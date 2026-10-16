"""Typed, JSON-serializable facts for SCOPE. No environment execution occurs here."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any

from .protocol import ScopeProtocol, require_name


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def require_hash(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA-256 digest")


class ExecutionRole(str, Enum):
    MAIN_ENTRY = "main_entry"
    MAIN_REPAIR = "main_repair"
    DIAGNOSTIC = "diagnostic"
    FOLLOWUP_ENTRY = "followup_entry"
    FOLLOWUP_REPAIR = "followup_repair"

    @property
    def is_search(self) -> bool:
        return self is not ExecutionRole.DIAGNOSTIC


class ExecutionStatus(str, Enum):
    RESERVED = "reserved"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"


class Termination(str, Enum):
    NORMAL = "normal"
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class ExecutionOutcome(str, Enum):
    SUCCESS = "success"
    TASK_FAILURE = "task_failure"
    PROGRAM_ERROR = "program_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class IssueKind(str, Enum):
    MODEL_ERROR = "model_error"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    REPLAY_UNAVAILABLE = "replay_unavailable"


@dataclass(frozen=True)
class EvidenceRef:
    """Reference relative to the archive directory, not a claim of tape validity.

    The artifact producer supplies the digest. M1 stores references, not payloads;
    consumers must verify payload bytes and provenance before using the evidence.
    """

    kind: str
    path: str
    sha256: str

    def __post_init__(self) -> None:
        require_name(self.kind, "evidence kind")
        require_hash(self.sha256)
        if not isinstance(self.path, str) or not self.path or "\\" in self.path or ":" in self.path:
            raise ValueError("evidence path must be a relative POSIX path")
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or str(path) != self.path or self.path == ".":
            raise ValueError("evidence path must stay inside the archive")


@dataclass(frozen=True)
class ProgramVersion:
    source: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("program source must be nonempty text")
        require_hash(self.sha256)
        if content_hash(self.source) != self.sha256:
            raise ValueError("program source hash mismatch")

    @classmethod
    def from_source(cls, source: str) -> ProgramVersion:
        return cls(source=source, sha256=content_hash(source))


@dataclass(frozen=True)
class CampaignIdentity:
    """Caller-supplied fingerprints; creating one never probes services or Git.

    configuration_sha256 covers model settings, prompts, environment config and
    other learning inputs. api_sha256/runtime_sha256 identify the declared API
    surface and execution semantics; they do not prove determinism.
    """

    run_id: str
    protocol: ScopeProtocol
    api_sha256: str
    runtime_sha256: str
    configuration_sha256: str

    def __post_init__(self) -> None:
        require_name(self.run_id, "run ID")
        if not isinstance(self.protocol, ScopeProtocol):
            raise ValueError("protocol must be a ScopeProtocol")
        for digest in (self.api_sha256, self.runtime_sha256, self.configuration_sha256):
            require_hash(digest)

    @property
    def campaign_id(self) -> str:
        return content_hash(canonical_json(asdict(self)))


@dataclass(frozen=True)
class ExecutionResult:
    task_completed: bool | None
    termination: Termination
    evidence: tuple[EvidenceRef, ...] = ()
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "termination", Termination(self.termination))
        if self.task_completed is not None and type(self.task_completed) is not bool:
            raise ValueError("task_completed must be bool or None")
        if self.termination is Termination.NORMAL and self.task_completed is None:
            raise ValueError("normal termination requires an evaluator outcome")
        if not isinstance(self.evidence, (tuple, list)) or any(
            not isinstance(ref, EvidenceRef) for ref in self.evidence
        ):
            raise ValueError("evidence must contain EvidenceRef objects")
        object.__setattr__(self, "evidence", tuple(self.evidence))
        if len([ref for ref in self.evidence if ref.kind == "tape"]) > 1:
            raise ValueError("an execution can reference at most one tape")
        if self.error_code is not None:
            require_name(self.error_code, "error code")

    @property
    def outcome(self) -> ExecutionOutcome:
        if self.termination is Termination.INFRASTRUCTURE_ERROR:
            return ExecutionOutcome.INFRASTRUCTURE_ERROR
        if self.termination is not Termination.NORMAL:
            return ExecutionOutcome.PROGRAM_ERROR
        return ExecutionOutcome.SUCCESS if self.task_completed else ExecutionOutcome.TASK_FAILURE

    @property
    def success(self) -> bool:
        return self.outcome is ExecutionOutcome.SUCCESS

    @property
    def tape(self) -> EvidenceRef | None:
        return next((ref for ref in self.evidence if ref.kind == "tape"), None)


@dataclass(frozen=True)
class ExecutionRecord:
    execution_id: str
    seed: int
    source_sha256: str
    role: ExecutionRole
    started_revision: int
    status: ExecutionStatus = ExecutionStatus.RESERVED
    result: ExecutionResult | None = None
    completed_revision: int | None = None
    accepted_source_changed: bool = False

    def __post_init__(self) -> None:
        require_name(self.execution_id, "execution ID")
        require_hash(self.source_sha256)
        object.__setattr__(self, "role", ExecutionRole(self.role))
        object.__setattr__(self, "status", ExecutionStatus(self.status))
        if (
            type(self.seed) is not int
            or type(self.started_revision) is not int
            or self.started_revision < 1
        ):
            raise ValueError("invalid seed or started revision")
        if type(self.accepted_source_changed) is not bool:
            raise ValueError("accepted_source_changed must be bool")
        if self.status is ExecutionStatus.COMPLETED:
            if (
                not isinstance(self.result, ExecutionResult)
                or type(self.completed_revision) is not int
            ):
                raise ValueError("completed execution requires a result and completion revision")
            if self.completed_revision <= self.started_revision:
                raise ValueError("completion must follow reservation")
        elif (
            self.result is not None
            or self.completed_revision is not None
            or self.accepted_source_changed
        ):
            raise ValueError("unresolved execution cannot have a completed result")


@dataclass(frozen=True)
class SuccessWitness:
    seed: int
    execution_id: str
    source_sha256: str
    tape: EvidenceRef | None = None


@dataclass(frozen=True)
class IssueRecord:
    issue_id: str
    kind: IssueKind
    source_sha256: str
    code: str
    seed: int | None
    revision: int

    def __post_init__(self) -> None:
        require_name(self.issue_id, "issue ID")
        require_name(self.code, "issue code")
        require_hash(self.source_sha256)
        object.__setattr__(self, "kind", IssueKind(self.kind))
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("invalid issue revision")


@dataclass
class CampaignState:
    """Detached archive snapshot. Mutating it cannot mutate the live archive.

    Budget usage and current outcomes are derived from the execution ledger to
    avoid a second counter/outcome store drifting away from the underlying facts.
    """

    identity: CampaignIdentity
    initial_source_sha256: str
    working_source_sha256: str
    accepted_source_sha256: str
    programs: dict[str, ProgramVersion]
    revision: int = 0
    executions: dict[str, ExecutionRecord] = field(default_factory=dict)
    success_bank: dict[int, SuccessWitness] = field(default_factory=dict)
    latest_accepted_success_id: str | None = None
    issues: tuple[IssueRecord, ...] = ()

    @property
    def pending_execution(self) -> ExecutionRecord | None:
        return next(
            (r for r in self.executions.values() if r.status is not ExecutionStatus.COMPLETED), None
        )

    def count(self, seed: int, role: ExecutionRole) -> int:
        self.identity.protocol.require_development_seed(seed)
        role = ExecutionRole(role)
        return sum(r.seed == seed and r.role is role for r in self.executions.values())

    def latest_result(self, seed: int, source_sha256: str) -> ExecutionRecord | None:
        self.identity.protocol.require_development_seed(seed)
        records = [
            r
            for r in self.executions.values()
            if r.seed == seed
            and r.source_sha256 == source_sha256
            and r.status is ExecutionStatus.COMPLETED
        ]
        return max(records, key=lambda r: r.completed_revision or 0, default=None)

    def validate(self) -> None:
        """Validate references and rederive acceptance and witnesses on reload."""
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("invalid campaign revision")
        for key, program in self.programs.items():
            if key != program.sha256:
                raise ValueError("program key/hash mismatch")
        for digest in (
            self.initial_source_sha256,
            self.working_source_sha256,
            self.accepted_source_sha256,
        ):
            if digest not in self.programs:
                raise ValueError("program reference is missing")
        accepted = self.initial_source_sha256
        latest = None
        bank: dict[int, SuccessWitness] = {}
        pending = 0
        previous_end = 0
        seen_revisions: set[int] = set()
        for key, record in sorted(
            self.executions.items(), key=lambda item: item[1].started_revision
        ):
            self.identity.protocol.require_development_seed(record.seed)
            if key != record.execution_id or record.source_sha256 not in self.programs:
                raise ValueError("execution identity/source mismatch")
            if record.started_revision <= previous_end or record.started_revision > self.revision:
                raise ValueError("executions must be serial and within the campaign revision")
            seen_revisions.add(record.started_revision)
            if record.role is ExecutionRole.DIAGNOSTIC and record.source_sha256 != accepted:
                raise ValueError("diagnostic must execute the accepted program")
            if record.status is not ExecutionStatus.COMPLETED:
                if record.role.is_search and record.source_sha256 != self.working_source_sha256:
                    raise ValueError("pending search must reference the working program")
                pending += 1
                previous_end = self.revision
                continue
            end = record.completed_revision
            assert end is not None and record.result is not None
            if end > self.revision:
                raise ValueError("completion exceeds campaign revision")
            previous_end = end
            seen_revisions.add(end)
            changed = False
            if record.role.is_search and record.result.success:
                changed = accepted != record.source_sha256
                accepted = record.source_sha256
                latest = record.execution_id
                bank[record.seed] = SuccessWitness(
                    record.seed,
                    record.execution_id,
                    record.source_sha256,
                    record.result.tape,
                )
            if record.accepted_source_changed != changed:
                raise ValueError("accepted source update flag contradicts execution history")
        if pending > 1:
            raise ValueError("only one unresolved execution is allowed per task")
        if accepted != self.accepted_source_sha256 or latest != self.latest_accepted_success_id:
            raise ValueError("accepted program/latest success contradicts execution history")
        if bank != self.success_bank:
            raise ValueError("success bank contradicts execution history")
        for seed in self.identity.protocol.development_seeds:
            if (
                self.count(seed, ExecutionRole.MAIN_REPAIR)
                > self.identity.protocol.max_main_repairs
            ):
                raise ValueError("main repair budget exceeded")
            if (
                self.count(seed, ExecutionRole.FOLLOWUP_REPAIR)
                > self.identity.protocol.max_followup_repairs
            ):
                raise ValueError("follow-up repair budget exceeded")
        issue_ids: set[str] = set()
        for issue in self.issues:
            if issue.seed is not None:
                self.identity.protocol.require_development_seed(issue.seed)
            if issue.source_sha256 not in self.programs or issue.revision > self.revision:
                raise ValueError("issue refers to an unknown program or revision")
            if issue.issue_id in issue_ids or issue.revision in seen_revisions:
                raise ValueError("duplicate issue ID or revision")
            issue_ids.add(issue.issue_id)
            seen_revisions.add(issue.revision)

    def to_dict(self) -> dict[str, Any]:
        # JSON round trip normalizes tuple fields, Enum values and integer keys.
        return json.loads(canonical_json(asdict(self)))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CampaignState:
        data = dict(value)
        identity = dict(data["identity"])
        identity["protocol"] = ScopeProtocol(**identity["protocol"])
        data["identity"] = CampaignIdentity(**identity)
        data["programs"] = {key: ProgramVersion(**row) for key, row in data["programs"].items()}
        executions = {}
        for key, row in data["executions"].items():
            row = dict(row)
            if row["result"] is not None:
                result = dict(row["result"])
                result["evidence"] = tuple(EvidenceRef(**ref) for ref in result["evidence"])
                row["result"] = ExecutionResult(**result)
            executions[key] = ExecutionRecord(**row)
        data["executions"] = executions
        bank = {}
        for key, row in data["success_bank"].items():
            if str(int(key)) != key:
                raise ValueError("success bank keys must be canonical seed integers")
            row = dict(row)
            if row["tape"] is not None:
                row["tape"] = EvidenceRef(**row["tape"])
            bank[int(key)] = SuccessWitness(**row)
        data["success_bank"] = bank
        data["issues"] = tuple(IssueRecord(**row) for row in data["issues"])
        state = cls(**data)
        state.validate()
        return state
