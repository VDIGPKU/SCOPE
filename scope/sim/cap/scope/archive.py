"""Atomic, identity-bound persistence for a single SCOPE learning task.

This module stores supplied facts. It never executes source, calls a model,
launches a simulator, invokes Git, or reads environment variables for credentials.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from .records import (
    CampaignIdentity,
    CampaignState,
    ExecutionRecord,
    ExecutionResult,
    ExecutionRole,
    ExecutionStatus,
    IssueKind,
    IssueRecord,
    ProgramVersion,
    SuccessWitness,
    canonical_json,
    content_hash,
)

SCHEMA_VERSION = 1


class ArchiveError(ValueError):
    """Invalid or inconsistent archive state."""


class IdentityMismatchError(ArchiveError):
    """The requested learning protocol or execution identity has changed."""


class StaleArchiveError(ArchiveError):
    """Another archive handle has persisted a newer state; reopen before editing."""


class BudgetExhaustedError(ArchiveError):
    """All repair execution slots for this configuration have been reserved."""


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ArchiveError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class ScopeArchive:
    """One serial task archive with advisory locking and stale-write detection.

    Each handle is used serially. Snapshots are detached copies. Every successful
    mutation is durably saved; separate handles cannot silently overwrite changes.
    A reserved or interrupted execution retains its slot until reconciled with a
    supplied result. This is bookkeeping, not exactly-once environment execution.
    """

    def __init__(self, directory: Path, state: CampaignState) -> None:
        self.directory = directory
        self._state = state

    @property
    def path(self) -> Path:
        return self.directory / "archive.json"

    @classmethod
    def create(
        cls,
        directory: Path,
        identity: CampaignIdentity,
        initial_program: ProgramVersion,
    ) -> ScopeArchive:
        state = CampaignState(
            identity=identity,
            initial_source_sha256=initial_program.sha256,
            working_source_sha256=initial_program.sha256,
            accepted_source_sha256=initial_program.sha256,
            programs={initial_program.sha256: initial_program},
        )
        state.validate()
        directory = Path(directory).absolute()
        # Require a fresh directory, including for empty existing directories.
        # Never silently overwrite or adopt an earlier experiment's artifacts.
        directory.mkdir(parents=True, exist_ok=False)
        archive = cls(directory, state)
        with archive._lock():
            archive._write(state)
        return archive

    @classmethod
    def open(cls, directory: Path, *, expected_identity: CampaignIdentity) -> ScopeArchive:
        directory = Path(directory).absolute()
        state = cls._read(directory / "archive.json")
        if state.identity != expected_identity:
            raise IdentityMismatchError("archive identity differs from the requested campaign")
        return cls(directory, state)

    @staticmethod
    def _read(path: Path) -> CampaignState:
        try:
            envelope = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys)
            if set(envelope) != {"schema_version", "state_sha256", "state"}:
                raise ArchiveError("invalid archive envelope")
            if (
                type(envelope["schema_version"]) is not int
                or envelope["schema_version"] != SCHEMA_VERSION
            ):
                raise ArchiveError("unsupported archive schema version")
            if content_hash(canonical_json(envelope["state"])) != envelope["state_sha256"]:
                raise ArchiveError("archive checksum mismatch")
            return CampaignState.from_dict(envelope["state"])
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            raise ArchiveError(f"invalid archive: {error}") from error

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with (self.directory / ".archive.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _write(self, state: CampaignState) -> None:
        payload = state.to_dict()
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "state_sha256": content_hash(canonical_json(payload)),
            "state": payload,
        }
        descriptor, name = tempfile.mkstemp(prefix=".archive-", suffix=".tmp", dir=self.directory)
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
                destination.write(
                    json.dumps(envelope, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
                )
                destination.flush()
                os.fsync(destination.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _commit(self, state: CampaignState) -> None:
        previous_revision = state.revision
        state.revision = previous_revision + 1
        state.validate()
        with self._lock():
            disk = self._read(self.path)
            if disk.identity != self._state.identity:
                raise IdentityMismatchError("archive identity changed on disk")
            if disk.revision != previous_revision or disk.to_dict() != self._state.to_dict():
                raise StaleArchiveError("archive changed on disk; reopen before updating")
            self._write(state)
            self._state = state

    def snapshot(self) -> CampaignState:
        return copy.deepcopy(self._state)

    def _require_idle(self) -> None:
        if self._state.pending_execution is not None:
            raise ArchiveError(
                "resolve the pending execution before changing source or reserving another"
            )

    def set_working_program(self, program: ProgramVersion) -> None:
        """Install a candidate without accepting it or executing it."""
        self._require_idle()
        if not isinstance(program, ProgramVersion):
            raise TypeError("program must be a ProgramVersion")
        state = self.snapshot()
        state.programs[program.sha256] = program
        state.working_source_sha256 = program.sha256
        self._commit(state)

    def reserve_execution(self, seed: int, role: ExecutionRole) -> ExecutionRecord:
        """Reserve before launching. Only repair roles consume repair slots.

        Main/follow-up entries and diagnostics are counted separately. Their
        scheduling, eligibility and per-source-update diagnostic limit belong to M7.
        """
        self._require_idle()
        protocol = self._state.identity.protocol
        protocol.require_development_seed(seed)
        role = ExecutionRole(role)
        for limited_role, limit in (
            (ExecutionRole.MAIN_REPAIR, protocol.max_main_repairs),
            (ExecutionRole.FOLLOWUP_REPAIR, protocol.max_followup_repairs),
        ):
            if role is limited_role and self._state.count(seed, role) >= limit:
                raise BudgetExhaustedError(f"{role.value} budget exhausted for seed {seed}")
        source = (
            self._state.accepted_source_sha256
            if role is ExecutionRole.DIAGNOSTIC
            else self._state.working_source_sha256
        )
        record = ExecutionRecord(
            execution_id=uuid.uuid4().hex,
            seed=seed,
            source_sha256=source,
            role=role,
            started_revision=self._state.revision + 1,
        )
        state = self.snapshot()
        state.executions[record.execution_id] = record
        self._commit(state)
        return record

    def complete_execution(self, execution_id: str, result: ExecutionResult) -> ExecutionRecord:
        """Record a supplied outcome and atomically update acceptance and witnesses.

        Repeated completion is rejected, so a caller cannot silently change a
        historical result or treat a duplicate completion as another update.
        """
        if not isinstance(result, ExecutionResult):
            raise TypeError("result must be an ExecutionResult")
        record = self._state.executions[execution_id]
        if record.status is ExecutionStatus.COMPLETED:
            raise ArchiveError("execution is already completed")
        changed = (
            record.role.is_search
            and result.success
            and record.source_sha256 != self._state.accepted_source_sha256
        )
        record = replace(
            record,
            status=ExecutionStatus.COMPLETED,
            result=result,
            completed_revision=self._state.revision + 1,
            accepted_source_changed=changed,
        )
        state = self.snapshot()
        state.executions[execution_id] = record
        if record.role.is_search and result.success:
            state.accepted_source_sha256 = record.source_sha256
            state.latest_accepted_success_id = execution_id
            state.success_bank[record.seed] = SuccessWitness(
                record.seed,
                execution_id,
                record.source_sha256,
                result.tape,
            )
        self._commit(state)
        return record

    def mark_interrupted(self, execution_id: str) -> None:
        """Preserve an unresolved slot; do not infer success/failure or rerun it."""
        record = self._state.executions[execution_id]
        if record.status is not ExecutionStatus.RESERVED:
            raise ArchiveError("only a reserved execution can be marked interrupted")
        state = self.snapshot()
        state.executions[execution_id] = replace(record, status=ExecutionStatus.INTERRUPTED)
        self._commit(state)

    def record_issue(self, kind: IssueKind, code: str, *, seed: int | None = None) -> IssueRecord:
        """Record an operational issue without fabricating an environment trial.

        Only a short identifier is accepted, not a raw provider error payload
        potentially containing credentials. Detailed evidence needs its own
        reviewed artifact path in a later module.
        """
        if seed is not None:
            self._state.identity.protocol.require_development_seed(seed)
        issue = IssueRecord(
            issue_id=uuid.uuid4().hex,
            kind=kind,
            code=code,
            seed=seed,
            source_sha256=self._state.working_source_sha256,
            revision=self._state.revision + 1,
        )
        state = self.snapshot()
        state.issues += (issue,)
        self._commit(state)
        return issue
