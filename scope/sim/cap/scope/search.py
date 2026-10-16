"""M7 serial, resumable SCOPE search with explicit external-action receipts.

One atomic checkpoint owns BOTH the M1 ledger and scheduler state. M1 mutations
are staged in memory and committed with the scheduling decision, eliminating a
two-file transaction gap. External actions are claimed durably before dispatch;
an uncertain dispatched action is never rerun implicitly.
"""

from __future__ import annotations

import copy
import fcntl
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .archive import IdentityMismatchError, ScopeArchive, StaleArchiveError
from .critic import CriticReport, NodeAssessment, assess_execution
from .diagnostic_graph import (
    DiagnosticGraph,
    ExecutionEvidence,
    PublicEvidence,
    digest,
    fields,
    json_text,
    parse_json,
    read_public_artifact,
    require_text,
    save_artifact,
)
from .records import (
    CampaignIdentity,
    CampaignState,
    EvidenceRef,
    ExecutionOutcome,
    ExecutionResult,
    ExecutionRole,
    ExecutionStatus,
    IssueKind,
    ProgramVersion,
)
from .repair import RepairAttempt, install_candidate, propose_repair
from .replay import ReplayEnvironment, ReplayReport, ReplayStatus, replay
from .retrieval import retrieve_repair_context


class RecoveryRequiredError(RuntimeError):
    """A dispatched action has no durable receipt; reconcile it explicitly."""


class _StagedArchive(ScopeArchive):
    """Reuse M1's transitions; persistence belongs to SearchCampaign._transaction."""

    def _commit(self, state):
        state.revision += 1
        state.validate()
        self._state = state


@dataclass(frozen=True)
class ExecutionReceipt:
    result: ExecutionResult
    max_step: int = 0
    items: tuple[PublicEvidence, ...] = ()
    raw_evidence: tuple[EvidenceRef, ...] = ()


@dataclass(frozen=True)
class SearchAction:
    action_id: str
    kind: str
    status: str
    payload_json: str

    @property
    def payload(self):
        return parse_json(self.payload_json)


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


def _evidence(data, state, execution_id):
    record = state.executions[execution_id]
    items = []
    for row in data["items"]:
        row = dict(row)
        if row["artifact"] is not None:
            row["artifact"] = EvidenceRef(**row["artifact"])
        items.append(PublicEvidence(**row))
    result = ExecutionEvidence(
        state.identity.protocol,
        record,
        state.programs[record.source_sha256],
        data["api_sha256"],
        data["task_language"],
        data["api_specification"],
        data["max_step"],
        tuple(items),
        tuple(EvidenceRef(**r) for r in data["raw_evidence"]),
    )
    if result.to_dict() != data:
        raise ValueError("stored evidence differs from ledger")
    return result


def _assessment(data, evidence, graph):
    if data["evidence_sha256"] != evidence.sha256 or data["graph_sha256"] != graph.sha256:
        raise ValueError("stored assessment identity mismatch")
    # Requests/raw responses remain in the checkpoint verbatim. The restored
    # host assessment needs no model call and no reloading of image bytes.
    nodes = tuple(
        NodeAssessment(
            r["node_id"],
            r["raw_label"],
            r["label"],
            tuple(r["window"]) if r["window"] is not None else None,
            tuple(r["citations"]),
            r["hypothesis"],
            tuple(r["reasons"]),
        )
        for r in data["nodes"]
    )
    return CriticReport(
        graph,
        evidence,
        None,
        data["raw_response"],
        nodes,
        tuple(data["issues"]),
        data["status"],
        data["evidence_verified"],
    )


def _replay_report(data):
    row = dict(data)
    row["status"] = ReplayStatus(row["status"])
    row["assumptions"] = tuple(row["assumptions"])
    return ReplayReport(**row)


class SearchCampaign:
    """Single-coordinator task search, with stale-handle detection and recovery.

    Public snapshots are detached. create() requires a fresh directory, explicit
    random seed and a previously planned task graph. No model/environment work
    occurs until run_next() or a caller explicitly dispatches a claimed action.
    """

    def __init__(self, directory, data):
        self.directory = Path(directory).absolute()
        self._data = data

    @property
    def path(self):
        return self.directory / "search.json"

    @classmethod
    def create(
        cls,
        directory: Path,
        identity: CampaignIdentity,
        initial_program: ProgramVersion,
        *,
        graph: DiagnosticGraph,
        task_language: str,
        api_specification: str,
        random_seed: int,
        replay_environment: ReplayEnvironment,
    ):
        if type(random_seed) is not int:
            raise ValueError("an explicit integer sampling seed is required")
        require_text(task_language, "task language")
        require_text(api_specification, "API specification")
        if (graph.suite, graph.task, graph.api_sha256) != (
            identity.protocol.suite,
            identity.protocol.task,
            identity.api_sha256,
        ) or (replay_environment.api_sha256, replay_environment.runtime_sha256) != (
            identity.api_sha256,
            identity.runtime_sha256,
        ):
            raise ValueError("search graph/runtime identity mismatch")
        state = CampaignState(
            identity,
            initial_program.sha256,
            initial_program.sha256,
            initial_program.sha256,
            {initial_program.sha256: initial_program},
        )
        data = {
            "format": "scope-search",
            "version": 1,
            "revision": 0,
            "ledger": state.to_dict(),
            "graph": graph.to_dict(),
            "task_language": task_language,
            "api_specification": api_specification,
            "replay_environment": asdict(replay_environment),
            "random_seed": random_seed,
            "rng_state": random.Random(random_seed).getstate(),
            "main_index": 0,
            "turn": "main",
            "followups": [],
            "visit": None,
            "pending": None,
            "outcome": None,
            "need_critic": None,
            "update": None,
            "updates": [],
            "last_feedback": None,
            "evidence": {},
            "assessments": {},
            "attempts": {},
            "events": [],
            "frozen": None,
        }
        data = parse_json(json_text(data))
        cls._validate(data)
        campaign = cls(directory, data)
        campaign.directory.mkdir(parents=True, exist_ok=False)
        campaign._write(data)
        return campaign

    @classmethod
    def open(cls, directory: Path, *, expected_identity: CampaignIdentity):
        campaign = cls(directory, {})
        data = campaign._read()
        if CampaignState.from_dict(data["ledger"]).identity != expected_identity:
            raise IdentityMismatchError("search campaign identity changed")
        campaign._data = data
        return campaign

    def snapshot(self):
        return CampaignState.from_dict(copy.deepcopy(self._data["ledger"]))

    def state(self):
        return copy.deepcopy(self._data)

    def _read(self):
        if self.path.is_symlink():
            raise ValueError("search checkpoint cannot be a symlink")
        envelope = parse_json(self.path.read_text(encoding="utf-8"))
        fields(envelope, ("sha256", "data"))
        if digest(envelope["data"]) != envelope["sha256"]:
            raise ValueError("search checkpoint checksum mismatch")
        self._validate(envelope["data"])
        return envelope["data"]

    @staticmethod
    def _validate(data):
        if (
            data["format"] != "scope-search"
            or type(data["version"]) is not int
            or data["version"] != 1
        ):
            raise ValueError("unsupported search checkpoint")
        state = CampaignState.from_dict(data["ledger"])
        protocol = state.identity.protocol
        graph = DiagnosticGraph.from_dict(data["graph"])
        if (graph.suite, graph.task, graph.api_sha256) != (
            protocol.suite,
            protocol.task,
            state.identity.api_sha256,
        ):
            raise ValueError("checkpoint graph identity mismatch")
        environment = ReplayEnvironment(**data["replay_environment"])
        if (environment.api_sha256, environment.runtime_sha256) != (
            state.identity.api_sha256,
            state.identity.runtime_sha256,
        ):
            raise ValueError("checkpoint runtime mismatch")
        if type(data["main_index"]) is not int or not 0 <= data["main_index"] <= len(
            protocol.development_seeds
        ):
            raise ValueError("invalid main cursor")
        if data["turn"] not in ("main", "followup"):
            raise ValueError("invalid queue turn")
        random.Random().setstate(_tuples(data["rng_state"]))
        queued = set()
        for item in data["followups"]:
            protocol.require_development_seed(item["seed"])
            record = state.executions[item["failure_id"]]
            if (
                item["seed"] in queued
                or record.seed != item["seed"]
                or record.role is not ExecutionRole.DIAGNOSTIC
                or record.result is None
                or record.result.outcome
                not in (ExecutionOutcome.TASK_FAILURE, ExecutionOutcome.PROGRAM_ERROR)
            ):
                raise ValueError("invalid follow-up failure")
            queued.add(item["seed"])
        if data["visit"]:
            protocol.require_development_seed(data["visit"]["seed"])
            if data["visit"]["queue"] not in ("main", "followup"):
                raise ValueError("invalid active queue")
        pending = data["pending"]
        if pending:
            if pending["kind"] not in ("execute", "critic", "repair", "replay") or pending[
                "status"
            ] not in ("ready", "dispatched"):
                raise ValueError("invalid pending action")
            if pending["kind"] == "execute":
                if (
                    state.pending_execution is None
                    or state.pending_execution.execution_id != pending["payload"]["execution_id"]
                ):
                    raise ValueError("pending execution/action mismatch")
            elif state.pending_execution is not None:
                raise ValueError("execution reservation has no dispatch action")
        elif state.pending_execution is not None:
            raise ValueError("orphan execution reservation")
        for execution_id, row in data["evidence"].items():
            _evidence(row, state, execution_id)
        for execution_id, row in data["assessments"].items():
            _assessment(row, _evidence(data["evidence"][execution_id], state, execution_id), graph)
        for update in [*data["updates"], *([data["update"]] if data["update"] else [])]:
            trigger = state.executions[update["trigger"]]
            if not trigger.accepted_source_changed:
                raise ValueError("certification requires an accepted source change")
            sampled = update["sample"]
            if (
                len(sampled) > protocol.replay_sample_size
                or len(set(sampled)) != len(sampled)
                or not set(sampled) <= set(update["bank_before"])
            ):
                raise ValueError("invalid historical sample")
            if len(update["diagnostics"]) > protocol.max_diagnostics_per_update:
                raise ValueError("diagnostic budget exceeded")
        if data["frozen"] is not None:
            if (
                pending
                or data["visit"]
                or data["followups"]
                or data["update"]
                or data["main_index"] != len(protocol.development_seeds)
            ):
                raise ValueError("cannot freeze unfinished search")
            if data["frozen"]["sha256"] != state.accepted_source_sha256:
                raise ValueError("frozen source must be accepted")
            ProgramVersion(**data["frozen"])

    def _write(self, data):
        fd, name = tempfile.mkstemp(prefix=".search-", suffix=".tmp", dir=self.directory)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                stream.write(json_text({"sha256": digest(data), "data": data}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _transaction(self, operation):
        with (self.directory / ".search.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self._read() != self._data:
                raise StaleArchiveError("search changed on disk; reopen before continuing")
            data = copy.deepcopy(self._data)
            ledger = _StagedArchive(self.directory, CampaignState.from_dict(data["ledger"]))
            result = operation(data, ledger)
            data["ledger"] = ledger.snapshot().to_dict()
            data["revision"] += 1
            data = parse_json(json_text(data))
            self._validate(data)
            self._write(data)
            self._data = data
            return result

    @staticmethod
    def _event(data, kind, **details):
        data["events"].append({"index": len(data["events"]), "kind": kind, **details})

    @classmethod
    def _action(cls, data, kind, **payload):
        action_id = f"action_{len(data['events']):06d}"
        data["pending"] = {
            "action_id": action_id,
            "kind": kind,
            "status": "ready",
            "payload": payload,
        }
        cls._event(data, "action_ready", action_id=action_id, action_kind=kind, **payload)

    @classmethod
    def _execute(cls, data, ledger, seed, role):
        state = ledger.snapshot()
        bank = [
            state.success_bank[s].execution_id
            for s in state.identity.protocol.development_seeds
            if s in state.success_bank
        ]
        record = ledger.reserve_execution(seed, role)
        cls._action(
            data,
            "execute",
            execution_id=record.execution_id,
            seed=seed,
            role=role.value,
            source_sha256=record.source_sha256,
            bank_before=bank,
            repair_action_id=data["visit"].get("repair_action_id") if data["visit"] else None,
        )

    @classmethod
    def _end_visit(cls, data, reason):
        cls._event(data, "visit_end", **data["visit"], reason=reason)
        data["turn"] = "followup" if data["visit"]["queue"] == "main" else "main"
        data["visit"] = None

    @classmethod
    def _process_outcome(cls, data, ledger):
        outcome = data["outcome"]
        data["outcome"] = None
        state = ledger.snapshot()
        record = state.executions[outcome["execution_id"]]
        failed = record.result.outcome in (
            ExecutionOutcome.TASK_FAILURE,
            ExecutionOutcome.PROGRAM_ERROR,
        )
        if record.role is ExecutionRole.DIAGNOSTIC:
            if failed:
                data["followups"].append({"seed": record.seed, "failure_id": record.execution_id})
                cls._event(
                    data, "followup_enqueued", seed=record.seed, failure_id=record.execution_id
                )
            return
        if record.result.success:
            if record.accepted_source_changed:
                rng = random.Random()
                rng.setstate(_tuples(data["rng_state"]))
                before_rng = copy.deepcopy(data["rng_state"])
                bank = outcome["bank_before"]
                sample = rng.sample(
                    bank, min(len(bank), state.identity.protocol.replay_sample_size)
                )
                data["rng_state"] = rng.getstate()
                data["update"] = {
                    "trigger": record.execution_id,
                    "candidate_sha256": record.source_sha256,
                    "bank_before": bank,
                    "sample": sample,
                    "rng_before": before_rng,
                    "rng_after": rng.getstate(),
                    "reports": [],
                    "diagnostic_cursor": 0,
                    "diagnostics": [],
                    "skipped": [],
                }
                cls._event(
                    data,
                    "accepted_update_sample",
                    trigger=record.execution_id,
                    bank_before=bank,
                    sample=sample,
                )
            else:
                cls._end_visit(data, "search_success_same_source")
        elif failed:
            visit = data["visit"]
            role = (
                ExecutionRole.MAIN_REPAIR
                if visit["queue"] == "main"
                else ExecutionRole.FOLLOWUP_REPAIR
            )
            limit = (
                state.identity.protocol.max_main_repairs
                if visit["queue"] == "main"
                else state.identity.protocol.max_followup_repairs
            )
            if state.count(record.seed, role) < limit:
                visit.update(phase="repair", failure_id=record.execution_id)
            else:
                cls._end_visit(data, "repair_budget_exhausted")
        else:
            cls._end_visit(data, "infrastructure_error_no_task_outcome")

    @classmethod
    def _advance(cls, data, ledger):
        while data["pending"] is None and data["frozen"] is None:
            state = ledger.snapshot()
            protocol = state.identity.protocol
            if data["need_critic"]:
                cls._action(data, "critic", execution_id=data["need_critic"])
                break
            if data["outcome"]:
                cls._process_outcome(data, ledger)
                continue
            update = data["update"]
            if update is not None:
                if len(update["reports"]) < len(update["sample"]):
                    cls._action(
                        data,
                        "replay",
                        witness_execution_id=update["sample"][len(update["reports"])],
                        candidate_sha256=update["candidate_sha256"],
                    )
                    break
                while update["diagnostic_cursor"] < len(update["reports"]):
                    row = update["reports"][update["diagnostic_cursor"]]
                    update["diagnostic_cursor"] += 1
                    witness = state.executions[row["witness_execution_id"]]
                    pending_seeds = {q["seed"] for q in data["followups"]}
                    if data["visit"] and data["visit"]["queue"] == "followup":
                        pending_seeds.add(data["visit"]["seed"])
                    reason = None
                    if row["status"] == ReplayStatus.CERTIFIED.value:
                        reason = "certified"
                    elif witness.seed in pending_seeds:
                        reason = "pending_followup"
                    elif (
                        state.count(witness.seed, ExecutionRole.FOLLOWUP_REPAIR)
                        >= protocol.max_followup_repairs
                    ):
                        reason = "followup_budget_exhausted"
                    elif len(update["diagnostics"]) >= protocol.max_diagnostics_per_update:
                        reason = "diagnostic_budget_exhausted"
                    if reason:
                        update["skipped"].append({"seed": witness.seed, "reason": reason})
                        continue
                    cls._execute(data, ledger, witness.seed, ExecutionRole.DIAGNOSTIC)
                    update["diagnostics"].append(data["pending"]["payload"]["execution_id"])
                    break
                if data["pending"]:
                    break
                data["updates"].append(update)
                data["last_feedback"] = len(data["updates"]) - 1
                data["update"] = None
                cls._end_visit(data, "accepted_update_complete")
                continue
            if data["visit"] is not None:
                visit = data["visit"]
                if visit["phase"] == "repair":
                    cls._action(data, "repair", failure_id=visit["failure_id"], seed=visit["seed"])
                elif visit["phase"] == "entry":
                    role = (
                        ExecutionRole.MAIN_ENTRY
                        if visit["queue"] == "main"
                        else ExecutionRole.FOLLOWUP_ENTRY
                    )
                    cls._execute(data, ledger, visit["seed"], role)
                elif visit["phase"] == "validate":
                    role = (
                        ExecutionRole.MAIN_REPAIR
                        if visit["queue"] == "main"
                        else ExecutionRole.FOLLOWUP_REPAIR
                    )
                    cls._execute(data, ledger, visit["seed"], role)
                else:
                    raise ValueError("invalid visit phase")
                break
            main_available = data["main_index"] < len(protocol.development_seeds)
            if not main_available and not data["followups"]:
                data["frozen"] = asdict(state.programs[state.accepted_source_sha256])
                cls._event(data, "frozen", source_sha256=state.accepted_source_sha256)
                break
            queue = data["turn"]
            if queue == "main" and not main_available:
                queue = "followup"
            elif queue == "followup" and not data["followups"]:
                queue = "main"
            if queue == "main":
                seed = protocol.development_seeds[data["main_index"]]
                data["main_index"] += 1
                visit = {"queue": queue, "seed": seed, "phase": "entry"}
            else:
                followup = data["followups"].pop(0)
                seed = followup["seed"]
                if (
                    state.count(seed, ExecutionRole.FOLLOWUP_REPAIR)
                    >= protocol.max_followup_repairs
                ):
                    cls._event(
                        data, "followup_skipped", seed=seed, reason="repair_budget_exhausted"
                    )
                    continue
                failure = state.executions[followup["failure_id"]]
                visit = {
                    "queue": queue,
                    "seed": seed,
                    "failure_id": failure.execution_id,
                    "phase": "repair"
                    if failure.source_sha256 == state.working_source_sha256
                    else "entry",
                }
            data["visit"] = visit
            cls._event(data, "visit_start", **visit)

    def next_action(self) -> SearchAction | None:
        if not self._data["pending"] and not self._data["frozen"]:
            self._transaction(self._advance)
        pending = self._data["pending"]
        return (
            SearchAction(
                pending["action_id"],
                pending["kind"],
                pending["status"],
                json_text(pending["payload"]),
            )
            if pending
            else None
        )

    @staticmethod
    def _pending(data, action_id, status):
        row = data["pending"]
        if row is None or row["action_id"] != action_id:
            raise ValueError("action receipt is stale or already completed")
        if row["status"] != status:
            raise RecoveryRequiredError(
                "action already dispatched; supply a receipt or reconcile it"
            )
        return row

    def claim(self, action_id):
        def operation(data, ledger):
            self._pending(data, action_id, "ready")["status"] = "dispatched"
            self._event(data, "action_dispatched", action_id=action_id)

        self._transaction(operation)

    def interrupt(self, action_id, *, error_code="external_action_interrupted"):
        from .protocol import require_name

        require_name(error_code, "error code")

        def operation(data, ledger):
            row = self._pending(data, action_id, "dispatched")
            if row["kind"] == "execute":
                record = ledger.snapshot().executions[row["payload"]["execution_id"]]
                if record.status is ExecutionStatus.RESERVED:
                    ledger.mark_interrupted(record.execution_id)
            self._event(data, "action_interrupted", action_id=action_id, error_code=error_code)

        self._transaction(operation)

    def retry_nonexecution(self, action_id):
        """Explicit recovery choice; execution actions cannot be reset/retried."""

        def operation(data, ledger):
            row = self._pending(data, action_id, "dispatched")
            if row["kind"] == "execute":
                raise RecoveryRequiredError(
                    "execution outcome must be reconciled; no automatic rerun"
                )
            row["status"] = "ready"
            self._event(data, "explicit_nonexecution_retry", action_id=action_id)

        self._transaction(operation)

    def evidence(self, execution_id):
        return _evidence(self._data["evidence"][execution_id], self.snapshot(), execution_id)

    def repair_context(self):
        pending = self._data["pending"]
        if pending is None or pending["kind"] != "repair":
            raise ValueError("no pending repair")
        data = self._data
        feedback = (
            data["updates"][data["last_feedback"]] if data["last_feedback"] is not None else None
        )
        state = self.snapshot()
        selected = {pending["payload"]["failure_id"]}
        if state.latest_accepted_success_id:
            selected.add(state.latest_accepted_success_id)
        if feedback:
            selected.update(feedback["diagnostics"])
        evidence = {key: self.evidence(key) for key in selected}
        graph = DiagnosticGraph.from_dict(data["graph"])
        reports = {
            key: _assessment(data["assessments"][key], evidence[key], graph)
            for key in selected
            if key in data["assessments"]
        }
        return retrieve_repair_context(
            state,
            pending["payload"]["failure_id"],
            graph=graph,
            evidence=evidence,
            archive_root=self.directory,
            assessments=reports,
            replay_reports=tuple(_replay_report(r) for r in feedback["reports"])
            if feedback
            else (),
            diagnostic_execution_ids=tuple(feedback["diagnostics"]) if feedback else (),
            repair_attempts=tuple(data["attempts"][key] for key in sorted(data["attempts"])),
        )

    def complete(self, action_id, receipt):
        def operation(data, ledger):
            action = self._pending(data, action_id, "dispatched")
            payload = action["payload"]
            state = ledger.snapshot()
            kind = action["kind"]
            if kind == "execute":
                if not isinstance(receipt, ExecutionReceipt):
                    raise TypeError("expected ExecutionReceipt")
                record = ledger.complete_execution(payload["execution_id"], receipt.result)
                item = ExecutionEvidence(
                    state.identity.protocol,
                    record,
                    state.programs[record.source_sha256],
                    state.identity.api_sha256,
                    data["task_language"],
                    data["api_specification"],
                    receipt.max_step,
                    receipt.items,
                    receipt.raw_evidence,
                )
                data["evidence"][record.execution_id] = item.to_dict()
                data["outcome"] = {
                    "execution_id": record.execution_id,
                    "bank_before": payload["bank_before"],
                }
                data["need_critic"] = (
                    record.execution_id
                    if record.result.outcome is not ExecutionOutcome.INFRASTRUCTURE_ERROR
                    else None
                )
                self._event(
                    data,
                    "execution_completed",
                    execution_id=record.execution_id,
                    seed=record.seed,
                    role=record.role.value,
                    source_sha256=record.source_sha256,
                    outcome=record.result.outcome.value,
                    accepted_source_changed=record.accepted_source_changed,
                    repair_action_id=payload["repair_action_id"],
                )
            elif kind == "critic":
                item = _evidence(
                    data["evidence"][payload["execution_id"]], state, payload["execution_id"]
                )
                if (
                    not isinstance(receipt, CriticReport)
                    or receipt.evidence.sha256 != item.sha256
                    or receipt.graph.to_dict() != data["graph"]
                ):
                    raise ValueError("critic receipt identity mismatch")
                data["assessments"][payload["execution_id"]] = receipt.to_dict()
                data["need_critic"] = None
            elif kind == "replay":
                witness = state.executions[payload["witness_execution_id"]]
                if not isinstance(receipt, ReplayReport) or (
                    receipt.candidate_sha256,
                    receipt.witness_execution_id,
                    receipt.witness_source_sha256,
                    receipt.tape_sha256,
                    receipt.api_sha256,
                    receipt.runtime_sha256,
                ) != (
                    payload["candidate_sha256"],
                    witness.execution_id,
                    witness.source_sha256,
                    witness.result.tape.sha256 if witness.result.tape else None,
                    state.identity.api_sha256,
                    state.identity.runtime_sha256,
                ):
                    raise ValueError("replay receipt identity mismatch")
                if receipt.status is ReplayStatus.CERTIFIED and (
                    not receipt.full_match
                    or not receipt.normal_termination
                    or receipt.tape_sha256 is None
                    or receipt.consumed_events != receipt.total_events
                ):
                    raise ValueError("invalid certificate")
                data["update"]["reports"].append(receipt.to_dict())
            else:
                if not isinstance(receipt, RepairAttempt):
                    raise TypeError("expected M6 RepairAttempt")
                if receipt.context.sha256 != self.repair_context().sha256:
                    raise ValueError("repair receipt context mismatch")
                data["attempts"][action_id] = receipt.to_dict()
                if receipt.issue is not None:
                    ledger.record_issue(
                        IssueKind.MODEL_ERROR, "repair_proposal_failed", seed=payload["seed"]
                    )
                    self._end_visit(data, "repair_model_error")
                else:
                    install_candidate(ledger, receipt)
                    data["visit"].update(phase="validate", repair_action_id=action_id)
            self._event(data, "action_completed", action_id=action_id, action_kind=kind)
            data["pending"] = None

        self._transaction(operation)

    def run_next(
        self, *, executor, critic_model, planner, generator, replayer=None,
        candidate_validator=None,
    ):
        """Perform one explicitly claimed external action. Exceptions require recovery.

        executor(action, reserved_record, program) returns ExecutionReceipt and
        owns fresh reset/capture. replayer(action, witness, program) may be supplied
        for tests; the default calls M4 with the recorded tape only.
        """
        action = self.next_action()
        if action is None:
            return None
        self.claim(action.action_id)
        payload = action.payload
        state = self.snapshot()
        try:
            if action.kind == "execute":
                record = state.executions[payload["execution_id"]]
                receipt = executor(action, record, state.programs[record.source_sha256])
            elif action.kind == "critic":
                receipt = assess_execution(
                    DiagnosticGraph.from_dict(self._data["graph"]),
                    self.evidence(payload["execution_id"]),
                    archive_root=self.directory,
                    model=critic_model,
                )
            elif action.kind == "repair":
                receipt = propose_repair(
                    self.repair_context(), planner=planner, generator=generator,
                    candidate_validator=candidate_validator,
                )
            else:
                witness = state.executions[payload["witness_execution_id"]]
                program = state.programs[payload["candidate_sha256"]]
                if replayer is not None:
                    receipt = replayer(action, witness, program)
                else:
                    ref = witness.result.tape
                    # M4 reports missing tapes as unavailable. Verify a present
                    # artifact before deriving the tape directory from its path.
                    try:
                        if ref:
                            read_public_artifact(self.directory, ref)
                    except (OSError, ValueError):
                        receipt = ReplayReport(
                            ReplayStatus.UNAVAILABLE,
                            "public_tape_unavailable",
                            program.sha256,
                            witness.execution_id,
                            witness.source_sha256,
                            ref.sha256,
                            state.identity.api_sha256,
                            state.identity.runtime_sha256,
                        )
                    else:
                        tape_dir = (
                            (self.directory / ref.path).parent
                            if ref
                            else self.directory / "missing-tape"
                        )
                        receipt = replay(
                            tape_dir,
                            program,
                            witness=witness,
                            environment=ReplayEnvironment(**self._data["replay_environment"]),
                        )
            self.complete(action.action_id, receipt)
        except Exception as error:
            self.interrupt(action.action_id, error_code=type(error).__name__)
            raise RecoveryRequiredError(
                f"{action.kind} action requires reconciliation: {type(error).__name__}"
            ) from None
        return action

    @property
    def frozen_program(self):
        return ProgramVersion(**self._data["frozen"]) if self._data["frozen"] else None

    def export_frozen(self, path: Path):
        if self.frozen_program is None:
            raise ValueError("search is not complete")
        save_artifact(
            path,
            {
                "format": "scope-frozen-program",
                "version": 1,
                "campaign_id": self.snapshot().identity.campaign_id,
                "program": asdict(self.frozen_program),
                "search_sha256": digest(self._data),
            },
        )
