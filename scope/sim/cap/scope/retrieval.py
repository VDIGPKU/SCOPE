"""M6 development-only retrieval, with each execution's own source identity.

Inputs are trusted host archive snapshots and M4/M5 reports. No filesystem
discovery, model calls, replay or environment execution occurs here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .critic import MAX_FRAME_BYTES, MAX_REQUEST_BYTES, CriticReport, ModelRequest, _request
from .diagnostic_graph import (
    DiagnosticGraph,
    ExecutionEvidence,
    digest,
    json_text,
    read_public_artifact,
    save_artifact,
    warn_budget,
)
from .records import CampaignState, ExecutionOutcome, ExecutionRole, ExecutionStatus
from .replay import ReplayReport, ReplayStatus


@dataclass(frozen=True)
class RepairContext:
    """Immutable public snapshot; JSON access returns a detached copy."""

    request: ModelRequest
    state_sha256: str

    @property
    def payload(self):
        return self.request.payload

    @property
    def sha256(self):
        return digest(self.request.to_dict())

    def save(self, path: Path):
        save_artifact(
            path,
            {
                "format": "scope-repair-context",
                "version": 1,
                "context_sha256": self.sha256,
                "state_sha256": self.state_sha256,
                "request": self.request.to_dict(),
            },
        )


def _search_history(state, failure, attempts):
    """Bounded ledger facts and prior plans; no extra frames or model calls.

    Prior diagnoses are explicitly hypotheses. Validation of a proposal says
    nothing about its effect; only a completed execution supplies an outcome.
    """
    cutoff = failure.completed_revision
    executions = sorted(
        (
            record
            for record in state.executions.values()
            if record.status is ExecutionStatus.COMPLETED
            and record.role.is_search
            and record.completed_revision <= cutoff
        ),
        key=lambda record: record.completed_revision,
    )
    streak = []
    for record in reversed(executions):
        if record.result.outcome not in (
            ExecutionOutcome.TASK_FAILURE,
            ExecutionOutcome.PROGRAM_ERROR,
        ):
            break
        streak.append(record)

    def fact(record):
        return {
            "execution_id": record.execution_id,
            "seed": record.seed,
            "source_sha256": record.source_sha256,
            "role": record.role.value,
            "outcome": record.result.outcome.value,
            "error_code": record.result.error_code,
            "completed_revision": record.completed_revision,
        }

    # Include a previous configuration even when the last few executions all
    # belong to the current seed; these are outcome facts, not physical causes.
    recent = executions[-8:]
    other_seed = next((r for r in reversed(executions) if r.seed != failure.seed), None)
    if other_seed is not None and other_seed not in recent:
        recent = [other_seed, *recent[-7:]]
    plans = []
    for attempt in attempts[-6:]:
        context = attempt["planning_request"]["payload"]["context"]
        if context["revision"] >= cutoff:
            continue
        prior = context["records"][0]["evidence"]
        plan, candidate = attempt["plan"], attempt["candidate"]
        outcomes = [
            fact(r)
            for r in executions
            if candidate is not None
            and r.source_sha256 == candidate["sha256"]
            and r.started_revision > context["revision"]
        ]
        plans.append(
            {
                "failure_execution_id": prior["execution_id"],
                "base_source_sha256": plan["base_source_sha256"] if plan else None,
                "candidate_source_sha256": candidate["sha256"] if candidate else None,
                "level": plan["level"] if plan else None,
                "issue": attempt["issue"],
                "diagnosis_hypothesis": plan["diagnosis"][:2048] if plan else None,
                "expected_effect_hypothesis": plan["expected_effect"][:2048] if plan else None,
                "changes": [
                    {**change, "decision": change["decision"][:1024]}
                    for change in plan["changes"][:8]
                ]
                if plan
                else [],
                "summary_truncated": bool(
                    plan
                    and (
                        len(plan["diagnosis"]) > 2048
                        or len(plan["expected_effect"]) > 2048
                        or len(plan["changes"]) > 8
                        or any(len(c["decision"]) > 1024 for c in plan["changes"])
                    )
                ),
                "candidate_execution_outcomes": outcomes[-4:],
            }
        )
    return {
        "through_failure_revision": cutoff,
        "completed_search_executions": len(executions),
        "successful_search_executions": sum(r.result.success for r in executions),
        "consecutive_search_failures": len(streak),
        "consecutive_failure_seeds": sorted({r.seed for r in streak}),
        "initial_source_lines": len(
            state.programs[state.initial_source_sha256].source.splitlines()
        ),
        "working_source_lines": len(
            state.programs[state.working_source_sha256].source.splitlines()
        ),
        "accepted_source_lines": len(
            state.programs[state.accepted_source_sha256].source.splitlines()
        ),
        "recent_executions": [fact(r) for r in recent],
        "recent_repairs": plans,
        "meaning": "Outcomes are ledger facts; prior diagnoses and expected effects are hypotheses. "
        "Line count is not a quality score. Missing candidate outcomes mean untested, not successful.",
    }


def retrieve_repair_context(
    state: CampaignState,
    failure_execution_id: str,
    *,
    graph: DiagnosticGraph,
    evidence: Mapping[str, ExecutionEvidence],
    archive_root: Path,
    assessments: Mapping[str, CriticReport] | None = None,
    replay_reports: Sequence[ReplayReport] = (),
    diagnostic_execution_ids: Sequence[str] = (),
    repair_attempts: Sequence[Mapping] = (),
) -> RepairContext:
    """Select a failure and latest accepted-source search success, plus feedback.

    Missing selected evidence is an error, never a reason to silently omit a
    success. Unselected evidence is not read. Scheduling/staleness of a failure
    relative to the working program is explicitly exposed, and belongs to M7.
    """
    state.validate()
    if state.pending_execution is not None:
        raise ValueError("resolve the pending execution before repair")
    protocol = state.identity.protocol
    if (graph.suite, graph.task, graph.api_sha256) != (
        protocol.suite,
        protocol.task,
        state.identity.api_sha256,
    ):
        raise ValueError("graph/campaign identity mismatch")
    failure = state.executions[failure_execution_id]
    if failure.result is None or failure.result.outcome not in (
        ExecutionOutcome.TASK_FAILURE,
        ExecutionOutcome.PROGRAM_ERROR,
    ):
        raise ValueError("repair requires a completed task/program failure")
    assessments = {} if assessments is None else assessments
    selected = [("current_failure", failure_execution_id)]
    if state.latest_accepted_success_id is not None:
        selected.append(("accepted_success", state.latest_accepted_success_id))
    if len(diagnostic_execution_ids) > 3 or len(set(diagnostic_execution_ids)) != len(
        diagnostic_execution_ids
    ):
        raise ValueError("expected at most three distinct diagnostic records")
    for execution_id in diagnostic_execution_ids:
        record = state.executions[execution_id]
        if record.role is not ExecutionRole.DIAGNOSTIC:
            raise ValueError("diagnostic feedback must reference a diagnostic execution")
        if execution_id == failure_execution_id:
            continue
        selected.append(("environment_diagnostic", execution_id))
    records, frames, citation_catalog = [], [], []
    for selection, execution_id in selected:
        record = state.executions[execution_id]
        protocol.require_development_seed(record.seed)
        if record.status is not ExecutionStatus.COMPLETED:
            raise ValueError("retrieval requires completed executions")
        item = evidence[execution_id]
        if (
            item.protocol != protocol
            or item.execution != record
            or item.program != state.programs[record.source_sha256]
            or item.api_sha256 != state.identity.api_sha256
        ):
            raise ValueError("evidence/archive identity mismatch")
        verified = _request(item, archive_root, graph)
        frames.extend(verified.frames)

        def cite(evidence_id, step, kind, record=record):
            citation_catalog.append(
                {
                    "execution_id": record.execution_id,
                    "evidence_id": evidence_id,
                    "step": step,
                    "kind": kind,
                    "source_sha256": record.source_sha256,
                }
            )

        cite("$outcome", None, "execution_outcome")
        for public_item in item.items:
            cite(public_item.evidence_id, public_item.step, public_item.kind)
        raw_text = []
        for ref in item.raw_evidence:
            # A tape manifest is not a human-readable full trace; its public
            # events/summaries are already projected in ExecutionEvidence.items.
            if ref.kind != "tape":
                raw_id = f"$raw_{len(raw_text)}"
                cite(raw_id, None, ref.kind)
                raw_text.append(
                    {
                        "evidence_id": raw_id,
                        "artifact": ref.path,
                        "sha256": ref.sha256,
                        "kind": ref.kind,
                        "text": read_public_artifact(archive_root, ref).decode("utf-8"),
                    }
                )
        annotation = None
        if execution_id in assessments:
            report = assessments[execution_id]
            if report.evidence.sha256 != item.sha256 or report.graph != graph:
                raise ValueError("assessment/evidence identity mismatch")
            annotation = report.to_dict()
            for key in ("request", "evidence", "graph"):
                annotation.pop(key)
            if not report.evidence_verified:
                annotation["usable"] = False
                annotation["usable_frontier"] = ()
        records.append(
            {
                "selection": selection,
                "execution_role": record.role.value,
                "evidence_sha256": item.sha256,
                "evidence": item.to_dict(),
                "raw_public_text": raw_text,
                "assessment": annotation,
                "guidance": "frontier" if annotation and annotation["usable_frontier"] else "raw",
            }
        )
    feedback = []
    if len(replay_reports) > 3:
        raise ValueError("expected at most three replay reports")
    seen = set()
    for report in replay_reports:
        witness = state.executions[report.witness_execution_id]
        protocol.require_development_seed(witness.seed)
        if (
            witness.result is None
            or not witness.result.success
            or not witness.role.is_search
            or report.witness_source_sha256 != witness.source_sha256
            or report.candidate_sha256 not in state.programs
            or report.api_sha256 != state.identity.api_sha256
            or report.runtime_sha256 != state.identity.runtime_sha256
            or report.tape_sha256 != (witness.result.tape.sha256 if witness.result.tape else None)
        ):
            raise ValueError("replay/archive identity mismatch")
        status = ReplayStatus(report.status)
        if status is ReplayStatus.CERTIFIED and (
            not report.full_match or not report.normal_termination or report.tape_sha256 is None
        ):
            raise ValueError("invalid certification feedback")
        key = (report.candidate_sha256, report.witness_execution_id)
        if key in seen:
            raise ValueError("duplicate replay feedback")
        seen.add(key)
        feedback_id = f"$replay_{len(feedback)}"
        citation_catalog.append(
            {
                "execution_id": witness.execution_id,
                "evidence_id": feedback_id,
                "step": None,
                "kind": "offline_replay",
                "source_sha256": witness.source_sha256,
                "candidate_sha256": report.candidate_sha256,
            }
        )
        feedback.append(
            {
                "evidence_id": feedback_id,
                "kind": "offline_replay",
                "report": report.to_dict(),
                "seed": witness.seed,
                "candidate_is_working": report.candidate_sha256 == state.working_source_sha256,
                "candidate_is_accepted": report.candidate_sha256 == state.accepted_source_sha256,
                "meaning": {
                    ReplayStatus.CERTIFIED: "certificate_for_recorded_instance_only",
                    ReplayStatus.DIVERGED: "behavior_differs_not_a_task_failure",
                    ReplayStatus.UNAVAILABLE: "no_certificate_no_task_outcome",
                }[status],
            }
        )
    payload = {
        "format": "scope-repair-input",
        "version": 1,
        "campaign_id": state.identity.campaign_id,
        "revision": state.revision,
        "graph": graph.to_dict(),
        "graph_sha256": graph.sha256,
        "working_program": {
            "source": state.programs[state.working_source_sha256].source,
            "sha256": state.working_source_sha256,
        },
        "accepted_program": {
            "source": state.programs[state.accepted_source_sha256].source,
            "sha256": state.accepted_source_sha256,
        },
        "failure_matches_working": failure.source_sha256 == state.working_source_sha256,
        "search_history": _search_history(state, failure, repair_attempts),
        "records": records,
        "replay_feedback": feedback,
        # Copyable output objects match the plan schema exactly. Descriptive
        # metadata is separate so copying it cannot invalidate a citation.
        "citation_catalog": [
            {key: item[key] for key in ("execution_id", "evidence_id", "step")}
            for item in citation_catalog
        ],
        "citation_metadata": citation_catalog,
    }
    encoded = json_text(payload)
    warn_budget("repair context bytes", len(encoded.encode()), MAX_REQUEST_BYTES)
    warn_budget("repair frame attachment bytes", sum(len(f.data) for f in frames), MAX_FRAME_BYTES)
    return RepairContext(
        ModelRequest("repair_context", "", encoded, tuple(frames)), digest(state.to_dict())
    )
