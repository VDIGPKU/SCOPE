"""Task-shared diagnostic DAG and public evidence identities for SCOPE M5.

The host checks structure and provenance; it does not infer physical truth from
a node description or a model judgment. No model/simulator imports are needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .protocol import ScopeProtocol, require_name
from .records import EvidenceRef, ExecutionRecord, ExecutionStatus, ProgramVersion, require_hash

MAX_NODES = 128
MAX_TEXT = 128 * 1024


def warn_budget(label: str, size: int, limit: int) -> None:
    """Data budgets are advisory; log counts without exposing payload contents."""
    if size > limit:
        logging.getLogger(__name__).warning(
            "SCOPE budget warning: %s (%d > %d); continuing without truncation",
            label, size, limit,
        )


def json_text(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )


def digest(value: Any) -> str:
    return hashlib.sha256(json_text(value).encode("utf-8")).hexdigest()


def require_text(value, label, *, maximum=MAX_TEXT):
    if type(value) is not str or not value.strip():
        raise ValueError(f"invalid {label}")
    warn_budget(label + " bytes", len(value.encode("utf-8", errors="surrogatepass")), maximum)


def parse_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError("nonfinite JSON value")

    return json.loads(text, object_pairs_hook=unique, parse_constant=nonfinite)


def parse_model_json(text):
    """Accept a whole JSON reply optionally wrapped in one Markdown fence.

    Callers retain the original response verbatim. Never extract a JSON fragment
    from prose, and keep persisted artifacts on the strict parse_json path.
    """
    stripped = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", stripped, re.DOTALL | re.IGNORECASE
    )
    return parse_json(fenced.group(1) if fenced else stripped)


def fields(value, expected):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError("unexpected JSON fields")


def save_artifact(path: Path, value) -> None:
    """New file only; references can use the file hash in the M1 evidence ledger."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json_text({"sha256": digest(value), "data": value}) + "\n")


@dataclass(frozen=True)
class CheckNode:
    node_id: str
    check: str
    parents: tuple[str, ...] = ()
    invariant: bool = False

    def __post_init__(self):
        require_name(self.node_id, "node ID")
        require_text(self.check, "observable check", maximum=4096)
        if type(self.invariant) is not bool or type(self.parents) not in (tuple, list):
            raise ValueError("invalid node fields")
        object.__setattr__(self, "parents", tuple(self.parents))
        for parent in self.parents:
            require_name(parent, "parent ID")
        if len(set(self.parents)) != len(self.parents) or self.node_id in self.parents:
            raise ValueError("duplicate parent or self edge")


@dataclass(frozen=True)
class DiagnosticGraph:
    suite: str
    task: str
    api_sha256: str
    nodes: tuple[CheckNode, ...]

    def __post_init__(self):
        ScopeProtocol(self.suite, self.task)
        require_hash(self.api_sha256)
        if type(self.nodes) not in (tuple, list) or not self.nodes:
            raise ValueError("graph must have at least one node")
        warn_budget("graph nodes", len(self.nodes), MAX_NODES)
        object.__setattr__(self, "nodes", tuple(self.nodes))
        if any(not isinstance(node, CheckNode) for node in self.nodes):
            raise ValueError("graph nodes must be CheckNode objects")
        ids = {node.node_id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate node ID")
        if any(not set(node.parents) <= ids for node in self.nodes):
            raise ValueError("missing prerequisite node")
        self.topological_order()

    def topological_order(self) -> tuple[str, ...]:
        remaining = list(self.nodes)
        ordered = []
        visited = set()
        while remaining:
            ready = [node for node in remaining if set(node.parents) <= visited]
            if not ready:
                raise ValueError("diagnostic graph contains a cycle")
            for node in ready:
                ordered.append(node.node_id)
                visited.add(node.node_id)
                remaining.remove(node)
        return tuple(ordered)

    def ancestors(self) -> dict[str, frozenset[str]]:
        nodes = {node.node_id: node for node in self.nodes}
        result = {}
        for node_id in self.topological_order():
            parents = nodes[node_id].parents
            result[node_id] = frozenset(parents).union(*(result[parent] for parent in parents))
        return result

    def to_dict(self):
        return {
            "format": "scope-diagnostic-graph",
            "version": 1,
            "suite": self.suite,
            "task": self.task,
            "api_sha256": self.api_sha256,
            "nodes": [{**asdict(node), "parents": list(node.parents)} for node in self.nodes],
        }

    @property
    def sha256(self):
        return digest(self.to_dict())

    def save(self, path: Path):
        save_artifact(path, self.to_dict())

    @classmethod
    def from_dict(cls, value):
        fields(value, ("format", "version", "suite", "task", "api_sha256", "nodes"))
        if (
            value["format"] != "scope-diagnostic-graph"
            or type(value["version"]) is not int
            or value["version"] != 1
        ):
            raise ValueError("unsupported graph format")
        if type(value["nodes"]) is not list:
            raise ValueError("nodes must be a list")
        for node in value["nodes"]:
            fields(node, ("node_id", "check", "parents", "invariant"))
        return cls(
            value["suite"],
            value["task"],
            value["api_sha256"],
            tuple(CheckNode(**node) for node in value["nodes"]),
        )

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        if path.is_symlink():
            raise ValueError("invalid graph file")
        warn_budget("graph file bytes", path.stat().st_size, 1024 * 1024)
        envelope = parse_json(path.read_text(encoding="utf-8"))
        fields(envelope, ("sha256", "data"))
        if digest(envelope["data"]) != envelope["sha256"]:
            raise ValueError("graph checksum mismatch")
        return cls.from_dict(envelope["data"])


@dataclass(frozen=True)
class PublicEvidence:
    """A trusted adapter's catalog entry; the critic may only cite this identity.

    Step 0 is entry; steps 1..N are ordered public API events. An image's step is
    supplied by the capture adapter, not inferred from filenames or model text.
    """

    evidence_id: str
    execution_id: str
    step: int
    kind: str
    text: str = ""
    artifact: EvidenceRef | None = None
    mime_type: str | None = None

    def __post_init__(self):
        require_name(self.evidence_id, "evidence ID")
        require_name(self.execution_id, "evidence execution ID")
        if (
            type(self.step) is not int
            or self.step < 0
            or self.kind not in ("frame", "api", "observation", "error")
        ):
            raise ValueError("invalid public evidence step/kind")
        if type(self.text) is not str:
            raise ValueError("invalid public evidence text")
        warn_budget("public evidence text characters", len(self.text), MAX_TEXT)
        if self.kind == "frame":
            if not isinstance(self.artifact, EvidenceRef) or self.artifact.kind != "public_frame":
                raise ValueError("frame requires an explicit public_frame artifact")
            if self.mime_type not in ("image/png", "image/jpeg", "image/webp"):
                raise ValueError("unsupported public frame MIME type")
        elif self.artifact is not None or self.mime_type is not None or not self.text.strip():
            raise ValueError("non-frame evidence requires public text, not arbitrary files")


@dataclass(frozen=True)
class ExecutionEvidence:
    protocol: ScopeProtocol
    execution: ExecutionRecord
    program: ProgramVersion
    api_sha256: str
    task_language: str
    api_specification: str
    max_step: int
    items: tuple[PublicEvidence, ...]
    raw_evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self):
        self.protocol.require_development_seed(self.execution.seed)
        if self.execution.status is ExecutionStatus.RESERVED:
            raise ValueError("diagnostics require a completed or interrupted execution")
        if self.program.sha256 != self.execution.source_sha256:
            raise ValueError("evidence source does not match execution")
        require_hash(self.api_sha256)
        require_text(self.task_language, "public task language")
        require_text(self.api_specification, "public API specification")
        if type(self.max_step) is not int or not 0 <= self.max_step <= 10000:
            raise ValueError("invalid execution step range")
        if type(self.items) not in (tuple, list) or len(self.items) > 20000:
            raise ValueError("invalid evidence items")
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "raw_evidence", tuple(self.raw_evidence))
        ids = set()
        api_steps = []
        for item in self.items:
            if (
                not isinstance(item, PublicEvidence)
                or item.execution_id != self.execution.execution_id
            ):
                raise ValueError("evidence belongs to another execution")
            if item.evidence_id in ids or item.step > self.max_step:
                raise ValueError("duplicate evidence ID or out-of-range step")
            if item.kind == "frame" and (
                self.execution.result is None or item.artifact not in self.execution.result.evidence
            ):
                raise ValueError("frame is not registered in this M1 execution")
            ids.add(item.evidence_id)
            if item.kind == "api":
                api_steps.append(item.step)
        if sorted(api_steps) != list(range(1, self.max_step + 1)):
            raise ValueError("public API steps must be unique and contiguous")
        if any(
            not isinstance(ref, EvidenceRef)
            or ref.kind not in ("tape", "public_trace", "public_observation", "public_error")
            for ref in self.raw_evidence
        ):
            raise ValueError("raw evidence must reference explicit public artifacts")
        if any(
            self.execution.result is None or ref not in self.execution.result.evidence
            for ref in self.raw_evidence
        ):
            raise ValueError("raw artifact is not registered in this M1 execution")

    def to_dict(self):
        # Explicit fields prevent private adapter/env state reaching a prompt.
        result = self.execution.result
        return {
            "suite": self.protocol.suite,
            "task": self.protocol.task,
            "execution_id": self.execution.execution_id,
            "seed": self.execution.seed,
            "source_sha256": self.program.sha256,
            "source": self.program.source,
            "api_sha256": self.api_sha256,
            "task_language": self.task_language,
            "api_specification": self.api_specification,
            "max_step": self.max_step,
            "outcome": None if result is None else result.outcome.value,
            "termination": None if result is None else result.termination.value,
            "error_code": None if result is None else result.error_code,
            "items": [asdict(item) for item in self.items],
            "raw_evidence": [asdict(ref) for ref in self.raw_evidence],
        }

    @property
    def sha256(self):
        return digest(self.to_dict())

    @classmethod
    def from_tape(
        cls,
        *,
        protocol,
        execution,
        program,
        tape,
        task_language,
        api_specification,
        public_items=(),
        summaries=None,
    ):
        """Project a verified M3 tape into critic-visible API events.

        Image sampling and semantic summaries are explicit adapter inputs; no
        simulator state is inspected and no replay-ready requirement is imposed.
        """
        manifest = tape.manifest
        identity = manifest["identity"]
        if (
            identity["execution_id"] != execution.execution_id
            or identity["source_sha256"] != program.sha256
        ):
            raise ValueError("tape execution/source mismatch")
        ref = execution.result.tape if execution.result else None
        if ref is None or ref.sha256 != tape.sha256:
            raise ValueError("tape is not referenced by this execution")
        summaries = {} if summaries is None else summaries
        events = manifest["events"]
        if any(
            type(step) is not int or not 1 <= step <= len(events) or type(text) is not str
            for step, text in summaries.items()
        ):
            raise ValueError("invalid public API summaries")
        items = tuple(
            PublicEvidence(
                f"api_{index + 1:06d}",
                execution.execution_id,
                index + 1,
                "api",
                json_text(
                    {
                        "kind": event["kind"],
                        "name": event["name"],
                        "outcome": event["outcome"],
                        "summary": summaries.get(index + 1, ""),
                    }
                ),
            )
            for index, event in enumerate(events)
        ) + tuple(public_items)
        return cls(
            protocol,
            execution,
            program,
            identity["api_sha256"],
            task_language,
            api_specification,
            len(events),
            items,
            (ref,),
        )


@dataclass(frozen=True)
class GraphContext:
    """Entry-only context, usable before program execution or evaluator feedback.

    The trusted observation capture adapter assigns provenance. No source,
    outcome, later frames or runtime artifacts are accepted by this interface.
    """

    protocol: ScopeProtocol
    execution_id: str
    seed: int
    api_sha256: str
    task_language: str
    api_specification: str
    items: tuple[PublicEvidence, ...]

    def __post_init__(self):
        self.protocol.require_development_seed(self.seed)
        require_name(self.execution_id, "entry execution ID")
        require_hash(self.api_sha256)
        require_text(self.task_language, "task language")
        require_text(self.api_specification, "API specification")
        if type(self.items) not in (tuple, list) or not 1 <= len(self.items) <= 256:
            raise ValueError("planning requires initial public observations")
        object.__setattr__(self, "items", tuple(self.items))
        ids = set()
        for item in self.items:
            if (
                not isinstance(item, PublicEvidence)
                or item.step != 0
                or item.kind not in ("observation", "frame")
            ):
                raise ValueError("graph planning accepts only entry observations/frames")
            if item.execution_id != self.execution_id or item.evidence_id in ids:
                raise ValueError("invalid initial observation provenance")
            ids.add(item.evidence_id)

    def to_dict(self):
        return {
            "suite": self.protocol.suite,
            "task": self.protocol.task,
            "execution_id": self.execution_id,
            "seed": self.seed,
            "api_sha256": self.api_sha256,
            "task_language": self.task_language,
            "api_specification": self.api_specification,
            "initial_evidence": [asdict(item) for item in self.items],
        }

    @property
    def sha256(self):
        return digest(self.to_dict())

    @classmethod
    def from_execution(cls, evidence: ExecutionEvidence):
        return cls(
            evidence.protocol,
            evidence.execution.execution_id,
            evidence.execution.seed,
            evidence.api_sha256,
            evidence.task_language,
            evidence.api_specification,
            tuple(
                item
                for item in evidence.items
                if item.step == 0 and item.kind in ("observation", "frame")
            ),
        )


def read_public_artifact(root: Path, ref: EvidenceRef, *, limit=16 * 1024 * 1024) -> bytes:
    root = Path(root).resolve()
    path = root
    for component in Path(ref.path).parts:
        path = path / component
        if path.is_symlink():
            raise ValueError("symlinked public artifact")
    if not path.is_file():
        raise ValueError("missing public artifact")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != ref.sha256:
        raise ValueError("public artifact checksum mismatch")
    warn_budget("public artifact bytes", len(data), limit)
    return data
