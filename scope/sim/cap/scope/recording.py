"""Stateful, serial API recording at an explicitly declared public boundary.

No simulator imports, monkey patches, program replay, or success certification.
Recording failures invalidate the tape without replacing a program's result or
exception. A journal published before each call leaves interrupted calls pending.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .protocol import require_name
from .records import EvidenceRef, require_hash
from .state_codec import (
    DEFAULT_LIMITS,
    CodecLimits,
    InvalidState,
    PublicObjectSpec,
    StateEncoder,
    StateRestorer,
    StateSnapshot,
    UnsupportedState,
)

FORMAT = "scope-api-tape"
VERSION = 1
# These exact builtin types have an args-only contract. No dynamic imports or
# arbitrary exception constructors are needed by the future replay consumer.
SUPPORTED_EXCEPTIONS = (
    Exception,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    RuntimeError,
    AssertionError,
    ZeroDivisionError,
    NotImplementedError,
    OverflowError,
)
EXCLUDED_BEHAVIORS = (
    "unwrapped external calls, file/network/clock/random inputs",
    "unregistered or asynchronously changing public state and dynamic properties",
    "threads, async code, generators, and callbacks across the API boundary",
    "exception identity, retention/mutation, traceback/frame inspection, chaining, and custom attributes",
    "wrapper introspection, numeric object IDs, finalizers, and weak references",
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False)


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidState("duplicate tape JSON key")
        result[key] = value
    return result


@dataclass(frozen=True)
class TapeIdentity:
    execution_id: str
    source_sha256: str
    api_sha256: str
    runtime_sha256: str

    def __post_init__(self):
        require_name(self.execution_id, "execution ID")
        for digest in (self.source_sha256, self.api_sha256, self.runtime_sha256):
            require_hash(digest)


@dataclass(frozen=True)
class BoundarySpec:
    """A trusted adapter's declaration, never inferred from wrapping functions.

    declared_complete asserts that this program stays within the stated domain
    and that inputs/public state/endpoints cover its relevant external boundary.
    """

    profile: str
    declared_complete: bool = False

    def __post_init__(self):
        require_name(self.profile, "boundary profile")
        if type(self.declared_complete) is not bool:
            raise ValueError("declared_complete must be bool")


@dataclass(frozen=True)
class TapeLimits:
    max_events: int = 10000
    max_storage_bytes: int = 1024 * 1024 * 1024
    max_manifest_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("tape limits must be positive integers")


DEFAULT_TAPE_LIMITS = TapeLimits()


def _complete(manifest):
    return (
        manifest["status"] == "finished"
        and manifest["termination"] == "normal"
        and not manifest["issues"]
        and manifest["entry"] is not None
        and manifest["terminal"] is not None
        and all(
            e["pre"] is not None
            and e["post"] is not None
            and e["outcome"] in ("return", "exception")
            for e in manifest["events"]
        )
    )


class TapeRecorder:
    """One execution, with one encoder shared by entry, calls, and termination.

    Pass only program-visible data as inputs/public_state, not the API receiver
    or simulator. Readers are zero-argument public getters (e.g. a dynamic
    property); they are explicit events. Plain registered fields can be read
    directly when they change only in deterministic program code or wrapped APIs.
    """

    def __init__(
        self,
        directory: Path,
        *,
        identity: TapeIdentity,
        boundary: BoundarySpec,
        functions: dict[str, Callable],
        inputs: dict[str, Any] | None = None,
        public_state: dict[str, Any] | None = None,
        readers: dict[str, Callable] | None = None,
        specs: tuple[PublicObjectSpec, ...] = (),
        codec_limits: CodecLimits = DEFAULT_LIMITS,
        limits: TapeLimits = DEFAULT_TAPE_LIMITS,
    ):
        if not isinstance(identity, TapeIdentity) or not isinstance(boundary, BoundarySpec):
            raise ValueError("identity and boundary must be explicit typed declarations")
        inputs = {} if inputs is None else inputs
        public_state = {} if public_state is None else public_state
        readers = {} if readers is None else readers
        for mapping in (inputs, public_state, functions, readers):
            if type(mapping) is not dict or any(type(k) is not str for k in mapping):
                raise ValueError("boundary mappings must be plain dictionaries with string keys")
            for name in mapping:
                require_name(name, "boundary name")
        if any(not callable(fn) for fn in (*functions.values(), *readers.values())):
            raise ValueError("endpoints must be callable")
        if set(functions) & set(readers):
            raise ValueError("functions and readers must have distinct names")
        self.directory = Path(directory)
        self._encoder = StateEncoder(specs=specs, limits=codec_limits)
        self._validator = StateRestorer(specs=specs, limits=codec_limits)
        self._codec_limits = codec_limits
        self._limits = limits
        self._inputs, self._public = inputs, public_state
        self._owner: int | None = None
        self._depth = 0
        self._snapshot_count = 0
        self._storage_bytes = 0
        self._disabled = False
        self._io_failed = False
        self._manifest = {
            "format": FORMAT,
            "version": VERSION,
            "identity": asdict(identity),
            "boundary": {**asdict(boundary), "exclusions": list(EXCLUDED_BEHAVIORS)},
            "endpoints": [
                {"name": name, "kind": kind}
                for kind, mapping in (("call", functions), ("read", readers))
                for name in sorted(mapping)
            ],
            "status": "created",
            "termination": None,
            "entry": None,
            "events": [],
            "terminal": None,
            "terminal_exception": None,
            "issues": [],
            "complete": False,
            "replay_ready": False,
        }
        self.functions = MappingProxyType(
            {n: self._wrap(n, f, "call") for n, f in functions.items()}
        )
        self.readers = MappingProxyType({n: self._wrap(n, f, "read") for n, f in readers.items()})
        # Setup errors are raised before any user program or API is executed.
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "snapshots").mkdir()
        self._publish()

    @property
    def manifest(self) -> dict[str, Any]:
        """Detached live metadata; disk may be older after a storage failure."""
        self._flags()
        return json.loads(_json(self._manifest))

    def _flags(self):
        self._manifest["complete"] = _complete(self._manifest)
        self._manifest["replay_ready"] = (
            self._manifest["complete"] and self._manifest["boundary"]["declared_complete"]
        )

    def _issue(self, stage, code, event=None):
        self._disabled = True
        # Record codes, not arbitrary exception messages or object reprs.
        row = {"stage": stage, "code": code, "event": event}
        if row not in self._manifest["issues"]:
            self._manifest["issues"].append(row)

    def mark_unavailable(self, code: str = "uncovered_external_input") -> None:
        """Trusted adapter reports a discovered boundary gap during execution."""
        require_name(code, "unavailable reason")
        self._issue("boundary", code)
        self._checkpoint()

    def _publish(self):
        self._flags()
        payload = (_json(self._manifest) + "\n").encode("utf-8")
        if len(payload) > self._limits.max_manifest_bytes:
            raise UnsupportedState("tape manifest size limit exceeded")
        temporary = self.directory / ".tape.json.tmp"
        with temporary.open("wb") as stream:
            stream.write(payload)
        temporary.replace(self.directory / "tape.json")

    def _checkpoint(self):
        if self._io_failed:
            return
        try:
            self._publish()
        except Exception:
            self._io_failed = True
            self._issue("storage", "manifest_write_failed")

    def _capture(self, roots, stage, event=None):
        if self._disabled:
            return None
        try:
            snapshot = self._encoder.capture(roots)
            # Capture alone may allow transitions (e.g. bytearray resizing)
            # that cannot be restored in place. Validate detached restoration.
            self._validator.restore(snapshot)
            # Actual storage accounting, including JSON envelope and duplicate
            # payloads across snapshots (M2 snapshots remain independently usable).
            metadata_size = len(
                (
                    json.dumps(
                        {"sha256": snapshot.sha256, "state": snapshot.metadata},
                        indent=2,
                        ensure_ascii=True,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            size = metadata_size + sum(map(len, snapshot.blobs.values()))
            if self._storage_bytes + size > self._limits.max_storage_bytes:
                raise UnsupportedState("tape storage limit exceeded")
            relative = f"snapshots/{self._snapshot_count:06d}"
            snapshot.save(self.directory / relative, limits=self._codec_limits)
            self._snapshot_count += 1
            self._storage_bytes += size
            return {"path": relative, "sha256": snapshot.sha256}
        except Exception as error:
            self._issue(stage, type(error).__name__, event)
            return None

    def _roots(self, args, kwargs):
        return {"inputs": self._inputs, "public": self._public, "args": args, "kwargs": kwargs}

    def _exception(self, error, stage, event=None):
        cls = type(error)
        supported = (
            cls in SUPPORTED_EXCEPTIONS
            and not error.__dict__
            and error.__cause__ is None
            and error.__context__ is None
            and not error.__suppress_context__
        )
        if not supported:
            self._issue(stage, "unsupported_exception", event)
        return {"module": cls.__module__, "name": cls.__qualname__, "supported": supported}

    def _wrap(self, name, fn, kind):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            if self._manifest["status"] != "running":
                raise RuntimeError("recorded endpoints require an active recorder.run()")
            if threading.get_ident() != self._owner:
                self._issue("boundary", "concurrent_call")
                return fn(*args, **kwargs)
            if self._depth:
                return fn(*args, **kwargs)  # Internal implementation, not a second public event.
            if self._disabled:
                return fn(*args, **kwargs)
            index = len(self._manifest["events"])
            if index >= self._limits.max_events:
                self._issue("pre", "event_limit", index)
                self._checkpoint()
                return fn(*args, **kwargs)
            if kind == "read" and (args or kwargs):
                self._issue("pre", "reader_has_arguments", index)
            roots = self._roots(args, kwargs)
            event = {
                "index": index,
                "kind": kind,
                "name": name,
                "outcome": "pending",
                "pre": self._capture(roots, "pre", index),
                "post": None,
                "exception": None,
            }
            self._manifest["events"].append(event)
            self._checkpoint()  # Publish pending call BEFORE invoking the real API.
            self._depth += 1
            try:
                result = fn(*args, **kwargs)
            except BaseException as error:
                event["outcome"] = "exception"
                event["exception"] = self._exception(error, "exception", index)
                event["post"] = self._capture(
                    {**roots, "exception_args": BaseException.args.__get__(error)},
                    "post",
                    index,
                )
                raise  # Keep the original exception object and its mutations.
            else:
                event["outcome"] = "return"
                event["post"] = self._capture({**roots, "result": result}, "post", index)
                return result
            finally:
                self._depth -= 1
                self._checkpoint()

        return wrapped

    def run(self, program: Callable, *args, **kwargs):
        """Execute one complete synchronous program; propagate its exact result.

        The host supplies the entry callable and fingerprints. This layer does
        not execute source text or inspect globals/closures to infer inputs.
        """
        if self._manifest["status"] != "created":
            raise RuntimeError("a recorder can execute only one program")
        if not callable(program):
            raise TypeError("program must be callable")
        self._owner = threading.get_ident()
        self._manifest["status"] = "running"
        roots = self._roots(args, kwargs)
        self._manifest["entry"] = self._capture(roots, "entry")
        self._checkpoint()
        try:
            result = program(*args, **kwargs)
        except BaseException as error:
            self._manifest["termination"] = (
                "exception" if isinstance(error, Exception) else "interrupted"
            )
            self._manifest["terminal_exception"] = self._exception(error, "terminal")
            self._manifest["terminal"] = self._capture(
                {**roots, "exception_args": BaseException.args.__get__(error)},
                "terminal",
            )
            raise
        else:
            self._manifest["termination"] = "normal"
            self._manifest["terminal"] = self._capture({**roots, "result": result}, "terminal")
            return result
        finally:
            self._manifest["status"] = "finished"
            self._checkpoint()


class RecordedTape:
    """Validated, detached tape and snapshots prepared for offline replay."""

    def __init__(self, manifest, snapshots, path, digest):
        self._json = _json(manifest)
        self._snapshots = MappingProxyType(snapshots)
        self.path = path
        self.sha256 = digest

    @property
    def manifest(self):
        return json.loads(self._json)

    @property
    def replay_ready(self):
        return self.manifest["replay_ready"]

    def snapshot(self, reference: dict[str, str]) -> StateSnapshot:
        snapshot = self._snapshots[reference["path"]]
        if snapshot.sha256 != reference["sha256"]:
            raise InvalidState("snapshot reference checksum mismatch")
        return snapshot

    def evidence_ref(self, archive_root: Path) -> EvidenceRef:
        """M1 reference; tape may still be incomplete/unavailable."""
        relative = self.path.resolve().relative_to(Path(archive_root).resolve()).as_posix()
        return EvidenceRef("tape", relative, self.sha256)

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        specs: tuple[PublicObjectSpec, ...] = (),
        codec_limits: CodecLimits = DEFAULT_LIMITS,
        limits: TapeLimits = DEFAULT_TAPE_LIMITS,
        expected_sha256: str | None = None,
    ) -> RecordedTape:
        directory = Path(directory)
        path = directory / "tape.json"
        try:
            if (
                directory.is_symlink()
                or path.is_symlink()
                or path.stat().st_size > limits.max_manifest_bytes
            ):
                raise InvalidState("invalid tape manifest path/size")
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            if expected_sha256 is not None:
                require_hash(expected_sha256)
                if digest != expected_sha256:
                    raise InvalidState("tape manifest checksum mismatch")
            manifest = json.loads(payload, object_pairs_hook=_pairs)
            references = _validate_manifest(manifest, limits)
            snapshots = {}
            restorer = StateRestorer(specs=specs, limits=codec_limits)
            size = 0
            for index, (ref, expected_roots) in enumerate(references):
                if ref != {"path": f"snapshots/{index:06d}", "sha256": ref.get("sha256")}:
                    raise InvalidState("snapshot paths must follow capture order")
                require_hash(ref["sha256"])
                folder = directory / ref["path"]
                if (directory / "snapshots").is_symlink() or folder.is_symlink():
                    raise InvalidState("symlinked snapshot path")
                snapshot = StateSnapshot.load(folder, limits=codec_limits)
                if snapshot.sha256 != ref["sha256"]:
                    raise InvalidState("snapshot checksum mismatch")
                size += (folder / "state.json").stat().st_size + sum(
                    map(len, snapshot.blobs.values())
                )
                if size > limits.max_storage_bytes:
                    raise InvalidState("tape storage limit exceeded")
                if set(snapshot.metadata["roots"]) != expected_roots:
                    raise InvalidState("snapshot has incorrect boundary roots")
                roots = restorer.restore(snapshot)
                if type(roots["inputs"]) is not dict or type(roots["public"]) is not dict:
                    raise InvalidState("inputs/public state must be dictionaries")
                if type(roots["args"]) is not tuple or type(roots["kwargs"]) is not dict:
                    raise InvalidState("request arguments have incorrect types")
                if any(type(k) is not str for k in roots["kwargs"]):
                    raise InvalidState("keyword argument names must be strings")
                if "exception_args" in roots and type(roots["exception_args"]) is not tuple:
                    raise InvalidState("exception arguments must be a tuple")
                snapshots[ref["path"]] = snapshot
            _validate_root_identity(manifest, snapshots)
            return cls(manifest, snapshots, path, digest)
        except (OSError, ValueError, TypeError, KeyError, RecursionError) as error:
            if isinstance(error, InvalidState):
                raise
            raise InvalidState(f"invalid tape: {type(error).__name__}") from error


def _validate_root_identity(manifest, snapshots):
    """Root objects stay fixed; their contents may change across the execution."""

    def roots(ref):
        return snapshots[ref["path"]].metadata["roots"]

    def same(left, right, names):
        if left is not None and right is not None:
            a, b = roots(left), roots(right)
            if any(a[name] != b[name] for name in names):
                raise InvalidState("boundary root identity changed")

    entry = manifest["entry"]
    for event in manifest["events"]:
        for ref in (event["pre"], event["post"]):
            same(entry, ref, ("inputs", "public"))
        same(event["pre"], event["post"], ("args", "kwargs"))
    same(entry, manifest["terminal"], ("inputs", "public", "args", "kwargs"))


def _validate_manifest(m, limits):
    def check(condition, message):
        if not condition:
            raise InvalidState(message)

    check(
        type(m) is dict
        and set(m)
        == {
            "format",
            "version",
            "identity",
            "boundary",
            "endpoints",
            "status",
            "termination",
            "entry",
            "events",
            "terminal",
            "terminal_exception",
            "issues",
            "complete",
            "replay_ready",
        },
        "invalid tape fields",
    )
    check(
        m["format"] == FORMAT and type(m["version"]) is int and m["version"] == VERSION,
        "unsupported tape format/version",
    )
    TapeIdentity(**m["identity"])
    check(
        type(m["boundary"]) is dict
        and set(m["boundary"]) == {"profile", "declared_complete", "exclusions"},
        "invalid boundary fields",
    )
    BoundarySpec(m["boundary"]["profile"], m["boundary"]["declared_complete"])
    check(m["boundary"]["exclusions"] == list(EXCLUDED_BEHAVIORS), "unsupported boundary domain")
    check(type(m["endpoints"]) is list, "invalid endpoints")
    endpoints = set()
    for endpoint in m["endpoints"]:
        check(type(endpoint) is dict and set(endpoint) == {"name", "kind"}, "invalid endpoint")
        require_name(endpoint["name"], "endpoint")
        check(endpoint["kind"] in ("call", "read"), "invalid endpoint kind")
        check(endpoint["name"] not in endpoints, "duplicate endpoint")
        endpoints.add(endpoint["name"])
    endpoint_pairs = {(e["kind"], e["name"]) for e in m["endpoints"]}
    check(m["status"] in ("created", "running", "finished"), "invalid status")
    check(m["termination"] in (None, "normal", "exception", "interrupted"), "invalid termination")
    check(
        (m["status"] == "finished") == (m["termination"] is not None), "status/termination mismatch"
    )
    check(
        type(m["events"]) is list and len(m["events"]) <= limits.max_events, "invalid events/limit"
    )
    check(type(m["issues"]) is list, "invalid issues")
    for issue in m["issues"]:
        check(type(issue) is dict and set(issue) == {"stage", "code", "event"}, "invalid issue")
        require_name(issue["stage"], "stage")
        require_name(issue["code"], "issue code")
        check(
            issue["event"] is None
            or type(issue["event"]) is int
            and 0 <= issue["event"] <= len(m["events"]),
            "invalid issue event index",
        )
    roots = {"inputs", "public", "args", "kwargs"}
    references = []

    def reference(ref, expected):
        if ref is not None:
            check(
                type(ref) is dict and set(ref) == {"path", "sha256"}, "invalid snapshot reference"
            )
            references.append((ref, expected))

    def exception(info):
        check(
            type(info) is dict and set(info) == {"module", "name", "supported"}, "invalid exception"
        )
        check(
            type(info["module"]) is str
            and type(info["name"]) is str
            and type(info["supported"]) is bool,
            "invalid exception fields",
        )
        if info["supported"]:
            check(
                info["module"] == "builtins"
                and info["name"] in {c.__name__ for c in SUPPORTED_EXCEPTIONS},
                "unknown supported exception",
            )
        else:
            check(bool(m["issues"]), "unsupported exception requires unavailable issue")

    reference(m["entry"], roots)
    for index, event in enumerate(m["events"]):
        check(
            type(event) is dict
            and set(event) == {"index", "kind", "name", "outcome", "pre", "post", "exception"},
            "invalid event fields",
        )
        check(type(event["index"]) is int and event["index"] == index, "event order mismatch")
        check((event["kind"], event["name"]) in endpoint_pairs, "unknown endpoint")
        check(event["outcome"] in ("pending", "return", "exception"), "invalid event outcome")
        reference(event["pre"], roots)
        if event["outcome"] == "exception":
            exception(event["exception"])
        else:
            check(event["exception"] is None, "unexpected exception descriptor")
        if event["outcome"] == "pending":
            check(
                index == len(m["events"]) - 1
                and event["post"] is None
                and m["status"] == "running",
                "pending call must end a running tape",
            )
        else:
            reference(
                event["post"],
                roots | ({"result"} if event["outcome"] == "return" else {"exception_args"}),
            )
    if m["termination"] in ("exception", "interrupted"):
        exception(m["terminal_exception"])
    else:
        check(m["terminal_exception"] is None, "unexpected terminal exception")
    check(
        m["terminal"] is None or m["status"] == "finished", "unfinished tape has terminal snapshot"
    )
    reference(
        m["terminal"], roots | ({"result"} if m["termination"] == "normal" else {"exception_args"})
    )
    if m["status"] == "created":
        check(not m["events"] and m["entry"] is None, "created tape has execution data")
    check(type(m["complete"]) is bool and m["complete"] == _complete(m), "false completeness claim")
    check(
        type(m["replay_ready"]) is bool
        and m["replay_ready"] == (m["complete"] and m["boundary"]["declared_complete"]),
        "false replay readiness claim",
    )
    return references
