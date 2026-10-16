"""Private tape-only worker. Invoked by replay(), never with a real API object."""

from __future__ import annotations

import json
import math
import resource
import sys
from pathlib import Path
from types import MappingProxyType

from ._replay_policy import ProfileError, compile_source, execution_globals
from .recording import SUPPORTED_EXCEPTIONS, RecordedTape
from .records import ProgramVersion
from .replay import ReplayEnvironment, ReplayLimits, replay_runtime_fingerprint
from .state_codec import StateMismatch, StateRestorer, StateUnavailable


class _ReplayAbort(BaseException):
    """Separate from recorded API exceptions; failure remains sticky if suppressed."""


class _Endpoints(dict):
    def __init__(self, values, unavailable):
        super().__init__(values)
        self._unavailable = unavailable

    def __missing__(self, key):
        self._unavailable("undeclared_endpoint", actual=str(key)[:160])


class _Session:
    def __init__(self, tape, publish):
        self.tape, self.publish = tape, publish
        self.manifest = tape.manifest
        self.events = self.manifest["events"]
        self.restorer = StateRestorer()
        self.entry = self.restorer.restore(tape.snapshot(self.manifest["entry"]))
        self.consumed = 0
        self.fault = None
        self.normal = False
        self.phase = "entry"

    def result(self, *, finished=False):
        return {
            "status": "certified" if finished and self.fault is None else "unavailable",
            "reason": "full_match" if finished and self.fault is None else "running",
            "event_index": None,
            "phase": None,
            "path": None,
            "expected": None,
            "actual": None,
            **(self.fault or {}),
            "consumed_events": self.consumed,
            "total_events": len(self.events),
            "normal_termination": self.normal,
            "full_match": finished and self.normal and self.fault is None,
            "finished": finished,
        }

    def fail(self, status, reason, **details):
        if self.fault is None:
            self.fault = {
                "status": status,
                "reason": reason,
                "event_index": self.consumed,
                "phase": self.phase,
                **details,
            }
            self.publish(self.result())
        raise _ReplayAbort()

    def unavailable(self, reason, **details):
        self.fail("unavailable", reason, **details)

    def match(self, reference, roots):
        try:
            self.restorer.bind(self.tape.snapshot(reference), roots)
        except StateMismatch as error:
            self.fail(
                "diverged",
                "state_mismatch",
                path=error.path,
                expected=error.expected,
                actual=error.actual,
            )
        except StateUnavailable:
            self.unavailable("unsupported_candidate_state")

    def endpoint(self, name, kind):
        def call(*args, **kwargs):
            if self.fault is not None:
                raise _ReplayAbort()
            self.phase = "request"
            if self.consumed == len(self.events):
                self.fail("diverged", "extra_call", expected="end_of_tape", actual=f"{kind}:{name}")
            event = self.events[self.consumed]
            if (kind, name) != (event["kind"], event["name"]):
                self.fail(
                    "diverged",
                    "request_mismatch",
                    expected=f"{event['kind']}:{event['name']}",
                    actual=f"{kind}:{name}",
                )
            self.phase = "pre_state"
            roots = {
                "inputs": self.entry["inputs"],
                "public": self.entry["public"],
                "args": args,
                "kwargs": kwargs,
            }
            self.match(event["pre"], roots)
            self.phase = "post_state"
            try:
                restored = self.restorer.restore(self.tape.snapshot(event["post"]))
            except StateUnavailable:
                self.unavailable("post_state_unavailable")
            self.consumed += 1
            self.publish(self.result())
            self.phase = "program"
            if event["outcome"] == "exception":
                cls = next(
                    c for c in SUPPORTED_EXCEPTIONS if c.__name__ == event["exception"]["name"]
                )
                # Construction is limited to M3's exact args-only builtin types.
                error = cls(*restored["exception_args"])
                error.args = restored["exception_args"]
                raise error
            return restored["result"]

        return call

    def run(self, code, entrypoint):
        self.publish(self.result())
        table = {
            kind: MappingProxyType(
                _Endpoints(
                    {
                        e["name"]: self.endpoint(e["name"], kind)
                        for e in self.manifest["endpoints"]
                        if e["kind"] == kind
                    },
                    self.unavailable,
                )
            )
            for kind in ("call", "read")
        }
        namespace = execution_globals(
            table["call"],
            table["read"],
            self.entry["inputs"],
            self.entry["public"],
            self.unavailable,
        )
        try:
            exec(code, namespace)
            self.phase = "program"
            result = namespace[entrypoint](*self.entry["args"], **self.entry["kwargs"])
            self.normal = True
            if self.fault is not None:
                return self.result(finished=True)
            self.phase = "termination"
            if self.consumed != len(self.events):
                self.fail(
                    "diverged",
                    "unconsumed_tape",
                    expected=self.events[self.consumed]["name"],
                    actual="normal_termination",
                )
            # Conservative terminal contract: final public state and program return
            # must also match. Never restore terminal state to make a mismatch pass.
            self.match(self.manifest["terminal"], {**self.entry, "result": result})
        except _ReplayAbort:
            pass
        except (MemoryError, RecursionError):
            if self.fault is None:
                self.fault = {
                    "status": "unavailable",
                    "reason": "resource_limit",
                    "event_index": self.consumed,
                    "phase": self.phase,
                }
        except NameError:
            if self.fault is None:
                self.fault = {
                    "status": "unavailable",
                    "reason": "unknown_program_capability",
                    "event_index": self.consumed,
                    "phase": self.phase,
                }
        except BaseException as error:
            if self.fault is None:
                self.fault = {
                    "status": "diverged",
                    "reason": "program_exception",
                    "event_index": self.consumed,
                    "phase": self.phase,
                    "actual": type(error).__name__,
                }
        return self.result(finished=True)


def _unavailable(reason, **details):
    return {"status": "unavailable", "reason": reason, "finished": True, **details}


def run_request(request, publish):
    environment = ReplayEnvironment(**request["environment"])
    limits = ReplayLimits(**request["limits"])
    resource.setrlimit(resource.RLIMIT_AS, (limits.max_memory_bytes, limits.max_memory_bytes))
    cpu = max(1, math.ceil(limits.timeout_seconds) + 1)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    candidate = ProgramVersion(**request["candidate"])
    if len(candidate.source.encode("utf-8")) > limits.max_source_bytes:
        return _unavailable("source_limit")
    if environment.runtime_sha256 != replay_runtime_fingerprint():
        return _unavailable("worker_runtime_mismatch")
    try:
        tape = RecordedTape.load(
            Path(request["tape_directory"]), expected_sha256=request["tape_sha256"]
        )
    except StateUnavailable:
        return _unavailable("invalid_or_unsupported_tape")
    m = tape.manifest
    if not tape.replay_ready:
        return _unavailable("tape_not_ready")
    for key, expected in (
        ("execution_id", request["witness_execution_id"]),
        ("source_sha256", request["witness_source_sha256"]),
        ("api_sha256", environment.api_sha256),
        ("runtime_sha256", environment.runtime_sha256),
    ):
        if m["identity"][key] != expected:
            return _unavailable(
                "identity_mismatch",
                path=f"identity.{key}",
                expected=expected,
                actual=m["identity"][key],
            )
    if m["boundary"]["profile"] != environment.boundary_profile:
        return _unavailable("boundary_profile_mismatch")
    try:
        code = compile_source(candidate.source, environment.entrypoint)
    except ProfileError as error:
        return _unavailable(
            error.reason, phase="source", path=f"line:{error.line}" if error.line else None
        )
    return _Session(tape, publish).run(code, environment.entrypoint)


def main():
    request_path, result_path = map(Path, sys.argv[1:])

    def publish(result):
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=True, allow_nan=False), encoding="utf-8"
        )
        temporary.replace(result_path)

    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        result = run_request(request, publish)
    except (MemoryError, RecursionError):
        result = _unavailable("resource_limit")
    except Exception:
        result = _unavailable("worker_internal_error")
    publish(result)


if __name__ == "__main__":
    main()
