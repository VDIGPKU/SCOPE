"""M4 offline replay launcher and instance-specific certification reports.

Only a successful M1 execution can serve as a witness. Candidate source runs in
a fresh bounded worker with tape-only endpoints, never real API implementations.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

from .protocol import require_name
from .records import ExecutionRecord, ExecutionStatus, ProgramVersion, content_hash, require_hash

REPLAY_PROFILE = "scope-offline-python-v1"
ASSUMPTIONS = (
    "The trusted M1 witness is successful on the recorded execution instance.",
    "The declared public boundary includes all behavior-relevant external inputs and effects.",
    "Program and environment decisions are deterministic for that instance and runtime.",
    "M2/M3 supported state, alias and exception semantics apply; introspection is excluded.",
    "Certification applies to the recorded instance, not a new reset with the same seed.",
)


class ReplayStatus(str, Enum):
    CERTIFIED = "certified"
    DIVERGED = "diverged"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ReplayEnvironment:
    api_sha256: str
    runtime_sha256: str
    boundary_profile: str
    entrypoint: str = "main"

    def __post_init__(self):
        require_hash(self.api_sha256)
        require_hash(self.runtime_sha256)
        require_name(self.boundary_profile, "boundary profile")
        if not self.entrypoint.isidentifier() or self.entrypoint.startswith("_"):
            raise ValueError("entrypoint must be a public Python identifier")


@dataclass(frozen=True)
class ReplayLimits:
    timeout_seconds: float = 10.0
    max_memory_bytes: int = 2 * 1024**3
    max_source_bytes: int = 128 * 1024

    def __post_init__(self):
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout must be positive and finite")
        if any(type(v) is not int or v < 1 for v in (self.max_memory_bytes, self.max_source_bytes)):
            raise ValueError("memory/source limits must be positive integers")


DEFAULT_REPLAY_LIMITS = ReplayLimits()


def replay_runtime_fingerprint() -> str:
    """Shared by recording adapters and workers using this execution profile."""
    import numpy as np

    return content_hash(
        json.dumps(
            {
                "profile": REPLAY_PROFILE,
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "numpy": np.__version__,
                "byteorder": sys.byteorder,
                "machine": platform.machine(),
            },
            sort_keys=True,
        )
    )


@dataclass(frozen=True)
class ReplayReport:
    status: ReplayStatus
    reason: str
    candidate_sha256: str
    witness_execution_id: str
    witness_source_sha256: str
    tape_sha256: str | None
    api_sha256: str
    runtime_sha256: str
    event_index: int | None = None
    phase: str | None = None
    path: str | None = None
    expected: str | None = None
    actual: str | None = None
    consumed_events: int = 0
    total_events: int = 0
    normal_termination: bool = False
    full_match: bool = False
    elapsed_seconds: float = 0.0
    execution_profile: str = REPLAY_PROFILE
    assumptions: tuple[str, ...] = ASSUMPTIONS

    def to_dict(self):
        return {**asdict(self), "status": self.status.value}

    def save(self, path: Path) -> None:
        """Write a new report; never overwrite an existing artifact."""
        with Path(path).open("x", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write("\n")


def replay(
    tape_directory: Path,
    candidate: ProgramVersion,
    *,
    witness: ExecutionRecord,
    environment: ReplayEnvironment,
    limits: ReplayLimits = DEFAULT_REPLAY_LIMITS,
) -> ReplayReport:
    """Compare one candidate to one archived success, with a wall-clock timeout.

    Witness/evidence must come from a trusted M1 archive. No task outcome is
    inferred from divergence or unavailability, and no source is accepted here.
    """
    started = time.monotonic()
    base = {
        "candidate_sha256": candidate.sha256,
        "witness_execution_id": witness.execution_id,
        "witness_source_sha256": witness.source_sha256,
        "tape_sha256": witness.result.tape.sha256
        if witness.result and witness.result.tape
        else None,
        "api_sha256": environment.api_sha256,
        "runtime_sha256": environment.runtime_sha256,
    }

    def report(status, reason, **details):
        return ReplayReport(
            ReplayStatus(status),
            reason,
            **base,
            **details,
            elapsed_seconds=time.monotonic() - started,
        )

    if (
        witness.status is not ExecutionStatus.COMPLETED
        or not witness.result
        or not witness.result.success
        or not witness.role.is_search
    ):
        return report("unavailable", "witness_not_successful_search")
    if witness.result.tape is None:
        return report("unavailable", "missing_witness_tape")
    if len(candidate.source.encode("utf-8")) > limits.max_source_bytes:
        return report("unavailable", "source_limit")
    request = {
        "tape_directory": str(Path(tape_directory).absolute()),
        "candidate": asdict(candidate),
        "witness_execution_id": witness.execution_id,
        "witness_source_sha256": witness.source_sha256,
        "tape_sha256": witness.result.tape.sha256,
        "environment": asdict(environment),
        "limits": asdict(limits),
    }
    # No inherited credentials, service addresses, user Python paths or GPU config.
    worker_env = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[4]),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "LANG": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    with tempfile.TemporaryDirectory(prefix="scope-replay-") as temporary:
        directory = Path(temporary)
        request_path, result_path = directory / "request.json", directory / "result.json"
        request_path.write_text(json.dumps(request, ensure_ascii=True), encoding="utf-8")
        timed_out = False
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "scope.sim.cap.scope._replay_worker",
                    str(request_path),
                    str(result_path),
                ],
                cwd=directory,
                env=worker_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            return report("unavailable", "worker_start_failed")
        try:
            process.wait(timeout=limits.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        try:
            if result_path.stat().st_size > 64 * 1024:
                raise ValueError("oversized worker report")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            finished = result.pop("finished")
            status, reason = result.pop("status"), result.pop("reason")
            if status == "certified" and not (
                finished
                and result.get("normal_termination") is True
                and result.get("full_match") is True
                and type(result.get("consumed_events")) is int
                and result["consumed_events"] == result.get("total_events")
            ):
                raise ValueError("incomplete worker certification claim")
            if timed_out or process.returncode != 0 or not finished:
                # A previously detected mismatch remains the first failure.
                if status == "certified" or reason == "running":
                    status, reason = "unavailable", "timeout" if timed_out else "worker_exit"
                result["normal_termination"] = result["full_match"] = False
            return report(status, reason, **result)
        except (OSError, ValueError, TypeError, KeyError):
            return report("unavailable", "timeout" if timed_out else "worker_failed")
