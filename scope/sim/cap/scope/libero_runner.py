"""M8 fresh-process LIBERO execution and same-interpreter M4 replay."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from .diagnostic_graph import json_text, parse_json, read_public_artifact
from .libero import BOUNDARY_PROFILE, SIM_ROOT, LiberoSettings, receipt_from_files
from .records import Termination
from .replay import ReplayEnvironment, ReplayReport, ReplayStatus


def worker_environment(gpu: str, service_ports: dict[str, int]):
    if not gpu or not gpu.isascii() or not gpu.isdigit():
        raise ValueError("GPU selection must be one explicit numeric CUDA index")
    allowed = (
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "LD_LIBRARY_PATH",
        "LIBERO_CONFIG_PATH",
        "ROBOT_DESCRIPTIONS_CACHE",
        "HF_HUB_OFFLINE",
    )
    environment = {key: os.environ[key] for key in allowed if key in os.environ}
    environment.update(
        PYTHONPATH=str(SIM_ROOT.parent.parent),
        CUDA_VISIBLE_DEVICES=gpu,
        MUJOCO_GL="egl",
        # LIBERO's robosuite EGL enumerates physical devices, independently of
        # CUDA's logical remapping. It also checks membership at import time.
        MUJOCO_EGL_DEVICE_ID=gpu,
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        PYTHONHASHSEED="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    for name, port in service_ports.items():
        if (
            name not in ("SAM3_PORT", "GRASPNET_PORT", "PYROKI_PORT")
            or type(port) is not int
            or not 1 <= port <= 65535
        ):
            raise ValueError("invalid perception service port")
        environment[name] = str(port)
    return environment


def run_worker(python, request, directory, *, gpu, service_ports, timeout):
    directory = Path(directory)
    request_path, result_path = directory / "request.json", directory / "result.json"
    with request_path.open("x", encoding="utf-8") as stream:
        stream.write(json_text(request) + "\n")
    command = [
        str(python),
        "-m",
        "scope.sim.cap.scope._libero_worker",
        str(request_path),
        str(result_path),
    ]
    timed_out = False
    with (directory / "worker.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            cwd=SIM_ROOT,
            env=worker_environment(gpu, service_ports),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    if timed_out:
        return {
            "termination": Termination.TIMEOUT.value,
            "task_completed": None,
            "error_code": "worker_timeout",
        }
    if not result_path.exists():
        raise RuntimeError("worker exited without a durable receipt; reconcile execution")
    result = parse_json(result_path.read_text())
    if process.returncode != 0:
        raise RuntimeError("worker process did not exit cleanly; reconcile execution")
    return result


def worker_facts(python, *, gpu, service_ports):
    with tempfile.TemporaryDirectory(prefix="scope-libero-describe-") as name:
        return run_worker(
            python, {"mode": "describe"}, name, gpu=gpu, service_ports=service_ports, timeout=60
        )


class LiberoExecutor:
    def __init__(
        self,
        *,
        python: Path,
        root: Path,
        suite: str,
        task: str,
        gpu: str,
        settings: LiberoSettings,
        facts: dict,
        service_ports: dict[str, int],
        timeout_seconds: float = 600.0,
        task_language: str | None = None,
    ):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("execution timeout must be positive")
        self.python, self.root = Path(python).absolute(), Path(root).absolute()
        self.suite, self.task, self.gpu = suite, task, gpu
        self.settings, self.facts, self.service_ports = settings, facts, dict(service_ports)
        self.timeout_seconds, self.task_language = timeout_seconds, task_language

    def _request(self, directory, execution_id, seed, program=None):
        return {
            "mode": "execute" if program else "probe",
            "root": str(self.root),
            "directory": str(directory),
            "suite": self.suite,
            "task": self.task,
            "seed": seed,
            "execution_id": execution_id,
            "program": asdict(program) if program else None,
            "settings": asdict(self.settings),
            "expected_language": self.task_language,
            "api_sha256": self.facts["api_sha256"],
            "runtime_sha256": self.facts["runtime_sha256"],
        }

    def probe(self, seed, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        request = self._request(directory, "scope_bootstrap", seed)
        result = run_worker(
            self.python,
            request,
            directory,
            gpu=self.gpu,
            service_ports=self.service_ports,
            timeout=self.timeout_seconds,
        )
        if (
            result.get("mode") != "probe"
            or not result.get("task_language")
            or result.get("termination") is not None
        ):
            hint = (
                "SCOPE requires PyRoKi FK diagnostics; restart the updated PyRoKi service. "
                if result.get("error_code") == "IkPoseUnavailableError" else ""
            )
            raise RuntimeError(
                hint + "initial public observation probe failed "
                f"(termination={result.get('termination')}, "
                f"error={result.get('error_code')}, stage={result.get('error_stage')}); "
                f"inspect {directory / 'result.json'}, "
                f"{directory / 'worker_error.json'} and {directory / 'worker.log'}"
            )
        return result

    def __call__(self, action, record, program):
        if (
            action.payload["execution_id"] != record.execution_id
            or record.source_sha256 != program.sha256
        ):
            raise ValueError("execution action/source mismatch")
        directory = self.root / "executions" / record.execution_id
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "program.py").open("x", encoding="utf-8") as stream:
            stream.write(program.source)
        result = run_worker(
            self.python,
            self._request(directory, record.execution_id, record.seed, program),
            directory,
            gpu=self.gpu,
            service_ports=self.service_ports,
            timeout=self.timeout_seconds,
        )
        return receipt_from_files(directory, self.root, record.execution_id, result)

    def replayer(self, action, witness, program):
        """M4 must use the same Python/NumPy interpreter that recorded the tape."""
        directory = self.root / "replays" / action.action_id
        # Each explicit retry has its own artifacts, never overwrite old evidence.
        directory.mkdir(parents=True, exist_ok=True)
        import uuid

        attempt = directory / uuid.uuid4().hex
        attempt.mkdir()
        ref = witness.result.tape
        try:
            if ref:
                read_public_artifact(self.root, ref)
        except (OSError, ValueError):
            return ReplayReport(
                ReplayStatus.UNAVAILABLE,
                "public_tape_unavailable",
                program.sha256,
                witness.execution_id,
                witness.source_sha256,
                ref.sha256,
                self.facts["api_sha256"],
                self.facts["runtime_sha256"],
            )
        request = {
            "mode": "replay",
            "program": asdict(program),
            "witness": asdict(witness),
            "tape_directory": str(
                (self.root / ref.path).parent if ref else self.root / "missing-tape"
            ),
            "environment": asdict(
                ReplayEnvironment(
                    self.facts["api_sha256"], self.facts["runtime_sha256"], BOUNDARY_PROFILE
                )
            ),
        }
        result = run_worker(
            self.python, request, attempt, gpu=self.gpu, service_ports={}, timeout=30
        )
        if result.get("termination") == "timeout":
            return ReplayReport(
                ReplayStatus.UNAVAILABLE,
                "replay_worker_timeout",
                program.sha256,
                witness.execution_id,
                witness.source_sha256,
                ref.sha256 if ref else None,
                self.facts["api_sha256"],
                self.facts["runtime_sha256"],
            )
        row = result["report"]
        row["status"] = ReplayStatus(row["status"])
        row["assumptions"] = tuple(row["assumptions"])
        return ReplayReport(**row)

    def self_match(self, record, program):
        from .search import SearchAction

        action = SearchAction("self_match_" + record.execution_id, "replay", "ready", "{}")
        return self.replayer(action, record, program)
