"""Private M8 process entry. Never receives model credentials or endpoint URLs."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from pathlib import Path

from .diagnostic_graph import parse_json
from .libero import api_fingerprint, api_specification, execute_public_program, write_json
from .records import EvidenceRef, ExecutionRecord, ExecutionResult, ProgramVersion
from .replay import ReplayEnvironment, replay, replay_runtime_fingerprint


def main():
    request_path, result_path = map(Path, sys.argv[1:])
    request = parse_json(request_path.read_text())
    if request["mode"] == "describe":
        versions = {}
        for name in (
            "numpy",
            "Pillow",
            "imageio",
            "imageio-ffmpeg",
            "torch",
            "mujoco",
            "libero",
            "robosuite",
            "open3d",
            "scipy",
        ):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        result = {
            "api_sha256": api_fingerprint(),
            "runtime_sha256": replay_runtime_fingerprint(),
            "api_specification": api_specification(),
            "python": platform.python_version(),
            "packages": versions,
        }
    elif request["mode"] == "replay":
        row = request["witness"]
        row["result"]["evidence"] = tuple(EvidenceRef(**r) for r in row["result"]["evidence"])
        row["result"] = ExecutionResult(**row["result"])
        witness = ExecutionRecord(**row)
        result = {
            "report": replay(
                Path(request["tape_directory"]),
                ProgramVersion(**request["program"]),
                witness=witness,
                environment=ReplayEnvironment(**request["environment"]),
            ).to_dict()
        }
    elif request["mode"] in ("probe", "execute"):
        result = execute_public_program(request)
    else:
        raise ValueError("unknown worker mode")
    write_json(result_path, result)


if __name__ == "__main__":
    main()
