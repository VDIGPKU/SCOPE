#!/usr/bin/env python3
"""Run SAM 3, Contact-GraspNet and PyRoKi until interrupted."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--graspnet-gpu", default=None)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()
    environment = os.environ.copy()
    if args.env_file.is_file():
        environment.update({k: v for k, v in dotenv_values(args.env_file).items() if v})
    environment["PYTHONPATH"] = str(ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
    names = ("sam3", "contact_graspnet", "pyroki")
    ports = [environment.get(key) or default for key, default in
             (("SAM3_PORT", "8114"), ("GRASPNET_PORT", "8115"), ("PYROKI_PORT", "8116"))]
    args.log_dir.mkdir(parents=True, exist_ok=True)
    children, logs = [], []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        for name, port in zip(names, ports):
            command = [sys.executable, "-u", "-m", f"scope.sim.cap.serving.launch_{name}_server",
                       "--host", "127.0.0.1", "--port", str(port)]
            env = environment.copy()
            env["CUDA_VISIBLE_DEVICES"] = (args.graspnet_gpu or args.gpu) if name == "contact_graspnet" else args.gpu
            if name == "sam3" and env.get("SAM3_CHECKPOINT_PATH"):
                command += ["--checkpoint-path", env["SAM3_CHECKPOINT_PATH"]]
            if name == "pyroki":
                env["JAX_PLATFORMS"] = "cpu"
                command += ["--robot", "panda_description", "--target-link", "panda_hand"]
            log = (args.log_dir / f"{name}.log").open("a")
            logs.append(log)
            children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=log))
            print(f"{name}: port {port}, log {args.log_dir / (name + '.log')}", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(1)
        failed = [(name, child.returncode) for name, child in zip(names, children) if child.poll() is not None]
        raise SystemExit(f"Service exited: {failed}; see {args.log_dir}")
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
