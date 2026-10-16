#!/usr/bin/env python3
"""Write LIBERO-Pro's path configuration for a downloaded source tree."""
import argparse
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("external/LIBERO-PRO"))
    parser.add_argument("--config-dir", type=Path, default=Path("external/libero-config"))
    args = parser.parse_args()
    benchmark = args.root.resolve() / "libero/libero"
    if not (benchmark / "benchmark").is_dir():
        parser.error("--root must point to the extracted LIBERO-Pro source")
    config = {
        "benchmark_root": str(benchmark),
        "bddl_files": str(benchmark / "bddl_files"),
        "init_states": str(benchmark / "init_files"),
        "assets": str(benchmark / "assets"),
        "datasets": str(args.root.resolve() / "datasets"),
    }
    args.config_dir.mkdir(parents=True, exist_ok=True)
    target = args.config_dir / "config.yaml"
    target.write_text(yaml.safe_dump(config))
    print(f"LIBERO_CONFIG_PATH={args.config_dir.resolve()}")


if __name__ == "__main__":
    main()
