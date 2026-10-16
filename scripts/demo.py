#!/usr/bin/env python3
"""Run SCOPE on one LIBERO-Pro drawer task."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scope.sim.cap.scope.libero_learning import main

if __name__ == "__main__":
    raise SystemExit(main())
