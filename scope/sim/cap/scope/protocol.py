"""LIBERO-Pro task identity, seed partitions and SCOPE's paper budgets."""

from __future__ import annotations

import re
from dataclasses import dataclass

LIBERO_PRO_SUITES = (
    "libero_object_swap",
    "libero_object_task",
    "libero_goal_swap",
    "libero_goal_task",
    "libero_spatial_swap",
    "libero_spatial_task",
)
DEVELOPMENT_SEEDS = tuple(range(51, 66))
HELD_OUT_SEEDS = tuple(range(1, 51))


def require_name(value: str, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{label} must be a nonempty identifier without path separators")


def _seeds(values: tuple[int, ...], allowed: tuple[int, ...], label: str) -> tuple[int, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError(f"{label} must be a nonempty sequence")
    if any(type(seed) is not int or seed not in allowed for seed in values):
        raise ValueError(f"{label} must be within {allowed[0]}..{allowed[-1]}")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must not contain duplicates")
    # Visit order is part of the protocol: never silently sort it.
    return tuple(values)


@dataclass(frozen=True)
class ScopeProtocol:
    """One task's learning contract; subsets are explicitly reported as such.

    Task names are checked structurally here. Benchmark registry validation belongs
    to the LIBERO adapter so this module never imports simulator dependencies.
    """

    suite: str
    task: str
    development_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    held_out_seeds: tuple[int, ...] = HELD_OUT_SEEDS
    max_main_repairs: int = 3
    max_followup_repairs: int = 1
    replay_sample_size: int = 3
    max_diagnostics_per_update: int = 3

    def __post_init__(self) -> None:
        if self.suite not in LIBERO_PRO_SUITES:
            raise ValueError("suite must be one of the six LIBERO-Pro suites")
        require_name(self.task, "task")
        object.__setattr__(
            self,
            "development_seeds",
            _seeds(
                self.development_seeds,
                DEVELOPMENT_SEEDS,
                "development seeds",
            ),
        )
        object.__setattr__(
            self,
            "held_out_seeds",
            _seeds(
                self.held_out_seeds,
                HELD_OUT_SEEDS,
                "held-out seeds",
            ),
        )
        for name, expected in (
            ("max_main_repairs", 3),
            ("max_followup_repairs", 1),
            ("replay_sample_size", 3),
            ("max_diagnostics_per_update", 3),
        ):
            value = getattr(self, name)
            if type(value) is not int or value != expected:
                raise ValueError(f"paper protocol requires {name}={expected}")

    def require_development_seed(self, seed: int) -> None:
        if type(seed) is not int or seed not in self.development_seeds:
            raise ValueError("execution must use an authorized development seed")

    def configuration_id(self, seed: int) -> str:
        self.require_development_seed(seed)
        return f"{self.suite}/{self.task}/seed_{seed:03d}"

    @property
    def development_scope(self) -> str:
        return "full" if set(self.development_seeds) == set(DEVELOPMENT_SEEDS) else "subset"

    @property
    def held_out_scope(self) -> str:
        return "full" if set(self.held_out_seeds) == set(HELD_OUT_SEEDS) else "subset"
