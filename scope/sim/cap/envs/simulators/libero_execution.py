"""Host execution limits, independent of simulator and scene state."""


class EpisodeStepLimitError(RuntimeError):
    """A host-counted LIBERO episode has no remaining control steps."""

    def __init__(self, max_steps: int, consumed_steps: int):
        super().__init__("simulation_step_limit")
        self.max_steps = max_steps
        self.consumed_steps = consumed_steps


def check_step_budget(consumed_steps: int, max_steps: int) -> None:
    if consumed_steps >= max_steps:
        raise EpisodeStepLimitError(max_steps, consumed_steps)


class MotionConvergenceError(RuntimeError):
    """A bounded joint motion ended before reaching its commanded joints."""

    def __init__(self, joint_error_norm_rad: float, tolerance_rad: float, motion_steps: int):
        super().__init__("joint_motion_not_converged")
        self.joint_error_norm_rad = float(joint_error_norm_rad)
        self.tolerance_rad = float(tolerance_rad)
        self.motion_steps = motion_steps


def check_motion_convergence(joint_error_norm_rad: float, tolerance_rad: float, steps: int) -> None:
    # Match the existing servo's strict error < tolerance stopping condition.
    # NaN must not be interpreted as convergence.
    if not joint_error_norm_rad < tolerance_rad:
        raise MotionConvergenceError(joint_error_norm_rad, tolerance_rad, steps)
