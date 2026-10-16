"""IK quality checks using robot-model FK, never simulator scene state."""

class IkPoseError(RuntimeError):
    def __init__(self, details):
        super().__init__("ik_pose_not_converged")
        self.details = details


class IkPoseUnavailableError(RuntimeError):
    """The IK service did not supply usable FK evidence for its solution."""


def pose_errors(target, actual):
    """Return translation meters and shortest rotation radians for WXYZ+XYZ."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    target = np.asarray(target, dtype=float).reshape(7)
    actual = np.asarray(actual, dtype=float).reshape(7)
    if not np.isfinite(target).all() or not np.isfinite(actual).all():
        raise ValueError("nonfinite pose")
    desired = Rotation.from_quat(target[[1, 2, 3, 0]])
    achieved = Rotation.from_quat(actual[[1, 2, 3, 0]])
    return float(np.linalg.norm(target[4:] - actual[4:])), float((desired.inv() * achieved).magnitude())


def solve_ik_checked(ik_solve_fn, quaternion, hand_position, measured_cfg, *, max_iters=5,
                     position_tolerance_m=0.005, orientation_tolerance_rad=0.05):
    """Bounded refinement; a stable joint vector alone does not certify a pose.

    Tolerances apply to the robot-model panda_hand pose, not physical arrival.
    The eighth configuration coordinate is a solver gripper coordinate, not the
    observation's normalized gripper opening. The caller supplies that mapping.
    """
    import numpy as np

    if type(max_iters) is not int or max_iters < 1:
        raise ValueError("invalid IK iteration budget")
    if not (np.isfinite(position_tolerance_m) and position_tolerance_m > 0
            and np.isfinite(orientation_tolerance_rad) and orientation_tolerance_rad > 0):
        raise ValueError("invalid IK tolerance")
    target = np.concatenate([quaternion, hand_position]).astype(float)
    pose_errors(target, target)  # Validate finite pose and nonzero quaternion.
    target[:4] /= np.linalg.norm(target[:4])
    cfg = np.asarray(measured_cfg, dtype=float).reshape(8).copy()
    if not np.isfinite(cfg).all():
        raise ValueError("nonfinite measured joints")
    for iteration in range(1, max_iters + 1):
        report = ik_solve_fn(target_pose_wxyz_xyz=target.copy(), prev_cfg=cfg.copy(), return_pose=True)
        try:
            cfg = np.asarray(report["joint_positions"], dtype=float).reshape(8).copy()
            achieved = np.asarray(report["achieved_pose_wxyz_xyz"], dtype=float).reshape(7)
            if not np.isfinite(cfg).all():
                raise ValueError("nonfinite IK solution")
            position_error, orientation_error = pose_errors(target, achieved)
        except (KeyError, TypeError, ValueError) as error:
            raise IkPoseUnavailableError("IK service requires finite joint and FK pose diagnostics") from error
        details = {
            "reason": "ik_pose_not_converged",
            "target_hand_pose_wxyz_xyz": target.tolist(),
            "achieved_hand_pose_wxyz_xyz": achieved.tolist(),
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "position_tolerance_m": position_tolerance_m,
            "orientation_tolerance_rad": orientation_tolerance_rad,
            "iterations": iteration,
        }
        if position_error <= position_tolerance_m and orientation_error <= orientation_tolerance_rad:
            return cfg
    raise IkPoseError(details)
