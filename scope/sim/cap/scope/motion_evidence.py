"""Motion endpoint residuals derived exclusively from public commands and state."""

from scope.sim.cap.utils.ik_validation import pose_errors


def motion_target(name, args, kwargs):
    if name not in ("goto_pose", "move_to_joints"):
        return None
    import numpy as np

    def argument(index, key):
        return args[index] if len(args) > index else kwargs[key]

    if name == "goto_pose":
        return {
            "reference": "requested_final_tcp",
            "position": np.asarray(argument(0, "position"), dtype=float).reshape(3).copy(),
            "quaternion_wxyz": np.asarray(argument(1, "quaternion_wxyz"), dtype=float).reshape(4).copy(),
        }
    if name == "move_to_joints":
        return {
            "reference": "requested_joint_target",
            "joints": np.asarray(argument(0, "joints"), dtype=float).reshape(7).copy(),
        }
    return None


def motion_feedback(target, observation):
    import numpy as np
    from scipy.spatial.transform import Rotation

    if target["reference"] == "requested_joint_target":
        actual = np.asarray(observation["robot_joint_pos"], dtype=float).reshape(8)[:7]
        error = float(np.linalg.norm(actual - target["joints"]))
        if not np.isfinite(error):
            raise ValueError("nonfinite joint residual")
        return {"reference": target["reference"], "joint_error_norm_rad": error}
    hand = np.asarray(observation["robot_cartesian_pos"], dtype=float).reshape(8)
    rotation = Rotation.from_quat(hand[[4, 5, 6, 3]])
    tcp = hand[:3] + rotation.apply([0., 0., 0.1])
    desired = np.concatenate([target["quaternion_wxyz"], target["position"]])
    measured = np.concatenate([hand[3:7], tcp])
    position_error, orientation_error = pose_errors(desired, measured)
    return {
        "reference": target["reference"],
        "requested_tcp_position": target["position"].tolist(),
        "measured_tcp_position": tcp.tolist(),
        "position_error_m": position_error,
        "orientation_error_rad": orientation_error,
        "interpretation": "Distance to requested final TCP, including any clamping/fallback effects; "
                          "on exception, motion may have stopped at an approach waypoint.",
    }
