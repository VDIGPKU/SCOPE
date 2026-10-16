# SPDX-FileCopyrightText: Copyright (c) 2026 Max Fu
# SPDX-License-Identifier: MIT
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import pathlib
import time
from typing import Any

import numpy as np
import open3d as o3d
import viser.transforms as vtf
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as SciRotation

from scope.sim.cap.utils.ik_validation import (
    IkPoseError,
    IkPoseUnavailableError,
    solve_ik_checked,
)

logger = logging.getLogger(__name__)

from scope.sim.cap.envs.base import (
    BaseEnv,
)
from scope.sim.cap.integrations.base_api import ApiBase
from scope.sim.cap.integrations.franka.common import (
    apply_tcp_offset,
    close_gripper as _close_gripper,
    extract_arm_joints,
    get_oriented_bounding_box_from_3d_points as _get_obb,
    open_gripper as _open_gripper,
    solve_ik_with_convergence,
)
from scope.sim.cap.integrations.vision.graspnet import init_contact_graspnet, init_contact_graspnet_point_clouds
from scope.sim.cap.integrations.vision.sam3 import init_sam3, init_sam3_point_prompt
from scope.sim.cap.integrations.motion.pyroki import init_pyroki

from scope.sim.cap.utils.camera_utils import obs_get_rgb
from scope.sim.cap.utils.depth_utils import (
    deproject_pixel_to_camera,
    depth_color_to_pointcloud,
    depth_to_pointcloud,
    depth_to_rgb,
)



from sklearn.cluster import DBSCAN

# ------------------------------- Control API ------------------------------
class FrankaLiberoApiReduced(ApiBase):
    """Robot control helpers for Franka.
    """
    _TCP_OFFSET = np.array([0.0, 0.0, -0.1], dtype=np.float64)

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        # Initialize perception models for non-privileged API
        # if self.use_sam3:
        self.sam3_seg_fn = init_sam3()
        self.sam3_point_prompt_fn = init_sam3_point_prompt()

        self.grasp_net_plan_fn = init_contact_graspnet()
        self.grasp_net_plan_point_clouds_fn = init_contact_graspnet_point_clouds()
        self.ik_solve_fn = init_pyroki()
        self.camera_name = "agentview"
        self.wrist_camera_name = "robot0_eye_in_hand"
        self.cfg = None

        # JIT warmup: trigger JAX compilation with a dummy IK call
        try:
            self.ik_solve_fn(
                target_pose_wxyz_xyz=np.array([1, 0, 0, 0, 0.3, 0, 0.5]),
                prev_cfg=None,
            )
        except Exception:
            pass


    def functions(self) -> dict[str, Any]:
        fns = {}
        fns["get_observation"] = self.get_observation
        fns["segment_sam3_text_prompt"] = self.segment_sam3_text_prompt
        fns["segment_sam3_point_prompt"] = self.segment_sam3_point_prompt
        fns["plan_grasp"] = self.plan_grasp
        fns["plan_grasp_from_point_clouds"] = self.plan_grasp_from_point_clouds
        fns["get_oriented_bounding_box_from_3d_points"] = (
            self.get_oriented_bounding_box_from_3d_points
        )
        fns["solve_ik"] = self.solve_ik
        fns["move_to_joints"] = self.move_to_joints
        fns["open_gripper"] = self.open_gripper
        fns["close_gripper"] = self.close_gripper
        fns["goto_pose"] = self.goto_pose

        fns["goto_home_joint_position"] = self.goto_home_joint_position
        fns["subsample_point_cloud"] = self.subsample_point_cloud
        fns["filter_noise"] = self.filter_noise

        return fns


    def get_observation(self) -> dict[str, Any]:
        """Get the observation of the environment.
        Returns:
            observation:
                A dictionary containing the observation of the environment.
                The dictionary contains the following keys:
                - ["agentview"]["images"]["rgb"]: Current color camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["agentview"]["images"]["depth"]: Current depth camera image as a numpy array of shape (H, W), dtype float32.
                - ["agentview"]["intrinsics"]: Camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["agentview"]["pose_mat"]: Camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot0_eye_in_hand"]["images"]["rgb"]: Current wrist camera image as a numpy array of shape (H, W, 3), dtype uint8.
                - ["robot0_eye_in_hand"]["images"]["depth"]: Current wrist camera depth image as a numpy array of shape (H, W), dtype float32.
                - ["robot0_eye_in_hand"]["intrinsics"]: Wrist camera intrinsic matrix as a numpy array of shape (3, 3), dtype float64.
                - ["robot0_eye_in_hand"]["pose_mat"]: Wrist camera extrinsic matrix as a numpy array of shape (4, 4), dtype float64.
                - ["robot_cartesian_pos"]: Current end-effector (panda_hand) pose in the robot/world frame as a numpy array of shape (8,), dtype float64. The first 3 elements are the robot's end-effector XYZ, the next 4 elements are the quaternion wxyz, and the last element is the gripper position normalized, 0 (closed) to 1 (open).
                - ["robot_joint_pos"]: Current arm joint positions followed by normalized gripper position as a numpy array of shape (8,), dtype float64. The first 7 elements are arm joint angles; the last is gripper position, 0 (closed) to 1 (open).

        Motion frame convention:
            robot_cartesian_pos reports the panda_hand origin, not the TCP.
            solve_ik and goto_pose accept TCP positions. For the reported hand
            pose (p_hand, R), p_tcp = p_hand + R @ [0, 0, 0.1] meters.
            Compare p_tcp with a commanded TCP target when checking arrival.
        """
        obs = self._env.get_observation()
        obs[self.camera_name]["images"]["depth"] = obs[self.camera_name]["images"]["depth"].squeeze(-1)
        obs[self.wrist_camera_name]["images"]["depth"] = obs[self.wrist_camera_name]["images"]["depth"].squeeze(-1)
        return obs


    def segment_sam3_point_prompt(
        self,
        rgb: np.ndarray,
        point_coords: tuple[float, float],
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image, optionally conditioned on an image coordinate point prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            point_coords:
                (x, y) pixel coordinates of the point prompt.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool,
                              where True means the pixel belongs to the instance.
                  - "score": float confidence score.

        Example:
            >>> rgb = obs["agentview"]["images"]["rgb"]
            >>> masks = segment_sam3_point_prompt(rgb, (100, 100))
        """
        return self.sam3_point_prompt_fn(Image.fromarray(rgb), point_coords)

    def segment_sam3_text_prompt(
        self,
        rgb: np.ndarray,
        text_prompt: str,
    ) -> list[dict[str, Any]]:
        """Run SAM3 segmentation on an RGB image conditioned on a text prompt.

        Args:
            rgb:
                RGB image array of shape (H, W, 3), dtype uint8.
            text_prompt:
                Text prompt for SAM3 segmentation.

        Returns:
            masks:
                A list of dictionaries. Each dict may contain:

                  - "mask":  np.ndarray of shape (H, W), dtype bool,
                              where True means the pixel belongs to the instance.
                  - "box": list [x1, y1, x2, y2] in pixel coordinates.
                  - "score": float confidence score.
        Note:
            Returns an empty list if no results are found.

        Example:
            >>> rgb = obs["agentview"]["images"]["rgb"]
            >>> masks = segment_sam3(rgb, text_prompt="red mug")
        """
        results = self.sam3_seg_fn(rgb, text_prompt=text_prompt)
        if len(results) == 0:
            print(f"[segment_sam3_text_prompt] SAM3 returned no results for prompt: '{text_prompt}'")
            return []
        return results

    # --------------------------------------------------------------------- #
    # Molmo point prompt
    # --------------------------------------------------------------------- #

    def get_oriented_bounding_box_from_3d_points(self, points: np.ndarray) -> dict[str, Any]:
        """Get the oriented bounding box from 3D points.

        Args:
            points: np.ndarray: The 3D points to get the oriented bounding box from.
                Shape: (N, 3), dtype float64.

        Returns:
            dict[str, Any]: The oriented bounding box. The dictionary contains the following keys:
                - "center": np.ndarray: The center of the oriented bounding box in point cloud frame.
                - "extent": np.ndarray: The extent of the oriented bounding box.
                - "R": np.ndarray: The rotation matrix of the oriented bounding box in point cloud frame.

        Example:
            >>> points = np.random.randn((100, 3))
            >>> obb = get_oriented_bounding_box_from_3d_points(points)
        """
        return _get_obb(points)

    # --------------------------------------------------------------------- #
    # Grasp planner (Contact-GraspNet)
    # --------------------------------------------------------------------- #
    def plan_grasp(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        segmentation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plan grasp candidates using Contact-GraspNet for a single instance.

        This is a thin wrapper around the Contact-GraspNet planner. It does not
        apply any camera/world transforms or TCP offsets: the caller is
        responsible for transforming the resulting grasp poses into the desired
        frame and applying TCP offsets if necessary.

        Args:
            depth:
                Depth image in meters.
                Shape: (H, W) or (H, W, 1), dtype float32/float64.
            intrinsics:
                Camera intrinsic matrix.
                Shape: (3, 3), dtype float64.
            segmentation:
                Instance segmentation map where each integer > 0 corresponds to a
                unique object instance ID.
                Shape: (H, W) or (H, W, 1), dtype int32/int64.

        Returns:
            grasp_poses:
                np.ndarray of shape (K, 4, 4), dtype float64.
                Homogeneous transforms for each candidate grasp IN THE CAMERA FRAME.
            grasp_scores:
                np.ndarray of shape (K,), dtype float64.
                Confidence score for each candidate grasp.

        Example:
            >>> cam = obs["agentview"]
            >>> rgb = cam["images"]["rgb"]
            >>> depth = cam["images"]["depth"][:, :, 0]
            >>> sam3_results = sam3_seg_fn(rgb, text_prompt="red mug")
            >>> best = max(sam3_results, key=lambda d: d["score"])
            >>> mask = best["mask"]
            >>> K = cam["intrinsics"]
            >>> grasp_sample_tf, grasp_scores = plan_grasp(
            ...     depth=depth,
            ...     intrinsics=K,
            ...     segmentation=mask,
            ... )
            >>> best_idx = grasp_scores.argmax()
            >>> best_T = grasp_poses[best_idx]  # (4, 4)
            >>> camera_extrinsics = cam["pose_mat"]
            >>> grasp_sample_world_frame = camera_extrinsics @ best_T
        """
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[:, :, 0]

        grasp_sample, grasp_scores, _ = self.grasp_net_plan_fn(
            depth,
            intrinsics,
            segmentation,
            1,
        )

        assert len(grasp_sample) > 0, "No grasp candidates found"

        grasp_sample_tf = (
            vtf.SE3.from_matrix(grasp_sample) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))
        ).as_matrix()
        return grasp_sample_tf, grasp_scores

    def plan_grasp_from_point_clouds(
        self,
        pc_full: np.ndarray,
        pc_segment: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Plan grasp candidates using Contact-GraspNet given a full point cloud and a segmented point cloud of the object wanting to be grasped. These point clouds can be composed from multiple viewpoints.

        Args:
            pc_full:
                Point cloud of the full environment, including the object to be grasped.
                Shape: (N, 3), dtype float32.
            pc_segment:
                Point cloud of the segmented object to be grasped.
                Shape: (N, 3), dtype float32.
        Returns:
            grasp_sample_tf: (4, 4) homogeneous transform for the grasp pose in THE POINT CLOUD FRAME.
            grasp_scores: (N,) array of grasp scores.
        """
        grasp_sample, grasp_scores, _ = self.grasp_net_plan_point_clouds_fn(pc_full, pc_segment, segmap_id=1)
        
        assert len(grasp_sample) > 0, "No grasp candidates found"

        grasp_sample_tf = (
            vtf.SE3.from_matrix(grasp_sample) @ vtf.SE3.from_translation(np.array([0, 0, 0.12]))
        ).as_matrix()
        return grasp_sample_tf, grasp_scores

    # --------------------------------------------------------------------- #
    # IK / motion primitives
    # --------------------------------------------------------------------- #
    def solve_ik(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
    ) -> np.ndarray:
        """Solve inverse kinematics for a TCP target using the panda_hand link.

        The TCP is 0.1 m along the hand's local +Z axis. This method converts
        the TCP target to a hand-origin target internally:
        p_hand = p_tcp + R @ [0, 0, -0.1]. Do not apply this offset again.
        get_observation reports the hand origin; its observed TCP position is
        p_hand + R @ [0, 0, 0.1]. Quaternions are WXYZ.

        Positions are clamped to world XYZ bounds [-0.1, -0.5, 0.005] through
        [0.75, 0.5, 0.9] meters. If the requested orientation cannot be solved,
        top-down, 45-degree tilt and side-approach orientations are tried.
        Returned joints can therefore realize a different orientation; inspect
        the observed robot pose before assuming the requested pose was reached.

        In SCOPE, each solve uses measured arm joints as its initial IK reference.
        Robot-model FK must meet 0.005 m hand-position and 0.05 rad orientation
        tolerances within five solves per orientation. This checks the IK model,
        not physical arrival or task success. Other callers retain legacy IK.

        Args:
            position:
                Target TCP position in world frame, not the hand origin.
                Shape: (3,), dtype float64.
            quaternion_wxyz:
                Target orientation as a unit quaternion in world frame.
                Shape: (4,), [w, x, y, z], dtype float64.

        Returns:
            joints:
                np.ndarray of shape (7,), dtype float64.
                Joint angles for the 7 DoF Franka arm.

        Example:
            >>> target_pos = np.array([0.5, 0.0, 0.3])
            >>> target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity, wxyz
            >>> joints = solve_ik(target_pos, target_quat)
            >>> move_to_joints(joints)
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        if getattr(self, "validate_ik_pose", False) and not np.isfinite(pos).all():
            raise ValueError("nonfinite TCP target")
        # Clamp to reachable workspace to avoid IK timeouts
        pos = np.clip(pos, [-0.1, -0.5, 0.005], [0.75, 0.5, 0.9])
        quat_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)
        offset_pos = apply_tcp_offset(pos, quat_wxyz, self._TCP_OFFSET)

        checked = getattr(self, "validate_ik_pose", False)
        previous = self.cfg
        if checked:
            # Public measured arm joints, not the last unexecuted solver result.
            measured = np.asarray(self._env.get_observation()["robot_joint_pos"], dtype=float).reshape(8)
            if not np.isfinite(measured).all():
                raise ValueError("nonfinite measured joints")
            # panda_hand FK is independent of the finger coordinate. A fixed
            # zero is valid; normalized gripper opening is not a joint in meters.
            previous = np.concatenate([measured[:7], [0.0]])
        last_pose_error = None
        # Try requested orientation first, then fallbacks
        orientations = [
            ("requested", quat_wxyz),
            ("top-down", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)),
            ("45-tilt", np.array([0.707, 0.707, 0.0, 0.0], dtype=np.float64)),
            ("side-approach", np.array([0.707, 0.0, 0.707, 0.0], dtype=np.float64)),
        ]

        for label, quat in orientations:
            try:
                off_pos = apply_tcp_offset(pos, quat, self._TCP_OFFSET) if label != "requested" else offset_pos
                solver = solve_ik_checked if checked else solve_ik_with_convergence
                candidate = solver(self.ik_solve_fn, quat, off_pos, previous)
                joints = extract_arm_joints(candidate)
                self.cfg = candidate
                if label != "requested":
                    logger.info("IK solved with fallback orientation: %s", label)
                return joints
            except IkPoseUnavailableError:
                # Older services need an update, not another orientation retry.
                raise
            except IkPoseError as error:
                last_pose_error = error
                continue
            except Exception:
                logger.warning("IK failed with orientation '%s', trying next fallback", label)
                continue

        if last_pose_error is not None:
            raise last_pose_error
        raise RuntimeError(
            f"IK failed for position {pos} with all orientation fallbacks"
        )

    # Single arm control APIs

    def move_to_joints(self, joints: np.ndarray) -> None:
        """Move the robot to a given joint configuration in a blocking manner.
        Interpolation is slightly rudimentary so it is recommended to keep pre-grasp cartesian offsets smaller e.g. 0.075m.

        Args:
            joints:
                Target joint angles for the 7-DoF Franka arm.
                Shape: (7,), dtype float64.

        Returns:
            None

        Example:
            >>> joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.8]) # this is an example home joint configuration for the Franka arm
            >>> move_to_joints(joints)
        """
        joints = np.asarray(joints, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)

        # self._env.move_to_joints_non_blocking(joints)

    def open_gripper(self) -> None:
        """Open gripper fully.

        Args:
            None
        """
        _open_gripper(self._env, steps=30)

    def close_gripper(self) -> None:
        """Close gripper fully.

        Args:
            None
        """
        _close_gripper(self._env, steps=30)

    def goto_pose(
        self,
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        z_approach: float = 0.0,
    ) -> None:
        """Go to pose using Inverse Kinematics with optional approach motion.

        Combines solve_ik + move_to_joints into a single call. If z_approach > 0,
        first moves to position + [0, 0, z_approach] in WORLD coordinates,
        then moves to the final position. This is a vertical approach, even
        when the gripper is tilted. For approach along another direction,
        explicitly construct TCP waypoints and call goto_pose on each.

        position is a TCP target. solve_ik applies the 0.1 m hand/TCP offset
        internally; do not compensate it again. Observation reports the hand
        origin, so compare its derived TCP position with the target. Workspace
        clamping and orientation fallbacks are inherited from solve_ik.

        Args:
            position: (3,) TCP XYZ target in meters (world frame).
            quaternion_wxyz: (4,) WXYZ unit quaternion (world frame).
            z_approach: Positive world Z offset for approach (meters). Default 0.0.

        Example:
            >>> goto_pose(np.array([0.5, 0.0, 0.3]), np.array([0.0, 1.0, 0.0, 0.0]), z_approach=0.075)
        """
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        quat = np.asarray(quaternion_wxyz, dtype=np.float64).reshape(4)

        if z_approach > 0.0:
            approach_pos = pos.copy()
            approach_pos[2] += z_approach
            joints = self.solve_ik(approach_pos, quat)
            self.move_to_joints(joints)

        joints = self.solve_ik(pos, quat)
        self.move_to_joints(joints)
    
    def goto_home_joint_position(self) -> None:
        """Return the arm to its reset joint configuration with high manipulability"""
        home = getattr(self._env, "home_joint_position", None)
        if home is None:
            raise RuntimeError("Home joint position is unavailable in the current environment.")
        joints = np.asarray(home, dtype=np.float64).reshape(7)
        self._env.move_to_joints_blocking(joints)
    
    def subsample_point_cloud(self, pc: np.ndarray, max_points: int = 10000) -> np.ndarray:
        """Randomly subsample a point cloud to a maximum number of points.
        
        Args:
            pc: (N, 3) array of points.
            max_points: The maximum number of points to subsample to. Default is 10000.
        Returns:
            subsampled_pc: (M, 3) array of subsampled points where M <= max_points.
        """
        if len(pc) > max_points:
            return pc[np.random.choice(len(pc), max_points, replace=False)]
        return pc
    
    def filter_noise(self, points: np.ndarray, colors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Filter noise from the point cloud.
        Args:
            points: (N, 3) array of points.
            colors: (N, 3) array of colors. Optional, default is None.
        Returns:
            (M, 3) array of points.
            (M, 3) array of colors. Optional, default is None.
        """
        eps = 0.005  # Maximum distance between samples to be neighbors
        min_samples = 10
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)
        filtered_pointcloud = points[labels != -1]
        if colors is not None:
            filtered_colors = colors[labels != -1]
        else:
            filtered_colors = None
        return filtered_pointcloud, filtered_colors

    





    

