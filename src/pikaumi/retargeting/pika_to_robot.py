"""Coordinate retargeting from Pika Sense tabletop/Vive TCP trajectories into robot world EE targets."""

from __future__ import annotations

import numpy as np

from pikaumi.calibration.deployment import TableCalibration, load_table_calibration
from pikaumi.dataset.raw import extract_episode_tcp_targets
from pikaumi.robots.kinematics import BimanualIKSolver, posture_seed
from pikaumi.robots.registry import RobotProfile
from pikaumi.robots.utils import mat_to_pose7, pose7_to_mat


def map_table_tcp_to_robot_ee(
    profile: RobotProfile,
    tcp_poses_table_by_side: dict[str, np.ndarray],
    *,
    table_calib: TableCalibration | None = None,
    retarget_mode: str = "absolute-table",
) -> tuple[dict[str, np.ndarray], float | None]:
    """Map Pika Sense TCP trajectories in table frame into target robot `ee_link` poses in robot world frame.

    Modes:
      - `"absolute-table"`: `T_robot_ee(t) = T_robot_table @ T_table_pika(t) @ T_pika_to_ee`.
        Preserves metric tabletop height (`z=0` contact) and bimanual world geometry.
      - `"relative-anchor"`: Anchors `t=0` at the robot's nominal `home_q` EE pose while applying
        table-frame displacements rotated into robot base coordinates (useful for single-arm / uncalibrated preview).

    Returns `(ee_targets_by_side, table_z_in_robot_world)`.
    """
    tcal = table_calib or load_table_calibration(profile.name, require_physical=False)
    T_rt = pose7_to_mat(tcal.T_robot_table_pose7)
    T_pika_ee = pose7_to_mat(profile.pika_to_ee_pose7)
    table_z_robot = float(T_rt[2, 3])

    ee_targets: dict[str, np.ndarray] = {}
    if retarget_mode == "absolute-table":
        for side in profile.active_sides:
            src_side = side if side in tcp_poses_table_by_side else "right"
            src = tcp_poses_table_by_side[src_side]
            out = np.zeros_like(src)
            for t in range(len(src)):
                T_tp = pose7_to_mat(src[t])
                out[t] = mat_to_pose7(T_rt @ T_tp @ T_pika_ee)
            ee_targets[side] = out
        return ee_targets, table_z_robot

    if retarget_mode == "relative-anchor":
        solver = BimanualIKSolver(profile)
        home_fk = solver.fk_poses7(posture_seed(profile))
        R_rt = T_rt[:3, :3]
        for side in profile.active_sides:
            src_side = side if side in tcp_poses_table_by_side else "right"
            src = tcp_poses_table_by_side[src_side]
            T_home = pose7_to_mat(home_fk[side])
            T_src0 = T_rt @ pose7_to_mat(src[0]) @ T_pika_ee
            out = np.zeros_like(src)
            for t in range(len(src)):
                T_cur = T_rt @ pose7_to_mat(src[t]) @ T_pika_ee
                dp_world = R_rt @ (src[t, :3] - src[0, :3])
                R_rel = T_cur[:3, :3] @ T_src0[:3, :3].T
                T_tgt = np.eye(4, dtype=np.float64)
                T_tgt[:3, 3] = T_home[:3, 3] + dp_world
                T_tgt[:3, :3] = R_rel @ T_home[:3, :3]
                out[t] = mat_to_pose7(T_tgt)
            ee_targets[side] = out
        return ee_targets, None

    raise ValueError(f"Unknown retarget_mode={retarget_mode!r}")


def prepare_episode_robot_targets(
    dataset_dir: str,
    episode_idx: int,
    profile: RobotProfile,
    *,
    table_calib: TableCalibration | None = None,
    retarget_mode: str = "absolute-table",
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, float | None]:
    """Load `pika_raw` episode and map TCP poses + widths to `profile`."""
    tcp_table, widths, ts = extract_episode_tcp_targets(dataset_dir, episode_idx)
    ee_targets, table_z = map_table_tcp_to_robot_ee(
        profile,
        tcp_table,
        table_calib=table_calib,
        retarget_mode=retarget_mode,
    )
    active_widths = {
        side: widths[side if side in widths else "right"]
        for side in profile.active_sides
    }
    return ee_targets, active_widths, ts, table_z
