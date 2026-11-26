"""Multi-robot kinematic screening & trajectory metric evaluation (`pika dataset screen`)."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from pikaumi.calibration.deployment import load_table_calibration
from pikaumi.config import QUALITY_CONFIG, load_yaml, save_json_atomic, sha256_object
from pikaumi.dataset.raw import load_dataset_episodes, load_dataset_info
from pikaumi.retargeting.cstro import CStroConfig, solve_episode_cstro
from pikaumi.retargeting.pika_to_robot import prepare_episode_robot_targets
from pikaumi.robots.collision import evaluate_trajectory_clearance
from pikaumi.robots.kinematics import (
    BimanualIKSolver,
    solve_episode_dls_myopic,
    solve_episode_handumi_lm,
)
from pikaumi.robots.registry import RobotProfile, load_robot_profile
from pikaumi.robots.utils import pose7_to_mat, pose_errors_pos_deg


def solve_episode_with_method(
    profile: RobotProfile,
    ee_targets_by_side: dict[str, np.ndarray],
    widths_by_side: dict[str, np.ndarray],
    *,
    solver_name: str = "cstro",
    fps: float = 30.0,
    table_z_robot: float | None = None,
    cstro_config: CStroConfig | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Dispatch `dls_myopic`, `handumi_lm`, or `cstro` and measure wall-clock solve time (s)."""
    t0 = time.perf_counter()
    if solver_name == "dls_myopic":
        q_traj, w_out = solve_episode_dls_myopic(profile, ee_targets_by_side, widths_by_side)
    elif solver_name == "handumi_lm":
        q_traj, w_out = solve_episode_handumi_lm(profile, ee_targets_by_side, widths_by_side)
    elif solver_name == "cstro":
        q_traj, w_out = solve_episode_cstro(
            profile,
            ee_targets_by_side,
            widths_by_side,
            fps=fps,
            table_z_robot=table_z_robot,
            config=cstro_config,
        )
    else:
        raise ValueError(f"Unknown solver_name={solver_name!r}")
    elapsed_s = time.perf_counter() - t0
    return q_traj, w_out, elapsed_s


def evaluate_retargeted_trajectory(
    profile: RobotProfile,
    q_traj: np.ndarray,
    ee_targets_by_side: dict[str, np.ndarray],
    widths_by_side: dict[str, np.ndarray],
    *,
    fps: float = 30.0,
    table_z_robot: float | None = None,
    screening_thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive tracking, smoothness, bimanual coupling, limit, and collision metrics."""
    thresh = screening_thresholds or (load_yaml(QUALITY_CONFIG).get("screening") or {})
    max_pos_m = float(thresh.get("max_pos_error_m", 0.030))
    warn_pos_m = float(thresh.get("warn_pos_error_m", 0.015))
    max_rot_deg = float(thresh.get("max_rot_error_deg", 45.0))
    max_swing_deg = float(thresh.get("max_base_swing_deg", 60.0))
    min_self_clear = float(thresh.get("min_self_collision_clearance_m", -0.002))
    min_table_z = float(thresh.get("min_table_z_m", -0.015))

    solver = BimanualIKSolver(profile)
    sides = profile.active_sides
    T = len(q_traj)
    dt = 1.0 / float(fps)

    fk_by_side: dict[str, np.ndarray] = {
        s: np.zeros((T, 7), dtype=np.float64) for s in sides
    }
    for t in range(T):
        fk_t = solver.fk_poses7(q_traj[t])
        for s in sides:
            fk_by_side[s][t] = fk_t[s]

    pos_errs_all: list[np.ndarray] = []
    rot_errs_all: list[np.ndarray] = []
    for s in sides:
        pe, re = pose_errors_pos_deg(fk_by_side[s], ee_targets_by_side[s])
        pos_errs_all.append(pe)
        rot_errs_all.append(re)

    pos_cat = np.concatenate(pos_errs_all)
    rot_cat = np.concatenate(rot_errs_all)
    mean_pos_mm = float(np.mean(pos_cat) * 1000.0)
    max_pos_mm = float(np.max(pos_cat) * 1000.0)
    mean_rot_deg = float(np.mean(rot_cat))
    max_rot_deg_obs = float(np.max(rot_cat))

    # Bimanual relative SE(3) error (overall & during co-grasp w < 0.04m)
    rel_pos_mm = 0.0
    rel_rot_deg = 0.0
    cograsp_rel_pos_mm = 0.0
    if "left" in sides and "right" in sides:
        rel_pe = np.zeros(T, dtype=np.float64)
        rel_re = np.zeros(T, dtype=np.float64)
        for t in range(T):
            TL_c = pose7_to_mat(fk_by_side["left"][t])
            TR_c = pose7_to_mat(fk_by_side["right"][t])
            TL_t = pose7_to_mat(ee_targets_by_side["left"][t])
            TR_t = pose7_to_mat(ee_targets_by_side["right"][t])
            T_rel_c = np.linalg.inv(TL_c) @ TR_c
            T_rel_t = np.linalg.inv(TL_t) @ TR_t
            rel_pe[t] = float(np.linalg.norm(T_rel_c[:3, 3] - T_rel_t[:3, 3]))
            R_diff = T_rel_c[:3, :3] @ T_rel_t[:3, :3].T
            rel_re[t] = float(np.degrees(np.linalg.norm(Rotation.from_matrix(R_diff).as_rotvec())))
        rel_pos_mm = float(np.mean(rel_pe) * 1000.0)
        rel_rot_deg = float(np.mean(rel_re))
        cg_mask = (widths_by_side["left"] < 0.040) & (widths_by_side["right"] < 0.040)
        cograsp_rel_pos_mm = float(np.mean(rel_pe[cg_mask]) * 1000.0) if np.any(cg_mask) else rel_pos_mm

    # Active joint velocity compliance, jerk, and base-yaw swing
    X_act = q_traj[:, list(solver.active_indices)]
    if T >= 2:
        vel = np.abs(np.diff(X_act, axis=0)) / dt
        vel_viol_frac = float(np.mean(vel > (solver.vel_limits[None, :] * 1.001)))
        max_vel_ratio = float(np.max(vel / np.maximum(solver.vel_limits[None, :], 1e-6)))
    else:
        vel_viol_frac = 0.0
        max_vel_ratio = 0.0

    if T >= 4:
        jerk = np.diff(X_act, n=3, axis=0) / (dt**3)
        rms_jerk = float(np.sqrt(np.mean(jerk**2)))
    else:
        rms_jerk = 0.0

    # Base-yaw swing across episode (peak-to-peak excursion on first arm joint of each side)
    base_swings_deg: list[float] = []
    for s in sides:
        j0_idx = profile.arms[s].arm_actuated_indices[0]
        swing_rad = float(np.max(q_traj[:, j0_idx]) - np.min(q_traj[:, j0_idx]))
        base_swings_deg.append(float(np.degrees(swing_rad)))
    max_base_swing_obs_deg = max(base_swings_deg) if base_swings_deg else 0.0

    # Joint position limit check
    lim_viol = int(
        np.sum((X_act < solver.lower[None, :] - 1e-4) | (X_act > solver.upper[None, :] + 1e-4))
    )

    ee_links = tuple(profile.arms[s].ee_link for s in sides)
    col_summary = evaluate_trajectory_clearance(
        profile.urdf_model,
        q_traj,
        table_z_in_robot_world=table_z_robot,
        ee_links=ee_links,
        table_tolerance_m=min_table_z,
    )

    reasons: list[str] = []
    warnings: list[str] = []
    if max_pos_mm / 1000.0 > max_pos_m:
        reasons.append(f"max_pos_err_mm:{max_pos_mm:.1f}>{max_pos_m*1000:.1f}")
    elif max_pos_mm / 1000.0 > warn_pos_m:
        warnings.append(f"warn_pos_err_mm:{max_pos_mm:.1f}")

    if max_rot_deg_obs > max_rot_deg:
        reasons.append(f"max_rot_err_deg:{max_rot_deg_obs:.1f}>{max_rot_deg:.1f}")
    if max_base_swing_obs_deg > max_swing_deg:
        reasons.append(f"base_swing_deg:{max_base_swing_obs_deg:.1f}>{max_swing_deg:.1f}")
    if col_summary.min_inter_arm_clearance_m < min_self_clear:
        reasons.append(f"inter_arm_collision_m:{col_summary.min_inter_arm_clearance_m:.4f}")
    if table_z_robot is not None and col_summary.table_penetration_frames > 0:
        reasons.append(f"table_penetration_frames:{col_summary.table_penetration_frames}")
    if lim_viol > 0:
        reasons.append(f"joint_limit_violations:{lim_viol}")
    if vel_viol_frac > 0.05:
        warnings.append(f"vel_limit_violation_frac:{vel_viol_frac:.3f}")

    verdict = "REJECT" if reasons else ("WARN" if warnings else "OK")
    return {
        "verdict": verdict,
        "mean_pos_err_mm": round(mean_pos_mm, 4),
        "max_pos_err_mm": round(max_pos_mm, 4),
        "mean_rot_err_deg": round(mean_rot_deg, 4),
        "max_rot_err_deg": round(max_rot_deg_obs, 4),
        "bimanual_rel_pos_err_mm": round(rel_pos_mm, 4),
        "cograsp_rel_pos_err_mm": round(cograsp_rel_pos_mm, 4),
        "bimanual_rel_rot_err_deg": round(rel_rot_deg, 4),
        "urdf_vel_violation_fraction": round(vel_viol_frac, 5),
        "max_vel_limit_ratio": round(max_vel_ratio, 4),
        "rms_jerk_rad_s3": round(rms_jerk, 3),
        "max_base_swing_deg": round(max_base_swing_obs_deg, 3),
        "min_inter_arm_clearance_m": round(col_summary.min_inter_arm_clearance_m, 4),
        "min_table_clearance_m": round(col_summary.min_table_clearance_m, 4),
        "reasons": reasons,
        "warnings": warnings,
    }


def compute_solve_cache_signature(
    dataset_info: dict[str, Any],
    profile: RobotProfile,
    table_sha256: str,
    solver: str,
    retarget_mode: str,
    cstro_config: CStroConfig | None = None,
    episodes: list[dict[str, Any]] | None = None,
) -> str:
    from dataclasses import asdict
    from pikaumi.config import sha256_file

    cal = dataset_info.get("calibration") or {}
    per_ep_cals = (
        [
            {
                "ep": int(e.get("episode_index", idx)),
                "cal": e.get("calibration"),
            }
            for idx, e in enumerate(episodes)
            if "calibration" in e
        ]
        if episodes
        else None
    )
    return sha256_object(
        {
            "robot": profile.name,
            "robot_sha256": profile.sha256,
            "urdf_sha256": sha256_file(profile.urdf_path),
            "table_sha256": table_sha256,
            "pika_tcp_sha256": cal.get("pika_tcp_sha256", "none"),
            "spatial_sha256": cal.get("spatial_sha256", "none"),
            "T_table_vive": cal.get("T_table_vive"),
            "T_tracker_tcp": cal.get("T_tracker_tcp"),
            "per_episode_calibrations": per_ep_cals,
            "solver": solver,
            "retarget_mode": retarget_mode,
            "cstro_config": asdict(cstro_config or CStroConfig()) if solver == "cstro" else None,
        }
    )


def screen_dataset_for_robots(
    dataset_dir: str | Path,
    *,
    robots: list[str],
    solver: str = "cstro",
    retarget_mode: str = "absolute-table",
) -> dict[str, dict[str, Any]]:
    """Screen all episodes in `dataset_dir` across `robots` and cache `_solves_{robot}_{solver}.npz`."""
    root = Path(dataset_dir)
    info = load_dataset_info(root)
    episodes = load_dataset_episodes(root)
    fps = float(info.get("fps", 30.0))
    results: dict[str, dict[str, Any]] = {}

    for rname in robots:
        prof = load_robot_profile(rname)
        tcal = load_table_calibration(rname, require_physical=False)
        mode = "relative-anchor" if len(prof.active_sides) == 1 and retarget_mode == "absolute-table" else retarget_mode
        sig = compute_solve_cache_signature(info, prof, tcal.sha256, solver, mode, episodes=episodes)

        counts = {"OK": 0, "WARN": 0, "REJECT": 0}
        ep_reports: list[dict[str, Any]] = []
        cache_arrays: dict[str, Any] = {"signature": np.array(sig)}

        for ep_meta in episodes:
            ep_idx = int(ep_meta["episode_index"])
            ee_tgts, widths, _, table_z = prepare_episode_robot_targets(
                str(root), ep_idx, prof, table_calib=tcal, retarget_mode=mode
            )
            q_traj, w_robot, solve_s = solve_episode_with_method(
                prof, ee_tgts, widths, solver_name=solver, fps=fps, table_z_robot=table_z
            )
            metrics = evaluate_retargeted_trajectory(
                prof, q_traj, ee_tgts, widths, fps=fps, table_z_robot=table_z
            )
            metrics["episode_index"] = ep_idx
            metrics["solve_time_s"] = round(solve_s, 4)
            counts[metrics["verdict"]] += 1
            ep_reports.append(metrics)

            cache_arrays[f"ep_{ep_idx:06d}_q"] = q_traj.astype(np.float32)
            for s, w_arr in w_robot.items():
                cache_arrays[f"ep_{ep_idx:06d}_w_{s}"] = w_arr.astype(np.float32)

        cache_path = root / "meta" / f"_solves_{rname}_{solver}.npz"
        np.savez_compressed(cache_path, **cache_arrays)

        rep = {
            "dataset_dir": str(root),
            "robot": rname,
            "solver": solver,
            "retarget_mode": mode,
            "cache_signature": sig,
            "cache_file": str(cache_path.name),
            "counts": counts,
            "episodes": ep_reports,
        }
        save_json_atomic(root / "meta" / f"screen_{rname}.json", rep)
        results[rname] = rep

    return results
