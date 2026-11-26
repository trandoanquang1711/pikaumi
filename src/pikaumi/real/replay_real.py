"""Physical trajectory preflight & guarded `piper_sdk` SocketCAN replay (`pika replay-real`)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from pikaumi.calibration.deployment import load_table_calibration
from pikaumi.config import save_json_atomic
from pikaumi.dataset.raw import load_dataset_info
from pikaumi.dataset.screening import solve_episode_with_method
from pikaumi.real.base import (
    ArmStateMode,
    GuardedRobotController,
    MockBimanualController,
    RealRobotBackend,
    require_live_authorization,
)
from pikaumi.real.piper_driver import PiperCanBackend, load_piper_can_settings
from pikaumi.real.streamer import next_periodic_deadline
from pikaumi.retargeting.pika_to_robot import prepare_episode_robot_targets
from pikaumi.robots.collision import evaluate_config_clearance
from pikaumi.robots.kinematics import BimanualIKSolver
from pikaumi.robots.registry import RobotProfile, load_robot_profile
from pikaumi.robots.utils import pose7_to_mat


def _resolve_real_motion_limits(profile: RobotProfile) -> tuple[float, float]:
    """Return `(max_vel_deg_s, max_acc_deg_s2)` supporting both flat and nested `real.control` schemas."""
    rc = profile.real_config or {}
    ctrl = rc.get("control") if isinstance(rc.get("control"), dict) else {}

    vel_candidates: list[float] = []
    if "max_joint_speed_deg_s" in rc:
        vel_candidates.append(float(rc["max_joint_speed_deg_s"]))
    if "max_joint_speed_rad_s" in rc:
        vel_candidates.append(math.degrees(float(rc["max_joint_speed_rad_s"])))
    if "max_joint_speed_deg_s" in ctrl:
        vel_candidates.append(float(ctrl["max_joint_speed_deg_s"]))
    if "max_joint_speed_rad_s" in ctrl:
        vel_candidates.append(math.degrees(float(ctrl["max_joint_speed_rad_s"])))
    max_vel_deg_s = min(vel_candidates) if vel_candidates else 180.0

    acc_candidates: list[float] = []
    if "max_joint_acceleration_deg_s2" in rc:
        acc_candidates.append(float(rc["max_joint_acceleration_deg_s2"]))
    if "max_joint_acceleration_rad_s2" in rc:
        acc_candidates.append(math.degrees(float(rc["max_joint_acceleration_rad_s2"])))
    if "max_joint_acceleration_deg_s2" in ctrl:
        acc_candidates.append(float(ctrl["max_joint_acceleration_deg_s2"]))
    if "max_joint_acceleration_rad_s2" in ctrl:
        acc_candidates.append(math.degrees(float(ctrl["max_joint_acceleration_rad_s2"])))
    max_acc_deg_s2 = min(acc_candidates) if acc_candidates else 720.0

    return max_vel_deg_s, max_acc_deg_s2


def _infer_widths_from_urdf_q(profile: RobotProfile, q_traj: np.ndarray) -> dict[str, np.ndarray]:
    """Recover per-side physical gripper width (m) from URDF gripper joint positions."""
    T = len(q_traj)
    out: dict[str, np.ndarray] = {}
    for side in profile.active_sides:
        g_specs = profile.arms[side].gripper_joints
        if not g_specs:
            out[side] = np.full(T, 0.5 * profile.gripper_max_width_m, dtype=np.float64)
            continue
        gs = g_specs[0]
        span = gs.open_value - gs.closed_value
        if abs(span) < 1e-9:
            out[side] = np.full(T, 0.5 * profile.gripper_max_width_m, dtype=np.float64)
        else:
            opening = np.clip((q_traj[:, gs.actuated_index] - gs.closed_value) / span, 0.0, 1.0)
            out[side] = opening * profile.gripper_max_width_m
    return out


def preflight_real_trajectory(
    profile: RobotProfile,
    q_traj: np.ndarray,
    *,
    fps: float = 30.0,
    speed_scale: float = 0.5,
    table_z_robot: float | None = None,
    enforce_clearance: bool = False,
) -> dict[str, Any]:
    """Verify scaled trajectory against finite values, URDF joint limits, speed/acc bounds, and optional clearance."""
    solver = BimanualIKSolver(profile)
    arr = np.asarray(q_traj, dtype=np.float64)
    reasons: list[str] = []

    if arr.ndim != 2 or arr.shape[1] != profile.urdf_model.num_actuated:
        reasons.append(f"invalid_q_traj_shape:{arr.shape}")
        return {
            "passed": False,
            "robot": profile.name,
            "fault_mode": str(profile.real_config.get("fault_mode", "HOLD_LAST_SAFE")),
            "speed_scale": speed_scale,
            "peak_joint_speed_deg_s": 0.0,
            "peak_joint_acceleration_deg_s2": 0.0,
            "reasons": reasons,
        }

    if not np.all(np.isfinite(arr)):
        reasons.append("non_finite_joint_values")
        return {
            "passed": False,
            "robot": profile.name,
            "fault_mode": str(profile.real_config.get("fault_mode", "HOLD_LAST_SAFE")),
            "speed_scale": speed_scale,
            "peak_joint_speed_deg_s": float("nan"),
            "peak_joint_acceleration_deg_s2": float("nan"),
            "reasons": reasons,
        }

    X = arr[:, list(solver.active_indices)]
    dt_eff = 1.0 / (float(fps) * max(float(speed_scale), 0.05))
    max_vel_deg_s, max_acc_deg_s2 = _resolve_real_motion_limits(profile)

    if len(X) > 0:
        pos_tol = 1e-4
        if bool(np.any(X < solver.lower[None, :] - pos_tol)) or bool(
            np.any(X > solver.upper[None, :] + pos_tol)
        ):
            reasons.append("exceeds_urdf_joint_limits")

    vel_rad_s = np.abs(np.diff(X, axis=0)) / dt_eff if len(X) >= 2 else np.zeros((1, solver.n_active))
    acc_rad_s2 = np.abs(np.diff(X, n=2, axis=0)) / (dt_eff**2) if len(X) >= 3 else np.zeros((1, solver.n_active))

    peak_vel_deg_s = float(np.degrees(np.max(vel_rad_s)))
    peak_acc_deg_s2 = float(np.degrees(np.max(acc_rad_s2)))
    urdf_vel_ok = bool(np.all(vel_rad_s <= solver.vel_limits[None, :] * 1.001))

    if not urdf_vel_ok:
        reasons.append("exceeds_urdf_velocity_limits")
    if peak_vel_deg_s > max_vel_deg_s:
        reasons.append(f"peak_vel_deg_s:{peak_vel_deg_s:.1f}>{max_vel_deg_s:.1f}")
    if peak_acc_deg_s2 > max_acc_deg_s2:
        reasons.append(f"peak_acc_deg_s2:{peak_acc_deg_s2:.1f}>{max_acc_deg_s2:.1f}")

    if enforce_clearance and len(arr) > 0:
        min_inter_arm = float(profile.real_config.get("min_inter_arm_clearance_m", 0.005))
        min_table = float(profile.real_config.get("min_table_clearance_m", -0.005))
        ee_links = tuple(profile.arms[s].ee_link for s in profile.active_sides)
        worst_inter = float("inf")
        worst_table = float("inf")
        for t in range(len(arr)):
            inter_clr, table_clr = evaluate_config_clearance(
                profile.urdf_model,
                arr[t],
                table_z_in_robot_world=table_z_robot,
                ee_links=ee_links,
            )
            worst_inter = min(worst_inter, inter_clr)
            worst_table = min(worst_table, table_clr)
        if worst_inter < min_inter_arm:
            reasons.append(f"inter_arm_clearance_m:{worst_inter:.4f}<{min_inter_arm:.4f}")
        if table_z_robot is not None and worst_table < min_table:
            reasons.append(f"table_clearance_m:{worst_table:.4f}<{min_table:.4f}")

    return {
        "passed": len(reasons) == 0,
        "robot": profile.name,
        "fault_mode": str(profile.real_config.get("fault_mode", "HOLD_LAST_SAFE")),
        "speed_scale": speed_scale,
        "peak_joint_speed_deg_s": round(peak_vel_deg_s, 2),
        "peak_joint_acceleration_deg_s2": round(peak_acc_deg_s2, 2),
        "reasons": reasons,
    }


def smooth_joint_trajectory(
    profile: RobotProfile,
    q_traj: np.ndarray,
    widths_by_side: dict[str, np.ndarray],
    *,
    fps: float = 30.0,
    cutoff_hz: float = 3.0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Apply zero-phase 4th-order Butterworth low-pass filtering to arm joints and gripper widths.

    Preserves timing (zero phase lag), clamps filtered joints strictly within URDF bounds, and
    returns copies unchanged when `cutoff_hz <= 0` or the trajectory is too short to filter.
    """
    q_out = np.asarray(q_traj, dtype=np.float64).copy()
    w_out = {s: np.asarray(v, dtype=np.float64).copy() for s, v in widths_by_side.items()}
    if cutoff_hz <= 0.0 or len(q_out) < 16 or fps <= 0.0:
        return q_out, w_out
    nyq = 0.5 * float(fps)
    wn = float(cutoff_hz) / nyq
    if wn <= 0.0 or wn >= 0.95:
        return q_out, w_out

    from scipy.signal import butter, filtfilt

    b, a = butter(4, wn, btype="low")
    padlen = min(3 * max(len(a), len(b)), len(q_out) - 1)
    lower_all = np.asarray(profile.urdf_model.lower_limits, dtype=np.float64)
    upper_all = np.asarray(profile.urdf_model.upper_limits, dtype=np.float64)
    for side in profile.active_sides:
        idxs = list(profile.arms[side].arm_actuated_indices)
        if idxs:
            filt_arm = filtfilt(b, a, q_out[:, idxs], axis=0, padlen=padlen)
            q_out[:, idxs] = np.clip(filt_arm, lower_all[idxs][None, :], upper_all[idxs][None, :])
        if side in w_out and len(w_out[side]) == len(q_out):
            w_filt = filtfilt(b, a, w_out[side], axis=0, padlen=padlen)
            w_out[side] = np.clip(w_filt, 0.0, float(profile.gripper_max_width_m))
    return q_out, w_out


def resample_trajectory_cubic(
    q_traj: np.ndarray,
    widths_by_side: dict[str, np.ndarray],
    *,
    source_hz: float,
    target_hz: float,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Upsample a time-scaled trajectory with a clamped C^2 CubicSpline (zero start/end velocity)."""
    arr = np.asarray(q_traj, dtype=np.float64)
    if len(arr) < 4 or target_hz <= source_hz + 1e-6 or source_hz <= 0.0:
        return arr, {s: np.asarray(v, dtype=np.float64) for s, v in widths_by_side.items()}, float(source_hz)

    from scipy.interpolate import CubicSpline

    t_src = np.arange(len(arr), dtype=np.float64) / float(source_hz)
    t_end = float(t_src[-1])
    n_dst = max(len(arr), int(round(t_end * float(target_hz))) + 1)
    t_dst = np.linspace(0.0, t_end, n_dst, dtype=np.float64)
    actual_hz = (n_dst - 1) / t_end if t_end > 0.0 else float(target_hz)

    cs = CubicSpline(t_src, arr, axis=0, bc_type="clamped")
    q_dense = cs(t_dst)
    w_dense = {
        s: np.interp(t_dst, t_src, np.asarray(v, dtype=np.float64))
        for s, v in widths_by_side.items()
    }
    return q_dense, w_dense, actual_hz


def execute_real_trajectory(
    backend: RealRobotBackend,
    profile: RobotProfile,
    q_traj: np.ndarray,
    widths_by_side: dict[str, np.ndarray],
    *,
    fps: float = 30.0,
    speed_scale: float = 0.5,
    ramp_speed_deg_s: float | None = None,
    smooth_cutoff_hz: float = 0.0,
    stream_hz: float | None = None,
    tracking_filter_hz: float | None = None,
    table_z_robot: float | None = None,
    enforce_clearance: bool = False,
    sleep_realtime: bool = True,
    dry_run: bool = False,
    stop_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Preflight, slowly ramp to start, and stream `q_traj` through `GuardedRobotController`.

    On completion, fault, or `KeyboardInterrupt`, always latches `HOLD_LAST_SAFE` via
    `backend.hold_last_safe()` and disconnects cleanly WITHOUT releasing holding torque.
    """
    q_orig_len = int(len(q_traj))
    if smooth_cutoff_hz > 0.0 and q_orig_len >= 16:
        q_work, widths_work = smooth_joint_trajectory(
            profile,
            q_traj,
            widths_by_side,
            fps=fps,
            cutoff_hz=smooth_cutoff_hz,
        )
    else:
        q_work = np.asarray(q_traj, dtype=np.float64)
        widths_work = {s: np.asarray(v, dtype=np.float64) for s, v in widths_by_side.items()}

    report = preflight_real_trajectory(
        profile,
        q_work,
        fps=fps,
        speed_scale=speed_scale,
        table_z_robot=table_z_robot,
        enforce_clearance=enforce_clearance,
    )
    report["dry_run"] = bool(dry_run)
    report["total_frames"] = q_orig_len
    report["executed_frames"] = 0
    report["smooth_cutoff_hz"] = float(smooth_cutoff_hz)
    report["tracking_filter_hz"] = None if tracking_filter_hz is None else float(tracking_filter_hz)
    report["ramped_to_start"] = False
    report["completed"] = False
    report["final_mode"] = ArmStateMode.HOLD_LAST_SAFE.value
    report["fault_reasons"] = list(report["reasons"])

    if not report["passed"]:
        return report
    if q_orig_len == 0:
        report["completed"] = True
        return report

    effective_hz = max(float(fps) * max(float(speed_scale), 0.05), 1.0)
    if stream_hz is not None and float(stream_hz) > effective_hz and len(q_work) >= 4:
        q_stream, widths_stream, loop_hz = resample_trajectory_cubic(
            q_work,
            widths_work,
            source_hz=effective_hz,
            target_hz=float(stream_hz),
        )
    else:
        q_stream = q_work
        widths_stream = widths_work
        loop_hz = effective_hz

    report["stream_hz"] = round(float(loop_hz), 2)
    report["streamed_steps"] = 0
    period_s = 1.0 / loop_hz
    max_vel_deg_s, _ = _resolve_real_motion_limits(profile)
    max_vel_rad_s = math.radians(max_vel_deg_s)
    # Allow nominal scaled motion with tight jump guard
    step_limit_rad = max(0.08, max_vel_rad_s * period_s * 1.5)

    guard = GuardedRobotController(
        profile=profile,
        max_step_rad=step_limit_rad,
        table_z_robot=table_z_robot,
        enforce_clearance=enforce_clearance,
    )

    widths_0 = {s: float(widths_stream[s][0]) for s in profile.active_sides if s in widths_stream}
    streamed = 0
    interrupted = False

    backend.connect()
    try:
        ramped = backend.ramp_to_q(
            q_stream[0],
            widths_0,
            speed_deg_s=ramp_speed_deg_s,
            stop_event=stop_event,
        )
        report["ramped_to_start"] = bool(ramped)
        if not ramped:
            guard.latch_hold_last_safe("ramp_to_start_timeout_or_aborted")
            backend.hold_last_safe()
            report["final_mode"] = guard.mode.value
            report["fault_reasons"] = list(guard.fault_reasons)
            return report

        if tracking_filter_hz is not None and float(tracking_filter_hz) > 0.0:
            set_tf = getattr(backend, "set_tracking_filter", None)
            if callable(set_tf):
                set_tf(float(tracking_filter_hz))

        guard.seed_state(
            q_stream[0],
            widths_0,
            mode=ArmStateMode.DRY_RUN_MOCK if dry_run else ArmStateMode.ACTIVE_TRACKING,
        )

        deadline = time.monotonic()
        for t in range(len(q_stream)):
            if stop_event is not None and stop_event.is_set():
                interrupted = True
                break
            try:
                backend.check_health()
            except BaseException as exc:
                guard.latch_hold_last_safe(f"backend_health_fault:{exc}")
                backend.hold_last_safe()
                break

            w_t = {
                s: float(widths_stream[s][t])
                for s in profile.active_sides
                if s in widths_stream
            }
            q_cmd, w_cmd, mode = guard.filter_command(q_stream[t], w_t, input_healthy=True)
            if mode == ArmStateMode.HOLD_LAST_SAFE:
                backend.hold_last_safe()
                break

            backend.write_target(q_cmd, w_cmd)
            streamed += 1

            if sleep_realtime and t + 1 < len(q_stream):
                deadline = next_periodic_deadline(deadline, time.monotonic(), period_s)
                sleep_s = deadline - time.monotonic()
                if sleep_s > 0.0:
                    if stop_event is not None:
                        if stop_event.wait(sleep_s):
                            interrupted = True
                            break
                    else:
                        time.sleep(sleep_s)

        # Let the smooth 3rd-order tracking filter glide cleanly to v=0 at the final target
        if (
            sleep_realtime
            and not interrupted
            and streamed == len(q_stream)
            and guard.mode != ArmStateMode.HOLD_LAST_SAFE
            and tracking_filter_hz is not None
            and float(tracking_filter_hz) > 0.0
        ):
            settle_s = min(0.35, max(0.10, 0.8 / float(tracking_filter_hz)))
            settle_end = time.monotonic() + settle_s
            while time.monotonic() < settle_end:
                if stop_event is not None and stop_event.is_set():
                    break
                backend.check_health()
                time.sleep(0.01)
    except KeyboardInterrupt:
        interrupted = True
        guard.latch_hold_last_safe("operator_keyboard_interrupt")
        backend.hold_last_safe()
    except BaseException as exc:
        guard.latch_hold_last_safe(f"runtime_exception:{exc}")
        backend.hold_last_safe()
        raise
    finally:
        backend.hold_last_safe()
        backend.disconnect()

    report["streamed_steps"] = streamed
    if len(q_stream) == q_orig_len:
        executed_frames = streamed
    elif streamed >= len(q_stream):
        executed_frames = q_orig_len
    else:
        executed_frames = int(round(streamed * q_orig_len / max(len(q_stream), 1)))
    report["executed_frames"] = executed_frames
    report["interrupted"] = interrupted
    report["completed"] = (
        streamed == len(q_stream)
        and not interrupted
        and guard.mode != ArmStateMode.HOLD_LAST_SAFE
    )
    report["final_mode"] = guard.mode.value
    report["fault_reasons"] = list(guard.fault_reasons)
    return report


def run_replay_real_cli(args: argparse.Namespace) -> int:
    prof = load_robot_profile(args.robot)
    if args.live:
        tcal = require_live_authorization(
            robot=args.robot,
            live=args.live,
            operator_supported=args.operator_supported,
        )
    else:
        tcal = load_table_calibration(args.robot, require_physical=False)

    table_z_robot = float(pose7_to_mat(tcal.T_robot_table_pose7)[2, 3])
    ds_root = Path(args.dataset_dir)
    info = load_dataset_info(ds_root)
    fps = float(info.get("fps", 30.0))
    npz_path = ds_root / "data" / f"episode_{int(args.episode):06d}.npz"
    q_traj: np.ndarray | None = None
    widths_by_side: dict[str, np.ndarray] | None = None

    if npz_path.is_file():
        with np.load(npz_path) as z:
            if "urdf_q_full" in z:
                if info.get("robot") and info.get("robot") != args.robot:
                    raise ValueError(
                        f"Joint dataset robot={info.get('robot')!r} does not match --robot={args.robot!r}"
                    )
                if args.live:
                    ds_calib_sha = info.get("table_calib_sha256")
                    if not ds_calib_sha or ds_calib_sha != tcal.sha256:
                        raise PermissionError(
                            f"Refusing live replay of pre-solved joint trajectory with mismatched/missing "
                            f"table_calib_sha256 (dataset={ds_calib_sha!r}, active physical calib={tcal.sha256!r}). "
                            "Re-convert the dataset with the verified physical table calibration before live replay."
                        )
                q_traj = z["urdf_q_full"].astype(np.float64)
                widths_by_side = _infer_widths_from_urdf_q(prof, q_traj)

    if q_traj is None:
        mode = "relative-anchor" if len(prof.active_sides) == 1 else "absolute-table"
        ee_tgts, widths_by_side, _, table_z_robot = prepare_episode_robot_targets(
            str(ds_root), args.episode, prof, table_calib=tcal, retarget_mode=mode
        )
        q_traj, _, _ = solve_episode_with_method(
            prof, ee_tgts, widths_by_side, solver_name="cstro", fps=fps, table_z_robot=table_z_robot
        )

    assert widths_by_side is not None
    smooth_cutoff_hz = float(getattr(args, "smooth_cutoff_hz", 3.0))
    stream_hz_arg = getattr(args, "stream_hz", 100.0)
    stream_hz = None if stream_hz_arg is None or float(stream_hz_arg) <= 0.0 else float(stream_hz_arg)
    tracking_filter_arg = getattr(args, "tracking_filter_hz", 5.0)
    tracking_filter_hz = (
        None if tracking_filter_arg is None or float(tracking_filter_arg) <= 0.0 else float(tracking_filter_arg)
    )

    if args.live:
        settings = load_piper_can_settings(
            prof,
            rig_config_path=getattr(args, "rig_config", None),
            left_port=getattr(args, "left_port", None),
            right_port=getattr(args, "right_port", None),
            speed_percent=getattr(args, "speed_percent", None),
            sdk_backend=getattr(args, "sdk", None),
            firmware=getattr(args, "firmware", None),
            enable_arms=bool(getattr(args, "enable_arms", False)),
        )
        backend: RealRobotBackend = PiperCanBackend(prof, settings)
        report = execute_real_trajectory(
            backend,
            prof,
            q_traj,
            widths_by_side,
            fps=fps,
            speed_scale=args.speed_scale,
            ramp_speed_deg_s=getattr(args, "ramp_speed_deg_s", None),
            smooth_cutoff_hz=smooth_cutoff_hz,
            stream_hz=stream_hz,
            tracking_filter_hz=tracking_filter_hz,
            table_z_robot=table_z_robot,
            enforce_clearance=True,
            sleep_realtime=True,
            dry_run=False,
        )
    elif getattr(args, "mock_robot", False):
        mock_backend = MockBimanualController(prof)
        report = execute_real_trajectory(
            mock_backend,
            prof,
            q_traj,
            widths_by_side,
            fps=fps,
            speed_scale=args.speed_scale,
            ramp_speed_deg_s=getattr(args, "ramp_speed_deg_s", None),
            smooth_cutoff_hz=smooth_cutoff_hz,
            stream_hz=None,
            tracking_filter_hz=None,
            table_z_robot=table_z_robot,
            enforce_clearance=False,
            sleep_realtime=False,
            dry_run=True,
        )
        report["mock_robot"] = True
    else:
        q_pf, _ = (
            smooth_joint_trajectory(prof, q_traj, widths_by_side, fps=fps, cutoff_hz=smooth_cutoff_hz)
            if smooth_cutoff_hz > 0.0 and len(q_traj) >= 16
            else (q_traj, widths_by_side)
        )
        report = preflight_real_trajectory(prof, q_pf, fps=fps, speed_scale=args.speed_scale)
        report["smooth_cutoff_hz"] = smooth_cutoff_hz
        report["dry_run"] = True

    if args.json_out:
        save_json_atomic(args.json_out, report)
    ok = bool(report["passed"]) and (not args.live or bool(report.get("completed", False)))
    print(
        f"[{'OK' if ok else 'FAIL'}] replay-real ({args.robot}, dry_run={not args.live}): "
        f"peak_vel={report['peak_joint_speed_deg_s']} deg/s, "
        f"peak_acc={report['peak_joint_acceleration_deg_s2']} deg/s^2"
        + (
            f", frames={report.get('executed_frames', 0)}/{report.get('total_frames', len(q_traj))}"
            if "executed_frames" in report
            else ""
        )
    )
    return 0 if ok else 1
