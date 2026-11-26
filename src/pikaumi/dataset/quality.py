"""Raw Pika Sense episode QA gating (`pika dataset validate`): tracking, sync, velocity, freeze & jaw motion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pikaumi.config import QUALITY_CONFIG, load_yaml, save_json_atomic
from pikaumi.dataset.raw import load_dataset_episodes, load_dataset_info, load_episode_raw
from pikaumi.robots.utils import normalize_quat_xyzw


def _longest_true_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for v in mask:
        if bool(v):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def validate_dataset(
    dataset_dir: str | Path,
    *,
    quality_config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    qcfg = load_yaml(quality_config_path or QUALITY_CONFIG)
    info = load_dataset_info(root)
    episodes = load_dataset_episodes(root)
    fps = float(info.get("fps", 30.0))

    min_frames = int(qcfg.get("min_frames", 10))
    min_dur_s = float(qcfg.get("min_duration_s", 1.0))
    warmup_s = float(qcfg.get("warmup_s", 0.2))
    max_bad_track_frac = float(qcfg.get("max_bad_tracking_fraction", 0.01))
    max_skew_s = float(qcfg.get("max_sync_skew_ms", 40.0)) / 1000.0
    max_lr_skew_s = float(qcfg.get("max_lr_image_skew_ms", 10.0)) / 1000.0
    max_sensor_gap_ms = float(qcfg.get("max_sensor_gap_ms", 30.0))
    max_bad_sync_frac = float(qcfg.get("max_bad_sync_fraction", 0.01))
    max_lin_speed = float(qcfg.get("max_linear_speed_m_s", 2.5))
    max_ang_speed = float(qcfg.get("max_angular_speed_rad_s", 15.0))
    max_frozen_s = float(qcfg.get("max_frozen_pose_ms", 250.0)) / 1000.0
    min_jaw_range_m = float(qcfg.get("min_gripper_range_m", 0.002))

    per_ep: list[dict[str, Any]] = []
    passed_count = 0

    for ep_meta in episodes:
        ep_idx = int(ep_meta["episode_index"])
        ep = load_episode_raw(root, ep_idx)
        ts = ep["timestamps_s"]
        state = ep["observation_state"]
        sync_skew = ep["sync_skew_s"]
        lr_skew = ep.get("lr_image_skew_s")
        T = len(ts)
        reasons: list[str] = []

        finite_ok = (
            bool(np.isfinite(ts).all())
            and bool(np.isfinite(state).all())
            and bool(np.isfinite(sync_skew).all())
            and (lr_skew is None or bool(np.isfinite(lr_skew).all()))
        )
        if not finite_ok:
            reasons.append("non_finite_values")

        dur_s = float(ts[-1] - ts[0] + 1.0 / fps) if T > 0 and np.isfinite(ts).all() else 0.0
        if T < min_frames:
            reasons.append(f"too_few_frames:{T}<{min_frames}")
        if dur_s < min_dur_s:
            reasons.append(f"too_short_s:{dur_s:.2f}<{min_dur_s:.2f}")

        # Apply configured warmup_s skip window when sufficient frames remain
        if T >= 2 and np.isfinite(ts).all() and warmup_s > 0.0:
            warm_mask = (ts - ts[0]) >= warmup_s
            if int(np.sum(warm_mask)) >= 2:
                eval_idx = np.flatnonzero(warm_mask)
            else:
                eval_idx = np.arange(T)
        else:
            eval_idx = np.arange(T)

        bad_track = ~(ep["left_accurate"][eval_idx] & ep["right_accurate"][eval_idx]) if len(eval_idx) > 0 else np.array([True])
        bad_track_frac = float(np.mean(bad_track)) if len(eval_idx) > 0 else 1.0
        if bad_track_frac > max_bad_track_frac:
            reasons.append(f"tracking_dropout_fraction:{bad_track_frac:.3f}>{max_bad_track_frac:.3f}")

        bad_sync_frac = float(np.mean(sync_skew[eval_idx] > max_skew_s)) if len(eval_idx) > 0 else 1.0
        if bad_sync_frac > max_bad_sync_frac:
            reasons.append(f"sync_skew_fraction:{bad_sync_frac:.3f}>{max_bad_sync_frac:.3f}")

        bad_lr_skew_frac = 0.0
        if lr_skew is not None and len(eval_idx) > 0 and np.isfinite(lr_skew).all():
            bad_lr_skew_frac = float(np.mean(lr_skew[eval_idx] > max_lr_skew_s))
            if bad_lr_skew_frac > max_bad_sync_frac:
                reasons.append(f"lr_image_skew_fraction:{bad_lr_skew_frac:.3f}>{max_bad_sync_frac:.3f}")

        max_obs_lin = 0.0
        max_obs_ang = 0.0
        max_obs_frozen_s = 0.0
        max_gap_excess_ms = float(ep_meta.get("max_sensor_gap_ms", 0.0) or 0.0)
        if finite_ok and len(eval_idx) >= 2:
            ts_e = ts[eval_idx]
            st_e = state[eval_idx]
            dt_raw = np.diff(ts_e)
            nominal_dt_ms = 1000.0 / fps
            max_gap_excess_ms = max(
                max_gap_excess_ms,
                max(0.0, float(np.max(dt_raw) * 1000.0) - nominal_dt_ms),
            )
            if max_gap_excess_ms > max_sensor_gap_ms:
                reasons.append(f"sensor_gap_ms:{max_gap_excess_ms:.1f}>{max_sensor_gap_ms:.1f}")

            dt = np.maximum(dt_raw, 1.0 / (fps * 4.0))
            for side_name, p0 in (("left", 0), ("right", 8)):
                pos = st_e[:, p0 : p0 + 3]
                quat = normalize_quat_xyzw(st_e[:, p0 + 3 : p0 + 7])
                dpos = np.linalg.norm(np.diff(pos, axis=0), axis=1)
                lin_vel = dpos / dt
                dots = np.clip(np.abs(np.sum(quat[:-1] * quat[1:], axis=1)), 0.0, 1.0)
                dang = 2.0 * np.arccos(dots)
                ang_vel = dang / dt

                max_obs_lin = max(max_obs_lin, float(np.max(lin_vel)))
                max_obs_ang = max(max_obs_ang, float(np.max(ang_vel)))
                if float(np.max(lin_vel)) > max_lin_speed:
                    reasons.append(f"{side_name}_linear_speed_spike:{float(np.max(lin_vel)):.2f}>{max_lin_speed:.2f}")
                if float(np.max(ang_vel)) > max_ang_speed:
                    reasons.append(f"{side_name}_angular_speed_spike:{float(np.max(ang_vel)):.2f}>{max_ang_speed:.2f}")

                frozen_steps = _longest_true_run((dpos < 1e-7) & (dang < 1e-7))
                frozen_s = frozen_steps / fps
                max_obs_frozen_s = max(max_obs_frozen_s, frozen_s)
                if frozen_s >= max_frozen_s:
                    reasons.append(f"{side_name}_frozen_pose_s:{frozen_s:.3f}>={max_frozen_s:.3f}")

        if finite_ok and T > 0:
            jaw_span_l = float(np.max(state[:, 7]) - np.min(state[:, 7]))
            jaw_span_r = float(np.max(state[:, 15]) - np.min(state[:, 15]))
            if max(jaw_span_l, jaw_span_r) < min_jaw_range_m:
                reasons.append(f"no_jaw_motion_m:{max(jaw_span_l, jaw_span_r):.4f}<{min_jaw_range_m:.4f}")
        else:
            jaw_span_l = 0.0
            jaw_span_r = 0.0

        ok = len(reasons) == 0
        if ok:
            passed_count += 1
        per_ep.append(
            {
                "episode_index": ep_idx,
                "passed": ok,
                "num_frames": T,
                "bad_tracking_fraction": bad_track_frac,
                "bad_lr_image_skew_fraction": bad_lr_skew_frac,
                "max_sensor_gap_excess_ms": max_gap_excess_ms,
                "max_linear_speed_m_s": max_obs_lin,
                "max_angular_speed_rad_s": max_obs_ang,
                "max_frozen_pose_s": max_obs_frozen_s,
                "jaw_span_m": {"left": jaw_span_l, "right": jaw_span_r},
                "reasons": reasons,
            }
        )

    report = {
        "dataset_dir": str(root),
        "num_episodes": len(per_ep),
        "num_passed": passed_count,
        "num_rejected": len(per_ep) - passed_count,
        "episodes": per_ep,
    }
    save_json_atomic(root / "meta" / "quality_report.json", report)
    return report
