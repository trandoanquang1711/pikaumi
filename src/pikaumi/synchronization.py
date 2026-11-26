"""Causal 40 ms lag-compensated sensor grid synchronization (`SE(3)` SLERP + linear jaw width)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from pikaumi.robots.utils import normalize_quat_xyzw

DEFAULT_SYNC_LAG_S = 0.040


def build_causal_sync_grid(
    start_ts_s: float,
    end_ts_s: float,
    *,
    fps: float = 30.0,
    lag_s: float = DEFAULT_SYNC_LAG_S,
) -> np.ndarray:
    """Construct uniform query timestamps `t_k` shifted back by `lag_s` so interpolation never extrapolates."""
    dt = 1.0 / float(fps)
    t0 = float(start_ts_s)
    t_last = float(end_ts_s) - float(lag_s)
    if t_last <= t0:
        return np.zeros((0,), dtype=np.float64)
    n = int(np.floor((t_last - t0) / dt)) + 1
    return t0 + np.arange(n, dtype=np.float64) * dt


def interpolate_se3_trajectory(
    src_ts_s: np.ndarray,
    src_poses7: np.ndarray,
    query_ts_s: np.ndarray,
    *,
    accurate_flags: np.ndarray | None = None,
    max_gap_s: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate `(N, 7)` `SE(3)` poses onto `query_ts_s` via linear XYZ + quaternion SLERP.

    Returns `(interp_poses7, valid_mask, nearest_skew_s)`.
    """
    ts = np.asarray(src_ts_s, dtype=np.float64)
    poses = np.asarray(src_poses7, dtype=np.float64)
    q_ts = np.asarray(query_ts_s, dtype=np.float64)
    if len(q_ts) == 0:
        return (
            np.zeros((0, 7), dtype=np.float64),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=np.float64),
        )
    if len(ts) < 2:
        raise ValueError("Require at least 2 source pose samples for SE(3) interpolation")

    order = np.argsort(ts)
    ts = ts[order]
    poses = poses[order]
    acc = np.ones(len(ts), dtype=bool) if accurate_flags is None else np.asarray(accurate_flags, dtype=bool)[order]

    # Ensure strictly increasing timestamps for scipy Slerp
    keep = np.concatenate([[True], np.diff(ts) > 1e-9])
    ts = ts[keep]
    poses = poses[keep]
    acc = acc[keep]

    quats = normalize_quat_xyzw(poses[:, 3:7])
    # Hemisphere continuity for smooth SLERP
    for i in range(1, len(quats)):
        if float(np.dot(quats[i - 1], quats[i])) < 0.0:
            quats[i] = -quats[i]

    clamped_q = np.clip(q_ts, ts[0], ts[-1])
    pos_interp = np.column_stack([np.interp(clamped_q, ts, poses[:, d]) for d in range(3)])
    slerp = Slerp(ts, Rotation.from_quat(quats))
    quat_interp = normalize_quat_xyzw(slerp(clamped_q).as_quat())

    idx_hi = np.clip(np.searchsorted(ts, q_ts, side="right"), 1, len(ts) - 1)
    idx_lo = idx_hi - 1
    in_bounds = (q_ts >= ts[0]) & (q_ts <= ts[-1])
    gap = ts[idx_hi] - ts[idx_lo]
    skew = np.minimum(np.abs(q_ts - ts[idx_lo]), np.abs(ts[idx_hi] - q_ts))
    valid = in_bounds & (gap <= max_gap_s) & acc[idx_lo] & acc[idx_hi]

    out_poses = np.hstack([pos_interp, quat_interp])
    return out_poses, valid, skew


def interpolate_jaw_widths(
    src_ts_s: np.ndarray,
    src_widths_m: np.ndarray,
    query_ts_s: np.ndarray,
    *,
    max_width_m: float = 0.10,
    max_gap_s: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly interpolate Pika jaw widths onto `query_ts_s` clamped to `[0, max_width_m]`."""
    ts = np.asarray(src_ts_s, dtype=np.float64)
    w = np.asarray(src_widths_m, dtype=np.float64)
    q_ts = np.asarray(query_ts_s, dtype=np.float64)
    if len(q_ts) == 0:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=bool),
            np.zeros((0,), dtype=np.float64),
        )
    order = np.argsort(ts)
    ts = ts[order]
    w = w[order]
    keep = np.concatenate([[True], np.diff(ts) > 1e-9])
    ts = ts[keep]
    w = w[keep]

    clamped_q = np.clip(q_ts, ts[0], ts[-1])
    w_interp = np.clip(np.interp(clamped_q, ts, w), 0.0, max_width_m)

    idx_hi = np.clip(np.searchsorted(ts, q_ts, side="right"), 1, len(ts) - 1)
    idx_lo = idx_hi - 1
    in_bounds = (q_ts >= ts[0]) & (q_ts <= ts[-1])
    gap = ts[idx_hi] - ts[idx_lo]
    skew = np.minimum(np.abs(q_ts - ts[idx_lo]), np.abs(ts[idx_hi] - q_ts))
    valid = in_bounds & (gap <= max_gap_s)
    return w_interp, valid, skew


@dataclass
class SustainedHealthGate:
    """Requires continuous stream health for `hold_duration_s` before arming episode capture."""

    hold_duration_s: float = 0.35
    _healthy_since_s: float | None = None

    def update(self, now_s: float, is_healthy: bool) -> bool:
        if not is_healthy:
            self._healthy_since_s = None
            return False
        if self._healthy_since_s is None:
            self._healthy_since_s = float(now_s)
        return (float(now_s) - self._healthy_since_s) >= self.hold_duration_s
