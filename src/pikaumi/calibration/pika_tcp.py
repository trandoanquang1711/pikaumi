"""Layer 4 Calibration: Pika Sense Tracker-to-TCP pivot translation & 3-axis SVD orientation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from pikaumi.config import sha256_object
from pikaumi.robots.utils import pose7_to_mat


def solve_tcp_pivot_translation(
    poses7: np.ndarray,
    *,
    max_rms_m: float = 0.005,
    max_err_m: float = 0.010,
    max_condition_number: float = 500.0,
) -> dict[str, Any]:
    """Solve `R_i @ p_tcp_in_tracker + t_i = p_world_pivot` via linear least squares."""
    arr = np.asarray(poses7, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 7 or len(arr) < 6:
        raise ValueError("Pivot calibration requires at least 6 diverse (N, 7) tracker poses")

    n = len(arr)
    A = np.zeros((3 * n, 6), dtype=np.float64)
    b = np.zeros(3 * n, dtype=np.float64)
    mats = [pose7_to_mat(p) for p in arr]
    for i, T in enumerate(mats):
        A[3 * i : 3 * i + 3, 0:3] = T[:3, :3]
        A[3 * i : 3 * i + 3, 3:6] = -np.eye(3, dtype=np.float64)
        b[3 * i : 3 * i + 3] = -T[:3, 3]

    sol, _, _, svals = np.linalg.lstsq(A, b, rcond=None)
    p_tcp = sol[0:3]
    p_pivot = sol[3:6]
    cond = float(svals[0] / max(svals[-1], 1e-12))

    errs = np.array([np.linalg.norm(T[:3, :3] @ p_tcp + T[:3, 3] - p_pivot) for T in mats], dtype=np.float64)
    rms_m = float(np.sqrt(np.mean(errs**2)))
    max_m = float(np.max(errs))

    reasons: list[str] = []
    if rms_m > max_rms_m:
        reasons.append(f"pivot_rms_m:{rms_m:.4f}>{max_rms_m:.4f}")
    if max_m > max_err_m:
        reasons.append(f"pivot_max_m:{max_m:.4f}>{max_err_m:.4f}")
    if cond > max_condition_number:
        reasons.append(f"pivot_ill_conditioned:{cond:.1f}>{max_condition_number:.1f}")

    return {
        "tcp_translation_m": p_tcp.tolist(),
        "pivot_world_m": p_pivot.tolist(),
        "samples": n,
        "rms_error_m": rms_m,
        "max_error_m": max_m,
        "condition_number": cond,
        "passed": len(reasons) == 0,
        "reasons": reasons,
    }


def solve_tcp_axes_svd(
    snapshots: dict[str, np.ndarray],
    *,
    expected_distance_m: float = 0.080,
    max_orthogonality_deg: float = 5.0,
) -> dict[str, Any]:
    """Solve `R_tracker_tcp` from signed pure-translation snapshots `(a0, ax, ay, az[, b0, bx, by, bz])`.

    Projects world displacements `(p_{k} - p_{0})` into the tracker body frame `R_0^T (p_k - p_0)`
    and projects onto `SO(3)` via SVD (`U @ diag(1, 1, det(U V^T)) @ V^T`).
    """
    groups = [("a0", "ax", "ay", "az")]
    if all(k in snapshots for k in ("b0", "bx", "by", "bz")):
        groups.append(("b0", "bx", "by", "bz"))

    cols_accum = np.zeros((3, 3), dtype=np.float64)
    raw_orth_errors_deg: list[float] = []
    step_lengths_m: list[float] = []

    for k0, kx, ky, kz in groups:
        T0 = pose7_to_mat(snapshots[k0])
        R0 = T0[:3, :3]
        p0 = T0[:3, 3]
        raw_vecs: list[np.ndarray] = []
        for idx, kk in enumerate((kx, ky, kz)):
            pk = np.asarray(snapshots[kk], dtype=np.float64)[:3]
            v_local = R0.T @ (pk - p0)
            norm = float(np.linalg.norm(v_local))
            if norm < 0.015:
                raise ValueError(f"Axis snapshot {kk} displacement too small ({norm*1000:.1f} mm)")
            step_lengths_m.append(norm)
            unit_v = v_local / norm
            cols_accum[:, idx] += unit_v
            raw_vecs.append(unit_v)
        # Pairwise orthogonality check from 90 deg
        for i in range(3):
            for j in range(i + 1, 3):
                ang = float(np.degrees(np.arccos(np.clip(np.dot(raw_vecs[i], raw_vecs[j]), -1.0, 1.0))))
                raw_orth_errors_deg.append(abs(ang - 90.0))

    M = cols_accum / float(len(groups))
    det_m = float(np.linalg.det(M))
    U, _, Vt = np.linalg.svd(M)
    det_sign = 1.0 if np.linalg.det(U @ Vt) >= 0.0 else -1.0
    R_tracker_tcp = U @ np.diag([1.0, 1.0, det_sign]) @ Vt
    quat_xyzw = Rotation.from_matrix(R_tracker_tcp).as_quat()

    max_orth_deg = float(max(raw_orth_errors_deg)) if raw_orth_errors_deg else 0.0
    reasons: list[str] = []
    if det_m <= 0.0:
        reasons.append(f"left_handed_axes_frame:det={det_m:.4f}")
    if max_orth_deg > max_orthogonality_deg:
        reasons.append(f"axis_orthogonality_deg:{max_orth_deg:.2f}>{max_orthogonality_deg:.2f}")

    return {
        "quaternion_xyzw": quat_xyzw.tolist(),
        "rotation_matrix": R_tracker_tcp.tolist(),
        "raw_axes_determinant": det_m,
        "max_orthogonality_error_deg": max_orth_deg,
        "mean_step_length_m": float(np.mean(step_lengths_m)),
        "expected_distance_m": expected_distance_m,
        "passed": len(reasons) == 0,
        "reasons": reasons,
    }


def load_snapshots_directory(directory: str | Path) -> dict[str, np.ndarray]:
    root = Path(directory)
    out: dict[str, np.ndarray] = {}
    for name in ("a0", "ax", "ay", "az", "b0", "bx", "by", "bz"):
        f = root / f"{name}.json"
        if f.is_file():
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                pose = data.get("pose7")
                if pose is None:
                    pose = data.get("pose")
                if pose is None:
                    pose = data.get("T_vive_tracker")
                if pose is None:
                    pose = data
                if isinstance(pose, dict) and "position" in pose and "quaternion" in pose:
                    pose = [*pose["position"], *pose["quaternion"]]
            else:
                pose = data
            out[name] = np.asarray(pose, dtype=np.float64)
    return out


def compute_pika_tcp_sha256(pika_tcp_yaml: dict[str, Any]) -> str:
    core = {
        "device": pika_tcp_yaml.get("device", "vive_pika_sense_v1"),
        "gripper_max_width_m": float(pika_tcp_yaml.get("gripper_max_width_m", 0.10)),
        "sides": {
            s: {
                "position": [round(float(v), 8) for v in side_cfg["position"]],
                "quaternion": [round(float(v), 8) for v in side_cfg["quaternion"]],
            }
            for s, side_cfg in sorted((pika_tcp_yaml.get("sides") or {}).items())
        },
    }
    return sha256_object(core)
