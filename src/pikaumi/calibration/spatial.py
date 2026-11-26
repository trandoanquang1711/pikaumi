"""Layer 3 Calibration: Kannala-Brandt fisheye intrinsics, hand-eye mount, and ChArUco tabletop `T_table_vive`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from pikaumi.config import sha256_object
from pikaumi.robots.utils import (
    mat_to_pose7,
    normalize_quat_xyzw,
    pose7_to_mat,
    se3_world_residual,
)


@dataclass(frozen=True)
class FisheyeIntrinsicsGate:
    passed: bool
    rms_reprojection_px: float
    accepted_views: int
    reasons: list[str]


def verify_fisheye_intrinsics_dict(
    payload: dict[str, Any],
    *,
    max_rms_px: float = 0.80,
    min_views: int = 12,
) -> FisheyeIntrinsicsGate:
    rms = float(payload.get("rms_reprojection_error_px", float("inf")))
    views = int(payload.get("accepted_view_count", 0))
    K = np.asarray(payload.get("K", np.zeros((3, 3))), dtype=np.float64)
    D = np.asarray(payload.get("D", []), dtype=np.float64)
    reasons: list[str] = []
    if K.shape != (3, 3) or K[0, 0] <= 10.0 or K[1, 1] <= 10.0:
        reasons.append("invalid_K_matrix")
    if D.size != 4:
        reasons.append(f"invalid_kannala_brandt_D_len:{D.size}")
    if rms > max_rms_px:
        reasons.append(f"rms_reprojection_px:{rms:.3f}>{max_rms_px:.3f}")
    if views < min_views:
        reasons.append(f"accepted_views:{views}<{min_views}")
    return FisheyeIntrinsicsGate(
        passed=len(reasons) == 0,
        rms_reprojection_px=rms,
        accepted_views=views,
        reasons=reasons,
    )


def average_se3_poses7(poses7: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Robust Huber SE(3) average returning `(mean_pose7, rms_pos_m, max_pos_m)`."""
    arr = np.asarray(poses7, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 7 or len(arr) == 0:
        raise ValueError("Expected non-empty (N, 7) poses7 array")
    pos_med = np.median(arr[:, :3], axis=0)
    quats = normalize_quat_xyzw(arr[:, 3:7])
    q0 = quats[0]
    aligned = np.where((quats @ q0)[:, None] < 0.0, -quats, quats)
    M = aligned.T @ aligned
    eigvals, eigvecs = np.linalg.eigh(M)
    q_mean = normalize_quat_xyzw(eigvecs[:, int(np.argmax(eigvals))])

    x0 = np.concatenate([pos_med, Rotation.from_quat(q_mean).as_rotvec()])
    mats = [pose7_to_mat(p) for p in arr]

    def residual(x: np.ndarray) -> np.ndarray:
        T_est = np.eye(4, dtype=np.float64)
        T_est[:3, 3] = x[:3]
        T_est[:3, :3] = Rotation.from_rotvec(x[3:6]).as_matrix()
        out = np.zeros(6 * len(mats), dtype=np.float64)
        for i, T_obs in enumerate(mats):
            pe, re = se3_world_residual(T_est, T_obs)
            out[6 * i : 6 * i + 3] = pe
            out[6 * i + 3 : 6 * i + 6] = 0.20 * re
        return out

    sol = least_squares(residual, x0, loss="huber", f_scale=0.005, max_nfev=100)
    T_opt = np.eye(4, dtype=np.float64)
    T_opt[:3, 3] = sol.x[:3]
    T_opt[:3, :3] = Rotation.from_rotvec(sol.x[3:6]).as_matrix()
    pose_opt = mat_to_pose7(T_opt)

    pos_errs = np.linalg.norm(arr[:, :3] - pose_opt[:3], axis=1)
    return pose_opt, float(np.sqrt(np.mean(pos_errs**2))), float(np.max(pos_errs))


def solve_hand_eye_mount(
    T_vive_tracker_seq: np.ndarray,
    T_cam_board_seq: np.ndarray,
    *,
    max_rms_m: float = 0.005,
) -> dict[str, Any]:
    """Jointly solve rigid hand-eye mount `T_tracker_cam` and fixed board `T_vive_board` via Huber SE(3).

    Observation model: `T_vive_tracker[i] @ T_tracker_cam @ T_cam_board[i] = T_vive_board`.
    """
    trk = np.asarray(T_vive_tracker_seq, dtype=np.float64)
    cam = np.asarray(T_cam_board_seq, dtype=np.float64)
    if trk.shape != cam.shape or trk.ndim != 2 or trk.shape[1] != 7 or len(trk) < 6:
        raise ValueError("Require matching (N>=6, 7) T_vive_tracker and T_cam_board sequences")

    T_wt = [pose7_to_mat(p) for p in trk]
    T_cb = [pose7_to_mat(p) for p in cam]

    # Multi-start initialization to avoid local minima
    rng = np.random.default_rng(0)
    inits = [np.zeros(12, dtype=np.float64)]
    for _ in range(4):
        inits.append(rng.uniform(-0.3, 0.3, size=12))

    def residual(params: np.ndarray) -> np.ndarray:
        T_tc = np.eye(4, dtype=np.float64)
        T_tc[:3, 3] = params[0:3]
        T_tc[:3, :3] = Rotation.from_rotvec(params[3:6]).as_matrix()

        T_wb = np.eye(4, dtype=np.float64)
        T_wb[:3, 3] = params[6:9]
        T_wb[:3, :3] = Rotation.from_rotvec(params[9:12]).as_matrix()

        res = np.zeros(6 * len(T_wt), dtype=np.float64)
        for i in range(len(T_wt)):
            T_pred = T_wt[i] @ T_tc @ T_cb[i]
            pe, re = se3_world_residual(T_pred, T_wb)
            res[6 * i : 6 * i + 3] = pe
            res[6 * i + 3 : 6 * i + 6] = 0.15 * re
        return res

    best_sol = None
    for x0 in inits:
        # Seed board translation near first tracker pose
        x0_adj = x0.copy()
        x0_adj[6:9] = T_wt[0][:3, 3] + x0[6:9] * 0.1
        sol = least_squares(residual, x0_adj, loss="huber", f_scale=0.004, max_nfev=300)
        if best_sol is None or sol.cost < best_sol.cost:
            best_sol = sol

    assert best_sol is not None
    p = best_sol.x
    T_tc = np.eye(4, dtype=np.float64)
    T_tc[:3, 3] = p[0:3]
    T_tc[:3, :3] = Rotation.from_rotvec(p[3:6]).as_matrix()
    T_wb = np.eye(4, dtype=np.float64)
    T_wb[:3, 3] = p[6:9]
    T_wb[:3, :3] = Rotation.from_rotvec(p[9:12]).as_matrix()

    pos_errs = [
        float(np.linalg.norm((T_wt[i] @ T_tc @ T_cb[i])[:3, 3] - T_wb[:3, 3]))
        for i in range(len(T_wt))
    ]
    rms_m = float(np.sqrt(np.mean(np.square(pos_errs))))
    max_m = float(np.max(pos_errs))
    mount_pose7 = mat_to_pose7(T_tc)
    return {
        "T_tracker_cam": mount_pose7.tolist(),
        "T_vive_board": mat_to_pose7(T_wb).tolist(),
        "rms_error_m": rms_m,
        "max_error_m": max_m,
        "num_views": len(T_wt),
        "passed": rms_m <= max_rms_m,
    }


def solve_session_table_from_vive(
    T_vive_tracker_seq: np.ndarray,
    T_cam_table_seq: np.ndarray,
    T_tracker_cam_pose7: np.ndarray,
    *,
    spatial_payload_for_hash: dict[str, Any],
    max_rms_m: float = 0.008,
) -> dict[str, Any]:
    """Compute tabletop `T_table_vive` from multi-view ChArUco observations and bind `spatial_sha256`.

    Averages `T_vive_table[i] = T_vive_tracker[i] @ T_tracker_cam @ T_cam_table[i]` directly in the
    table-local pose neighborhood and inverts the averaged transform once to avoid lever-arm noise
    amplification when the Vive origin is far from the table.
    """
    trk = np.asarray(T_vive_tracker_seq, dtype=np.float64)
    cam = np.asarray(T_cam_table_seq, dtype=np.float64)
    T_tc = pose7_to_mat(T_tracker_cam_pose7)
    samples_vt: list[np.ndarray] = []
    for i in range(len(trk)):
        T_vt = pose7_to_mat(trk[i]) @ T_tc @ pose7_to_mat(cam[i])
        samples_vt.append(mat_to_pose7(T_vt))

    mean_vt, rms_m, max_m = average_se3_poses7(np.stack(samples_vt, axis=0))
    mean_tv = mat_to_pose7(np.linalg.inv(pose7_to_mat(mean_vt)))
    spatial_sha = sha256_object(spatial_payload_for_hash)
    passed = rms_m <= max_rms_m
    return {
        "schema_version": 1,
        "kind": "pikaumi_session_table_calibration",
        "verified": bool(passed),
        "spatial_sha256": spatial_sha,
        "T_table_vive": {
            "position": mean_tv[:3].tolist(),
            "quaternion": mean_tv[3:7].tolist(),
        },
        "metrics": {
            "num_observations": len(samples_vt),
            "rms_error_m": rms_m,
            "max_error_m": max_m,
            "gate_max_rms_m": max_rms_m,
        },
    }
