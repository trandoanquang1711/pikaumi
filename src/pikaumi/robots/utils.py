"""Exact SE(3) and SO(3) Lie-group utilities using `pose7 = [x, y, z, qx, qy, qz, qw]`."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

IDENTITY_POSE7 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).copy()
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    valid = norm > 1e-12
    safe = np.where(valid, norm, 1.0)
    q = q / safe
    ident = np.zeros_like(q)
    ident[..., -1] = 1.0
    return np.where(valid, q, ident)


def pose7_to_mat(pose7: np.ndarray) -> np.ndarray:
    p = np.asarray(pose7, dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = p[:3]
    T[:3, :3] = Rotation.from_quat(normalize_quat_xyzw(p[3:7])).as_matrix()
    return T


def mat_to_pose7(mat: np.ndarray) -> np.ndarray:
    T = np.asarray(mat, dtype=np.float64)
    quat = Rotation.from_matrix(T[:3, :3]).as_quat()
    return np.concatenate([T[:3, 3], normalize_quat_xyzw(quat)]).astype(np.float64)


def rpy_xyz_to_pose7(xyz: list[float] | np.ndarray, rpy: list[float] | np.ndarray) -> np.ndarray:
    quat = Rotation.from_euler("xyz", np.asarray(rpy, dtype=np.float64)).as_quat()
    return np.concatenate([np.asarray(xyz, dtype=np.float64), normalize_quat_xyzw(quat)])


def pose_inv(pose7: np.ndarray) -> np.ndarray:
    return mat_to_pose7(np.linalg.inv(pose7_to_mat(pose7)))


def pose_mul(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    return mat_to_pose7(pose7_to_mat(pose_a) @ pose7_to_mat(pose_b))


def pose_between(pose_from: np.ndarray, pose_to: np.ndarray) -> np.ndarray:
    return mat_to_pose7(np.linalg.inv(pose7_to_mat(pose_from)) @ pose7_to_mat(pose_to))


def skew3(vec: np.ndarray) -> np.ndarray:
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def so3_left_jacobian_inv(phi: np.ndarray) -> np.ndarray:
    """Exact inverse left Jacobian J_l^{-1}(phi) of SO(3).

    Satisfies d/d(delta_omega_world) log(exp([delta_omega]_x) R_cur R_tgt^T) = J_l^{-1}(phi).
    """
    theta = float(np.linalg.norm(phi))
    if theta < 1e-7:
        return np.eye(3, dtype=np.float64) - 0.5 * skew3(phi)
    half = 0.5 * theta
    cot_half = half / np.tan(half)
    S = skew3(phi)
    coeff = (1.0 - cot_half) / (theta * theta)
    return np.eye(3, dtype=np.float64) - 0.5 * S + coeff * (S @ S)


def fast_so3_log(R: np.ndarray) -> np.ndarray:
    """Closed-form SO(3) logarithm `log(R) in R^3` matching `Rotation.from_matrix(R).as_rotvec()`."""
    cos_theta = float(np.clip(0.5 * (R[0, 0] + R[1, 1] + R[2, 2] - 1.0), -1.0, 1.0))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    if cos_theta > 0.999999:
        return 0.5 * vee
    if cos_theta < -0.9995:
        return Rotation.from_matrix(R).as_rotvec()
    theta = float(np.arccos(cos_theta))
    return (0.5 * theta / float(np.sin(theta))) * vee


def se3_world_residual(T_cur: np.ndarray, T_tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return `(pos_err, rot_err_world)` in R^3 x R^3.

    `pos_err = p_cur - p_tgt`, `rot_err_world = log(R_cur @ R_tgt.T)` so a small
    world angular perturbation `exp([dw]_x) R_cur` changes `rot_err_world` by `J_l^{-1}(phi) @ dw`.
    """
    pos_err = T_cur[:3, 3] - T_tgt[:3, 3]
    R_err = T_cur[:3, :3] @ T_tgt[:3, :3].T
    rot_err = fast_so3_log(R_err)
    return pos_err, rot_err


def pose_errors_pos_deg(poses_cur: np.ndarray, poses_tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-frame Euclidean position error (m) and geodesic rotation error (deg)."""
    cur = np.asarray(poses_cur, dtype=np.float64)
    tgt = np.asarray(poses_tgt, dtype=np.float64)
    pos_err = np.linalg.norm(cur[..., :3] - tgt[..., :3], axis=-1)
    q_cur = normalize_quat_xyzw(cur[..., 3:7])
    q_tgt = normalize_quat_xyzw(tgt[..., 3:7])
    dots = np.clip(np.abs(np.sum(q_cur * q_tgt, axis=-1)), 0.0, 1.0)
    rot_deg = np.degrees(2.0 * np.arccos(dots))
    return pos_err, rot_deg


def slerp_quat_xyzw(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    qa = normalize_quat_xyzw(q0)
    qb = normalize_quat_xyzw(q1)
    if float(np.dot(qa, qb)) < 0.0:
        qb = -qb
    rots = Rotation.from_quat(np.stack([qa, qb], axis=0))
    from scipy.spatial.transform import Slerp

    interp = Slerp([0.0, 1.0], rots)([float(np.clip(alpha, 0.0, 1.0))])[0]
    return normalize_quat_xyzw(interp.as_quat())
