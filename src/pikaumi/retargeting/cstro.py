"""`C-STRO`: Contact- & Coupling-Aware Spatio-Temporal Retargeting Optimization.

Two-stage non-causal retargeting engine designed to overcome myopic frame-by-frame IK failure modes:
  Stage 1 — Multi-Branch Kinematic Graph Viterbi DP Seeding:
      Generates diverse posture/branch candidates at keyframes via fast damped Gauss-Newton steps and
      finds the minimum-cost homotopy path via dynamic programming, eliminating base-yaw (>60 deg)
      branch traps and wrist lockups.
  Stage 2 — Global Zero-Phase Block-Sparse SE(3) Trajectory Optimization:
      Jointly optimizes `Q[0:T]` with exact vectorized `scipy.sparse.csr_matrix` Jacobians combining:
        (a) Contact/aperture-modulated SE(3) pose precision weights,
        (b) Bimanual relative SE(3) co-grasp coupling (`^L T_R(t)` preservation),
        (c) Zero-phase centered acceleration smoothing + URDF velocity-limit hinge barrier,
        (d) Analytical link-capsule inter-arm & tabletop clearance repulsion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.transform import Rotation

from pikaumi.robots.collision import (
    distal_capsules_by_links,
    segment_segment_distance,
    world_capsules_for_config,
    world_shared_capsules_for_config,
)
from pikaumi.robots.kinematics import BimanualIKSolver, posture_seed
from pikaumi.robots.registry import RobotProfile
from pikaumi.robots.utils import (
    fast_so3_log,
    pose7_to_mat,
    se3_world_residual,
    skew3,
    so3_left_jacobian_inv,
)


@dataclass(frozen=True)
class CStroConfig:
    # Ablation switches
    use_viterbi_seed: bool = True
    use_bimanual_coupling: bool = True
    use_contact_weights: bool = True
    use_zero_phase_smoothing: bool = True
    use_velocity_barrier: bool = True
    use_capsule_clearance: bool = True
    use_table_clearance: bool = True

    # Stage 1 Viterbi parameters
    keyframe_stride: int = 10
    num_branch_perturbations: int = 4

    # Stage 2 residual weights
    pos_weight: float = 130.0
    ori_weight: float = 7.0
    bimanual_pos_weight: float = 85.0
    bimanual_ori_weight: float = 5.0
    posture_weight: float = 1.5
    limit_margin_weight: float = 4.0
    vel_reg_weight: float = 0.8
    acc_reg_weight: float = 0.018
    vel_barrier_weight: float = 8.0
    vel_limit_fraction: float = 0.90
    capsule_clearance_margin_m: float = 0.025
    capsule_barrier_weight: float = 60.0
    table_clearance_margin_m: float = 0.015
    table_barrier_weight: float = 80.0
    max_nfev: int = 25


def compute_contact_precision_multipliers(
    ee_targets_by_side: dict[str, np.ndarray],
    widths_by_side: dict[str, np.ndarray],
    *,
    fps: float = 30.0,
    table_z_robot: float | None = None,
    enabled: bool = True,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute per-arm contact precision multipliers `alpha[side][t]` and bimanual co-grasp weight `beta[t]`."""
    sides = tuple(ee_targets_by_side.keys())
    T = len(ee_targets_by_side[sides[0]])
    if not enabled:
        ones = {s: np.ones(T, dtype=np.float64) for s in sides}
        return ones, np.ones(T, dtype=np.float64)

    alpha: dict[str, np.ndarray] = {}
    close_scores: dict[str, np.ndarray] = {}
    for s in sides:
        w = np.asarray(widths_by_side[s], dtype=np.float64)
        dw = np.abs(np.gradient(w) * float(fps)) if T >= 2 else np.zeros(T, dtype=np.float64)
        c_score = 1.0 / (1.0 + np.exp((w - 0.035) / 0.008))
        close_scores[s] = c_score
        trans_score = np.clip(dw / 0.08, 0.0, 1.0)
        if table_z_robot is not None:
            z_rel = ee_targets_by_side[s][:, 2] - float(table_z_robot)
            table_score = 1.0 / (1.0 + np.exp((z_rel - 0.055) / 0.015))
        else:
            table_score = np.zeros(T, dtype=np.float64)

        alpha[s] = 1.0 + 0.55 * c_score + 0.35 * trans_score + 0.30 * table_score

    if "left" in close_scores and "right" in close_scores:
        co_grasp = close_scores["left"] * close_scores["right"]
        beta = 0.35 + 0.85 * co_grasp
    else:
        beta = np.zeros(T, dtype=np.float64)
    return alpha, beta


def _point_spatial_jacobian_on_link(
    model,
    joint_frames: dict[str, tuple[np.ndarray, np.ndarray]],
    link_name: str,
    point_world: np.ndarray,
    active_indices: tuple[int, ...],
) -> np.ndarray:
    """Exact `(3, n_active)` linear velocity Jacobian of a world point rigidly attached to `link_name`."""
    col_map = {act_idx: col_i for col_i, act_idx in enumerate(active_indices)}
    J_pt = np.zeros((3, len(active_indices)), dtype=np.float64)
    for j in model.ancestors[link_name]:
        if j.joint_type == "fixed":
            continue
        mult = 1.0
        act_idx = j.actuated_index
        if act_idx is None and j.mimic_joint is not None:
            ref = model.joint_by_name[j.mimic_joint]
            act_idx = ref.actuated_index
            mult = j.mimic_multiplier
        if act_idx is None or act_idx not in col_map:
            continue
        col = col_map[act_idx]
        origin_w, axis_w = joint_frames[j.name]
        if j.joint_type in ("revolute", "continuous"):
            J_pt[:, col] += mult * np.cross(axis_w, point_world - origin_w)
        elif j.joint_type == "prismatic":
            J_pt[:, col] += mult * axis_w
    return J_pt


def _fast_gauss_newton_frame(
    solver: BimanualIKSolver,
    q_nom: np.ndarray,
    x_init: np.ndarray,
    T_tgts: dict[str, np.ndarray],
    x_anchor: np.ndarray,
    *,
    iters: int = 8,
    w_anchor: float = 2.0,
) -> np.ndarray:
    """Zero-overhead projected Gauss-Newton single-frame IK solve in pure NumPy."""
    sides = tuple(T_tgts.keys())
    D = solver.n_active
    x = np.clip(x_init.copy(), solver.lower, solver.upper)
    post_mask_f = solver.posture_mask.astype(np.float64)
    eye_D = np.eye(D, dtype=np.float64)

    for _ in range(iters):
        q_cur = solver.embed_active(q_nom, x)
        fk_pair = solver.model.forward_kinematics_all(q_cur)
        r_list: list[np.ndarray] = []
        J_list: list[np.ndarray] = []
        pose_err_sq = 0.0
        for s in sides:
            ee = solver.profile.arms[s].ee_link
            T_cur, J_pos, J_rot = solver.model.link_pose_and_spatial_jacobian(
                q_cur, ee, solver.active_indices, precomputed_fk=fk_pair
            )
            pe, re = se3_world_residual(T_cur, T_tgts[s])
            pose_err_sq += float(np.dot(pe, pe)) + 0.01 * float(np.dot(re, re))
            r_list.append(125.0 * pe)
            r_list.append(6.0 * re)
            J_list.append(125.0 * J_pos)
            J_list.append(6.0 * (so3_left_jacobian_inv(re) @ J_rot))

        r_list.append(w_anchor * (x - x_anchor))
        J_list.append(w_anchor * eye_D)
        r_list.append(1.5 * post_mask_f * (x - solver.posture_vals))
        J_list.append(np.diag(1.5 * post_mask_f))

        r = np.concatenate(r_list)
        J = np.vstack(J_list)
        H = J.T @ J + 1.5 * eye_D
        dx = -np.linalg.solve(H, J.T @ r)
        dx = np.clip(dx, -0.35, 0.35)
        x = np.clip(x + dx, solver.lower, solver.upper)
        if pose_err_sq < 1e-8 and float(np.max(np.abs(dx))) < 1e-4:
            break
    return x


def _viterbi_multi_branch_seed(
    solver: BimanualIKSolver,
    ee_targets_by_side: dict[str, np.ndarray],
    *,
    fps: float,
    cfg: CStroConfig,
) -> np.ndarray:
    """Stage 1: Multi-branch Viterbi DP over keyframe candidate IK branches, followed by warm-started GN refinement."""
    profile = solver.profile
    sides = profile.active_sides
    T = len(ee_targets_by_side[sides[0]])
    q_nom = posture_seed(profile)
    x_nom = solver.extract_active(q_nom)

    T_all = {
        s: [pose7_to_mat(ee_targets_by_side[s][t]) for t in range(T)]
        for s in sides
    }

    # Always compute the causal warm-started rollout first; in Stage 1 Viterbi DP it serves as
    # Candidate k=0 (guaranteeing monotonic optimality: Viterbi never degrades a clean sequential rollout
    # while escaping branch traps whenever an explorer branch k>=1 achieves lower path cost).
    x_seq = np.zeros((T, solver.n_active), dtype=np.float64)
    x_cur = _fast_gauss_newton_frame(
        solver, q_nom, x_nom, {s: T_all[s][0] for s in sides}, x_nom, iters=12, w_anchor=1.0
    )
    x_seq[0] = x_cur
    for t in range(1, T):
        x_cur = _fast_gauss_newton_frame(
            solver, q_nom, x_cur, {s: T_all[s][t] for s in sides}, x_cur, iters=6, w_anchor=4.0
        )
        x_seq[t] = x_cur

    if T == 1 or not cfg.use_viterbi_seed:
        return x_seq

    stride = max(1, min(int(cfg.keyframe_stride), max(1, T // 2)))
    keyframes = list(range(0, T, stride))
    if keyframes[-1] != T - 1:
        keyframes.append(T - 1)
    M = len(keyframes)

    col_map = {act_idx: c for c, act_idx in enumerate(solver.active_indices)}
    base_cols = [
        col_map[profile.arms[s].arm_actuated_indices[0]]
        for s in sides
        if profile.arms[s].arm_actuated_indices[0] in col_map
    ]

    K = max(2, int(cfg.num_branch_perturbations))
    rng = np.random.default_rng(2026)
    branch_offsets = [np.zeros(solver.n_active, dtype=np.float64)]
    for k in range(1, K):
        if k == 1:
            off = np.full(solver.n_active, 0.35, dtype=np.float64)
        elif k == 2:
            off = np.full(solver.n_active, -0.35, dtype=np.float64)
        else:
            off = rng.uniform(-0.45, 0.45, size=solver.n_active)
        branch_offsets.append(off)

    cand_x = np.zeros((M, K, solver.n_active), dtype=np.float64)
    node_cost = np.zeros((M, K), dtype=np.float64)

    for m, t_kf in enumerate(keyframes):
        T_tgts = {s: T_all[s][t_kf] for s in sides}
        for k in range(K):
            if k == 0:
                xk = x_seq[t_kf].copy()
            else:
                x_seed = np.clip(x_nom + branch_offsets[k], solver.lower + 0.02, solver.upper - 0.02)
                xk = _fast_gauss_newton_frame(
                    solver, q_nom, x_seed, T_tgts, x_nom, iters=10, w_anchor=0.8
                )
            cand_x[m, k] = xk

            c_val = 0.0
            q_sol = solver.embed_active(q_nom, xk)
            link_tf, _ = profile.urdf_model.forward_kinematics_all(q_sol)
            for s in sides:
                pe, re = se3_world_residual(link_tf[profile.arms[s].ee_link], T_tgts[s])
                c_val += 2500.0 * float(np.dot(pe, pe)) + 15.0 * float(np.dot(re, re))
            if np.any(solver.posture_mask):
                diff_p = (xk - solver.posture_vals) * solver.posture_mask
                c_val += 0.5 * float(np.dot(diff_p, diff_p))
            for bc in base_cols:
                yaw_exc = abs(float(xk[bc] - x_nom[bc]))
                if yaw_exc > np.deg2rad(50.0):
                    c_val += 40.0 * (yaw_exc - np.deg2rad(50.0)) ** 2
            node_cost[m, k] = c_val

    dp = np.full((M, K), np.inf, dtype=np.float64)
    parent = np.zeros((M, K), dtype=np.int64)
    # Small hysteresis preference (0.15) toward k=0 so explorer branches are chosen only when
    # they genuinely reduce tracking/swing/velocity cost over the causal warm-start rollout.
    node_cost[:, 1:] += 0.15
    dp[0] = node_cost[0]

    for m in range(1, M):
        dt_m = max((keyframes[m] - keyframes[m - 1]) / float(fps), 1.0 / float(fps))
        for k in range(K):
            for j in range(K):
                dq = cand_x[m, k] - cand_x[m - 1, j]
                vel = np.abs(dq) / dt_m
                vel_excess = np.maximum(0.0, vel - solver.vel_limits)
                yaw_jump_pen = sum(12.0 * float(dq[bc] ** 2) for bc in base_cols)
                edge_c = 1.5 * float(np.dot(dq, dq)) / dt_m + 30.0 * float(np.dot(vel_excess, vel_excess)) + yaw_jump_pen
                cand_total = dp[m - 1, j] + edge_c + node_cost[m, k]
                if cand_total < dp[m, k]:
                    dp[m, k] = cand_total
                    parent[m, k] = j

    best_path = [0] * M
    best_path[M - 1] = int(np.argmin(dp[M - 1]))
    for m in range(M - 1, 0, -1):
        best_path[m - 1] = int(parent[m, best_path[m]])

    if all(b == 0 for b in best_path):
        return x_seq

    x_interp = np.zeros((T, solver.n_active), dtype=np.float64)
    for m in range(M - 1):
        t0, t1 = keyframes[m], keyframes[m + 1]
        if best_path[m] == 0 and best_path[m + 1] == 0:
            x_interp[t0 : t1 + 1] = x_seq[t0 : t1 + 1]
        else:
            x0_kf = cand_x[m, best_path[m]]
            x1_kf = cand_x[m + 1, best_path[m + 1]]
            for t in range(t0, t1 + 1):
                u = (t - t0) / max(t1 - t0, 1)
                x_interp[t] = (1.0 - u) * x0_kf + u * x1_kf

    x_seeded = np.zeros_like(x_interp)
    for t in range(T):
        x_prev = x_seeded[t - 1] if t > 0 else x_interp[0]
        x_seeded[t] = _fast_gauss_newton_frame(
            solver, q_nom, x_interp[t], {s: T_all[s][t] for s in sides}, x_prev, iters=4, w_anchor=2.5
        )
    return x_seeded


def build_cstro_problem(
    profile: RobotProfile,
    ee_targets_by_side: dict[str, np.ndarray],
    pika_widths_by_side: dict[str, np.ndarray],
    *,
    fps: float = 30.0,
    table_z_robot: float | None = None,
    config: CStroConfig | None = None,
):
    """Construct the Stage 2 global residual and vectorized sparse CSR Jacobian `(fun, jac, x0, lb, ub)`."""
    cfg = config or CStroConfig()
    solver = BimanualIKSolver(profile)
    sides = profile.active_sides
    is_bimanual = ("left" in sides and "right" in sides) and cfg.use_bimanual_coupling
    has_caps = (len(sides) == 2) and cfg.use_capsule_clearance
    has_table = (table_z_robot is not None) and cfg.use_table_clearance
    T = len(ee_targets_by_side[sides[0]])
    D = solver.n_active
    dt = 1.0 / float(fps)
    q_nom = posture_seed(profile)

    x_init_2d = _viterbi_multi_branch_seed(solver, ee_targets_by_side, fps=fps, cfg=cfg)
    x0 = np.clip(x_init_2d.ravel(), np.tile(solver.lower, T) + 1e-7, np.tile(solver.upper, T) - 1e-7)

    alpha, beta = compute_contact_precision_multipliers(
        ee_targets_by_side,
        pika_widths_by_side,
        fps=fps,
        table_z_robot=table_z_robot,
        enabled=cfg.use_contact_weights,
    )

    T_tgts = {
        s: [pose7_to_mat(ee_targets_by_side[s][t]) for t in range(T)]
        for s in sides
    }
    p_rel_tgt = np.zeros((T, 3), dtype=np.float64)
    R_rel_tgt = [np.eye(3, dtype=np.float64) for _ in range(T)]
    if is_bimanual:
        for t in range(T):
            TL = T_tgts["left"][t]
            TR = T_tgts["right"][t]
            p_rel_tgt[t] = TL[:3, :3].T @ (TR[:3, 3] - TL[:3, 3])
            R_rel_tgt[t] = TL[:3, :3].T @ TR[:3, :3]

    dense_rows_per_frame = (
        6 * len(sides)
        + (6 if is_bimanual else 0)
        + (1 if has_caps else 0)
        + (1 if has_table else 0)
    )
    unary_rows_per_frame = dense_rows_per_frame + 2 * D
    n_unary = T * unary_rows_per_frame
    n_vel = max(0, T - 1) * D
    n_vel_bar = max(0, T - 1) * D if cfg.use_velocity_barrier else 0
    n_acc = max(0, T - 2) * D if cfg.use_zero_phase_smoothing else 0
    total_rows = n_unary + n_vel + n_vel_bar + n_acc

    vel_thresh = cfg.vel_limit_fraction * solver.vel_limits
    post_mask_f = solver.posture_mask.astype(np.float64)
    usable_f = solver.margin_usable.astype(np.float64)

    # Precompute static COO index templates for fast vectorized CSR assembly!
    # 1) Per-frame dense block (T, dense_rows_per_frame, D)
    t_idx = np.arange(T, dtype=np.int32)
    r_dense = (
        t_idx[:, None, None] * unary_rows_per_frame
        + np.arange(dense_rows_per_frame, dtype=np.int32)[None, :, None]
        + np.zeros((1, 1, D), dtype=np.int32)
    ).ravel()
    c_dense = (
        t_idx[:, None, None] * D
        + np.zeros((1, dense_rows_per_frame, 1), dtype=np.int32)
        + np.arange(D, dtype=np.int32)[None, None, :]
    ).ravel()

    # 2) Per-frame diagonal posture + limit_margin blocks (T, 2*D)
    d_idx = np.arange(D, dtype=np.int32)
    r_post = (t_idx[:, None] * unary_rows_per_frame + dense_rows_per_frame + d_idx[None, :]).ravel()
    r_marg = (t_idx[:, None] * unary_rows_per_frame + dense_rows_per_frame + D + d_idx[None, :]).ravel()
    c_diag = (t_idx[:, None] * D + d_idx[None, :]).ravel()
    v_post_const = np.tile(cfg.posture_weight * post_mask_f, T)

    # 3) Temporal velocity & acceleration constant sparse blocks
    w_v = cfg.vel_reg_weight if cfg.use_zero_phase_smoothing else 3.5
    if T >= 2:
        tv_idx = np.arange(T - 1, dtype=np.int32)
        r_vel = (n_unary + tv_idx[:, None] * D + d_idx[None, :]).ravel()
        c_vel0 = (tv_idx[:, None] * D + d_idx[None, :]).ravel()
        c_vel1 = ((tv_idx[:, None] + 1) * D + d_idx[None, :]).ravel()
    else:
        r_vel = c_vel0 = c_vel1 = np.zeros(0, dtype=np.int32)

    if cfg.use_velocity_barrier and T >= 2:
        r_vbar = (n_unary + n_vel + tv_idx[:, None] * D + d_idx[None, :]).ravel()
    else:
        r_vbar = np.zeros(0, dtype=np.int32)

    w_a = cfg.acc_reg_weight / (dt * dt)
    if cfg.use_zero_phase_smoothing and T >= 3:
        ta_idx = np.arange(T - 2, dtype=np.int32)
        base_acc = n_unary + n_vel + n_vel_bar
        r_acc = (base_acc + ta_idx[:, None] * D + d_idx[None, :]).ravel()
        c_accm = (ta_idx[:, None] * D + d_idx[None, :]).ravel()
        c_acc0 = ((ta_idx[:, None] + 1) * D + d_idx[None, :]).ravel()
        c_accp = ((ta_idx[:, None] + 2) * D + d_idx[None, :]).ravel()
    else:
        r_acc = c_accm = c_acc0 = c_accp = np.zeros(0, dtype=np.int32)

    def _fallback_sep_normal(p0_a: np.ndarray, p1_a: np.ndarray, p0_b: np.ndarray, p1_b: np.ndarray) -> np.ndarray:
        cand = np.cross(p1_a - p0_a, p1_b - p0_b)
        n = float(np.linalg.norm(cand))
        if n > 1e-8:
            return cand / n
        mid_diff = 0.5 * (p0_a + p1_a) - 0.5 * (p0_b + p1_b)
        n = float(np.linalg.norm(mid_diff))
        if n > 1e-8:
            return mid_diff / n
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)

    def eval_res_and_jac(x_flat: np.ndarray) -> tuple[np.ndarray, csr_matrix]:
        X = x_flat.reshape(T, D)
        res = np.zeros(total_rows, dtype=np.float64)
        J_dense_vals = np.zeros((T, dense_rows_per_frame, D), dtype=np.float64)

        for t in range(T):
            xt = X[t]
            q_cur = solver.embed_active(q_nom, xt)
            fk_pair = profile.urdf_model.forward_kinematics_all(q_cur)
            link_tf, joint_frames = fk_pair

            r_base = t * unary_rows_per_frame
            dr = 0
            per_side_kin: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for s in sides:
                ee = profile.arms[s].ee_link
                T_cur, J_pos, J_rot = profile.urdf_model.link_pose_and_spatial_jacobian(
                    q_cur, ee, solver.active_indices, precomputed_fk=fk_pair
                )
                per_side_kin[s] = (T_cur, J_pos, J_rot)
                pe, re = se3_world_residual(T_cur, T_tgts[s][t])
                Jl_inv = so3_left_jacobian_inv(re)
                wp = cfg.pos_weight * alpha[s][t]
                wo = cfg.ori_weight * alpha[s][t]
                res[r_base + dr : r_base + dr + 3] = wp * pe
                res[r_base + dr + 3 : r_base + dr + 6] = wo * re
                J_dense_vals[t, dr : dr + 3, :] = wp * J_pos
                J_dense_vals[t, dr + 3 : dr + 6, :] = wo * (Jl_inv @ J_rot)
                dr += 6

            if is_bimanual:
                TL_cur, Jp_L, Jr_L = per_side_kin["left"]
                TR_cur, Jp_R, Jr_R = per_side_kin["right"]
                RL = TL_cur[:3, :3]
                pL = TL_cur[:3, 3]
                RR = TR_cur[:3, :3]
                pR = TR_cur[:3, 3]

                p_rel_cur = RL.T @ (pR - pL)
                e_prel = p_rel_cur - p_rel_tgt[t]
                J_prel = RL.T @ (Jp_R - Jp_L) + (RL.T @ skew3(pR - pL)) @ Jr_L

                R_rel_cur = RL.T @ RR
                e_rrel = fast_so3_log(R_rel_cur @ R_rel_tgt[t].T)
                J_rrel = so3_left_jacobian_inv(e_rrel) @ RL.T @ (Jr_R - Jr_L)

                wb_p = cfg.bimanual_pos_weight * beta[t]
                wb_o = cfg.bimanual_ori_weight * beta[t]
                res[r_base + dr : r_base + dr + 3] = wb_p * e_prel
                res[r_base + dr + 3 : r_base + dr + 6] = wb_o * e_rrel
                J_dense_vals[t, dr : dr + 3, :] = wb_p * J_prel
                J_dense_vals[t, dr + 3 : dr + 6, :] = wb_o * J_rrel
                dr += 6

            left_caps = right_caps = None
            left_distal = right_distal = None
            if has_caps or has_table:
                left_caps, right_caps = world_capsules_for_config(profile.urdf_model, q_cur, link_tf=link_tf)
                left_distal = distal_capsules_by_links(left_caps, n_distinct_links=3)
                right_distal = distal_capsules_by_links(right_caps, n_distinct_links=3)

            if has_caps and left_caps is not None and right_caps is not None:
                # Smooth L2-aggregated multi-pair capsule barrier across all active penetrating pairs
                # (including distal arms vs movable torso/shared links) with non-zero zero-dist fallback.
                margin_c = cfg.capsule_clearance_margin_m
                sum_p2 = 0.0
                sum_p_grad = np.zeros(D, dtype=np.float64)
                shared_caps = world_shared_capsules_for_config(profile.urdf_model, q_cur, link_tf=link_tf)

                pair_groups: list[tuple[list, list]] = [(left_caps, right_caps)]
                if shared_caps and left_distal is not None and right_distal is not None:
                    pair_groups.append((left_distal + right_distal, shared_caps))

                for group_a, group_b in pair_groups:
                    for cap_a, p0_a, p1_a in group_a:
                        mid_a = 0.5 * (p0_a + p1_a)
                        rad_bound_a = 0.5 * float(np.linalg.norm(p1_a - p0_a)) + cap_a.radius + margin_c
                        for cap_b, p0_b, p1_b in group_b:
                            mid_b = 0.5 * (p0_b + p1_b)
                            dx_m = mid_a - mid_b
                            max_r = rad_bound_a + 0.5 * float(np.linalg.norm(p1_b - p0_b)) + cap_b.radius
                            if float(dx_m[0] * dx_m[0] + dx_m[1] * dx_m[1] + dx_m[2] * dx_m[2]) > max_r * max_r:
                                continue
                            dist, c_a, c_b = segment_segment_distance(p0_a, p1_a, p0_b, p1_b)
                            clear = dist - (cap_a.radius + cap_b.radius)
                            if clear < margin_c:
                                pen = margin_c - clear
                                n_hat = (c_a - c_b) / dist if dist > 1e-8 else _fallback_sep_normal(p0_a, p1_a, p0_b, p1_b)
                                J_ca = _point_spatial_jacobian_on_link(
                                    profile.urdf_model, joint_frames, cap_a.link_name, c_a, solver.active_indices
                                )
                                J_cb = _point_spatial_jacobian_on_link(
                                    profile.urdf_model, joint_frames, cap_b.link_name, c_b, solver.active_indices
                                )
                                grad_pen = -(n_hat @ (J_ca - J_cb))
                                sum_p2 += pen * pen
                                sum_p_grad += pen * grad_pen
                if sum_p2 > 0.0:
                    norm_p = float(np.sqrt(sum_p2))
                    res[r_base + dr] = cfg.capsule_barrier_weight * norm_p
                    J_dense_vals[t, dr, :] = cfg.capsule_barrier_weight * (sum_p_grad / norm_p)
                dr += 1

            if has_table and table_z_robot is not None:
                # Smooth L2-aggregated tabletop clearance barrier on EE links and distal wrist/forearm/gripper capsules.
                margin_t = cfg.table_clearance_margin_m
                tz = float(table_z_robot)
                sum_t2 = 0.0
                sum_t_grad = np.zeros(D, dtype=np.float64)
                for s in sides:
                    T_cur, J_pos, _ = per_side_kin[s]
                    clear = float(T_cur[2, 3]) - tz
                    if clear < margin_t:
                        pen = margin_t - clear
                        sum_t2 += pen * pen
                        sum_t_grad += pen * (-J_pos[2, :])
                if left_distal is not None and right_distal is not None:
                    for cap, p0_w, p1_w in (left_distal + right_distal):
                        p_low = p0_w if p0_w[2] <= p1_w[2] else p1_w
                        clear = float(p_low[2]) - cap.radius - tz
                        if clear < margin_t:
                            pen = margin_t - clear
                            J_pt = _point_spatial_jacobian_on_link(
                                profile.urdf_model, joint_frames, cap.link_name, p_low, solver.active_indices
                            )
                            sum_t2 += pen * pen
                            sum_t_grad += pen * (-J_pt[2, :])
                if sum_t2 > 0.0:
                    norm_t = float(np.sqrt(sum_t2))
                    res[r_base + dr] = cfg.table_barrier_weight * norm_t
                    J_dense_vals[t, dr, :] = cfg.table_barrier_weight * (sum_t_grad / norm_t)
                dr += 1

            # Posture & limit_margin residuals
            res[r_base + dr : r_base + dr + D] = cfg.posture_weight * post_mask_f * (xt - solver.posture_vals)
            low_v = np.maximum(0.0, solver.margin_lower - xt)
            high_v = np.maximum(0.0, xt - solver.margin_upper)
            res[r_base + dr + D : r_base + dr + 2 * D] = cfg.limit_margin_weight * usable_f * (high_v - low_v)

        # Vectorized diagonal margin derivatives
        active_marg = usable_f[None, :] * (
            (X > solver.margin_upper[None, :]).astype(np.float64)
            + (X < solver.margin_lower[None, :]).astype(np.float64)
        )
        v_marg = (cfg.limit_margin_weight * active_marg).ravel()

        rows_parts = [r_dense, r_post, r_marg]
        cols_parts = [c_dense, c_diag, c_diag]
        vals_parts = [J_dense_vals.ravel(), v_post_const, v_marg]

        if T >= 2:
            dX = X[1:] - X[:-1]
            res[n_unary : n_unary + n_vel] = (w_v * dX).ravel()
            n_e = r_vel.size
            rows_parts.extend([r_vel, r_vel])
            cols_parts.extend([c_vel0, c_vel1])
            vals_parts.extend([np.full(n_e, -w_v), np.full(n_e, w_v)])

            if cfg.use_velocity_barrier:
                inv_dt = 1.0 / dt
                V = dX * inv_dt
                abs_V = np.abs(V)
                excess = np.maximum(0.0, abs_V - vel_thresh[None, :])
                res[n_unary + n_vel : n_unary + n_vel + n_vel_bar] = (
                    cfg.vel_barrier_weight * excess * np.sign(V)
                ).ravel()
                g_bar = (
                    cfg.vel_barrier_weight * inv_dt * (abs_V > vel_thresh[None, :]).astype(np.float64)
                ).ravel()
                rows_parts.extend([r_vbar, r_vbar])
                cols_parts.extend([c_vel0, c_vel1])
                vals_parts.extend([-g_bar, g_bar])

        if cfg.use_zero_phase_smoothing and T >= 3:
            acc_X = X[2:] - 2.0 * X[1:-1] + X[:-2]
            base_acc = n_unary + n_vel + n_vel_bar
            res[base_acc : base_acc + n_acc] = (w_a * acc_X).ravel()
            n_a = r_acc.size
            rows_parts.extend([r_acc, r_acc, r_acc])
            cols_parts.extend([c_accm, c_acc0, c_accp])
            vals_parts.extend([np.full(n_a, w_a), np.full(n_a, -2.0 * w_a), np.full(n_a, w_a)])

        all_r = np.concatenate(rows_parts)
        all_c = np.concatenate(cols_parts)
        all_v = np.concatenate(vals_parts)
        J_csr = coo_matrix((all_v, (all_r, all_c)), shape=(total_rows, T * D)).tocsr()
        return res, J_csr

    cache: dict[str, object] = {}

    def fun(x: np.ndarray) -> np.ndarray:
        r, J_csr = eval_res_and_jac(x)
        cache["x"] = x.copy()
        cache["J"] = J_csr
        return r

    def jac(x: np.ndarray) -> csr_matrix:
        last_x = cache.get("x")
        if isinstance(last_x, np.ndarray) and np.array_equal(last_x, x):
            return cache["J"]  # type: ignore[return-value]
        _, J_csr = eval_res_and_jac(x)
        return J_csr

    lb = np.tile(solver.lower, T)
    ub = np.tile(solver.upper, T)
    return fun, jac, x0, lb, ub, solver, q_nom


def solve_episode_cstro(
    profile: RobotProfile,
    ee_targets_by_side: dict[str, np.ndarray],
    pika_widths_by_side: dict[str, np.ndarray],
    *,
    fps: float = 30.0,
    table_z_robot: float | None = None,
    config: CStroConfig | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Solve full-episode retargeting with `C-STRO` (Stage 1 Viterbi + Stage 2 Global Sparse SE(3) Optimization)."""
    cfg = config or CStroConfig()
    sides = profile.active_sides
    T = len(ee_targets_by_side[sides[0]])

    fun, jac, x0, lb, ub, solver, q_nom = build_cstro_problem(
        profile,
        ee_targets_by_side,
        pika_widths_by_side,
        fps=fps,
        table_z_robot=table_z_robot,
        config=cfg,
    )

    tr_maxiter = max(45, min(200, int(1.5 * T)))
    sol = least_squares(
        fun,
        x0,
        jac=jac,
        bounds=(lb, ub),
        method="trf",
        tr_solver="lsmr",
        x_scale="jac",
        tr_options={"maxiter": tr_maxiter},
        max_nfev=int(cfg.max_nfev),
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-4,
    )

    X_opt = sol.x.reshape(T, solver.n_active)
    q_traj = np.zeros((T, profile.urdf_model.num_actuated), dtype=np.float64)
    widths_out = {s: np.zeros(T, dtype=np.float64) for s in sides}
    for t in range(T):
        qt = solver.embed_active(q_nom, X_opt[t])
        for s in sides:
            qt, rw = profile.map_gripper_width_to_q(qt, s, float(pika_widths_by_side[s][t]))
            widths_out[s][t] = rw
        q_traj[t] = qt

    return q_traj, widths_out
