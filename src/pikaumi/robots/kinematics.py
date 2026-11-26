"""Analytical multi-arm URDF IK solvers & baseline trajectory retargeters (`dls_myopic`, `handumi_lm`)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from pikaumi.robots.registry import RobotProfile
from pikaumi.robots.utils import (
    mat_to_pose7,
    pose7_to_mat,
    se3_world_residual,
    so3_left_jacobian_inv,
)


@dataclass(frozen=True)
class IKSolverWeights:
    pos_weight: float = 125.0
    ori_weight: float = 5.0
    rest_weight: float = 4.0
    posture_weight: float = 2.0
    limit_margin_weight: float = 5.0
    limit_margin_rad: float = np.deg2rad(8.0)
    max_joint_delta: float | None = 0.35


def build_replay_weights(profile: RobotProfile) -> IKSolverWeights:
    base = profile.ik_weights
    rep = profile.replay_ik_weights
    return IKSolverWeights(
        pos_weight=float(rep.get("pos", base.get("pos", 125.0))),
        ori_weight=float(rep.get("ori", base.get("ori", 5.0))),
        rest_weight=float(rep.get("rest", base.get("rest", 4.0))),
        posture_weight=float(rep.get("posture", 2.0)),
        limit_margin_weight=float(rep.get("limit_margin", 5.0)),
        max_joint_delta=profile.replay_max_joint_delta,
    )


def posture_seed(profile: RobotProfile) -> np.ndarray:
    return np.where(np.isnan(profile.posture_q), profile.home_q, profile.posture_q).astype(np.float64)


class BimanualIKSolver:
    """Analytical-Jacobian kinematic solver for any 1-arm or 2-arm `RobotProfile`."""

    def __init__(self, profile: RobotProfile, weights: IKSolverWeights | None = None) -> None:
        self.profile = profile
        self.model = profile.urdf_model
        self.weights = weights or build_replay_weights(profile)

        # Active movable arm joint indices across active sides (excluding grippers & locked torso)
        active: list[int] = []
        locked_set = set(profile.locked_actuated_indices)
        for side in profile.active_sides:
            for idx in profile.arms[side].arm_actuated_indices:
                if idx not in locked_set and idx not in active:
                    active.append(idx)
        self.active_indices = tuple(active)
        self.n_active = len(self.active_indices)
        self.lower = self.model.lower_limits[list(self.active_indices)]
        self.upper = self.model.upper_limits[list(self.active_indices)]
        self.vel_limits = self.model.velocity_limits[list(self.active_indices)]

        # Posture preferences on active indices
        post_active = profile.posture_q[list(self.active_indices)]
        self.posture_mask = ~np.isnan(post_active)
        self.posture_vals = np.where(self.posture_mask, post_active, 0.0)

        # Soft limit margin bounds
        margin = float(self.weights.limit_margin_rad)
        usable = (self.upper - self.lower) > 3.0 * margin
        self.margin_usable = usable
        self.margin_lower = np.where(usable, self.lower + margin, self.lower)
        self.margin_upper = np.where(usable, self.upper - margin, self.upper)

    def extract_active(self, q_full: np.ndarray) -> np.ndarray:
        return np.asarray(q_full, dtype=np.float64)[list(self.active_indices)].copy()

    def embed_active(self, q_base: np.ndarray, q_active: np.ndarray) -> np.ndarray:
        out = np.asarray(q_base, dtype=np.float64).copy()
        out[list(self.active_indices)] = np.asarray(q_active, dtype=np.float64)
        return out

    def fk_poses7(self, q_full: np.ndarray) -> dict[str, np.ndarray]:
        link_tf, _ = self.model.forward_kinematics_all(q_full)
        return {
            side: mat_to_pose7(link_tf[self.profile.arms[side].ee_link])
            for side in self.profile.active_sides
        }

    def solve_frame_lm(
        self,
        q_full_seed: np.ndarray,
        targets_pose7: dict[str, np.ndarray],
        *,
        q_prev_full: np.ndarray | None = None,
        max_nfev: int = 30,
        apply_delta_cap: bool = True,
    ) -> np.ndarray:
        """Frame-by-frame Levenberg-Marquardt / TRF solve matching HandUMI's per-frame formulation."""
        w = self.weights
        x0 = np.clip(self.extract_active(q_full_seed), self.lower + 1e-7, self.upper - 1e-7)
        x_prev = self.extract_active(q_prev_full if q_prev_full is not None else q_full_seed)
        T_tgts = {s: pose7_to_mat(targets_pose7[s]) for s in self.profile.active_sides if s in targets_pose7}
        sides = tuple(T_tgts.keys())

        w_pos = float(max(w.pos_weight, 0.0))
        w_ori = float(max(w.ori_weight, 0.0))
        w_rest = float(max(w.rest_weight, 0.0))
        w_post = float(max(w.posture_weight, 0.0))
        w_marg = float(max(w.limit_margin_weight, 0.0))

        n_pose = 6 * len(sides)
        n_rows = n_pose + self.n_active + (self.n_active if w_post > 0 else 0) + (self.n_active if w_marg > 0 else 0)

        def fun_and_jac(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            q_cur = self.embed_active(q_full_seed, x)
            res = np.zeros(n_rows, dtype=np.float64)
            J = np.zeros((n_rows, self.n_active), dtype=np.float64)
            row = 0
            for side in sides:
                ee = self.profile.arms[side].ee_link
                T_cur, J_pos, J_rot = self.model.link_pose_and_spatial_jacobian(
                    q_cur, ee, self.active_indices
                )
                p_err, r_err = se3_world_residual(T_cur, T_tgts[side])
                Jl_inv = so3_left_jacobian_inv(r_err)
                res[row : row + 3] = w_pos * p_err
                res[row + 3 : row + 6] = w_ori * r_err
                J[row : row + 3, :] = w_pos * J_pos
                J[row + 3 : row + 6, :] = w_ori * (Jl_inv @ J_rot)
                row += 6

            res[row : row + self.n_active] = w_rest * (x - x_prev)
            J[row : row + self.n_active, :] = w_rest * np.eye(self.n_active)
            row += self.n_active

            if w_post > 0:
                mask_f = self.posture_mask.astype(np.float64)
                res[row : row + self.n_active] = w_post * mask_f * (x - self.posture_vals)
                J[row : row + self.n_active, :] = np.diag(w_post * mask_f)
                row += self.n_active

            if w_marg > 0:
                low_v = np.maximum(0.0, self.margin_lower - x)
                high_v = np.maximum(0.0, x - self.margin_upper)
                usable_f = self.margin_usable.astype(np.float64)
                res[row : row + self.n_active] = w_marg * usable_f * (high_v - low_v)
                grad_m = usable_f * ((x > self.margin_upper).astype(np.float64) + (x < self.margin_lower).astype(np.float64))
                J[row : row + self.n_active, :] = np.diag(w_marg * grad_m)

            return res, J

        cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

        def fun(x: np.ndarray) -> np.ndarray:
            r, J = fun_and_jac(x)
            cache["last"] = (x.copy(), r, J)
            return r

        def jac(x: np.ndarray) -> np.ndarray:
            last = cache.get("last")
            if last is not None and np.array_equal(last[0], x):
                return last[2]
            _, J = fun_and_jac(x)
            return J

        sol = least_squares(
            fun,
            x0,
            jac=jac,
            bounds=(self.lower, self.upper),
            method="trf",
            max_nfev=max_nfev,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        x_sol = sol.x
        if apply_delta_cap and w.max_joint_delta is not None:
            delta = np.clip(x_sol - x_prev, -float(w.max_joint_delta), float(w.max_joint_delta))
            x_sol = np.clip(x_prev + delta, self.lower, self.upper)
        return self.embed_active(q_full_seed, x_sol)

    def solve_frame_dls(
        self,
        q_full_seed: np.ndarray,
        targets_pose7: dict[str, np.ndarray],
        *,
        damping: float = 0.04,
        iters: int = 8,
        max_step: float = 0.18,
    ) -> np.ndarray:
        """Pure myopic Damped Least Squares (Baseline 1: no posture or lookahead)."""
        x = np.clip(self.extract_active(q_full_seed), self.lower, self.upper)
        T_tgts = {s: pose7_to_mat(targets_pose7[s]) for s in self.profile.active_sides if s in targets_pose7}
        sides = tuple(T_tgts.keys())
        for _ in range(iters):
            q_cur = self.embed_active(q_full_seed, x)
            errs: list[np.ndarray] = []
            jacs: list[np.ndarray] = []
            for side in sides:
                ee = self.profile.arms[side].ee_link
                T_cur, J_pos, J_rot = self.model.link_pose_and_spatial_jacobian(
                    q_cur, ee, self.active_indices
                )
                p_err, r_err = se3_world_residual(T_cur, T_tgts[side])
                errs.append(np.concatenate([-p_err, -0.35 * r_err]))
                jacs.append(np.vstack([J_pos, 0.35 * so3_left_jacobian_inv(r_err) @ J_rot]))
            e = np.concatenate(errs)
            if float(np.linalg.norm(e)) < 1e-5:
                break
            J = np.vstack(jacs)
            JJt = J @ J.T + (damping * damping) * np.eye(J.shape[0])
            dx = J.T @ np.linalg.solve(JJt, e)
            dx = np.clip(dx, -max_step, max_step)
            x = np.clip(x + dx, self.lower, self.upper)
        return self.embed_active(q_full_seed, x)


def solve_episode_dls_myopic(
    profile: RobotProfile,
    ee_targets_by_side: dict[str, np.ndarray],
    pika_widths_by_side: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Baseline 1: Sequential myopic Damped Least Squares without posture or global smoothing."""
    solver = BimanualIKSolver(profile)
    first_side = profile.active_sides[0]
    T = len(ee_targets_by_side[first_side])
    q_traj = np.zeros((T, profile.urdf_model.num_actuated), dtype=np.float64)
    widths_out = {s: np.zeros(T, dtype=np.float64) for s in profile.active_sides}

    q_cur = profile.home_q.copy()
    for t in range(T):
        tgt = {s: ee_targets_by_side[s][t] for s in profile.active_sides}
        q_cur = solver.solve_frame_dls(q_cur, tgt, iters=20 if t == 0 else 8)
        for s in profile.active_sides:
            q_cur, rw = profile.map_gripper_width_to_q(q_cur, s, float(pika_widths_by_side[s][t]))
            widths_out[s][t] = rw
        q_traj[t] = q_cur
    return q_traj, widths_out


def solve_episode_handumi_lm(
    profile: RobotProfile,
    ee_targets_by_side: dict[str, np.ndarray],
    pika_widths_by_side: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Baseline 2: HandUMI's sequential warm-started LM with `posture_q`, `limit_margin`, and `max_joint_delta`."""
    solver = BimanualIKSolver(profile)
    first_side = profile.active_sides[0]
    T = len(ee_targets_by_side[first_side])
    q_traj = np.zeros((T, profile.urdf_model.num_actuated), dtype=np.float64)
    widths_out = {s: np.zeros(T, dtype=np.float64) for s in profile.active_sides}

    q_cur = posture_seed(profile)
    # Initial unrestricted solve (matching HandUMI replay_in_sim.py initial_solve_iterations)
    first_tgt = {s: ee_targets_by_side[s][0] for s in profile.active_sides}
    q_cur = solver.solve_frame_lm(
        q_cur, first_tgt, q_prev_full=q_cur, max_nfev=60, apply_delta_cap=False
    )
    for t in range(T):
        tgt = {s: ee_targets_by_side[s][t] for s in profile.active_sides}
        q_cur = solver.solve_frame_lm(
            q_cur,
            tgt,
            q_prev_full=q_cur,
            max_nfev=25,
            apply_delta_cap=(t > 0),
        )
        for s in profile.active_sides:
            q_cur, rw = profile.map_gripper_width_to_q(q_cur, s, float(pika_widths_by_side[s][t]))
            widths_out[s][t] = rw
        q_traj[t] = q_cur
    return q_traj, widths_out
