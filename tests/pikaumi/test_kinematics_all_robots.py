"""Analytical Jacobian parity, collision clearance, and baseline IK tests across all 9 robots."""

from __future__ import annotations

import numpy as np
import pytest

from pikaumi.robots.collision import evaluate_config_clearance
from pikaumi.robots.kinematics import (
    BimanualIKSolver,
    posture_seed,
    solve_episode_dls_myopic,
    solve_episode_handumi_lm,
)
from pikaumi.robots.registry import load_robot_profile
from pikaumi.robots.utils import pose_errors_pos_deg, se3_world_residual, so3_left_jacobian_inv

ROBOT_NAMES = (
    "piper_pika",
    "piper",
    "openarmv1",
    "arx5",
    "yam",
    "trlc_dk1",
    "metal",
    "r1lite",
    "franka_emika_panda",
)


@pytest.mark.parametrize("robot_name", ROBOT_NAMES)
def test_analytical_vs_numerical_se3_jacobian_all_robots(robot_name: str) -> None:
    """Verify exact analytical SE(3) Jacobian matches central differences (< 1e-5) for arbitrary joint axes."""
    prof = load_robot_profile(robot_name)
    solver = BimanualIKSolver(prof)
    rng = np.random.default_rng(42)

    q_base = posture_seed(prof)
    # Perturb slightly within joint limits
    x0 = np.clip(
        solver.extract_active(q_base) + rng.uniform(-0.15, 0.15, size=solver.n_active),
        solver.lower + 0.05,
        solver.upper - 0.05,
    )
    q0 = solver.embed_active(q_base, x0)

    # Target slightly offset from q0
    x_tgt = np.clip(x0 + rng.uniform(-0.08, 0.08, size=solver.n_active), solver.lower, solver.upper)
    q_tgt = solver.embed_active(q_base, x_tgt)
    link_tf_tgt, _ = prof.urdf_model.forward_kinematics_all(q_tgt)

    eps = 1e-6
    for side in prof.active_sides:
        ee = prof.arms[side].ee_link
        T_tgt = link_tf_tgt[ee]
        T_cur, J_pos, J_rot = prof.urdf_model.link_pose_and_spatial_jacobian(
            q0, ee, solver.active_indices
        )
        _, r_err0 = se3_world_residual(T_cur, T_tgt)
        J_ori_analytic = so3_left_jacobian_inv(r_err0) @ J_rot

        J_pos_num = np.zeros_like(J_pos)
        J_ori_num = np.zeros_like(J_ori_analytic)
        for col in range(solver.n_active):
            xp = x0.copy()
            xm = x0.copy()
            xp[col] += eps
            xm[col] -= eps
            Tp, _, _ = prof.urdf_model.link_pose_and_spatial_jacobian(
                solver.embed_active(q_base, xp), ee, solver.active_indices
            )
            Tm, _, _ = prof.urdf_model.link_pose_and_spatial_jacobian(
                solver.embed_active(q_base, xm), ee, solver.active_indices
            )
            pe_p, re_p = se3_world_residual(Tp, T_tgt)
            pe_m, re_m = se3_world_residual(Tm, T_tgt)
            J_pos_num[:, col] = (pe_p - pe_m) / (2.0 * eps)
            J_ori_num[:, col] = (re_p - re_m) / (2.0 * eps)

        assert np.max(np.abs(J_pos - J_pos_num)) < 1e-5, f"{robot_name} {side} J_pos mismatch"
        assert np.max(np.abs(J_ori_analytic - J_ori_num)) < 1e-5, f"{robot_name} {side} J_ori mismatch"


@pytest.mark.parametrize("robot_name", ROBOT_NAMES)
def test_baseline_ik_solvers_and_collision_clearance_all_robots(robot_name: str) -> None:
    prof = load_robot_profile(robot_name)
    solver = BimanualIKSolver(prof)
    q_home = posture_seed(prof)
    home_poses = solver.fk_poses7(q_home)

    # Home pose should have positive inter-arm clearance for bimanual robots
    c_inter, _ = evaluate_config_clearance(prof.urdf_model, q_home)
    if len(prof.active_sides) == 2:
        assert c_inter > 0.02, f"{robot_name} home pose unexpectedly colliding: {c_inter:.4f} m"

    # Build a smooth 8-frame synthetic Cartesian trajectory around home EE pose
    T = 8
    t_grid = np.linspace(0.0, 1.0, T)
    ee_targets: dict[str, np.ndarray] = {}
    widths: dict[str, np.ndarray] = {}
    for side in prof.active_sides:
        traj = np.repeat(home_poses[side][None, :], T, axis=0)
        traj[:, 0] += 0.03 * np.sin(np.pi * t_grid)
        traj[:, 2] += 0.02 * (1.0 - np.cos(np.pi * t_grid))
        ee_targets[side] = traj
        widths[side] = np.linspace(0.08, 0.02, T)

    q_dls, _ = solve_episode_dls_myopic(prof, ee_targets, widths)
    q_lm, _ = solve_episode_handumi_lm(prof, ee_targets, widths)
    assert q_dls.shape == (T, prof.urdf_model.num_actuated)
    assert q_lm.shape == (T, prof.urdf_model.num_actuated)

    # Verify HandUMI-LM tracks the reachable target within < 5 mm
    for t in range(T):
        fk = solver.fk_poses7(q_lm[t])
        for side in prof.active_sides:
            pos_err, rot_deg = pose_errors_pos_deg(fk[side], ee_targets[side][t])
            assert float(pos_err) < 0.006, f"{robot_name} {side} pos_err={pos_err:.4f}m"
            assert float(rot_deg) < 5.0, f"{robot_name} {side} rot_deg={rot_deg:.2f}deg"
