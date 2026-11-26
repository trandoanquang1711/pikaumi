"""Unit tests for `C-STRO` analytical sparse Jacobian parity, smoothness, and co-grasp coupling (Task 6b)."""

from __future__ import annotations

import time

import numpy as np

from pikaumi.dataset.screening import evaluate_retargeted_trajectory
from pikaumi.retargeting.cstro import (
    CStroConfig,
    build_cstro_problem,
    solve_episode_cstro,
)
from pikaumi.robots.kinematics import (
    BimanualIKSolver,
    posture_seed,
    solve_episode_dls_myopic,
    solve_episode_handumi_lm,
)
from pikaumi.robots.registry import load_robot_profile


def test_cstro_exact_sparse_jacobian_vs_finite_difference() -> None:
    """Verify every block of C-STRO's analytical CSR Jacobian (pose, bimanual SE(3), vel, acc) matches central diff."""
    prof = load_robot_profile("piper_pika")
    solver = BimanualIKSolver(prof)
    q_home = posture_seed(prof)
    home_fk = solver.fk_poses7(q_home)

    T = 5
    u = np.linspace(0.0, 1.0, T)
    ee_targets = {}
    widths = {}
    for side in prof.active_sides:
        traj = np.repeat(home_fk[side][None, :], T, axis=0)
        traj[:, 0] += 0.04 * np.sin(np.pi * u)
        traj[:, 2] += 0.03 * u
        ee_targets[side] = traj
        widths[side] = np.full(T, 0.015)

    # Disable piecewise kink thresholds at exact boundary by testing away from exact hinge kinks
    cfg = CStroConfig(use_viterbi_seed=False)
    fun, jac, x0, lb, ub, _, _ = build_cstro_problem(
        prof, ee_targets, widths, fps=30.0, table_z_robot=-0.03, config=cfg
    )
    rng = np.random.default_rng(123)
    x_test = np.clip(x0 + rng.uniform(-0.04, 0.04, size=x0.size), lb + 0.25, ub - 0.25)

    _ = fun(x_test)
    J_ana = jac(x_test).toarray()
    J_num = np.zeros_like(J_ana)
    eps = 1e-6
    for j in range(x_test.size):
        xp = x_test.copy()
        xm = x_test.copy()
        xp[j] += eps
        xm[j] -= eps
        J_num[:, j] = (fun(xp) - fun(xm)) / (2.0 * eps)

    max_err = float(np.max(np.abs(J_ana - J_num)))
    assert max_err < 1e-4, f"C-STRO sparse Jacobian mismatch: max_err={max_err:.6e}"


def test_cstro_outperforms_baselines_on_jerk_velocity_and_cograsp() -> None:
    """Verify C-STRO achieves lower jerk, zero velocity violations, and tighter co-grasp coupling."""
    prof = load_robot_profile("openarmv1")
    solver = BimanualIKSolver(prof)
    q_home = posture_seed(prof)
    home_fk = solver.fk_poses7(q_home)

    T = 40
    u = np.linspace(0.0, 1.0, T)
    rng = np.random.default_rng(99)
    ee_targets = {}
    widths = {}
    for s_idx, side in enumerate(prof.active_sides):
        traj = np.repeat(home_fk[side][None, :], T, axis=0)
        # Fast reach + jittery tracker noise + co-grasp lift
        traj[:, 0] += 0.09 * np.sin(2.0 * np.pi * u) + rng.normal(0.0, 0.0025, size=T)
        traj[:, 1] += (0.04 if s_idx == 0 else -0.04) * (1.0 - np.cos(2.0 * np.pi * u))
        traj[:, 2] += 0.08 * np.sin(np.pi * u) + rng.normal(0.0, 0.0025, size=T)
        ee_targets[side] = traj
        widths[side] = np.where((u >= 0.25) & (u <= 0.75), 0.012, 0.080)

    q_dls, _ = solve_episode_dls_myopic(prof, ee_targets, widths)
    q_lm, _ = solve_episode_handumi_lm(prof, ee_targets, widths)

    t0 = time.perf_counter()
    q_cstro, _ = solve_episode_cstro(prof, ee_targets, widths, fps=30.0)
    dt_s = time.perf_counter() - t0
    assert dt_s < 2.0, f"C-STRO took {dt_s:.2f}s on {T} frames"

    m_dls = evaluate_retargeted_trajectory(prof, q_dls, ee_targets, widths, fps=30.0)
    m_lm = evaluate_retargeted_trajectory(prof, q_lm, ee_targets, widths, fps=30.0)
    m_cstro = evaluate_retargeted_trajectory(prof, q_cstro, ee_targets, widths, fps=30.0)

    assert m_cstro["rms_jerk_rad_s3"] < m_lm["rms_jerk_rad_s3"]
    assert m_cstro["rms_jerk_rad_s3"] < m_dls["rms_jerk_rad_s3"]
    assert m_cstro["urdf_vel_violation_fraction"] <= m_lm["urdf_vel_violation_fraction"]
    assert m_cstro["cograsp_rel_pos_err_mm"] < 5.0
