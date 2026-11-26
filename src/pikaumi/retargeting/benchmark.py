"""Reproducible 9-robot retargeting benchmark & C-STRO ablation suite (`pika benchmark-retarget`)."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from pikaumi.calibration.deployment import load_table_calibration
from pikaumi.config import save_json_atomic
from pikaumi.dataset.raw import (
    generate_synthetic_bimanual_dataset,
    load_dataset_episodes,
    load_dataset_info,
)
from pikaumi.dataset.screening import (
    evaluate_retargeted_trajectory,
    solve_episode_with_method,
)
from pikaumi.retargeting.cstro import CStroConfig
from pikaumi.retargeting.pika_to_robot import prepare_episode_robot_targets
from pikaumi.robots.registry import list_robot_names, load_robot_profile


def run_retarget_benchmark(
    *,
    dataset_dir: Path | None = None,
    robots: list[str] | None = None,
    output_dir: Path,
    include_ablations: bool = True,
) -> dict[str, Any]:
    """Compare `dls_myopic`, `handumi_lm`, and `cstro` (+ ablations) across target robots."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tmp_ctx = None
    if dataset_dir is None:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="pikaumi_bench_")
        ds_path = generate_synthetic_bimanual_dataset(
            Path(tmp_ctx.name) / "bench_towel_fold",
            num_episodes=3,
            frames_per_episode=45,
            fps=30,
            include_branch_challenge=True,
        )
    else:
        ds_path = Path(dataset_dir)

    target_robots = (
        list(list_robot_names())
        if (robots is None or robots == ["all"] or "all" in robots)
        else list(robots)
    )
    info = load_dataset_info(ds_path)
    episodes = load_dataset_episodes(ds_path)
    fps = float(info.get("fps", 30.0))

    methods: list[tuple[str, str, CStroConfig | None]] = [
        ("dls_myopic", "dls_myopic", None),
        ("handumi_lm", "handumi_lm", None),
        ("cstro", "cstro", CStroConfig()),
    ]
    if include_ablations:
        methods.extend(
            [
                ("cstro_no_viterbi", "cstro", CStroConfig(use_viterbi_seed=False)),
                ("cstro_no_bimanual", "cstro", CStroConfig(use_bimanual_coupling=False)),
                ("cstro_no_zerophase", "cstro", CStroConfig(use_zero_phase_smoothing=False)),
            ]
        )

    rows: list[dict[str, Any]] = []
    for rname in target_robots:
        prof = load_robot_profile(rname)
        tcal = load_table_calibration(rname, require_physical=False)
        mode = "relative-anchor" if len(prof.active_sides) == 1 else "absolute-table"

        prepared = [
            prepare_episode_robot_targets(
                str(ds_path), int(ep["episode_index"]), prof, table_calib=tcal, retarget_mode=mode
            )
            for ep in episodes
        ]

        for label, solver_key, cfg in methods:
            ep_metrics: list[dict[str, Any]] = []
            solve_times: list[float] = []
            pass_count = 0
            for ee_tgts, widths, _, table_z in prepared:
                q_traj, _, dt_s = solve_episode_with_method(
                    prof,
                    ee_tgts,
                    widths,
                    solver_name=solver_key,
                    fps=fps,
                    table_z_robot=table_z,
                    cstro_config=cfg,
                )
                m = evaluate_retargeted_trajectory(
                    prof, q_traj, ee_tgts, widths, fps=fps, table_z_robot=table_z
                )
                ep_metrics.append(m)
                solve_times.append(dt_s)
                if m["verdict"] != "REJECT":
                    pass_count += 1

            agg = {
                "robot": rname,
                "dof_per_arm": len(prof.arms[prof.active_sides[0]].arm_joint_names),
                "arms": len(prof.active_sides),
                "method": label,
                "pass_rate_pct": round(100.0 * pass_count / max(len(ep_metrics), 1), 1),
                "mean_pos_err_mm": round(float(np.mean([m["mean_pos_err_mm"] for m in ep_metrics])), 3),
                "max_pos_err_mm": round(float(np.max([m["max_pos_err_mm"] for m in ep_metrics])), 3),
                "mean_rot_err_deg": round(float(np.mean([m["mean_rot_err_deg"] for m in ep_metrics])), 3),
                "cograsp_rel_pos_err_mm": round(float(np.mean([m["cograsp_rel_pos_err_mm"] for m in ep_metrics])), 3),
                "urdf_vel_viol_pct": round(100.0 * float(np.mean([m["urdf_vel_violation_fraction"] for m in ep_metrics])), 2),
                "rms_jerk_rad_s3": round(float(np.mean([m["rms_jerk_rad_s3"] for m in ep_metrics])), 2),
                "max_base_swing_deg": round(float(np.max([m["max_base_swing_deg"] for m in ep_metrics])), 2),
                "mean_solve_time_s": round(float(np.mean(solve_times)), 4),
            }
            rows.append(agg)

    if tmp_ctx is not None:
        tmp_ctx.cleanup()

    # Format Markdown summary table
    md_lines = [
        "# Multi-Robot Retargeting Benchmark (`C-STRO` vs. `HandUMI-LM` vs. `Myopic-DLS`)",
        "",
        "| Robot | Arms x DoF | Method | Pass % | Pos Err (mm) | Co-Grasp Rel (mm) | Vel Viol (%) | RMS Jerk (rad/s^3) | Max Base Swing (deg) | Solve (s) |",
        "| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| `{r['robot']}` | {r['arms']}x{r['dof_per_arm']} | `{r['method']}` | "
            f"{r['pass_rate_pct']:.1f}% | {r['mean_pos_err_mm']:.2f} | {r['cograsp_rel_pos_err_mm']:.2f} | "
            f"{r['urdf_vel_viol_pct']:.2f}% | {r['rms_jerk_rad_s3']:.1f} | {r['max_base_swing_deg']:.1f} | "
            f"{r['mean_solve_time_s']:.3f} |"
        )
    md_text = "\n".join(md_lines) + "\n"
    (out / "benchmark_table.md").write_text(md_text, encoding="utf-8")

    summary = {
        "num_robots": len(target_robots),
        "robots": target_robots,
        "num_episodes_per_robot": len(episodes),
        "fps": fps,
        "rows": rows,
        "markdown_table_path": str(out / "benchmark_table.md"),
    }
    save_json_atomic(out / "benchmark_results.json", summary)
    return summary


def run_benchmark_cli(
    *,
    dataset_dir: Path | None,
    robots: list[str],
    output_dir: Path,
    include_ablations: bool,
) -> int:
    summary = run_retarget_benchmark(
        dataset_dir=dataset_dir,
        robots=robots,
        output_dir=output_dir,
        include_ablations=include_ablations,
    )
    print((Path(summary["markdown_table_path"])).read_text(encoding="utf-8"))
    print(f"[OK] Saved benchmark JSON & Markdown table to {output_dir}")
    return 0
