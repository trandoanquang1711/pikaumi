"""Multi-robot screening, SHA-256 solve cache conversion, follower layout export & benchmark tests (Task 7)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pikaumi.cli import main
from pikaumi.dataset.raw import generate_synthetic_bimanual_dataset


def test_screen_all_nine_robots_and_convert_with_cache(tmp_path: Path) -> None:
    ds_dir = generate_synthetic_bimanual_dataset(
        tmp_path / "raw_bench",
        num_episodes=2,
        frames_per_episode=18,
    )

    # Screen across all 9 embodiments via CLI
    rc = main(["dataset", "screen", str(ds_dir), "--robots", "all", "--solver", "cstro"])
    assert rc == 0

    for rname in (
        "piper_pika",
        "piper",
        "openarmv1",
        "arx5",
        "yam",
        "trlc_dk1",
        "metal",
        "r1lite",
        "franka_emika_panda",
    ):
        s_file = ds_dir / "meta" / f"screen_{rname}.json"
        assert s_file.is_file(), f"Missing screening report for {rname}"
        cache_file = ds_dir / "meta" / f"_solves_{rname}_cstro.npz"
        assert cache_file.is_file(), f"Missing solve cache for {rname}"

    # Convert piper_pika, piper, and openarmv1 reusing cached solves, then export follower layouts
    for rname, expected_dim, layout_name in (
        ("piper_pika", 14, "bi_piper_pika_follower"),
        ("piper", 14, "bi_piper_follower"),
        ("openarmv1", 16, "bi_openarm_follower"),
    ):
        joint_dir = tmp_path / f"joints_{rname}"
        assert (
            main(
                [
                    "convert",
                    str(ds_dir),
                    "--output-dir",
                    str(joint_dir),
                    "--format",
                    "robot-joints",
                    "--robot",
                    rname,
                    "--solver",
                    "cstro",
                    "--allow-flagged-episodes",
                ]
            )
            == 0
        )
        j_info = json.loads((joint_dir / "meta" / "info.json").read_text(encoding="utf-8"))
        assert j_info["features"]["observation.state"]["shape"] == [expected_dim]

        with np.load(joint_dir / "data" / "episode_000000.npz") as z:
            st = z["observation_state"]
            ac = z["action"]
            # action[t] == state[t+1]
            assert np.allclose(ac[:-1], st[1:], atol=1e-6)

        ext_dir = tmp_path / f"ext_{layout_name}"
        assert (
            main(
                [
                    "dataset",
                    "export",
                    str(joint_dir),
                    "--layout",
                    layout_name,
                    "--output-dir",
                    str(ext_dir),
                ]
            )
            == 0
        )
        ext_info = json.loads((ext_dir / "meta" / "info.json").read_text(encoding="utf-8"))
        assert ext_info["external_layout"] == layout_name


def test_benchmark_retarget_cli_outputs_tables(tmp_path: Path) -> None:
    ds_dir = generate_synthetic_bimanual_dataset(
        tmp_path / "mini_ds",
        num_episodes=1,
        frames_per_episode=16,
    )
    bench_out = tmp_path / "bench_out"
    rc = main(
        [
            "benchmark-retarget",
            "--dataset-dir",
            str(ds_dir),
            "--robots",
            "piper_pika",
            "openarmv1",
            "--include-ablations",
            "--output-dir",
            str(bench_out),
        ]
    )
    assert rc == 0
    assert (bench_out / "benchmark_results.json").is_file()
    assert (bench_out / "benchmark_table.md").is_file()
    res = json.loads((bench_out / "benchmark_results.json").read_text(encoding="utf-8"))
    assert len(res["rows"]) == 2 * 6  # 2 robots x (3 methods + 3 ablations)
