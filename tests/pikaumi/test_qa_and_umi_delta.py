"""Unit tests for raw QA, direction reversal detection, curation, merging, and `umi-delta` export (Task 5)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pikaumi.cli import main
from pikaumi.dataset import (
    analyze_dataset_direction,
    curate_dataset,
    export_umi_delta,
    extract_episode_tcp_targets,
    generate_synthetic_bimanual_dataset,
    merge_datasets,
    validate_dataset,
)
from pikaumi.robots.utils import pose7_to_mat


def test_validate_direction_curate_and_merge(tmp_path: Path) -> None:
    # Build dataset with 3 good episodes + 1 reversed episode (ep 3) + 1 tracking-dropout episode (ep 4)
    raw_dir = generate_synthetic_bimanual_dataset(
        tmp_path / "raw_mixed",
        num_episodes=3,
        frames_per_episode=45,
        include_reversed_episode=True,
        include_bad_tracking_episode=True,
    )

    q_rep = validate_dataset(raw_dir)
    assert q_rep["num_episodes"] == 5
    assert q_rep["num_rejected"] == 1
    assert q_rep["episodes"][4]["passed"] is False

    d_rep = analyze_dataset_direction(raw_dir)
    assert 3 in d_rep["flagged_reversed_episodes"]
    assert 0 not in d_rep["flagged_reversed_episodes"]

    curated_dir = tmp_path / "curated"
    man = curate_dataset(raw_dir, curated_dir)
    assert man["kept_episodes"] == 3
    assert len(man["dropped_episodes"]) == 2

    # Merge two clean datasets
    merged_dir = tmp_path / "merged"
    prov = merge_datasets([curated_dir, curated_dir], merged_dir, task="fold towel merged")
    assert prov["num_episodes"] == 6


def test_umi_delta_contract_matches_gemini_md(tmp_path: Path) -> None:
    raw_dir = generate_synthetic_bimanual_dataset(
        tmp_path / "raw_clean",
        num_episodes=4,
        frames_per_episode=30,
    )
    umi_dir = tmp_path / "umi_delta_out"
    rc = main(["convert", str(raw_dir), "--output-dir", str(umi_dir), "--format", "umi-delta"])
    assert rc == 0

    info = json.loads((umi_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info["features"]["observation.state"]["shape"] == [11]
    assert info["features"]["action"]["shape"] == [14]

    reports = json.loads((umi_dir / "meta" / "pika_source_reports.json").read_text(encoding="utf-8"))
    flags = reports["mandatory_trainer_flags"]
    assert flags["policy.use_delta_action"] is False
    assert flags["use_corr_state"] is False
    assert flags["drop_n_last_frames"] == 49
    assert set(reports["train_raw_episodes"]).isdisjoint(set(reports["val_raw_episodes"]))

    # Verify exact mathematical identity: integrating TCP-local action delta at t recovers TCP at t+1!
    tcp_by_side, widths_by_side, _ = extract_episode_tcp_targets(raw_dir, 0)
    with np.load(umi_dir / "data" / "episode_000000.npz") as z:
        state_11d = z["observation_state"]
        action_14d = z["action"]

    assert state_11d.shape == (29, 11)
    assert action_14d.shape == (29, 14)

    for t in range(29):
        T_l_t = pose7_to_mat(tcp_by_side["left"][t])
        T_r_t = pose7_to_mat(tcp_by_side["right"][t])
        T_rel = np.linalg.inv(T_l_t) @ T_r_t
        assert np.allclose(state_11d[t, 0:3], T_rel[:3, 3], atol=1e-6)
        assert np.allclose(state_11d[t, 3:6], T_rel[:3, 0], atol=1e-6)
        assert np.allclose(state_11d[t, 6:9], T_rel[:3, 1], atol=1e-6)
        assert np.isclose(state_11d[t, 9], widths_by_side["left"][t], atol=1e-6)
        assert np.isclose(state_11d[t, 10], widths_by_side["right"][t], atol=1e-6)

        # Left local delta integration
        T_delta_l = np.eye(4)
        T_delta_l[:3, 3] = action_14d[t, 0:3]
        T_delta_l[:3, :3] = Rotation.from_rotvec(action_14d[t, 3:6]).as_matrix()
        T_l_next_pred = T_l_t @ T_delta_l
        assert np.allclose(T_l_next_pred, pose7_to_mat(tcp_by_side["left"][t + 1]), atol=1e-5)
        assert np.isclose(action_14d[t, 6], widths_by_side["left"][t + 1], atol=1e-6)

        # Right local delta integration
        T_delta_r = np.eye(4)
        T_delta_r[:3, 3] = action_14d[t, 7:10]
        T_delta_r[:3, :3] = Rotation.from_rotvec(action_14d[t, 10:13]).as_matrix()
        T_r_next_pred = T_r_t @ T_delta_r
        assert np.allclose(T_r_next_pred, pose7_to_mat(tcp_by_side["right"][t + 1]), atol=1e-5)
        assert np.isclose(action_14d[t, 13], widths_by_side["right"][t + 1], atol=1e-6)
