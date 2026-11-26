"""Regression tests for Round 2 code review fixes across dataset, collision, sensor env, and calibration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from pikaumi.calibration.cli_handlers import import_legacy_calibration_dir
from pikaumi.calibration.deployment import solve_table_from_three_contact_points
from pikaumi.calibration.pika_tcp import compute_pika_tcp_sha256
from pikaumi.calibration.verify import verify_calibration_bundle
from pikaumi.config import DEFAULT_DATA_CALIB_DIR, load_yaml, save_yaml_atomic
from pikaumi.dataset.canonical import convert_dataset_to_robot_joints
from pikaumi.dataset.curation import curate_dataset
from pikaumi.dataset.external_layouts import export_external_follower_layout
from pikaumi.dataset.merging import merge_datasets
from pikaumi.dataset.quality import validate_dataset
from pikaumi.dataset.raw import (
    apply_tracking_status_with_padding,
    generate_synthetic_bimanual_dataset,
    load_episode_raw,
)
from pikaumi.dataset.screening import compute_solve_cache_signature
from pikaumi.dataset.umi_delta import export_umi_delta
from pikaumi.robots.collision import distal_capsules_by_links, world_capsules_for_config
from pikaumi.robots.registry import load_robot_profile


ROOT = Path(__file__).resolve().parents[2]


def _load_start_pika_sensors_module():
    path = ROOT / "Pika_Data" / "scripts" / "start_pika_sensors.py"
    spec = importlib.util.spec_from_file_location("start_pika_sensors_r2", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_dummy_mp4(path: Path, *, width: int = 64, height: int = 48, frames: int = 16, fps: int = 30) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    assert writer.isOpened()
    for i in range(frames):
        img = np.full((height, width, 3), fill_value=(i * 10) % 255, dtype=np.uint8)
        writer.write(img)
    writer.release()


def test_critical_2_1_chw_video_metadata_and_full_norm_stats_all_exporters(tmp_path: Path) -> None:
    raw_ds = generate_synthetic_bimanual_dataset(tmp_path / "raw_ds", num_episodes=1, frames_per_episode=16)
    _write_dummy_mp4(raw_ds / "videos" / "left_wrist" / "episode_000000.mp4", width=64, height=48, frames=16)
    _write_dummy_mp4(raw_ds / "videos" / "right_wrist" / "episode_000000.mp4", width=64, height=48, frames=16)

    # 1. umi-delta exporter
    umi_out = tmp_path / "umi_out"
    export_umi_delta(raw_ds, umi_out)
    umi_info = json.loads((umi_out / "meta" / "info.json").read_text(encoding="utf-8"))
    for cam in ("observation.images.left_wrist", "observation.images.right_wrist"):
        feat = umi_info["features"][cam]
        assert feat["shape"] == [3, 48, 64]
        assert feat["names"] == ["channels", "height", "width"]
    umi_stats = json.loads((umi_out / "meta" / "abs_action_norm_stats.json").read_text(encoding="utf-8"))
    assert "observation.state" in umi_stats and "action" in umi_stats

    # 2. canonical robot-joints exporter
    joints_out = tmp_path / "joints_out"
    convert_dataset_to_robot_joints(raw_ds, joints_out, robot="piper")
    j_info = json.loads((joints_out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert j_info.get("table_calib_sha256")
    for cam in ("observation.images.left_wrist", "observation.images.right_wrist"):
        feat = j_info["features"][cam]
        assert feat["shape"] == [3, 48, 64]
        assert feat["names"] == ["channels", "height", "width"]
    j_stats = json.loads((joints_out / "meta" / "abs_action_norm_stats.json").read_text(encoding="utf-8"))
    assert "observation.state" in j_stats and "action" in j_stats

    # 3. external follower layout exporter (bi_piper_follower)
    ext_out = tmp_path / "ext_out"
    export_external_follower_layout(
        joints_out,
        layout="bi_piper_follower",
        output_dir=ext_out,
    )
    ext_info = json.loads((ext_out / "meta" / "info.json").read_text(encoding="utf-8"))
    for cam in ("observation.images.left_wrist", "observation.images.right_wrist"):
        feat = ext_info["features"][cam]
        assert feat["shape"] == [3, 48, 64]
        assert feat["names"] == ["channels", "height", "width"]
    ext_stats = json.loads((ext_out / "meta" / "abs_action_norm_stats.json").read_text(encoding="utf-8"))
    assert "observation.state" in ext_stats and "action" in ext_stats
    assert (ext_out / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    tbl = pq.read_table(ext_out / "data" / "episode_000000.parquet")
    assert "index" in tbl.column_names and "next.done" in tbl.column_names


def test_critical_2_2_start_pika_sensors_lowercase_env_export_and_distinct_validation(tmp_path: Path) -> None:
    mod = _load_start_pika_sensors_module()

    # Lowercase env keys required by sensor_tools
    env = mod.build_sensor_child_env({}, {"left": "LHR-0AE19791", "right": "LHR-8A6C081E"})
    assert env["pika_L_code"] == "LHR-0AE19791"
    assert env["pika_R_code"] == "LHR-8A6C081E"
    assert "PIKA_L_CODE" not in env and "PIKA_R_CODE" not in env

    # Reject identical or single-sided Vive codes
    with pytest.raises(ValueError, match="distinct"):
        mod.validate_vive_trackers({"left": "LHR-0AE19791", "right": "LHR-0AE19791"})
    with pytest.raises(ValueError, match="both 'left' and 'right'"):
        mod.validate_vive_trackers({"left": "LHR-0AE19791"})

    # Match optional `export ` prefix in startup bash script
    script_path = tmp_path / "start_multi_sensor.bash"
    script_path.write_text(
        "#!/bin/bash\ncamera_fps=30\nexport pika_L_code=LHR-OLD1\npika_R_code=LHR-OLD2\n",
        encoding="utf-8",
    )
    out = mod.configured_script(
        script_path,
        vive_trackers={"left": "LHR-0AE19791", "right": "LHR-8A6C081E"},
    )
    assert "export pika_L_code=LHR-0AE19791" in out
    assert "pika_R_code=LHR-8A6C081E" in out

    # Also verify start.sh forwards lowercase pika_L_code / pika_R_code
    start_sh = (ROOT / "Pika_Data" / "start.sh").read_text(encoding="utf-8")
    assert "-e pika_L_code=" in start_sh
    assert "-e pika_R_code=" in start_sh


def test_high_3_2_sensor_gap_qc_and_curate_preserves_lr_image_skew(tmp_path: Path) -> None:
    from pikaumi.dataset.raw import write_episode_table

    raw_ds = generate_synthetic_bimanual_dataset(tmp_path / "raw_gap", num_episodes=2, frames_per_episode=60)

    # Re-write episode 0 with explicit lr_image_skew_s column to verify curate_dataset preserves it
    ep0 = load_episode_raw(raw_ds, 0)
    skew = np.full(len(ep0["timestamps_s"]), 0.004, dtype=np.float64)
    write_episode_table(
        0,
        raw_ds / "data",
        timestamps_s=ep0["timestamps_s"],
        state_16d=ep0["observation_state"],
        action_16d=ep0["action"],
        left_accurate=ep0["left_accurate"],
        right_accurate=ep0["right_accurate"],
        sync_skew_s=ep0["sync_skew_s"],
        lr_image_skew_s=skew,
    )

    # Inject pre-resampling sensor gap > 30 ms into episode 1 metadata
    eps_path = raw_ds / "meta" / "episodes.json"
    eps = json.loads(eps_path.read_text(encoding="utf-8"))
    eps[1]["max_sensor_gap_ms"] = 85.0
    eps_path.write_text(json.dumps(eps), encoding="utf-8")

    qrep = validate_dataset(raw_ds)
    assert qrep["episodes"][0]["passed"] is True
    assert qrep["episodes"][1]["passed"] is False
    assert any("sensor_gap_ms" in r for r in qrep["episodes"][1]["reasons"])

    # Curate episode 0 (quality_report.json already saved by validate_dataset) and verify lr_image_skew_s survives
    curated = tmp_path / "curated"
    curate_dataset(raw_ds, curated)
    cur_ep = load_episode_raw(curated, 0)
    assert "lr_image_skew_s" in cur_ep
    assert np.allclose(cur_ep["lr_image_skew_s"], skew, atol=1e-6)


def test_high_3_3_distal_capsules_include_wrist_links_on_piper_pika() -> None:
    prof = load_robot_profile("piper_pika")
    left_caps, right_caps = world_capsules_for_config(prof.urdf_model, prof.home_q)

    # Old [-2:] only captured left_gripper_base (2 capsules) and missed link5/link6!
    old_links = {c.link_name for c, _, _ in left_caps[-2:]}
    assert old_links == {"left_gripper_base"}

    distal_l = distal_capsules_by_links(left_caps, n_distinct_links=3)
    distal_r = distal_capsules_by_links(right_caps, n_distinct_links=3)
    distal_l_links = {c.link_name for c, _, _ in distal_l}
    distal_r_links = {c.link_name for c, _, _ in distal_r}
    assert {"left_link5", "left_link6", "left_gripper_base"}.issubset(distal_l_links)
    assert {"right_link5", "right_link6", "right_gripper_base"}.issubset(distal_r_links)


def test_high_3_4_umi_delta_v3_columns_flagged_filter_padding_and_lerobot_indices(tmp_path: Path) -> None:
    # 1. Tracking dropout padding: 50ms pre-dropout and 250ms post-recovery
    ts = np.arange(0.0, 1.0, 0.025, dtype=np.float64)  # 40 Hz (25 ms step)
    st_ts = np.array([0.0, 0.30, 0.40], dtype=np.float64)
    st_acc = np.array([True, False, True], dtype=bool)
    acc = apply_tracking_status_with_padding(ts, st_ts, st_acc, pre_pad_s=0.05, post_pad_s=0.25)
    # t=0.25..0.30 (pre-pad 50ms) and t=0.40..0.65 (post-pad 250ms) must all be False
    assert bool(acc[np.argmin(np.abs(ts - 0.275))]) is False
    assert bool(acc[np.argmin(np.abs(ts - 0.625))]) is False
    assert bool(acc[np.argmin(np.abs(ts - 0.100))]) is True
    assert bool(acc[np.argmin(np.abs(ts - 0.800))]) is True

    # 2. umi_delta writes index, next.done, chunk-000, and skips episodes flagged in quality_report.json
    raw_ds = generate_synthetic_bimanual_dataset(tmp_path / "raw_umi_v3", num_episodes=2, frames_per_episode=40)
    (raw_ds / "meta" / "quality_report.json").write_text(
        json.dumps(
            {
                "episodes": [
                    {"episode_index": 0, "passed": True, "reasons": []},
                    {"episode_index": 1, "passed": False, "reasons": ["synthetic_fail"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    umi_out = tmp_path / "umi_filtered"
    summary = export_umi_delta(raw_ds, umi_out, min_segment_seconds=1.0)
    assert summary["num_episodes"] == 1
    assert (umi_out / "data" / "chunk-000" / "episode_000000.parquet").is_file()
    tbl = pq.read_table(umi_out / "data" / "episode_000000.parquet")
    assert "index" in tbl.column_names and "next.done" in tbl.column_names
    done_col = tbl["next.done"].to_pylist()
    assert done_col[-1] is True and not any(done_col[:-1])

    # 3. Verify prepare_pika_vla populates lerobot_episode_indices in source reports
    vla_src = (ROOT / "Pika_Data" / "data_tools" / "prepare_pika_vla.py").read_text(encoding="utf-8")
    assert '"lerobot_episode_indices"' in vla_src


def test_medium_4_1_solve_cache_invalidates_on_table_vive_and_per_episode_calib() -> None:
    prof = load_robot_profile("piper_pika")
    info_a = {
        "fps": 30,
        "num_episodes": 2,
        "calibration": {
            "spatial_sha256": "abc123",
            "T_table_vive": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
    }
    info_b = {
        "fps": 30,
        "num_episodes": 2,
        "calibration": {
            "spatial_sha256": "abc123",
            "T_table_vive": [0.15, -0.05, 0.0, 0.0, 0.0, 0.0, 1.0],
        },
    }
    sig_a = compute_solve_cache_signature(info_a, prof, "sha_table", "absolute-table", 30, "cstro")
    sig_b = compute_solve_cache_signature(info_b, prof, "sha_table", "absolute-table", 30, "cstro")
    assert sig_a != sig_b

    ep_a = [{"episode_index": 0, "calibration": {"T_table_vive": [0.0, 0.0, 0.0, 0, 0, 0, 1]}}]
    ep_b = [{"episode_index": 0, "calibration": {"T_table_vive": [0.2, 0.0, 0.0, 0, 0, 0, 1]}}]
    sig_ep_a = compute_solve_cache_signature(info_a, prof, "sha_table", "absolute-table", 30, "cstro", episodes=ep_a)
    sig_ep_b = compute_solve_cache_signature(info_a, prof, "sha_table", "absolute-table", 30, "cstro", episodes=ep_b)
    assert sig_ep_a != sig_ep_b


def test_medium_4_2_merge_same_basename_sources_disambiguates_source_episode(tmp_path: Path) -> None:
    ds1 = generate_synthetic_bimanual_dataset(tmp_path / "session_a" / "curated", num_episodes=1, frames_per_episode=12)
    ds2 = generate_synthetic_bimanual_dataset(tmp_path / "session_b" / "curated", num_episodes=1, frames_per_episode=12)
    merged = tmp_path / "merged_same_leaf"
    merge_datasets([ds1, ds2], merged)
    eps = json.loads((merged / "meta" / "episodes.json").read_text(encoding="utf-8"))
    assert eps[0]["source_episode"] != eps[1]["source_episode"]
    assert eps[0]["source_episode"] == "s00_curated/episode0"
    assert eps[1]["source_episode"] == "s01_curated/episode0"


def test_medium_4_4_verify_tcp_sha256_and_reject_inverted_table_normal(tmp_path: Path) -> None:
    # 1. Tampering with pika_tcp.yaml after storing pika_tcp_sha256 must fail verify_calibration_bundle
    cal_dir = tmp_path / "cal_bundle"
    import_legacy_calibration_dir(DEFAULT_DATA_CALIB_DIR, cal_dir)
    tcp_path = cal_dir / "pika_tcp.yaml"
    tcp_doc = load_yaml(tcp_path)
    tcp_doc["pika_tcp_sha256"] = compute_pika_tcp_sha256(tcp_doc)
    save_yaml_atomic(tcp_path, tcp_doc)
    assert verify_calibration_bundle(cal_dir)["passed"] is True

    # Tamper with left TCP position without updating pika_tcp_sha256
    tcp_doc["sides"]["left"]["position"][0] += 0.02
    save_yaml_atomic(tcp_path, tcp_doc)
    rep = verify_calibration_bundle(cal_dir)
    assert rep["passed"] is False
    assert "sha256_mismatch:pika_tcp.yaml" in rep["reasons"]

    # 2. Inverted 3-point contact order (origin -> +X -> -Y) must raise ValueError instead of flipping silently
    with pytest.raises(ValueError, match="Inverted table normal"):
        solve_table_from_three_contact_points(
            np.array([0.35, 0.0, -0.02]),
            np.array([0.55, 0.0, -0.02]),
            np.array([0.35, -0.20, -0.02]),
        )
