"""Unit tests for causal 40 ms synchronization, Pika jaw gestures, and deferred TCP `pika_raw` dataset (Task 4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pikaumi.dataset.raw import (
    extract_episode_tcp_targets,
    generate_synthetic_bimanual_dataset,
    import_pika_episodes,
    load_dataset_info,
)
from pikaumi.robots.utils import rpy_xyz_to_pose7
from pikaumi.sensors.gestures import BilateralClapArbiter, DoubleClapDetector
from pikaumi.synchronization import (
    SustainedHealthGate,
    build_causal_sync_grid,
    interpolate_jaw_widths,
    interpolate_se3_trajectory,
)


def test_causal_sync_slerp_and_sustained_health_gate() -> None:
    grid = build_causal_sync_grid(10.0, 11.0, fps=30.0, lag_s=0.040)
    assert len(grid) == 29
    assert grid[-1] <= 11.0 - 0.040 + 1e-9

    src_ts = np.linspace(10.0, 11.0, 101)
    poses = np.zeros((101, 7), dtype=np.float64)
    for i, t in enumerate(src_ts):
        poses[i] = rpy_xyz_to_pose7([t - 10.0, 0.0, 0.2], [0.0, 0.0, 0.5 * (t - 10.0)])
    # Flip sign on half the quaternions to test SLERP hemisphere continuity
    poses[50:, 3:7] *= -1.0

    interp, valid, skew = interpolate_se3_trajectory(src_ts, poses, grid)
    assert np.all(valid)
    assert np.max(skew) <= 0.006
    assert np.allclose(interp[:, 0], grid - 10.0, atol=1e-4)

    widths = np.linspace(-0.02, 0.14, 101)
    w_interp, w_val, _ = interpolate_jaw_widths(src_ts, widths, grid, max_width_m=0.10)
    assert np.all(w_val)
    assert np.min(w_interp) >= 0.0 and np.max(w_interp) <= 0.10

    gate = SustainedHealthGate(hold_duration_s=0.30)
    assert not gate.update(0.0, True)
    assert not gate.update(0.20, True)
    assert gate.update(0.31, True)
    assert not gate.update(0.35, False)
    assert not gate.update(0.50, True)


def test_double_clap_and_bilateral_arbiter() -> None:
    det = DoubleClapDetector()
    # Clap 1 at t=0.1, open at t=0.25, clap 2 at t=0.40 -> triggers!
    assert not det.step(0.00, 0.08)
    assert not det.step(0.10, 0.01)
    assert not det.step(0.25, 0.08)
    assert det.step(0.40, 0.01)

    arb = BilateralClapArbiter(bilateral_window_s=0.25)
    # Unilateral left double-clap -> toggle_record after settle window
    for t, wl in ((0.0, 0.08), (0.10, 0.01), (0.22, 0.08), (0.35, 0.01)):
        assert arb.step(t, wl, 0.08) is None
    assert arb.step(0.65, 0.08, 0.08) == "toggle_record"

    # Simultaneous left + right double-clap -> discard_episode immediately
    arb2 = BilateralClapArbiter(bilateral_window_s=0.25)
    for t, w in ((1.0, 0.08), (1.10, 0.01), (1.22, 0.08)):
        assert arb2.step(t, w, w) is None
    assert arb2.step(1.35, 0.01, 0.01) == "discard_episode"


def test_deferred_tcp_reapplication_and_legacy_episode_import(tmp_path: Path) -> None:
    ds_dir = generate_synthetic_bimanual_dataset(tmp_path / "raw_ds", num_episodes=2, frames_per_episode=20)
    tcp0, widths0, ts = extract_episode_tcp_targets(ds_dir, 0)
    assert len(ts) == 20
    assert tcp0["left"].shape == (20, 7)
    assert np.all((widths0["left"] >= 0.0) & (widths0["left"] <= 0.10))

    # Post-hoc TCP override shifts reconstructed TCP without re-recording Vive trajectories!
    override_tcp = {
        "left": np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        "right": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    }
    tcp1, _, _ = extract_episode_tcp_targets(ds_dir, 0, override_T_tracker_tcp=override_tcp)
    delta_norm = np.linalg.norm(tcp1["left"][:, :3] - tcp0["left"][:, :3], axis=1)
    assert np.all(delta_norm > 0.03)

    # Test importing legacy Pika_Data JSONL episode directory
    legacy_ep = tmp_path / "legacy_src" / "episode0"
    (legacy_ep / "pose").mkdir(parents=True)
    (legacy_ep / "gripper").mkdir(parents=True)
    pose_lines = [
        json.dumps({"timestamp": 100.0 + i * 0.02, "pose7": [0.1, 0.02 * i, 0.2, 0.0, 0.0, 0.0, 1.0], "accurate": True})
        for i in range(60)
    ]
    grip_lines = [
        json.dumps({"timestamp": 100.0 + i * 0.02, "distance": 80.0 - i})
        for i in range(60)
    ]
    (legacy_ep / "pose" / "pika_pose_l.jsonl").write_text("\n".join(pose_lines), encoding="utf-8")
    (legacy_ep / "pose" / "pika_pose_r.jsonl").write_text("\n".join(pose_lines), encoding="utf-8")
    (legacy_ep / "gripper" / "pika_gripper_l.jsonl").write_text("\n".join(grip_lines), encoding="utf-8")
    (legacy_ep / "gripper" / "pika_gripper_r.jsonl").write_text("\n".join(grip_lines), encoding="utf-8")

    imported_dir = tmp_path / "imported_raw"
    info = import_pika_episodes(tmp_path / "legacy_src", imported_dir, fps=30)
    assert info["num_episodes"] == 1
    assert load_dataset_info(imported_dir)["state_dim"] == 16


def test_native_pika_ros_json_and_wrist_mp4_import(tmp_path: Path) -> None:
    """Guard against float64 Unix-epoch precision loss, config.json crashes, and missing wrist MP4s."""
    import cv2

    ep_dir = tmp_path / "pika_ros_raw" / "episode3"
    for rel in (
        "localization/pose/pika_l",
        "localization/pose/pika_r",
        "localization/status/pika_l",
        "localization/status/pika_r",
        "gripper/encoder/pika_l",
        "gripper/encoder/pika_r",
        "camera/color/pikaFisheyeCamera_l",
        "camera/color/pikaFisheyeCamera_r",
    ):
        d = ep_dir / rel
        d.mkdir(parents=True)
        # Non-numeric metadata file inside stream folder must be ignored cleanly
        (d / "config.json").write_text('{"stream": "meta"}', encoding="utf-8")

    (ep_dir / "instructions.json").write_text(json.dumps({"instructions": ["fold red towel"]}), encoding="utf-8")
    base_epoch = 1793350664
    img_bgr = np.full((60, 80, 3), (30, 120, 220), dtype=np.uint8)
    for i in range(45):
        ts_str = f"{base_epoch}.{i * 25000:06d}"
        for side in ("l", "r"):
            (ep_dir / f"localization/pose/pika_{side}/{ts_str}.json").write_text(
                json.dumps({"x": 0.15 if side == "r" else -0.15, "y": 0.002 * i, "z": 0.18, "roll": 0.0, "pitch": 0.0, "yaw": 1.57}),
                encoding="utf-8",
            )
            (ep_dir / f"localization/status/pika_{side}/{ts_str}.json").write_text(
                json.dumps({"accurate": True}), encoding="utf-8"
            )
            (ep_dir / f"gripper/encoder/pika_{side}/{ts_str}.json").write_text(
                json.dumps({"distance": 75.0}), encoding="utf-8"
            )
            cv2.imwrite(str(ep_dir / f"camera/color/pikaFisheyeCamera_{side}/{ts_str}.jpg"), img_bgr)

    out_dir = tmp_path / "pika_imported_mp4"
    info = import_pika_episodes(tmp_path / "pika_ros_raw", out_dir, fps=30)
    assert info["num_episodes"] == 1
    assert (out_dir / "frames/episode_000000_thumbs.npz").is_file()
    assert (out_dir / "videos/left_wrist/episode_000000.mp4").is_file()
    assert (out_dir / "videos/right_wrist/episode_000000.mp4").is_file()


def test_causal_wrist_pair_avoids_one_frame_split() -> None:
    """When grid lands between two cameras separated by 0.8ms, snap causally instead of splitting by 33ms."""
    from pikaumi.dataset.raw import _align_causal_wrist_pair

    # 30 Hz cameras with slight clock drift and 0.8ms L-R offset
    k = np.arange(120, dtype=np.float64)
    ts_r = k * 0.03335
    ts_l = ts_r + 0.0008
    grid = np.arange(0.035, 3.8, 1.0 / 30.0)

    idx_l, idx_r = _align_causal_wrist_pair(ts_l, ts_r, grid, max_age_s=0.040)
    # Strictly causal: never peek ahead of grid
    assert np.all(ts_l[idx_l] <= grid + 1e-12)
    assert np.all(ts_r[idx_r] <= grid + 1e-12)
    # Pair stays locked within 1ms (never split by a full 33.3ms frame)
    assert float(np.max(np.abs(ts_l[idx_l] - ts_r[idx_r]))) < 0.002


