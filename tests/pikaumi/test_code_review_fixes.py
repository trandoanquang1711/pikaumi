"""Comprehensive regression tests verifying all code-review fixes across Sections 2, 3, 4, and 5."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pikaumi.calibration.pika_tcp import load_snapshots_directory, solve_tcp_axes_svd
from pikaumi.calibration.spatial import solve_session_table_from_vive
from pikaumi.calibration.vive import evaluate_stationary_vive_window
from pikaumi.cli import main
from pikaumi.dataset.canonical import convert_dataset_to_robot_joints
from pikaumi.dataset.curation import _link_or_copy
from pikaumi.dataset.direction import analyze_dataset_direction
from pikaumi.dataset.external_layouts import export_external_follower_layout
from pikaumi.dataset.merging import merge_datasets
from pikaumi.dataset.quality import validate_dataset
from pikaumi.dataset.raw import (
    default_calibration_metadata,
    generate_synthetic_bimanual_dataset,
    import_pika_episodes,
    load_episode_raw,
    write_episode_table,
)
from pikaumi.dataset.screening import compute_solve_cache_signature
from pikaumi.dataset.umi_delta import export_umi_delta
from pikaumi.retargeting.cstro import CStroConfig
from pikaumi.robots.registry import load_robot_profile
from pikaumi.robots.utils import mat_to_pose7, normalize_quat_xyzw, pose7_to_mat
from pikaumi.sensors.gestures import BilateralClapArbiter, DoubleClapDetector


# ==============================================================================
# Section 2: Multi-robot configs, external layouts, C-STRO & collision geometry
# ==============================================================================


def test_pika_to_ee_home_alignment_for_fixed_robots() -> None:
    """Confirm metal, openarmv1, trlc_dk1 TCP orientations align cleanly in table frame near home_q."""
    from pikaumi.calibration.deployment import load_table_calibration

    for name in ("metal", "openarmv1", "trlc_dk1"):
        prof = load_robot_profile(name)
        tcal = load_table_calibration(name, require_physical=False)
        R_robot_table = pose7_to_mat(tcal.T_robot_table_pose7)[:3, :3]
        q_home = prof.home_q.copy()
        link_tf, _ = prof.urdf_model.forward_kinematics_all(q_home)
        R_pika_to_ee = pose7_to_mat(prof.pika_to_ee_pose7)[:3, :3]
        for side, arm in prof.arms.items():
            R_ee = link_tf[arm.ee_link][:3, :3]
            R_pika_in_table = R_robot_table.T @ R_ee @ R_pika_to_ee.T
            # In table coordinates (+Y forward, +Z up), home TCP +X should point forward (+Y_table)
            # and home TCP +Z should point upward (+Z_table)!
            assert R_pika_in_table[1, 0] > 0.5, f"{name} {side} TCP +X misaligned in table frame: {R_pika_in_table[:, 0]}"
            assert R_pika_in_table[2, 2] > 0.5, f"{name} {side} TCP +Z misaligned in table frame: {R_pika_in_table[:, 2]}"


def test_r1lite_locks_torso_joints_and_piper_uses_70mm_normalized() -> None:
    r1 = load_robot_profile("r1lite")
    assert len(r1.locked_actuated_indices) >= 3
    for jname in ("torso_joint1", "torso_joint2", "torso_joint3"):
        idx = r1.urdf_model.actuated_joint_names.index(jname)
        assert idx in r1.locked_actuated_indices

    piper = load_robot_profile("piper")
    assert np.isclose(piper.gripper_max_width_m, 0.07)
    assert piper.gripper_retarget_mode == "normalized"


def test_metal_urdf_topological_order_and_shared_torso_capsules() -> None:
    metal = load_robot_profile("metal")
    names = [j.name for j in metal.urdf_model.joints]
    # Parent links must always be visited before children; FK must produce finite transforms for all links.
    link_tf, _ = metal.urdf_model.forward_kinematics_all(metal.home_q)
    assert len(link_tf) >= len(names)
    for T in link_tf.values():
        assert np.all(np.isfinite(T))
    # Shared torso capsules exist for torso-bearing robots like r1lite
    r1 = load_robot_profile("r1lite")
    shared_caps = [c for c in r1.urdf_model.capsules if c.side == "shared"]
    assert len(shared_caps) > 0


def test_external_layouts_exact_piper_and_openarm_encodings(tmp_path: Path) -> None:
    raw_dir = generate_synthetic_bimanual_dataset(tmp_path / "raw", num_episodes=1, frames_per_episode=12)

    # 1) bi_piper_follower: exact sign flips, [-100, 100] firmware scaling, [0, 100] gripper, urdf_q_full preserved
    canon_piper = tmp_path / "canon_piper"
    convert_dataset_to_robot_joints(raw_dir, canon_piper, robot="piper", solver="dls_myopic")
    ext_piper = tmp_path / "ext_piper"
    summary_p = export_external_follower_layout(canon_piper, layout="bi_piper_follower", output_dir=ext_piper)
    assert summary_p["external_layout"] == "bi_piper_follower"
    info_p = json.loads((ext_piper / "meta" / "info.json").read_text(encoding="utf-8"))
    assert info_p["features"]["observation.state"]["names"][:7] == [
        "left_shoulder_pan.pos",
        "left_shoulder_lift.pos",
        "left_elbow_flex.pos",
        "left_forearm_roll.pos",
        "left_wrist_flex.pos",
        "left_wrist_roll.pos",
        "left_gripper.pos",
    ]
    assert (ext_piper / "meta" / "tasks.jsonl").is_file()
    assert (ext_piper / "meta" / "stats.json").is_file()
    assert (ext_piper / "meta" / "abs_action_norm_stats.json").is_file()
    with np.load(ext_piper / "data" / "episode_000000.npz") as z:
        assert "urdf_q_full" in z
        assert np.all(z["observation_state"] >= -100.01) and np.all(z["observation_state"] <= 100.01)
        assert np.all(z["observation_state"][:, 6] >= -0.01) and np.all(z["observation_state"][:, 6] <= 100.01)
    # Replay-joints must work directly on the exported external layout because urdf_q_full is preserved!
    assert main(["replay-joints", str(ext_piper), "--episode", "0", "--output-dir", str(tmp_path / "rep_ext")]) == 0

    # 2) bi_openarm_follower: degrees for arm joints, [0, -60] deg for gripper
    canon_oa = tmp_path / "canon_oa"
    convert_dataset_to_robot_joints(raw_dir, canon_oa, robot="openarmv1", solver="dls_myopic")
    ext_oa = tmp_path / "ext_oa"
    export_external_follower_layout(canon_oa, layout="bi_openarm_follower", output_dir=ext_oa)
    with np.load(canon_oa / "data" / "episode_000000.npz") as z_src, np.load(
        ext_oa / "data" / "episode_000000.npz"
    ) as z_dst:
        assert np.allclose(z_dst["observation_state"][:, :7], np.degrees(z_src["observation_state"][:, :7]), atol=1e-4)
        assert np.all(z_dst["observation_state"][:, 7] <= 0.01)
        assert np.all(z_dst["observation_state"][:, 7] >= -60.01)


def test_cstro_cache_signature_invalidates_on_config_change() -> None:
    prof = load_robot_profile("piper_pika")
    info = {"calibration": {"T_table_vive": [0, 0, 0, 0, 0, 0, 1]}}
    sig1 = compute_solve_cache_signature(
        info,
        prof,
        "tblhash",
        "cstro",
        "absolute-table",
        cstro_config=CStroConfig(table_barrier_weight=12.0),
    )
    sig2 = compute_solve_cache_signature(
        info,
        prof,
        "tblhash",
        "cstro",
        "absolute-table",
        cstro_config=CStroConfig(table_barrier_weight=24.0),
    )
    assert sig1 != sig2


# ==============================================================================
# Section 3: Dataset pipeline, dropout splitting, QA, merging, direction, utils
# ==============================================================================


def test_normalize_quat_xyzw_handles_2d_zero_rows() -> None:
    qs = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]], dtype=np.float64)
    out = normalize_quat_xyzw(qs)
    assert out.shape == (2, 4)
    assert np.allclose(out[0], [0.0, 0.0, 0.0, 1.0])
    assert np.allclose(out[1], [0.0, 0.0, 0.0, 1.0])


def test_curation_link_or_copy_same_file_is_safe(tmp_path: Path) -> None:
    f = tmp_path / "ep.npz"
    f.write_bytes(b"payload123")
    _link_or_copy(f, f)
    assert f.is_file() and f.read_bytes() == b"payload123"


def test_raw_import_tracking_status_unit_inference_and_natural_sort(tmp_path: Path) -> None:
    raw_src = tmp_path / "pika_raw_src"
    for ep_name in ("episode10", "episode2"):
        ep = raw_src / ep_name
        (ep / "pose").mkdir(parents=True)
        (ep / "gripper").mkdir(parents=True)
        ts_vals = [100.0 + i / 30.0 for i in range(15)]
        for side_char in ("l", "r"):
            pose_lines = [
                json.dumps({"timestamp": f"{t:.6f}", "pose7": [0.1, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0], "accurate": True})
                for t in ts_vals
            ]
            (ep / "pose" / f"pika_pose_{side_char}.jsonl").write_text("\n".join(pose_lines), encoding="utf-8")
            # Millimeter stream that touches 0.15 mm (below 0.25) and 80.0 mm -> must scale whole stream by 1/1000!
            grip_lines = [
                json.dumps({"timestamp": f"{t:.6f}", "distance": 0.15 if i == 0 else 80.0})
                for i, t in enumerate(ts_vals)
            ]
            (ep / "gripper" / f"pika_gripper_{side_char}.jsonl").write_text("\n".join(grip_lines), encoding="utf-8")
        # Inject tracking_status.json dropout in the middle of episode2
        if ep_name == "episode2":
            status_payload = {
                "sides": {
                    "left": [
                        [int(ts_vals[0] * 1e9), True],
                        [int(ts_vals[5] * 1e9), False],
                        [int(ts_vals[9] * 1e9), True],
                    ],
                    "right": [[int(ts_vals[0] * 1e9), True]],
                }
            }
            (ep / "tracking_status.json").write_text(json.dumps(status_payload), encoding="utf-8")

    out_dir = tmp_path / "imported"
    import_pika_episodes(raw_src, out_dir, fps=30)
    eps = json.loads((out_dir / "meta" / "episodes.json").read_text(encoding="utf-8"))
    # Natural sort: episode2 BEFORE episode10!
    assert [e["source_episode"] for e in eps] == ["episode2", "episode10"]

    ep0 = load_episode_raw(out_dir, 0)
    # Tracking status dropout must have marked middle frames inaccurate
    assert not bool(np.all(ep0["left_accurate"]))
    # Stream-level mm scaling: 0.15 mm -> 0.00015 m (NOT 0.15 m!), 80.0 mm -> 0.08 m
    assert ep0["observation_state"][0, 7] < 0.005
    assert np.isclose(ep0["observation_state"][-1, 7], 0.08, atol=1e-3)

    # Missing encoder must raise FileNotFoundError instead of fabricating 0.05 m
    bad_src = tmp_path / "bad_no_grip"
    ep_bad = bad_src / "episode0"
    (ep_bad / "pose").mkdir(parents=True)
    for side_char in ("l", "r"):
        (ep_bad / "pose" / f"pika_pose_{side_char}.jsonl").write_text(
            "\n".join(
                json.dumps({"timestamp": f"{100.0 + i/30.0:.6f}", "pose7": [0, 0, 0.2, 0, 0, 0, 1]})
                for i in range(6)
            ),
            encoding="utf-8",
        )
    with pytest.raises(FileNotFoundError):
        import_pika_episodes(bad_src, tmp_path / "bad_out", fps=30)


def test_dropout_segment_splitting_in_umi_delta_and_canonical(tmp_path: Path) -> None:
    raw_dir = generate_synthetic_bimanual_dataset(
        tmp_path / "raw_drop",
        num_episodes=1,
        frames_per_episode=30,
        include_bad_tracking_episode=True,
    )
    umi_dir = tmp_path / "umi_split"
    res = export_umi_delta(raw_dir, umi_dir)
    # Episode 0 has 1 segment (29 transitions); episode 1 has a middle dropout -> split into 2 clean segments!
    assert res["num_episodes"] == 3
    assert (umi_dir / "meta" / "tasks.jsonl").is_file()
    assert (umi_dir / "meta" / "stats.json").is_file()
    stats = json.loads((umi_dir / "meta" / "stats.json").read_text(encoding="utf-8"))
    assert "q01" in stats["action"] and "q99" in stats["action"]

    # Same splitting in canonical joint export
    canon_dir = tmp_path / "canon_split"
    res_c = convert_dataset_to_robot_joints(
        raw_dir,
        canon_dir,
        robot="piper_pika",
        solver="dls_myopic",
        allow_flagged_episodes=True,
    )
    assert res_c["num_episodes"] == 3
    assert (canon_dir / "meta" / "tasks.jsonl").is_file()
    assert (canon_dir / "meta" / "stats.json").is_file()


def test_quality_rejects_nan_and_enforces_lr_image_skew_and_merge_keeps_calibration(tmp_path: Path) -> None:
    ds1 = generate_synthetic_bimanual_dataset(tmp_path / "s1", num_episodes=1, frames_per_episode=30)
    ds2 = generate_synthetic_bimanual_dataset(tmp_path / "s2", num_episodes=1, frames_per_episode=30)
    # Give ds2 a distinct T_table_vive in info.json
    info2 = json.loads((ds2 / "meta" / "info.json").read_text(encoding="utf-8"))
    info2["calibration"]["T_table_vive"] = [0.25, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0]
    (ds2 / "meta" / "info.json").write_text(json.dumps(info2), encoding="utf-8")

    merged = tmp_path / "merged"
    merge_datasets([ds1, ds2], merged)
    eps = json.loads((merged / "meta" / "episodes.json").read_text(encoding="utf-8"))
    assert eps[0]["source_episode"] == "s1/episode0"
    assert eps[1]["source_episode"] == "s2/episode0"
    assert np.allclose(eps[1]["calibration"]["T_table_vive"][:3], [0.25, -0.1, 0.0])

    # Inject NaN into episode 0 and excessive lr_image_skew_s into episode 1
    ep0 = load_episode_raw(merged, 0)
    st0 = ep0["observation_state"].copy()
    st0[5, 0] = np.nan
    write_episode_table(
        0,
        merged / "data",
        timestamps_s=ep0["timestamps_s"],
        state_16d=st0,
        action_16d=ep0["action"],
        left_accurate=ep0["left_accurate"],
        right_accurate=ep0["right_accurate"],
        sync_skew_s=ep0["sync_skew_s"],
    )
    ep1 = load_episode_raw(merged, 1)
    write_episode_table(
        1,
        merged / "data",
        timestamps_s=ep1["timestamps_s"],
        state_16d=ep1["observation_state"],
        action_16d=ep1["action"],
        left_accurate=ep1["left_accurate"],
        right_accurate=ep1["right_accurate"],
        sync_skew_s=ep1["sync_skew_s"],
        lr_image_skew_s=np.full(len(ep1["timestamps_s"]), 0.080),
    )
    qa = validate_dataset(merged)
    assert not qa["episodes"][0]["passed"]
    assert any("non_finite_values" in r for r in qa["episodes"][0]["reasons"])
    assert not qa["episodes"][1]["passed"]
    assert any("lr_image_skew_fraction" in r for r in qa["episodes"][1]["reasons"])


def test_direction_check_handles_small_n_and_mixed_thumbnails(tmp_path: Path) -> None:
    ds = generate_synthetic_bimanual_dataset(tmp_path / "small", num_episodes=2, frames_per_episode=20)
    # Remove thumbnail for episode 1 to verify mixed thumbnail datasets don't crash np.stack
    (ds / "frames" / "episode_000001_thumbs.npz").unlink()
    rep = analyze_dataset_direction(ds)
    assert rep["consensus_defined"] is False
    assert rep["flagged_reversed_episodes"] == []


def test_default_calibration_loads_legacy_pika_tcp_json(tmp_path: Path) -> None:
    T_left = np.eye(4)
    T_left[:3, 3] = [0.02, -0.01, 0.08]
    T_right = np.eye(4)
    T_right[:3, 3] = [-0.02, 0.01, 0.07]
    (tmp_path / "left_pika_tcp.json").write_text(json.dumps({"transform": T_left.tolist()}), encoding="utf-8")
    (tmp_path / "right_pika_tcp.json").write_text(json.dumps({"transform": T_right.tolist()}), encoding="utf-8")
    cal = default_calibration_metadata(tmp_path)
    assert np.allclose(cal["T_tracker_tcp"]["left"][:3], [0.02, -0.01, 0.08], atol=1e-6)
    assert np.allclose(cal["T_tracker_tcp"]["right"][:3], [-0.02, 0.01, 0.07], atol=1e-6)


# ==============================================================================
# Section 4: Calibration math, CLI wiring, and gesture edge cases
# ==============================================================================


def test_vive_gate_catches_symmetric_orientation_oscillation() -> None:
    # Alternating +4 deg and -4 deg around Z: unsigned angle from q0 has std=0 if ref=0, or biased;
    # geodesic RMS from mean rotation MUST be ~4.0 deg > 0.5 deg -> FAIL!
    poses = np.zeros((20, 7), dtype=np.float64)
    for i in range(20):
        ang = np.radians(4.0 if i % 2 == 0 else -4.0)
        poses[i, 3:7] = Rotation.from_euler("z", ang).as_quat()
    rep = evaluate_stationary_vive_window(poses, max_ori_std_deg=0.5)
    assert not rep.passed
    assert rep.orientation_std_deg > 3.5


def test_pika_tcp_axes_rejects_left_handed_and_accepts_pose_key(tmp_path: Path) -> None:
    # Left-handed frame: +X, +Y, -Z
    snaps = {
        "a0": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        "ax": np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        "ay": np.array([0.0, 0.08, 0.0, 0.0, 0.0, 0.0, 1.0]),
        "az": np.array([0.0, 0.0, -0.08, 0.0, 0.0, 0.0, 1.0]),
    }
    res = solve_tcp_axes_svd(snaps)
    assert not res["passed"]
    assert any("left_handed_axes_frame" in r for r in res["reasons"])

    # Verify load_snapshots_directory accepts legacy pika_tcp_axes.py `"pose"` key
    for k, v in snaps.items():
        (tmp_path / f"{k}.json").write_text(json.dumps({"pose": v.tolist()}), encoding="utf-8")
    loaded = load_snapshots_directory(tmp_path)
    assert set(loaded.keys()) == {"a0", "ax", "ay", "az"}


def test_spatial_table_from_vive_avoids_lever_arm_noise_amplification() -> None:
    # Table located 5 meters from Vive origin with 0.2 deg orientation noise:
    # Averaging T_vive_table directly yields sub-mm table position RMS (~0 mm), whereas inverting first
    # amplifies 0.2 deg * 5 m = 17.5 mm > 8 mm gate!
    T_vt_true = np.eye(4)
    T_vt_true[:3, 3] = [5.0, 0.0, 0.0]
    T_tc = np.eye(4)
    rng = np.random.default_rng(42)
    trk_seq = []
    cam_seq = []
    for _ in range(12):
        noise_rot = Rotation.from_euler("xyz", rng.normal(0.0, np.radians(0.2), size=3)).as_matrix()
        T_vt_obs = T_vt_true.copy()
        T_vt_obs[:3, :3] = noise_rot @ T_vt_true[:3, :3]
        trk_seq.append(mat_to_pose7(T_vt_obs))
        cam_seq.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    res = solve_session_table_from_vive(
        np.asarray(trk_seq),
        np.asarray(cam_seq),
        mat_to_pose7(T_tc),
        spatial_payload_for_hash={"cameras": {}},
        max_rms_m=0.008,
    )
    assert res["verified"] is True
    assert res["metrics"]["rms_error_m"] < 0.001


def test_calibrate_cli_subcommands_end_to_end(tmp_path: Path) -> None:
    # vive
    vive_json = tmp_path / "vive.json"
    poses = [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] for _ in range(15)]
    vive_json.write_text(json.dumps({"poses7": poses}), encoding="utf-8")
    assert main(["calibrate", "vive", "--poses-json", str(vive_json), "--output", str(tmp_path / "vive.yaml")]) == 0

    # gripper
    grip_json = tmp_path / "grip.json"
    grip_json.write_text(json.dumps({"closed_m": [0.001] * 10, "open_m": [0.099] * 10}), encoding="utf-8")
    assert (
        main(["calibrate", "gripper", "--side", "left", "--samples-json", str(grip_json), "--output", str(tmp_path / "grip.yaml")])
        == 0
    )

    # table-robot
    contacts_json = tmp_path / "contacts.json"
    contacts_json.write_text(
        json.dumps({"origin": [0.34, 0.0, -0.03], "x_axis": [0.34, -0.20, -0.03], "y_plane": [0.54, 0.0, -0.03]}),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "calibrate",
                "table-robot",
                "--robot",
                "piper_pika",
                "--contacts-json",
                str(contacts_json),
                "--output",
                str(tmp_path / "table_piper.yaml"),
                "--verified",
            ]
        )
        == 0
    )


def test_gestures_ignore_initial_closed_and_preserve_expired_pending() -> None:
    # Starting with jaw closed at t=0 must NOT arm clap 1 until after opening!
    det = DoubleClapDetector()
    assert det.step(0.00, 0.005) is False
    assert det.step(0.10, 0.080) is False  # now armed
    assert det.step(0.20, 0.005) is False  # clap 1 (NOT clap 2!)
    assert det.step(0.30, 0.080) is False
    assert det.step(0.40, 0.005) is True   # clap 2 fires here!

    # Expired unilateral pending event must be emitted even when the other hand double-claps on the same tick
    arb = BilateralClapArbiter(bilateral_window_s=0.25)
    # Arm both hands open
    assert arb.step(0.0, 0.08, 0.08) is None
    # Left double-clap at t=0.40
    assert arb.step(0.10, 0.005, 0.08) is None
    assert arb.step(0.20, 0.08, 0.08) is None
    assert arb.step(0.40, 0.005, 0.08) is None
    # Re-open left; right begins double-clap at t=0.50 and completes at t=0.80 (> 0.40 + 0.25 = 0.65)
    assert arb.step(0.50, 0.08, 0.005) is None
    assert arb.step(0.60, 0.08, 0.08) is None
    # At t=0.80, left's pending event expired AND right fires -> MUST emit left's "toggle_record" now,
    # and keep right pending to emit its own "toggle_record" at t=1.20!
    assert arb.step(0.80, 0.08, 0.005) == "toggle_record"
    assert arb.step(1.20, 0.08, 0.08) == "toggle_record"


# ==============================================================================
# Section 5: Letterbox aspect ratio preservation in prepare_pika_dataset.py
# ==============================================================================


def test_prepare_pika_dataset_letterbox_preserves_4_3_aspect_ratio() -> None:
    import prepare_pika_dataset as ppd

    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    out, (sx, sy, px, py) = ppd._letterbox_rgb(img, (224, 224))
    assert out.shape == (224, 224, 3)
    assert np.isclose(sx, sy, atol=1e-3)
    assert px == 0 and py == 28  # 640x480 -> 224x168 centered with 28px top/bottom bars
    assert np.all(out[:28, :, :] == 0)
    assert np.all(out[28:196, :, :] == 200)
    assert np.all(out[196:, :, :] == 0)

    K = np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
    K_out = ppd._letterbox_intrinsics(K, 640, 480, 224, 224)
    assert np.isclose(K_out[0, 0], K_out[1, 1], atol=1e-6)
    assert np.isclose(K_out[0, 2], 112.0, atol=1e-6)
    assert np.isclose(K_out[1, 2], 112.0, atol=1e-6)
