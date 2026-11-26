"""Unit tests for all 5 gated calibration layers and SHA-256 provenance verification (Task 3)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pikaumi.calibration import (
    calibrate_gripper_encoder,
    evaluate_stationary_vive_window,
    load_table_calibration,
    save_physical_table_calibration,
    solve_hand_eye_mount,
    solve_session_table_from_vive,
    solve_table_from_three_contact_points,
    solve_tcp_axes_svd,
    solve_tcp_pivot_translation,
    verify_calibration_bundle,
)
from pikaumi.cli import main
from pikaumi.config import DEFAULT_DATA_CALIB_DIR, load_yaml, save_yaml_atomic
from pikaumi.robots.utils import mat_to_pose7, pose7_to_mat, rpy_xyz_to_pose7


def test_layer1_vive_and_layer2_gripper_gates() -> None:
    rng = np.random.default_rng(7)
    poses = np.repeat(np.array([[0.2, -0.1, 0.5, 0.0, 0.0, 0.0, 1.0]]), 20, axis=0)
    poses[:, :3] += rng.normal(0.0, 0.0004, size=(20, 3))
    rep = evaluate_stationary_vive_window(poses, np.ones(20, dtype=bool))
    assert rep.passed and rep.position_std_mm < 1.5

    rep_bad = evaluate_stationary_vive_window(poses, np.array([True] * 19 + [False]))
    assert not rep_bad.passed

    gcal = calibrate_gripper_encoder(
        "left",
        closed_samples_m=np.full(15, 0.001),
        open_samples_m=np.full(15, 0.099),
    )
    assert gcal.passed
    assert np.isclose(gcal.map_width_m(0.001), 0.0)
    assert np.isclose(gcal.map_width_m(0.099), 0.10)


def test_layer3_hand_eye_and_session_table_sha256() -> None:
    rng = np.random.default_rng(11)
    T_tc_gt = pose7_to_mat(rpy_xyz_to_pose7([0.03, -0.02, 0.05], [0.2, -0.1, 0.4]))
    T_wb_gt = pose7_to_mat(rpy_xyz_to_pose7([0.45, 0.05, -0.15], [0.0, 0.0, 0.25]))

    trk_seq: list[np.ndarray] = []
    cam_seq: list[np.ndarray] = []
    for _ in range(12):
        rpy = rng.uniform(-0.7, 0.7, size=3)
        xyz = np.array([0.25, 0.0, 0.25]) + rng.uniform(-0.12, 0.12, size=3)
        T_wt = pose7_to_mat(rpy_xyz_to_pose7(xyz, rpy))
        T_cb = np.linalg.inv(T_tc_gt) @ np.linalg.inv(T_wt) @ T_wb_gt
        trk_seq.append(mat_to_pose7(T_wt))
        cam_seq.append(mat_to_pose7(T_cb))

    trk_arr = np.stack(trk_seq)
    cam_arr = np.stack(cam_seq)
    he = solve_hand_eye_mount(trk_arr, cam_arr)
    assert he["passed"] and he["rms_error_m"] < 1e-4
    assert np.allclose(np.asarray(he["T_tracker_cam"])[:3], T_tc_gt[:3, 3], atol=1e-3)

    spatial_doc = {"schema_version": 1, "mounts": {"left": {"T_tracker_cam": he["T_tracker_cam"]}}}
    ses = solve_session_table_from_vive(
        trk_arr,
        cam_arr,
        np.asarray(he["T_tracker_cam"]),
        spatial_payload_for_hash=spatial_doc,
    )
    assert ses["verified"] is True
    T_tv_gt = mat_to_pose7(np.linalg.inv(T_wb_gt))
    assert np.allclose(ses["T_table_vive"]["position"], T_tv_gt[:3], atol=1e-3)


def test_layer4_tcp_pivot_and_svd_axes() -> None:
    p_tcp_gt = np.array([0.012, -0.004, 0.145], dtype=np.float64)
    p_piv_gt = np.array([0.35, -0.10, 0.02], dtype=np.float64)
    rng = np.random.default_rng(19)
    poses: list[np.ndarray] = []
    for _ in range(14):
        R = Rotation.from_euler("xyz", rng.uniform(-0.9, 0.9, size=3)).as_matrix()
        t = p_piv_gt - R @ p_tcp_gt + rng.normal(0.0, 0.0005, size=3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        poses.append(mat_to_pose7(T))

    piv = solve_tcp_pivot_translation(np.stack(poses))
    assert piv["passed"]
    assert np.allclose(piv["tcp_translation_m"], p_tcp_gt, atol=0.0015)

    # 3-axis SVD orientation recovery
    R_trk_tcp_gt = Rotation.from_euler("xyz", [0.15, -0.25, 0.40]).as_matrix()
    R_w_trk = Rotation.from_euler("xyz", [-0.3, 0.2, 0.5]).as_matrix()
    p0 = np.array([0.2, 0.1, 0.3])
    q0 = Rotation.from_matrix(R_w_trk).as_quat()
    step = 0.080
    snaps = {
        "a0": np.concatenate([p0, q0]),
        "ax": np.concatenate([p0 + R_w_trk @ R_trk_tcp_gt[:, 0] * step, q0]),
        "ay": np.concatenate([p0 + R_w_trk @ R_trk_tcp_gt[:, 1] * step, q0]),
        "az": np.concatenate([p0 + R_w_trk @ R_trk_tcp_gt[:, 2] * step, q0]),
    }
    ax = solve_tcp_axes_svd(snaps)
    assert ax["passed"]
    assert np.allclose(np.asarray(ax["rotation_matrix"]), R_trk_tcp_gt, atol=1e-5)


def test_layer5_deployment_gate_and_legacy_import_verify_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pikaumi.calibration.deployment as deployment_mod

    monkeypatch.setattr(deployment_mod, "TABLE_LOCAL_CALIB_DIR", tmp_path / "empty_local_calibs")
    # Simulation table must block require_physical=True
    with pytest.raises(PermissionError):
        load_table_calibration("piper_pika", require_physical=True)

    # Three-point contact table calibration + verified physical save
    pose7 = solve_table_from_three_contact_points(
        np.array([0.35, 0.0, -0.02]),
        np.array([0.55, 0.0, -0.02]),
        np.array([0.35, 0.20, -0.02]),
    )
    phys_yaml = save_physical_table_calibration(
        tmp_path / "piper_pika_local.yaml",
        robot="piper_pika",
        T_robot_table_pose7=pose7,
        verified=True,
    )
    loaded = load_table_calibration("piper_pika", path=phys_yaml, require_physical=True)
    assert loaded.verified and loaded.scope == "physical"

    # Import real workspace data/calibration JSONs and run `pika calibrate verify`
    cal_out = tmp_path / "imported_calib"
    assert main(["calibrate", "import-legacy", "--legacy-dir", str(DEFAULT_DATA_CALIB_DIR), "--output-dir", str(cal_out)]) == 0

    # Construct synthetic two-tip touch where left & right TCP tips meet within 2 mm
    tcp_doc = load_yaml(cal_out / "pika_tcp.yaml")
    p_l = np.asarray(tcp_doc["sides"]["left"]["position"])
    p_r = np.asarray(tcp_doc["sides"]["right"]["position"])
    touch_pt = np.array([0.30, 0.0, 0.10])
    touch_file = tmp_path / "touch.json"
    touch_file.write_text(
        json.dumps(
            {
                "tip_touch": {
                    "left_tracker_pose7": [*(touch_pt - p_l).tolist(), 0.0, 0.0, 0.0, 1.0],
                    "right_tracker_pose7": [*(touch_pt - p_r + np.array([0.002, 0.0, 0.0])).tolist(), 0.0, 0.0, 0.0, 1.0],
                }
            }
        ),
        encoding="utf-8",
    )
    ver_json = tmp_path / "verify_report.json"
    assert (
        main(
            [
                "calibrate",
                "verify",
                "--calibration-dir",
                str(cal_out),
                "--touch-json",
                str(touch_file),
                "--json-out",
                str(ver_json),
            ]
        )
        == 0
    )

    # Tampering with spatial.yaml after session.yaml binding must trigger SHA-256 mismatch
    save_yaml_atomic(
        cal_out / "session.yaml",
        {
            "spatial_sha256": "deadbeef" * 8,
            "T_table_vive": {"position": [0.0, 0.0, 0.0], "quaternion": [0.0, 0.0, 0.0, 1.0]},
        },
    )
    rep_tampered = verify_calibration_bundle(cal_out)
    assert not rep_tampered["passed"]
    assert any("sha256_mismatch" in r for r in rep_tampered["reasons"])
