"""Offline unit tests for live ChArUco + Vive calibration observation capture."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pikaumi.calibration.capture import (
    CharucoBoardSpec,
    DockerRos1CharucoCaptureSource,
    capture_charuco_observations,
    detect_charuco_fisheye_pose,
    interpolate_pose7_pair,
    is_view_diverse,
    load_board_spec_from_configs,
)
from pikaumi.calibration.cli_handlers import import_legacy_calibration_dir
from pikaumi.cli import main
from pikaumi.config import save_yaml_atomic
from pikaumi.robots.utils import mat_to_pose7, pose7_to_mat


def _render_synthetic_fisheye_charuco(
    spec: CharucoBoardSpec,
    K: np.ndarray,
    D: np.ndarray,
    T_cam_board: np.ndarray,
    img_size: tuple[int, int] = (640, 480),
) -> np.ndarray:
    """Render an exact synthetic Kannala-Brandt fisheye view of the ChArUco board."""
    w_px, h_px = img_size
    dict_id = getattr(cv2.aruco, spec.dictionary)
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    board = cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        spec.square_length_m,
        spec.marker_length_m,
        dictionary,
    )
    ppm = 2000  # 2 px/mm texture
    tex_w = int(round(spec.squares_x * spec.square_length_m * ppm))
    tex_h = int(round(spec.squares_y * spec.square_length_m * ppm))
    tex = board.generateImage((tex_w, tex_h), marginSize=0)

    # Back-project every fisheye pixel ray onto the board plane z_board = 0
    u, v = np.meshgrid(np.arange(w_px, dtype=np.float64), np.arange(h_px, dtype=np.float64))
    pts = np.stack([u, v], axis=-1).reshape(-1, 1, 2)
    undist = cv2.fisheye.undistortPoints(pts, K, D.reshape(4, 1)).reshape(-1, 2)
    rays_cam = np.concatenate([undist, np.ones((len(undist), 1), dtype=np.float64)], axis=1)

    T_board_cam = np.linalg.inv(T_cam_board)
    R_bc = T_board_cam[:3, :3]
    t_bc = T_board_cam[:3, 3]
    dirs_b = rays_cam @ R_bc.T
    valid_dir = np.abs(dirs_b[:, 2]) > 1e-6
    lam = np.where(valid_dir, -t_bc[2] / np.where(valid_dir, dirs_b[:, 2], 1.0), -1.0)
    pts_b = t_bc[None, :] + lam[:, None] * dirs_b

    board_w_m = spec.squares_x * spec.square_length_m
    board_h_m = spec.squares_y * spec.square_length_m
    in_bounds = (
        (lam > 0.05)
        & (pts_b[:, 0] >= 0.0)
        & (pts_b[:, 0] < board_w_m)
        & (pts_b[:, 1] >= 0.0)
        & (pts_b[:, 1] < board_h_m)
    )
    map_x = np.full((h_px * w_px,), -1.0, dtype=np.float32)
    map_y = np.full((h_px * w_px,), -1.0, dtype=np.float32)
    map_x[in_bounds] = (pts_b[in_bounds, 0] * ppm - 0.5).astype(np.float32)
    map_y[in_bounds] = (pts_b[in_bounds, 1] * ppm - 0.5).astype(np.float32)
    gray = cv2.remap(
        tex,
        map_x.reshape(h_px, w_px),
        map_y.reshape(h_px, w_px),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=220,
    )
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_board_to_table_transform_centers_origin_and_points_z_up() -> None:
    spec = CharucoBoardSpec(squares_x=4, squares_y=5, square_length_m=0.045, marker_length_m=0.035)
    T_bt = spec.board_to_table_transform()
    # Origin of {Table} sits at exact geometric center of 4x5 board
    p_origin_in_board = (T_bt @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
    assert np.allclose(p_origin_in_board, [0.090, 0.1125, 0.0])
    # +X right unchanged, +Y forward maps to -Y board, +Z up maps to -Z board (out of page)
    assert np.allclose(T_bt[:3, 0], [1.0, 0.0, 0.0])
    assert np.allclose(T_bt[:3, 1], [0.0, -1.0, 0.0])
    assert np.allclose(T_bt[:3, 2], [0.0, 0.0, -1.0])
    assert np.linalg.det(T_bt[:3, :3]) == pytest.approx(1.0)


def test_detect_charuco_fisheye_pose_recovers_ground_truth() -> None:
    spec = CharucoBoardSpec()
    K = np.array([[204.0, 0.0, 320.0], [0.0, 205.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([-0.025, 0.08, -0.20, 0.14], dtype=np.float64)

    T_cb_gt = np.eye(4, dtype=np.float64)
    T_cb_gt[:3, :3] = Rotation.from_euler("xyz", [12.0, -10.0, 5.0], degrees=True).as_matrix()
    T_cb_gt[:3, 3] = np.array([-0.08, -0.10, 0.34], dtype=np.float64)

    bgr = _render_synthetic_fisheye_charuco(spec, K, D, T_cb_gt)
    det = detect_charuco_fisheye_pose(bgr, K, D, spec, min_corners=6, max_reprojection_px=2.5)
    assert det is not None
    assert det.num_corners >= 8
    assert det.reprojection_rms_px < 1.5

    T_cb_est = pose7_to_mat(det.T_cam_board_pose7)
    pos_err_m = float(np.linalg.norm(T_cb_est[:3, 3] - T_cb_gt[:3, 3]))
    rot_err_deg = float(
        np.degrees(
            np.arccos(np.clip((np.trace(T_cb_est[:3, :3].T @ T_cb_gt[:3, :3]) - 1.0) * 0.5, -1.0, 1.0))
        )
    )
    assert pos_err_m < 0.003
    assert rot_err_deg < 1.0

    # Verify T_cam_table = T_cam_board @ T_board_table
    T_ct_expected = T_cb_est @ spec.board_to_table_transform()
    assert np.allclose(pose7_to_mat(det.T_cam_table_pose7), T_ct_expected, atol=1e-6)


def test_interpolate_pose7_pair_and_view_diversity() -> None:
    p0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    q1 = Rotation.from_euler("z", 20.0, degrees=True).as_quat()
    p1 = np.concatenate([[0.10, 0.0, 0.0], q1])

    mid = interpolate_pose7_pair(1000, p0, 10_001_000, p1, 5_001_000, max_gap_ns=25_000_000)
    assert mid is not None
    assert np.allclose(mid[:3], [0.05, 0.0, 0.0], atol=1e-6)
    ang = Rotation.from_quat(mid[3:7]).as_euler("xyz", degrees=True)[2]
    assert ang == pytest.approx(10.0, abs=1e-4)

    # Rejects stale gap
    assert interpolate_pose7_pair(0, p0, 60_000_000, p1, 30_000_000, max_gap_ns=25_000_000) is None

    # Diversity check
    assert is_view_diverse(p0, [], min_rot_deg=6.0, min_trans_m=0.025)
    assert not is_view_diverse(p0, [p0], min_rot_deg=6.0, min_trans_m=0.025)
    assert is_view_diverse(p1, [p0], min_rot_deg=6.0, min_trans_m=0.025)


def test_docker_charuco_source_command_and_end_to_end_hand_eye_and_session(tmp_path: Path) -> None:
    import shlex
    from pikaumi.real.teleop import DockerRos1PikaPoseSource

    src_l = DockerRos1CharucoCaptureSource(side="left", container_name="pika-ros")
    cmd_l = src_l.build_exec_command()
    inner_l = cmd_l[-1]
    assert "/root/pika_ros/install/setup.bash" in inner_l
    assert "/pika_pose_l" in inner_l
    assert "/pika_localization_status_l" in inner_l
    assert "/camera_fisheye_l/color/image_raw" in inner_l

    # Guard against json.dumps bash-quoting bug: extracted `-c` argument must compile as valid Python
    tokens_l = shlex.split(inner_l)
    c_idx = tokens_l.index("-c")
    compile(tokens_l[c_idx + 1], "<capture_bridge>", "exec")

    teleop_cmd = DockerRos1PikaPoseSource(container_name="pika-ros").build_exec_command()
    teleop_tokens = shlex.split(teleop_cmd[-1])
    t_idx = teleop_tokens.index("-c")
    compile(teleop_tokens[t_idx + 1], "<teleop_bridge>", "exec")

    src_r = DockerRos1CharucoCaptureSource(side="right", container_name="pika-ros")
    cmd_r = " ".join(src_r.build_exec_command())
    assert "/pika_pose_r" in cmd_r
    assert "/camera_fisheye_r/color/image_raw" in cmd_r

    # Create spatial.yaml with synthetic K, D
    K = np.array([[204.0, 0.0, 320.0], [0.0, 205.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([-0.025, 0.08, -0.20, 0.14], dtype=np.float64)
    spec = CharucoBoardSpec()
    spatial_yaml = tmp_path / "spatial.yaml"
    save_yaml_atomic(
        spatial_yaml,
        {
            "schema_version": 1,
            "kind": "pikaumi_spatial_calibration",
            "cameras": {
                "left": {
                    "K": K.tolist(),
                    "D": D.tolist(),
                    "image_width": 640,
                    "image_height": 480,
                    "board": spec.to_dict(),
                }
            },
            "mounts": {"left": {"T_tracker_cam": [0, 0, 0, 0, 0, 0, 1], "provisional": True}},
        },
    )

    loaded_spec = load_board_spec_from_configs(spatial_yaml=spatial_yaml, side="left")
    assert loaded_spec == spec

    # True hand-eye mount T_tracker_cam and fixed board T_vive_board
    T_tc_gt = np.eye(4, dtype=np.float64)
    T_tc_gt[:3, :3] = Rotation.from_euler("xyz", [5.0, -8.0, 3.0], degrees=True).as_matrix()
    T_tc_gt[:3, 3] = np.array([0.03, -0.02, 0.04], dtype=np.float64)

    T_vb_gt = np.eye(4, dtype=np.float64)
    T_vb_gt[:3, :3] = Rotation.from_euler("xyz", [175.0, 4.0, -6.0], degrees=True).as_matrix()
    T_vb_gt[:3, 3] = np.array([0.10, 0.25, -0.15], dtype=np.float64)

    # Synthesize 8 diverse views
    angles = [
        (0, 0, 0, -0.09, -0.11, 0.34),
        (16, -10, 8, -0.06, -0.13, 0.36),
        (-15, 12, -10, -0.11, -0.08, 0.35),
        (12, 16, 15, -0.07, -0.09, 0.38),
        (-14, -14, -12, -0.10, -0.12, 0.33),
        (20, 4, -18, -0.05, -0.10, 0.37),
        (-8, 18, 12, -0.12, -0.09, 0.35),
        (10, -18, -8, -0.08, -0.12, 0.39),
    ]
    frames = []
    for i, (rx, ry, rz, tx, ty, tz) in enumerate(angles):
        T_cb = np.eye(4, dtype=np.float64)
        T_cb[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        T_cb[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
        # T_vive_tracker @ T_tracker_cam @ T_cam_board = T_vive_board
        T_vt = T_vb_gt @ np.linalg.inv(T_cb) @ np.linalg.inv(T_tc_gt)
        pose_vt = mat_to_pose7(T_vt)
        bgr = _render_synthetic_fisheye_charuco(spec, K, D, T_cb)
        stamp_ns = (i + 1) * 100_000_000
        frames.append(
            {
                "image_stamp_ns": stamp_ns,
                "pose_prev_ns": stamp_ns - 2_000_000,
                "pose_prev": pose_vt.tolist(),
                "pose_next_ns": stamp_ns + 2_000_000,
                "pose_next": pose_vt.tolist(),
                "accurate": True,
                "image_bgr": bgr,
            }
        )

    # 1. Capture hand-eye observations and solve via CLI
    he_json = tmp_path / "obs_handeye_left.json"
    capture_charuco_observations(
        mode="hand-eye",
        side="left",
        spatial_yaml=spatial_yaml,
        output_json=he_json,
        target_views=8,
        preview=False,
        frame_iterable=frames,
    )
    he_yaml = tmp_path / "hand_eye_left.yaml"
    rc = main(
        [
            "calibrate",
            "hand-eye",
            "--observations-json",
            str(he_json),
            "--side",
            "left",
            "--spatial-yaml",
            str(spatial_yaml),
            "--output",
            str(he_yaml),
        ]
    )
    assert rc == 0

    # 2. Capture session observations and solve session.yaml via CLI
    ses_json = tmp_path / "obs_table.json"
    capture_charuco_observations(
        mode="session",
        side="left",
        spatial_yaml=spatial_yaml,
        output_json=ses_json,
        target_views=6,
        preview=False,
        frame_iterable=frames,
    )
    ses_yaml = tmp_path / "session.yaml"
    rc_ses = main(
        [
            "calibrate",
            "session",
            "--observations-json",
            str(ses_json),
            "--spatial-yaml",
            str(spatial_yaml),
            "--output",
            str(ses_yaml),
        ]
    )
    assert rc_ses == 0
    assert ses_yaml.is_file()


def test_headless_cv2_imshow_falls_back_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pikaumi.calibration.capture as cap_mod

    K = np.array([[204.0, 0.0, 320.0], [0.0, 205.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([-0.025, 0.08, -0.20, 0.14], dtype=np.float64)
    spec = CharucoBoardSpec()
    spatial_yaml = tmp_path / "spatial.yaml"
    save_yaml_atomic(
        spatial_yaml,
        {
            "schema_version": 1,
            "kind": "pikaumi_spatial_calibration",
            "cameras": {
                "left": {
                    "K": K.tolist(),
                    "D": D.tolist(),
                    "image_width": 640,
                    "image_height": 480,
                    "board": spec.to_dict(),
                }
            },
            "mounts": {"left": {"T_tracker_cam": [0, 0, 0, 0, 0, 0, 1], "provisional": True}},
        },
    )

    frames = []
    for i, (rx, ry, rz) in enumerate([(0, 0, 0), (14, -10, 6), (-12, 12, -8), (8, 15, 10)]):
        T_cb = np.eye(4, dtype=np.float64)
        T_cb[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        T_cb[:3, 3] = np.array([-0.08, -0.10, 0.35], dtype=np.float64)
        pose_vt = mat_to_pose7(np.linalg.inv(T_cb))
        bgr = _render_synthetic_fisheye_charuco(spec, K, D, T_cb)
        stamp_ns = (i + 1) * 100_000_000
        frames.append(
            {
                "image_stamp_ns": stamp_ns,
                "pose_prev_ns": stamp_ns - 1_000_000,
                "pose_prev": pose_vt.tolist(),
                "pose_next_ns": stamp_ns + 1_000_000,
                "pose_next": pose_vt.tolist(),
                "accurate": True,
                "image_bgr": bgr,
            }
        )

    class _FakeDockerSrc:
        def __init__(self, **_: object) -> None:
            pass

        def __enter__(self) -> "_FakeDockerSrc":
            return self

        def __exit__(self, *_: object) -> None:
            pass

        def iter_frames(self):
            yield from frames

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(cap_mod, "DockerRos1CharucoCaptureSource", _FakeDockerSrc)
    monkeypatch.setattr(
        cv2,
        "imshow",
        lambda *a, **kw: (_ for _ in ()).throw(cv2.error("headless build: cvShowImage not implemented")),
    )

    logs: list[str] = []
    out_json = tmp_path / "obs_headless.json"
    payload = capture_charuco_observations(
        mode="session",
        side="left",
        spatial_yaml=spatial_yaml,
        output_json=out_json,
        target_views=3,
        min_interval_s=0.0,
        preview=True,
        progress_cb=logs.append,
    )
    assert payload["num_views"] == 3
    assert any("OpenCV GUI (cv2.imshow) is unavailable" in line for line in logs)


def test_stationary_motion_gate_rejects_moving_frames(tmp_path: Path) -> None:
    K = np.array([[204.0, 0.0, 320.0], [0.0, 205.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.array([-0.025, 0.08, -0.20, 0.14], dtype=np.float64)
    spec = CharucoBoardSpec()
    spatial_yaml = tmp_path / "spatial.yaml"
    save_yaml_atomic(
        spatial_yaml,
        {
            "schema_version": 1,
            "kind": "pikaumi_spatial_calibration",
            "cameras": {
                "left": {
                    "K": K.tolist(),
                    "D": D.tolist(),
                    "image_width": 640,
                    "image_height": 480,
                    "board": spec.to_dict(),
                }
            },
            "mounts": {"left": {"T_tracker_cam": [0, 0, 0, 0, 0, 0, 1], "provisional": True}},
        },
    )

    angles = [(0, 0, 0), (14, -10, 6), (-12, 12, -8), (8, 15, 10), (-10, -14, 5)]
    frames = []
    for i, (rx, ry, rz) in enumerate(angles):
        T_cb = np.eye(4, dtype=np.float64)
        T_cb[:3, :3] = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_matrix()
        T_cb[:3, 3] = np.array([-0.08, -0.10, 0.35], dtype=np.float64)
        pose_vt = mat_to_pose7(np.linalg.inv(T_cb))
        bgr = _render_synthetic_fisheye_charuco(spec, K, D, T_cb)
        stamp_ns = (i + 1) * 100_000_000
        # First 2 frames are moving (15 mm / 5 deg), last 3 are stationary (0.8 mm / 0.3 deg)
        is_moving = i < 2
        frames.append(
            {
                "image_stamp_ns": stamp_ns,
                "pose_prev_ns": stamp_ns - 1_000_000,
                "pose_prev": pose_vt.tolist(),
                "pose_next_ns": stamp_ns + 1_000_000,
                "pose_next": pose_vt.tolist(),
                "stationary_pos_mm": 15.0 if is_moving else 0.8,
                "stationary_rot_deg": 5.0 if is_moving else 0.3,
                "stationary_samples": 12,
                "accurate": True,
                "image_bgr": bgr,
            }
        )

    out_json = tmp_path / "obs_stationary.json"
    payload = capture_charuco_observations(
        mode="session",
        side="left",
        spatial_yaml=spatial_yaml,
        output_json=out_json,
        target_views=5,
        max_motion_mm=4.0,
        max_motion_deg=2.0,
        preview=False,
        frame_iterable=frames,
    )
    assert payload["num_views"] == 3


