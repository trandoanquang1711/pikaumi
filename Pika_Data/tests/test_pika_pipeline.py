"""Regression tests for sensor timing, geometry and the portable HDF5 contract."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

h5py = pytest.importorskip("h5py", reason="h5py is required for HDF5 export pipeline tests")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_tools"))


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pipeline = load("pika_pipeline", "data_tools/prepare_pika_dataset.py")
handeye = load("pika_handeye", "scripts/solve_right_handeye.py")
vla = load("pika_vla", "data_tools/prepare_pika_vla.py")


def test_nanosecond_epoch_precision():
    assert pipeline.timestamp_ns("1793350664.000217.json") == 1793350664000217000


def test_no_interpolation_across_dropouts_and_exact_sample_is_valid():
    left, right, alpha, valid = pipeline.brackets(
        np.array([0, 10, 100]), np.array([0, 5, 50, 100]), 20)
    assert valid.tolist() == [True, True, False, True]
    assert alpha[1] == .5
    assert pipeline.valid_runs(valid) == [(0, 2)]


def test_causal_camera_never_uses_future_frame():
    index, offsets, valid = pipeline.sample_images(
        np.array([10, 30, 50]), np.array([0, 20, 49, 50]), "causal", 25)
    assert valid.tolist() == [False, True, True, True]
    assert np.all(offsets[valid] <= 0)
    assert index.tolist() == [0, 0, 1, 2]


def test_action_rotation_uses_short_arc_and_gripper_metres():
    pose = np.zeros((2, 2, 7))
    q = Rotation.from_euler("z", [[179], [-179]], degrees=True).as_quat()
    pose[:, :, 3:] = q[:, None, :]
    action = pipeline.cartesian_actions(pose, np.array([[.08, .07], [.03, .02]]))
    assert action.shape == (1, 14)
    assert np.isclose(action[0, 5], np.deg2rad(2))
    assert np.isclose(action[0, 6], .03)
    assert np.isclose(action[0, 13], .02)


def test_handeye_full_transform_keeps_fixed_board():
    rng = np.random.default_rng(8)
    rx = Rotation.from_euler("xyz", [.2, -.3, .1]).as_matrix()
    tx = np.array([.04, -.02, .1])
    fixed_r = Rotation.from_euler("xyz", [.5, .2, -.8]).as_matrix()
    fixed_t = np.array([.4, .1, .3])
    for _ in range(20):
        rg = Rotation.random(random_state=rng).as_matrix()
        tg = rng.normal(0, .1, 3)
        rc = rx.T @ rg.T @ fixed_r
        tc = rx.T @ (rg.T @ (fixed_t - tg) - tx)
        r, t = handeye.board_in_base(rg, tg, rx, tx, rc, tc)
        np.testing.assert_allclose(r, fixed_r, atol=1e-12)
        np.testing.assert_allclose(t, fixed_t, atol=1e-12)


def test_handeye_pose_pairing_and_invalid_quaternion():
    poses = [(0, np.eye(3), np.zeros(3)),
             (10, Rotation.from_euler("z", 90, degrees=True).as_matrix(), np.ones(3))]
    r, t = handeye.interpolate_pose(poses, [0, 10], 5, 20)
    np.testing.assert_allclose(t, .5)
    assert np.isclose(Rotation.from_matrix(r).as_euler("xyz", degrees=True)[2], 45)
    assert handeye.interpolate_pose(poses, [0, 10], 5, 2) is None
    with pytest.raises(ValueError):
        handeye.quat_to_rotation(0, 0, 0, 0)


def test_handeye_rejects_unobservable_single_axis_motion():
    with pytest.raises(ValueError,match="non-parallel"):
        handeye.validate_excitation(Rotation.from_euler("z",[[0],[.1],[.2],[.3]]).as_matrix())
    handeye.validate_excitation(Rotation.from_euler("xyz",[[0,0,0],[.2,0,0],[0,.3,0]]).as_matrix())


def synthetic_episode(root, image_fps=30):
    episode = root / "episode3"
    episode.mkdir()
    (episode / "instructions.json").write_text(json.dumps({"instructions": ["fold a towel"]}))
    epoch = 1793350664
    def filename(ns, extension):
        return f"{epoch + ns // 10**9}.{(ns % 10**9) // 1000:06d}{extension}"
    for side, label in pipeline.SIDES.items():
        for relative in [f"localization/pose/{label}", f"gripper/encoder/{label}"]:
            p = episode / relative
            p.mkdir(parents=True)
            for ns in range(0, 200_000_001, 10_000_000):
                if "pose" in relative:
                    row = dict(x=ns / 1e9, y=0 if side == "left" else .5, z=.1,
                               roll=0, pitch=0, yaw=0)
                else:
                    row = dict(angle=1., distance=.08-ns/1e9*.1)
                (p / filename(ns, ".json")).write_text(json.dumps(row))
    for key, relative in pipeline.IMAGE_STREAMS.items():
        p = episode / relative
        p.mkdir(parents=True)
        (p / "config.json").write_text(json.dumps({
            "width":8,"height":6,"K":[4,0,4,0,4,3,0,0,1],"D":[0,0,0,0,0],
            "distortion_model":"plumb_bob"}))
        for ns in pipeline.timestamp_grid(0, 200_000_000, image_fps):
            if key.endswith("depth"):
                image = np.full((6, 8), 50000, dtype=np.uint16)
            else:
                image = np.full((6,8,3), [0,0,255], dtype=np.uint8)
            cv2.imwrite(str(p / filename(ns, ".png")), image)
    # Exercise fallback from an episode without local intrinsic JSON files.
    c = root / "episode1/calibration"
    c.mkdir(parents=True)
    for side in pipeline.SIDES:
        (c / f"{side}_fisheye_intrinsics.json").write_text(json.dumps({
            "model":"fisheye","image_width":8,"image_height":6,
            "K":[[4,0,4],[0,4,3],[0,0,1]],"D":[0,0,0,0]}))
    return episode


def test_portable_export_preserves_rgb_depth_and_transition_alignment(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    synthetic_episode(source)
    output = tmp_path / "processed"
    pipeline.main(["--data-root",str(source),"--output-dir",str(output),
                   "--episodes","episode3","--image-size","4","3"])
    assert not (output / "vista_replay").exists()
    report = json.loads((output / "report.json").read_text())
    assert report["training_ready"] is False
    assert report["quality_checks"]["tracking_status_checked"] is False
    # Relocate the HDF5 on its own: no image paths or VDS filenames may depend on its original folder.
    relocated = tmp_path / "relocated.hdf5"
    shutil.copyfile(output / "pika_source_20hz.hdf5", relocated)
    shutil.rmtree(output)
    shutil.rmtree(source)
    with h5py.File(relocated) as f:
        assert not f.attrs["training_ready"]
        demo = f["data/demo_0"]
        assert f["data"].attrs["total"] == 4
        assert demo["actions"].shape == (4,14)
        assert demo["obs/state"].shape == (4,16)
        assert demo["obs/left_fisheye_rgb"].shape == (4,3,4,3)
        np.testing.assert_array_equal(demo["obs/left_fisheye_rgb"][0,0,0], [255,0,0])
        assert demo["obs/left_depth"].dtype == np.uint16
        assert demo["obs/left_depth"][0].max() == 50000
        np.testing.assert_allclose(demo["next_obs/eef_pose"][:-1], demo["obs/eef_pose"][1:])
        np.testing.assert_allclose(demo["actions"][:,6], demo["next_obs/gripper_width"][:,0])
        # Shared-world geometry survives export; the right arm is not zeroed independently.
        np.testing.assert_allclose(demo["obs/eef_pose"][:,1,1], .5)
        assert not demo.attrs["robot_replay_validated"]
        assert demo["rewards"][:].sum() == 0
        assert demo["dones"][:].tolist() == [0,0,0,1]
        np.testing.assert_allclose(f["source_episodes/episode3/images/left_fisheye_rgb"].attrs["K"],
                                   [[2,0,2],[0,2,1.5],[0,0,1]])


def test_conflicting_calibration_requires_explicit_selection(tmp_path):
    episode = synthetic_episode(tmp_path)
    other = tmp_path / "episode2/calibration"
    other.mkdir(parents=True)
    d = json.loads((tmp_path/"episode1/calibration/left_fisheye_intrinsics.json").read_text())
    d["K"][0][0] = 5
    (other/"left_fisheye_intrinsics.json").write_text(json.dumps(d))
    with pytest.raises(ValueError, match="Conflicting"):
        pipeline.resolve_intrinsic(tmp_path,episode,"left")


@pytest.mark.parametrize("fps", [10, 30])
def test_custom_fps_reaches_hdf5_labels_and_quality_screen(tmp_path, fps):
    raw = tmp_path / "raw"
    raw.mkdir()
    episode = synthetic_episode(raw)
    exported = tmp_path / "export"
    pipeline.main(["--data-root", str(raw), "--episodes", "episode3", "--output-dir", str(exported),
                   "--vista-only", "--fps", str(fps), "--image-size", "4", "3"])
    source = exported / f"pika_source_{fps}hz.hdf5"
    calibration = tmp_path / "tcp"
    calibration.mkdir()
    for side in ("left", "right"):
        (calibration / f"{side}_pika_tcp.json").write_text(json.dumps({"transform": np.eye(4).tolist()}))
    output = tmp_path / "labels"
    vla.prepare(source, episode, calibration, output, min_segment_seconds=0.1)
    report = json.loads((output / "report.json").read_text())
    assert report["fps"] == fps
    assert report["min_segment_transitions"] == int(np.ceil(0.1 * fps))
    with np.load(output / "labels.npz") as labels:
        assert len(labels["action"]) == round(0.2 * fps)
        np.testing.assert_allclose(labels["action"][:, 0], 1 / fps, atol=1e-8)
    with pytest.raises(ValueError, match="No contiguous segment"):
        vla.prepare(source, episode, calibration, tmp_path / "rejected", min_segment_seconds=0.1,
                    max_speed_m_s=0.5)


def test_causal_wrist_pair_uses_the_earlier_pair_within_age_limit():
    left = np.array([0, 33, 66])
    right = np.array([2, 42, 75])
    grid = np.array([34, 41, 84])
    (li, lo, lv), (ri, ro, rv) = pipeline.sample_wrist_pair(left, right, grid, "causal", 40)
    np.testing.assert_array_equal(left[li], [0, 33, 66])
    np.testing.assert_array_equal(right[ri], [2, 2, 75])
    assert lv.all() and rv.all()
    assert ((-40 <= lo) & (lo <= 0) & (-40 <= ro) & (ro <= 0)).all()


@pytest.mark.parametrize("wrist_camera,fps", [("fisheye", 20), ("d405", 60), ("d405", 90)])
def test_vista_source_exports_with_only_wrist_cameras(tmp_path, wrist_camera, fps):
    source = tmp_path / "raw"
    source.mkdir()
    episode = synthetic_episode(source, image_fps=max(30, fps))
    for key, relative in pipeline.IMAGE_STREAMS.items():
        if f"_{wrist_camera}_rgb" not in key:
            shutil.rmtree(episode / relative)
    if wrist_camera == "d405":
        shutil.rmtree(source / "episode1")
        (episode / "capture_metadata.json").write_text(json.dumps({"wrist_camera": wrist_camera, "export_fps": fps}))
    output = tmp_path / "export"
    pipeline.main(["--data-root", str(source), "--output-dir", str(output),
                   "--episodes", episode.name, "--vista-only"])
    source_file = output / f"pika_source_{fps}hz.hdf5"
    with h5py.File(source_file) as file:
        assert file.attrs["fps"] == fps
        assert file.attrs["wrist_pairing"] == "causal_pair"
        assert file.attrs["max_image_skew_ms"] == (10 if wrist_camera == "fisheye" else 40)
        group = file[f"source_episodes/{episode.name}"]
        assert set(group["images"]) == {f"{side}_{wrist_camera}_rgb" for side in pipeline.SIDES}
        assert group.attrs["wrist_camera"] == wrist_camera
        assert file["data"].attrs["total"] == fps // 5
    calibration = tmp_path / "tcp"
    calibration.mkdir()
    for side in pipeline.SIDES:
        (calibration / f"{side}_pika_tcp.json").write_text(json.dumps({"transform": np.eye(4).tolist()}))
    prepared = tmp_path / "prepared"
    vla.prepare(source_file, episode, calibration, prepared, min_segment_seconds=0.1)
    report = json.loads((prepared / "report.json").read_text())
    assert report["cameras"]["observation.images.left_wrist"] == f"images/left_{wrist_camera}_rgb/frames"
    assert report["fps"] == fps


def test_motion_screen_rejects_stale_pose_and_its_recovery(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "prepare_pika_dataset", pipeline)
    times = np.arange(0, 601, 10, dtype=np.int64) * 1_000_000
    grid = times[::5]
    pose = np.zeros((len(times), 6))
    pose[:, 0] = times * 1e-9 * 0.01
    left_pose = pose.copy()
    left_pose[10:41] = left_pose[10]

    def read_pose(path, extensions, fields):
        return pipeline.Stream([], times, left_pose if path.name == "pika_l" else pose)

    monkeypatch.setattr(pipeline, "read_stream", read_pose)
    group = {f"sync/pose_{side}/{edge}_timestamp_ns": grid
             for side in pipeline.SIDES for edge in ("left", "right")}
    rejected, report = vla.motion_screen(group, tmp_path, grid)
    np.testing.assert_array_equal(np.flatnonzero(rejected), np.arange(2, 9))
    left = report["sides"]["left"]
    assert left["flagged_raw_intervals"] == []
    assert left["suspected_stale_pose_intervals"][0]["held_until_s"] == pytest.approx(0.4)
    assert report["sides"]["right"]["suspected_stale_pose_intervals"] == []


def test_tracking_screen_cuts_bracketed_actions_at_bad_status_and_gap(tmp_path):
    origin, ms = 1_794_000_000_000_000_000, 1_000_000
    times = origin + np.arange(0, 2101, 50, dtype=np.int64) * ms
    group = {f"sync/pose_{side}/{edge}_timestamp_ns": times + offset * ms
             for side in pipeline.SIDES for edge, offset in (("left", -5), ("right", 5))}
    # A left False arrives between observations. The right stream later goes quiet.
    left = [[origin + t * ms, t != 710] for t in range(0, 2101, 10)]
    right = [[origin + t * ms, True] for t in range(0, 2101, 10) if not 1100 < t < 1550]
    path = tmp_path / "tracking_status.json"
    path.write_text(json.dumps({"version": 1, "clock": "ros_receive_time_ns",
                               "sides": {"left": left, "right": right}}))
    rejected, report = vla.tracking_screen(group, tmp_path, times)
    kept_ms = (times[:-1][~rejected] - origin) // ms
    assert kept_ms.tolist() == [*range(300, 601, 50), *range(1000, 1201, 50), *range(1850, 2051, 50)]
    assert report["sides"]["left"]["inaccurate_samples"] == 1
    assert report["sides"]["right"]["max_status_gap_ms"] == 450

    path.write_text(json.dumps({"version": 1, "clock": "ros_receive_time_ns", "sides": {"left": left}}))
    rejected, _ = vla.tracking_screen(group, tmp_path, times)
    assert rejected.all()
    path.unlink()
    rejected, report = vla.tracking_screen(group, tmp_path, times)
    assert not rejected.any() and report["checked"] is False

