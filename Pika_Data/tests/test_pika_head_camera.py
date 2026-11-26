"""Offline D435i launch, dropout, calibration and three-view regression tests."""
import ast
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "data_tools"))
from start_pika_sensors import head_launch
from pika_sync import analyze
from prepare_pika_dataset import sample_head
from test_pika_sync import live_sample


def test_serial_validation_and_launch_lifecycle(tmp_path):
    launch = tmp_path / "installed.launch"
    launch.write_text('<launch><arg name="camera_fps" default="30"/></launch>')
    for serial in ("", 12345, "123;exit"):
        with pytest.raises(ValueError, match="serial"):
            head_launch(launch, dict(enabled=True, serial=serial))
    root = ET.fromstring(head_launch(launch, dict(enabled=True, serial="317622076044", enable_depth=True)))
    includes = root.findall("include")
    assert len(includes) == 2
    params = {arg.get("name"): arg.get("value") for arg in includes[1]}
    assert params["serial_no"] == "317622076044"
    assert params["align_depth"] == "true"
    assert params["enable_pointcloud"] == "false"
    assert params["enable_gyro"] == "false"
    assert json.loads(root.find("rosparam").text)["enabled"]
    assert len(ET.fromstring(head_launch(launch, {})).findall("include")) == 1


def test_capture_profile_and_required_head_files(tmp_path):
    # Load the pure launch/save helpers without ROS dependencies.
    tree = ast.parse((ROOT / "scripts/pika_capture_service.py").read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {"selected_head_profile", "capture_launch", "write_capture_metadata"}]
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "LAUNCH" for t in n.targets))
    namespace = {"json": json, "rospy": SimpleNamespace(get_param=lambda *a: dict(enabled=True, enable_depth=True))}
    exec(compile(ast.Module(body=[assignment, *nodes], type_ignores=[]), "capture", "exec"), namespace)
    root = ET.fromstring(namespace["capture_launch"]())
    cameras = json.loads(root.find("rosparam").text)["dataInfo"]["camera"]
    assert cameras["color"]["names"][-1] == "headCamera"
    assert cameras["depth"]["topics"] == ["/camera_head/aligned_depth_to_color/image_raw"]
    with pytest.raises(ValueError, match="head camera"):
        namespace["write_capture_metadata"](tmp_path)
    for kind in ("color", "depth"):
        directory = tmp_path / "camera" / kind / "headCamera"
        directory.mkdir(parents=True)
        (directory / "config.json").write_text('{}')
        (directory / "1.000000.png").touch()
    namespace["write_capture_metadata"](tmp_path)
    assert json.loads((tmp_path / "capture_metadata.json").read_text())["head_camera"]["enabled"]


def test_head_pairing_never_uses_future_and_rejects_dropout():
    grid = np.array([100, 200, 300])
    index, offsets, valid = sample_head(np.array([80, 105, 180, 400]), grid,
                                       np.array([[90, 95], [190, 195], [290, 295]]), 40)
    assert index.tolist() == [0, 2, 2]
    assert valid.tolist() == [True, True, False]
    assert (offsets[valid] <= 0).all()


@pytest.mark.parametrize("fault", [None, "missing", "clock", "dropout"])
def test_live_head_preflight(fault):
    data = live_sample()
    rows = np.array(data["streams"]["left_image"], dtype=np.int64)
    rows += 12_000_000
    if fault == "missing":
        rows = rows[:0]
    elif fault == "clock":
        rows[:, 0] -= 10**9
    elif fault == "dropout":
        rows = rows[[0, -1]]
    data["streams"]["head_image"] = rows.tolist()
    assert analyze(data)["ready"] == (fault is None)


def test_head_hdf5_roundtrip_and_missing_required_camera(tmp_path):
    h5py = pytest.importorskip("h5py")
    from test_pika_pipeline import synthetic_episode, pipeline, vla
    episode = synthetic_episode(tmp_path)
    for kind, source in (("color", "camera/color/pikaDepthCamera_l"), ("depth", "camera/depth/pikaDepthCamera_l")):
        shutil.copytree(episode / source, episode / f"camera/{kind}/headCamera")
    (episode / "capture_metadata.json").write_text(json.dumps({"head_camera": dict(enabled=True, enable_depth=True, serial="317622076044")}))
    output = tmp_path / "export"
    pipeline.main(["--data-root", str(tmp_path), "--output-dir", str(output), "--episodes", "episode3", "--vista-only"])
    with h5py.File(output / "pika_source_20hz.hdf5") as file:
        group = file["source_episodes/episode3"]
        assert set(vla.dataset_cameras(group)) == {"observation.images.left_wrist", "observation.images.right_wrist", "observation.images.head"}
        image = group["images/head_rgb"]
        np.testing.assert_array_equal(image["frames"][0, 0, 0], [255, 0, 0])
        assert (image["offset_ns"][:] <= 0).all()
        np.testing.assert_array_equal(image.attrs["K"], [[4,0,4],[0,4,3],[0,0,1]])
        depth = group["images/head_depth"]
        assert depth["frames"].dtype == np.uint16
        assert depth["frames"][0].max() == 50000
        assert depth.attrs["depth_scale_m_per_unit"] == 0.001
    calibration = tmp_path / "tcp"
    calibration.mkdir()
    for side in ("left", "right"):
        (calibration / f"{side}_pika_tcp.json").write_text(json.dumps({"transform": np.eye(4).tolist()}))
    labels_dir = tmp_path / "labels"
    source = output / "pika_source_20hz.hdf5"
    vla.prepare(source, episode, calibration, labels_dir, min_segment_seconds=0.1)
    report = json.loads((labels_dir / "report.json").read_text())
    assert report["head_sync"]["rows_rejected"] == 0
    with h5py.File(source) as file, np.load(labels_dir / "labels.npz") as labels:
        group = file["source_episodes/episode3"]
        packaged = tmp_path / "packaged"
        packaged.mkdir()
        provenance = vla.write_dataset_provenance(packaged, group, labels["source_row"], report)
        with np.load(packaged / provenance["timestamps"]) as times:
            np.testing.assert_array_equal(times["timestamp_ns"], labels["timestamp_ns"])
            assert "head_timestamp_ns" in times
        info = json.loads((packaged / provenance["calibration"]).read_text())
        assert info["observation.images.head"]["image_size"] == [8, 6]
    # Preparation must recheck the head camera even if motion and wrists are valid.
    with h5py.File(source, "r+") as file:
        stamps = file["source_episodes/episode3/images/head_rgb/timestamp_ns"]
        stamps[:] = stamps[:] + 1_000_000_000
    with pytest.raises(ValueError, match="No contiguous segment"):
        vla.prepare(source, episode, calibration, tmp_path / "bad_labels", min_segment_seconds=0.1)
    shutil.rmtree(episode / "camera/color/headCamera")
    with pytest.raises(FileNotFoundError):
        pipeline.main(["--data-root", str(tmp_path), "--output-dir", str(tmp_path / "bad"), "--episodes", "episode3", "--vista-only"])
