"""Validate raw mount selection and cleanup when an external recorder fails."""

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def storage(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "data_tools"))
    import pika_storage
    return pika_storage


def test_host_mount_mapping_uses_path_boundaries_and_writable_mounts(storage):
    mounts = [dict(Type="bind", Source="/storage", Destination="/broad", RW=True),
              dict(Type="bind", Source="/storage/raw", Destination="/capture", RW=True),
              dict(Type="bind", Source="/storage/raw/private", Destination="/readonly", RW=False)]
    assert storage.container_directory(Path("/storage/raw/new batch"), mounts) == "/capture/new batch"
    assert storage.container_directory(Path("/storage2/raw"), mounts) is None
    assert storage.container_directory(Path("/storage/raw"), mounts[2:]) is None


def test_external_recorder_copies_stream_config_and_is_removed_after_failure(tmp_path, monkeypatch, storage):
    raw = tmp_path / "other disk" / "raw"
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "inspect":
            metadata = [True, [], "sha256:test-image", "host", "sensor-host"]
            return SimpleNamespace(stdout="\n".join(json.dumps(value) for value in metadata))
        if command[1] == "exec":
            return SimpleNamespace(stdout="http://127.0.0.1:11451")
        if command[1] == "cp" and command[-1] == "-":
            return SimpleNamespace(stdout=b"current stream configuration")
        if command[1] == "cp":
            assert kwargs["input"] == b"current stream configuration"
        return SimpleNamespace(stdout="", returncode=0)
    monkeypatch.setattr(storage.subprocess, "run", run)
    probes = []
    monkeypatch.setattr(storage, "check_shared_directory", lambda *args: probes.append(args))
    with pytest.raises(RuntimeError, match="recorder failed"):
        with storage.capture_storage(raw, "sensors", "/root/pika_ros") as (container, target):
            assert target == "/home/agilex/data"
            assert probes == [(raw, container, target)]
            raise RuntimeError("recorder failed")
    launch = next(command for command, _ in calls if command[1] == "run")
    assert launch[launch.index("--mount") + 1] == f"type=bind,source={raw},target=/home/agilex/data"
    assert "ROS_MASTER_URI=http://127.0.0.1:11451" in launch
    assert launch[-2:] == ["sha256:test-image", "infinity"]
    assert calls[-1][0] == ["docker", "rm", "--force", container]


def test_existing_raw_mount_reuses_sensor_container(tmp_path, monkeypatch, storage):
    raw = tmp_path / "batch"
    metadata = [True, [dict(Type="bind", Source=str(tmp_path), Destination="/data", RW=True)],
                "sha256:image", "host", "sensors"]
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="\n".join(json.dumps(value) for value in metadata))
    monkeypatch.setattr(storage.subprocess, "run", run)
    monkeypatch.setattr(storage, "check_shared_directory", lambda *args: None)
    with storage.capture_storage(raw, "sensors", "/root/pika_ros") as target:
        assert target == ("sensors", "/data/batch")
    assert len(calls) == 1


def test_record_flag_reaches_actual_recorder_environment(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "data_tools"))
    import pika_data
    import pika_storage

    raw = tmp_path / "raw disk"
    @contextmanager
    def capture(root, sensor_container, ros_root):
        assert root == raw
        yield "external-recorder", "/external raw"
    monkeypatch.setattr(pika_storage, "capture_storage", capture)
    calls = []
    monkeypatch.setattr(pika_data.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    pika_data.main(["record", "10", "fold a towel", "--data-root", str(raw)])
    command, kwargs = calls[0]
    assert command[-2:] == ["10", "fold a towel"]
    assert kwargs["env"]["PIKA_CONTAINER"] == "external-recorder"
    assert kwargs["env"]["PIKA_CONTAINER_DATA_DIR"] == "/external raw"
    assert kwargs["env"]["PIKA_DATA_DIR"] == str(raw)


def test_output_root_routes_source_labels_dataset_and_cache(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "data_tools"))
    import pika_data

    raw, output = tmp_path / "raw", tmp_path / "output disk"
    (raw / "episode9").mkdir(parents=True)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    for side in ("left", "right"):
        (calibration / f"{side}_fisheye_intrinsics.json").write_text("{}")
    monkeypatch.setattr(pika_data, "export_python", lambda: "/example/python")
    calls = []
    cache = []
    def run(command, **kwargs):
        calls.append(command)
        if command[1].endswith("prepare_pika_dataset.py"):
            directory = Path(command[command.index("--output-dir") + 1])
            directory.mkdir(parents=True)
            (directory / "pika_source_20hz.hdf5").touch()
        elif command[2] == "prepare":
            directory = Path(command[6])
            directory.mkdir(parents=True)
            for name in ("labels.npz", "report.json"):
                (directory / name).touch()
        else:
            env = kwargs["env"]
            cache.append(Path(env["HF_DATASETS_CACHE"]))
            assert cache[-1].is_relative_to(output)
            assert Path(env["TMPDIR"]).is_relative_to(output)
            assert Path(env["TMPDIR"]).is_dir()
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pika_data.subprocess, "run", run)
    pika_data.main(["export", "9", "--data-root", str(raw), "--output-root", str(output),
                    "--calibration-dir", str(calibration)])
    pika_data.main(["prepare", "9", "--data-root", str(raw), "--output-root", str(output)])
    pika_data.main(["lerobot", "9", "--output-root", str(output)])
    assert str(output / "source/episode9") in calls[0]
    assert str(output / "source/episode9/pika_source_20hz.hdf5") in calls[1]
    assert str(output / "prepared/episode9") in calls[1]
    assert calls[2][calls[2].index("--output-dir") + 1] == str(output / "lerobot")
    assert not cache[0].parent.exists()


def test_session_resume_reads_external_raw_root_without_docker(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "data_tools"))
    monkeypatch.delenv("PIKA_DATA_DIR", raising=False)
    import pika_data
    import pika_session

    raw = tmp_path / "external raw"
    directory = tmp_path / "saved session"
    directory.mkdir()
    (directory / "session.json").write_text(json.dumps({"raw_root": str(raw), "episodes": []}))
    opened = []
    monkeypatch.setattr(pika_session, "open_session", lambda *args: opened.append(args))
    monkeypatch.setattr(pika_data.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected Docker call"))
    pika_data.main(["session", "--session-dir", str(directory)])
    assert opened[0][-1] == raw
