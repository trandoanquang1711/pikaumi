"""Per-task camera selection stays local to capture/preflight and survives resume."""
import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_tools"))
from pika_session import CaptureClient, Session
from test_pika_data_cli import cli


@pytest.mark.parametrize("flag,enabled", [(None, False), ("--use-head", True), ("--no-use-head", False)])
def test_record_flag_reaches_external_recorder_environment(tmp_path, monkeypatch, flag, enabled):
    import pika_storage
    @contextmanager
    def storage(*args):
        yield "external-recorder", "/capture"
    monkeypatch.setattr(pika_storage, "capture_storage", storage)
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    args = ["record", "12", "pick and place", "--data-root", str(tmp_path)]
    cli.main(args + ["--wrist-camera", "d405"] + ([flag] if flag else []))
    assert calls[-1][1]["env"]["PIKA_USE_HEAD"] == str(enabled).lower()
    assert calls[-1][1]["env"]["PIKA_WRIST_CAMERA"] == "d405"
    assert calls[-1][1]["env"]["PIKA_CONTAINER"] == "external-recorder"


@pytest.mark.parametrize("enabled", [False, True])
def test_session_uses_same_flag_for_preflight_and_capture(tmp_path, monkeypatch, enabled):
    from pika_options import head_arguments
    client = CaptureClient("sensors", "/capture", "/ros", tmp_path / "capture.log",
                           settings={"use_head": enabled, "wrist_camera": "d405"})
    checks, starts = [], []
    monkeypatch.setattr("pika_session.subprocess.run", lambda cmd, **kw: checks.append(cmd) or SimpleNamespace(returncode=0))
    monkeypatch.setattr("pika_session.subprocess.Popen", lambda cmd, **kw: starts.append(cmd) or SimpleNamespace())
    monkeypatch.setattr(client, "response", lambda: "")
    monkeypatch.setattr(client, "call", lambda *a, **kw: None)
    client.start(1, "fold towel")
    assert head_arguments(enabled)[0] in checks[0]
    assert checks[0][checks[0].index("--wrist-camera") + 1] == "d405"
    assert f"PIKA_USE_HEAD={str(enabled).lower()}" in starts[0]
    assert "PIKA_WRIST_CAMERA=d405" in starts[0]


def test_session_mode_is_persisted_and_cannot_change_on_resume(tmp_path):
    raw = tmp_path / "raw"
    session = Session(tmp_path, raw, {"use_head": True, "wrist_camera": "d405"})
    assert json.loads(session.manifest.read_text())["settings"]["use_head"] is True
    assert Session(tmp_path, raw).settings["use_head"] is True
    assert Session(tmp_path, raw).settings["wrist_camera"] == "d405"
    assert session.settings["max_image_skew_ms"] == 40
    with pytest.raises(ValueError, match="use_head=True"):
        Session(tmp_path, raw, {"use_head": False})
    with pytest.raises(ValueError, match="wrist_camera=d405"):
        Session(tmp_path, raw, {"wrist_camera": "fisheye"})


def test_legacy_head_session_infers_its_existing_mode(tmp_path):
    raw = tmp_path / "raw"
    episode = raw / "episode2"
    episode.mkdir(parents=True)
    (episode / "capture_metadata.json").write_text(json.dumps({"head_camera": {"enabled": True}}))
    directory = tmp_path / "session"
    directory.mkdir()
    (directory / "session.json").write_text(json.dumps({"raw_root": str(raw), "episodes": [{"episode": 2}]}))
    assert Session(directory, raw).settings["use_head"] is True


@pytest.mark.parametrize("requested,available", [("false", True), ("true", True), ("true", False)])
def test_capture_selects_local_profile_without_changing_global_ros_params(tmp_path, monkeypatch, requested, available):
    monkeypatch.setenv("PIKA_USE_HEAD", requested)
    tree = ast.parse((ROOT / "scripts/pika_capture_service.py").read_text())
    names = {"selected_head_profile", "wrist_camera_profile", "capture_launch", "write_capture_metadata"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    launch = next(node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "LAUNCH" for t in node.targets))
    profile = dict(enabled=available, enable_depth=False)
    namespace = {"json": json, "os": os, "rospy": SimpleNamespace(get_param=lambda *args: profile)}
    exec(compile(ast.Module(body=[launch, *nodes], type_ignores=[]), "capture", "exec"), namespace)
    if requested == "true" and not available:
        with pytest.raises(ValueError, match="sensors --use-head"):
            namespace["capture_launch"]()
    else:
        xml = namespace["capture_launch"]()
        if requested == "false":
            assert "headCamera" not in xml
            namespace["write_capture_metadata"](tmp_path)
            assert not json.loads((tmp_path / "capture_metadata.json").read_text())["head_camera"]["enabled"]
        else:
            camera = json.loads(ET.fromstring(xml).find("rosparam").text)["dataInfo"]["camera"]
            assert camera["color"]["names"][-1] == "headCamera"
    assert profile == dict(enabled=available, enable_depth=False)


@pytest.mark.parametrize("flag", ["--use-head", "--no-use-head"])
def test_sensors_passes_override_without_editing_yaml(monkeypatch, flag):
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    before = (ROOT / "config/sensors.yaml").read_text()
    cli.main(["sensors", flag])
    assert f"PIKA_USE_HEAD={str(flag == '--use-head').lower()}" in calls[0]
    assert (ROOT / "config/sensors.yaml").read_text() == before


@pytest.mark.parametrize("flag", ["--use-head", "--no-use-head"])
def test_status_forwards_mode_to_live_probe(monkeypatch, flag):
    calls = []
    monkeypatch.setattr(cli, "status", lambda *args: None)
    monkeypatch.setattr(cli, "export_python", lambda: "/python")
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    cli.main(["status", "--check-streams", flag, "--wrist-camera", "d405"])
    assert flag in calls[0]
    assert calls[0][calls[0].index("--wrist-camera") + 1] == "d405"


@pytest.mark.parametrize("flag", ["--use-head", "--no-use-head"])
def test_live_probe_receives_process_local_mode(monkeypatch, flag):
    import pika_sync
    from test_pika_sync import live_sample
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=json.dumps(live_sample()))
    monkeypatch.setattr(pika_sync.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["pika_sync", "--container", "pika", "--ros-root", "/ros",
                                     flag, "--wrist-camera", "d405"])
    pika_sync.main()
    assert f"PIKA_USE_HEAD={str(flag == '--use-head').lower()}" in calls[0]
    assert "PIKA_WRIST_CAMERA=d405" in calls[0]
