"""Check FPS timing, saved processing settings and sensor startup overrides."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_tools"))
from pika_options import FPS_CHOICES, SESSION_DEFAULTS
from pika_session import Session
from prepare_pika_dataset import timestamp_grid


@pytest.mark.parametrize("fps", FPS_CHOICES)
def test_timestamp_grid_has_no_epoch_precision_loss_or_accumulated_rate_error(fps):
    start = 1794022000123456789
    grid = timestamp_grid(start, start + 1000 * 10**9, fps)
    assert len(grid) == 1000 * fps + 1
    assert grid[-1] - grid[0] == 1000 * 10**9
    expected = np.rint(np.arange(len(grid)) * (1e9 / fps)).astype(np.int64)
    np.testing.assert_array_equal(grid - start, expected)


def test_session_remembers_settings_and_rejects_an_incompatible_resume(tmp_path):
    raw = tmp_path / "raw"
    directory = tmp_path / "session"
    directory.mkdir()
    settings = {"fps": 10, "image_size": [320, 240], "min_segment_seconds": 0.5,
                "repo_id": "local/towel_test"}
    original = Session(directory, raw, settings)
    original.write_manifest([{"episode": 9, "instruction": "fold a towel"}])
    reopened = Session(directory, raw)
    assert reopened.settings == {**SESSION_DEFAULTS, **settings}
    assert reopened.saved == original.saved
    with pytest.raises(ValueError, match="fps=10"):
        Session(directory, raw, {"fps": 30})


def test_legacy_session_gets_the_original_processing_defaults(tmp_path):
    raw = tmp_path / "raw"
    (tmp_path / "session.json").write_text(json.dumps({"raw_root": str(raw), "episodes": []}))
    session = Session(tmp_path, raw)
    assert session.settings == SESSION_DEFAULTS


def test_session_passes_custom_processing_flags_to_all_stages(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    directory = tmp_path / "session"
    directory.mkdir()
    session = Session(directory, raw, {"fps": 60, "image_size": [320, 240],
                                      "max_speed_m_s": 1.0, "repo_id": "local/my_towels"})
    session.write_manifest([{"episode": 9, "instruction": "fold a towel"}])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("pika_session.subprocess.run", run)
    session.convert(lambda progress: None)
    assert calls[0][calls[0].index("--fps") + 1] == "60"
    image_index = calls[0].index("--image-size")
    assert calls[0][image_index + 1:image_index + 3] == ["320", "240"]
    assert str(directory / "source/episode9/pika_source_60hz.hdf5") in calls[1]
    assert calls[1][calls[1].index("--max-speed-m-s") + 1] == "1.0"
    assert calls[2][calls[2].index("--repo-id") + 1] == "local/my_towels"


@pytest.mark.parametrize("fps,wrist_camera", [(15, "fisheye"), (60, "d405"), (90, "d405")])
def test_camera_override_preserves_startup_paths_and_only_changes_requested_rate(tmp_path, fps, wrist_camera):
    spec = importlib.util.spec_from_file_location("pika_sensor_options", ROOT / "scripts/start_pika_sensors.py")
    sensor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sensor)
    path = tmp_path / "installed scripts/start_multi_sensor.bash"
    path.parent.mkdir()
    original = ('SCRIPT_DIR=$(dirname "$(readlink -f "$0")")\n'
                'camera_fps=30\n'
                'serial=260522277249\n'
                'printf "%s|%s|%s" "$camera_fps" "$SCRIPT_DIR" "$serial"\n')
    path.write_text(original)
    script = sensor.configured_script(path, fps, wrist_camera=wrist_camera)
    result = subprocess.run(["bash", "-c", script, str(path)], check=True, capture_output=True, text=True)
    assert result.stdout == f"{fps}|{path.parent}|260522277249"
    assert path.read_text() == original
    with pytest.raises(ValueError, match="15, 30, 60, or 90"):
        sensor.configured_script(path, 20)
    with pytest.raises(ValueError, match="60 requires --wrist-camera d405"):
        sensor.configured_script(path, 60)


@pytest.mark.parametrize("fps", [None, 15, 30])
def test_startup_applies_both_serials_even_without_an_fps_override(tmp_path, fps):
    spec = importlib.util.spec_from_file_location("pika_sensor_serials", ROOT / "scripts/start_pika_sensors.py")
    sensor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sensor)
    path = tmp_path / "startup.bash"
    original = ('camera_fps=30\nl_depth_camera_no=111111111111\nr_depth_camera_no="260522279678"\n'
                'printf "%s|%s|%s" "$camera_fps" "$l_depth_camera_no" "$r_depth_camera_no"\n')
    path.write_text(original)
    script = sensor.configured_script(path, fps, {"left": "260522277249", "right": "260622272056"})
    result = subprocess.run(["bash", "-c", script], check=True, capture_output=True, text=True)
    assert result.stdout == f"{fps or 30}|260522277249|260622272056"
    assert path.read_text() == original
    with pytest.raises(ValueError, match="strings of digits"):
        sensor.configured_script(path, fps, {"right": "123; exit 7"})
    path.write_text("camera_fps=30\n")
    with pytest.raises(ValueError, match="r_depth_camera_no"):
        sensor.configured_script(path, fps, {"right": "260622272056"})


@pytest.mark.parametrize("arguments", [
    ["session", "--fps", "4"], ["session", "--fps", "91"],
    ["record", "9", "--fps", "91"],
    ["export", "9", "--image-size", "0", "480"],
    ["session", "--max-image-offset-ms", "nan"],
    ["prepare", "9", "--max-speed-m-s", "-1"],
    ["status", "--timeout", "inf"], ["sensors", "--camera-fps", "20"],
    ["sensors", "--camera-fps", "60"],
    ["sensors", "--camera-fps", "90"],
])
def test_invalid_flags_fail_before_touching_docker(arguments, monkeypatch):
    import pika_data
    monkeypatch.setattr(pika_data.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected process"))
    with pytest.raises(SystemExit):
        pika_data.main(arguments)
