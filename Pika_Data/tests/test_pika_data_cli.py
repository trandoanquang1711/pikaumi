"""Check Docker dispatch and offline data export without connecting to hardware."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_tools"))
spec = importlib.util.spec_from_file_location("pika_data", ROOT / "data_tools/pika_data.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def test_specific_docker_mount_and_path_boundary():
    mounts = [dict(Type="bind", Source="/host/root", Destination="/home"),
              dict(Type="bind", Source="/host/pika raw", Destination="/home/agilex/data"),
              dict(Type="bind", Source="/wrong", Destination="/home/agilex/database")]
    assert cli.data_root_from_mounts(mounts, "/home/agilex/data/recordings") == Path("/host/pika raw/recordings")
    with pytest.raises(ValueError):
        cli.data_root_from_mounts(mounts[1:], "/home/agilex/data-other")
    with pytest.raises(ValueError):
        cli.data_root_from_mounts(mounts, "/home/agilex/data/../other")


@pytest.mark.parametrize("value", ["../episode8", "8; reboot", "-1", "episode", "8/other"])
def test_reject_invalid_episode_before_subprocess(value, monkeypatch):
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("unexpected process"))
    with pytest.raises(SystemExit):
        cli.main(["export", value])


def test_record_dispatch_preserves_instruction_as_one_argument(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "resolve_data_root", lambda *args: tmp_path)
    monkeypatch.setattr(cli.subprocess, "run", lambda args, **kwargs: calls.append(args))
    instruction = 'fold towel; $(touch /tmp/not-a-command) "left"'
    cli.main(["record", "episode9", instruction])
    assert calls == [["bash", str(ROOT / "scripts/record_pika_episode.sh"), "multi", "9", instruction]]


def test_sensor_start_loads_existing_tracker_environment(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda args, **kwargs: calls.append(args))
    monkeypatch.setenv("PIKA_CONTAINER", "test-pika")
    monkeypatch.setenv("PIKA_ROS_ROOT", "/root/pika ros")
    cli.main(["sensors"])
    assert calls[0][:6] == ["docker", "exec", "-it", "test-pika", "bash", "-lic"]
    assert calls[0][-5:-3] == ["_", "/root/pika ros"]
    assert calls[0][-2] == ""  # No FPS override; still apply configured serials.
    assert calls[0][-1] == (ROOT / "config/sensors.yaml").read_text()


def test_export_explicit_root_needs_no_docker_and_wont_overwrite(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    (raw / "episode9").mkdir(parents=True)
    cal = tmp_path / "calibration"
    cal.mkdir()
    for side in ("left", "right"):
        (cal / f"{side}_fisheye_intrinsics.json").write_text("{}")
    destination = tmp_path / "output"
    monkeypatch.setattr(cli, "export_python", lambda: "/example/python")
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda args, **kwargs: calls.append(args))
    arguments = ["export", "9", "--data-root", str(raw), "--calibration-dir", str(cal),
                 "--output-dir", str(destination)]
    cli.main(arguments)
    assert calls == [["/example/python", str(ROOT / "data_tools/prepare_pika_dataset.py"),
                      "--data-root", str(raw), "--episodes", "episode9",
                      "--calibration-dir", str(cal), "--output-dir", str(destination),
                      "--fps", "20", "--max-image-offset-ms", "40.0", "--max-sensor-gap-ms", "30.0", "--vista-only"]]
    destination.mkdir()
    with pytest.raises(ValueError, match="will not be overwritten"):
        cli.main(arguments)
    assert len(calls) == 1


def test_export_reads_actual_docker_mount_when_unconfigured(tmp_path, monkeypatch):
    monkeypatch.delenv("PIKA_DATA_DIR", raising=False)
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout=json.dumps([
            dict(Type="bind", Source=str(tmp_path), Destination="/home/agilex/data")]))
    monkeypatch.setattr(cli.subprocess, "run", run)
    assert cli.resolve_data_root(None, "pika-ros", "/home/agilex/data") == tmp_path
    assert calls == [["docker", "inspect", "--format", "{{json .Mounts}}", "pika-ros"]]


def test_no_operation_only_shows_help(monkeypatch, capsys):
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("unexpected process"))
    cli.main([])
    assert "export" in capsys.readouterr().out


def test_session_convert_reuses_completed_episodes_on_retry(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "data_tools"))
    import pika_session

    raw = tmp_path / "raw"
    raw.mkdir()
    directory = tmp_path / "session"
    directory.mkdir()
    session = pika_session.Session(directory, raw)
    assert session.next_episode(10) == 10
    (raw / "episode10").mkdir()
    (raw / "episode11").mkdir()
    assert session.next_episode(10) == 12
    session.write_manifest([{"episode": n, "instruction": "fold a towel"} for n in (10, 11)])

    calls = []
    def run(command, **kwargs):
        calls.append(command[1:3])
        if command[1] == "export":
            output = Path(command[command.index("--output-dir") + 1])
            output.mkdir(parents=True)
            (output / "pika_source_20hz.hdf5").touch()
        if command[1] == "prepare":
            if command[2] == "11" and calls.count(["prepare", "11"]) == 1:
                return SimpleNamespace(returncode=1)
            output = Path(command[command.index("--prepared-root") + 1]) / f"episode{command[2]}"
            output.mkdir(parents=True)
            (output / "labels.npz").touch()
            (output / "report.json").touch()
        if command[1] == "lerobot":
            assert command[2:4] == ["10", "11"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pika_session.subprocess, "run", run)
    with pytest.raises(ValueError, match="episode11"):
        session.convert(lambda message: None)
    assert calls == [["export", "10"], ["prepare", "10"], ["export", "11"], ["prepare", "11"]]

    reopened = pika_session.Session(directory, raw)
    assert reopened.saved == session.saved
    assert reopened.convert(lambda message: None) == directory / "lerobot"
    assert calls[-2:] == [["prepare", "11"], ["lerobot", "10"]]


@pytest.mark.parametrize("command", ("piper-trial", "source-plan", "episode-plan"))
def test_retired_replay_commands_do_not_start_a_subprocess(command, monkeypatch):
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: pytest.fail("retired command must not start"))
    with pytest.raises(SystemExit) as exc:
        cli.main([command])
    assert exc.value.code == 2
