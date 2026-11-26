"""Failed captures can be removed without leaving reusable stale labels."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "data_tools"))
    import pika_session

    monkeypatch.setattr(pika_session, "SESSIONS", tmp_path / "sessions")
    raw, directory = tmp_path / "raw", tmp_path / "session"
    raw.mkdir()
    directory.mkdir()
    model = pika_session.Session(directory, raw)
    for episode in (10, 11):
        for parent in (raw, directory / "source", directory / "prepared"):
            path = parent / f"episode{episode}"
            path.mkdir(parents=True)
            (path / "sample").write_text(str(episode))
    model.write_manifest([{"episode": episode, "instruction": "fold a towel"} for episode in (10, 11)])
    return model


def test_discard_removes_only_target_and_updates_resume(session):
    from pika_session import Session

    assert session.discard(11) == []
    for parent in (session.raw_root, session.directory / "source", session.directory / "prepared"):
        assert not (parent / "episode11").exists()
        assert (parent / "episode10/sample").read_text() == "10"
    assert [item["episode"] for item in Session(session.directory, session.raw_root).saved] == [10]
    assert session.next_episode(11) == 11
    with pytest.raises(ValueError, match="was not found"):
        session.discard(11)


def test_manifest_failure_restores_all_episode_directories(session, monkeypatch):
    previous = session.manifest.read_text()
    def fail(episodes):
        raise OSError("disk full")
    monkeypatch.setattr(session, "write_manifest", fail)
    with pytest.raises(OSError, match="disk full"):
        session.discard(11)
    for parent in (session.raw_root, session.directory / "source", session.directory / "prepared"):
        assert (parent / "episode11/sample").read_text() == "11"
    assert session.manifest.read_text() == previous


def test_discard_rejects_active_capture_and_published_dataset(session):
    from pika_storage import episode_lock

    with episode_lock(session.raw_root, 11):
        with pytest.raises(ValueError, match="is in use"):
            session.discard(11)
    published = session.directory / "lerobot_2"
    published.mkdir()
    (published / "pika_source_reports.json").write_text(json.dumps([{"episode": "episode11"}]))
    with pytest.raises(ValueError, match="is already in"):
        session.discard(11)
    assert (session.raw_root / "episode11/sample").is_file()
    assert len(session.saved) == 2


def test_symlink_rejection_restores_already_staged_raw(session):
    from pika_storage import discard_paths

    link = session.directory / "symlink"
    link.symlink_to(session.raw_root / "episode10", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        discard_paths([session.raw_root / "episode11", link])
    assert (session.raw_root / "episode11/sample").is_file()
    assert (link / "sample").read_text() == "10"


def test_cleanup_failure_reports_quarantine_after_committing_manifest(session, monkeypatch):
    import pika_storage

    def fail(path):
        raise PermissionError("root owned")
    monkeypatch.setattr(pika_storage, "remove_directory", fail)
    errors = session.discard(11)
    assert len(errors) == 3
    assert all(".discarded-episode11-" in error for error in errors)
    assert [item["episode"] for item in session.saved] == [10]
    assert not (session.raw_root / "episode11").exists()
    assert list(session.raw_root.glob(".discarded-episode11-*"))


def test_cli_session_and_raw_only_discard(session, monkeypatch):
    import pika_data

    monkeypatch.delenv("PIKA_DATA_DIR", raising=False)
    monkeypatch.setattr(pika_data.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected Docker call"))
    pika_data.main(["discard", "episode11", "--session-dir", str(session.directory)])
    assert json.loads(session.manifest.read_text())["episodes"] == session.saved[:1]
    pika_data.main(["discard", "10", "--data-root", str(session.raw_root)])
    assert not (session.raw_root / "episode10").exists()
    # Raw-only mode does not silently edit some other session's manifest/cache.
    assert (session.directory / "prepared/episode10/sample").is_file()


def test_cli_refuses_deletion_while_gui_is_open(session):
    from pika_data import main
    from pika_session import capture_window_lock

    with capture_window_lock():
        with pytest.raises(ValueError, match="recording window is open"):
            main(["discard", "11", "--session-dir", str(session.directory)])
    assert (session.raw_root / "episode11/sample").is_file()


def test_client_releases_episode_lock_when_recorder_pipe_breaks(session):
    from pika_session import CaptureClient
    from pika_storage import episode_lock

    class BrokenInput:
        def write(self, value):
            raise BrokenPipeError()
    client = CaptureClient("sensors", "/raw", "/ros", session.directory / "capture.log", session.raw_root)
    client.active = True
    client.recording.enter_context(episode_lock(session.raw_root, 11))
    client.process = SimpleNamespace(stdin=BrokenInput())
    with pytest.raises(ValueError, match="Recorder exited"):
        client.call("discard")
    with episode_lock(session.raw_root, 11):
        pass


@pytest.mark.parametrize("episode", ["-1", "../episode11", "11/other"])
def test_discard_rejects_invalid_episode_before_touching_files(session, episode):
    from pika_data import main

    with pytest.raises(SystemExit):
        main(["discard", episode, "--data-root", str(session.raw_root)])
    assert (session.raw_root / "episode11/sample").is_file()
