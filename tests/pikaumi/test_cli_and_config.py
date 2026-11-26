"""Tests for `pikaumi` CLI, config loaders, SHA-256 provenance, and safety skill checks."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from pikaumi.cli import build_parser, main
from pikaumi.config import (
    ASSETS_DIR,
    CONFIGS_DIR,
    REPO_ROOT,
    load_rig_config,
    save_yaml_atomic,
    sha256_object,
)
from pikaumi.doctor import run_doctor_checks


def test_config_paths_and_sha256_determinism(tmp_path: Path) -> None:
    assert ASSETS_DIR.is_dir()
    assert CONFIGS_DIR.is_dir()

    rig = load_rig_config()
    assert "recording" in rig
    assert rig["recording"]["fps"] == 30

    h1 = sha256_object({"b": [1, 2, 3], "a": "pika"})
    h2 = sha256_object({"a": "pika", "b": [1, 2, 3]})
    assert h1 == h2 and len(h1) == 64

    yaml_out = tmp_path / "sample.yaml"
    save_yaml_atomic(yaml_out, {"robot": "piper_pika", "stroke_m": 0.10})
    assert yaml_out.is_file()


def test_doctor_and_safety_skill_present(tmp_path: Path) -> None:
    checks = run_doctor_checks()
    by_name = {c.name: c for c in checks}
    assert by_name["pika-piper-control-guard"].status == "OK"
    assert by_name["module:numpy"].status == "OK"
    assert by_name["module:scipy"].status == "OK"
    assert by_name["embodiment_profiles"].status == "OK"

    json_report = tmp_path / "doctor.json"
    rc = main(["doctor", "--json-out", str(json_report)])
    assert rc == 0
    payload = json.loads(json_report.read_text(encoding="utf-8"))
    assert payload["passed"] is True


def test_cli_parser_and_robots_subcommands(tmp_path: Path) -> None:
    parser = build_parser()
    assert parser.prog == "pika"

    assert main(["robots", "list"]) == 0
    info_json = tmp_path / "piper_pika_info.json"
    assert main(["robots", "info", "piper_pika", "--json-out", str(info_json)]) == 0
    info = json.loads(info_json.read_text(encoding="utf-8"))
    assert info["name"] == "piper_pika"
    assert info["gripper_max_width_m"] == 0.10


def test_pika_shell_launcher_executes() -> None:
    launcher = REPO_ROOT / "pika"
    assert launcher.is_file()
    res = subprocess.run([str(launcher), "--version"], capture_output=True, text=True, check=True)
    assert "pikaumi" in res.stdout
