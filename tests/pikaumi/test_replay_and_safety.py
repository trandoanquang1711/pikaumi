"""Tests for simulation replay, physical preflight, HOLD_LAST_SAFE policy bridge & AST safety invariants (Task 8)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from pikaumi.cli import main
from pikaumi.config import PACKAGE_DIR
from pikaumi.dataset.raw import generate_synthetic_bimanual_dataset
from pikaumi.real.base import (
    ArmStateMode,
    MockBimanualController,
    require_live_authorization,
)
from pikaumi.robots.registry import load_robot_profile


def test_replay_and_replay_joints_and_replay_real_dry_run(tmp_path: Path) -> None:
    ds_dir = generate_synthetic_bimanual_dataset(tmp_path / "raw_ds", num_episodes=1, frames_per_episode=15)

    rep_out = tmp_path / "replay_out"
    assert main(["replay", str(ds_dir), "--episode", "0", "--robot", "piper_pika", "--output-dir", str(rep_out)]) == 0
    assert (rep_out / "replay_piper_pika_ep0000.png").is_file()
    assert (rep_out / "replay_piper_pika_ep0000.json").is_file()

    joints_dir = tmp_path / "joints_piper_pika"
    assert (
        main(
            [
                "convert",
                str(ds_dir),
                "--output-dir",
                str(joints_dir),
                "--format",
                "robot-joints",
                "--robot",
                "piper_pika",
                "--allow-flagged-episodes",
            ]
        )
        == 0
    )

    repj_out = tmp_path / "replay_joints_out"
    assert main(["replay-joints", str(joints_dir), "--episode", "0", "--output-dir", str(repj_out)]) == 0
    assert (repj_out / "replay_joints_piper_pika_ep0000.png").is_file()

    # replay-real dry-run preflight
    preflight_json = tmp_path / "preflight.json"
    assert (
        main(
            [
                "replay-real",
                str(ds_dir),
                "--episode",
                "0",
                "--robot",
                "piper_pika",
                "--speed-scale",
                "0.4",
                "--json-out",
                str(preflight_json),
            ]
        )
        == 0
    )
    pf = json.loads(preflight_json.read_text(encoding="utf-8"))
    assert pf["dry_run"] is True
    assert pf["fault_mode"] == "HOLD_LAST_SAFE"

    # replay-real must also accept already-converted <robot>-joints datasets directly
    preflight_joints_json = tmp_path / "preflight_from_joints.json"
    assert (
        main(
            [
                "replay-real",
                str(joints_dir),
                "--episode",
                "0",
                "--robot",
                "piper_pika",
                "--speed-scale",
                "0.4",
                "--json-out",
                str(preflight_joints_json),
            ]
        )
        == 0
    )
    pf_j = json.loads(preflight_joints_json.read_text(encoding="utf-8"))
    assert pf_j["dry_run"] is True
    assert pf_j["fault_mode"] == "HOLD_LAST_SAFE"


def test_policy_bridge_and_hold_last_safe_state_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pikaumi.calibration.deployment as deployment_mod

    monkeypatch.setattr(deployment_mod, "TABLE_LOCAL_CALIB_DIR", tmp_path / "empty_local_calibs")
    # Live execution must be blocked without --operator-supported AND verified physical local table YAML
    with pytest.raises(PermissionError):
        require_live_authorization(robot="piper_pika", live=False, operator_supported=True)
    with pytest.raises(PermissionError):
        require_live_authorization(robot="piper_pika", live=True, operator_supported=False)
    with pytest.raises(PermissionError):
        require_live_authorization(robot="piper_pika", live=True, operator_supported=True)

    # Mock policy bridge runs cleanly across embodiments
    for rname in ("piper_pika", "openarmv1", "arx5"):
        out_json = tmp_path / f"pb_{rname}.json"
        assert main(["policy-bridge", "--robot", rname, "--steps", "6", "--fps", "30", "--out", str(out_json)]) == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["dry_run"] is True and payload["mock_robot"] is True

    # Verify HOLD_LAST_SAFE latches permanently on unhealthy input or jump without releasing torque
    prof = load_robot_profile("piper_pika")
    ctrl = MockBimanualController(prof)
    q_safe, w_safe, mode = ctrl.filter_command(prof.home_q, {"left": 0.06, "right": 0.06}, input_healthy=True)
    assert mode == ArmStateMode.DRY_RUN_MOCK

    # Inject huge jump -> must latch HOLD_LAST_SAFE and hold q_safe
    q_jump = q_safe + 1.5
    q_out, w_out, mode2 = ctrl.filter_command(q_jump, {"left": 0.0, "right": 0.0}, input_healthy=True)
    assert mode2 == ArmStateMode.HOLD_LAST_SAFE
    assert np.allclose(q_out, q_safe)
    assert w_out == w_safe


def test_ast_zero_disable_arm_or_reset_calls_in_real_package() -> None:
    """Firmware S-V1.8-9 safety check: `src/pikaumi/real` must never call DisableArm or Reset."""
    forbidden_calls = {"DisableArm", "ResetArm", "Reset"}
    real_dir = PACKAGE_DIR / "real"
    for py_file in sorted(real_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in forbidden_calls:
                    raise AssertionError(f"Forbidden hardware call {node.func.attr} in {py_file}")
                if isinstance(node.func, ast.Name) and node.func.id in forbidden_calls:
                    raise AssertionError(f"Forbidden hardware call {node.func.id} in {py_file}")
