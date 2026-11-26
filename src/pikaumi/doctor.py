"""Read-only environment, calibration, skill, and robot-asset readiness inspector (`pika doctor`)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
import shutil
import sys

from pikaumi.config import (
    ASSETS_DIR,
    CONFIGS_DIR,
    DEFAULT_DATA_CALIB_DIR,
    PIKA_LEGACY_DIR,
    QUALITY_CONFIG,
    REPO_ROOT,
    RIG_CONFIG,
    RIG_EXAMPLE_CONFIG,
    ROBOTS_CONFIG_DIR,
    save_json_atomic,
)


@dataclass(frozen=True)
class CheckResult:
    category: str
    name: str
    status: str  # "OK" | "WARN" | "FAIL"
    detail: str


def run_doctor_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    results.append(
        CheckResult(
            "python",
            "interpreter",
            "OK",
            f"{sys.executable} ({sys.version.split()[0]})",
        )
    )
    for mod, required in (
        ("numpy", True),
        ("scipy", True),
        ("yaml", True),
        ("cv2", True),
        ("pyarrow", False),
        ("av", False),
        ("h5py", False),
        ("pyroki", False),
    ):
        try:
            importlib.import_module(mod)
            results.append(CheckResult("python", f"module:{mod}", "OK", "importable"))
        except Exception:
            status = "FAIL" if required else "WARN"
            tier = "required" if required else "optional extra"
            results.append(CheckResult("python", f"module:{mod}", status, f"missing ({tier})"))

    for cmd in ("git", "docker", "uv"):
        found = shutil.which(cmd) or (
            str(Path.home() / ".local/bin/uv")
            if cmd == "uv" and (Path.home() / ".local/bin/uv").is_file()
            else None
        )
        results.append(
            CheckResult(
                "host",
                f"cmd:{cmd}",
                "OK" if found else "WARN",
                found or "not on PATH",
            )
        )

    skill_guard = REPO_ROOT / ".gemini" / "skills" / "pika-piper-control-guard" / "SKILL.md"
    results.append(
        CheckResult(
            "safety",
            "pika-piper-control-guard",
            "OK" if skill_guard.is_file() else "WARN",
            str(skill_guard),
        )
    )

    results.append(
        CheckResult(
            "config",
            "rig.yaml",
            "OK" if RIG_CONFIG.is_file() else "WARN",
            f"active={RIG_CONFIG}"
            if RIG_CONFIG.is_file()
            else f"local rig.yaml not created yet (template: {RIG_EXAMPLE_CONFIG})",
        )
    )
    results.append(
        CheckResult(
            "config",
            "quality.yaml",
            "OK" if QUALITY_CONFIG.is_file() else "WARN",
            str(QUALITY_CONFIG),
        )
    )

    robot_yamls = sorted(ROBOTS_CONFIG_DIR.glob("*.yaml")) if ROBOTS_CONFIG_DIR.is_dir() else []
    results.append(
        CheckResult(
            "robots",
            "embodiment_profiles",
            "OK" if robot_yamls else "WARN",
            f"{len(robot_yamls)} profile(s) in {ROBOTS_CONFIG_DIR}: "
            + ", ".join(p.stem for p in robot_yamls),
        )
    )
    results.append(
        CheckResult(
            "assets",
            "assets_dir",
            "OK" if ASSETS_DIR.is_dir() else "WARN",
            str(ASSETS_DIR),
        )
    )
    results.append(
        CheckResult(
            "legacy",
            "Pika_Data",
            "OK" if PIKA_LEGACY_DIR.is_dir() else "WARN",
            str(PIKA_LEGACY_DIR),
        )
    )
    results.append(
        CheckResult(
            "calibration",
            "data_calibration_dir",
            "OK" if DEFAULT_DATA_CALIB_DIR.is_dir() else "WARN",
            str(DEFAULT_DATA_CALIB_DIR),
        )
    )
    return results


def run_doctor_cli(json_out: str | None = None, strict: bool = False) -> int:
    checks = run_doctor_checks()
    print("=" * 72)
    print(f" pikaumi Doctor — Workspace & Readiness Check ({CONFIGS_DIR.parent})")
    print("=" * 72)
    failures = 0
    for item in checks:
        if item.status == "FAIL":
            failures += 1
        elif strict and item.status == "WARN":
            failures += 1
        print(f"  [{item.status:<4}] {item.category:<12} {item.name:<26} {item.detail}")
    if json_out:
        save_json_atomic(
            json_out,
            {
                "passed": failures == 0,
                "checks": [asdict(c) for c in checks],
            },
        )
    return 1 if failures > 0 else 0
