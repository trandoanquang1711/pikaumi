"""Portable path resolution, YAML loaders, and SHA-256 provenance fingerprinting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[1]


def _resolve_bundled_dir(name: str) -> Path:
    repo_candidate = REPO_ROOT / name
    if repo_candidate.is_dir():
        return repo_candidate
    wheel_candidate = PACKAGE_DIR / name
    return wheel_candidate


ASSETS_DIR = _resolve_bundled_dir("assets")
CONFIGS_DIR = _resolve_bundled_dir("configs")
PIKA_LEGACY_DIR = REPO_ROOT / "Pika_Data"
DEFAULT_DATA_CALIB_DIR = REPO_ROOT / "data" / "calibration"

RIG_CONFIG = CONFIGS_DIR / "rig.yaml"
RIG_EXAMPLE_CONFIG = CONFIGS_DIR / "rig.example.yaml"
QUALITY_CONFIG = CONFIGS_DIR / "quality.yaml"
ROBOTS_CONFIG_DIR = CONFIGS_DIR / "robots"
PIKA_TCP_CALIB_DIR = CONFIGS_DIR / "calibration" / "pika_tcp"
TABLE_SIM_CALIB_DIR = CONFIGS_DIR / "calibration" / "table" / "sim"
TABLE_LOCAL_CALIB_DIR = CONFIGS_DIR / "calibration" / "table" / "local"
USER_CACHE_DIR = Path.home() / ".cache" / "pikaumi"


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize any JSON-compatible nested object deterministically for SHA-256."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def sha256_object(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def save_yaml_atomic(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    tmp.replace(target)
    return target


def save_json_atomic(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    tmp.replace(target)
    return target


def load_rig_config(path: str | Path | None = None) -> dict[str, Any]:
    candidate = Path(path) if path is not None else RIG_CONFIG
    if candidate.is_file():
        return load_yaml(candidate)
    if RIG_EXAMPLE_CONFIG.is_file():
        return load_yaml(RIG_EXAMPLE_CONFIG)
    return {}
