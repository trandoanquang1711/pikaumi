"""Layer 5 Calibration: Robot-to-table deployment calibration (`scope: simulation|physical`, `verified: bool`)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from pikaumi.config import (
    TABLE_LOCAL_CALIB_DIR,
    TABLE_SIM_CALIB_DIR,
    load_yaml,
    save_yaml_atomic,
    sha256_file,
)
from pikaumi.robots.utils import mat_to_pose7, normalize_quat_xyzw


@dataclass(frozen=True)
class TableCalibration:
    robot: str
    scope: str  # "simulation" | "physical"
    verified: bool
    source: str
    T_robot_table_pose7: np.ndarray
    sha256: str
    path: Path


def load_table_calibration(
    robot: str,
    *,
    path: str | Path | None = None,
    require_physical: bool = False,
) -> TableCalibration:
    if path is not None:
        cfg_path = Path(path)
    else:
        local_cand = TABLE_LOCAL_CALIB_DIR / f"{robot}.yaml"
        sim_cand = TABLE_SIM_CALIB_DIR / f"{robot}.yaml"
        cfg_path = local_cand if local_cand.is_file() else sim_cand

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Table calibration YAML not found for robot={robot!r} ({cfg_path})")

    raw = load_yaml(cfg_path)
    scope = str(raw.get("scope", "simulation"))
    verified = bool(raw.get("verified", False))
    if require_physical and (scope != "physical" or not verified):
        raise PermissionError(
            f"Physical execution requires scope='physical' and verified=True in {cfg_path} "
            f"(got scope={scope!r}, verified={verified})"
        )

    cal = raw.get("calibration") or {}
    rft = cal.get("robot_from_table")
    if isinstance(rft, dict):
        pos = np.asarray(rft["position"], dtype=np.float64)
        quat = normalize_quat_xyzw(np.asarray(rft["quaternion"], dtype=np.float64))
        pose7 = np.concatenate([pos, quat])
    elif "T_robot_table" in raw:
        pose7 = np.asarray(raw["T_robot_table"], dtype=np.float64)
        pose7[3:7] = normalize_quat_xyzw(pose7[3:7])
    else:
        raise ValueError(f"Missing calibration.robot_from_table in {cfg_path}")

    return TableCalibration(
        robot=str(raw.get("robot", robot)),
        scope=scope,
        verified=verified,
        source=str(raw.get("source", cfg_path.name)),
        T_robot_table_pose7=pose7,
        sha256=sha256_file(cfg_path),
        path=cfg_path,
    )


def solve_table_from_three_contact_points(
    p_origin_robot: np.ndarray,
    p_x_axis_robot: np.ndarray,
    p_y_plane_robot: np.ndarray,
) -> np.ndarray:
    """Solve `T_robot_table` (`pose7`) from 3 stationary TCP tabletop contact points in robot base frame."""
    p0 = np.asarray(p_origin_robot, dtype=np.float64)
    px = np.asarray(p_x_axis_robot, dtype=np.float64)
    py = np.asarray(p_y_plane_robot, dtype=np.float64)

    x_hat = px - p0
    nx = float(np.linalg.norm(x_hat))
    if nx < 0.05:
        raise ValueError("Table +X contact point must be >= 50 mm from origin")
    x_hat = x_hat / nx

    y_raw = py - p0
    z_hat = np.cross(x_hat, y_raw)
    nz = float(np.linalg.norm(z_hat))
    if nz < 1e-4:
        raise ValueError("Table contact points are collinear")
    z_hat = z_hat / nz
    if z_hat[2] < 0.0:
        raise ValueError(
            "Inverted table normal (z_hat[2] < 0): check 3-point contact order (origin -> +X -> +Y)"
        )
    y_hat = np.cross(z_hat, x_hat)
    y_hat = y_hat / float(np.linalg.norm(y_hat))

    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = x_hat
    T[:3, 1] = y_hat
    T[:3, 2] = z_hat
    T[:3, 3] = p0
    return mat_to_pose7(T)


def save_physical_table_calibration(
    path: str | Path,
    *,
    robot: str,
    T_robot_table_pose7: np.ndarray,
    verified: bool,
    source: str = "contact_three_point",
    extra_metrics: dict[str, Any] | None = None,
) -> Path:
    pose = np.asarray(T_robot_table_pose7, dtype=np.float64)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pikaumi_robot_table_calibration",
        "robot": robot,
        "scope": "physical",
        "source": source,
        "verified": bool(verified),
        "calibration": {
            "frame_convention": "pose7=[x,y,z,qx,qy,qz,qw], meters, xyzw quaternion",
            "robot_from_table": {
                "position": Rotation.identity().apply(pose[:3]).tolist(),
                "quaternion": normalize_quat_xyzw(pose[3:7]).tolist(),
            },
        },
    }
    if extra_metrics:
        payload["metrics"] = extra_metrics
    return save_yaml_atomic(path, payload)
