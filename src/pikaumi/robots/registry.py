"""Declarative robot embodiment registry backed by `configs/robots/*.yaml`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pikaumi.config import REPO_ROOT, ROBOTS_CONFIG_DIR, load_yaml, sha256_file
from pikaumi.robots.loader import URDFModel, load_urdf_model
from pikaumi.robots.utils import rpy_xyz_to_pose7


@dataclass(frozen=True)
class GripperJointSpec:
    name: str
    actuated_index: int
    closed_value: float
    open_value: float


@dataclass(frozen=True)
class ArmSpec:
    side: str
    ee_link: str
    declared_joint_names: tuple[str, ...]
    arm_joint_names: tuple[str, ...]
    arm_actuated_indices: tuple[int, ...]
    gripper_joints: tuple[GripperJointSpec, ...]


@dataclass(frozen=True)
class RobotProfile:
    name: str
    kind: str
    config_path: Path
    sha256: str
    urdf_relpath: str
    urdf_path: Path
    pkg_root: Path
    urdf_model: URDFModel
    arms: dict[str, ArmSpec]
    home_q: np.ndarray
    posture_q: np.ndarray  # NaN = free joint
    gripper_max_width_m: float
    gripper_retarget_mode: str  # "physical-width" | "normalized"
    pika_to_ee_pose7: np.ndarray
    ik_weights: dict[str, float]
    replay_ik_weights: dict[str, float]
    replay_max_joint_delta: float
    max_reach_m: float
    manipulation_lock_joints: tuple[str, ...]
    locked_actuated_indices: tuple[int, ...]
    real_config: dict[str, Any]

    @property
    def active_sides(self) -> tuple[str, ...]:
        return tuple(s for s in ("left", "right") if s in self.arms)

    def map_gripper_width_to_q(self, q_base: np.ndarray, side: str, pika_width_m: float, source_max_m: float = 0.10) -> tuple[np.ndarray, float]:
        """Map Pika jaw width (m) onto the robot's URDF gripper joints and physical width (m)."""
        q_out = np.asarray(q_base, dtype=np.float64).copy()
        if side not in self.arms:
            return q_out, 0.0
        if self.gripper_retarget_mode == "normalized":
            opening = float(np.clip(pika_width_m / max(source_max_m, 1e-6), 0.0, 1.0))
            robot_width_m = opening * self.gripper_max_width_m
        else:
            robot_width_m = float(np.clip(pika_width_m, 0.0, self.gripper_max_width_m))
            opening = float(np.clip(robot_width_m / max(self.gripper_max_width_m, 1e-6), 0.0, 1.0))

        for gspec in self.arms[side].gripper_joints:
            q_out[gspec.actuated_index] = gspec.closed_value + opening * (gspec.open_value - gspec.closed_value)
        return q_out, robot_width_m

    def map_robot_width_to_pika(self, side: str, robot_width_m: float, source_max_m: float = 0.10) -> float:
        """Inverse of `map_gripper_width_to_q`: map physical robot width (m) back into Pika `[0, source_max_m]`."""
        if side not in self.arms:
            return 0.0
        w = float(np.clip(robot_width_m, 0.0, self.gripper_max_width_m))
        if self.gripper_retarget_mode == "normalized":
            opening = float(np.clip(w / max(self.gripper_max_width_m, 1e-6), 0.0, 1.0))
            return opening * float(source_max_m)
        return float(np.clip(w, 0.0, source_max_m))

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "sha256": self.sha256,
            "urdf": self.urdf_relpath,
            "num_actuated_urdf_joints": self.urdf_model.num_actuated,
            "active_sides": list(self.active_sides),
            "arms": {
                s: {
                    "ee_link": a.ee_link,
                    "arm_dof": len(a.arm_joint_names),
                    "arm_joints": list(a.arm_joint_names),
                    "gripper_joints": [g.name for g in a.gripper_joints],
                }
                for s, a in self.arms.items()
            },
            "gripper_max_width_m": self.gripper_max_width_m,
            "gripper_retarget_mode": self.gripper_retarget_mode,
            "num_visual_meshes": len(self.urdf_model.mesh_paths),
        }


def list_robot_names(config_dir: Path = ROBOTS_CONFIG_DIR) -> tuple[str, ...]:
    if not config_dir.is_dir():
        return ()
    return tuple(sorted(p.stem for p in config_dir.glob("*.yaml")))


def _resolve_asset_path(rel_str: str) -> Path:
    cand = REPO_ROOT / rel_str
    if cand.exists():
        return cand.resolve()
    from pikaumi.config import PACKAGE_DIR

    return (PACKAGE_DIR / rel_str).resolve()


def load_robot_profile(name: str, config_dir: Path = ROBOTS_CONFIG_DIR) -> RobotProfile:
    cfg_path = Path(config_dir) / f"{name}.yaml"
    if not cfg_path.is_file():
        avail = ", ".join(list_robot_names(config_dir))
        raise ValueError(f"Unknown robot {name!r}; available: {avail}")
    raw = load_yaml(cfg_path)
    digest = sha256_file(cfg_path)

    urdf_rel = str(raw["urdf"])
    pkg_rel = str(raw.get("pkg_root", Path(urdf_rel).parent))
    urdf_path = _resolve_asset_path(urdf_rel)
    pkg_root = _resolve_asset_path(pkg_rel)
    model = load_urdf_model(urdf_path, pkg_root=pkg_root)

    # Resolve home_q (YAML may either list full URDF actuated vector or active-arm subset)
    raw_home = raw.get("home_q")
    if isinstance(raw_home, dict):
        default_key = str(raw.get("default_home_pose") or next(iter(raw_home.keys())))
        home_list = list(raw_home[default_key])
    elif isinstance(raw_home, list):
        home_list = list(raw_home)
    else:
        home_list = [0.0] * model.num_actuated

    declared_arm_joints_ordered: list[str] = []
    arms_raw = raw.get("arms", {})
    for s in ("left", "right"):
        if s in arms_raw:
            declared_arm_joints_ordered.extend(arms_raw[s].get("joint_names", []))

    home_q = np.zeros(model.num_actuated, dtype=np.float64)
    if len(home_list) == model.num_actuated:
        home_q[:] = np.asarray(home_list, dtype=np.float64)
    elif len(home_list) == len(declared_arm_joints_ordered):
        for jn, val in zip(declared_arm_joints_ordered, home_list, strict=True):
            home_q[model.actuated_name_to_index[jn]] = float(val)
    else:
        n_copy = min(len(home_list), model.num_actuated)
        home_q[:n_copy] = np.asarray(home_list[:n_copy], dtype=np.float64)

    raw_posture = raw.get("posture_q")
    posture_q = np.full(model.num_actuated, np.nan, dtype=np.float64)
    if isinstance(raw_posture, list):
        vals = [np.nan if v is None else float(v) for v in raw_posture]
        if len(vals) == model.num_actuated:
            posture_q[:] = np.asarray(vals, dtype=np.float64)
        elif len(vals) == len(declared_arm_joints_ordered):
            for jn, val in zip(declared_arm_joints_ordered, vals, strict=True):
                posture_q[model.actuated_name_to_index[jn]] = val

    arms: dict[str, ArmSpec] = {}
    for side in ("left", "right"):
        if side not in arms_raw:
            continue
        ar = arms_raw[side]
        ee_link = str(ar["ee_link"])
        if ee_link not in model.links:
            raise ValueError(f"Robot {name} side {side} ee_link {ee_link!r} not in URDF links")
        jnames = tuple(str(x) for x in ar.get("joint_names", []))
        g_raw = ar.get("gripper_joints", [])
        g_names_set: set[str] = set()
        g_specs: list[GripperJointSpec] = []
        for g_item in g_raw:
            gname = str(g_item["name"])
            g_names_set.add(gname)
            g_idx = model.actuated_name_to_index[gname]
            closed_v = float(g_item.get("closed", 0.0))
            open_v = float(
                g_item["open"]
                if "open" in g_item and g_item["open"] is not None
                else model.upper_limits[g_idx]
            )
            g_specs.append(
                GripperJointSpec(
                    name=gname,
                    actuated_index=g_idx,
                    closed_value=closed_v,
                    open_value=open_v,
                )
            )
        arm_only = tuple(jn for jn in jnames if jn not in g_names_set)
        arm_indices = tuple(model.actuated_name_to_index[jn] for jn in arm_only)
        arms[side] = ArmSpec(
            side=side,
            ee_link=ee_link,
            declared_joint_names=jnames,
            arm_joint_names=arm_only,
            arm_actuated_indices=arm_indices,
            gripper_joints=tuple(g_specs),
        )

    manip_block = raw.get("manipulation") or {}
    raw_locks = raw.get("manipulation_lock_joints")
    if raw_locks is None:
        raw_locks = manip_block.get("lock_joints", [])
    lock_names = tuple(str(x) for x in (raw_locks or []))
    lock_indices = tuple(
        model.actuated_name_to_index[x] for x in lock_names if x in model.actuated_name_to_index
    )

    ik_w = {k: float(v) for k, v in (raw.get("ik_weights") or {}).items() if isinstance(v, (int, float))}
    replay_block = raw.get("replay") or {}
    rep_w = {
        k: float(v)
        for k, v in (replay_block.get("ik_weights") or {}).items()
        if isinstance(v, (int, float))
    }
    rep_max_delta = float(replay_block.get("max_joint_delta", ik_w.get("max_joint_delta", 0.30)))
    max_reach = float(ik_w.get("max_reach", 0.70))
    pika_rpy = raw.get("pika_to_ee_rpy", [0.0, 0.0, 0.0])

    return RobotProfile(
        name=name,
        kind=str(raw.get("kind", name)),
        config_path=cfg_path,
        sha256=digest,
        urdf_relpath=urdf_rel,
        urdf_path=urdf_path,
        pkg_root=pkg_root,
        urdf_model=model,
        arms=arms,
        home_q=home_q,
        posture_q=posture_q,
        gripper_max_width_m=float(raw.get("gripper_max_width_m", 0.080)),
        gripper_retarget_mode=str(replay_block.get("gripper_retarget", "physical-width")),
        pika_to_ee_pose7=rpy_xyz_to_pose7([0.0, 0.0, 0.0], pika_rpy),
        ik_weights=ik_w,
        replay_ik_weights=rep_w,
        replay_max_joint_delta=rep_max_delta,
        max_reach_m=max_reach,
        manipulation_lock_joints=lock_names,
        locked_actuated_indices=lock_indices,
        real_config=dict(raw.get("real") or {}),
    )
