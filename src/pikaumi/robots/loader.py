"""General URDF parser supporting arbitrary joint axes, mimic joints, mesh resolution & capsules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from pikaumi.config import ASSETS_DIR


@dataclass(frozen=True)
class URDFJoint:
    name: str
    joint_type: str  # "revolute" | "continuous" | "prismatic" | "fixed"
    parent_link: str
    child_link: str
    origin_transform: np.ndarray  # (4, 4)
    axis: np.ndarray  # (3,) unit vector
    axis_skew: np.ndarray  # (3, 3) precomputed [axis]_x
    axis_skew_sq: np.ndarray  # (3, 3) precomputed [axis]_x^2
    lower: float
    upper: float
    velocity: float
    effort: float
    actuated_index: int | None  # index in `actuated_joint_names` if directly actuated
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@dataclass(frozen=True)
class LinkCapsule:
    """Link segment capsule between parent and child joint frames (in parent_link coordinates)."""

    link_name: str
    side: str  # "left" | "right" | "shared"
    p0_local: np.ndarray  # (3,)
    p1_local: np.ndarray  # (3,)
    radius: float


def _parse_vec3(raw: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not raw:
        return np.array(default, dtype=np.float64)
    arr = np.fromstring(raw.strip(), sep=" ", dtype=np.float64)
    if arr.size != 3:
        raise ValueError(f"Expected 3 floats in URDF attribute, got {raw!r}")
    return arr


def _resolve_mesh_path(filename: str, urdf_dir: Path, pkg_root: Path) -> Path:
    if filename.startswith("package://"):
        stripped = filename[len("package://") :]
        parts = stripped.split("/", 1)
        rel = parts[1] if len(parts) > 1 else parts[0]
        pkg_name = parts[0]
        candidates = [
            pkg_root / rel,
            pkg_root / stripped,
            ASSETS_DIR / pkg_name / rel,
            urdf_dir / rel,
            urdf_dir.parent / rel,
        ]
        # Handle package://piper_description/meshes/... inside assets/piper
        if pkg_name == "piper_description":
            candidates.insert(0, ASSETS_DIR / "piper" / rel)
        elif pkg_name == "pika_gripper_description":
            candidates.insert(0, ASSETS_DIR / "pika_gripper" / rel)
        for cand in candidates:
            if cand.is_file():
                return cand.resolve()
        return candidates[0]
    cand = (urdf_dir / filename).resolve()
    if cand.is_file():
        return cand
    cand_pkg = (pkg_root / filename).resolve()
    return cand_pkg


class URDFModel:
    """Self-contained kinematic tree and capsule collision approximation parsed from URDF XML."""

    def __init__(
        self,
        name: str,
        urdf_path: Path,
        root_link: str,
        links: tuple[str, ...],
        joints: tuple[URDFJoint, ...],
        actuated_joint_names: tuple[str, ...],
        mesh_paths: tuple[Path, ...],
        capsules: tuple[LinkCapsule, ...],
    ) -> None:
        self.name = name
        self.urdf_path = urdf_path
        self.root_link = root_link
        self.links = links
        self.joints = joints
        self.actuated_joint_names = actuated_joint_names
        self.actuated_name_to_index = {n: i for i, n in enumerate(actuated_joint_names)}
        self.mesh_paths = mesh_paths
        self.capsules = capsules

        self.joint_by_child: dict[str, URDFJoint] = {j.child_link: j for j in joints}
        self.joint_by_name: dict[str, URDFJoint] = {j.name: j for j in joints}
        self.children_by_parent: dict[str, list[URDFJoint]] = {lnk: [] for lnk in links}
        for j in joints:
            self.children_by_parent.setdefault(j.parent_link, []).append(j)

        # Topological traversal order from root_link
        order: list[URDFJoint] = []
        stack = list(self.children_by_parent.get(root_link, ()))
        while stack:
            j = stack.pop(0)
            order.append(j)
            stack.extend(self.children_by_parent.get(j.child_link, ()))
        self.topo_joints: tuple[URDFJoint, ...] = tuple(order)

        # Precompute ancestor chain for each link
        self.ancestors: dict[str, tuple[URDFJoint, ...]] = {root_link: ()}
        for j in self.topo_joints:
            self.ancestors[j.child_link] = (*self.ancestors[j.parent_link], j)

        actuated = [j for j in joints if j.actuated_index is not None]
        actuated.sort(key=lambda item: int(item.actuated_index))  # type: ignore[arg-type]
        self.lower_limits = np.array([j.lower for j in actuated], dtype=np.float64)
        self.upper_limits = np.array([j.upper for j in actuated], dtype=np.float64)
        self.velocity_limits = np.array([j.velocity for j in actuated], dtype=np.float64)

    @property
    def num_actuated(self) -> int:
        return len(self.actuated_joint_names)

    def _joint_scalar_value(self, joint: URDFJoint, q_actuated: np.ndarray) -> float:
        if joint.actuated_index is not None:
            return float(q_actuated[joint.actuated_index])
        if joint.mimic_joint is not None:
            ref = self.joint_by_name[joint.mimic_joint]
            if ref.actuated_index is not None:
                return float(joint.mimic_multiplier * q_actuated[ref.actuated_index] + joint.mimic_offset)
        return 0.0

    def forward_kinematics_all(
        self, q_actuated: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
        """Compute world transforms `T_W_link` for every link and `(origin_W, axis_W)` for every joint."""
        q = np.asarray(q_actuated, dtype=np.float64)
        eye3 = np.eye(3, dtype=np.float64)
        link_tf: dict[str, np.ndarray] = {self.root_link: np.eye(4, dtype=np.float64)}
        joint_frames: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for j in self.topo_joints:
            T_parent = link_tf[j.parent_link]
            T_pre = T_parent @ j.origin_transform
            origin_w = T_pre[:3, 3]
            axis_w = T_pre[:3, :3] @ j.axis

            if j.joint_type == "fixed":
                link_tf[j.child_link] = T_pre
            elif j.joint_type in ("revolute", "continuous"):
                val = self._joint_scalar_value(j, q)
                s_v = float(np.sin(val))
                c_v = 1.0 - float(np.cos(val))
                R_mot = eye3 + s_v * j.axis_skew + c_v * j.axis_skew_sq
                T_child = T_pre.copy()
                T_child[:3, :3] = T_pre[:3, :3] @ R_mot
                link_tf[j.child_link] = T_child
            elif j.joint_type == "prismatic":
                val = self._joint_scalar_value(j, q)
                T_child = T_pre.copy()
                T_child[:3, 3] = T_pre[:3, 3] + T_pre[:3, :3] @ (j.axis * val)
                link_tf[j.child_link] = T_child
            else:
                link_tf[j.child_link] = T_pre
            joint_frames[j.name] = (origin_w, axis_w)

        return link_tf, joint_frames

    def link_pose_and_spatial_jacobian(
        self,
        q_actuated: np.ndarray,
        link_name: str,
        active_indices: tuple[int, ...] | None = None,
        *,
        precomputed_fk: tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return `(T_W_link, J_pos, J_rot_world)` for `link_name`.

        `J_pos` and `J_rot_world` have shape `(3, len(cols))` where `cols` is `active_indices`
        (or all `num_actuated` joints when `active_indices is None`).
        """
        link_tf, joint_frames = (
            precomputed_fk if precomputed_fk is not None else self.forward_kinematics_all(q_actuated)
        )
        T_ee = link_tf[link_name]
        p_ee = T_ee[:3, 3]

        if active_indices is None:
            col_map = {idx: idx for idx in range(self.num_actuated)}
            ncols = self.num_actuated
        else:
            col_map = {act_idx: col_i for col_i, act_idx in enumerate(active_indices)}
            ncols = len(active_indices)

        J_pos = np.zeros((3, ncols), dtype=np.float64)
        J_rot = np.zeros((3, ncols), dtype=np.float64)

        for j in self.ancestors[link_name]:
            if j.joint_type == "fixed":
                continue
            mult = 1.0
            act_idx = j.actuated_index
            if act_idx is None and j.mimic_joint is not None:
                ref = self.joint_by_name[j.mimic_joint]
                act_idx = ref.actuated_index
                mult = j.mimic_multiplier
            if act_idx is None or act_idx not in col_map:
                continue
            col = col_map[act_idx]
            origin_w, axis_w = joint_frames[j.name]
            if j.joint_type in ("revolute", "continuous"):
                J_pos[:, col] += mult * np.cross(axis_w, p_ee - origin_w)
                J_rot[:, col] += mult * axis_w
            elif j.joint_type == "prismatic":
                J_pos[:, col] += mult * axis_w

        return T_ee, J_pos, J_rot


def _infer_side(link_name: str) -> str:
    low = link_name.lower()
    if "left" in low or low.startswith("izq_"):
        return "left"
    if "right" in low or low.startswith("der_"):
        return "right"
    return "shared"


def load_urdf_model(urdf_path: str | Path, pkg_root: str | Path | None = None) -> URDFModel:
    path = Path(urdf_path).resolve()
    root_pkg = Path(pkg_root).resolve() if pkg_root is not None else path.parent
    xml_root = ET.parse(path).getroot()
    robot_name = xml_root.attrib.get("name", path.stem)

    links = tuple(el.attrib["name"] for el in xml_root.findall("link"))
    child_links: set[str] = set()
    actuated_names: list[str] = []
    parsed_joints: list[URDFJoint] = []

    for j_el in xml_root.findall("joint"):
        jname = j_el.attrib["name"]
        jtype = j_el.attrib["type"]
        parent = j_el.find("parent").attrib["link"]  # type: ignore[union-attr]
        child = j_el.find("child").attrib["link"]  # type: ignore[union-attr]
        child_links.add(child)

        orig_el = j_el.find("origin")
        xyz = _parse_vec3(orig_el.attrib.get("xyz") if orig_el is not None else None, (0.0, 0.0, 0.0))
        rpy = _parse_vec3(orig_el.attrib.get("rpy") if orig_el is not None else None, (0.0, 0.0, 0.0))
        T_orig = np.eye(4, dtype=np.float64)
        T_orig[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        T_orig[:3, 3] = xyz

        ax_el = j_el.find("axis")
        axis = _parse_vec3(ax_el.attrib.get("xyz") if ax_el is not None else None, (0.0, 0.0, 1.0))
        norm = float(np.linalg.norm(axis))
        axis = axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

        lim_el = j_el.find("limit")
        if jtype == "continuous":
            lower, upper = -2.0 * np.pi, 2.0 * np.pi
            vel = float(lim_el.attrib.get("velocity", "6.283")) if lim_el is not None else 6.283
            eff = float(lim_el.attrib.get("effort", "100.0")) if lim_el is not None else 100.0
        elif lim_el is not None:
            lower = float(lim_el.attrib.get("lower", "-3.14159"))
            upper = float(lim_el.attrib.get("upper", "3.14159"))
            vel = float(lim_el.attrib.get("velocity", "3.14159"))
            eff = float(lim_el.attrib.get("effort", "100.0"))
        else:
            lower, upper, vel, eff = 0.0, 0.0, 0.0, 0.0

        mimic_el = j_el.find("mimic")
        mimic_joint = mimic_el.attrib.get("joint") if mimic_el is not None else None
        mimic_mult = float(mimic_el.attrib.get("multiplier", "1.0")) if mimic_el is not None else 1.0
        mimic_off = float(mimic_el.attrib.get("offset", "0.0")) if mimic_el is not None else 0.0

        act_idx: int | None = None
        if jtype != "fixed" and mimic_joint is None:
            act_idx = len(actuated_names)
            actuated_names.append(jname)

        ax_x, ax_y, ax_z = float(axis[0]), float(axis[1]), float(axis[2])
        ax_skew = np.array(
            [[0.0, -ax_z, ax_y], [ax_z, 0.0, -ax_x], [-ax_y, ax_x, 0.0]],
            dtype=np.float64,
        )
        parsed_joints.append(
            URDFJoint(
                name=jname,
                joint_type=jtype,
                parent_link=parent,
                child_link=child,
                origin_transform=T_orig,
                axis=axis,
                axis_skew=ax_skew,
                axis_skew_sq=ax_skew @ ax_skew,
                lower=lower,
                upper=upper,
                velocity=max(vel, 0.2),
                effort=eff,
                actuated_index=act_idx,
                mimic_joint=mimic_joint,
                mimic_multiplier=mimic_mult,
                mimic_offset=mimic_off,
            )
        )

    roots = [lnk for lnk in links if lnk not in child_links]
    if not roots:
        raise ValueError(f"URDF {path} has no root link")
    root_link = roots[0]

    # Collect visual mesh paths to verify asset completeness
    meshes: list[Path] = []
    seen_meshes: set[Path] = set()
    for m_el in xml_root.findall(".//visual//mesh"):
        fn = m_el.attrib.get("filename")
        if fn:
            resolved = _resolve_mesh_path(fn, path.parent, root_pkg)
            if resolved not in seen_meshes:
                seen_meshes.add(resolved)
                meshes.append(resolved)

    # Generate lightweight link-bone capsules for movable arm links (excluding fixed pedestal/torso brackets)
    parent_joint_map = {j.child_link: j for j in parsed_joints}

    def _has_movable_ancestor(lnk_name: str) -> bool:
        cur = lnk_name
        while cur in parent_joint_map:
            pj = parent_joint_map[cur]
            if pj.joint_type != "fixed":
                return True
            cur = pj.parent_link
        return False

    children_map: dict[str, list[URDFJoint]] = {}
    for j in parsed_joints:
        children_map.setdefault(j.parent_link, []).append(j)

    # Traverse links in kinematic topological BFS order from root_link so distal forearm/wrist
    # capsules always appear at the end of per-side lists regardless of XML declaration order.
    topo_links: list[str] = []
    queue = [root_link]
    visited: set[str] = {root_link}
    while queue:
        cur = queue.pop(0)
        topo_links.append(cur)
        for ch in children_map.get(cur, ()):
            if ch.child_link not in visited:
                visited.add(ch.child_link)
                queue.append(ch.child_link)
    for lnk in children_map:
        if lnk not in visited:
            topo_links.append(lnk)

    capsules: list[LinkCapsule] = []
    for lnk in topo_links:
        ch_list = children_map.get(lnk)
        if not ch_list or not _has_movable_ancestor(lnk):
            continue
        side = _infer_side(lnk)
        if side not in ("left", "right", "shared"):
            continue
        for ch in ch_list:
            offset = ch.origin_transform[:3, 3]
            seg_len = float(np.linalg.norm(offset))
            if seg_len >= 0.03:
                if side == "shared":
                    radius = 0.055
                else:
                    radius = 0.030 if "gripper" not in lnk and "tcp" not in ch.child_link else 0.020
                capsules.append(
                    LinkCapsule(
                        link_name=lnk,
                        side=side,
                        p0_local=np.zeros(3, dtype=np.float64),
                        p1_local=offset.copy(),
                        radius=radius,
                    )
                )

    return URDFModel(
        name=robot_name,
        urdf_path=path,
        root_link=root_link,
        links=links,
        joints=tuple(parsed_joints),
        actuated_joint_names=tuple(actuated_names),
        mesh_paths=tuple(meshes),
        capsules=tuple(capsules),
    )
