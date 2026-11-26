"""Capsule-capsule inter-arm clearance and tabletop HalfSpace collision queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pikaumi.robots.loader import LinkCapsule, URDFModel


def segment_segment_distance(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Shortest Euclidean distance and closest points `(c1, c2)` between 3D segments `[p1, q1]` and `[p2, q2]`."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    eps = 1e-12
    if a <= eps and e <= eps:
        return float(np.linalg.norm(p1 - p2)), p1.copy(), p2.copy()
    if a <= eps:
        s = 0.0
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= eps:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            if denom != 0.0:
                s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))

    c1 = p1 + s * d1
    c2 = p2 + t * d2
    return float(np.linalg.norm(c1 - c2)), c1, c2


@dataclass(frozen=True)
class CollisionCheckSummary:
    min_inter_arm_clearance_m: float
    min_table_clearance_m: float
    inter_arm_collision_frames: int
    table_penetration_frames: int


def world_capsules_for_config(
    model: URDFModel,
    q_actuated: np.ndarray,
    link_tf: dict[str, np.ndarray] | None = None,
) -> tuple[list[tuple[LinkCapsule, np.ndarray, np.ndarray]], list[tuple[LinkCapsule, np.ndarray, np.ndarray]]]:
    """Transform left and right link capsules into world coordinates `(cap, p0_w, p1_w)`."""
    if link_tf is None:
        link_tf, _ = model.forward_kinematics_all(q_actuated)
    left_caps: list[tuple[LinkCapsule, np.ndarray, np.ndarray]] = []
    right_caps: list[tuple[LinkCapsule, np.ndarray, np.ndarray]] = []
    for cap in model.capsules:
        T = link_tf.get(cap.link_name)
        if T is None:
            continue
        R = T[:3, :3]
        t = T[:3, 3]
        p0_w = R @ cap.p0_local + t
        p1_w = R @ cap.p1_local + t
        if cap.side == "left":
            left_caps.append((cap, p0_w, p1_w))
        elif cap.side == "right":
            right_caps.append((cap, p0_w, p1_w))
    return left_caps, right_caps


def world_shared_capsules_for_config(
    model: URDFModel,
    q_actuated: np.ndarray,
    link_tf: dict[str, np.ndarray] | None = None,
) -> list[tuple[LinkCapsule, np.ndarray, np.ndarray]]:
    """Transform movable shared/torso link capsules into world coordinates `(cap, p0_w, p1_w)`."""
    if link_tf is None:
        link_tf, _ = model.forward_kinematics_all(q_actuated)
    shared_caps: list[tuple[LinkCapsule, np.ndarray, np.ndarray]] = []
    for cap in model.capsules:
        if cap.side != "shared":
            continue
        T = link_tf.get(cap.link_name)
        if T is None:
            continue
        R = T[:3, :3]
        t = T[:3, 3]
        shared_caps.append((cap, R @ cap.p0_local + t, R @ cap.p1_local + t))
    return shared_caps


def distal_capsules_by_links(
    caps: list[tuple[LinkCapsule, np.ndarray, np.ndarray]],
    n_distinct_links: int = 3,
) -> list[tuple[LinkCapsule, np.ndarray, np.ndarray]]:
    """Select all capsules belonging to the last `n_distinct_links` distinct links (topologically distal).

    Prevents links with multiple collision capsules (e.g. two-capsule `gripper_base` on Piper/Pika)
    from pushing wrist/forearm links (`link5`, `link6`) out of distal tabletop and torso clearance checks.
    """
    if not caps or n_distinct_links <= 0:
        return []
    distinct_links: list[str] = []
    for cap, _, _ in reversed(caps):
        if cap.link_name not in distinct_links:
            distinct_links.append(cap.link_name)
            if len(distinct_links) >= n_distinct_links:
                break
    selected = set(distinct_links)
    return [entry for entry in caps if entry[0].link_name in selected]


def evaluate_config_clearance(
    model: URDFModel,
    q_actuated: np.ndarray,
    *,
    table_z_in_robot_world: float | None = None,
    ee_links: tuple[str, ...] = (),
) -> tuple[float, float]:
    """Return `(min_inter_arm_clearance_m, min_table_clearance_m)` for a single joint configuration.

    Positive values mean collision-free separation; negative values indicate capsule penetration.
    """
    link_tf, _ = model.forward_kinematics_all(q_actuated)
    left_caps, right_caps = world_capsules_for_config(model, q_actuated, link_tf=link_tf)
    shared_caps = world_shared_capsules_for_config(model, q_actuated, link_tf=link_tf)
    left_distal = distal_capsules_by_links(left_caps, n_distinct_links=3)
    right_distal = distal_capsules_by_links(right_caps, n_distinct_links=3)

    min_inter = float("inf")
    for cap_l, p0_l, p1_l in left_caps:
        for cap_r, p0_r, p1_r in right_caps:
            dist, _, _ = segment_segment_distance(p0_l, p1_l, p0_r, p1_r)
            clearance = dist - (cap_l.radius + cap_r.radius)
            if clearance < min_inter:
                min_inter = clearance
    # Check distal forearm/wrist/gripper capsules against movable torso/shared links (e.g. r1lite)
    if shared_caps:
        for group in (left_distal, right_distal):
            for cap_a, p0_a, p1_a in group:
                for cap_s, p0_s, p1_s in shared_caps:
                    dist, _, _ = segment_segment_distance(p0_a, p1_a, p0_s, p1_s)
                    clearance = dist - (cap_a.radius + cap_s.radius)
                    if clearance < min_inter:
                        min_inter = clearance
    if min_inter == float("inf"):
        min_inter = 1.0

    min_table = 1.0
    if table_z_in_robot_world is not None:
        # Check EE points and distal forearm/wrist/gripper capsules (selected by distinct link names)
        z_vals: list[float] = []
        for ee in ee_links:
            if ee in link_tf:
                z_vals.append(float(link_tf[ee][2, 3]) - float(table_z_in_robot_world))
        for group in (left_distal, right_distal):
            for cap, p0_w, p1_w in group:
                z_vals.append(min(float(p0_w[2]), float(p1_w[2])) - cap.radius - float(table_z_in_robot_world))
        if z_vals:
            min_table = min(z_vals)

    return float(min_inter), float(min_table)


def evaluate_trajectory_clearance(
    model: URDFModel,
    q_traj: np.ndarray,
    *,
    table_z_in_robot_world: float | None = None,
    ee_links: tuple[str, ...] = (),
    table_tolerance_m: float = -0.015,
) -> CollisionCheckSummary:
    min_inter = float("inf")
    min_table = float("inf")
    inter_collisions = 0
    table_hits = 0

    for t in range(len(q_traj)):
        c_inter, c_table = evaluate_config_clearance(
            model,
            q_traj[t],
            table_z_in_robot_world=table_z_in_robot_world,
            ee_links=ee_links,
        )
        min_inter = min(min_inter, c_inter)
        min_table = min(min_table, c_table)
        if c_inter < 0.0:
            inter_collisions += 1
        if table_z_in_robot_world is not None and c_table < table_tolerance_m:
            table_hits += 1

    return CollisionCheckSummary(
        min_inter_arm_clearance_m=float(min_inter if min_inter != float("inf") else 1.0),
        min_table_clearance_m=float(min_table if min_table != float("inf") else 1.0),
        inter_arm_collision_frames=inter_collisions,
        table_penetration_frames=table_hits,
    )
