"""Asset completeness and declarative YAML config tests across all 9 target robot embodiments."""

from __future__ import annotations

import numpy as np

from pikaumi.config import TABLE_SIM_CALIB_DIR, load_yaml
from pikaumi.robots.registry import list_robot_names, load_robot_profile

EXPECTED_ROBOTS = {
    "piper_pika",
    "piper",
    "openarmv1",
    "arx5",
    "yam",
    "trlc_dk1",
    "metal",
    "r1lite",
    "franka_emika_panda",
}


def test_all_nine_robot_profiles_and_meshes_resolve() -> None:
    registered = set(list_robot_names())
    assert EXPECTED_ROBOTS.issubset(registered)

    for name in sorted(EXPECTED_ROBOTS):
        prof = load_robot_profile(name)
        assert prof.urdf_path.is_file(), f"Missing URDF for {name}: {prof.urdf_path}"
        assert len(prof.urdf_model.mesh_paths) > 0, f"No visual meshes parsed for {name}"
        missing = [m for m in prof.urdf_model.mesh_paths if not m.is_file()]
        assert not missing, f"Robot {name} has missing mesh files: {missing[:3]}"

        # Table sim calibration exists for every embodiment
        sim_table = TABLE_SIM_CALIB_DIR / f"{name}.yaml"
        assert sim_table.is_file(), f"Missing sim table YAML for {name}: {sim_table}"
        t_cfg = load_yaml(sim_table)
        assert t_cfg["robot"] == name
        rft = t_cfg["calibration"]["robot_from_table"]
        assert len(rft["position"]) == 3 and len(rft["quaternion"]) == 4


def test_piper_pika_geometry_contract() -> None:
    """Verify `GEMINI.md` Pika-on-Piper 100 mm stroke and ~194/195 mm axial TCP offset."""
    prof = load_robot_profile("piper_pika")
    assert prof.active_sides == ("left", "right")
    assert np.isclose(prof.gripper_max_width_m, 0.10)
    assert prof.gripper_retarget_mode == "physical-width"

    # At zero arm angles, left_link6 -> left_tcp should reflect ~194.8 mm axial offset along +X of link6
    q_zero = np.zeros(prof.urdf_model.num_actuated, dtype=np.float64)
    link_tf, _ = prof.urdf_model.forward_kinematics_all(q_zero)
    for side in ("left", "right"):
        T_flange = link_tf[f"{side}_link6"]
        T_tcp = link_tf[f"{side}_tcp"]
        T_flange_tcp = np.linalg.inv(T_flange) @ T_tcp
        axial_dist = float(np.linalg.norm(T_flange_tcp[:3, 3]))
        assert 0.190 <= axial_dist <= 0.200, f"Unexpected {side} Pika TCP offset: {axial_dist:.4f} m"

    # Gripper mapping: in piper_pika URDF, q=0.0 is open (100 mm) and q=-0.05 is closed (0 mm)
    q_open, rw_open = prof.map_gripper_width_to_q(q_zero, "left", 0.10)
    q_closed, rw_closed = prof.map_gripper_width_to_q(q_zero, "left", 0.00)
    assert np.isclose(rw_open, 0.10)
    assert np.isclose(rw_closed, 0.00)
    for gspec in prof.arms["left"].gripper_joints:
        assert np.isclose(q_open[gspec.actuated_index], gspec.open_value)
        assert np.isclose(q_open[gspec.actuated_index], 0.0)
        assert np.isclose(q_closed[gspec.actuated_index], gspec.closed_value)
        assert np.isclose(q_closed[gspec.actuated_index], -0.05)

    # Confirm physical URDF finger link separation is strictly larger when open than when closed
    tf_open, _ = prof.urdf_model.forward_kinematics_all(q_open)
    tf_closed, _ = prof.urdf_model.forward_kinematics_all(q_closed)
    d_open = float(np.linalg.norm(tf_open["left_link7"][:3, 3] - tf_open["left_link8"][:3, 3]))
    d_closed = float(np.linalg.norm(tf_closed["left_link7"][:3, 3] - tf_closed["left_link8"][:3, 3]))
    assert d_open > d_closed + 0.09, f"Expected ~100 mm stroke open-closed delta, got {d_open - d_closed:.4f} m"
