"""Headless 3-view kinematic skeleton preview renderer & diagnostics (`pika replay`, `pika replay-joints`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pikaumi.calibration.deployment import load_table_calibration
from pikaumi.config import save_json_atomic
from pikaumi.dataset.raw import load_dataset_info
from pikaumi.dataset.screening import (
    evaluate_retargeted_trajectory,
    solve_episode_with_method,
)
from pikaumi.retargeting.pika_to_robot import prepare_episode_robot_targets
from pikaumi.robots.registry import RobotProfile, load_robot_profile


def render_kinematic_preview_strip(
    profile: RobotProfile,
    q_actuated: np.ndarray,
    *,
    width: int = 720,
    height: int = 260,
) -> np.ndarray:
    """Render a headless 3-panel (Top XY, Front YZ, Side XZ) kinematic stick-figure image (`uint8 BGR`)."""
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    panel_w = width // 3
    link_tf, _ = profile.urdf_model.forward_kinematics_all(q_actuated)

    panels = [
        ("Top (X-Y)", 0, 1, 220.0),
        ("Front (Y-Z)", 1, 2, 220.0),
        ("Side (X-Z)", 0, 2, 220.0),
    ]
    for p_idx, (title, ax_u, ax_v, scale) in enumerate(panels):
        x_off = p_idx * panel_w
        cx = x_off + panel_w // 2
        cy = int(height * 0.68)
        cv2.rectangle(canvas, (x_off + 2, 2), (x_off + panel_w - 2, height - 2), (55, 55, 55), 1)
        cv2.putText(canvas, title, (x_off + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)

        for j in profile.urdf_model.topo_joints:
            Tp = link_tf[j.parent_link]
            Tc = link_tf[j.child_link]
            p0 = Tp[:3, 3]
            p1 = Tc[:3, 3]
            u0 = int(round(cx + p0[ax_u] * scale))
            v0 = int(round(cy - p0[ax_v] * scale))
            u1 = int(round(cx + p1[ax_u] * scale))
            v1 = int(round(cy - p1[ax_v] * scale))
            low = j.child_link.lower()
            color = (240, 180, 70) if "left" in low else ((90, 140, 250) if "right" in low else (170, 170, 170))
            cv2.line(canvas, (u0, v0), (u1, v1), color, 2, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (u1, v1), 3, (245, 245, 245), -1)

    return canvas


def replay_raw_episode(
    dataset_dir: Path,
    *,
    episode_idx: int,
    robot: str,
    solver: str,
    output_dir: Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = load_dataset_info(dataset_dir)
    fps = float(info.get("fps", 30.0))

    prof = load_robot_profile(robot)
    tcal = load_table_calibration(robot, require_physical=False)
    mode = "relative-anchor" if len(prof.active_sides) == 1 else "absolute-table"
    ee_tgts, widths, ts, table_z = prepare_episode_robot_targets(
        str(dataset_dir), episode_idx, prof, table_calib=tcal, retarget_mode=mode
    )
    q_traj, _, solve_s = solve_episode_with_method(
        prof, ee_tgts, widths, solver_name=solver, fps=fps, table_z_robot=table_z
    )
    metrics = evaluate_retargeted_trajectory(
        prof, q_traj, ee_tgts, widths, fps=fps, table_z_robot=table_z
    )

    # Render start/mid/end keyframe strip PNG
    idxs = [0, len(q_traj) // 2, len(q_traj) - 1]
    strips = [render_kinematic_preview_strip(prof, q_traj[i]) for i in idxs]
    grid_img = np.vstack(strips)
    preview_png = out / f"replay_{robot}_ep{episode_idx:04d}.png"
    cv2.imwrite(str(preview_png), grid_img)

    summary = {
        "mode": "replay_raw",
        "robot": robot,
        "solver": solver,
        "episode_index": episode_idx,
        "num_frames": int(len(ts)),
        "solve_time_s": round(solve_s, 4),
        "preview_png": str(preview_png),
        "metrics": metrics,
    }
    save_json_atomic(out / f"replay_{robot}_ep{episode_idx:04d}.json", summary)
    return summary


def _resolve_pybullet_urdf(profile: RobotProfile, tmp_dir: Path) -> Path:
    """Rewrite ROS `package://<pkg>/` URIs and prune duplicate-parent dummy roots for PyBullet."""
    import xml.etree.ElementTree as ET

    raw_xml = profile.urdf_path.read_text(encoding="utf-8")
    pkg_abs = str(profile.pkg_root.resolve()).rstrip("/") + "/"
    for prefix in {f"package://{profile.pkg_root.name}/", f"package://{profile.kind}/"}:
        raw_xml = raw_xml.replace(prefix, pkg_abs)

    root = ET.fromstring(raw_xml)
    seen_children: set[str] = set()
    to_remove_joints: list[ET.Element] = []
    for j_el in list(root.findall("joint")):
        child_el = j_el.find("child")
        if child_el is None:
            continue
        child_name = child_el.get("link", "")
        if child_name in seen_children:
            to_remove_joints.append(j_el)
        else:
            seen_children.add(child_name)
    for j_el in to_remove_joints:
        root.remove(j_el)

    # Remove orphan links that are neither the primary root nor referenced by any remaining joint
    referenced_links: set[str] = set()
    for j_el in root.findall("joint"):
        p_el = j_el.find("parent")
        c_el = j_el.find("child")
        if p_el is not None and p_el.get("link"):
            referenced_links.add(str(p_el.get("link")))
        if c_el is not None and c_el.get("link"):
            referenced_links.add(str(c_el.get("link")))
    for l_el in list(root.findall("link")):
        lname = l_el.get("name", "")
        if lname and lname not in referenced_links:
            root.remove(l_el)

    resolved = tmp_dir / f"{profile.urdf_path.stem}_pybullet.urdf"
    ET.ElementTree(root).write(str(resolved), encoding="utf-8", xml_declaration=True)
    return resolved


def render_pybullet_joint_replay_video(
    joint_dataset_dir: Path,
    *,
    episode_idx: int,
    output_mp4: Path,
    gui: bool = False,
    camera_yaw_deg: float = 135.0,
    camera_pitch_deg: float = -28.0,
    camera_distance_m: float = 1.15,
) -> dict[str, Any]:
    """Replay a converted joint-action episode in PyBullet 3D URDF simulation and export an MP4 video.

    Renders the 3D robot + calibrated tabletop side-by-side with the synchronized real wrist camera
    streams (when present in `joint_dataset_dir/videos/`). Purely offline simulation — zero hardware I/O.
    """
    import tempfile
    import time

    try:
        import pybullet as pb
    except ImportError as exc:
        raise ImportError(
            "PyBullet is required for 3D simulation replay. Install with: "
            "`uv pip install --python .venv-pika-dataset/bin/python pybullet`"
        ) from exc

    ds_dir = Path(joint_dataset_dir)
    info = json.loads((ds_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    robot = str(info["robot"])
    fps = float(info.get("fps", 30.0))
    prof = load_robot_profile(robot)
    tcal = load_table_calibration(robot, require_physical=False)
    table_pos = np.asarray(tcal.T_robot_table_pose7[:3], dtype=np.float64)
    table_quat = np.asarray(tcal.T_robot_table_pose7[3:7], dtype=np.float64)

    with np.load(ds_dir / "data" / f"episode_{episode_idx:06d}.npz") as z:
        q_traj = z["urdf_q_full"].astype(np.float64)
        ts = z["timestamps_s"].astype(np.float64)
        act_arr = z["action"].astype(np.float64) if "action" in z else None

    # Open optional synchronized wrist camera videos for side-by-side comparison
    wrist_caps: dict[str, Any] = {}
    for cam_key in ("left_wrist", "right_wrist"):
        for cand in (
            ds_dir / "videos" / cam_key / f"episode_{episode_idx:06d}.mp4",
            ds_dir / "videos" / "chunk-000" / f"observation.images.{cam_key}" / f"episode_{episode_idx:06d}.mp4",
        ):
            if cand.is_file():
                cap = cv2.VideoCapture(str(cand))
                if cap.isOpened():
                    wrist_caps[cam_key] = cap
                break

    out_path = Path(output_mp4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sim_w, sim_h = 720, 540
    side_w = 360 if wrist_caps else 0
    total_w, total_h = sim_w + side_w, sim_h

    # Parse mimic joints from URDF so parallel gripper fingers move symmetrically in PyBullet
    import xml.etree.ElementTree as ET

    mimic_map: dict[str, tuple[str, float, float]] = {}
    tree = ET.parse(prof.urdf_path)
    for j_el in tree.getroot().findall("joint"):
        m_el = j_el.find("mimic")
        if m_el is not None and j_el.get("name") and m_el.get("joint"):
            mimic_map[str(j_el.get("name"))] = (
                str(m_el.get("joint")),
                float(m_el.get("multiplier", "1.0")),
                float(m_el.get("offset", "0.0")),
            )

    cid = pb.connect(pb.GUI if gui else pb.DIRECT)
    try:
        pb.resetSimulation(physicsClientId=cid)
        pb.setGravity(0.0, 0.0, -9.81, physicsClientId=cid)

        with tempfile.TemporaryDirectory(prefix="pikaumi_pb_") as tmp_s:
            urdf_pb = _resolve_pybullet_urdf(prof, Path(tmp_s))
            robot_id = pb.loadURDF(
                str(urdf_pb),
                basePosition=[0.0, 0.0, 0.0],
                useFixedBase=True,
                flags=pb.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                physicsClientId=cid,
            )

        # Draw physical calibrated tabletop slab (top face at calibrated table_pos[2])
        slab_h = 0.02
        slab_col = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=[0.45, 0.55, slab_h / 2.0],
            rgbaColor=[0.84, 0.80, 0.74, 0.92],
            physicsClientId=cid,
        )
        pb.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=slab_col,
            basePosition=[float(table_pos[0]), float(table_pos[1]), float(table_pos[2] - slab_h / 2.0)],
            baseOrientation=table_quat.tolist(),
            physicsClientId=cid,
        )
        # Draw a small dark marker pad at calibrated {Table} origin (0,0,0)
        pad_vis = pb.createVisualShape(
            pb.GEOM_BOX,
            halfExtents=[0.09, 0.11, 0.001],
            rgbaColor=[0.18, 0.32, 0.48, 0.85],
            physicsClientId=cid,
        )
        pb.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=pad_vis,
            basePosition=[float(table_pos[0]), float(table_pos[1]), float(table_pos[2] + 0.001)],
            baseOrientation=table_quat.tolist(),
            physicsClientId=cid,
        )

        # Map joint name -> PyBullet joint index
        pb_joint_idx: dict[str, int] = {}
        for j_i in range(pb.getNumJoints(robot_id, physicsClientId=cid)):
            j_info = pb.getJointInfo(robot_id, j_i, physicsClientId=cid)
            j_name = j_info[1].decode("utf-8")
            pb_joint_idx[j_name] = j_i

        actuated_names = list(prof.urdf_model.actuated_joint_names)
        cam_target = [float(table_pos[0] - 0.05), float(table_pos[1]), float(table_pos[2] + 0.18)]
        if gui:
            pb.resetDebugVisualizerCamera(
                cameraDistance=camera_distance_m,
                cameraYaw=camera_yaw_deg,
                cameraPitch=camera_pitch_deg,
                cameraTargetPosition=cam_target,
                physicsClientId=cid,
            )

        view_mat = pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=cam_target,
            distance=camera_distance_m,
            yaw=camera_yaw_deg,
            pitch=camera_pitch_deg,
            roll=0.0,
            upAxisIndex=2,
        )
        proj_mat = pb.computeProjectionMatrixFOV(
            fov=52.0,
            aspect=float(sim_w) / float(sim_h),
            nearVal=0.05,
            farVal=5.0,
        )
        renderer = pb.ER_BULLET_HARDWARE_OPENGL if gui else pb.ER_TINY_RENDERER

        # Prefer H.264 via PyAV for universal playback, fallback to OpenCV mp4v
        av_container = None
        av_stream = None
        cv_writer = None
        try:
            import av  # type: ignore

            av_container = av.open(str(out_path), mode="w")
            av_stream = av_container.add_stream("libx264", rate=int(round(fps)))
            av_stream.width = total_w
            av_stream.height = total_h
            av_stream.pix_fmt = "yuv420p"
            av_stream.options = {"crf": "20", "preset": "fast"}
        except Exception:
            if av_container is not None:
                av_container.close()
                av_container = None
                av_stream = None
            cv_writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (total_w, total_h)
            )

        keyframe_bgrs: list[np.ndarray] = []
        key_indices = {0, len(q_traj) // 3, (2 * len(q_traj)) // 3, len(q_traj) - 1}

        try:
            for i in range(len(q_traj)):
                q = q_traj[i]
                q_by_name = {name: float(q[k]) for k, name in enumerate(actuated_names)}
                for j_name, q_val in q_by_name.items():
                    if j_name in pb_joint_idx:
                        pb.resetJointState(robot_id, pb_joint_idx[j_name], q_val, physicsClientId=cid)
                for m_name, (parent_j, mult, off) in mimic_map.items():
                    if m_name in pb_joint_idx and parent_j in q_by_name:
                        pb.resetJointState(
                            robot_id,
                            pb_joint_idx[m_name],
                            q_by_name[parent_j] * mult + off,
                            physicsClientId=cid,
                        )

                _, _, rgba, _, _ = pb.getCameraImage(
                    width=sim_w,
                    height=sim_h,
                    viewMatrix=view_mat,
                    projectionMatrix=proj_mat,
                    lightDirection=[0.6, -0.5, 1.2],
                    renderer=renderer,
                    physicsClientId=cid,
                )
                rgb_sim = np.asarray(rgba, dtype=np.uint8).reshape(sim_h, sim_w, 4)[:, :, :3]
                bgr_sim = cv2.cvtColor(rgb_sim, cv2.COLOR_RGB2BGR)

                # Overlay telemetry header
                t_sec = float(ts[i] - ts[0]) if len(ts) > i else i / fps
                hud = f"PyBullet 3D Sim | {robot} | ep{episode_idx:04d} | frame {i+1}/{len(q_traj)} ({t_sec:.2f}s)"
                if act_arr is not None and act_arr.shape[1] >= 14:
                    hud += f" | grip L={act_arr[i,6]*1000:.0f}mm R={act_arr[i,13]*1000:.0f}mm"
                cv2.rectangle(bgr_sim, (0, 0), (sim_w, 30), (24, 24, 28), -1)
                cv2.putText(bgr_sim, hud, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)

                if wrist_caps:
                    panel = np.full((sim_h, side_w, 3), 24, dtype=np.uint8)
                    half_h = sim_h // 2
                    for slot, (ck, label) in enumerate((("left_wrist", "Real Left Wrist"), ("right_wrist", "Real Right Wrist"))):
                        cap = wrist_caps.get(ck)
                        if cap is not None:
                            ok, fr = cap.read()
                            if ok and fr is not None:
                                resized = cv2.resize(fr, (side_w, half_h))
                                y0 = slot * half_h
                                panel[y0 : y0 + half_h, :] = resized
                                cv2.rectangle(panel, (0, y0), (170, y0 + 24), (24, 24, 28), -1)
                                cv2.putText(
                                    panel, label, (8, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA
                                )
                    frame_bgr = np.hstack([bgr_sim, panel])
                else:
                    frame_bgr = bgr_sim

                if i in key_indices:
                    keyframe_bgrs.append(frame_bgr.copy())

                if av_container is not None and av_stream is not None:
                    import av  # type: ignore

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    av_fr = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                    for packet in av_stream.encode(av_fr):
                        av_container.mux(packet)
                elif cv_writer is not None:
                    cv_writer.write(frame_bgr)

                if gui:
                    time.sleep(max(0.0, 1.0 / fps))

            if av_container is not None and av_stream is not None:
                for packet in av_stream.encode():
                    av_container.mux(packet)
        finally:
            if av_container is not None:
                av_container.close()
            if cv_writer is not None:
                cv_writer.release()
            for cap in wrist_caps.values():
                cap.release()
    finally:
        pb.disconnect(physicsClientId=cid)

    # Save a 2x2 contact sheet PNG of the 3D PyBullet replay for quick visual inspection
    contact_png = out_path.with_suffix(".png")
    if len(keyframe_bgrs) == 4:
        top = np.hstack([cv2.resize(keyframe_bgrs[0], (total_w // 2, total_h // 2)), cv2.resize(keyframe_bgrs[1], (total_w // 2, total_h // 2))])
        bot = np.hstack([cv2.resize(keyframe_bgrs[2], (total_w // 2, total_h // 2)), cv2.resize(keyframe_bgrs[3], (total_w // 2, total_h // 2))])
        cv2.imwrite(str(contact_png), np.vstack([top, bot]))
    elif keyframe_bgrs:
        cv2.imwrite(str(contact_png), keyframe_bgrs[0])

    return {
        "video_mp4": str(out_path),
        "contact_sheet_png": str(contact_png),
        "num_frames": int(len(q_traj)),
        "fps": fps,
        "resolution": [total_w, total_h],
        "with_wrist_videos": bool(wrist_caps),
    }


def replay_joint_episode(
    joint_dataset_dir: Path,
    *,
    episode_idx: int,
    output_dir: Path,
    sim_video: bool = False,
    gui: bool = False,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((Path(joint_dataset_dir) / "meta" / "info.json").read_text(encoding="utf-8"))
    robot = str(info["robot"])
    prof = load_robot_profile(robot)

    with np.load(Path(joint_dataset_dir) / "data" / f"episode_{episode_idx:06d}.npz") as z:
        q_traj = z["urdf_q_full"].astype(np.float64)
        ts = z["timestamps_s"].astype(np.float64)

    idxs = [0, len(q_traj) // 2, len(q_traj) - 1]
    strips = [render_kinematic_preview_strip(prof, q_traj[i]) for i in idxs]
    preview_png = out / f"replay_joints_{robot}_ep{episode_idx:04d}.png"
    cv2.imwrite(str(preview_png), np.vstack(strips))

    summary: dict[str, Any] = {
        "mode": "replay_joints",
        "robot": robot,
        "episode_index": episode_idx,
        "num_frames": int(len(ts)),
        "preview_png": str(preview_png),
    }
    if sim_video or gui:
        mp4_path = out / f"sim_replay_{robot}_ep{episode_idx:04d}.mp4"
        sim_rep = render_pybullet_joint_replay_video(
            Path(joint_dataset_dir),
            episode_idx=episode_idx,
            output_mp4=mp4_path,
            gui=gui,
        )
        summary["pybullet_sim"] = sim_rep
    save_json_atomic(out / f"replay_joints_{robot}_ep{episode_idx:04d}.json", summary)
    return summary


def dispatch_replay(args: argparse.Namespace) -> int:
    if args.command == "replay":
        rep = replay_raw_episode(
            Path(args.dataset_dir),
            episode_idx=args.episode,
            robot=args.robot,
            solver=args.solver,
            output_dir=Path(args.output_dir),
        )
        print(
            f"[OK] Replay preview saved to {rep['preview_png']} "
            f"(verdict={rep['metrics']['verdict']}, pos_err={rep['metrics']['mean_pos_err_mm']:.2f}mm)"
        )
        return 0

    rep = replay_joint_episode(
        Path(args.dataset_dir),
        episode_idx=args.episode,
        output_dir=Path(args.output_dir),
        sim_video=bool(getattr(args, "sim_video", False)),
        gui=bool(getattr(args, "gui", False)),
    )
    print(f"[OK] Joint replay preview saved to {rep['preview_png']}")
    if "pybullet_sim" in rep:
        print(f"[OK] PyBullet 3D simulation video saved to {rep['pybullet_sim']['video_mp4']}")
    return 0
