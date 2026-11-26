"""Unified lazy-loaded CLI for `pikaumi` (`pika` / `pk`).

Covers all 5 stages of the standardized Dual Pika Sense -> Multi-Robot workflow:
  1. Setup & Calibration:  `doctor`, `setup`, `calibrate`
  2. Capture & Sensors:    `sensors`, `status`, `record`, `session`, `import-raw`
  3. QA & Curation:        `dataset {validate,direction,screen,qa,curate,merge,export}`
  4. Retarget & Benchmark: `robots`, `convert`, `benchmark-retarget`
  5. Replay & Control:     `replay`, `replay-joints`, `replay-real`, `policy-bridge`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from pikaumi import __version__
from pikaumi.config import PIKA_LEGACY_DIR, RIG_CONFIG, RIG_EXAMPLE_CONFIG


def _delegate_legacy_pika_data(subcmd: str, rest: list[str]) -> int:
    script = PIKA_LEGACY_DIR / "data_tools" / "pika_data.py"
    if not script.is_file():
        raise FileNotFoundError(f"Legacy Pika_Data router not found: {script}")
    completed = subprocess.run([sys.executable, str(script), subcmd, *rest], check=False)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pika",
        description="pikaumi — Robot-free Dual Pika Sense capture, calibration, and C-STRO multi-robot retargeting",
    )
    parser.add_argument("--version", action="version", version=f"pikaumi {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- doctor ---
    p_doc = sub.add_parser("doctor", help="Read-only environment, skill, config & robot asset check")
    p_doc.add_argument("--json-out", default=None, help="Optional JSON output path")
    p_doc.add_argument("--strict", action="store_true", help="Treat warnings as failures")

    # --- setup ---
    p_setup = sub.add_parser("setup", help="Initialize local configs/rig.yaml or check hardware readiness")
    p_setup.add_argument("--check", action="store_true", help="Run read-only readiness check (`doctor`)")
    p_setup.add_argument("--init-rig", action="store_true", help="Create configs/rig.yaml from template if absent")
    p_setup.add_argument("--force", action="store_true", help="Overwrite configs/rig.yaml when --init-rig is set")

    # --- robots ---
    p_robots = sub.add_parser("robots", help="Inspect registered robot embodiments and URDF kinematics")
    r_sub = p_robots.add_subparsers(dest="robots_cmd", required=True)
    r_sub.add_parser("list", help="List all available robot embodiments")
    p_rinfo = r_sub.add_parser("info", help="Show detailed joint/gripper/TCP config for a robot")
    p_rinfo.add_argument("robot", help="Robot profile name (e.g. piper_pika, piper, openarmv1, arx5)")
    p_rinfo.add_argument("--json-out", default=None)

    # --- calibrate ---
    p_cal = sub.add_parser("calibrate", help="5-layer gated calibration & SHA-256 provenance verification")
    c_sub = p_cal.add_subparsers(dest="calib_cmd", required=True)

    p_c_imp = c_sub.add_parser("import-legacy", help="Import existing data/calibration JSONs into standardized YAML")
    p_c_imp.add_argument("--legacy-dir", default=None, help="Source legacy calibration directory")
    p_c_imp.add_argument("--output-dir", required=True, help="Output directory for standardized YAML files")

    p_c_vive = c_sub.add_parser("vive", help="Layer 1: Gate a stationary Vive tracker pose window for stability")
    p_c_vive.add_argument("--poses-json", required=True, help="JSON file with poses7 and optional accurate flags")
    p_c_vive.add_argument("--max-pos-std-mm", type=float, default=1.5)
    p_c_vive.add_argument("--max-ori-std-deg", type=float, default=0.5)
    p_c_vive.add_argument("--output", default=None, help="Optional YAML report path")

    p_c_grip = c_sub.add_parser("gripper", help="Layer 2: Calibrate Pika Sense jaw encoder zero-point & span")
    p_c_grip.add_argument("--side", choices=("left", "right"), required=True)
    p_c_grip.add_argument("--samples-json", required=True, help="JSON file with closed_m and open_m sample lists")
    p_c_grip.add_argument("--max-width-m", type=float, default=0.10)
    p_c_grip.add_argument("--output", default=None, help="Optional YAML report path")

    p_c_cap = c_sub.add_parser(
        "capture-charuco",
        help="Capture live ChArUco + Vive observations from pika-ros Docker for hand-eye or session",
    )
    p_c_cap.add_argument("--mode", choices=("hand-eye", "session"), required=True)
    p_c_cap.add_argument("--side", choices=("left", "right"), required=True)
    p_c_cap.add_argument("--spatial-yaml", required=True, help="Path to spatial.yaml containing fisheye K/D")
    p_c_cap.add_argument("--output", required=True, help="Output observations JSON path")
    p_c_cap.add_argument("--views", type=int, default=None, help="Target view count (default: 30 hand-eye, 15 session)")
    p_c_cap.add_argument("--min-corners", type=int, default=None, help="Minimum detected ChArUco corners per frame")
    p_c_cap.add_argument("--max-reprojection-px", type=float, default=2.5)
    p_c_cap.add_argument("--max-pose-gap-ms", type=float, default=25.0)
    p_c_cap.add_argument("--min-rot-deg", type=float, default=None)
    p_c_cap.add_argument("--min-trans-mm", type=float, default=None)
    p_c_cap.add_argument("--max-motion-mm", type=float, default=4.0, help="Max Vive motion over stationary window (mm)")
    p_c_cap.add_argument("--max-motion-deg", type=float, default=2.0, help="Max Vive rotation over stationary window (deg)")
    p_c_cap.add_argument("--stationary-window-ms", type=float, default=220.0, help="Stationary pose window before frame (ms)")
    p_c_cap.add_argument("--squares-x", type=int, default=None)
    p_c_cap.add_argument("--squares-y", type=int, default=None)
    p_c_cap.add_argument("--square-m", type=float, default=None)
    p_c_cap.add_argument("--marker-m", type=float, default=None)
    p_c_cap.add_argument("--dictionary", default=None)
    p_c_cap.add_argument("--container", default="pika-ros")
    p_c_cap.add_argument("--no-preview", action="store_true", help="Disable OpenCV live preview window")
    p_c_cap.add_argument("--rig-config", default=None)

    p_c_he = c_sub.add_parser("hand-eye", help="Layer 3: Solve rigid T_tracker_cam hand-eye mount via Huber SE(3)")
    p_c_he.add_argument("--observations-json", required=True, help="JSON with T_vive_tracker and T_cam_board sequences")
    p_c_he.add_argument("--side", choices=("left", "right"), required=True)
    p_c_he.add_argument("--max-rms-m", type=float, default=0.005)
    p_c_he.add_argument("--spatial-yaml", default=None, help="Optional spatial.yaml to update mount in-place")
    p_c_he.add_argument("--output", required=True, help="Output YAML path")

    p_c_piv = c_sub.add_parser("tcp-pivot", help="Solve tracker-to-TCP pivot offset from multi-orientation poses")
    p_c_piv.add_argument("--poses-json", required=True, help="JSON file containing poses7 [[x,y,z,qx,qy,qz,qw], ...]")
    p_c_piv.add_argument("--side", choices=("left", "right"), required=True)
    p_c_piv.add_argument("--output", required=True)

    p_c_axes = c_sub.add_parser("tcp-axes", help="Solve 3-axis TCP rotation via SVD from signed translation snapshots")
    p_c_axes.add_argument("--directory", required=True, help="Snapshot directory containing a0,ax,ay,az[,b0,bx,by,bz]")
    p_c_axes.add_argument("--distance-mm", type=float, default=80.0)
    p_c_axes.add_argument("--output", required=True)

    p_c_ses = c_sub.add_parser("session", help="Compute tabletop T_table_vive from ChArUco observations")
    p_c_ses.add_argument("--observations-json", required=True, help="JSON with tracker & camera-board poses")
    p_c_ses.add_argument("--spatial-yaml", required=True, help="Path to spatial.yaml (intrinsics + mount)")
    p_c_ses.add_argument("--output", required=True, help="Output session.yaml")

    p_c_tr = c_sub.add_parser("table-robot", help="Layer 5: Solve physical T_robot_table from 3 TCP contact points")
    p_c_tr.add_argument("--robot", default="piper_pika", help="Target robot profile name")
    p_c_tr.add_argument("--contacts-json", required=True, help="JSON with origin, x_axis, y_plane [x,y,z] points")
    p_c_tr.add_argument("--output", required=True, help="Output physical table calibration YAML")
    p_c_tr.add_argument("--verified", action="store_true", help="Mark physical calibration as verified")

    p_c_ctr = c_sub.add_parser(
        "capture-table-robot",
        help="Layer 5: Interactive READ-ONLY 3-point tabletop capture (Teach Mode CAN FK / manual / mock)",
    )
    p_c_ctr.add_argument("--robot", default="piper_pika", help="Target robot profile name")
    p_c_ctr.add_argument("--side", choices=("left", "right"), default="left", help="Arm used to touch the 3 points")
    p_c_ctr.add_argument("--source", choices=("can", "manual", "mock"), default="can", help="Measurement source")
    p_c_ctr.add_argument("--port", default=None, help="Optional CAN port override (default: can0 left / can1 right)")
    p_c_ctr.add_argument("--sdk", choices=("auto", "pyagxarm", "piper_sdk"), default=None)
    p_c_ctr.add_argument("--firmware", default=None)
    p_c_ctr.add_argument("--rig-config", default=None)
    p_c_ctr.add_argument("--samples", type=int, default=15, help="Stationary joint samples per point")
    p_c_ctr.add_argument("--max-spread-deg", type=float, default=1.5, help="Max allowed joint spread during touch")
    p_c_ctr.add_argument("--contacts-json", default="data/calibration_std/contacts_piper.json")
    p_c_ctr.add_argument("--output", default=None, help="Output YAML path (default: configs/calibration/table/local/<robot>.yaml)")
    p_c_ctr.add_argument("--unverified", action="store_true", help="Save with verified=false instead of verified=true")

    p_c_ver = c_sub.add_parser("verify", help="Verify static SHA-256 chain & optional two-tip touch / table height")
    p_c_ver.add_argument("--calibration-dir", required=True)
    p_c_ver.add_argument("--touch-json", default=None, help="Optional two-tip touch & table contact poses JSON")
    p_c_ver.add_argument("--json-out", default=None)

    # --- live sensor & legacy wrappers (safe passthrough) ---
    for legacy_cmd, desc in (
        ("sensors", "Launch Docker ROS 1 Pika sensors (single-instance locked)"),
        ("status", "Check live ROS 1 sensor stream synchronization & tracking accuracy"),
        ("session", "Open multi-episode Tkinter session capture & conversion GUI"),
        ("record", "Record a single Dual Pika Sense episode from terminal"),
        ("discard", "Discard a recorded episode cleanly"),
    ):
        p_leg = sub.add_parser(legacy_cmd, help=desc, add_help=False)
        p_leg.set_defaults(_legacy_cmd=legacy_cmd)

    # --- import-raw ---
    p_imp = sub.add_parser("import-raw", help="Import Pika raw/HDF5 episodes into robot-agnostic pika_raw dataset")
    p_imp.add_argument("--input-dir", required=True, help="Raw episode root or HDF5 source directory")
    p_imp.add_argument("--output-dir", required=True, help="Target pika_raw dataset directory")
    p_imp.add_argument("--calibration-dir", default=None, help="Calibration directory to fingerprint & embed")
    p_imp.add_argument("--fps", type=int, default=30)
    p_imp.add_argument("--task", default="bimanual manipulation")

    # --- dataset ---
    p_ds = sub.add_parser("dataset", help="Dataset validation, direction check, multi-robot screening, curation & export")
    d_sub = p_ds.add_subparsers(dest="dataset_cmd", required=True)

    p_val = d_sub.add_parser("validate", help="Validate raw sensor sync, tracking accuracy, kinematics & jaw motion")
    p_val.add_argument("dataset_dir")
    p_val.add_argument("--quality-config", default=None)

    p_dir = d_sub.add_parser("direction", help="Detect accidentally recorded scene-reset / reversed episodes")
    p_dir.add_argument("dataset_dir")

    p_scr = d_sub.add_parser("screen", help="Screen retargetability, collision, limits & base swing on target robots")
    p_scr.add_argument("dataset_dir")
    p_scr.add_argument("--robots", nargs="+", default=["piper_pika"], help="Robot names or 'all'")
    p_scr.add_argument("--solver", choices=("cstro", "handumi_lm", "dls_myopic"), default="cstro")
    p_scr.add_argument("--retarget-mode", choices=("absolute-table", "relative-anchor"), default="absolute-table")

    p_qa = d_sub.add_parser("qa", help="Run validate -> direction -> screen -> analyze in one shot")
    p_qa.add_argument("dataset_dir")
    p_qa.add_argument("--robots", nargs="+", default=["piper_pika"])
    p_qa.add_argument("--solver", choices=("cstro", "handumi_lm", "dls_myopic"), default="cstro")

    p_cur = d_sub.add_parser("curate", help="Build a clean subset dataset excluding rejected episodes")
    p_cur.add_argument("dataset_dir")
    p_cur.add_argument("--output-dir", required=True)
    p_cur.add_argument("--robot", default=None, help="Optional robot screening verdict to enforce")
    p_cur.add_argument("--exclude", nargs="*", type=int, default=[])

    p_mrg = d_sub.add_parser("merge", help="Merge multiple compatible pika_raw datasets with provenance")
    p_mrg.add_argument("sources", nargs="+")
    p_mrg.add_argument("--output-dir", required=True)
    p_mrg.add_argument("--task", default=None)

    p_exp = d_sub.add_parser("export", help="Export canonical joint dataset to external follower plugin layout")
    p_exp.add_argument("dataset_dir")
    p_exp.add_argument("--layout", required=True, choices=("bi_piper_pika_follower", "bi_piper_follower", "bi_openarm_follower"))
    p_exp.add_argument("--output-dir", required=True)

    # --- convert ---
    p_conv = sub.add_parser("convert", help="Convert pika_raw dataset to umi-delta (11D/14D VLA) or <robot>-joints")
    p_conv.add_argument("dataset_dir")
    p_conv.add_argument("--output-dir", required=True)
    p_conv.add_argument("--format", choices=("robot-joints", "umi-delta"), default="robot-joints")
    p_conv.add_argument("--robot", default="piper_pika", help="Target robot when --format=robot-joints")
    p_conv.add_argument("--solver", choices=("cstro", "handumi_lm", "dls_myopic"), default="cstro")
    p_conv.add_argument("--retarget-mode", choices=("absolute-table", "relative-anchor"), default="absolute-table")
    p_conv.add_argument("--allow-flagged-episodes", action="store_true")

    # --- convert-v21 ---
    p_v21 = sub.add_parser(
        "convert-v21",
        help="Parallel direct raw Dual Pika -> LeRobot v2.1 joint-action dataset exporter (Parquet + H.264 MP4)",
    )
    p_v21.add_argument("raw_dir", help="Raw Dual Pika root directory containing episode0/, episode1/, ...")
    p_v21.add_argument("--output-dir", required=True, help="Target LeRobot v2.1 dataset directory")
    p_v21.add_argument("--calibration-dir", default=None, help="Calibration directory (session.yaml + pika_tcp.yaml)")
    p_v21.add_argument("--robot", default="piper_pika", help="Target robot profile (default: piper_pika)")
    p_v21.add_argument("--solver", choices=("cstro", "handumi_lm", "dls_myopic"), default="cstro")
    p_v21.add_argument("--retarget-mode", choices=("absolute-table", "relative-anchor"), default="absolute-table")
    p_v21.add_argument("--exclude", default="", help="Comma-separated raw episode IDs to skip (e.g. '18,62,63,64')")
    p_v21.add_argument("--fps", type=int, default=30)
    p_v21.add_argument("--min-segment-seconds", type=float, default=1.0)
    p_v21.add_argument("--workers", type=int, default=8, help="Parallel worker processes (default: 8)")
    p_v21.add_argument("--require-physical-table", action="store_true", help="Require verified physical table calibration")
    p_v21.add_argument("--table-calib", default=None, help="Optional explicit table calibration YAML path")
    p_v21.add_argument("--task", default="bimanual manipulation", help="Fallback task prompt if instructions.json is absent")

    # --- benchmark-retarget ---
    p_bench = sub.add_parser(
        "benchmark-retarget",
        help="Run reproducible C-STRO vs. HandUMI-LM vs. Myopic-DLS multi-robot retargeting ablation",
    )
    p_bench.add_argument("--dataset-dir", default=None, help="Optional pika_raw dataset (defaults to synthetic bimanual towel-fold suite)")
    p_bench.add_argument("--robots", nargs="+", default=["all"], help="Robots to evaluate or 'all'")
    p_bench.add_argument("--output-dir", required=True, help="Directory for benchmark JSON and Markdown tables")
    p_bench.add_argument("--include-ablations", action="store_true", help="Include C-STRO component ablations")

    # --- replay & replay-joints ---
    p_rep = sub.add_parser("replay", help="Offline 3D kinematic retargeting & trajectory diagnostics preview")
    p_rep.add_argument("dataset_dir")
    p_rep.add_argument("--episode", type=int, default=0)
    p_rep.add_argument("--robot", default="piper_pika")
    p_rep.add_argument("--solver", choices=("cstro", "handumi_lm", "dls_myopic"), default="cstro")
    p_rep.add_argument("--output-dir", required=True)

    p_repj = sub.add_parser("replay-joints", help="Validate & render FK / PyBullet 3D sim for a converted joint dataset")
    p_repj.add_argument("dataset_dir")
    p_repj.add_argument("--episode", type=int, default=0)
    p_repj.add_argument("--output-dir", required=True)
    p_repj.add_argument("--sim-video", action="store_true", help="Render 3D PyBullet URDF simulation MP4 (+ wrist videos)")
    p_repj.add_argument("--gui", action="store_true", help="Open interactive 3D PyBullet GUI window while replaying")

    # --- replay-real (dry-run by default) ---
    p_repr = sub.add_parser(
        "replay-real",
        help="Preflight & guarded pyAgxArm / piper_sdk SocketCAN replay for physical Dual Piper",
    )
    p_repr.add_argument("dataset_dir")
    p_repr.add_argument("--episode", type=int, default=0)
    p_repr.add_argument("--robot", default="piper_pika")
    p_repr.add_argument("--sdk", choices=("auto", "pyagxarm", "piper_sdk"), default=None, help="Piper SDK backend (default: auto -> prefers pyAgxArm, falls back to piper_sdk)")
    p_repr.add_argument("--firmware", default=None, help="Piper firmware version string (default: S-V1.8-9 -> PiperFW.V189)")
    p_repr.add_argument("--speed-scale", type=float, default=0.5)
    p_repr.add_argument("--ramp-speed-deg-s", type=float, default=None, help="Slow ramp-to-start joint speed (deg/s)")
    p_repr.add_argument(
        "--smooth-cutoff-hz",
        type=float,
        default=3.0,
        help="Zero-phase Butterworth low-pass cutoff (Hz) on episode trajectory before streaming (0 to disable)",
    )
    p_repr.add_argument(
        "--stream-hz",
        type=float,
        default=100.0,
        help="Clamped C^2 CubicSpline streaming rate (Hz) to eliminate stop-and-go steps when speed-scale < 1 (0 to disable)",
    )
    p_repr.add_argument(
        "--tracking-filter-hz",
        type=float,
        default=5.0,
        help="3rd-order critically-damped online tracking filter cutoff (Hz) in the 200 Hz CAN streamer (0 for trapezoidal)",
    )
    p_repr.add_argument("--dry-run", action="store_true", default=True, help="Offline preflight only (default)")
    p_repr.add_argument("--mock-robot", action="store_true", help="Run full offline ramp+stream against mock robot")
    p_repr.add_argument("--live", action="store_true", help="Execute live over SocketCAN (requires --operator-supported & verified physical table)")
    p_repr.add_argument("--operator-supported", action="store_true")
    p_repr.add_argument("--enable-arms", action="store_true", help="Explicitly energize Piper arms (auto_enable=false by default)")
    p_repr.add_argument("--left-port", default=None, help="Left CAN port override (default from rig.yaml / can0)")
    p_repr.add_argument("--right-port", default=None, help="Right CAN port override (default from rig.yaml / can1)")
    p_repr.add_argument("--speed-percent", type=int, default=None, help="Piper speed percentage (1..100)")
    p_repr.add_argument("--rig-config", default=None, help="Optional rig.yaml override")
    p_repr.add_argument("--json-out", default=None)

    # --- teleop (Flow 2: live Dual Pika Sense -> Dual Piper online retargeting) ---
    p_tel = sub.add_parser(
        "teleop",
        help="Live online Dual Pika Sense -> Dual Piper retargeting via pyAgxArm / piper_sdk with HOLD_LAST_SAFE",
    )
    p_tel.add_argument("--robot", default="piper_pika")
    p_tel.add_argument("--sdk", choices=("auto", "pyagxarm", "piper_sdk"), default=None, help="Piper SDK backend (default: auto -> prefers pyAgxArm)")
    p_tel.add_argument("--firmware", default=None, help="Piper firmware version string (default: S-V1.8-9 -> PiperFW.V189)")
    p_tel.add_argument("--source", choices=("mock", "udp", "docker-ros1"), default="mock")
    p_tel.add_argument("--retarget-mode", choices=("absolute-table", "relative-anchor"), default="absolute-table")
    p_tel.add_argument("--calibration-dir", default=None, help="Optional session calibration directory (session.yaml + pika_tcp.yaml)")
    p_tel.add_argument("--fps", type=float, default=30.0)
    p_tel.add_argument("--steps", type=int, default=None, help="Optional step cap (default: 10 in dry-run, unlimited in --live)")
    p_tel.add_argument("--stale-timeout-s", type=float, default=0.25)
    p_tel.add_argument("--ramp-speed-deg-s", type=float, default=None, help="Slow ramp speed to first pose (deg/s)")
    p_tel.add_argument("--udp-host", default="127.0.0.1")
    p_tel.add_argument("--udp-port", type=int, default=9876)
    p_tel.add_argument("--dry-run", action="store_true", default=True)
    p_tel.add_argument("--mock-robot", action="store_true", default=True)
    p_tel.add_argument("--live", action="store_true", help="Execute live over SocketCAN (requires --operator-supported & verified physical table)")
    p_tel.add_argument("--operator-supported", action="store_true")
    p_tel.add_argument("--enable-arms", action="store_true", help="Explicitly energize Piper arms (auto_enable=false by default)")
    p_tel.add_argument("--left-port", default=None)
    p_tel.add_argument("--right-port", default=None)
    p_tel.add_argument("--speed-percent", type=int, default=None)
    p_tel.add_argument("--rig-config", default=None)
    p_tel.add_argument("--out", default=None)

    # --- policy-bridge (Flow 3: live/mock SE(3) VLA policy bridge) ---
    p_pb = sub.add_parser("policy-bridge", help="Multi-robot SE(3) VLA policy bridge with pyAgxArm / piper_sdk & HOLD_LAST_SAFE")
    p_pb.add_argument("--robot", default="piper_pika")
    p_pb.add_argument("--sdk", choices=("auto", "pyagxarm", "piper_sdk"), default=None, help="Piper SDK backend (default: auto -> prefers pyAgxArm)")
    p_pb.add_argument("--firmware", default=None, help="Piper firmware version string (default: S-V1.8-9 -> PiperFW.V189)")
    p_pb.add_argument("--policy-url", default=None, help="HTTP JSON VLA policy endpoint (e.g. http://127.0.0.1:8000/act)")
    p_pb.add_argument("--task", default="", help="Natural-language task prompt forwarded to the VLA policy server")
    p_pb.add_argument("--n-action-steps", type=int, default=1, help="Number of steps to execute per policy action chunk")
    p_pb.add_argument("--rpc-timeout-s", type=float, default=0.5, help="Policy server RPC timeout before latching HOLD_LAST_SAFE")
    p_pb.add_argument("--dry-run", action="store_true", default=True)
    p_pb.add_argument("--mock-robot", action="store_true", default=True)
    p_pb.add_argument("--live", action="store_true", help="Execute live over SocketCAN (requires --operator-supported, --policy-url, verified physical table)")
    p_pb.add_argument("--operator-supported", action="store_true")
    p_pb.add_argument("--enable-arms", action="store_true", help="Explicitly energize Piper arms (auto_enable=false by default)")
    p_pb.add_argument("--left-port", default=None)
    p_pb.add_argument("--right-port", default=None)
    p_pb.add_argument("--speed-percent", type=int, default=None)
    p_pb.add_argument("--rig-config", default=None)
    p_pb.add_argument("--fps", type=int, choices=(20, 30), default=30)
    p_pb.add_argument("--steps", type=int, default=10, help="Number of steps (0 = unlimited in --live)")
    p_pb.add_argument("--calibration-dir", default=None, help="Optional calibration directory override")
    p_pb.add_argument("--out", default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] in {"sensors", "status", "session", "record", "discard"}:
        return _delegate_legacy_pika_data(raw_argv[0], raw_argv[1:])

    parser = build_parser()
    args = parser.parse_args(raw_argv)

    if args.command == "doctor":
        from pikaumi.doctor import run_doctor_cli

        return run_doctor_cli(json_out=args.json_out, strict=args.strict)

    if args.command == "setup":
        from pikaumi.doctor import run_doctor_cli

        if args.init_rig:
            if RIG_CONFIG.exists() and not args.force:
                print(f"[INFO] {RIG_CONFIG} already exists (use --force to overwrite).")
            else:
                RIG_CONFIG.parent.mkdir(parents=True, exist_ok=True)
                RIG_CONFIG.write_text(RIG_EXAMPLE_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
                print(f"[OK] Created {RIG_CONFIG} from {RIG_EXAMPLE_CONFIG}")
        if args.check or not args.init_rig:
            return run_doctor_cli()
        return 0

    if args.command == "robots":
        from pikaumi.robots.registry import list_robot_names, load_robot_profile
        from pikaumi.config import save_json_atomic

        if args.robots_cmd == "list":
            names = list_robot_names()
            for name in names:
                prof = load_robot_profile(name)
                arms_desc = ", ".join(f"{s}:{len(a.arm_joint_names)}DoF" for s, a in prof.arms.items())
                print(
                    f"  {name:<20} arms=[{arms_desc}]  max_width={prof.gripper_max_width_m*1000:.0f}mm  urdf={prof.urdf_relpath}"
                )
            return 0
        prof = load_robot_profile(args.robot)
        info = prof.to_summary_dict()
        if args.json_out:
            save_json_atomic(args.json_out, info)
        for k, v in info.items():
            print(f"{k}: {v}")
        return 0

    if args.command == "calibrate":
        from pikaumi.calibration.cli_handlers import dispatch_calibrate

        return dispatch_calibrate(args)

    if args.command == "import-raw":
        from pikaumi.dataset.raw import import_pika_episodes

        summary = import_pika_episodes(
            input_dir=Path(args.input_dir),
            output_dir=Path(args.output_dir),
            calibration_dir=Path(args.calibration_dir) if args.calibration_dir else None,
            fps=args.fps,
            task=args.task,
        )
        print(f"[OK] Imported {summary['num_episodes']} episode(s) into {args.output_dir}")
        return 0

    if args.command == "dataset":
        from pikaumi.dataset.cli_handlers import dispatch_dataset

        return dispatch_dataset(args)

    if args.command == "convert":
        from pikaumi.dataset.cli_handlers import dispatch_convert

        return dispatch_convert(args)

    if args.command == "convert-v21":
        from pikaumi.dataset.lerobot_v21 import export_raw_to_lerobot_v21

        exclude_ids = {int(x.strip()) for x in str(args.exclude).split(",") if x.strip()}
        export_raw_to_lerobot_v21(
            raw_dir=Path(args.raw_dir),
            output_dir=Path(args.output_dir),
            calibration_dir=Path(args.calibration_dir) if args.calibration_dir else None,
            robot=args.robot,
            solver=args.solver,
            retarget_mode=args.retarget_mode,
            exclude_ids=exclude_ids,
            fps=args.fps,
            min_segment_seconds=args.min_segment_seconds,
            workers=args.workers,
            require_physical_table=bool(args.require_physical_table),
            table_calib_path=Path(args.table_calib) if args.table_calib else None,
            default_task=args.task,
        )
        return 0

    if args.command == "benchmark-retarget":
        from pikaumi.retargeting.benchmark import run_benchmark_cli

        return run_benchmark_cli(
            dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
            robots=args.robots,
            output_dir=Path(args.output_dir),
            include_ablations=args.include_ablations,
        )

    if args.command in {"replay", "replay-joints"}:
        from pikaumi.sim.replay import dispatch_replay

        return dispatch_replay(args)

    if args.command == "replay-real":
        from pikaumi.real.replay_real import run_replay_real_cli

        return run_replay_real_cli(args)

    if args.command == "teleop":
        from pikaumi.real.teleop import run_teleop_cli

        return run_teleop_cli(args)

    if args.command == "policy-bridge":
        from pikaumi.real.policy_bridge import run_policy_bridge_cli

        return run_policy_bridge_cli(args)

    parser.error(f"Unhandled command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
