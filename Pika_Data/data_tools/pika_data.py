#!/usr/bin/env python3
"""Record Dual Pika demonstrations, generate relative TCP labels, and export VISTA datasets."""
import argparse
from contextlib import ExitStack
from datetime import datetime
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

from pika_options import (CALIBRATION, WORKSPACE, EXPORT_DEFAULTS, PREPARE_DEFAULTS, TIMING_KEYS, add_export_options,
                          add_prepare_options, add_timing_options, add_head_option, head_arguments, option_arguments, positive_float, source_filename)

PIKA = Path(__file__).resolve().parents[1]
PIKA_CLI = WORKSPACE / "pika"
PREPARED_ROOT = WORKSPACE / "data/pika_vla"
HOST_DATA = PIKA / "data"
VENV_PYTHON = WORKSPACE / ".venv/bin/python"
DATASET_PYTHON = WORKSPACE / ".venv-pika-dataset/bin/python"
CPU_PYTHON = WORKSPACE / ".venv-pika/bin/python"


def episode_index(value):
    match = re.fullmatch(r"(?:episode)?([0-9]+)", value)
    if not match:
        raise argparse.ArgumentTypeError("Enter an episode number, such as 9 or episode9.")
    return str(int(match.group(1)))


def data_root_from_mounts(mounts, container_path):
    """Select the most specific mount, including data paths inside a mount."""
    target = PurePosixPath(container_path)
    if not target.is_absolute() or ".." in target.parts:
        raise ValueError("PIKA_CONTAINER_DATA_DIR must be an absolute path without '..'.")
    matches = []
    for mount in mounts:
        destination = PurePosixPath(mount["Destination"])
        try:
            relative = target.relative_to(destination)
        except ValueError:
            continue
        if mount.get("Type") not in ("bind", "volume") or not mount.get("Source"):
            continue
        source = Path(mount["Source"])
        if not source.is_absolute():
            continue
        matches.append((len(destination.parts), source / str(relative)))
    if not matches:
        raise ValueError("Could not locate the Docker data directory on the host. Pass --data-root PATH.")
    return max(matches, key=lambda item: item[0])[1]


def resolve_data_root(explicit, container, container_path):
    configured = explicit or os.environ.get("PIKA_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Mounts}}", container],
            check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        if container == "pika-ros" and container_path == "/home/agilex/data" and HOST_DATA.is_dir():
            return HOST_DATA
        raise ValueError("Could not inspect the Docker mount. Pass --data-root PATH or set PIKA_DATA_DIR.") from None
    return data_root_from_mounts(json.loads(result.stdout), container_path).resolve()


def export_python():
    configured = os.environ.get("PIKA_PYTHON")
    candidates = ([configured] if configured else [
        str(VENV_PYTHON), str(DATASET_PYTHON), str(CPU_PYTHON), shutil.which("python3"), str(Path.home() / "anaconda3/bin/python3")])
    for candidate in dict.fromkeys(candidates):
        if not candidate or not Path(candidate).is_file():
            continue
        try:
            result = subprocess.run([candidate, "-c", "import cv2, h5py, numpy, scipy"],
                                    capture_output=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    raise ValueError("Run ./install.sh to create .venv, or set PIKA_PYTHON to a Python environment with numpy, scipy, h5py, and opencv.")


def export_source(args, container, container_path):
    root = resolve_data_root(args.data_root, container, container_path)
    episode = f"episode{args.episode}"
    source = root / episode
    if not source.is_dir():
        raise ValueError(f"Episode not found: {source}")
    destination = (args.output_dir or (
        args.output_root / "source" / episode if args.output_root else
        WORKSPACE / "data/pika_exports" / episode)).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"Output already exists and will not be overwritten: {destination}")
    calibration = args.calibration_dir.expanduser().resolve()
    for side in ("left", "right"):
        if not (calibration / f"{side}_fisheye_intrinsics.json").is_file():
            raise ValueError(f"Missing {side} camera intrinsics in {calibration}")
    command = [export_python(), str(PIKA / "data_tools/prepare_pika_dataset.py"),
               "--data-root", str(root), "--episodes", episode,
               "--calibration-dir", str(calibration), "--output-dir", str(destination)]
    command.extend(option_arguments(vars(args), EXPORT_DEFAULTS))
    if args.vista_only:
        command.append("--vista-only")
    if getattr(args, "exclude_head", False):
        command.append("--exclude-head")
    print(f"Source: {source}\nOutput: {destination}", flush=True)
    subprocess.run(command, check=True)
    next_step = f"./pika_data.sh prepare {args.episode}"
    if args.data_root or os.environ.get("PIKA_DATA_DIR"):
        next_step += f" --data-root {shlex.quote(str(root))}"
    if args.output_dir or args.fps != 20:
        next_step += f" --source {shlex.quote(str(destination / source_filename(args.fps)))}"
    if args.output_root:
        next_step += f" --output-root {shlex.quote(str(args.output_root))}"
    print(f"Exported {args.fps} Hz source HDF5 with 14-dimensional actions. Next step: {next_step}")


def status(container, container_path, ros_root, raw_root=None, sample_timeout=5.0):
    result = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", container],
                            check=True, capture_output=True, text=True, timeout=10)
    if result.stdout.strip() != "true":
        raise ValueError(f"Container {container} is not running.")
    print(f"Docker container {container} is running.", flush=True)
    print(f"Host data: {resolve_data_root(raw_root, container, container_path)}", flush=True)
    for suffix in ("l", "r"):
        topic = f"/pika_localization_status_{suffix}"
        result = subprocess.run([
            "docker", "exec", container, "bash", "-lc",
            'source "$1/install/setup.bash" && timeout "$3" rostopic echo -n 1 "$2"',
            "_", ros_root, topic, f"{sample_timeout:g}s"],
            capture_output=True, text=True, timeout=sample_timeout + 5)
        sample = result.stdout.strip() or result.stderr.strip() or f"No sample received within {sample_timeout:g} seconds."
        print(f"{topic}:\n{sample}", flush=True)
        if result.returncode or not re.search(r"accurate:\s*(?:True|true)\b", result.stdout):
            raise ValueError("No valid tracking sample. Check the sensors and Vive before recording.")
    print("Both tracking samples are valid. Tracking may change during recording.", flush=True)


def main(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    delegated_pika_cmds = {
        "policy-bridge": ["policy-bridge"],
        "review": ["replay"],
        "review-joints": ["replay-joints"],
        "piper-contact-calibration": ["calibrate", "table-robot"],
    }
    if arguments and arguments[0] in delegated_pika_cmds:
        subprocess.run([str(PIKA_CLI), *delegated_pika_cmds[arguments[0]], *arguments[1:]], check=True)
        return
    if arguments and arguments[0] == "tcp-axes":
        subprocess.run([export_python(), str(PIKA / "data_tools/pika_tcp_axes.py"), *arguments[1:]], check=True)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation")
    sub.add_parser("policy-bridge", help="Run the standardized multi-robot policy bridge (--dry-run by default)")
    sub.add_parser("tcp-axes", help="Measure Pika or Piper TCP axes from pose feedback")
    sub.add_parser("piper-contact-calibration", help="Estimate T_robot_table via ./pika calibrate table-robot")
    sub.add_parser("review", help="Render offline 3D kinematic retargeting preview via ./pika replay")
    sub.add_parser("review-joints", help="Validate & render FK diagnostics for converted joints via ./pika replay-joints")
    sensors = sub.add_parser("sensors", help="Start Pika sensors in a dedicated terminal and keep it open")
    sensors.add_argument("--camera-fps", type=int, choices=(15, 30),
                         help="Raw camera frame rate. Uses the startup script setting by default, currently 30")
    check = sub.add_parser("status", help="Read Docker and tracking status")
    raw_help = "Host directory containing raw episodes. Defaults to PIKA_DATA_DIR or the Docker mount"
    output_help = "Host output directory for source/, prepared/, lerobot/, or capture sessions"
    check.add_argument("--data-root", type=Path, help=raw_help)
    check.add_argument("--timeout", type=positive_float, default=5.0,
                       help="Tracking timeout per hand in seconds. Default: 5")
    check.add_argument("--check-streams", action="store_true",
                       help="Also check six streams and timestamp alignment over approximately three seconds")
    add_timing_options(check)
    check.set_defaults(**{key: EXPORT_DEFAULTS[key] for key in TIMING_KEYS})
    record = sub.add_parser("record", help="Record a dual-Pika episode. Press Enter to finish")
    record.add_argument("episode", type=episode_index, help="Unused episode number, such as 10 or episode10")
    record.add_argument("instruction", nargs="?", help="Task instruction. Omit to enter it interactively")
    record.add_argument("--data-root", type=Path, help=raw_help)
    session = sub.add_parser("session", help="Open the recording, saving, and conversion window")
    session.add_argument("--session-dir", type=Path, help="Reopen a session with its saved raw path and processing settings")
    session.add_argument("--start-episode", type=lambda value: int(episode_index(value)),
                         help="Find the first free episode number at or above N. Defaults to the next number in the raw directory")
    session.add_argument("--instruction", help="Initial task instruction. Defaults to the last instruction in the session, if available")
    session.add_argument("--data-root", type=Path, help=raw_help)
    session.add_argument("--output-root", type=Path, help=output_help)
    add_export_options(session)
    add_prepare_options(session)
    session.add_argument("--calibration-dir", type=Path, help="Directory containing camera intrinsics and Pika TCP files. Default: data/calibration")
    session.add_argument("--repo-id", help="Dataset namespace/name. Default: local/dual_pika")
    discard = sub.add_parser("discard", help="Permanently delete a failed recording before conversion")
    discard.add_argument("episode", type=episode_index, help="Episode to discard, such as 10 or episode10")
    discard.add_argument("--data-root", type=Path, help=raw_help)
    discard.add_argument("--session-dir", type=Path,
                         help="Session containing the episode. Removes raw data and cache and updates session.json. Omit to delete raw data only")
    sync = sub.add_parser("sync", help="Optionally generate sync.txt with the legacy Pika tool")
    sync.add_argument("episode", type=episode_index, help="Recorded episode number, such as 9")
    sync.add_argument("--data-root", type=Path, help=raw_help)
    export = sub.add_parser("export", help="Export source HDF5 from raw streams without requiring sync.txt")
    export.add_argument("episode", type=episode_index, help="Raw episode number to export, such as 9")
    export.add_argument("--data-root", type=Path, help=raw_help)
    export.add_argument("--exclude-head", action="store_true", help="Exclude head from exported data and synchronization")
    export.add_argument("--output-dir", type=Path,
                        help="Exact HDF5 output directory. Defaults to data/pika_exports/episodeN or output-root/source/episodeN")
    export.add_argument("--output-root", type=Path, help=output_help)
    export.add_argument("--calibration-dir", type=Path, default=CALIBRATION,
                        help="Fisheye intrinsics directory. Default: data/calibration")
    add_export_options(export)
    export.set_defaults(**EXPORT_DEFAULTS)
    export.set_defaults(max_image_skew_ms=None)
    profile = export.add_mutually_exclusive_group()
    profile.add_argument("--vista-only", dest="vista_only", action="store_true", default=True,
                         help="Export wrist fisheye images, poses, and gripper data for VISTA (default)")
    profile.add_argument("--all-streams", dest="vista_only", action="store_false",
                         help="Include D405 RGB and depth. Requires both streams in the recording")
    prepare = sub.add_parser("prepare", help="Filter tracking jumps and generate relative TCP labels")
    prepare.add_argument("episode", type=episode_index, help="Episode number to label, such as 9")
    prepare.add_argument("--data-root", type=Path, help=raw_help)
    prepare.add_argument("--source", type=Path, help="Exact source HDF5 path. Defaults to the only file in the episode export directory")
    prepare.add_argument("--prepared-root", type=Path,
                         help="Parent directory for episodeN labels. Defaults to data/pika_vla or output-root/prepared")
    prepare.add_argument("--output-root", type=Path, help=output_help)
    add_prepare_options(prepare)
    prepare.set_defaults(**PREPARE_DEFAULTS)
    prepare.add_argument("--calibration-dir", type=Path, default=CALIBRATION,
                         help="Calibration directory containing Pika TCP files, optionally under tcp/. Default: data/calibration")
    lerobot = sub.add_parser("lerobot", help="Package prepared episodes into a VLA dataset")
    lerobot.add_argument("episodes", type=episode_index, nargs="+", help="Prepared episode numbers, such as 8 9")
    lerobot.add_argument("--prepared-root", type=Path,
                         help="Input label directory. Defaults to data/pika_vla or output-root/prepared")
    lerobot.add_argument("--output-dir", type=Path,
                         help="Create .venv-pika using the README, or set PIKA_VLA_PYTHON to the VISTA Python environment.")
    lerobot.add_argument("--output-root", type=Path, help=output_help)
    lerobot.add_argument("--repo-id", default="local/dual_pika", help="Dataset namespace/name. Default: local/dual_pika")
    lerobot.add_argument("--format-version", choices=("v2.1", "v3.0"), default="v3.0")
    lerobot.add_argument("--representation", choices=("umi", "pika-tcp", "piper-joint"), default="umi",
                         help="Export UMI deltas, absolute Pika TCP poses, or Piper joint targets")
    lerobot.add_argument("--ik-calibration", type=Path,
                         help="Dual Piper base and TCP transforms for the piper-joint representation")
    for head_parser in (sensors, check, record, session):
        add_head_option(head_parser, default=False if head_parser is record else None)
    args = parser.parse_args(argv)
    if args.operation is None:
        parser.print_help()
        return
    container = os.environ.get("PIKA_CONTAINER", "pika-ros")
    container_path = os.environ.get("PIKA_CONTAINER_DATA_DIR", "/home/agilex/data")
    ros_root = os.environ.get("PIKA_ROS_ROOT", "/root/pika_ros")
    output_root = getattr(args, "output_root", None)
    if output_root:
        args.output_root = output_root = output_root.expanduser().resolve()
    if args.operation in ("prepare", "lerobot"):
        args.prepared_root = (args.prepared_root or (
            output_root / "prepared" if output_root else PREPARED_ROOT)).expanduser().resolve()
    if args.operation == "export":
        export_source(args, container, container_path)
    elif args.operation == "prepare":
        episode = f"episode{args.episode}"
        source = args.source
        if source is None:
            directory = (output_root / "source" if output_root else WORKSPACE / "data/pika_exports") / episode
            candidates = sorted(directory.glob("pika_source_*hz.hdf5"))
            if len(candidates) > 1:
                raise ValueError(f"Multiple HDF5 files found in {directory}. Select one with --source PATH.")
            source = candidates[0] if candidates else directory / source_filename(20)
        source = source.expanduser().resolve()
        if not source.is_file():
            raise ValueError(f"Missing source HDF5 at {source}. Run ./pika_data.sh export {args.episode}.")
        output = args.prepared_root / episode
        if output.exists():
            raise ValueError(f"Output already exists and will not be overwritten: {output}")
        raw = resolve_data_root(args.data_root, container, container_path) / episode
        if not raw.is_dir():
            raise ValueError(f"Episode not found: {raw}")
        calibration = args.calibration_dir.expanduser().resolve()
        tcp_calibration = calibration / "tcp"
        if not tcp_calibration.is_dir():
            tcp_calibration = calibration
        command = [export_python(), str(PIKA / "data_tools/prepare_pika_vla.py"), "prepare",
                   str(source), str(raw), str(tcp_calibration), str(output)]
        command.extend(option_arguments(vars(args), PREPARE_DEFAULTS))
        subprocess.run(command, check=True)
    elif args.operation == "lerobot":
        if args.representation == "pika-tcp" and args.format_version != "v3.0":
            parser.error(f"--representation {args.representation} requires LeRobot v3.0")
        if (args.representation == "piper-joint") != (args.ik_calibration is not None):
            parser.error("--ik-calibration is required for piper-joint and applies only to that representation")
        args.output_dir = (args.output_dir or (
            output_root / "lerobot" if output_root else WORKSPACE / "data/lerobot/dual_pika"
        )).expanduser().resolve()
        if len(set(args.episodes)) != len(args.episodes):
            raise ValueError("List each episode only once.")
        prepared = [args.prepared_root / f"episode{n}" for n in args.episodes]
        for n, path in zip(args.episodes, prepared):
            if not all((path / name).is_file() for name in ("report.json", "labels.npz")):
                raise ValueError(f"Missing labels at {path}. Run ./pika_data.sh prepare {n}.")
        if args.output_dir.exists():
            raise ValueError(f"Output already exists at {args.output_dir}. Choose a new --output-dir.")
        python = os.environ.get("PIKA_VLA_PYTHON") or (
            str(VENV_PYTHON) if VENV_PYTHON.is_file() else (
                str(DATASET_PYTHON) if DATASET_PYTHON.is_file() else (
                    str(CPU_PYTHON) if CPU_PYTHON.is_file() else (shutil.which("python") or shutil.which("python3")))))
        if not python or not shutil.which(python):
            raise ValueError("Run ./install.sh to create .venv (or .venv-pika-dataset), or set PIKA_VLA_PYTHON.")
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(WORKSPACE / "src"), str(PIKA / "data_tools"), env.get("PYTHONPATH")]))
        env["HF_HUB_OFFLINE"] = "1"
        env["use_corr_state"] = "false"
        # The reader verifies the dataset before publication, so its Arrow cache is temporary.
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".pika-cache-", dir=args.output_dir.parent) as cache:
            env["HF_DATASETS_CACHE"] = str(Path(cache) / "datasets")
            temporary = Path(cache) / "tmp"
            temporary.mkdir()
            env.update(TMPDIR=str(temporary), TMP=str(temporary), TEMP=str(temporary))
            subprocess.run([python, str(PIKA / "data_tools/prepare_pika_vla.py"), "lerobot",
                            *map(str, prepared), "--output-dir", str(args.output_dir),
                            "--repo-id", args.repo_id,
                            *(["--representation", args.representation] if args.representation != "umi" else []),
                            *(["--ik-calibration", str(args.ik_calibration.expanduser())] if args.ik_calibration else []),
                            *(["--format-version", args.format_version] if args.format_version != "v3.0" else [])], check=True, env=env)
    elif args.operation == "status":
        status(container, container_path, ros_root, args.data_root, args.timeout)
        if args.check_streams:
            subprocess.run([export_python(), str(PIKA / "data_tools/pika_sync.py"),
                            "--container", container, "--ros-root", ros_root,
                            *option_arguments(vars(args), TIMING_KEYS), *head_arguments(args.use_head)], check=True)
    elif args.operation == "session":
        from pika_session import open_session

        explicit = args.data_root or os.environ.get("PIKA_DATA_DIR")
        if not explicit and args.session_dir:
            manifest = args.session_dir.expanduser().resolve() / "session.json"
            if manifest.is_file():
                explicit = json.loads(manifest.read_text())["raw_root"]
        raw_root = resolve_data_root(explicit, container, container_path)
        open_session(args, container, container_path, ros_root, raw_root)
    elif args.operation == "discard":
        from pika_session import Session, capture_window_lock
        from pika_storage import discard_paths, episode_lock

        with capture_window_lock():
            if args.session_dir:
                directory = args.session_dir.expanduser().resolve()
                manifest = directory / "session.json"
                if not manifest.is_file():
                    raise ValueError(f"Missing session manifest: {manifest}")
                explicit = args.data_root or os.environ.get("PIKA_DATA_DIR") or json.loads(manifest.read_text())["raw_root"]
                raw_root = resolve_data_root(explicit, container, container_path)
                errors = Session(directory, raw_root).discard(args.episode)
            else:
                raw_root = resolve_data_root(args.data_root, container, container_path)
                path = raw_root / f"episode{args.episode}"
                with episode_lock(raw_root, args.episode):
                    if not path.exists() and not path.is_symlink():
                        raise ValueError(f"Episode not found: {path}")
                    errors = discard_paths([path])
            if errors:
                raise ValueError("Episode removed from the session. Some files could not be deleted. " + " ".join(errors))
            print(f"Deleted episode{args.episode} at {raw_root}. You can record this number again.")
    elif args.operation == "sensors":
        # The existing Pika setup stores tracker serial exports in ~/.bashrc.
        code = (PIKA / "scripts/start_pika_sensors.py").read_text()
        config = (PIKA / "config/sensors.yaml").read_text()
        head_env = [] if args.use_head is None else ["-e", f"PIKA_USE_HEAD={str(args.use_head).lower()}"]
        subprocess.run(["docker", "exec", "-it", *head_env, container, "bash", "-lic",
                        'source "$1/install/setup.bash" && exec python3 -c "$2" "$1" "$3" "$4"',
                        "_", ros_root, code, str(args.camera_fps) if args.camera_fps is not None else "", config],
                       check=True)
    else:
        script = "record_pika_episode.sh" if args.operation == "record" else "sync_pika_episode.sh"
        command = ["bash", str(PIKA / "scripts" / script), "multi", args.episode]
        if args.operation == "record" and args.instruction is not None:
            command.append(args.instruction)
        with ExitStack() as resources:
            if args.operation == "record":
                from pika_storage import episode_lock

                raw_root = resolve_data_root(args.data_root, container, container_path)
                resources.enter_context(episode_lock(raw_root, args.episode))
            if args.data_root or os.environ.get("PIKA_DATA_DIR"):
                from pika_storage import capture_storage

                raw_root = resolve_data_root(args.data_root, container, container_path)
                recorder, target = resources.enter_context(capture_storage(raw_root, container, ros_root))
                env = os.environ.copy()
                env.update(PIKA_CONTAINER=recorder, PIKA_CONTAINER_DATA_DIR=target,
                           PIKA_DATA_DIR=str(raw_root))
                if args.operation == "record":
                    env["PIKA_USE_HEAD"] = str(args.use_head).lower()
                print(f"Host raw directory: {raw_root}", flush=True)
                subprocess.run(command, check=True, env=env)
            else:
                if args.operation == "record":
                    env = os.environ.copy()
                    env["PIKA_USE_HEAD"] = str(args.use_head).lower()
                    subprocess.run(command, check=True, env=env)
                else:
                    subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
    except subprocess.CalledProcessError as error:
        if error.stderr:
            print(error.stderr.strip(), file=sys.stderr)
        raise SystemExit(error.returncode)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Capture or export stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
