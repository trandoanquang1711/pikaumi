#!/usr/bin/env python3
"""Set instructions, validate, and export synchronized Pika episodes."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


STREAMS = {
    "rgb/left_d405": "camera/color/pikaDepthCamera_l",
    "rgb/right_d405": "camera/color/pikaDepthCamera_r",
    "rgb/left_fisheye": "camera/color/pikaFisheyeCamera_l",
    "rgb/right_fisheye": "camera/color/pikaFisheyeCamera_r",
    "depth/left_d405": "camera/depth/pikaDepthCamera_l",
    "depth/right_d405": "camera/depth/pikaDepthCamera_r",
    "pose/left": "localization/pose/pika_l",
    "pose/right": "localization/pose/pika_r",
    "gripper/left": "gripper/encoder/pika_l",
    "gripper/right": "gripper/encoder/pika_r",
    "imu/left": "imu/9axis/pika_l",
    "imu/right": "imu/9axis/pika_r",
}
INTRINSICS = ("left_fisheye_intrinsics.json", "right_fisheye_intrinsics.json")


def synced_files(directory):
    names = (directory / "sync.txt").read_text().splitlines()
    return [directory / name.strip() for name in names if name.strip()]


def timestamp(path):
    return float(path.name.rsplit(".", 1)[0])


def validate(episode):
    failures, counts = [], {}
    for name, relative in STREAMS.items():
        sync = episode / relative / "sync.txt"
        if not sync.is_file():
            failures.append(f"missing {relative}/sync.txt")
            continue
        counts[name] = len(sync.read_text().splitlines())
    if len(counts) != len(STREAMS):
        failures.append("one or more required streams are missing")
    elif not counts or min(counts.values()) == 0:
        failures.append("one or more streams have zero synchronized frames")
    elif len(set(counts.values())) != 1:
        failures.append("synchronized-frame counts do not match")

    try:
        instruction = json.loads((episode / "instructions.json").read_text())["instructions"]
        if not instruction or instruction == ["null"]:
            failures.append("task instruction is missing")
    except Exception:
        failures.append("instructions.json is missing or invalid")
        instruction = []

    for name in INTRINSICS:
        if not (episode / "calibration" / name).is_file():
            failures.append(f"missing calibration/{name}")

    depth = next((episode / STREAMS["depth/left_d405"]).glob("*.png"), None)
    if depth is None:
        failures.append("no raw depth PNG found")
    else:
        header = depth.read_bytes()[:26]
        if header[:8] != b"\x89PNG\r\n\x1a\n" or header[24] != 16 or header[25] != 0:
            failures.append("depth must be 16-bit grayscale PNG")
    return failures, counts, instruction


def set_instruction(episode, instruction):
    path = episode / "instructions.json"
    path.write_text(json.dumps({"instructions": [instruction]}, indent=2) + "\n")
    print(f"Wrote {path}")


def json_rows(files, kind):
    rows = []
    for file in files:
        value = json.loads(file.read_text())
        if kind == "pose":
            rows.append([value[key] for key in ("x", "y", "z", "roll", "pitch", "yaw")])
        elif kind == "gripper":
            rows.append([value["angle"], value["distance"]])
        else:
            orientation = value["orientation"]
            gyro = value["angular_velocity"]
            accel = value["linear_acceleration"]
            rows.append([orientation[k] for k in ("x", "y", "z", "w")] +
                        [gyro[k] for k in ("x", "y", "z")] +
                        [accel[k] for k in ("x", "y", "z")])
    return np.asarray(rows, dtype=np.float32)


def write_file_stream(group, files, h5py):
    encoded = h5py.vlen_dtype(np.dtype("uint8"))
    data = group.create_dataset("encoded", (len(files),), dtype=encoded)
    for index, file in enumerate(files):
        data[index] = np.frombuffer(file.read_bytes(), dtype=np.uint8)
    group.create_dataset("timestamps", data=np.asarray([timestamp(file) for file in files]))
    group.attrs["encoding"] = "jpeg" if files[0].suffix.lower() == ".jpg" else "png"


def export_hdf5(episode, destination):
    try:
        import h5py
    except ModuleNotFoundError as error:
        raise RuntimeError("HDF5 export requires h5py. Run this command inside the pika-ros container.") from error
    failures, counts, instruction = validate(episode)
    if failures:
        raise RuntimeError("Cannot export invalid episode:\n- " + "\n- ".join(failures))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(destination, "w") as output:
        output.attrs["format"] = "pika-synchronized-hdf5-v1"
        output.attrs["frame_count"] = next(iter(counts.values()))
        output.attrs["instruction"] = json.dumps(instruction)
        output.attrs["pose_units"] = "metres, radians (x,y,z,roll,pitch,yaw)"
        output.attrs["imu_note"] = "orientation is unavailable (all-zero quaternion); use Vive pose orientation"

        calibration = output.create_group("calibration")
        string_type = h5py.string_dtype(encoding="utf-8")
        for file in (episode / "calibration").glob("*.json"):
            calibration.create_dataset(file.name, data=file.read_text(), dtype=string_type)

        for name, relative in STREAMS.items():
            files = synced_files(episode / relative)
            group = output.require_group(name)
            if name.startswith(("rgb/", "depth/")):
                write_file_stream(group, files, h5py)
            else:
                kind = name.split("/", 1)[0]
                group.create_dataset("data", data=json_rows(files, kind))
                group.create_dataset("timestamps", data=np.asarray([timestamp(file) for file in files]))
        output.create_dataset("stream_names", data=np.asarray(list(STREAMS), dtype=string_type))
    print(f"Wrote {destination}")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    set_parser = commands.add_parser("set-instruction")
    set_parser.add_argument("episode", type=Path)
    set_parser.add_argument("instruction")
    check_parser = commands.add_parser("validate")
    check_parser.add_argument("episodes", nargs="+", type=Path)
    export_parser = commands.add_parser("export-hdf5")
    export_parser.add_argument("episode", type=Path)
    export_parser.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.command == "set-instruction":
        set_instruction(args.episode, args.instruction)
    elif args.command == "validate":
        status = 0
        for episode in args.episodes:
            failures, counts, instruction = validate(episode)
            print(f"{episode}: {next(iter(counts.values()), 0)} synchronized frames")
            print(f"  instruction: {instruction}")
            if failures:
                status = 1
                for failure in failures:
                    print(f"  FAIL: {failure}")
            else:
                print("  PASS")
        raise SystemExit(status)
    else:
        export_hdf5(args.episode, args.output)


if __name__ == "__main__":
    main()
