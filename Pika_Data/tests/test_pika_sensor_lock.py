"""Do not allow two respawning sensor launches to fight for ROS names/USB."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/start_pika_sensors.py"


@pytest.fixture
def sensor():
    spec = importlib.util.spec_from_file_location("pika_sensor_lock", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lock_prevents_concurrent_launch_and_is_released_on_close(tmp_path, sensor):
    lock, proc = tmp_path / "sensors.lock", tmp_path / "proc"
    proc.mkdir()
    fd = sensor.acquire_sensor_lock(lock, proc)
    try:
        assert not os.get_inheritable(fd)
        with pytest.raises(ValueError, match="Keep only one sensor session"):
            sensor.acquire_sensor_lock(lock, proc)
    finally:
        os.close(fd)
    os.close(sensor.acquire_sensor_lock(lock, proc))


@pytest.mark.parametrize("args", [
    ["python3", "/opt/ros/noetic/bin/roslaunch", "sensor_tools", "open_multi_sensor.launch"],
    ["bash", "/root/pika_ros/install/share/sensor_tools/scripts/start_multi_sensor.bash"],
    ["python3", "/root/pika_ros/install/share/sensor_tools/scripts/usb_camera.py"],
    ["/root/pika_ros/install/lib/sensor_tools/serial_gripper_imu"],
    ["/root/pika_ros/install/lib/pika_locator/pika_double_locator_node"],
    ["/root/pika_ros/install/lib/survive-cli", "--force-calibrate"],
])
def test_legacy_sensor_processes_are_detected_without_a_lock(tmp_path, sensor, args):
    proc = tmp_path / "proc"
    process = proc / "99999999"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in args))
    with pytest.raises(ValueError, match="99999999"):
        sensor.acquire_sensor_lock(tmp_path / "lock", proc)
    (process / "cmdline").unlink()
    os.close(sensor.acquire_sensor_lock(tmp_path / "lock", proc))


def test_source_code_arguments_are_not_mistaken_for_live_sensor_processes(tmp_path, sensor):
    process = tmp_path / "proc/99999999"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python3\0-c\0print('open_multi_sensor.launch')\0")
    os.close(sensor.acquire_sensor_lock(tmp_path / "lock", process.parent))


def test_lock_is_not_leaked_to_detached_children(tmp_path, sensor):
    proc, lock = tmp_path / "proc", tmp_path / "lock"
    proc.mkdir()
    fd = sensor.acquire_sensor_lock(lock, proc)
    # Spawn a detached child while parent holds the lock; child must NOT inherit fd.
    child = subprocess.Popen(
        ["/bin/bash", "-c", "printf ready; read -r done"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        assert child.stdout.read(5) == b"ready"
        # While parent still holds fd, another acquisition must fail.
        with pytest.raises(ValueError, match="Keep only one sensor session"):
            sensor.acquire_sensor_lock(lock, proc)
        # Once parent closes fd, lock must be free even while child is still alive!
        os.close(fd)
        fd = -1
        os.close(sensor.acquire_sensor_lock(lock, proc))
    finally:
        if fd >= 0:
            os.close(fd)
        child.communicate(b"done\n", timeout=5)
