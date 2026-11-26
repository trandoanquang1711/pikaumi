"""Map a host raw directory to the existing recorder or a temporary recorder container."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from uuid import uuid4


@contextmanager
def episode_lock(raw_root, episode):
    identity = hashlib.sha256(str(raw_root.resolve()).encode()).hexdigest()[:24]
    path = Path(tempfile.gettempdir()) / f"pika-episode-{os.getuid()}-{identity}-{int(episode)}.lock"
    with path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"episode{episode} is in use. Stop recording before deleting it.") from error
        yield


def remove_directory(path):
    try:
        shutil.rmtree(path)
    except PermissionError:
        # The native Docker recorder can create root-owned files on its bind mount.
        container = os.environ.get("PIKA_CONTAINER", "pika-ros")
        result = subprocess.run(["docker", "inspect", "--format", "{{.Image}}", container],
                                check=True, capture_output=True, text=True, timeout=10)
        if "," in str(path):
            raise ValueError("Paths used for deletion through Docker must not contain commas.")
        code = """from pathlib import Path
import shutil
for item in Path('/discard').iterdir():
    if item.is_dir() and not item.is_symlink():
        shutil.rmtree(item)
    else:
        item.unlink()
"""
        subprocess.run(["docker", "run", "--rm", "--network", "none",
                        "--mount", f"type=bind,source={path},target=/discard",
                        "--entrypoint", "python3", result.stdout.strip(), "-c", code],
                       check=True, capture_output=True, timeout=60)
        path.rmdir()


def discard_paths(paths, commit=None):
    """Remove episode references atomically, then reclaim their staged files."""
    staged = []
    try:
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"Episode or cache is a symbolic link: {path}")
            if not path.exists():
                continue
            if not path.is_dir():
                raise ValueError(f"Expected an episode directory: {path}")
            # A sibling rename also works for root-owned recorder directories:
            # changing their parent would require write permission on the child.
            target = path.with_name(f".discarded-{path.name}-{uuid4().hex}")
            path.rename(target)
            staged.append((path, target))
        if commit is not None:
            commit()
    except Exception:
        for original, target in reversed(staged):
            target.rename(original)
        raise
    errors = []
    for _, target in staged:
        try:
            remove_directory(target)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            errors.append(f"File still present at {target}: {error}")
    return errors


def container_directory(raw_root, mounts):
    matches = []
    for mount in mounts:
        if mount.get("Type") not in ("bind", "volume") or not mount.get("RW", True):
            continue
        source = Path(mount["Source"]).resolve()
        try:
            relative = raw_root.relative_to(source)
        except ValueError:
            continue
        target = PurePosixPath(mount["Destination"]) / relative.as_posix()
        matches.append((len(source.parts), str(target)))
    return max(matches, key=lambda item: item[0])[1] if matches else None


def check_shared_directory(raw_root, container, target):
    """Verify both sides see the same writable directory before recording anything."""
    with tempfile.NamedTemporaryFile(prefix=".pika-write-", dir=raw_root) as probe:
        token = uuid4().hex
        probe.write(token.encode())
        probe.flush()
        script = """from pathlib import Path
import sys
path = Path(sys.argv[1])
if path.read_text() != sys.argv[2]:
    raise ValueError('Host/Docker path mismatch')
path.write_text(sys.argv[2] + '-docker')
"""
        result = subprocess.run(
            ["docker", "exec", container, "python3", "-c", script,
             str(PurePosixPath(target) / Path(probe.name).name), token],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode or Path(probe.name).read_text() != token + "-docker":
            raise ValueError(f"Docker cannot read or write the raw directory {raw_root}. "
                             f"{result.stderr.strip()}")


@contextmanager
def capture_storage(raw_root, sensor_container, ros_root):
    raw_root = raw_root.expanduser().resolve()
    raw_root.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["docker", "inspect", "--format",
         "{{json .State.Running}}\n{{json .Mounts}}\n{{json .Image}}\n"
         "{{json .HostConfig.NetworkMode}}\n{{json .Config.Hostname}}", sensor_container],
        check=True, capture_output=True, text=True, timeout=10,
    )
    running, mounts, image, network, hostname = map(json.loads, result.stdout.splitlines())
    if not running:
        raise ValueError(f"Container {sensor_container} is not running. Start the sensors before recording.")
    target = container_directory(raw_root, mounts)
    if target is not None:
        check_shared_directory(raw_root, sensor_container, target)
        yield sensor_container, target
        return
    if network != "host":
        raise ValueError("Recording outside the current mount requires the sensor container to use --network host.")
    if "," in str(raw_root):
        raise ValueError("Docker --mount requires a raw directory path without commas.")

    recorder = f"pika-recorder-{uuid4().hex}"
    target = "/home/agilex/data"
    result = subprocess.run(
        ["docker", "exec", sensor_container, "bash", "-lc",
         'source "$1/install/setup.bash" && printf %s "${ROS_MASTER_URI:-http://127.0.0.1:11311}"',
         "_", ros_root], check=True, capture_output=True, text=True, timeout=10,
    )
    master_uri = result.stdout.strip()
    try:
        subprocess.run(
            ["docker", "run", "--detach", "--rm", "--name", recorder,
             "--network", "host", "--add-host", f"{hostname}:127.0.0.1",
             "--env", f"ROS_MASTER_URI={master_uri}", "--env", "ROS_IP=127.0.0.1",
             "--mount", f"type=bind,source={raw_root},target={target}",
             "--log-driver", "none", "--entrypoint", "sleep", image, "infinity"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        # Preserve the running sensor container's stream profile, including added IMU streams.
        config = f"{ros_root}/install/share/data_tools/config"
        archive = subprocess.run(["docker", "cp", f"{sensor_container}:{config}/.", "-"],
                                 check=True, capture_output=True, timeout=15)
        subprocess.run(["docker", "cp", "-", f"{recorder}:{config}"], input=archive.stdout,
                       check=True, capture_output=True, timeout=15)
        check_shared_directory(raw_root, recorder, target)
        print(f"Host raw directory: {raw_root}\nRecorder container: {recorder}", flush=True)
        yield recorder, target
    finally:
        subprocess.run(["docker", "rm", "--force", recorder],
                       capture_output=True, timeout=20)
