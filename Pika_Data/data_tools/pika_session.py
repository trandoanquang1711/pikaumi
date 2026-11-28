"""Desktop controls for collecting a batch before running the existing converter."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from datetime import datetime
import fcntl
import json
from pathlib import Path
import queue
import re
import subprocess
import tempfile
from uuid import uuid4

from pika_options import (EXPORT_DEFAULTS, PREPARE_DEFAULTS, SESSION_DEFAULTS, TIMING_KEYS, WORKSPACE,
                          head_arguments, option_arguments, session_settings, source_filename)


PIKA = Path(__file__).resolve().parents[1]
CLI = WORKSPACE / "pika_data.sh"
SESSIONS = WORKSPACE / "data/pika_sessions"


@contextmanager
def capture_window_lock():
    SESSIONS.mkdir(parents=True, exist_ok=True)
    with (SESSIONS / ".capture.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("A recording window is open. Discard the episode there or close the session before using discard.") from error
        yield


class CaptureClient:
    def __init__(self, container, container_path, ros_root, log, raw_root=None, settings=None):
        self.container = container
        self.container_path = container_path
        self.ros_root = ros_root
        self.log = log
        self.process = None
        self.active = False
        self.raw_root = raw_root
        self.storage = ExitStack()
        self.settings = session_settings(settings)
        self.recording = ExitStack()

    def response(self):
        line = self.process.stdout.readline()
        if not line:
            self.active = False
            raise ValueError(f"Recorder exited. See {self.log}.")
        result = json.loads(line)
        self.active = result["active"]
        if not result["ok"]:
            raise ValueError(f"{result['message']} Xem {self.log}.")
        return result.get("message", "")

    def start(self, episode, instruction):
        with self.log.open("a") as log:
            check = [str(CLI), "status", "--check-streams",
                     *option_arguments(self.settings, (*TIMING_KEYS, "wrist_camera")),
                     *head_arguments(self.settings["use_head"])]
            if self.raw_root is not None:
                check.extend(["--data-root", str(self.raw_root)])
            result = subprocess.run(check, stdout=log, stderr=subprocess.STDOUT, timeout=45)
            if result.returncode:
                raise ValueError(f"Sensor or timestamp alignment check failed. See {self.log}.")
            if self.process is None or self.process.poll() is not None:
                self.close()
                code = (PIKA / "scripts/pika_capture_service.py").read_text()
                try:
                    recorder, target = self.container, self.container_path
                    if self.raw_root is not None:
                        from pika_storage import capture_storage

                        recorder, target = self.storage.enter_context(
                            capture_storage(self.raw_root, self.container, self.ros_root))
                    self.process = subprocess.Popen(
                        ["docker", "exec", "-i", "-e", f"PIKA_USE_HEAD={str(self.settings['use_head']).lower()}",
                         "-e", f"PIKA_WRIST_CAMERA={self.settings['wrist_camera']}",
                         "-e", f"PIKA_EXPORT_FPS={self.settings['fps']}", recorder, "bash", "-lc",
                         'source "$1/install/setup.bash" && export ROS_NAMESPACE="$2" '
                         'ROS_LOG_DIR="$4/.ros_logs" && '
                         'exec python3 -u -c "$3" "$4"', "_", self.ros_root,
                         f"/pika_session_{uuid4().hex}", code, target],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                    )
                    self.response()
                except (OSError, ValueError, subprocess.SubprocessError):
                    self.close()
                    raise
        if self.raw_root is not None:
            from pika_storage import episode_lock

            self.recording.enter_context(episode_lock(self.raw_root, episode))
        self.call("start", episode=episode, instruction=instruction)

    def call(self, operation, **params):
        try:
            try:
                self.process.stdin.write(json.dumps({"operation": operation, **params}) + "\n")
                self.process.stdin.flush()
            except BrokenPipeError as error:
                self.active = False
                raise ValueError(f"Recorder exited. See {self.log}.") from error
            return self.response()
        finally:
            if not self.active:
                self.recording.close()

    def close(self):
        try:
            if self.process is not None:
                self.process.stdin.close()
                try:
                    self.process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                self.process.stdout.close()
        finally:
            self.process = None
            self.active = False
            self.storage.close()
            self.recording.close()


class Session:
    def __init__(self, directory, raw_root, settings=None):
        self.directory = directory
        self.raw_root = raw_root
        self.manifest = directory / "session.json"
        if self.manifest.is_file():
            data = json.loads(self.manifest.read_text())
            if Path(data["raw_root"]) != raw_root:
                raise ValueError(f"This session uses raw directory {data['raw_root']}.")
            self.saved = data["episodes"]
            saved_settings = data.get("settings", {})
            if "use_head" not in saved_settings:
                # Existing recordings predate this switch; preserve their camera schema.
                modes = set()
                for item in self.saved:
                    episode = raw_root / f"episode{item['episode']}"
                    metadata = episode / "capture_metadata.json"
                    if metadata.is_file():
                        modes.add(bool(json.loads(metadata.read_text()).get("head_camera", {}).get("enabled")))
                    elif episode.is_dir():
                        modes.add((episode / "camera/color/headCamera").is_dir())
                if len(modes) > 1:
                    raise ValueError("Session contains mixed head camera modes; use separate sessions per task.")
                saved_settings = {**saved_settings, "use_head": next(iter(modes), False)}
            self.settings = session_settings(saved_settings)
            for key, value in (settings or {}).items():
                if value != self.settings[key]:
                    raise ValueError(f"This session already has {key}={self.settings[key]}. "
                                     "Start a new session or choose a new output root to change processing settings.")
        else:
            if any(directory.iterdir()):
                raise ValueError(f"Session directory already contains data: {directory}")
            self.settings = session_settings(settings)
            self.write_manifest([])

    def write_manifest(self, episodes):
        pending = self.manifest.with_suffix(".tmp")
        data = {"raw_root": str(self.raw_root), "episodes": episodes, "settings": self.settings}
        pending.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        pending.replace(self.manifest)
        self.saved = episodes

    def next_episode(self, start):
        existing = {
            int(match[1]) for path in self.raw_root.glob("episode*")
            if (match := re.fullmatch(r"episode([0-9]+)", path.name))
        }
        episode = start if start is not None else max(existing, default=-1) + 1
        while episode in existing:
            episode += 1
        return episode

    def discard(self, episode):
        from pika_storage import discard_paths, episode_lock

        episode = int(episode)
        if episode < 0:
            raise ValueError("Episode number must be non-negative.")
        name = f"episode{episode}"
        with episode_lock(self.raw_root, episode):
            for dataset in self.directory.glob("lerobot*"):
                reports = dataset / "pika_source_reports.json"
                if reports.is_file() and any(
                        report["episode"] == name for report in json.loads(reports.read_text())):
                    raise ValueError(f"{name} is already in {dataset}. Discard episodes before converting. "
                                     "To exclude an episode from an existing dataset, export the retained episodes to a new destination.")
            raw = self.raw_root / name
            if not raw.exists() and not any(item["episode"] == episode for item in self.saved):
                raise ValueError(f"{name} was not found in the raw directory or session.")
            remaining = [item for item in self.saved if item["episode"] != episode]
            return discard_paths([raw, self.directory / "source" / name, self.directory / "prepared" / name],
                                 lambda: self.write_manifest(remaining))

    def convert(self, progress):
        sources = self.directory / "source"
        prepared = self.directory / "prepared"
        log_path = self.directory / "convert.log"
        output = self.directory / "lerobot"
        version = 2
        while output.exists():
            output = self.directory / f"lerobot_{version}"
            version += 1

        def run(step, *args):
            progress(step)
            command = [str(CLI), *map(str, args)]
            with log_path.open("a") as log:
                print(f"\n{step}", file=log, flush=True)
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise ValueError(f"{step} failed. See {log_path}.")

        episodes = [item["episode"] for item in self.saved]
        for episode in episodes:
            name = f"episode{episode}"
            hdf5 = sources / name / source_filename(self.settings["fps"])
            if not hdf5.is_file():
                run(f"Exporting {name}", "export", episode, "--vista-only",
                    "--data-root", self.raw_root, "--output-dir", hdf5.parent,
                    "--calibration-dir", self.settings["calibration_dir"],
                    *option_arguments(self.settings, EXPORT_DEFAULTS))
            if not all((prepared / name / file).is_file() for file in ("labels.npz", "report.json")):
                run(f"Filtering and labeling {name}", "prepare", episode, "--data-root", self.raw_root,
                    "--source", hdf5, "--prepared-root", prepared,
                    "--calibration-dir", self.settings["calibration_dir"],
                    *option_arguments(self.settings, PREPARE_DEFAULTS))
        run("Packaging LeRobot dataset", "lerobot", *episodes,
            "--prepared-root", prepared, "--output-dir", output, "--repo-id", self.settings["repo_id"])
        return output


def open_session(args, container, container_path, ros_root, raw_root):
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError as error:
        raise ValueError("Install python3-tk to open the recording window.") from error
    with capture_window_lock():
        try:
            root = tk.Tk()
        except tk.TclError as error:
            raise ValueError("Run ./pika_data.sh session from a terminal on the Ubuntu desktop.") from error
        if args.session_dir:
            directory = args.session_dir.expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
        else:
            parent = args.output_root or SESSIONS
            parent.mkdir(parents=True, exist_ok=True)
            directory = Path(tempfile.mkdtemp(prefix=f"{datetime.now():%Y%m%d-%H%M%S}_", dir=parent))
        raw_root.mkdir(parents=True, exist_ok=True)
        settings = {key: getattr(args, key) for key in SESSION_DEFAULTS if getattr(args, key, None) is not None}
        if "calibration_dir" in settings:
            settings["calibration_dir"] = str(settings["calibration_dir"].expanduser().resolve())
        session = Session(directory, raw_root, settings)
        client = CaptureClient(container, container_path, ros_root, directory / "capture.log", raw_root, session.settings)
        try:
            session_window(root, tk, ttk, session, client, args)
        finally:
            client.close()


def session_window(root, tk, ttk, session, client, args):
    root.title("Dual Pika Recording Session")
    frame = ttk.Frame(root, padding=16)
    frame.pack(fill="both", expand=True)
    message = tk.StringVar(value="Ready. Select Start recording when the scene is set up.")
    instruction = tk.StringVar(value=args.instruction or (session.saved[-1]["instruction"] if session.saved else ""))
    number = tk.StringVar()
    ttk.Label(frame, text=f"Raw: {session.raw_root}\nOutput: {session.directory}\n"
              f"Dataset: {session.settings['fps']} Hz · Images: {session.settings['image_size'] or 'original size'} · "
              f"Wrist: {session.settings['wrist_camera']} · Head: {'ON' if session.settings['use_head'] else 'OFF'}",
              wraplength=620).pack(anchor="w")
    ttk.Label(frame, text="Instruction").pack(anchor="w", pady=(12, 0))
    entry = ttk.Entry(frame, textvariable=instruction, width=64)
    entry.pack(fill="x")
    ttk.Label(frame, textvariable=number).pack(anchor="w", pady=8)
    buttons = ttk.Frame(frame)
    buttons.pack(fill="x")
    saved_list = tk.Listbox(frame, height=8, width=74)
    saved_list.pack(fill="both", expand=True, pady=8)
    for item in session.saved:
        saved_list.insert("end", f"episode{item['episode']}  ·  {item['instruction']}")
    ttk.Label(frame, textvariable=message, wraplength=620).pack(anchor="w")
    ttk.Label(frame, text="Discard permanently removes the raw recording and its cache before conversion.",
              wraplength=620).pack(anchor="w", pady=(8, 0))

    progress = queue.SimpleQueue()
    episode = None
    task = ""
    future = None
    discard_target = session.saved[-1]["episode"] if session.saved else None

    def refresh():
        available = future is None
        start_button.configure(state="normal" if available and not client.active else "disabled")
        save_button.configure(state="normal" if available and client.active else "disabled")
        convert_button.configure(state="normal" if available and not client.active and session.saved else "disabled")
        can_discard = discard_target is not None and (
            (session.raw_root / f"episode{discard_target}").exists()
            or any(item["episode"] == discard_target for item in session.saved))
        discard_button.configure(state="normal" if available and can_discard else "disabled",
                                 text=f"Discard episode{discard_target}" if can_discard else "Discard latest episode")
        entry.configure(state="normal" if available and not client.active else "disabled")
        if client.active:
            number.set(f"Recording episode{episode}")
        else:
            number.set(f"Next episode: episode{session.next_episode(args.start_episode)}")

    def submit(label, operation, done):
        nonlocal future
        message.set(label)
        future = executor.submit(operation)
        refresh()

        def poll():
            nonlocal future
            while not progress.empty():
                message.set(progress.get_nowait())
            if not future.done():
                root.after(100, poll)
                return
            completed, future = future, None
            try:
                message.set(done(completed.result()))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                message.set(str(error))
            refresh()

        root.after(100, poll)

    def start():
        nonlocal episode, task, discard_target
        task = instruction.get().strip()
        if not task:
            message.set("Enter a task instruction that matches the demonstration.")
            return
        episode = session.next_episode(args.start_episode)
        discard_target = episode
        submit(f"Checking sensors and timestamp alignment for episode{episode}…", lambda: client.start(episode, task),
               lambda _: f"Recording episode{episode}. Select Save when finished.")

    def save():
        def finish(tracking_summary):
            item = {"episode": episode, "instruction": task}
            session.write_manifest([*session.saved, item])
            saved_list.insert("end", f"episode{episode}  ·  {task}")
            if tracking_summary:
                return f"Saved episode{episode}. Tracking: {tracking_summary}. Review the filtering report after conversion."
            return f"Saved episode{episode}. Ready for the next recording."
        submit(f"Saving episode{episode}…", lambda: client.call("save"), finish)

    def convert():
        def finish(output):
            nonlocal discard_target
            discard_target = None
            return f"Dataset saved to {output}"
        submit("Processing saved episodes…", lambda: session.convert(progress.put),
               finish)

    def discard():
        target = discard_target
        if target is None:
            return
        def operation():
            if client.active:
                client.call("discard")
            return session.discard(target)
        def finish(errors):
            nonlocal discard_target, episode
            discard_target = None
            episode = None
            saved_list.delete(0, "end")
            for item in session.saved:
                saved_list.insert("end", f"episode{item['episode']}  ·  {item['instruction']}")
            if errors:
                return "Episode removed from the session. " + " ".join(errors)
            return f"Deleted episode{target}. You can record this number again."
        submit(f"Stopping and discarding episode{target}…", operation, finish)

    def close():
        if future is not None or client.active:
            message.set("Save or discard the current recording, or wait for the active operation to finish before closing.")
        else:
            root.destroy()

    start_button = ttk.Button(buttons, text="Start recording", command=start)
    save_button = ttk.Button(buttons, text="Save", command=save)
    discard_button = ttk.Button(buttons, text="Discard latest episode", command=discard)
    convert_button = ttk.Button(buttons, text="Convert session", command=convert)
    for button in (start_button, save_button, discard_button, convert_button):
        button.pack(side="left", padx=(0, 8))
    root.protocol("WM_DELETE_WINDOW", close)
    with ThreadPoolExecutor(max_workers=1) as executor:
        refresh()
        root.mainloop()
