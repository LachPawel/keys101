"""Record a narrated keyboard session: LeRobot's recorder plus two microphones on one clock.

- The robot side is `lerobot-record` itself: its `record()` runs in this process with the same flags, so
  every saved frame gets a `time.perf_counter` stamp. FrameClock wraps three LeRobotDataset methods for that.
- voice.wav: your narration from the AirPods, 16 kHz mono.
- room.wav: the MacBook mic hearing the notes, 16 kHz mono. The same mic feeds the `ears` camera.
- manifest.json: session clock, dataset, cameras, the audio clock fits and every episode's frame times,
  rewritten after each saved episode.

Keys during recording are lerobot-record's: right arrow ends an episode early, left arrow re-records it,
Esc stops the session.

    uv run --with-requirements scripts/requirements-keys.txt scripts/record_session.py --episodes 3
    uv run scripts/record_session.py --print-only          # the equivalent lerobot-record command
    uv run --with-requirements scripts/requirements-keys.txt scripts/record_session.py --no-robot 30

Extra lerobot-record flags go after `--`, e.g. `-- --display_data=true`.
"""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing
import queue
import shlex
import signal
import sys
import threading
import time
import wave
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from keys_common import EPISODE_SCRIPTS, MANIFEST, SPOKEN, load_json, write_json

DEFAULT_CAMERAS = Path(__file__).with_name("keys_cameras.json")


class WavRecorder:
    """Records one input device to a 16-bit mono WAV and fits sample index -> session time.

    Every audio block is stamped with PortAudio's ADC time of its first sample, which on macOS is the same
    clock as `time.perf_counter`. If the audio thread ever misses blocks, the gap is filled with silence, so
    the file stays aligned with the clock. A straight-line fit over all blocks gives the time of any sample.
    """

    # 100 ms blocks and PortAudio's "high" latency: a late callback costs nothing instead of dropped audio.
    BLOCK = 1600

    def __init__(self, path: Path, device: str, t0: float, rate: int = 16000):
        self.path, self.device, self.t0, self.rate = path, device, t0, rate
        self._queue: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue()
        self._anchors: list[tuple[int, float]] = []
        self._written = 0
        self._next_t: float | None = None
        self._gap_samples = 0
        self._overflows = 0
        self.device_name = device
        self.started = False

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.rate,
            dtype="int16",
            blocksize=self.BLOCK,
            latency="high",
            callback=self._callback,
        )
        self.device_name = sd.query_devices(self._stream.device)["name"]
        self._open_wav()
        self._writer = threading.Thread(target=self._write_loop, daemon=True)
        self._writer.start()
        self._stream.start()
        self.started = True

    def _open_wav(self) -> None:
        self._wav = wave.open(str(self.path), "wb")  # noqa: SIM115  stays open until stop()
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self.rate)

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status.input_overflow:
            self._overflows += 1
        if time_info.inputBufferAdcTime > 0:
            t_first = time_info.inputBufferAdcTime + (time.perf_counter() - time_info.currentTime)
        else:
            t_first = time.perf_counter() - frames / self.rate - float(self._stream.latency)
        self._queue.put((indata.copy(), t_first - self.t0))

    def _write_block(self, block: np.ndarray, t_first: float) -> None:
        if (
            self._next_t is not None
            and (gap := round((t_first - self._next_t) * self.rate)) > len(block) // 2
        ):
            self._wav.writeframes(np.zeros(gap, dtype=np.int16).tobytes())
            self._written += gap
            self._gap_samples += gap
        self._anchors.append((self._written, t_first))
        self._wav.writeframes(block.tobytes())
        self._written += len(block)
        self._next_t = t_first + len(block) / self.rate

    def _write_loop(self) -> None:
        while (item := self._queue.get()) is not None:
            self._write_block(*item)

    def stop(self) -> dict[str, Any]:
        self._stream.stop()
        self._stream.close()
        self._queue.put(None)
        self._writer.join()
        return self._finish()

    def _finish(self) -> dict[str, Any]:
        self._wav.close()
        nominal = 1 / self.rate
        if len(self._anchors) >= 2:
            n, t = np.asarray(self._anchors, dtype=np.float64).T
            slope, intercept = np.polyfit(n, t, 1)
            if abs(slope / nominal - 1) > 0.01:
                print(
                    f"WARNING {self.path.name}: audio clock off by {slope / nominal - 1:+.2%}; using nominal"
                )
                slope, intercept = nominal, float(np.median(t - n * nominal))
            residual_ms = float(np.std(t - (intercept + slope * n)) * 1000)
        else:
            slope, intercept, residual_ms = nominal, self._anchors[0][1] if self._anchors else 0.0, 0.0
        if self._gap_samples:
            print(
                f"WARNING {self.path.name}: {self._gap_samples / self.rate:.2f} s of audio missed, filled with silence"
            )
        return {
            "file": self.path.name,
            "device": self.device_name,
            "sample_rate": self.rate,
            "n_samples": self._written,
            "t_first_sample": float(intercept),
            "seconds_per_sample": float(slope),
            "fit_residual_ms": round(residual_ms, 3),
            "missed_s": round(self._gap_samples / self.rate, 3),
            "overflows": self._overflows,
        }


def _audio_worker(
    folder: str, devices: dict[str, str], t0: float, ready: Any, stop: Any, results: Any
) -> None:
    """Child process: records every device until `stop` is set. Ctrl-C is left to the parent."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    recorders = {
        name: WavRecorder(Path(folder) / f"{name}.wav", device, t0) for name, device in devices.items()
    }
    try:
        for recorder in recorders.values():
            recorder.start()
        ready.put({name: r.device_name for name, r in recorders.items()})
        stop.wait()
    except Exception as e:
        ready.put(f"{type(e).__name__}: {e}")
    finally:
        results.put({name: r.stop() for name, r in recorders.items() if r.started})


class AudioProcess:
    """Both microphones in their own process, so LeRobot's work (imports, robot connect, video encoding)
    never delays an audio callback. `time.perf_counter` is system-wide on macOS, so the stamps still line up.
    """

    def __init__(self, folder: Path, devices: dict[str, str], t0: float):
        ctx = multiprocessing.get_context("spawn")
        self._ready, self._results, self._stop = ctx.Queue(), ctx.Queue(), ctx.Event()
        self._process = ctx.Process(
            target=_audio_worker,
            args=(str(folder), devices, t0, self._ready, self._stop, self._results),
            daemon=True,
        )
        self.started = False
        self._results_cache: dict[str, Any] | None = None

    def start(self) -> dict[str, str]:
        self._process.start()
        self.started = True
        status = self._ready.get(timeout=30)
        if isinstance(status, str):
            raise RuntimeError(f"Could not open the microphones: {status}")
        return status

    def stop(self) -> dict[str, Any]:
        """Stop recording and return each WAV's clock fit; later calls return the same result."""
        if self._results_cache is None:
            self._stop.set()
            self._results_cache = self._results.get(timeout=30)
            self._process.join(timeout=10)
        return self._results_cache


class Prompter:
    """Shows one key at a time during an episode, every `interval` seconds, with the word to say.

    Runs on its own thread. The reset phase records no frames, so half a second without one means the
    episode is over; it then says so and stops.
    """

    def __init__(self, interval: float):
        self.interval = interval
        self.last_frame = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def frame(self) -> None:
        self.last_frame = time.perf_counter()

    def start(self, episode_index: int, name: str, keys: list[int]) -> None:
        self.stop()
        self._stop = threading.Event()
        self.last_frame = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run, args=(episode_index, name, keys, self._stop), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _wait(self, seconds: float, stop: threading.Event) -> bool:
        """Wait while the episode runs; False once it ended or the prompter was stopped."""
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            if stop.wait(0.05) or time.perf_counter() - self.last_frame > 0.5:
                return False
        return True

    def _run(self, episode_index: int, name: str, keys: list[int], stop: threading.Event) -> None:
        print(
            f"\n{'=' * 64}\n  EPISODE {episode_index} STARTED ({name}): {' '.join(map(str, keys))}\n"
            f"  Recording now. Own pace is fine; press the right arrow when the list is done.\n{'=' * 64}"
        )
        for i, key in enumerate(keys):
            upcoming = f"next {keys[i + 1]}" if i + 1 < len(keys) else "last one"
            print(
                f'\n      ######   {key}   ######    say "{SPOKEN[key]}", press {key}'
                f"      ({i + 1}/{len(keys)}, {upcoming})\n",
                flush=True,
            )
            if not self._wait(self.interval, stop):
                break
        else:
            self._wait(float("inf"), stop)
        if not stop.is_set():
            print(
                "\n  --- EPISODE OVER, not recording. Bring the arm back to the start; you can talk now. ---\n",
                flush=True,
            )


class FrameClock:
    """Stamps every frame LeRobot saves, by wrapping three public LeRobotDataset methods.

    - add_frame: stamp before the frame is buffered; the first frame of an episode starts the prompter.
    - clear_episode_buffer: lerobot-record's re-record (left arrow) drops the pending stamps too.
    - save_episode: the index is `dataset.num_episodes` before saving, the same one `record()` logs.
    - finalize: the dataset is complete, so the session saves its audio and manifest right away. LeRobot
      disconnects the cameras afterwards, and stopping the ears camera's stream can hang in CoreAudio.
    """

    METHODS = ("add_frame", "clear_episode_buffer", "save_episode", "finalize")

    def __init__(self, session: Session):
        self.session = session
        self._pending: list[float] = []

    def __enter__(self) -> FrameClock:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._cls = LeRobotDataset
        self._originals = {name: getattr(LeRobotDataset, name) for name in self.METHODS}
        add_frame, clear, save, finalize = (self._originals[name] for name in self.METHODS)
        clock = self

        def stamped_add_frame(dataset: Any, frame: dict) -> None:
            if not clock._pending:
                clock.session.announce(dataset.num_episodes)
            clock._pending.append(time.perf_counter() - clock.session.t0)
            clock.session.prompter.frame()
            add_frame(dataset, frame)

        def stamped_clear(dataset: Any, *args: Any, **kwargs: Any) -> None:
            clock._pending = []
            clock.session.prompter.stop()
            clear(dataset, *args, **kwargs)

        def stamped_save(dataset: Any, *args: Any, **kwargs: Any) -> None:
            index, stamps = dataset.num_episodes, clock._pending
            clock._pending = []
            save(dataset, *args, **kwargs)
            clock.session.add_episode(dataset, index, stamps)

        def finalize_and_save(dataset: Any) -> None:
            finalize(dataset)
            if clock.session.on_finalize is not None:
                clock.session.on_finalize()

        wrappers = (stamped_add_frame, stamped_clear, stamped_save, finalize_and_save)
        for name, fn in zip(self.METHODS, wrappers, strict=True):
            setattr(LeRobotDataset, name, fn)
        return self

    def __exit__(self, *exc: object) -> None:
        self.session.prompter.stop()
        for name, fn in self._originals.items():
            setattr(self._cls, name, fn)


class Session:
    def __init__(
        self,
        folder: Path,
        plan: list[str],
        cameras: dict | None,
        record_argv: list[str],
        prompt_interval: float = 6.0,
    ):
        import lerobot

        self.folder = folder
        self.plan = plan
        self.prompter = Prompter(prompt_interval)
        self.on_finalize: Callable[[], None] | None = None
        self.t0 = time.perf_counter()
        self.manifest: dict[str, Any] = {
            "session": folder.name,
            "clock": "seconds since t0 on time.perf_counter",
            "t0_unix": time.time(),
            "lerobot_version": lerobot.__version__,
            "record_argv": record_argv,
            "plan": plan,
            "cameras": cameras,
            "dataset": None,
            "audio": {},
            "episodes": [],
        }

    def script(self, episode_index: int) -> tuple[str, list[int]]:
        name = self.plan[episode_index % len(self.plan)]
        return name, EPISODE_SCRIPTS[name]

    def announce(self, episode_index: int) -> None:
        self.prompter.start(episode_index, *self.script(episode_index))

    def add_episode(self, dataset: Any, index: int, stamps: list[float]) -> None:
        name, keys = self.script(index)
        self.manifest["dataset"] = {"repo_id": dataset.repo_id, "root": str(dataset.root), "fps": dataset.fps}
        start_unix = self.manifest["t0_unix"] + stamps[0]
        self.manifest["episodes"].append(
            {
                "episode_index": index,
                "script": name,
                "keys": keys,
                "start": round(stamps[0], 5),
                "end": round(stamps[-1] + 1 / dataset.fps, 5),
                "start_unix": round(start_unix, 3),
                "start_local": datetime.fromtimestamp(start_unix).isoformat(timespec="milliseconds"),
                "n_frames": len(stamps),
                "frame_t": [round(s, 5) for s in stamps],
            }
        )
        self.write()

    def write(self) -> None:
        write_json(self.folder / MANIFEST, self.manifest)


def build_record_argv(args: argparse.Namespace, cameras: dict, extra: list[str]) -> list[str]:
    import json

    return [
        "--robot.type=so101_follower",
        f"--robot.port={args.robot_port}",
        f"--robot.id={args.robot_id}",
        f"--robot.cameras={json.dumps(cameras, separators=(',', ':'))}",
        "--teleop.type=so101_leader",
        f"--teleop.port={args.teleop_port}",
        f"--teleop.id={args.teleop_id}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.single_task={args.task}",
        f"--dataset.num_episodes={args.episodes}",
        f"--dataset.episode_time_s={args.episode_time}",
        f"--dataset.reset_time_s={args.reset_time}",
        f"--dataset.fps={args.fps}",
        # lerobot-record defaults that would break this setup: it pushes to the Hub, and it speaks
        # "Recording episode 1" into the narration mic, which Whisper hears as a key number.
        "--dataset.push_to_hub=false",
        "--play_sounds=false",
        *extra,
    ]


def check_cameras(out_dir: Path, cameras: dict, allow_change: bool) -> None:
    """Every session uses the same camera config, so all episodes can share one dataset later."""
    for path in sorted(out_dir.glob(f"*/{MANIFEST}")):
        previous = load_json(path).get("cameras")
        if previous is not None and previous != cameras and not allow_change:
            sys.exit(
                f"Camera config differs from session {path.parent.name}:\n  then: {previous}\n  now:  {cameras}\n"
                "Restore it, or pass --allow-camera-change to start a separate dataset."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument(
        "--plan", default="A,B,C", help=f"Episode scripts, cycled: {', '.join(EPISODE_SCRIPTS)}"
    )
    parser.add_argument("--episode-time", type=float, default=90)
    parser.add_argument("--interval", type=float, default=6.0, help="Seconds between key prompts")
    parser.add_argument("--reset-time", type=float, default=15)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--repo-id", default="PawelLach/so101_keys_narrated")
    parser.add_argument("--task", default="play the keyboard")
    parser.add_argument("--robot-port", default="/dev/tty.usbmodem5B415331661")
    parser.add_argument("--teleop-port", default="/dev/tty.usbmodem5B415332681")
    parser.add_argument("--robot-id", default="workshop_so101")
    parser.add_argument("--teleop-id", default="workshop_so101")
    parser.add_argument(
        "--voice-mic", default="AirPods Pro", help="Narration input device, or part of its name"
    )
    parser.add_argument(
        "--room-mic", default="MacBook Pro Microphone", help="Input device that hears the notes"
    )
    parser.add_argument("--cameras", type=Path, default=DEFAULT_CAMERAS)
    parser.add_argument("--allow-camera-change", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/keys_sessions"))
    parser.add_argument("--no-robot", type=float, metavar="SECONDS", help="Only record both mics, no arm")
    parser.add_argument("--print-only", action="store_true", help="Print the lerobot-record command and exit")
    args, extra = parser.parse_known_args()
    extra = [a for a in extra if a != "--"]

    plan = args.plan.split(",")
    if unknown := [p for p in plan if p not in EPISODE_SCRIPTS]:
        sys.exit(f"Unknown episode scripts {unknown}; choose from {list(EPISODE_SCRIPTS)}")
    cameras = load_json(args.cameras)
    argv = build_record_argv(args, cameras, extra)
    if args.print_only:
        print(shlex.join(["lerobot-record", *argv]))
        return

    if args.no_robot is None:
        check_cameras(args.out, cameras, args.allow_camera_change)
    folder = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True)
    session = Session(folder, plan, None if args.no_robot is not None else cameras, argv, args.interval)
    audio = AudioProcess(folder, {"voice": args.voice_mic, "room": args.room_mic}, session.t0)

    def save_audio() -> None:
        if audio.started:
            session.manifest["audio"] = audio.stop()
        session.write()

    def save_before_disconnect() -> None:
        save_audio()
        print(
            f"\nAll data saved in {folder}. If the program now hangs while disconnecting the ears camera "
            "(a macOS audio bug), close this terminal tab: nothing will be lost.\n",
            flush=True,
        )

    session.on_finalize = save_before_disconnect
    try:
        mics = audio.start()
        print("Mics: " + ", ".join(f"{name} = {device}" for name, device in mics.items()))
        if args.no_robot is not None:
            run_without_robot(session, args.no_robot)
        else:
            run_record(session, argv, cameras)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        save_audio()
        print(f"\nSession saved in {folder}")


def run_without_robot(session: Session, seconds: float) -> None:
    start = time.perf_counter() - session.t0
    print(f"Recording {seconds:.0f} s. Say a number, then press that key, a few times.")
    with contextlib.suppress(KeyboardInterrupt):
        time.sleep(seconds)
    session.manifest["no_robot"] = {"start": start, "end": time.perf_counter() - session.t0}


def run_record(session: Session, argv: list[str], cameras: dict) -> None:
    import draccus

    from lerobot.scripts.lerobot_record import RecordConfig, record
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    if any(cam.get("type") == "opencv_scaled" for cam in cameras.values()):
        import lerobot_camera_scaled  # noqa: F401  registers the "opencv_scaled" camera type
    if any(cam.get("type") == "spectrogram" for cam in cameras.values()):
        try:
            import lerobot_camera_spectrogram  # noqa: F401  registers the "spectrogram" camera type
        except ImportError:
            sys.exit(
                "The ears camera needs its plugin: run with --with-requirements scripts/requirements-keys.txt"
            )

    cfg = draccus.parse(config_class=RecordConfig, args=argv)
    with FrameClock(session):
        record(cfg)


if __name__ == "__main__":
    main()
