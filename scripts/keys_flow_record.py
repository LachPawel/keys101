#!/usr/bin/env python
"""Record note after note without stopping, with the arm staying over the keyboard.

    uv run --with-requirements scripts/requirements-keys.txt scripts/keys_flow_record.py --per-note 25

You hold the leader arm so the fingertip hovers over the keyboard. The script names a note, you move to it
and press it, and the moment MIDI hears a note the script names the next one. No stops, no return to the
parked pose.

MIDI cuts and labels the recording on its own:
  - one episode runs from just after the previous note sounded to just after this one sounded, so it shows
    "lift from the key you are on, go to the named key, press it";
  - its label is the named note ("play mi, the third key");
  - if the wrong key sounded, that episode is thrown away and the note is asked again later.

The piano must be running with the ears cable (keys_synth.py --ears "BlackHole 2ch"). Ctrl+C stops early;
everything recorded up to then is kept. The raw recording is saved before the dataset is written, and
`--from-raw <folder>` writes the dataset again from it.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from keys_common import load_json, write_json
from keys_play import ORDINALS, task_text
from keys_synth import note_name
from keys_teleop_check import TASK_NOTES

ROOT = Path(__file__).resolve().parent.parent
HOLD_S = 0.25  # an episode ends this long after its note sounds; the same delay the song player uses
BOUNCE_S = 0.7  # a second note-on of the same key this soon is the key bouncing, not a new press
MIN_S, MAX_S = 0.5, 8.0  # an episode shorter or longer than this is not a clean press
SPOKEN = {"do": "doh", "re": "ray", "mi": "mee", "fa": "fah", "sol": "soul", "la": "lah"}


@dataclass
class Segment:
    note: str
    start: float
    end: float
    heard: int

    @property
    def ok(self) -> bool:
        return self.heard % 12 == TASK_NOTES[self.note] and MIN_S <= self.end - self.start <= MAX_S


@dataclass
class Sequencer:
    """Which note to ask for next, and where one episode ends and the next begins."""

    queue: list[str]
    start: float = 0.0
    retries: int = 2
    index: int = 0
    segments: list[Segment] = field(default_factory=list)
    last_press: tuple[float, int] | None = None
    tries: dict[int, int] = field(default_factory=dict)

    @property
    def current(self) -> str | None:
        return self.queue[self.index] if self.index < len(self.queue) else None

    @property
    def upcoming(self) -> str | None:
        return self.queue[self.index + 1] if self.index + 1 < len(self.queue) else None

    def press(self, t: float, midi_note: int) -> Segment | None:
        """Feed every note-on. Returns the finished segment, or None for a bounce or after the end."""
        if self.last_press and self.last_press[1] == midi_note and t - self.last_press[0] < BOUNCE_S:
            return None
        self.last_press = (t, midi_note)
        note = self.current
        if note is None:
            return None
        begin = self.segments[-1].end if self.segments else self.start
        segment = Segment(note, begin, t + HOLD_S, midi_note)
        self.segments.append(segment)
        if not segment.ok and self.tries.get(self.index, 0) < self.retries:
            # Ask for it again at the end, so every note still gets its share of good presses.
            self.tries[len(self.queue)] = self.tries.get(self.index, 0) + 1
            self.queue.append(note)
        self.index += 1
        return segment


def balanced_queue(per_note: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    queue = [note for note in ORDINALS for _ in range(per_note)]
    rng.shuffle(queue)
    return [str(note) for note in queue]


def say(note: str | None) -> None:
    if note:
        subprocess.Popen(
            ["say", "-r", "230", SPOKEN[note]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )


def encode(image: np.ndarray, ext: str) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if ext == ".jpg" else []
    return cv2.imencode(ext, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params)[1].tobytes()


def decode(data: bytes) -> np.ndarray:
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def record(args: argparse.Namespace, session_dir: Path) -> dict[str, Any]:
    from keys_scripted import make_robot, open_midi

    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderTeleopConfig
    from lerobot.utils.robot_utils import precise_sleep

    cameras = load_json(args.cameras)
    robot = make_robot(
        argparse.Namespace(robot_port=args.robot_port, robot_id=args.robot_id, max_step=None), cameras
    )
    leader = SOLeader(SOLeaderTeleopConfig(port=args.teleop_port, id=args.teleop_id))
    queue = balanced_queue(args.per_note, args.seed)
    frames: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    midi = open_midi("auto", t0, session_dir / "midi.jsonl")
    if midi is None:
        sys.exit("No MIDI keyboard found. It cuts and labels the recording, so it has to be there.")

    robot.connect()
    leader.connect()
    sequencer = Sequencer(queue)
    handled = 0
    try:
        input(
            f"\n{len(queue)} notes, about {len(queue) * 3 / 60:.0f} minutes.\n"
            "Hold the leader so the fingertip hovers over the keyboard, then press Enter. "
        )
        sequencer.start = time.perf_counter() - t0
        handled = len(midi.events)
        print(
            f"\n>>> {sequencer.current.upper()}   (then {(sequencer.upcoming or 'the end').upper()})",
            flush=True,
        )
        say(sequencer.current)
        tick = 0
        loop_start = time.perf_counter()
        while sequencer.current is not None:
            obs = robot.get_observation()
            action = leader.get_action()
            robot.send_action(action)
            images = {k: v for k, v in obs.items() if isinstance(v, np.ndarray)}
            frames.append(
                {
                    "t": time.perf_counter() - t0,
                    "state": {k: float(v) for k, v in obs.items() if k not in images},
                    "action": {k: float(v) for k, v in action.items()},
                    "images": {
                        k: encode(v, ".png" if v.shape[0] <= 128 else ".jpg") for k, v in images.items()
                    },
                }
            )
            for t, midi_note, velocity in list(midi.events[handled:]):
                handled += 1
                if not velocity:
                    continue
                segment = sequencer.press(t, midi_note)
                if segment is None:
                    continue
                mark = "ok" if segment.ok else "WRONG KEY, asked again later"
                good = sum(s.ok for s in sequencer.segments)
                print(
                    f"    {segment.note:3s} -> {note_name(midi_note):3s} {mark}   [{good} good]", flush=True
                )
                if sequencer.current:
                    then = (sequencer.upcoming or "the end").upper()
                    print(f">>> {sequencer.current.upper()}   (then {then})", flush=True)
                    say(sequencer.current)
            tick += 1
            precise_sleep(loop_start + tick / args.fps - time.perf_counter())
        # Keep the last press's tail.
        end = time.perf_counter() + HOLD_S + 0.2
        while time.perf_counter() < end:
            obs = robot.get_observation()
            action = leader.get_action()
            robot.send_action(action)
            images = {k: v for k, v in obs.items() if isinstance(v, np.ndarray)}
            frames.append(
                {
                    "t": time.perf_counter() - t0,
                    "state": {k: float(v) for k, v in obs.items() if k not in images},
                    "action": {k: float(v) for k, v in action.items()},
                    "images": {
                        k: encode(v, ".png" if v.shape[0] <= 128 else ".jpg") for k, v in images.items()
                    },
                }
            )
            tick += 1
            precise_sleep(loop_start + tick / args.fps - time.perf_counter())
        print("\nAll notes done. You can put the arm down.")
        subprocess.Popen(["say", "done"])
    except KeyboardInterrupt:
        print("\nStopped early. Keeping what was recorded.")
    except Exception as error:  # noqa: BLE001  an arm dropping off USB must not lose the session
        print(f"\nRecording broke off ({type(error).__name__}: {error}). Keeping what was recorded.")
    finally:
        midi.close()
        for device in (leader, robot):
            try:
                device.disconnect()
            except Exception as error:  # noqa: BLE001
                print(f"(could not disconnect cleanly: {type(error).__name__})")

    raw = {
        "fps": args.fps,
        "repo_id": args.repo_id,
        "cameras": cameras,
        "robot_port": args.robot_port,
        "robot_id": args.robot_id,
        "frames": frames,
        "segments": [vars(s) for s in sequencer.segments],
    }
    with (session_dir / "raw.pkl").open("wb") as f:
        pickle.dump(raw, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Raw recording saved: {session_dir / 'raw.pkl'} ({len(frames)} frames).")
    return raw


def write_dataset(raw: dict[str, Any], session_dir: Path) -> Path | None:
    """Cut the recording into one episode per good press and write a LeRobot dataset."""
    from keys_scripted import make_robot

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    segments = [Segment(**s) for s in raw["segments"]]
    good = [s for s in segments if s.ok]
    print(f"\n{len(segments)} presses, {len(good)} good.")
    for note in ORDINALS:
        print(
            f"  {note:3s}: {sum(s.note == note for s in good)} good of {sum(s.note == note for s in segments)}"
        )
    if not good:
        print("Nothing to write.")
        return None

    # The robot object is only asked which features it has; it is not connected.
    robot = make_robot(
        argparse.Namespace(robot_port=raw["robot_port"], robot_id=raw["robot_id"], max_step=None),
        raw["cameras"],
    )
    features = {
        **hw_to_dataset_features(robot.action_features, "action"),
        **hw_to_dataset_features(robot.observation_features, "observation"),
    }
    root = session_dir / "dataset"
    dataset = LeRobotDataset.create(
        raw["repo_id"],
        raw["fps"],
        features=features,
        root=root,
        robot_type=robot.name,
        image_writer_threads=4 * len(raw["cameras"]),
    )
    times = np.array([f["t"] for f in raw["frames"]])
    written = 0
    print("Writing the dataset (video encoding takes a few minutes, no need to stay)...", flush=True)
    for segment in good:
        picked = np.flatnonzero((times >= segment.start) & (times < segment.end))
        if len(picked) < MIN_S * raw["fps"]:
            continue
        task = task_text(segment.note)
        for i in picked:
            frame = raw["frames"][i]
            obs = {**frame["state"], **{k: decode(v) for k, v in frame["images"].items()}}
            dataset.add_frame(
                {
                    **build_dataset_frame(dataset.features, obs, "observation"),
                    **build_dataset_frame(dataset.features, frame["action"], "action"),
                    "task": task,
                }
            )
        dataset.save_episode()
        written += 1
        if written % 25 == 0:
            print(f"  {written} of {len(good)} episodes written", flush=True)
    dataset.finalize()
    lengths = [s.end - s.start for s in good]
    write_json(
        session_dir / "session.json",
        {
            "repo_id": raw["repo_id"],
            "episodes": written,
            "median_seconds": float(np.median(lengths)),
            "segments": raw["segments"],
        },
    )
    print(f"\nDataset: {root}\n{written} episodes, median {np.median(lengths):.1f} s each.")
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--per-note", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--cameras", type=Path, default=ROOT / "scripts" / "keys_cameras_midi.json")
    parser.add_argument("--robot-port", default="/dev/tty.usbmodem5B415331661")
    parser.add_argument("--teleop-port", default="/dev/tty.usbmodem5B415332681")
    parser.add_argument("--robot-id", default="workshop_so101")
    parser.add_argument("--teleop-id", default="workshop_so101")
    parser.add_argument("--repo-id", default="PawelLach/so101_keys_flow")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "keys_flow")
    parser.add_argument("--from-raw", type=Path, help="A session folder: write its dataset again")
    args = parser.parse_args(argv)

    if args.from_raw:
        with (args.from_raw / "raw.pkl").open("rb") as f:
            raw = pickle.load(f)  # noqa: S301  our own recording
        write_dataset(raw, args.from_raw)
        return
    session_dir = args.out / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True)
    raw = record(args, session_dir)
    write_dataset(raw, session_dir)


if __name__ == "__main__":
    main()
