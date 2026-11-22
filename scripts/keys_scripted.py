#!/usr/bin/env python
"""Record the keyboard dataset with the follower arm alone. The arm plays the keys itself, from poses you
teach once by hand, so the script labels every episode ("press key N"). No leader arm, narration or press
detection is needed.

  teach   Torque off. Move the arm by hand to each pose and press Enter. Writes keys_poses.json.
  test    Plays every key once, slowly, without cameras or recording. Check that each key sounds.
  adjust  Holds a key's down pose with torque on; nudge it with the keyboard until it is spot on.
  song    Plays song.txt on an even beat, without recording.
  record  Plays the episode scripts and records the dataset: one episode per press ("press key N"), or
          with --mode phrase one episode per phrase ("play keys 1 2 5 5 5"). With --say the arm waits
          before each press while you say the key's word, so the ears camera hears the instruction.

While recording, a session folder next to the dataset keeps everything on one clock (seconds since t0 on
time.perf_counter): session.json (every frame's time, every press's frames and the notes it sounded),
midi.jsonl (the USB MIDI keyboard's note-ons and note-offs) and room.wav (the microphone behind the ears
camera). `test` with a MIDI keyboard plugged in writes keys_midi.json, the note under each sticker, and
`record` then says after each press whether the right note sounded.

A press: travel to above the key, press down, lift, hold still. Each episode starts where the previous
press ended (home for the first), like the chunks cut from the narrated session. The above pose and the
durations vary a little on every press, so the policy sees more than one path to each key.

  uv run scripts/keys_scripted.py teach
  uv run scripts/keys_scripted.py test
  uv run scripts/keys_scripted.py song
  uv run --with-requirements scripts/requirements-keys.txt scripts/keys_scripted.py record
  uv run --with-requirements scripts/requirements-keys.txt scripts/keys_scripted.py record --mode phrase
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from keys_common import (
    EPISODE_SCRIPTS,
    KEYS,
    SOLFEGE,
    SPOKEN,
    WAIT_TASK,
    load_json,
    read_song,
    task_for_key,
    write_json,
)

Pose = dict[str, float]

# Keys101 plays more than six keys. keys_layout.json in the repository root names the note under every
# sticker, in sticker order. With it, this script teaches, tests and records that many keys and labels each
# press with its note ("press E3"). Without it: the six sticker keys, labelled "press key 3".
# KEYS_LAYOUT=keys_layout_sharps.json picks another set of keys, with its own poses file (--poses).
LAYOUT = Path(os.environ.get("KEYS_LAYOUT") or Path(__file__).resolve().parent.parent / "keys_layout.json")


def read_layout(path: Path = LAYOUT) -> list[str]:
    return [str(name) for name in load_json(path)["names"]] if path.exists() else []


LAYOUT_NAMES = [] if "pytest" in sys.modules else read_layout()  # the offline tests use the six keys
if LAYOUT_NAMES:
    KEYS = tuple(range(1, len(LAYOUT_NAMES) + 1))

DEFAULT_CAMERAS = Path(__file__).with_name("keys_cameras.json")
DEFAULT_PHRASES = Path(__file__).with_name("phrases") / "song.txt"
SONGS = Path(__file__).with_name("songs")
# Joints in degrees. The gripper is 0 to 100 and stays as taught.
BODY = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
POSE_NAMES = ["rest", "home", *(f"{kind}_{key}" for key in KEYS for kind in ("down", "above"))]
HINTS = {
    "rest": "the arm folded in its rest position, where it can sit with torque off",
    "home": "above the middle of the keyboard, clear of every key",
    "down": "gripper closed, its tip pressing key {key} down far enough that it sounds",
    "above": "straight up from there, about 4 cm above key {key}",
}
PARK_SPEED = 30.0  # deg/s for the unrecorded moves to home and rest
SAY_WAIT = (1.6, 2.4)  # seconds the arm holds still before a press while you say its word (record --say)


@dataclass
class Timing:
    speed: float = 60.0  # mean joint speed of a travel, deg/s
    min_travel: float = 0.6
    press: float = 0.25  # down, and the same back up
    hold: float = 0.15
    settle: float = 0.5  # still above the key, so the policy learns to stop
    vary: float = 0.2  # travel and press durations change by up to this fraction
    jitter: float = 1.5  # degrees of random offset on the above pose (never on the press itself)
    # Long jumps across the keyboard. arc: the travel rises this many degrees (shoulder lift) per degree the
    # base turns, up to arc_max, so the tip never drags over the keys on its way. arrive: seconds the arm
    # waits above the key before pressing, so it has really stopped moving sideways. Both 0: as before.
    arc: float = 0.0
    arc_max: float = 14.0
    arrive: float = 0.0


@dataclass
class Segment:
    start: Pose
    end: Pose
    duration: float
    via: Pose | None = None  # the pose halfway through, for an arc that lifts over the keys


def min_jerk(s: float) -> float:
    """0 to 1 with zero velocity and acceleration at both ends."""
    s = min(max(s, 0.0), 1.0)
    return s**3 * (10 - 15 * s + 6 * s * s)


def pose_at(segments: list[Segment], t: float) -> Pose:
    """Commanded pose t seconds into a list of segments; the last end pose after they finish."""
    for seg in segments:
        if t < seg.duration:
            s = t / seg.duration
            a = min_jerk(s)
            pose = {k: seg.start[k] + a * (seg.end[k] - seg.start[k]) for k in seg.end}
            if seg.via is not None:
                # A bump that peaks at the via pose. It rises faster than the joints travel at the start
                # and drops slower at the end, so the arm leaves a key and lands on the next one mostly
                # vertically.
                bump = math.sin(math.pi * s) ** 2
                for k in pose:
                    pose[k] += bump * (seg.via[k] - (seg.start[k] + seg.end[k]) / 2)
            return pose
        t -= seg.duration
    return dict(segments[-1].end)


def travel_time(a: Pose, b: Pose, speed: float, min_s: float) -> float:
    return max(min_s, max(abs(b[f"{j}.pos"] - a[f"{j}.pos"]) for j in BODY) / speed)


def press_segments(
    start: Pose, poses: dict[str, Pose], key: int, timing: Timing, rng: np.random.Generator
) -> list[Segment]:
    def vary(seconds: float) -> float:
        return seconds * rng.uniform(1 - timing.vary, 1 + timing.vary)

    above = {
        k: v + (rng.uniform(-timing.jitter, timing.jitter) if k.removesuffix(".pos") in BODY else 0.0)
        for k, v in poses[f"above_{key}"].items()
    }
    down = poses[f"down_{key}"]
    travel = Segment(start, above, vary(travel_time(start, above, timing.speed, timing.min_travel)))
    if timing.arc:
        turn = abs(above["shoulder_pan.pos"] - start["shoulder_pan.pos"])
        rise = min(timing.arc_max, timing.arc * turn)
        # Up is the negative direction of the shoulder lift on this arm (above poses sit below down poses).
        travel.via = {
            k: (start[k] + above[k]) / 2 - (rise if k == "shoulder_lift.pos" else 0.0) for k in above
        }
        travel.duration += 2 * rise / timing.speed
    return [
        travel,
        *([Segment(above, above, timing.arrive)] if timing.arrive else []),
        Segment(above, down, vary(timing.press)),
        Segment(down, down, timing.hold),
        Segment(down, above, vary(timing.press)),
        Segment(above, above, timing.settle),
    ]


def press_sequence(plan: list[str], presses: int) -> list[int]:
    """The episode scripts in plan order, repeated, cut to `presses` keys. A + B + C cover every pair."""
    keys: list[int] = []
    for name in itertools.cycle(plan):
        keys += EPISODE_SCRIPTS[name]
        if len(keys) >= presses:
            return keys[:presses]
    raise AssertionError("unreachable")


def all_pairs_tour(keys: tuple[int, ...], rng: np.random.Generator) -> list[int]:
    """A walk over the keys in which every jump from one key to another, and every repeat of a key, happens
    exactly once: len(keys) squared presses after the first. It starts and ends on the first key. (An Euler
    circuit of the complete graph with loops, by Hierholzer's method, in a random order.)"""
    left = {key: [int(k) for k in rng.permutation(list(keys))] for key in keys}
    stack, circuit = [keys[0]], []
    while stack:
        here = stack[-1]
        if left[here]:
            stack.append(left[here].pop())
        else:
            circuit.append(stack.pop())
    return circuit[::-1]


def tour_sequence(keys: tuple[int, ...], presses: int, rng: np.random.Generator) -> list[int]:
    """Tours one after another, each in a new random order, cut to `presses` keys."""
    sequence: list[int] = []
    while len(sequence) < presses:
        tour = all_pairs_tour(keys, rng)
        sequence += tour if not sequence else tour[1:]  # the next tour starts where this one ended
    return sequence[:presses]


def task_for_phrase(keys: list[int]) -> str:
    return "play keys " + " ".join(str(key) for key in keys)


def plan_episodes(mode: str, plan: list[str], phrases: list[list[int]], presses: int) -> list[list[int]]:
    """The keys of every episode. press: one key each, from the episode scripts. phrase: the phrases in
    order, repeated until `presses` keys are planned (holds, the 0s, are dropped)."""
    if mode == "press":
        return [[key] for key in press_sequence(plan, presses)]
    phrases = [[key for key in phrase if key] for phrase in phrases]
    if not any(phrases):
        raise ValueError("The phrase file has no keys")
    episodes: list[list[int]] = []
    for phrase in itertools.cycle(p for p in phrases if p):
        if sum(len(e) for e in episodes) >= presses:
            return episodes
        episodes.append(phrase)
    raise AssertionError("unreachable")


class MidiLog:
    """Note-ons and note-offs of a USB MIDI keyboard on the session clock, appended to a JSONL file."""

    def __init__(self, port: str | None, t0: float, path: Path | None = None):
        import mido

        names = mido.get_input_names()
        matches = [n for n in names if port is None or port.lower() in n.lower()]
        if not matches:
            raise RuntimeError(
                f"no MIDI input{f' matching {port!r}' if port else ''} (found: {names or 'none'})"
            )
        self.name, self.t0 = matches[0], t0
        self.events: list[tuple[float, int, int]] = []  # (session time, note, velocity); velocity 0: released
        self._lock = threading.Lock()
        self._file = path.open("a") if path else None
        self._port = mido.open_input(self.name, callback=self._on_message)

    def _on_message(self, msg: Any) -> None:
        t = time.perf_counter() - self.t0
        if msg.type not in ("note_on", "note_off"):
            return
        velocity = msg.velocity if msg.type == "note_on" else 0
        with self._lock:
            self.events.append((t, msg.note, velocity))
            if self._file:
                row = {"t": round(t, 4), "note": msg.note, "velocity": velocity, "channel": msg.channel}
                self._file.write(json.dumps(row) + "\n")
                self._file.flush()

    def notes_between(self, start: float, end: float) -> list[int]:
        with self._lock:
            return [note for t, note, velocity in self.events if velocity and start <= t < end]

    def ons_since(self, start: float) -> list[tuple[float, int, int]]:
        """(session time, note, velocity) of every note-on from `start` on."""
        with self._lock:
            return [event for event in self.events if event[2] and event[0] >= start]

    def close(self) -> None:
        self._port.close()
        if self._file:
            self._file.close()


def open_midi(port: str, t0: float, path: Path | None = None) -> MidiLog | None:
    """port: "none", "auto" (the first input, or carry on without one) or part of an input's name."""
    if port == "none":
        return None
    try:
        midi = MidiLog(None if port == "auto" else port, t0, path)
    except (ImportError, RuntimeError) as e:
        if port != "auto":
            sys.exit(f"MIDI: {e}")
        print(
            f"WARNING: no MIDI ({e}). Presses are not checked or logged. With the keyboard plugged in, run "
            "under: uv run --with-requirements scripts/requirements-keys.txt"
        )
        return None
    print(f"MIDI input: {midi.name}")
    return midi


def load_key_notes(path: Path) -> dict[int, int]:
    return {int(k): int(v) for k, v in load_json(path)["notes"].items()} if path.exists() else {}


def judge_press(key: int, notes: list[int] | None, key_notes: dict[int, int]) -> tuple[bool | None, str]:
    """Did the press sound its key? None when it cannot be told (no MIDI, or no note known for the key)."""
    if notes is None:
        return None, ""
    if not notes:
        return False, "NO NOTE"
    if key not in key_notes:
        return None, f"heard {notes}"
    if notes == [key_notes[key]]:
        return True, "ok"
    if set(notes) == {key_notes[key]}:  # the right key, bouncing: it sounded more than once
        return True, f"ok (sounded {len(notes)} times)"
    return False, f"WRONG: heard {notes}, expected {key_notes[key]}"


def read_pose(robot: Any) -> Pose:
    return {f"{motor}.pos": float(v) for motor, v in robot.bus.sync_read("Present_Position").items()}


def play(
    robot: Any,
    segments: list[Segment],
    fps: int,
    on_frame: Callable[[dict[str, Any], Pose], None] | None = None,
) -> Pose:
    """Send the segments at `fps`. With on_frame, also read the robot and its cameras on every tick.

    Time on the trajectory advances one frame per tick, so a late tick slows the arm down instead of
    making it jump, and every recorded frame stays 1/fps apart on the trajectory.
    """
    from lerobot.utils.robot_utils import precise_sleep

    total = sum(s.duration for s in segments)
    start = time.perf_counter()
    for i in range(math.ceil(total * fps) + 1):  # the last tick sends the end pose itself
        target = pose_at(segments, i / fps)
        obs = robot.get_observation() if on_frame else None
        sent = robot.send_action(target)
        if on_frame and obs is not None:
            on_frame(obs, sent)
        precise_sleep(start + (i + 1) / fps - time.perf_counter())
    return dict(segments[-1].end)


def move_to(robot: Any, pose: Pose, fps: int, speed: float = PARK_SPEED) -> Pose:
    here = read_pose(robot)
    return play(robot, [Segment(here, pose, travel_time(here, pose, speed, 1.0))], fps)


def park(robot: Any, poses: dict[str, Pose], fps: int, ask: bool) -> None:
    """Move to rest, then disconnect (torque off). After a Ctrl+C, ask first: something may be in the way."""
    try:
        if ask:
            input("\nEnter: park in the rest pose and turn torque off. Ctrl+C: torque off right here. ")
        print("Parking.", flush=True)
        move_to(robot, poses["rest"], fps)
    except KeyboardInterrupt:
        print("\nTorque off where it is.")
    finally:
        robot.disconnect()


def make_robot(args: argparse.Namespace, cameras: dict[str, dict]) -> Any:
    import draccus

    from lerobot.cameras import CameraConfig
    from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401  registers the "opencv" camera type
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

    if any(cam.get("type") == "opencv_scaled" for cam in cameras.values()):
        import lerobot_camera_scaled  # noqa: F401  registers the "opencv_scaled" camera type
    if any(cam.get("type") == "spectrogram" for cam in cameras.values()):
        try:
            import lerobot_camera_spectrogram  # noqa: F401  registers the "spectrogram" camera type
        except ImportError:
            sys.exit(
                "The ears camera needs its plugin: "
                "uv run --with-requirements scripts/requirements-keys.txt scripts/keys_scripted.py ..."
            )
    config = SOFollowerRobotConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras={name: draccus.decode(CameraConfig, cam) for name, cam in cameras.items()},
        max_relative_target=args.max_step,
    )
    return SOFollower(config)


def lowered(poses: dict[str, Pose], above: float) -> dict[str, Pose]:
    """The poses with every `above` pose moved toward its `down` pose: `above` of the taught way up.
    An above pose taught high makes a long, slow press; 0.5 halves it without teaching again."""
    if above >= 1.0:
        return poses
    out = dict(poses)
    for key in KEYS:
        down, up = poses[f"down_{key}"], poses[f"above_{key}"]
        out[f"above_{key}"] = {k: down[k] + above * (up[k] - down[k]) for k in up}
    return out


def load_poses(path: Path) -> dict[str, Pose]:
    if not path.exists():
        sys.exit(f"No {path}. Teach the poses first: uv run scripts/keys_scripted.py teach")
    poses: dict[str, Pose] = load_json(path)["poses"]
    if missing := [n for n in POSE_NAMES if n not in poses]:
        sys.exit(f"{path} is missing {missing}. Teach them: uv run scripts/keys_scripted.py teach --only ...")
    return poses


def teach(args: argparse.Namespace) -> None:
    poses: dict[str, Pose] = load_json(args.poses)["poses"] if args.poses.exists() else {}
    names = [n for n in POSE_NAMES if not args.only or n in args.only or n.split("_")[-1] in args.only]
    robot = make_robot(args, {})
    robot.connect()
    try:
        robot.bus.disable_torque()
        print("Torque is off and the arm is limp. Hold it with one hand, press Enter with the other.")
        for name in names:
            kind, _, number = name.partition("_")
            if number and LAYOUT_NAMES:  # say the note too: "key 7 (C4)"
                number = f"{number} ({LAYOUT_NAMES[int(number) - 1]})"
            input(f"\n{name}: {HINTS[kind].format(key=number)}. Enter when it is there. ")
            poses[name] = read_pose(robot)
            write_json(args.poses, {"robot_id": args.robot_id, "poses": poses})
            print("  " + "  ".join(f"{k.removesuffix('.pos')} {v:.1f}" for k, v in poses[name].items()))
        for key in KEYS:
            if f"down_{key}" in poses and f"above_{key}" in poses:
                lift = travel_time(poses[f"down_{key}"], poses[f"above_{key}"], 1.0, 0.0)
                if lift < 2.0:
                    print(
                        f"Warning: above_{key} is only {lift:.1f} degrees from down_{key}. Teach it higher."
                    )
        print(f"\nSaved {args.poses}. Put the arm back in its rest pose before letting go.")
    except KeyboardInterrupt:
        print(f"\nStopped. Poses taught so far are in {args.poses}.")
    finally:
        robot.disconnect()


def test(args: argparse.Namespace) -> None:
    poses = lowered(load_poses(args.poses), args.above)
    timing = Timing(speed=args.speed, press=0.6, hold=0.5, settle=0.5, vary=0.0, jitter=0.0)
    if len(KEYS) != 6:
        timing.arc = 0.35
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    midi = open_midi(args.midi, t0)
    heard: dict[int, list[int]] = {}
    robot = make_robot(args, {})
    robot.connect()
    stopped = False
    try:
        print("Moving to home.", flush=True)
        pose = move_to(robot, poses["home"], args.fps)
        for key in args.keys:
            start = time.perf_counter() - t0
            pose = play(robot, press_segments(pose, poses, key, timing, rng), args.fps)
            if midi:
                heard[key] = midi.notes_between(start, time.perf_counter() - t0)
            print(f"key {key}" + (f": notes {heard[key] or 'NONE'}" if midi else ""), flush=True)
        move_to(robot, poses["home"], args.fps)
    except KeyboardInterrupt:
        stopped = True
    finally:
        if midi:
            midi.close()
            save_key_notes(args.key_notes, heard, midi.name)
        park(robot, poses, args.fps, ask=stopped)


def save_key_notes(path: Path, heard: dict[int, list[int]], port: str) -> None:
    """Write keys_midi.json when every key played sounded exactly one note of its own."""
    bad = [key for key, notes in heard.items() if len(notes) != 1]
    notes = {key: notes[0] for key, notes in heard.items() if len(notes) == 1}
    if bad or len(set(notes.values())) != len(notes) or not notes:
        print(f"Not saving {path}: keys {bad or sorted(notes)} did not each sound one note of their own.")
        return
    known = load_key_notes(path)
    write_json(path, {"port": port, "notes": {str(k): v for k, v in sorted({**known, **notes}.items())}})
    print(f"Saved {path}: {notes}")


# Keys for `adjust`: lower case moves 0.5 degrees, upper case 2. Key 1 is at the lowest shoulder_pan.
NUDGES = {
    "a": ("shoulder_pan", -1.0, "a/d  sideways, toward key 1 / key 6"),
    "d": ("shoulder_pan", 1.0, ""),
    "w": ("shoulder_lift", -1.0, "w/s  shoulder up / down"),
    "s": ("shoulder_lift", 1.0, ""),
    "q": ("elbow_flex", -1.0, "q/e  elbow"),
    "e": ("elbow_flex", 1.0, ""),
    "f": ("wrist_flex", -1.0, "f/r  wrist"),
    "r": ("wrist_flex", 1.0, ""),
}


def nudged(poses: dict[str, Pose], key: int, delta: Pose) -> dict[str, Pose]:
    """Poses with key's down pose moved by delta, and its above pose too, so the press stays straight."""
    out = dict(poses)
    for name in (f"down_{key}", f"above_{key}"):
        out[name] = {k: v + delta.get(k, 0.0) for k, v in poses[name].items()}
    return out


def read_char() -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def adjust(args: argparse.Namespace) -> None:
    poses = load_poses(args.poses)
    robot = make_robot(args, {})
    robot.connect()
    stopped = False
    help_text = "  ".join(text for _, _, text in NUDGES.values() if text)
    print(f"{help_text}\n  shift: 2 degree steps  p: lift and press again  Enter: save  x: skip key")
    try:
        print("Moving to home.", flush=True)
        move_to(robot, poses["home"], args.fps)
        for key in args.keys:
            print(f"\nkey {key}", flush=True)
            move_to(robot, poses[f"above_{key}"], args.fps)
            move_to(robot, poses[f"down_{key}"], args.fps)
            delta: Pose = {}
            while (c := read_char()) not in ("\n", "\r", "x"):
                if c.lower() in NUDGES:
                    joint, sign, _ = NUDGES[c.lower()]
                    step = 2.0 if c.isupper() else 0.5
                    delta[f"{joint}.pos"] = delta.get(f"{joint}.pos", 0.0) + sign * step
                    robot.send_action(nudged(poses, key, delta)[f"down_{key}"])
                elif c == "p":
                    trial = nudged(poses, key, delta)
                    move_to(robot, trial[f"above_{key}"], args.fps, speed=60.0)
                    move_to(robot, trial[f"down_{key}"], args.fps, speed=60.0)
                moved = "  ".join(f"{k.removesuffix('.pos')} {v:+.1f}" for k, v in delta.items() if v)
                print(f"\r  {moved or 'no change'}" + " " * 20, end="", flush=True)
            if c != "x" and any(delta.values()):
                poses = nudged(poses, key, delta)
                write_json(args.poses, {**load_json(args.poses), "poses": poses})
                print(f"\n  saved key {key}", flush=True)
            else:
                print(f"\n  key {key} unchanged", flush=True)
            move_to(robot, poses[f"above_{key}"], args.fps)
        move_to(robot, poses["home"], args.fps)
    except KeyboardInterrupt:
        stopped = True
    finally:
        park(robot, poses, args.fps, ask=stopped)


def song_segments(
    start: Pose,
    poses: dict[str, Pose],
    keys: list[int],
    speed: float,
    tap: float,
    lift: float,
    beat: float | None,
) -> tuple[list[Segment], float, list[float]]:
    """Moves that hit the keys on a beat, and the beat used and hit times. A 0 in `keys` (a '-' in the song)
    holds the previous key down one more beat.

    A repeated key taps from a partial lift (`tap` of the taught one). A new key is one arc from the
    previous key to the next, passing `lift` of the way up to their above poses halfway, without stopping.
    Each note starts with a wait that pads it to the beat, so the hits are evenly spaced. Without `beat`,
    the beat is the longest note's moves.
    """

    def move(a: Pose, b: Pose) -> Segment:
        return Segment(a, b, travel_time(a, b, speed, 0.1))

    def arc(a: Pose, b: Pose, prev: int, key: int) -> Segment:
        down = {k: (poses[f"down_{prev}"][k] + poses[f"down_{key}"][k]) / 2 for k in b}
        above = {k: (poses[f"above_{prev}"][k] + poses[f"above_{key}"][k]) / 2 for k in b}
        via = {k: down[k] + lift * (above[k] - down[k]) for k in b}
        # The bump goes up and back down, so it counts twice towards the distance.
        bump = max(abs(via[f"{j}.pos"] - (a[f"{j}.pos"] + b[f"{j}.pos"]) / 2) for j in BODY)
        return Segment(a, b, max(0.1, travel_time(a, b, speed, 0.0) + 2 * bump / speed), via)

    def hover(key: int) -> Pose:
        down, above = poses[f"down_{key}"], poses[f"above_{key}"]
        return {k: down[k] + tap * (above[k] - down[k]) for k in down}

    tune: list[list[int]] = []  # [key, beats]
    for key in keys:
        if key:
            tune.append([key, 1])
        elif tune:
            tune[-1][1] += 1
    hold = 0.04
    release = max(move(poses[f"down_{k}"], hover(k)).duration for k, _ in tune)
    notes: list[list[Segment]] = []
    pose, prev = start, None
    for key, _ in tune:
        down = poses[f"down_{key}"]
        if prev is None:
            approach = [move(pose, poses[f"above_{key}"]), move(poses[f"above_{key}"], down)]
        elif key == prev:
            approach = [move(pose, down)]
        else:
            approach = [arc(pose, down, prev, key)]
        notes.append(approach)
        pose, prev = hover(key), key

    need = max(sum(s.duration for s in approach) for approach in notes) + hold + release
    beat = need if beat is None else beat
    segments: list[Segment] = []
    hits: list[float] = []
    t = 0.0
    for approach, (key, beats) in zip(notes, tune, strict=True):
        down = poses[f"down_{key}"]
        pad = max(beat - sum(s.duration for s in approach) - hold - release, 0.0)
        held = hold + (beats - 1) * beat
        segments += [
            Segment(approach[0].start, approach[0].start, pad),
            *approach,
            Segment(down, down, held),
            Segment(down, hover(key), release),
        ]
        hits.append(t + pad + sum(s.duration for s in approach))
        t = hits[-1] + held + release
    return segments, beat, hits


def song_names() -> list[str]:
    return sorted(path.stem for path in SONGS.glob("*.txt"))


def song(args: argparse.Namespace) -> None:
    poses = load_poses(args.poses)
    path = SONGS / f"{args.name}.txt" if args.name else args.song
    if not path.exists():
        sys.exit(f"No {path}. Songs: {', '.join(song_names())}")
    repeat = args.repeat or (1 if args.name else 2)
    keys = [key for phrase in read_song(path) for key in phrase] * repeat
    plan = (poses["home"], poses, keys, args.speed, args.tap, args.lift)
    segments, beat, _ = song_segments(*plan, args.beat)
    shortest = song_segments(*plan, None)[1]
    if beat < shortest:
        print(f"Warning: a beat of {beat:.2f} s is too short for some jumps (they need {shortest:.2f} s).")
    robot = make_robot(args, {})
    robot.connect()
    stopped = False
    try:
        print("Moving to home.", flush=True)
        move_to(robot, poses["home"], args.fps)
        notes = sum(1 for key in keys if key)
        print(f"Playing {path.stem}: {notes} notes, one beat every {beat:.2f} s ({60 / beat:.0f} BPM).")
        play(robot, segments, args.fps)
        move_to(robot, poses[f"above_{next(k for k in reversed(keys) if k)}"], args.fps)
    except KeyboardInterrupt:
        stopped = True
    finally:
        park(robot, poses, args.fps, ask=stopped)


def record(args: argparse.Namespace) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

    poses = lowered(load_poses(args.poses), args.above)
    args.names = args.names or LAYOUT_NAMES or None
    if args.names and len(args.names) != len(KEYS):
        sys.exit(f"--names takes {len(KEYS)} note names, one per sticker, e.g. C3 D3 E3 F3 G3 A3")

    def key_task(key: int) -> str:
        """The episode's label: "press key 3", or with --names the note itself, "press E3". Note names
        keep two recordings on different keys apart when they are merged into one training set."""
        return f"press {args.names[key - 1]}" if args.names else task_for_key(key)

    if unknown := [p for p in args.plan.split(",") if p not in EPISODE_SCRIPTS]:
        sys.exit(f"Unknown episode scripts {unknown}; choose from {list(EPISODE_SCRIPTS)}")
    phrases = read_song(args.phrases) if args.mode == "phrase" else []
    if args.mode == "press" and len(KEYS) != 6:
        # The episode scripts A, B, C, S are written for six keys. For any other number: tours that make
        # every jump between two keys equally often.
        plan_rng = np.random.default_rng(args.seed)
        episodes = [[key] for key in tour_sequence(KEYS, args.presses, plan_rng)]
    else:
        episodes = plan_episodes(args.mode, args.plan.split(","), phrases, args.presses)
    n_presses = sum(len(keys) for keys in episodes)
    timing = Timing(speed=args.speed, jitter=args.jitter)
    if args.fast:  # snappier presses and shorter stops, for a policy that should react quickly
        timing = Timing(
            speed=args.speed, jitter=args.jitter, min_travel=0.4, press=0.18, hold=0.12, settle=0.3
        )
    if len(KEYS) != 6 or args.arc is not None:  # a wide keyboard: arcs on long jumps, a stop before pressing
        timing.arc = 0.35 if args.arc is None else args.arc
        timing.arrive = args.arrive
    rng = np.random.default_rng(args.seed)
    cameras = load_json(args.cameras)
    robot = make_robot(args, cameras)
    features = {
        **hw_to_dataset_features(robot.action_features, "action"),
        **hw_to_dataset_features(robot.observation_features, "observation"),
    }
    n_cameras = sum(isinstance(ft, tuple) for ft in robot.observation_features.values())
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = args.out / stamp
    session_dir = args.out / f"{stamp}_session"
    session_dir.mkdir(parents=True)

    # One clock for the frames, the MIDI notes and the microphones: seconds since t0.
    t0 = time.perf_counter()
    midi = open_midi(args.midi, t0, session_dir / "midi.jsonl")
    key_notes = load_key_notes(args.key_notes)
    if midi and not key_notes:
        print(f"No {args.key_notes}: notes are logged but not checked. `test` writes it.")
    devices = {}
    room = args.room_device
    if room == "auto":
        room = next((c.get("device") for c in cameras.values() if c.get("type") == "spectrogram"), "none")
    if room not in (None, "none"):
        devices["room"] = room
    if args.voice_device != "none":
        devices["voice"] = args.voice_device
    audio = None
    if devices:
        from record_session import AudioProcess

        audio = AudioProcess(session_dir, devices, t0)
        print(f"Microphones: {audio.start()}")

    session: dict[str, Any] = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "fps": args.fps,
        "repo_id": args.repo_id,
        "dataset_root": str(root),
        "cameras": cameras,
        "midi": {"port": midi.name, "file": "midi.jsonl", "key_notes": key_notes} if midi else None,
        "audio": {},
        "episodes": [],
    }
    dataset = LeRobotDataset.create(
        args.repo_id,
        args.fps,
        features=features,
        root=root,
        robot_type=robot.name,
        image_writer_threads=4 * n_cameras,
    )
    print(
        f"{len(episodes)} episodes, {n_presses} presses, about {n_presses * 3.5 / 60:.0f} min. "
        "Stay quiet: the ears camera listens."
    )
    robot.connect()
    saved, done, good, bad, stopped = 0, 0, 0, 0, False
    try:
        print("Moving to home.", flush=True)
        pose = move_to(robot, poses["home"], args.fps)
        for i, keys in enumerate(episodes):
            task = args.task or (key_task(keys[0]) if args.mode == "press" else task_for_phrase(keys))
            stamps: list[float] = []
            presses: list[dict[str, Any]] = []

            frame_task = [task]  # the task of the frames being recorded right now

            def add(
                obs: dict[str, Any],
                sent: Pose,
                frame_task: list[str] = frame_task,
                stamps: list[float] = stamps,
            ) -> None:
                stamps.append(time.perf_counter() - t0)
                dataset.add_frame(
                    {
                        **build_dataset_frame(dataset.features, obs, "observation"),
                        **build_dataset_frame(dataset.features, sent, "action"),
                        "task": frame_task[0],
                    }
                )

            for key in keys:
                segments = press_segments(pose, poses, key, timing, rng)
                first, start = len(stamps), time.perf_counter() - t0
                if args.say:
                    # The arm holds still while you say the word, so the ears hear it before any motion:
                    # the word is the only thing that tells the policy which key comes next. These frames
                    # are labelled "wait", the press itself "press key N": a policy steered by text
                    # learns to hold still until it is told a key.
                    words = SOLFEGE if args.words == "solfege" else SPOKEN
                    print(f"\n        >>>  SAY:  {words[key].upper()}  <<<", flush=True)
                    frame_task[0] = args.task or WAIT_TASK
                    play(robot, [Segment(pose, pose, float(rng.uniform(*SAY_WAIT)))], args.fps, add)
                    frame_task[0] = args.task or key_task(key)
                go = len(stamps)
                pose = play(robot, segments, args.fps, add)
                end = time.perf_counter() - t0
                notes = midi.notes_between(start, end) if midi else None
                ok, verdict = judge_press(key, notes, key_notes)
                good, bad = good + (ok is True), bad + (ok is False)
                done += 1
                presses.append(
                    {
                        "key": key,
                        "frame_start": first,
                        "frame_end": len(stamps),  # exclusive
                        "frame_go": go,  # the arm starts moving
                        "frame_down": go
                        + round(
                            sum(seg.duration for seg in segments[: 3 if timing.arrive else 2]) * args.fps
                        ),
                        "start": round(start, 4),
                        "end": round(end, 4),
                        "notes": notes,
                        "ok": ok,
                    }
                )
                print(f"[{done}/{n_presses}] episode {i}: {key_task(key)}  {verdict}", flush=True)
            dataset.save_episode()
            saved += 1
            session["episodes"].append(
                {
                    "episode_index": i,
                    "task": task,
                    "keys": keys,
                    "frame_times": [round(t, 4) for t in stamps],
                    "presses": presses,
                }
            )
            write_json(session_dir / "session.json", session)
    except KeyboardInterrupt:
        stopped = True
        print("\nStopped. The unfinished episode is dropped.")
    finally:
        # Audio and the session file first: disconnecting the ears camera can hang inside CoreAudio.
        if audio:
            session["audio"] = audio.stop()
        if midi:
            midi.close()
        write_json(session_dir / "session.json", session)
        dataset.clear_episode_buffer()
        dataset.finalize()
        checked = f" Presses that sounded their key: {good}/{good + bad}." if good + bad else ""
        print(
            f"\n{saved} episodes saved in {root} (repo id {args.repo_id}), session in {session_dir}.{checked} "
            "If the program hangs while disconnecting the ears camera (a macOS audio bug), close this "
            "terminal tab: nothing is lost.\n",
            flush=True,
        )
        park(robot, poses, args.fps, ask=stopped)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--robot-port", default="/dev/tty.usbmodem5B415331661")
    common.add_argument("--robot-id", default="workshop_so101")
    common.add_argument("--poses", type=Path, default=Path("keys_poses.json"))
    common.add_argument("--fps", type=int, help="Control rate in Hz (default: 50 for song, else 30)")
    common.add_argument(
        "--max-step", type=float, default=10.0, help="Largest move per frame from where the arm is (deg)"
    )
    common.add_argument(
        "--midi", default="auto", help="MIDI input: part of its name, auto (the first one) or none"
    )
    common.add_argument(
        "--above",
        type=float,
        default=1.0,
        help="test and record: hover this fraction of the taught way above each key (0.5: half as high)",
    )
    common.add_argument(
        "--key-notes", type=Path, default=Path("keys_midi.json"), help="MIDI note of each key, from `test`"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("teach", parents=[common], help="Teach the poses by hand")
    p.add_argument("--only", nargs="+", default=[], help="Pose names or key numbers to re-teach, e.g. 3 home")
    p.set_defaults(run=teach)

    p = sub.add_parser("test", parents=[common], help="Play each key once, slowly, no recording")
    p.add_argument("--keys", type=int, nargs="+", default=list(KEYS), choices=KEYS)
    p.add_argument("--speed", type=float, default=25.0, help="Mean joint speed while travelling (deg/s)")
    p.set_defaults(run=test)

    p = sub.add_parser("adjust", parents=[common], help="Nudge keys' poses with torque on")
    p.add_argument("keys", type=int, nargs="+", choices=KEYS)
    p.set_defaults(run=adjust)

    p = sub.add_parser("song", parents=[common], help="Play a song, no recording")
    p.add_argument("name", nargs="?", help=f"A song in scripts/songs: {', '.join(song_names())}")
    p.add_argument("--song", type=Path, default=Path("song.txt"), help="Song file, when no name is given")
    p.add_argument("--repeat", type=int, help="Times through the song (default: 2 for song.txt, else 1)")
    p.add_argument("--beat", type=float, help="Seconds between notes (default: as fast as the longest jump)")
    p.add_argument("--speed", type=float, default=150.0, help="Mean joint speed of each move (deg/s)")
    p.add_argument(
        "--tap", type=float, default=0.3, help="Lift between repeated notes, fraction of the taught"
    )
    p.add_argument("--lift", type=float, default=0.7, help="Arc height between keys, fraction of the taught")
    p.set_defaults(run=song)

    p = sub.add_parser("record", parents=[common], help="Play the scripts and record the dataset")
    p.add_argument("--presses", type=int, default=500)
    p.add_argument(
        "--mode", choices=("press", "phrase"), default="press", help="One episode per press, or per phrase"
    )
    p.add_argument(
        "--phrases", type=Path, default=DEFAULT_PHRASES, help="Phrase mode: a song file, one phrase per line"
    )
    p.add_argument(
        "--room-device", default="auto", help="Microphone saved as room.wav: auto (the ears camera's) or none"
    )
    p.add_argument("--voice-device", default="none", help="Narration microphone saved as voice.wav, or none")
    p.add_argument(
        "--say",
        action="store_true",
        help="Before each press, show its word and hold still about 2 s while you say it to the ears",
    )
    p.add_argument("--words", choices=("numbers", "solfege"), default="numbers", help="What --say shows")
    p.add_argument("--task", help="One task text for every frame, e.g. 'play the key you hear'")
    p.add_argument(
        "--names",
        nargs="+",
        help="The note under each sticker, e.g. C3 D3 E3 F3 G3 A3: episodes are labelled 'press E3'",
    )
    p.add_argument("--plan", default="A,B,C,S", help=f"Episode scripts, cycled: {', '.join(EPISODE_SCRIPTS)}")
    p.add_argument(
        "--speed", type=float, default=Timing.speed, help="Mean joint speed while travelling (deg/s)"
    )
    p.add_argument(
        "--jitter", type=float, default=Timing.jitter, help="Random offset of the above pose (deg)"
    )
    p.add_argument("--fast", action="store_true", help="Shorter presses and stops: a more reactive policy")
    p.add_argument(
        "--arc",
        type=float,
        help="Rise of a jump, degrees per degree the base turns (default 0.35 beyond 6 keys)",
    )
    p.add_argument(
        "--arrive", type=float, default=0.2, help="Seconds above the key before pressing, with --arc"
    )
    p.add_argument("--seed", type=int, help="Random seed for the variations")
    p.add_argument("--cameras", type=Path, default=DEFAULT_CAMERAS)
    p.add_argument("--repo-id", default="PawelLach/so101_keys_scripted")
    p.add_argument("--out", type=Path, default=Path("outputs/keys_scripted"))
    p.set_defaults(run=record)

    args = parser.parse_args()
    # Not set_defaults(fps=50) on the song parser: the parsers share --fps, so it would change them all.
    if args.fps is None:
        args.fps = 50 if args.command == "song" else 30
    args.run(args)


if __name__ == "__main__":
    main()
