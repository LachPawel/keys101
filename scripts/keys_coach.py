#!/usr/bin/env python
"""Keys101, the piano coach: the SO-101 teaches you a song on the MIDI keyboard, then plays it with you.

    R="uv run --with-requirements scripts/requirements-keys.txt"
    $R scripts/keys_coach.py --arm sim                 # no robot: the screen and the speaker stand in for it
    $R scripts/keys_coach.py --arm scripted            # the arm moves from taught poses (fast, no policy)
    $R scripts/keys_coach.py --arm policy              # the learned policy presses, taught poses point
    $R scripts/keys_coach.py teach --from C3 --to C5   # teach the arm where the keys are (keys_board.json)

Then open http://localhost:8780. Pick a song and a mode on the screen:

  Learn            The arm hovers over the next key and waits for you. A wrong key: it taps the right one and
                   John sings its name. Each phrase twice without a mistake, then the next. The second time the
                   arm only helps if you hesitate.
  Watch and repeat The arm plays a phrase, you play it back. MIDI checks each note.
  Play together    You play your part, the arm plays its part with you: one note per bar, the moment you
                   play that bar's first note. You lead, it follows.
  Jam              Call and response. Play a short phrase and stop: the arm answers it on its keys, echoed or
                   turned upside down. Then it is your turn again.

You lead everywhere: there is no metronome. The coach only listens to the MIDI keyboard. The arm's own notes
are told apart from yours because the coach knows which key the arm is about to press.

The keyboard (a USB MIDI keyboard without its own sound) needs the software piano running in another tab:
    $R scripts/keys_synth.py

This runs on its own, next to the other scripts: it changes nothing in the voice teacher or the stage display.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import queue
import re
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
LESSONS = Path(__file__).with_name("lessons")
BOARD = ROOT / "keys_board.json"
POSES = ROOT / "keys_poses.json"  # the sticker keys of keys_scripted.py, used when there is no board yet
LAYOUT = ROOT / "keys_layout.json"  # the note under every sticker, shared with keys_scripted.py
WEB = ROOT / "web"

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
SYLLABLE = {0: "do", 2: "re", 4: "mi", 5: "fa", 7: "sol", 9: "la", 11: "si"}  # what John sings on each key
# The stickers 1 to 6 of the policy's training: C4 D4 E4 F4 G4 A4. The learned policy can press only these.
# The sticker keys, which are also the keys the policy learns. keys_layout.json names them in sticker order:
# for Keys101 the bass C3 to A3 (1 to 6, the left hand it plays with you) and the melody C4 to A4 (7 to 12,
# which it shows you). Without the file: the six bass keys. Filled in below, once notes can be read.
STICKERS: dict[int, int] = {1: 48, 2: 50, 3: 52, 4: 53, 5: 55, 6: 57}
LOWEST = 48  # C3: the auto arm part stays at or above it
# The older `flow` policy (--prompts flow) learned the six keys from middle C, named do re mi fa sol la.
POLICY_WORDS = {60: "do", 62: "re", 64: "mi", 65: "fa", 67: "sol", 69: "la"}

Pose = dict[str, float]


# ------------------------------------------------------------------------------------------- notes


def note_midi(name: str) -> int:
    """'C4' -> 60, 'F#3' -> 54, 'Bb3' -> 58."""
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)", name.strip())
    if not m:
        raise ValueError(f"not a note: {name!r} (expected a letter, maybe # or b, and an octave, like F#3)")
    letter, accidental, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    pitch = letter + accidental
    pitch = FLATS.get(pitch, pitch)
    if pitch not in NAMES:  # Cb, Fb, E#, B#
        raise ValueError(f"not a note: {name!r}")
    return NAMES.index(pitch) + 12 * (octave + 1)


def read_stickers(path: Path = LAYOUT) -> dict[int, int]:
    """Sticker number -> MIDI note, from keys_layout.json; the six bass keys without it."""
    if not path.exists():
        return dict(STICKERS)
    return {i: note_midi(name) for i, name in enumerate(json.loads(path.read_text())["names"], start=1)}


def midi_name(midi: int) -> str:
    return f"{NAMES[midi % 12]}{midi // 12 - 1}"


def is_black(midi: int) -> bool:
    return NAMES[midi % 12].endswith("#")


def syllable(midi: int) -> str:
    return SYLLABLE.get(midi % 12, "la")


if "pytest" not in sys.modules:  # the offline tests use the six bass keys
    STICKERS = read_stickers()


# ------------------------------------------------------------------------------------------- songs


@dataclass
class Bar:
    you: list[int]
    arm: int | None = None


@dataclass
class Phrase:
    bars: list[Bar]

    @property
    def notes(self) -> list[int]:
        return [n for bar in self.bars for n in bar.you]

    def bar_starts(self) -> dict[int, int | None]:
        """Index of each bar's first note in `notes` -> the arm's note for that bar."""
        out: dict[int, int | None] = {}
        i = 0
        for bar in self.bars:
            out[i] = bar.arm
            i += len(bar.you)
        return out


@dataclass
class Song:
    name: str
    title: str
    about: str = ""
    level: int = 1
    phrases: list[Phrase] = field(default_factory=list)

    def info(self) -> dict[str, Any]:
        notes = [n for p in self.phrases for n in p.notes]
        return {
            "name": self.name,
            "title": self.title,
            "about": self.about,
            "level": self.level,
            "phrases": len(self.phrases),
            "notes": len(notes),
            "low": min(notes) if notes else None,
            "high": max(notes) if notes else None,
        }


def parse_lesson(text: str, name: str = "song") -> Song:
    song = Song(name=name, title=name)
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = re.split(r"(?:^|\s)#", raw, maxsplit=1)[0].strip()  # "D#4" is a note, " # x" a comment
        if not line:
            continue
        key, sep, value = line.partition(":")
        key = key.strip().lower()
        if not sep:
            raise ValueError(f"{name}:{line_no}: expected 'key: value', got {raw.strip()!r}")
        value = value.strip()
        try:
            if key == "title":
                song.title = value
            elif key == "about":
                song.about = value
            elif key == "level":
                song.level = int(value)
            elif key == "you":
                bars = [[note_midi(tok) for tok in part.split()] for part in value.split("|")]
                if any(not bar for bar in bars):
                    raise ValueError("a bar with no notes")
                song.phrases.append(Phrase([Bar(you=bar) for bar in bars]))
            elif key == "arm":
                if not song.phrases:
                    raise ValueError("an arm: line must follow a you: line")
                phrase = song.phrases[-1]
                parts = [part.split() for part in value.split("|")]
                if len(parts) != len(phrase.bars):
                    raise ValueError(f"{len(parts)} bars for the arm, {len(phrase.bars)} for you")
                for bar, part in zip(phrase.bars, parts, strict=True):
                    if len(part) != 1:
                        raise ValueError("one note (or -) per bar for the arm")
                    bar.arm = None if part[0] == "-" else note_midi(part[0])
            else:
                raise ValueError(f"unknown key {key!r}")
        except ValueError as e:
            raise ValueError(f"{name}:{line_no}: {e}") from None
    if not song.phrases:
        raise ValueError(f"{name}: no 'you:' lines")
    # Without arm lines the arm plays the first note of each bar an octave lower, while that stays at C3 or
    # above (the lowest key of a 25-key keyboard).
    if all(bar.arm is None for p in song.phrases for bar in p.bars):
        for p in song.phrases:
            for bar in p.bars:
                bar.arm = bar.you[0] - 12 if bar.you[0] - 12 >= LOWEST else None
    return song


def load_songs(folder: Path = LESSONS) -> dict[str, Song]:
    songs = {path.stem: parse_lesson(path.read_text(), path.stem) for path in sorted(folder.glob("*.txt"))}
    return dict(sorted(songs.items(), key=lambda kv: (kv[1].level, kv[1].title)))


# ------------------------------------------------------------------------------------------- board


@dataclass
class Board:
    """Where the arm finds each key: an `above` pose (pointing, about 3 cm over the key) and a `down` pose
    (pressing it). `reach` is every key the arm can press; a simulated arm reaches everything."""

    keys: dict[int, dict[str, Pose]] = field(default_factory=dict)
    home: Pose | None = None
    rest: Pose | None = None
    everything: bool = False  # a simulated arm: every key

    def reaches(self, midi: int) -> bool:
        return self.everything or (midi in self.keys and {"above", "down"} <= self.keys[midi].keys())

    @property
    def reach(self) -> list[int]:
        return sorted(m for m in self.keys if self.reaches(m))

    def snap(self, midi: int) -> int | None:
        """The key the arm should play for `midi`: itself, else the same note in the nearest octave it
        reaches, else None."""
        if self.reaches(midi):
            return midi
        same = [m for m in self.reach if m % 12 == midi % 12]
        return min(same, key=lambda m: (abs(m - midi), m)) if same else None

    def nearest(self, midi: int) -> int | None:
        """snap(), else the closest key at all (for an answer in a jam, where any nearby key will do)."""
        hit = self.snap(midi)
        if hit is not None or not self.reach:
            return hit
        return min(self.reach, key=lambda m: (abs(m - midi), m))


def load_board(
    board_path: Path = BOARD, poses_path: Path = POSES, stickers: dict[int, int] | None = None
) -> Board:
    """keys_board.json from `teach`, else the sticker keys taught for keys_scripted.py (keys_poses.json)."""
    if board_path.exists():
        data = json.loads(board_path.read_text())
        keys = {note_midi(name): poses for name, poses in data.get("keys", {}).items()}
        return Board(keys=keys, home=data.get("home"), rest=data.get("rest"))
    board = Board()
    if poses_path.exists():
        poses = json.loads(poses_path.read_text())["poses"]
        for key, midi in (stickers or STICKERS).items():
            if f"above_{key}" in poses and f"down_{key}" in poses:
                board.keys[midi] = {"above": poses[f"above_{key}"], "down": poses[f"down_{key}"]}
        board.home, board.rest = poses.get("home"), poses.get("rest")
    return board


def save_board(board: Board, path: Path = BOARD, robot_id: str = "") -> None:
    data = {
        "robot_id": robot_id,
        "home": board.home,
        "rest": board.rest,
        "keys": {midi_name(m): board.keys[m] for m in sorted(board.keys)},
    }
    path.write_text(json.dumps(data, indent=1) + "\n")


def white_keys(lo: int, hi: int) -> list[int]:
    return [m for m in range(lo, hi + 1) if not is_black(m)]


def fill_white(board: Board, lo: int, hi: int) -> list[int]:
    """Poses for the white keys between `lo` and `hi` that were not taught, from the ones that were: each
    joint as a smooth curve (a parabola through three taught keys, a line through two) over the white keys'
    positions. Keys are evenly spaced, so this lands close; re-teach any that miss with `teach --only`.
    Black keys sit further back and higher, so they are never filled in. Returns the keys it filled."""
    whites = white_keys(lo, hi)
    taught = [m for m in whites if board.reaches(m)]
    if len(taught) < 2:
        return []
    filled = []
    for m in whites:
        if board.reaches(m):
            continue
        x = whites.index(m)
        # The two or three taught keys closest to this one.
        near = sorted(taught, key=lambda t: abs(whites.index(t) - x))[: 3 if len(taught) >= 3 else 2]
        xs = [whites.index(t) for t in near]
        poses: dict[str, Pose] = {}
        for kind in ("above", "down"):
            pose: Pose = {}
            for joint in board.keys[near[0]][kind]:
                ys = [board.keys[t][kind][joint] for t in near]
                pose[joint] = round(_interpolate(xs, ys, x), 3)
            poses[kind] = pose
        board.keys[m] = poses
        filled.append(m)
    return filled


def _interpolate(xs: list[int], ys: list[float], x: float) -> float:
    """Lagrange polynomial through the points (degree 1 or 2)."""
    total = 0.0
    for i, (xi, yi) in enumerate(zip(xs, ys, strict=True)):
        term = yi
        for j, xj in enumerate(xs):
            if j != i:
                term *= (x - xj) / (xi - xj)
        total += term
    return total


# ------------------------------------------------------------------------------------------- state


class State:
    """Everything the page shows, sent to every open tab as it changes."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.version = 0
        self.press_id = 0
        self.john_id = 0
        self.data: dict[str, Any] = {
            "mode": "",
            "song": None,
            "songs": [],
            "phrase": 0,
            "phrases": 0,
            "notes": [],  # the current phrase: {"midi", "name", "status": todo | now | ok | miss}
            "target": None,  # the key you should play now
            "arm": {"at": None, "pressing": None, "how": "", "name": ""},
            "big": "Pick a song",
            "small": "Then choose how to practise it.",
            "tone": "",
            "presses": [],  # the newest key presses: {"id", "midi", "who": you | arm | ghost, "ok"}
            "keys_down": [],
            "range": [48, 72],
            "reach": [],
            "score": {"right": 0, "wrong": 0, "streak": 0},
            "john": None,  # {"id", "midi", "name", "seconds"}: sing this once
            "john_text": "",
            "style": "echo",
            "sound": {"you": False, "arm": False},
            "midi": False,
            "course": [],  # the lessons: {"id", "title", "goal", "status": done | open | locked}
            "lesson": None,  # the lesson running now
            "hearts": None,  # hearts left in it
            "xp": 0,
            "days": 0,  # days in a row with a finished lesson
        }

    def set(self, **changes: Any) -> None:
        with self.cond:
            self.data.update(changes)
            self.version += 1
            self.cond.notify_all()

    def get(self, key: str) -> Any:
        with self.cond:
            return self.data[key]

    def snapshot(self) -> str:
        with self.cond:
            return json.dumps(self.data)

    def press(self, midi: int, who: str, ok: bool | None = None) -> int:
        """A key went down (or the page should play one, who="ghost"). Returns its id for judge()."""
        with self.cond:
            self.press_id += 1
            presses = [*self.data["presses"], {"id": self.press_id, "midi": midi, "who": who, "ok": ok}][-24:]
            self.data["presses"] = presses
            self.version += 1
            self.cond.notify_all()
            return self.press_id

    def judge(self, press_id: int, ok: bool) -> None:
        """Right or wrong, for a press already shown: the page flashes the key green or red."""
        with self.cond:
            self.data["presses"] = [
                {**p, "ok": ok} if p["id"] == press_id else p for p in self.data["presses"]
            ]
            s = dict(self.data["score"])
            if ok:
                s["right"] += 1
                s["streak"] += 1
            else:
                s["wrong"] += 1
                s["streak"] = 0
            self.data["score"] = s
            self.version += 1
            self.cond.notify_all()

    def john(self, midi: int, seconds: float = 0.9) -> None:
        with self.cond:
            self.john_id += 1
            self.data["john"] = {"id": self.john_id, "midi": midi, "name": syllable(midi), "seconds": seconds}
            self.version += 1
            self.cond.notify_all()


# ------------------------------------------------------------------------------------------- ears


@dataclass
class Press:
    t: float
    midi: int
    who: str  # "you" or "arm"
    id: int = 0  # State.press's id, for State.judge


class Stopped(Exception):
    """The mode was stopped from the page."""


class Ears:
    """Every key that goes down, from the MIDI keyboard or a click on the page, told apart as yours or the
    arm's. Before a press the arm says which key it is about to play (`expect`); the first note-on of that key
    within the window is the arm's. Anything else is yours."""

    def __init__(self, state: State) -> None:
        self.state = state
        self.queue: queue.Queue[Press] = queue.Queue()
        self.lock = threading.Lock()
        self.expected: list[tuple[int, float]] = []  # (key, until when) the arm is about to press
        self.down: set[int] = set()
        self.port: Any = None
        self.stop = threading.Event()

    def open_midi(self, port: str) -> str | None:
        """port: "auto" (the first input), "none", or part of its name. Returns the input's name."""
        if port == "none":
            return None
        try:
            import mido
        except ImportError:
            print("No mido: run under uv run --with-requirements scripts/requirements-keys.txt", flush=True)
            return None
        names = [n for n in mido.get_input_names() if port == "auto" or port.lower() in n.lower()]
        names = [n for n in names if "blackhole" not in n.lower() and "through" not in n.lower()] or names
        if not names:
            print("No MIDI keyboard found: click the keys on the page instead.", flush=True)
            return None
        self.port = mido.open_input(names[0], callback=self._on_message)
        self.state.set(midi=True)
        return names[0]

    def _on_message(self, msg: Any) -> None:
        if msg.type == "note_on" and msg.velocity > 0:
            self.key_down(msg.note)
        elif msg.type in ("note_off", "note_on"):
            self.key_up(msg.note)

    def expect(self, midi: int, seconds: float) -> None:
        with self.lock:
            now = time.monotonic()
            self.expected = [(m, until) for m, until in self.expected if until > now]
            self.expected.append((midi, now + seconds))

    def unexpect(self, midi: int) -> None:
        with self.lock:
            self.expected = [(m, until) for m, until in self.expected if m != midi]

    def key_down(self, midi: int, who: str | None = None) -> None:
        now = time.monotonic()
        with self.lock:
            if who is None:
                who = "you"
                for i, (m, until) in enumerate(self.expected):
                    if m == midi and until > now:
                        who = "arm"
                        del self.expected[i]
                        break
            self.down.add(midi)
            down = sorted(self.down)
        self.state.set(keys_down=down)
        self.queue.put(Press(now, midi, who, self.state.press(midi, who)))

    def key_up(self, midi: int) -> None:
        with self.lock:
            self.down.discard(midi)
            down = sorted(self.down)
        self.state.set(keys_down=down)

    def flush(self) -> None:
        with contextlib.suppress(queue.Empty):
            while True:
                self.queue.get_nowait()

    def next(self, timeout: float | None, who: str = "you") -> Press | None:
        """The next press by `who`, or None after `timeout` seconds. Raises Stopped when stopped."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self.stop.is_set():
                raise Stopped
            left = 0.1 if deadline is None else min(0.1, deadline - time.monotonic())
            if left <= 0:
                return None
            try:
                press = self.queue.get(timeout=left)
            except queue.Empty:
                continue
            if press.who == who:
                return press

    def close(self) -> None:
        if self.port is not None:
            self.port.close()


# ------------------------------------------------------------------------------------------- arms


class Arm(Protocol):
    name: str
    board: Board

    def point(self, midi: int) -> None:
        """Hover over the key and wait there."""

    def press(self, midi: int, learned: bool = False) -> None:
        """Press the key and come back up over it. learned: let the policy do it, where there is one."""

    def away(self) -> None:
        """Out of the way of your hands: back to home."""

    def close(self) -> None: ...


class SimArm:
    """No robot. The page draws where the arm would be, and plays the arm's notes itself."""

    name = "no robot (screen)"

    def __init__(self, ears: Ears, state: State, board: Board | None = None, speed: float = 1.0) -> None:
        self.ears, self.state, self.speed = ears, state, speed
        self.board = board if board is not None and board.reach else Board(everything=True)
        self.at: int | None = None

    def _wait(self, seconds: float) -> None:
        if self.ears.stop.wait(seconds / self.speed):
            raise Stopped

    def point(self, midi: int) -> None:
        if self.at == midi:
            return
        self.state.set(arm={**self.state.get("arm"), "at": midi, "pressing": None})
        self._wait(0.35 if self.at is None else min(0.2 + 0.03 * abs(midi - self.at), 0.6))
        self.at = midi

    def press(self, midi: int, learned: bool = False) -> None:
        self.point(midi)
        self.state.set(arm={**self.state.get("arm"), "at": midi, "pressing": midi, "how": "sim"})
        self.ears.key_down(midi, who="arm")
        self._wait(0.25)
        self.ears.key_up(midi)
        self.state.set(arm={**self.state.get("arm"), "pressing": None})

    def away(self) -> None:
        self.at = None
        self.state.set(arm={**self.state.get("arm"), "at": None, "pressing": None})

    def close(self) -> None:
        pass


@dataclass
class Speeds:
    travel: float = 120.0  # mean joint speed between keys, deg/s
    min_travel: float = 0.25
    down: float = 0.12  # seconds from above the key to pressed, and the same back
    hold: float = 0.08
    fps: int = 50


class ScriptedArm:
    """The real arm, moving straight from taught poses (keys_board.json): fast and exact, no policy."""

    name = "scripted (taught poses)"

    def __init__(
        self, board: Board, ears: Ears, state: State, port: str, robot_id: str, speeds: Speeds
    ) -> None:
        import keys_scripted

        if not board.reach:
            sys.exit("The arm has no taught keys: run `keys_coach.py teach` first.")
        self.ks = keys_scripted
        self.board, self.ears, self.state, self.speeds = board, ears, state, speeds
        args = argparse.Namespace(robot_port=port, robot_id=robot_id, max_step=15.0)
        self.robot = keys_scripted.make_robot(args, {})
        self.robot.connect()
        self.pose = keys_scripted.read_pose(self.robot)
        self.at: int | None = None
        if board.home:
            self._move(board.home, slow=True)

    def _move(self, pose: Pose, seconds: float | None = None, slow: bool = False) -> None:
        if self.ears.stop.is_set():
            raise Stopped
        ks, sp = self.ks, self.speeds
        target = {**self.pose, **pose}
        if seconds is None:
            seconds = ks.travel_time(
                self.pose, target, 40.0 if slow else sp.travel, 1.0 if slow else sp.min_travel
            )
        self.pose = ks.play(self.robot, [ks.Segment(self.pose, target, seconds)], sp.fps)

    def point(self, midi: int) -> None:
        if self.at == midi or not self.board.reaches(midi):
            return
        self.state.set(arm={**self.state.get("arm"), "at": midi, "pressing": None})
        self._move(self.board.keys[midi]["above"])
        self.at = midi

    def press(self, midi: int, learned: bool = False) -> None:
        if not self.board.reaches(midi):
            return
        self.point(midi)
        keys = self.board.keys[midi]
        self.state.set(arm={**self.state.get("arm"), "pressing": midi, "how": "scripted"})
        self.ears.expect(midi, 2.0)
        self._move(keys["down"], self.speeds.down)
        self._move(keys["down"], self.speeds.hold)
        self._move(keys["above"], self.speeds.down)
        self.state.set(arm={**self.state.get("arm"), "pressing": None})

    def away(self) -> None:
        if self.board.home and self.at is not None:
            self._move(self.board.home)
        self.at = None
        self.state.set(arm={**self.state.get("arm"), "at": None, "pressing": None})

    def close(self) -> None:
        self.ears.stop.clear()
        with contextlib.suppress(Exception):
            if self.board.rest:
                self._move(self.board.rest, slow=True)
        self.robot.disconnect()


class PolicyArm:
    """The learned SmolVLA policy presses the six sticker keys it was trained on (C3 to A3, the bass, with
    --prompts scripted; C4 to A4 for the older flow policy); everything else
    (pointing, the keys it never learned, fast notes in Play together) moves from taught poses. One
    `lerobot-rollout` session owns the arm the whole time: keys_rollout.py adds /pause and /goto to it."""

    name = "learned policy + taught poses"

    def __init__(
        self,
        board: Board,
        ears: Ears,
        state: State,
        model: str,
        port: str,
        speeds: Speeds,
        prompts: str = "notes",
        policy_keys: list[int] | None = None,
    ) -> None:
        import keys_play

        if not board.reach:
            sys.exit("The arm has no taught keys: run `keys_coach.py teach` first.")
        self.board, self.ears, self.state, self.speeds = board, ears, state, speeds
        self.keys_play = keys_play
        # The words the policy was trained on. notes: "press E3" (keys_scripted.py record --names);
        # scripted: "press key 3"; flow: "play mi, the third key" (the older teleop data).
        self.prompts = prompts
        self.policy_keys = set(policy_keys or (POLICY_WORDS if prompts == "flow" else STICKERS.values()))
        task = keys_play.task_text("do")
        command, self.label = keys_play.build_command(model, task, 0.0, port, None, True, 60, 1.0)
        command.append("--interactive=true")
        if not any(arg.startswith("--inference.action_starvation_grace_s") for arg in command):
            command.append("--inference.action_starvation_grace_s=8")
        command[:1] = [sys.executable, str(Path(__file__).with_name("keys_rollout.py"))]
        print(f"Policy: {self.label}. Loading takes about 15 seconds.", flush=True)
        self.process = subprocess.Popen(command, cwd=keys_play.ROOT, stdin=subprocess.PIPE, text=True)
        self.at: int | None = None
        self.first = True
        self.state.set(big="Waking the arm", small="The policy loads in about 15 seconds.")
        if board.home:
            self.goto(board.home, 1.5)

    def tell(self, line: str) -> None:
        if self.process.poll() is not None or self.process.stdin is None:
            raise RuntimeError("the policy's session has ended: restart keys_coach.py")
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def goto(self, pose: Pose, seconds: float) -> None:
        if self.ears.stop.is_set():
            raise Stopped
        self.tell(f"/goto {json.dumps(pose, separators=(',', ':'))} {seconds:.2f}")
        if self.ears.stop.wait(seconds + 0.15):  # the session moves on its own clock; give it the time
            raise Stopped

    def point(self, midi: int) -> None:
        if self.at == midi or not self.board.reaches(midi):
            return
        self.state.set(arm={**self.state.get("arm"), "at": midi, "pressing": None})
        self.goto(self.board.keys[midi]["above"], 0.6)
        self.at = midi

    def press(self, midi: int, learned: bool = False) -> None:
        if learned and midi in self.policy_keys:
            self._policy_press(midi)
        elif self.board.reaches(midi):
            self.point(midi)
            keys = self.board.keys[midi]
            self.state.set(arm={**self.state.get("arm"), "pressing": midi, "how": "scripted"})
            self.ears.expect(midi, 2.0)
            self.goto(keys["down"], self.speeds.down + self.speeds.hold)
            self.goto(keys["above"], self.speeds.down)
            self.state.set(arm={**self.state.get("arm"), "pressing": None})

    def task(self, midi: int) -> str:
        """The words the policy was trained on for this key."""
        if self.prompts == "notes":  # keys_scripted.py record --names
            return f"press {midi_name(midi)}"
        if self.prompts == "scripted":  # keys_scripted.py record
            return f"press key {next(k for k, m in STICKERS.items() if m == midi)}"
        return self.keys_play.task_text(POLICY_WORDS[midi])

    def _policy_press(self, midi: int, any_note: bool = False) -> None:
        """One press by the policy. Returns once its note has sounded, or after 12 seconds. any_note: also
        return when a different key sounds (the scorer: a wrong key is an answer too)."""
        with contextlib.suppress(Exception):  # Reachy is the policy's camera: its head goes to the saved pose
            import keys_reachy

            keys_reachy.lock()
        self.state.set(arm={**self.state.get("arm"), "at": midi, "pressing": midi, "how": "policy"})
        self.ears.expect(midi, 60.0 if self.first else 12.0)
        self.tell(f"/subtask {self.task(midi)}")
        self.tell("/start")
        deadline = time.monotonic() + (60.0 if self.first else 12.0)
        self.first = False
        seen = self.state.press_id
        try:
            while time.monotonic() < deadline:
                if self.ears.stop.is_set():
                    raise Stopped
                with self.ears.lock:
                    pending = any(m == midi for m, _ in self.ears.expected)
                if not pending or (any_note and self.state.press_id != seen):  # the note sounded
                    break
                time.sleep(0.02)
            # A policy trained on keys_scripted.py's presses lifts off the key by itself: give it the time.
            time.sleep(self.keys_play.HOLD_S if self.prompts == "flow" else 0.7)
        finally:
            self.ears.unexpect(midi)
            # The flow policy only learned presses and leaves its finger on the key: lift it. Then wait in the air.
            self.tell("/pause 4" if self.prompts == "flow" else "/pause")
            self.at = None
            self.state.set(arm={**self.state.get("arm"), "pressing": None})
            time.sleep(0.5)

    def away(self) -> None:
        if self.board.home and self.at is not None:
            self.goto(self.board.home, 0.8)
        self.at = None
        self.state.set(arm={**self.state.get("arm"), "at": None, "pressing": None})

    def close(self) -> None:
        if self.process.poll() is None:
            with contextlib.suppress(Exception):
                self.tell("/stop")
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=25)


# ------------------------------------------------------------------------------------------- the coach


def answer_phrase(notes: list[int], style: str, board: Board, longest: int = 6) -> list[int]:
    """The arm's answer to your phrase in a jam, on keys it reaches. echo: your notes. mirror: your
    phrase upside down, turning around its first note (up becomes down)."""
    if not notes:
        return []
    notes = notes[:longest]
    if style == "mirror":
        notes = [2 * notes[0] - n for n in notes]
    out = [board.nearest(n) for n in notes]
    return [n for n in out if n is not None]


# The course: short lessons in a fixed order, each with one goal. A lesson opens when the one before it is
# done. Three hearts per lesson: a wrong key costs one, and at zero the lesson stops and can be tried again.
COURSE: list[dict[str, Any]] = [
    {"id": "mary-1", "title": "First notes", "goal": "Follow the arm: three keys, one short phrase",
     "mode": "learn", "song": "mary", "phrases": [0], "passes": 1},
    {"id": "mary-2", "title": "Finish the tune", "goal": "The second phrase, then on your own",
     "mode": "learn", "song": "mary", "phrases": [1], "passes": 2},
    {"id": "mary-3", "title": "Play it back", "goal": "The arm plays a phrase, you repeat it",
     "mode": "watch", "song": "mary"},
    {"id": "mary-4", "title": "Duet", "goal": "You play Mary, the arm plays the bass",
     "mode": "together", "song": "mary"},
    {"id": "ode-1", "title": "A new song", "goal": "Ode to Joy, the first two phrases",
     "mode": "learn", "song": "ode", "phrases": [0, 1], "passes": 1},
    {"id": "ode-2", "title": "Ode duet", "goal": "The whole song with the arm on bass",
     "mode": "together", "song": "ode"},
    {"id": "jam", "title": "Jam", "goal": "Play three phrases: the arm answers each one",
     "mode": "jam", "rounds": 3},
]  # fmt: skip
HEARTS = 3
XP_LESSON, XP_CLEAN = 10, 5  # for finishing a lesson, and extra for finishing it without a wrong key
PROGRESS = ROOT / "outputs" / "keys101_progress.json"


class OutOfHearts(Exception):
    """Three wrong keys in one lesson."""


def load_progress(path: Path = PROGRESS) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    return {
        "done": list(data.get("done", [])),
        "xp": int(data.get("xp", 0)),
        "days": list(data.get("days", [])),
    }


def streak(days: list[str], today: str) -> int:
    """Days in a row with a finished lesson, ending today or yesterday. days: ISO dates."""
    import datetime

    have = {datetime.date.fromisoformat(d) for d in days}
    day = datetime.date.fromisoformat(today)
    if day not in have:
        day -= datetime.timedelta(days=1)
    count = 0
    while day in have:
        count += 1
        day -= datetime.timedelta(days=1)
    return count


def course_view(progress: dict[str, Any]) -> list[dict[str, Any]]:
    """The lessons as the page shows them: done, open (the next one) or locked."""
    out, unlocked = [], True
    for lesson in COURSE:
        done = lesson["id"] in progress["done"]
        status = "done" if done else ("open" if unlocked else "locked")
        out.append({"id": lesson["id"], "title": lesson["title"], "goal": lesson["goal"], "status": status})
        unlocked = unlocked and done
    return out


@dataclass
class Settings:
    passes: int = 2  # clean runs of a phrase in Learn before the next one
    max_tries: int = 4  # runs of a phrase before Learn moves on anyway
    hesitate_s: float = 2.5  # second run of Learn: the arm points only if you wait this long
    phrase_end_s: float = 1.5  # Jam: this much silence ends your phrase
    repeat_timeout_s: float = 12.0  # Watch and repeat: the longest wait for your answer
    bass_learned: bool = (
        True  # Play together: the policy plays the bass (slower), else taught poses (instant)
    )


class Coach:
    def __init__(
        self,
        arm: Arm,
        ears: Ears,
        state: State,
        songs: dict[str, Song],
        settings: Settings,
        progress_path: Path = PROGRESS,
    ) -> None:
        self.arm, self.ears, self.state, self.songs, self.settings = arm, ears, state, songs, settings
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.hearts: int | None = None  # None outside a lesson: free practice costs nothing
        self.progress_path = progress_path
        self.progress = load_progress(self.progress_path)
        self.show_course()

    # ---- the course

    def show_course(self) -> None:
        today = time.strftime("%Y-%m-%d")
        self.state.set(
            course=course_view(self.progress),
            xp=self.progress["xp"],
            days=streak(self.progress["days"], today),
        )

    def start_lesson(self, lesson_id: str) -> None:
        view = {row["id"]: row for row in course_view(self.progress)}
        lesson = next((entry for entry in COURSE if entry["id"] == lesson_id), None)
        if lesson is None or view[lesson_id]["status"] == "locked":
            return
        self.start(lesson["mode"], lesson.get("song"), lesson)

    def finish_lesson(self, lesson: dict[str, Any]) -> None:
        clean = self.state.get("score")["wrong"] == 0
        gained = XP_LESSON + (XP_CLEAN if clean else 0)
        first_time = lesson["id"] not in self.progress["done"]
        if first_time:
            self.progress["done"].append(lesson["id"])
        self.progress["xp"] += gained
        today = time.strftime("%Y-%m-%d")
        if today not in self.progress["days"]:
            self.progress["days"].append(today)
        with contextlib.suppress(OSError):
            self.progress_path.parent.mkdir(parents=True, exist_ok=True)
            self.progress_path.write_text(json.dumps(self.progress, indent=1) + "\n")
        self.show_course()
        nxt = next((row for row in course_view(self.progress) if row["status"] == "open"), None)
        praise = "Not one wrong key." if clean else "Done."
        self.say(
            "Lesson complete",
            f"{praise} +{gained} XP. " + (f"Next: {nxt['title']}." if nxt else "That was the whole course."),
            "ok",
        )

    def reset_course(self) -> None:
        self.progress = {"done": [], "xp": 0, "days": []}
        with contextlib.suppress(OSError):
            self.progress_path.unlink()
        self.show_course()

    def wrong(self, press: Press) -> None:
        """A wrong key: the page shows it red, and in a lesson it costs a heart."""
        self.state.judge(press.id, False)
        if self.hearts is not None:
            self.hearts -= 1
            self.state.set(hearts=self.hearts)
            if self.hearts <= 0:
                raise OutOfHearts

    # ---- control, from the page

    def start(self, mode: str, song_name: str | None, lesson: dict[str, Any] | None = None) -> None:
        if mode != "jam" and (mode not in ("learn", "watch", "together") or song_name not in self.songs):
            return
        with self.lock:
            self.stop()
            self.ears.stop.clear()
            self.ears.flush()
            song = self.songs.get(song_name or "")
            self.state.set(
                mode=mode,
                song=song.info() if song else None,
                score={"right": 0, "wrong": 0, "streak": 0},
                phrase=0,
                phrases=0,
                tone="",
                notes=[],
                target=None,
                lesson=lesson["id"] if lesson else None,
                hearts=HEARTS if lesson else None,
            )
            self.hearts = HEARTS if lesson else None
            options = lesson or {}

            def run() -> None:
                if mode == "jam":
                    self.jam(options.get("rounds"))
                elif mode == "learn":
                    self.learn(song, options.get("phrases"), options.get("passes"))  # type: ignore[arg-type]
                elif mode == "watch":
                    self.watch(song)  # type: ignore[arg-type]
                else:
                    self.together(song)  # type: ignore[arg-type]

            self.thread = threading.Thread(target=self._run, args=(run, lesson), daemon=True)
            self.thread.start()

    def stop(self) -> None:
        thread = self.thread
        if thread is not None and thread.is_alive():
            self.ears.stop.set()
            thread.join(timeout=5.0)
        self.thread = None

    def _run(self, run: Callable[[], None], lesson: dict[str, Any] | None = None) -> None:
        try:
            run()
            if lesson is not None:
                self.finish_lesson(lesson)
        except OutOfHearts:
            self.state.set(
                big="Out of hearts", small="Three wrong keys. Take a breath and try the lesson again.",
                tone="miss", target=None,
            )  # fmt: skip
        except Stopped:
            self.state.set(big="Stopped", small="Pick a song or a mode.", target=None)
        except Exception as e:  # noqa: BLE001  the page must show it, the server must keep running
            self.state.set(big="Something went wrong", small=str(e)[:200], target=None)
            import traceback

            traceback.print_exc()
        finally:
            self.ears.stop.clear()
            with contextlib.suppress(Exception):
                self.arm.away()
            self.hearts = None
            self.state.set(mode="", target=None, lesson=None, hearts=None)

    # ---- helpers

    def say(self, big: str, small: str = "", tone: str = "") -> None:
        self.state.set(big=big, small=small, tone=tone)

    def show_phrase(self, phrase: Phrase, index: int, total: int) -> list[dict[str, Any]]:
        notes = [{"midi": n, "name": midi_name(n), "status": "todo"} for n in phrase.notes]
        self.state.set(notes=notes, phrase=index, phrases=total)
        return notes

    def mark(self, notes: list[dict[str, Any]], i: int, status: str) -> None:
        notes[i] = {**notes[i], "status": status}
        self.state.set(notes=list(notes))

    def pause(self, seconds: float) -> None:
        if self.ears.stop.wait(seconds):
            raise Stopped

    def ghost(self, midi: int) -> None:
        """A note the arm cannot reach: the page plays it instead."""
        self.state.press(midi, "ghost")
        self.pause(0.45)

    def arm_plays(self, midi: int, learned: bool = False) -> None:
        key = self.arm.board.snap(midi)
        if key is None:
            self.ghost(midi)
            return
        self.arm.press(key, learned=learned)

    # ---- the modes

    def learn(self, song: Song, only: list[int] | None = None, passes: int | None = None) -> None:
        """Follow the finger: the arm points, you press, a mistake gets the right key played for you.
        only: just these phrases (a lesson). passes: clean runs of a phrase before the next one."""
        st = self.settings
        passes = passes or st.passes
        total = len(song.phrases)
        for p_index, phrase in enumerate(song.phrases):
            if only is not None and p_index not in only:
                continue
            clean, tries = 0, 0
            while clean < passes and tries < st.max_tries:
                tries += 1
                guided = clean == 0
                if not guided:
                    self.arm.away()
                notes = self.show_phrase(phrase, p_index, total)
                if guided:
                    self.say(
                        f"Phrase {p_index + 1} of {total}", "Follow the arm: press the key it points at."
                    )
                else:
                    self.say("Now mostly on your own", "The arm only helps if you wait.")
                self.pause(0.8)
                mistakes = 0
                for i, note in enumerate(phrase.notes):
                    self.mark(notes, i, "now")
                    self.state.set(target=note)
                    reach = self.arm.board.reaches(note)
                    pointed = False
                    if guided and reach:
                        self.arm.point(note)
                        pointed = True
                    while True:
                        press = self.ears.next(None if pointed or not reach else st.hesitate_s)
                        if press is None:  # you hesitate: help
                            self.arm.point(note)
                            pointed = True
                            self.say("Here", f"{midi_name(note)}: the key under the arm.")
                            continue
                        if press.midi == note:
                            self.state.judge(press.id, True)
                            self.mark(notes, i, "ok")
                            break
                        mistakes += 1
                        self.mark(notes, i, "miss")
                        self.wrong(press)
                        self.say("Not that one", f"Listen: this is {midi_name(note)}. Now you.", "miss")
                        self.state.john(note)
                        self.arm_plays(note)
                        self.mark(notes, i, "now")
                        pointed = reach
                self.state.set(target=None)
                if mistakes:
                    clean = 0
                    self.say(
                        "Almost", f"{mistakes} wrong key{'s' if mistakes > 1 else ''}. Once more.", "miss"
                    )
                else:
                    clean += 1
                    self.say("Clean", "Again, without help." if clean < passes else "Well played.", "ok")
                self.pause(1.2)
            self.arm.away()
        if only is None:
            self.say(f"You can play {song.title}", "Try it with the arm: Play together.", "ok")

    def watch(self, song: Song) -> None:
        """The arm plays a phrase, you play it back, MIDI checks each note."""
        total = len(song.phrases)
        for p_index, phrase in enumerate(song.phrases):
            for attempt in range(3):
                notes = self.show_phrase(phrase, p_index, total)
                self.say(
                    "Watch and listen",
                    f"Phrase {p_index + 1} of {total}." + (" Once more." if attempt else ""),
                )
                self.pause(0.6)
                for i, note in enumerate(phrase.notes):
                    self.mark(notes, i, "now")
                    self.state.john(note, 0.6)
                    self.arm_plays(note, learned=True)
                    self.mark(notes, i, "todo")
                self.arm.away()
                self.ears.flush()
                self.say("Your turn", "Play it back.")
                right = 0
                for i, note in enumerate(phrase.notes):
                    self.mark(notes, i, "now")
                    press = self.ears.next(self.settings.repeat_timeout_s)
                    if press is None:
                        break
                    ok = press.midi == note
                    right += ok
                    self.mark(notes, i, "ok" if ok else "miss")
                    if ok:
                        self.state.judge(press.id, True)
                    else:
                        self.wrong(press)
                if right == len(phrase.notes):
                    self.say("Perfect", "Next phrase.", "ok")
                    self.pause(1.2)
                    break
                self.say(f"{right} of {len(phrase.notes)}", "Watch it again.", "miss")
                self.pause(1.5)
        self.say("Song done", f"{song.title}, phrase by phrase.", "ok")

    def together(self, song: Song) -> None:
        """You play your part, the arm plays one note per bar the moment you play that bar's first note."""
        total = len(song.phrases)
        arm_notes = [
            (p, i, bar_arm) for p, ph in enumerate(song.phrases) for i, bar_arm in ph.bar_starts().items()
        ]
        upcoming = [self.arm.board.snap(a) if a is not None else None for _, _, a in arm_notes]
        first = next((n for n in upcoming if n is not None), None)
        self.say("Play together", "Start when you are ready. You lead, the arm follows.")
        if first is not None:
            self.arm.point(first)
        k = 0  # index into arm_notes
        for p_index, phrase in enumerate(song.phrases):
            notes = self.show_phrase(phrase, p_index, total)
            starts = phrase.bar_starts()
            for i, note in enumerate(phrase.notes):
                self.mark(notes, i, "now")
                self.state.set(target=note)
                while True:
                    press = self.ears.next(8.0)
                    if press is None:
                        self.say("Your note", f"{midi_name(note)}, lit up below.")
                        continue
                    if press.midi == note:
                        self.state.judge(press.id, True)
                        self.mark(notes, i, "ok")
                        # Light your next key now: you keep playing while the arm presses its note.
                        if i + 1 < len(notes):
                            self.mark(notes, i + 1, "now")
                        self.state.set(target=phrase.notes[i + 1] if i + 1 < len(notes) else None)
                        break
                    self.mark(notes, i, "miss")
                    self.wrong(press)
                if i in starts:
                    if starts[i] is not None:
                        self.say("Together", f"Phrase {p_index + 1} of {total}.")
                        self.arm_plays(starts[i], learned=self.settings.bass_learned)  # type: ignore[arg-type]
                    k += 1
                    nxt = next((n for n in upcoming[k:] if n is not None), None)
                    if nxt is not None:
                        self.arm.point(nxt)
            self.state.set(target=None)
        s = self.state.get("score")
        self.say(f"You played {song.title} together", f"{s['right']} notes right, {s['wrong']} slips.", "ok")
        self.pause(0.5)

    def jam(self, stop_after: int | None = None) -> None:
        """Call and response: you play, you stop, the arm answers. stop_after: this many exchanges (a
        lesson); else until you stop it."""
        rounds = 0
        while stop_after is None or rounds < stop_after:
            self.say("Your turn", "Play a short phrase, then stop.")
            self.state.set(target=None)  # the last exchange stays on screen until you play
            self.ears.flush()
            first = self.ears.next(None)
            assert first is not None
            yours = [first.midi]
            self.state.set(notes=[{"midi": first.midi, "name": midi_name(first.midi), "status": "ok"}])
            while len(yours) < 8:
                press = self.ears.next(self.settings.phrase_end_s)
                if press is None:
                    break
                yours.append(press.midi)
                self.state.set(notes=[{"midi": n, "name": midi_name(n), "status": "ok"} for n in yours])
            style = self.state.get("style")
            if style == "turns":
                style = "echo" if rounds % 2 == 0 else "mirror"
            answer = answer_phrase(yours, style, self.arm.board)
            self.say("The arm answers", "Echo." if style == "echo" else "Upside down.")
            shown = [{"midi": n, "name": midi_name(n), "status": "ok"} for n in yours]
            shown += [{"midi": n, "name": midi_name(n), "status": "arm"} for n in answer]
            self.state.set(notes=shown)
            for note in answer:
                self.state.john(note, 0.45)
                self.arm_plays(note, learned=True)
            rounds += 1
            self.pause(0.6)


# ------------------------------------------------------------------------------------------- the page


TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".wasm": "application/wasm",
    ".woff2": "font/woff2",
    ".otf": "font/otf",
}


def make_handler(coach: Coach, state: State, ears: Ears, web: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: Any) -> None:
            return

        def do_GET(self) -> None:
            path = urlsplit(self.path).path
            if path == "/events":
                self.stream()
                return
            if path == "/state":
                self.send(200, "application/json", state.snapshot().encode())
                return
            name = unquote(path).lstrip("/") or "coach.html"
            file = (web / name).resolve()
            if not file.is_relative_to(web) or not file.is_file():
                self.send(404, "text/plain; charset=utf-8", b"Not found")
            else:
                self.send(200, TYPES.get(file.suffix.lower(), "application/octet-stream"), file.read_bytes())

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/cmd":
                self.send(404, "text/plain; charset=utf-8", b"Not found")
                return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 10_000)
                cmd = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self.send(400, "text/plain; charset=utf-8", b"Bad JSON")
                return
            what = cmd.get("do")
            if what == "start":
                threading.Thread(
                    target=coach.start, args=(cmd.get("mode", ""), cmd.get("song")), daemon=True
                ).start()
            elif what == "lesson" and isinstance(cmd.get("id"), str):
                threading.Thread(target=coach.start_lesson, args=(cmd["id"],), daemon=True).start()
            elif what == "reset_course":
                coach.reset_course()
            elif what == "stop":
                threading.Thread(target=coach.stop, daemon=True).start()
            elif what == "key" and isinstance(cmd.get("midi"), int) and 21 <= cmd["midi"] <= 108:
                ears.key_down(cmd["midi"])
                threading.Timer(0.2, ears.key_up, args=(cmd["midi"],)).start()
            elif what == "style" and cmd.get("style") in ("echo", "mirror", "turns"):
                state.set(style=cmd["style"])
            self.send(200, "application/json", b'{"ok":true}')

        def send(self, status: int, kind: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            try:
                while True:
                    with state.cond:
                        state.cond.wait_for(lambda seen=seen: state.version != seen, timeout=1.0)
                        body = json.dumps(state.data) if state.version != seen else None
                        seen = state.version
                    out = ": keep-alive\n\n" + (f"event: state\ndata: {body}\n\n" if body else "")
                    self.wfile.write(out.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    return Handler


# ------------------------------------------------------------------------------------------- teach


def teach(args: argparse.Namespace) -> None:
    """Torque off: move the arm by hand over each key and press Enter. Saves after every pose."""
    import keys_scripted as ks

    lo, hi = note_midi(args.lo), note_midi(args.hi)
    board = load_board(args.board, Path("/nonexistent")) if args.board.exists() else Board()
    if args.only:
        targets = [note_midi(n) for n in args.only if n not in ("home", "rest")]
        extra = [n for n in args.only if n in ("home", "rest")]
    elif args.corners:
        whites = white_keys(lo, hi)
        targets = sorted({whites[0], whites[len(whites) // 2], whites[-1]})
        extra = ["rest", "home"]
    else:
        targets = [m for m in range(lo, hi + 1) if args.black or not is_black(m)]
        extra = ["rest", "home"]
    robot = ks.make_robot(
        argparse.Namespace(robot_port=args.robot_port, robot_id=args.robot_id, max_step=10.0), {}
    )
    robot.connect()
    try:
        robot.bus.disable_torque()
        print("Torque is off: the arm is limp. Hold it with one hand, press Enter with the other.")
        for name in extra:
            hint = (
                "folded, where it can rest with torque off"
                if name == "rest"
                else "high over the keyboard's middle"
            )
            input(f"\n{name}: {hint}. Enter. ")
            setattr(board, name, ks.read_pose(robot))
            save_board(board, args.board, args.robot_id)
        for midi in targets:
            name = midi_name(midi)
            poses: dict[str, Pose] = {}
            input(f"\n{name} above: the gripper tip about 3 cm straight over {name}. Enter. ")
            poses["above"] = ks.read_pose(robot)
            input(f"{name} down: now press {name} down until it sounds. Enter. ")
            poses["down"] = ks.read_pose(robot)
            board.keys[midi] = poses
            save_board(board, args.board, args.robot_id)
        filled = fill_white(board, lo, hi) if args.corners or args.fill else []
        if filled:
            save_board(board, args.board, args.robot_id)
            print(
                f"\nFilled in {', '.join(midi_name(m) for m in filled)}. Check them with `keys_coach.py test`."
            )
        print(
            f"\nSaved {args.board}: {len(board.reach)} keys. Put the arm in its rest pose before letting go."
        )
    except KeyboardInterrupt:
        print(f"\nStopped. What you taught is in {args.board}.")
    finally:
        robot.disconnect()


def test(args: argparse.Namespace) -> None:
    """Press every taught key once and check on MIDI that it sounded."""
    state = State()
    ears = Ears(state)
    midi = ears.open_midi(args.midi)
    board = load_board(args.board)
    arm = ScriptedArm(board, ears, state, args.robot_port, args.robot_id, Speeds(travel=60.0, down=0.25))
    bad = []
    try:
        for key in board.reach if not args.only else [note_midi(n) for n in args.only]:
            ears.flush()
            arm.press(key)
            heard = []
            with contextlib.suppress(queue.Empty):
                end = time.monotonic() + 0.6
                while time.monotonic() < end:
                    heard.append(ears.queue.get(timeout=max(0.01, end - time.monotonic())).midi)
            ok = heard == [key] if midi else None
            print(
                f"{midi_name(key):4} {'ok' if ok else ('no MIDI' if ok is None else f'heard {heard}')}",
                flush=True,
            )
            if ok is False:
                bad.append(midi_name(key))
        arm.away()
    finally:
        arm.close()
        ears.close()
    if bad:
        print(f"Re-teach: keys_coach.py teach --only {' '.join(bad)}")


def score(args: argparse.Namespace) -> None:
    """Let the policy press every key it learned, `rounds` times each in a fixed shuffled order, and count on
    MIDI how often the right key sounded first. Two models scored with the same --rounds get the same order."""
    import random

    state = State()
    ears = Ears(state)
    if ears.open_midi(args.midi) is None:
        sys.exit("The scorer needs the MIDI keyboard: it says which key the policy pressed.")
    learned = [note_midi(n) for n in args.policy_keys] if args.policy_keys else None
    arm = PolicyArm(
        load_board(args.board), ears, state, args.model, args.robot_port, Speeds(), args.prompts, learned
    )
    keys = sorted(arm.policy_keys)
    rng = random.Random(101)
    order = [key for _ in range(args.rounds) for key in rng.sample(keys, len(keys))]
    rows: list[dict[str, Any]] = []
    try:
        for i, target in enumerate(order, start=1):
            ears.flush()
            asked = time.monotonic()
            arm._policy_press(target, any_note=True)
            heard: list[Press] = []
            with contextlib.suppress(queue.Empty):
                while True:
                    heard.append(ears.queue.get_nowait())
            first = heard[0] if heard else None
            verdict = "no note" if first is None else ("correct" if first.midi == target else "wrong key")
            rows.append(
                {
                    "asked": midi_name(target),
                    "heard": [midi_name(p.midi) for p in heard],
                    "verdict": verdict,
                    "seconds": round(first.t - asked, 2) if first else None,
                }
            )
            print(f"[{i}/{len(order)}] {midi_name(target):3} {verdict:9} {rows[-1]['heard']}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        arm.close()
        ears.close()
    if not rows:
        return
    right = [r for r in rows if r["verdict"] == "correct"]
    times = [r["seconds"] for r in right if r["seconds"] is not None]
    per_key = {
        midi_name(k): f"{sum(r['verdict'] == 'correct' for r in rows if r['asked'] == midi_name(k))}"
        f"/{sum(r['asked'] == midi_name(k) for r in rows)}"
        for k in keys
    }
    summary = {
        "model": args.model,
        "label": arm.label,
        "prompts": args.prompts,
        "presses": len(rows),
        "correct": len(right),
        "wrong_key": sum(r["verdict"] == "wrong key" for r in rows),
        "no_note": sum(r["verdict"] == "no note" for r in rows),
        "mean_seconds_to_note": round(sum(times) / len(times), 2) if times else None,
        "per_key": per_key,
        "rows": rows,
    }
    out = ROOT / "outputs" / "keys_scores"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"keys101_{args.model}_{time.strftime('%H%M%S')}.json"
    path.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"\n{args.model}: {len(right)}/{len(rows)} right, {summary['wrong_key']} wrong key, "
          f"{summary['no_note']} no note, {summary['mean_seconds_to_note']} s to the note on average")  # fmt: skip
    print(f"Per key: {per_key}\nSaved {path}")


# ------------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm", choices=("sim", "scripted", "policy"), default="sim")
    parser.add_argument(
        "--model", default="keys101", help="The policy for --arm policy (see keys_play.py MODELS)"
    )
    parser.add_argument(
        "--prompts",
        choices=("notes", "scripted", "flow"),
        default="notes",
        help='The policy\'s task words: notes "press E3", scripted "press key 3", flow "play mi, the third key"',
    )
    parser.add_argument(
        "--policy-keys",
        nargs="+",
        help="The keys the policy may press (default: every sticker key in keys_layout.json). "
        "Fewer, e.g. C3 D3 E3 F3 G3 A3, leaves the others to the taught poses",
    )
    parser.add_argument("--robot-port", default="/dev/tty.usbmodem5B415331661")
    parser.add_argument("--robot-id", default="workshop_so101")
    parser.add_argument("--midi", default="auto", help='Part of the keyboard\'s name, or "none"')
    parser.add_argument("--board", type=Path, default=BOARD)
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--range", nargs=2, default=("C3", "C5"), metavar=("LOW", "HIGH"), help="Keys drawn")
    parser.add_argument("--speed", type=float, default=120.0, help="Scripted moves between keys, deg/s")
    parser.add_argument(
        "--bass",
        choices=("policy", "poses"),
        default="policy",
        help="Play together: the learned policy plays the bass, or the taught poses (faster)",
    )
    parser.add_argument("--sound", choices=("auto", "all", "arm", "none"), default="auto",
                        help="What the page plays itself. auto: the arm's notes without a robot, your clicks without MIDI")  # fmt: skip
    sub = parser.add_subparsers(dest="command")
    t = sub.add_parser("teach", help="Teach the key poses by hand (torque off)")
    t.add_argument("--from", dest="lo", default="C3")
    t.add_argument("--to", dest="hi", default="C5")
    t.add_argument("--black", action="store_true", help="The black keys too")
    t.add_argument("--corners", action="store_true", help="Teach three white keys, fill in the rest")
    t.add_argument("--fill", action="store_true", help="Fill in the white keys left untaught")
    t.add_argument("--only", nargs="+", help="Re-teach these keys (or home, rest)")
    sc = sub.add_parser(
        "score", help="The policy presses every key it learned: how often is it the right one?"
    )
    sc.add_argument("--rounds", type=int, default=5, help="Presses per key")
    s = sub.add_parser("test", help="Press every taught key once and check it on MIDI")
    s.add_argument("--only", nargs="+", help="Only these keys")
    args = parser.parse_args(argv)
    if args.command == "teach":
        teach(args)
        return
    if args.command == "score":
        score(args)
        return
    if args.command == "test":
        test(args)
        return

    state = State()
    ears = Ears(state)
    midi = ears.open_midi(args.midi)
    songs = load_songs()
    board = load_board(args.board)
    speeds = Speeds(travel=args.speed)
    arm: Arm
    if args.arm == "scripted":
        arm = ScriptedArm(board, ears, state, args.robot_port, args.robot_id, speeds)
    elif args.arm == "policy":
        learned = [note_midi(n) for n in args.policy_keys] if args.policy_keys else None
        arm = PolicyArm(board, ears, state, args.model, args.robot_port, speeds, args.prompts, learned)
    else:
        arm = SimArm(ears, state, board if board.reach else None)
    lo, hi = note_midi(args.range[0]), note_midi(args.range[1])
    sound = {
        "auto": {"you": midi is None, "arm": args.arm == "sim"},
        "all": {"you": True, "arm": True},
        "arm": {"you": False, "arm": True},
        "none": {"you": False, "arm": False},
    }[args.sound]
    reach = list(range(lo, hi + 1)) if arm.board.everything else arm.board.reach
    state.set(
        songs=[song.info() for song in songs.values()],
        range=[lo, hi],
        reach=reach,
        sound=sound,
        arm={"at": None, "pressing": None, "how": "", "name": arm.name},
        big="Pick a song",
        small="Then choose how to practise it.",
    )
    coach = Coach(arm, ears, state, songs, Settings(bass_learned=args.bass == "policy"))
    web = WEB.resolve()
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(coach, state, ears, web))
    except OSError:
        arm.close()
        raise SystemExit(
            f"Port {args.port} is taken: is the coach already running? Try --port {args.port + 1}."
        ) from None
    server.daemon_threads = True
    server.block_on_close = False
    print(
        f"Keys101: http://localhost:{args.port}   arm: {arm.name}   MIDI: {midi or 'none (click the keys)'}"
    )
    print(f"Songs: {', '.join(songs)}. Ctrl+C stops.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        coach.stop()
        arm.close()
        ears.close()
        server.server_close()


if __name__ == "__main__":
    main()
