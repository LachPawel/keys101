#!/usr/bin/env python
"""The piano companion: it plays the keys you name, teaches you a tune, runs a singing warm-up and keeps
score with the MIDI keyboard.

Who presses the keys (--hands):
  scripted   The saved poses of keys_scripted.py. Nothing learned: the backup, and the baseline to beat.
  text       A learned policy steered by text. `lerobot-rollout --interactive=true` runs underneath and is
             sent "/subtask press key N" for a note and "/subtask wait" after it sounded.
  sound      A learned policy that hears. The same rollout with one fixed task; you say the word yourself,
             the companion only shows it and keeps score.
For text and sound, everything after `--` goes to lerobot-rollout (policy, robot, cameras, device).

  score   Presses per key in a shuffled order, each judged by its MIDI note. Writes a JSON report.
  play    Push to talk. Enter, then: "play do re mi", "play ode", "teach me ode", "sing", "stop".
  teach   The lesson: the hands show a few notes, you repeat them (any octave), MIDI checks, a missed
          note is shown again.
  sing    The hands play a note, you sing it back, the microphone answers higher, lower or well done.

  R="uv run --with-requirements scripts/requirements-keys.txt"
  $R scripts/keys_companion.py score --hands scripted --per-key 5
  $R scripts/keys_companion.py score --hands text --per-key 5 -- --policy.path=PawelLach/smolvla_keys_text \\
      --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B415331661 --robot.id=workshop_so101 \\
      --robot.cameras='{...}' --device=mps
  $R scripts/keys_companion.py teach ode --hands scripted --speak
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from keys_common import (
    KEYS,
    NOTE_WORDS,
    SOLFEGE,
    SPOKEN,
    WAIT_TASK,
    read_song,
    task_for_key,
    write_json,
)

SONGS = Path(__file__).with_name("songs")
SOUND_TASK = "play the key you hear"
FILLER = {"the", "note", "notes", "key", "keys", "please", "and", "then", "now", "i", "m", "im", "am"}
PLAY_WORDS = {"play", "playing", "press", "hit"}
SONG_ALIASES = {"ode": {"ode", "joy", "old"}, "warmup": {"warmup", "warm"}}
IN_TUNE_CENTS = 50.0


class Clock:
    def now(self) -> float:
        return time.perf_counter()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class Midi(Protocol):
    t0: float

    def ons_since(self, start: float) -> list[tuple[float, int, int]]: ...


class Hands(Protocol):
    def press(self, key: int) -> None: ...

    def rest(self) -> None: ...

    def away(self) -> None: ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------------------- commands


@dataclass
class Command:
    kind: str  # play, song, teach, sing, stop or unknown
    keys: list[int] = field(default_factory=list)
    song: str | None = None
    note: str = ""  # why a command was not understood, or what was ignored


def find_song(tokens: list[str], songs: list[str]) -> str | None:
    for song in songs:
        if SONG_ALIASES.get(song, {song}) & set(tokens):
            return song
    return None


def parse_command(text: str, songs: list[str]) -> Command:
    """What a transcribed sentence asks for. Note words count only after "play" or when nothing else was said,
    because many of them ("to", "so", "me", "a") are ordinary words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return Command("unknown", note="nothing heard")
    if {"stop", "quit", "bye"} & set(tokens):
        return Command("stop")
    song = find_song(tokens, songs)
    if {"teach", "learn", "lesson"} & set(tokens):
        return Command("teach", song=song or "ode")
    if {"sing", "singing"} & set(tokens):
        return Command("sing")
    verb = next((i for i, tok in enumerate(tokens) if tok in PLAY_WORDS), None)
    if verb is not None and song:
        return Command("song", song=song)
    said = tokens if verb is None else tokens[verb + 1 :]
    said = [
        tok
        for i, tok in enumerate(said)
        if tok not in FILLER and not (tok == "a" and said[i + 1 : i + 2] and said[i + 1] in FILLER)
    ]
    note = "sharps and flats are not taught yet" if {"sharp", "flat"} & set(said) else ""
    said = [tok for tok in said if tok not in ("sharp", "flat")]
    keys = [NOTE_WORDS[tok] for tok in said if tok in NOTE_WORDS]
    if keys and (verb is not None or len(keys) == len(said)):
        return Command("play", keys=keys, note=note)
    return Command("unknown", note=note or f"no note or command in {text!r}")


def song_keys(name: str) -> list[int]:
    path = SONGS / f"{name}.txt"
    if not path.exists():
        sys.exit(f"No song {name!r}. Songs: {', '.join(sorted(p.stem for p in SONGS.glob('*.txt')))}")
    return [key for phrase in read_song(path) for key in phrase if key]


def split_phrases(keys: list[int], size: int, count: int | None) -> list[list[int]]:
    phrases = [keys[i : i + size] for i in range(0, len(keys), size)]
    return phrases[:count] if count else phrases


# ----------------------------------------------------------------------------------------- scoring


def wait_for_notes(
    midi: Midi, since: float, count: int, timeout: float, clock: Clock, settle: float = 0.0
) -> list[tuple[float, int, int]]:
    """Note-ons after `since` (session time): returns once `count` arrived and `settle` more seconds
    passed, or at the timeout."""
    deadline = clock.now() + timeout
    while clock.now() < deadline and len(midi.ons_since(since)) < count:
        clock.sleep(0.02)
    if settle and len(midi.ons_since(since)) >= count:
        clock.sleep(settle)
    return midi.ons_since(since)


def judge_trial(
    key: int, ons: list[tuple[float, int, int]], since: float, key_notes: dict[int, int]
) -> dict[str, Any]:
    """One commanded press: right when the only note that sounded is the key's own."""
    heard = [note for _, note, _ in ons]
    return {
        "key": key,
        "heard": heard,
        "ok": heard == [key_notes[key]],
        "seconds": round(ons[0][0] - since, 3) if ons else None,
        "velocity": ons[0][2] if ons else None,
    }


def trial_order(per_key: int, seed: int, keys: tuple[int, ...] = KEYS) -> list[int]:
    """Every key `per_key` times, shuffled within each round so no key waits long for its turn."""
    rng = np.random.default_rng(seed)
    return [int(key) for _ in range(per_key) for key in rng.permutation(keys)]


def press_and_judge(
    key: int, hands: Hands, midi: Midi, key_notes: dict[int, int], clock: Clock, timeout: float, settle: float
) -> dict[str, Any]:
    since = clock.now() - midi.t0
    hands.press(key)
    ons = wait_for_notes(midi, since, 1, timeout, clock, settle)
    hands.rest()
    return judge_trial(key, ons, since, key_notes)


def summarize(trials: list[dict[str, Any]], key_notes: dict[int, int]) -> dict[str, Any]:
    key_of = {note: key for key, note in key_notes.items()}
    per_key = {
        key: [t["ok"] for t in trials if t["key"] == key] for key in sorted({t["key"] for t in trials})
    }
    confusion: dict[str, dict[str, int]] = {}
    for t in trials:
        got = "none" if not t["heard"] else "+".join(str(key_of.get(n, f"note{n}")) for n in t["heard"])
        row = confusion.setdefault(str(t["key"]), {})
        row[got] = row.get(got, 0) + 1
    hits = [t for t in trials if t["ok"]]
    return {
        "presses": len(trials),
        "right": len(hits),
        "rate": round(len(hits) / len(trials), 3) if trials else None,
        "per_key": {str(key): f"{sum(oks)}/{len(oks)}" for key, oks in per_key.items()},
        "asked_to_heard": confusion,
        "median_seconds": round(float(np.median([t["seconds"] for t in hits])), 2) if hits else None,
        "mean_velocity": round(float(np.mean([t["velocity"] for t in hits])), 1) if hits else None,
    }


def print_summary(summary: dict[str, Any], label: str) -> None:
    chance = 1 / max(len(summary["per_key"]), 1)
    print(
        f"\n{label}: {summary['right']}/{summary['presses']} right ({summary['rate']:.0%}, chance {chance:.0%})"
    )
    for key, row in summary["asked_to_heard"].items():
        heard = ", ".join(f"{got} x{n}" for got, n in sorted(row.items()))
        print(f"  key {key}: {summary['per_key'][key]}   heard: {heard}")
    if summary["right"]:
        print(
            f"  note after {summary['median_seconds']} s (median), velocity {summary['mean_velocity']} (mean)"
        )


# ------------------------------------------------------------------------------------------- hands


class ScriptedHands:
    """The saved poses: every press is the taught motion."""

    def __init__(self, args: argparse.Namespace) -> None:
        import keys_scripted as ks

        self.ks, self.fps = ks, args.fps
        self.poses = ks.load_poses(args.poses)
        self.timing = ks.Timing(vary=0.0, jitter=0.0)
        self.rng = np.random.default_rng(0)
        self.robot = ks.make_robot(args, {})
        self.robot.connect()
        self.pose = ks.move_to(self.robot, self.poses["home"], self.fps)

    def press(self, key: int) -> None:
        segments = self.ks.press_segments(self.pose, self.poses, key, self.timing, self.rng)
        self.pose = self.ks.play(self.robot, segments, self.fps)

    def rest(self) -> None:
        pass

    def away(self) -> None:
        self.pose = self.ks.move_to(self.robot, self.poses["home"], self.fps, speed=60.0)

    def close(self) -> None:
        self.ks.park(self.robot, self.poses, self.fps, ask=False)


class RolloutHands:
    """A learned policy inside `lerobot-rollout --interactive=true`, steered through its chat on stdin."""

    def __init__(
        self, mode: str, rollout_args: list[str], words: dict[int, str], process: Any = None
    ) -> None:
        self.mode, self.words = mode, words
        task = WAIT_TASK if mode == "text" else SOUND_TASK
        argv = [sys.executable, "-m", "lerobot.scripts.lerobot_rollout", "--strategy.type=base"]
        argv += ["--interactive=true", f"--task={task}", "--play_sounds=false", *rollout_args]
        self.process = process or subprocess.Popen(argv, stdin=subprocess.PIPE, text=True)  # noqa: S603

    def send(self, line: str) -> None:
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def start(self) -> None:
        input(
            "Wait until the rollout below has loaded the policy and connected the robot, then press Enter. "
        )
        self.send("/start")

    def press(self, key: int) -> None:
        if self.mode == "text":
            self.send(f"/subtask {task_for_key(key)}")
        else:
            print(f"\n        >>>  SAY:  {self.words[key].upper()}  <<<", flush=True)

    def rest(self) -> None:
        if self.mode == "text":
            self.send(f"/subtask {WAIT_TASK}")

    def away(self) -> None:
        pass  # a policy only knows presses: play your part an octave higher instead

    def close(self) -> None:
        self.send("/stop")
        self.process.stdin.close()
        self.process.wait(timeout=60)


# ------------------------------------------------------------------------------- lesson and singing


class Face:
    """What the companion says and how it reacts. Reachy Mini's voice and antennas plug in here."""

    def __init__(self, speak: bool = False) -> None:
        self.speak = speak

    def say(self, text: str) -> None:
        print(f"  Reachy: {text}", flush=True)
        if self.speak:
            subprocess.run(["say", text], check=False)  # noqa: S603, S607  macOS speech

    def react(self, good: bool) -> None:
        print("  Reachy: " + ("\\o/" if good else "(shakes head)"), flush=True)


def lesson(
    phrases: list[list[int]],
    hands: Hands,
    midi: Midi,
    key_notes: dict[int, int],
    face: Face,
    clock: Clock,
    tries: int = 3,
    press_timeout: float = 10.0,
    turn_seconds_per_note: float = 3.0,
) -> list[dict[str, Any]]:
    """Show a phrase, hear it back, show a missed note again. Your notes count in any octave."""
    report = []
    for number, phrase in enumerate(phrases, start=1):
        want = [key_notes[key] % 12 for key in phrase]
        learned = False
        for attempt in range(1, tries + 1):
            face.say("Watch." if attempt == 1 else "Once more. Watch.")
            for key in phrase:
                press_and_judge(key, hands, midi, key_notes, clock, press_timeout, settle=0.8)
            hands.away()
            face.say("Your turn.")
            since = clock.now() - midi.t0
            played = wait_for_notes(midi, since, len(phrase), turn_seconds_per_note * len(phrase) + 2, clock)
            got = [note % 12 for _, note, _ in played][: len(phrase)]
            report.append({"phrase": number, "attempt": attempt, "want": want, "got": got, "ok": got == want})
            if got == want:
                face.react(True)
                learned = True
                break
            face.react(False)
            miss = next((i for i, (a, b) in enumerate(zip(want, got, strict=False)) if a != b), len(got))
            face.say(f"Not quite. Note {miss + 1} is this one.")
            press_and_judge(phrase[miss], hands, midi, key_notes, clock, press_timeout, settle=0.8)
        if not learned:
            face.say("We will come back to that one.")
    face.say("That is the tune." if all(r["ok"] for r in report[-1:]) else "Good practice.")
    return report


def sung_pitch(audio: np.ndarray, rate: int, lo: float = 70.0, hi: float = 800.0) -> float | None:
    """The pitch of a sung note in Hz, from the loudest 0.4 s, or None when nobody sang."""
    size = int(0.4 * rate)
    if len(audio) < size:
        return None
    audio = audio.astype(np.float64)
    energy = np.convolve(audio**2, np.ones(size), mode="valid")
    start = int(np.argmax(energy))
    frame = audio[start : start + size]
    frame = frame - frame.mean()
    if np.sqrt(np.mean(frame**2)) < 0.005:
        return None
    spectrum = np.fft.rfft(frame * np.hanning(size), 2 * size)
    corr = np.fft.irfft(np.abs(spectrum) ** 2)[:size]
    corr /= corr[0] + 1e-12
    lags = np.arange(int(rate / hi), int(rate / lo))
    if corr[lags].max() < 0.3:  # no clear period: breath or noise
        return None
    # The first strong peak, not the strongest: later peaks are the same note an octave or more down.
    strong = lags[corr[lags] >= 0.85 * corr[lags].max()]
    lag = int(strong[0])
    while lag + 1 < size and corr[lag + 1] > corr[lag]:
        lag += 1
    a, b, c = corr[lag - 1], corr[lag], corr[lag + 1]
    shift = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else 0.0
    return float(rate / (lag + shift))


def judge_pitch(hz: float | None, target_midi: int) -> tuple[float | None, str]:
    """Cents above (+) or below (-) the target, in whichever octave you sang, and what to say."""
    if hz is None:
        return None, "I did not hear you. Sing la."
    target_hz = 440.0 * 2 ** ((target_midi - 69) / 12)
    cents = 1200 * np.log2(hz / target_hz)
    cents = float((cents + 600) % 1200 - 600)
    if abs(cents) <= IN_TUNE_CENTS:
        return cents, "Well done."
    return cents, "A little higher." if cents < 0 else "A little lower."


class Microphone:
    def __init__(self, device: str | None, rate: int = 16000) -> None:
        self.device, self.rate = device, rate

    def record(self, seconds: float) -> np.ndarray:
        import sounddevice as sd

        audio = sd.rec(
            int(seconds * self.rate), samplerate=self.rate, channels=1, dtype="float32", device=self.device
        )
        sd.wait()
        return audio[:, 0]


def singing(
    keys: list[int],
    hands: Hands,
    midi: Midi,
    key_notes: dict[int, int],
    mic: Any,
    face: Face,
    clock: Clock,
    tries: int = 3,
    press_timeout: float = 10.0,
) -> list[dict[str, Any]]:
    """Play a note, listen to you sing it, say higher, lower or well done. Then the next note."""
    report = []
    for key in keys:
        for attempt in range(1, tries + 1):
            press_and_judge(key, hands, midi, key_notes, clock, press_timeout, settle=0.8)
            clock.sleep(0.5)  # let the piano note die away before listening
            hz = sung_pitch(mic.record(1.5), mic.rate)
            cents, answer = judge_pitch(hz, key_notes[key])
            report.append({"key": key, "attempt": attempt, "hz": hz, "cents": cents, "answer": answer})
            face.say(answer)
            if answer == "Well done.":
                face.react(True)
                break
    return report


# ------------------------------------------------------------------------------------ voice commands


class Listener:
    """Push to talk: records until you stop speaking, cleans it with Voice Focus, transcribes it."""

    PROMPT = "Play do, re, mi, fa, sol, la. Play one, two, three. Play Ode to Joy. Teach me. Sing. Stop."

    def __init__(self, device: str | None, model: str, voice_focus: bool, rate: int = 16000) -> None:
        self.device, self.model, self.rate = device, model, rate
        self.key = self.vf_model = None
        if voice_focus:
            import audio_hooks

            self.key = audio_hooks.license_key()
            self.vf_model = audio_hooks._load_model(audio_hooks._sdk(), audio_hooks.VOICE_FOCUS_MODEL)
        # The first Whisper call loads the model, which can take a minute: do it now, not on your first command.
        print("Loading the speech models...", flush=True)
        self.transcribe(np.zeros(rate, dtype=np.float32))

    def record(self, max_s: float = 6.0, quiet_s: float = 0.8) -> np.ndarray:
        import sounddevice as sd

        block = self.rate // 20
        blocks: list[np.ndarray] = []
        floor, spoke, quiet = None, False, 0.0
        with sd.InputStream(
            device=self.device, channels=1, samplerate=self.rate, dtype="float32", blocksize=block
        ) as mic:
            while len(blocks) * block < max_s * self.rate:
                chunk = mic.read(block)[0][:, 0].copy()
                blocks.append(chunk)
                level = float(np.sqrt(np.mean(chunk**2)))
                floor = level if floor is None else min(floor, level)
                if level > max(4 * floor, 0.01):
                    spoke, quiet = True, 0.0
                elif spoke:
                    quiet += block / self.rate
                    if quiet >= quiet_s:
                        break
        return np.concatenate(blocks)

    def transcribe(self, audio: np.ndarray) -> str:
        import mlx_whisper

        if self.key:
            import audio_hooks

            audio = audio_hooks.enhance(audio, self.rate, self.key, self.vf_model)
        result = mlx_whisper.transcribe(
            audio.astype(np.float32),
            path_or_hf_repo=self.model,
            language="en",
            condition_on_previous_text=False,
            initial_prompt=self.PROMPT,
        )
        return str(result["text"]).strip()

    def hear(self) -> str:
        return self.transcribe(self.record())


# ------------------------------------------------------------------------------------------- main


@dataclass
class Session:
    hands: Hands
    midi: Any
    key_notes: dict[int, int]
    face: Face
    clock: Clock
    args: argparse.Namespace

    def press(self, keys: list[int]) -> list[dict[str, Any]]:
        trials = []
        for key in keys:
            trial = press_and_judge(
                key, self.hands, self.midi, self.key_notes, self.clock, self.args.timeout, 0.8
            )
            print(
                f"  key {key}: {'ok' if trial['ok'] else 'MISSED, heard ' + str(trial['heard'])}", flush=True
            )
            trials.append(trial)
            self.clock.sleep(self.args.gap)
        return trials

    def teach(self, song: str) -> list[dict[str, Any]]:
        phrases = split_phrases(song_keys(song), self.args.phrase_size, self.args.phrases)
        return lesson(phrases, self.hands, self.midi, self.key_notes, self.face, self.clock)

    def sing(self) -> list[dict[str, Any]]:
        mic = Microphone(self.args.mic)
        return singing(self.args.notes, self.hands, self.midi, self.key_notes, mic, self.face, self.clock)


def run_score(session: Session) -> Any:
    args = session.args
    trials = session.press(trial_order(args.per_key, args.seed, tuple(args.keys)))
    summary = summarize(trials, session.key_notes)
    print_summary(summary, args.label or args.hands)
    return {"summary": summary, "trials": trials}


def run_play(session: Session) -> Any:
    from transcribe import DEFAULT_MODEL

    args = session.args
    listener = Listener(args.mic, args.model or DEFAULT_MODEL, args.voice_focus)
    songs = sorted(path.stem for path in SONGS.glob("*.txt"))
    log = []
    while True:
        input("\nEnter, then speak (Ctrl+C quits). ")
        text = listener.hear()
        command = parse_command(text, songs)
        print(f"  heard: {text!r} -> {command.kind} {command.keys or command.song or ''} {command.note}")
        entry: dict[str, Any] = {"text": text, "kind": command.kind}
        if command.kind == "stop":
            log.append(entry)
            return log
        if command.kind == "play":
            entry["trials"] = session.press(command.keys)
        elif command.kind == "song":
            entry["trials"] = session.press(song_keys(command.song or "ode"))
        elif command.kind == "teach":
            entry["lesson"] = session.teach(command.song or "ode")
        elif command.kind == "sing":
            entry["singing"] = session.sing()
        log.append(entry)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    rollout_args = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--hands", choices=("scripted", "text", "sound"), default="scripted")
    common.add_argument("--midi", default="auto", help="MIDI input: part of its name, or auto")
    common.add_argument("--key-notes", type=Path, default=Path("keys_midi.json"))
    common.add_argument("--poses", type=Path, default=Path("keys_poses.json"))
    common.add_argument("--robot-port", default="/dev/tty.usbmodem5B415331661")
    common.add_argument("--robot-id", default="workshop_so101")
    common.add_argument("--fps", type=int, default=30)
    common.add_argument("--max-step", type=float, default=10.0)
    common.add_argument(
        "--words", choices=("numbers", "solfege"), default="numbers", help="What sound hands show"
    )
    common.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for a note after asking")
    common.add_argument("--gap", type=float, default=1.0, help="Seconds between presses")
    common.add_argument("--mic", help="Microphone for commands and singing (default: the system input)")
    common.add_argument(
        "--speak", action="store_true", help="Say Reachy's lines aloud (not with sound hands)"
    )
    common.add_argument("--phrase-size", type=int, default=4, help="Notes per lesson phrase")
    common.add_argument("--phrases", type=int, default=2, help="Lesson phrases (0: the whole song)")
    common.add_argument(
        "--notes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 4, 3, 2, 1], help="Singing notes"
    )
    common.add_argument("--out", type=Path, default=Path("outputs/keys_scores"))
    common.add_argument("--label", help="Name of this run in the report, e.g. smolvla_text")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("score", parents=[common], help="Shuffled presses judged by MIDI")
    p.add_argument("--per-key", type=int, default=5)
    p.add_argument("--keys", type=int, nargs="+", default=list(KEYS), choices=KEYS)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("play", parents=[common], help="Push-to-talk voice commands")
    p.add_argument("--model", help="mlx-whisper model (default: the one transcribe.py uses)")
    p.add_argument(
        "--no-voice-focus", dest="voice_focus", action="store_false", help="Raw microphone into Whisper"
    )
    p = sub.add_parser("teach", parents=[common], help="The lesson for one song")
    p.add_argument("song", nargs="?", default="ode")
    sub.add_parser("sing", parents=[common], help="The singing warm-up")
    args = parser.parse_args(argv)

    import keys_scripted as ks

    key_notes = ks.load_key_notes(args.key_notes)
    if not key_notes:
        sys.exit(f"No {args.key_notes}. Run `keys_scripted.py test` with the MIDI keyboard plugged in first.")
    clock = Clock()
    midi = ks.open_midi(args.midi, clock.now())
    if midi is None:
        sys.exit("The companion needs the MIDI keyboard: it is how every note is checked.")
    if args.hands == "sound" and args.speak:
        sys.exit("--speak with sound hands would talk into the robot's ears. Drop one of them.")
    words = SOLFEGE if args.words == "solfege" else SPOKEN
    hands: Any = (
        ScriptedHands(args) if args.hands == "scripted" else RolloutHands(args.hands, rollout_args, words)
    )
    session = Session(hands, midi, key_notes, Face(args.speak), clock, args)
    result: Any = None
    try:
        if args.hands != "scripted":
            hands.start()
        if args.command == "score":
            result = run_score(session)
        elif args.command == "play":
            result = run_play(session)
        elif args.command == "teach":
            result = session.teach(args.song)
        else:
            result = session.sing()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if result is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            name = (
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.command}_{args.label or args.hands}.json"
            )
            write_json(args.out / name, {"command": args.command, "hands": args.hands, "result": result})
            print(f"Report: {args.out / name}")
        midi.close()
        hands.close()


if __name__ == "__main__":
    main()
