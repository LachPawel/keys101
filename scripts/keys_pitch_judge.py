#!/usr/bin/env python
"""Can Gemini Decision 1 hear whether a sung note is flat, in tune or sharp?

It gets a reference tone and a sung note, and answers two questions with probabilities. Its answers are
compared with the pitch function the companion already uses (keys_companion.sung_pitch), which is the truth
here: it measures the note in cents.

    R="uv run --with-requirements scripts/requirements-keys.txt"
    $R scripts/keys_pitch_judge.py synth                 # 20 made-up "sung" notes with known errors, no microphone
    $R scripts/keys_pitch_judge.py record                # 20 notes you sing: it plays a tone, you sing it back
    $R scripts/keys_pitch_judge.py judge <folder>        # sends every clip to Gemini Decision 1, prints the tally
    $R scripts/keys_pitch_judge.py judge <folder> --no-reference   # only the sung note and the note's name
    $R scripts/keys_pitch_judge.py report <folder>       # the tally again, no API calls

`synth` is the easy case: clean tones with a little vibrato. If the model fails there, it fails on a voice.
`record` asks you to sing flat, in tune or sharp on purpose; the label is what you really sang, measured.

`judge` uploads the clips to Google. The key is read from GEMINI_API_KEY (or GOOGLE_API_KEY), and when
neither is set, from ~/code/codex/.env. Clips and results go to outputs/keys_pitch/<stamp>_<kind>.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import keys_companion as kc
import numpy as np
from keys_common import load_json, write_json, write_wav
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "keys_pitch"
ENV_FILE = Path.home() / "code" / "codex" / ".env"
KEY_NAMES = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/decisions"
MODEL = "gemini-decision-1"
RATE = 16000
NOTES = {"do": 60, "re": 62, "mi": 64, "fa": 65, "sol": 67, "la": 69}  # C4 to A4, the six stickers
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
LABELS = ("flat", "in_tune", "sharp")
# Cents off the target for the made-up notes: 4 in tune, 3 flat and 3 sharp per note.
SYNTH_OFFSETS = (-200, -100, -70, -30, 0, 0, 30, 70, 100, 200)
Post = Callable[[dict[str, Any], str], dict[str, Any]]


def api_key(env_file: Path = ENV_FILE) -> str:
    for name in KEY_NAMES:
        if key := os.environ.get(name):
            return key
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip().removeprefix("export ") in KEY_NAMES and value.strip():
                return value.strip().strip("\"'")
    sys.exit(f"No Gemini key: set GEMINI_API_KEY or add it to {env_file}")


# ------------------------------------------------------------------------------------------- sound


def midi_hz(midi: float) -> float:
    return float(440.0 * 2 ** ((midi - 69) / 12))


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def reference_tone(hz: float, seconds: float = 1.0, rate: int = RATE) -> NDArray[np.float32]:
    """A plucked, piano-like note: harmonics that die away."""
    t = np.arange(int(seconds * rate)) / rate
    tone = sum(np.sin(2 * np.pi * hz * k * t) / k**1.5 for k in range(1, 7))
    envelope = np.minimum(t / 0.005, 1.0) * np.exp(-t / 0.5) * np.minimum((seconds - t) / 0.05, 1.0)
    return (0.25 * tone * envelope).astype(np.float32)


def sung_tone(
    hz: float, rng: np.random.Generator, seconds: float = 1.6, rate: int = RATE
) -> NDArray[np.float32]:
    """A stand-in for a sung note: a held tone with harmonics, a little vibrato and breath noise."""
    t = np.arange(int(seconds * rate)) / rate
    vibrato = 2 ** (10 / 1200 * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 2 * np.pi)))  # 10 cents each way
    phase = 2 * np.pi * np.cumsum(hz * vibrato) / rate
    voice = sum(np.sin(k * phase) / k for k in range(1, 7))
    envelope = np.minimum(t / 0.08, 1.0) * np.minimum((seconds - t) / 0.1, 1.0)
    held = 0.2 * voice * envelope + 0.003 * rng.standard_normal(t.size)
    lead_in = 0.003 * rng.standard_normal(int(0.2 * rate))  # you start a moment after the tone
    return np.concatenate([lead_in, held]).astype(np.float32)


def label_for(cents: float, in_tune_cents: float = kc.IN_TUNE_CENTS) -> str:
    if abs(cents) <= in_tune_cents:
        return "in_tune"
    return "flat" if cents < 0 else "sharp"


def measure(audio: NDArray[np.float32], target_midi: int) -> tuple[float | None, float | None]:
    """The sung pitch in Hz, and its cents above (+) or below (-) the target in whichever octave it was sung."""
    hz = kc.sung_pitch(audio, RATE)
    return hz, kc.judge_pitch(hz, target_midi)[0]


def save_clip(
    folder: Path,
    index: int,
    note: str,
    asked: str,
    sung: NDArray[np.float32],
    reference: NDArray[np.float32],
) -> dict[str, Any] | None:
    """Write one clip and its reference tone. None when no clear note is in the recording."""
    target = NOTES[note]
    hz, cents = measure(sung, target)
    if hz is None or cents is None:
        return None
    clip, ref = f"clip_{index:02d}.wav", f"ref_{index:02d}.wav"
    write_wav(folder / clip, sung, RATE)
    write_wav(folder / ref, reference, RATE)
    return {
        "id": index,
        "note": note,
        "target_midi": target,
        "asked": asked,
        "clip": clip,
        "ref": ref,
        "hz": round(hz, 2),
        "cents": round(cents, 1),
        "label": label_for(cents),
    }


def new_folder(kind: str, out: Path | None) -> Path:
    folder = out or OUT / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{kind}"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def make_synth(
    folder: Path,
    notes: list[str],
    offsets: tuple[int, ...] = SYNTH_OFFSETS,
    octave: int = 0,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Made-up sung notes, `offsets` cents off each note. `octave` moves the voice, not the reference."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    clips = []
    for note in notes:
        for cents in offsets:
            sung_hz = midi_hz(NOTES[note] + 12 * octave + cents / 100)
            clip = save_clip(
                folder,
                len(clips) + 1,
                note,
                f"{cents:+d} cents",
                sung_tone(sung_hz, rng),
                reference_tone(midi_hz(NOTES[note])),
            )
            if clip is None:
                sys.exit(f"The pitch function found no note in a made-up tone at {sung_hz:.1f} Hz.")
            clips.append(clip)
    write_json(folder / "clips.json", {"kind": "synth", "octave": octave, "clips": clips})
    return clips


def record(folder: Path, notes: list[str], trials: int, mic: str | None, seed: int) -> list[dict[str, Any]]:
    """Play a tone, record you singing it back flat, in tune or sharp on purpose, measure what you sang."""
    import sounddevice as sd

    asks = [("FLAT (a bit too low)", "IN TUNE", "SHARP (a bit too high)")[i % 3] for i in range(trials)]
    random.Random(seed).shuffle(asks)
    microphone = kc.Microphone(mic, RATE)
    clips: list[dict[str, Any]] = []
    for trial, ask in enumerate(asks, start=1):
        note = notes[(trial - 1) % len(notes)]
        reference = reference_tone(midi_hz(NOTES[note]))
        for _attempt in range(3):
            input(f"\n[{trial}/{trials}] {note} ({note_name(NOTES[note])}). Sing it {ask}. Enter to hear it.")
            sd.play(reference, RATE)
            sd.wait()
            print("   Sing now...", flush=True)
            clip = save_clip(folder, trial, note, ask, microphone.record(2.0), reference)
            if clip is not None:
                print(f"   You sang {clip['cents']:+.0f} cents: {clip['label']}.")
                clips.append(clip)
                break
            print("   I did not hear a clear note. Again.")
        write_json(folder / "clips.json", {"kind": "sung", "clips": clips})
    return clips


# ------------------------------------------------------------------------------------ the decision model


def audio_block(path: Path) -> dict[str, str]:
    return {"type": "audio", "data": base64.b64encode(path.read_bytes()).decode(), "mime_type": "audio/wav"}


def build_request(clip: Path, reference: Path | None, target_midi: int) -> dict[str, Any]:
    """One request: the reference tone (unless left out), the sung note, and two questions about them."""
    if reference is not None:
        target = "the reference tone in recording 1"
        content: list[dict[str, str]] = [
            {"type": "text", "text": "Recording 1, the reference tone:"},
            audio_block(reference),
            {"type": "text", "text": "Recording 2, a person singing one note to match the reference tone:"},
            audio_block(clip),
        ]
    else:
        target = f"the note {note_name(target_midi)} ({midi_hz(target_midi):.1f} Hz)"
        content = [
            {"type": "text", "text": f"A person singing one note, aiming for {target}:"},
            audio_block(clip),
        ]
    scope = "Ignore octave differences, loudness and voice quality: judge only the pitch."
    return {
        "model": MODEL,
        "input": content,
        "questions": {
            "direction": {
                "type": "choice",
                "instructions": f"Compared with {target}, is the sung note flat, in tune or sharp? {scope}",
                "criteria": {
                    "flat": "The sung note is more than a quarter tone lower than the target pitch.",
                    "in_tune": "The sung note is within a quarter tone of the target pitch.",
                    "sharp": "The sung note is more than a quarter tone higher than the target pitch.",
                },
            },
            "in_tune": {
                "type": "noul",
                "instructions": f"Is the sung note within a quarter tone of {target}? {scope}",
            },
        },
    }


def http_post(payload: dict[str, Any], key: str, timeout: float = 60.0) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    for attempt in range(4):
        request = urllib.request.Request(API_URL, data=body, headers=headers, method="POST")  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310  # nosec B310
                return dict(json.loads(response.read()))
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:600]
            if error.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(2.0 * 2**attempt)
                continue
            sys.exit(f"Decisions API error {error.code}: {detail}")
    raise AssertionError("unreachable")


def results_file(folder: Path, use_reference: bool) -> Path:
    return folder / ("judged.json" if use_reference else "judged_no_reference.json")


def judge(
    folder: Path, key: str, use_reference: bool = True, post: Post = http_post, limit: int | None = None
) -> list[dict[str, Any]]:
    """Ask the model about every clip in the folder and save its answers next to the measured truth."""
    rows = []
    for clip in load_json(folder / "clips.json")["clips"][:limit]:
        reference = folder / clip["ref"] if use_reference else None
        start = time.perf_counter()
        result = post(build_request(folder / clip["clip"], reference, clip["target_midi"]), key)
        seconds = time.perf_counter() - start
        direction, in_tune = result["answers"]["direction"], result["answers"]["in_tune"]
        row = {
            **clip,
            "choice": direction["choice"],
            "probabilities": direction["probabilities"],
            "p_in_tune": in_tune["noul"],
            "seconds": round(seconds, 2),
            "input_tokens": result.get("usage", {}).get("total_input_tokens"),
        }
        mark = "ok   " if row["choice"] == row["label"] else "WRONG"
        print(
            f"  {clip['id']:2d} {clip['note']:3s} {clip['cents']:+7.1f} cents  truth {clip['label']:7s} "
            f"model {row['choice']:7s} {mark}  p(in tune) {row['p_in_tune']:.2f}  {seconds:.1f} s",
            flush=True,
        )
        rows.append(row)
    write_json(
        results_file(folder, use_reference), {"model": MODEL, "reference": use_reference, "rows": rows}
    )
    return rows


def ranks_higher(high: list[float], low: list[float]) -> float | None:
    """How often a value from `high` beats one from `low` (ties count half). 0.5 is chance, 1.0 is perfect."""
    if not high or not low:
        return None
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in high for b in low)
    return wins / (len(high) * len(low))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["label"] for row in rows)
    off = [row for row in rows if row["label"] != "in_tune"]
    direction_right = sum(
        (row["probabilities"]["flat"] > row["probabilities"]["sharp"]) == (row["label"] == "flat")
        for row in off
    )
    tokens = [row["input_tokens"] for row in rows if row.get("input_tokens")]
    return {
        "clips": len(rows),
        "labels": {label: labels.get(label, 0) for label in LABELS},
        "correct": sum(row["choice"] == row["label"] for row in rows),
        "majority_guess": max(labels.values(), default=0),
        "off_pitch": len(off),
        "direction_right": direction_right,
        "in_tune_separation": ranks_higher(
            [row["p_in_tune"] for row in rows if row["label"] == "in_tune"],
            [row["p_in_tune"] for row in off],
        ),
        "confusion": {
            label: dict(Counter(row["choice"] for row in rows if row["label"] == label)) for label in LABELS
        },
        "median_seconds": statistics.median(row["seconds"] for row in rows) if rows else None,
        "input_tokens": sum(tokens),
    }


def print_report(summary: dict[str, Any], title: str) -> None:
    n = summary["clips"]
    if not n:
        print("No clips judged.")
        return
    print(f"\n== {title}: {n} clips, truth from the pitch function {summary['labels']}")
    print(
        f"Flat, in tune or sharp:  {summary['correct']}/{n} right "
        f"(always guessing the commonest answer: {summary['majority_guess']}/{n})"
    )
    if summary["off_pitch"]:
        print(
            f"Direction when off pitch: {summary['direction_right']}/{summary['off_pitch']} right "
            "(a coin flip gets half)"
        )
    if summary["in_tune_separation"] is not None:
        print(
            f"In tune vs off pitch:     {summary['in_tune_separation']:.2f} "
            "(how often an in-tune clip gets the higher p(in tune): 0.5 is chance, 1.0 is perfect)"
        )
    print("Truth -> what the model said:")
    for label in LABELS:
        print(f"  {label:7s} -> {summary['confusion'][label]}")
    print(f"Median answer time {summary['median_seconds']:.1f} s, {summary['input_tokens']} input tokens.")


def report(folder: Path) -> None:
    found = False
    for use_reference in (True, False):
        path = results_file(folder, use_reference)
        if path.exists():
            found = True
            title = "with the reference tone" if use_reference else "note name only, no reference tone"
            print_report(summarize(load_json(path)["rows"]), title)
    if not found:
        sys.exit(f"No results in {folder}. Run `judge` first.")


# ------------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    notes = {"nargs": "+", "choices": list(NOTES), "default": ["mi", "sol"], "help": "Target notes"}

    p = sub.add_parser("synth", help="Make clips with known pitch errors")
    p.add_argument("--notes", **notes)
    p.add_argument("--octave", type=int, default=0, help="-1: the voice sings an octave below the tone")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)

    p = sub.add_parser("record", help="Record yourself singing the notes back")
    p.add_argument("--notes", **notes)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--mic", help="Microphone (default: the system input)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)

    p = sub.add_parser("judge", help="Send the clips to Gemini Decision 1 and print the tally")
    p.add_argument("folder", type=Path)
    p.add_argument("--no-reference", dest="reference", action="store_false", help="Send only the sung note")
    p.add_argument("--limit", type=int, help="Only the first N clips")

    p = sub.add_parser("report", help="Print the tally of an earlier run")
    p.add_argument("folder", type=Path)
    args = parser.parse_args(argv)

    if args.command == "synth":
        folder = new_folder("synth", args.out)
        clips = make_synth(folder, args.notes, octave=args.octave, seed=args.seed)
        print(f"{len(clips)} clips in {folder}")
        print(f"Next: scripts/keys_pitch_judge.py judge {folder}")
    elif args.command == "record":
        folder = new_folder("sung", args.out)
        clips = record(folder, args.notes, args.trials, args.mic, args.seed)
        print(f"\n{len(clips)} clips in {folder}: {dict(Counter(c['label'] for c in clips))}")
        print(f"Next: scripts/keys_pitch_judge.py judge {folder}")
    elif args.command == "judge":
        rows = judge(args.folder, api_key(), args.reference, limit=args.limit)
        title = "with the reference tone" if args.reference else "note name only, no reference tone"
        print_report(summarize(rows), title)
    else:
        report(args.folder)


if __name__ == "__main__":
    main()
