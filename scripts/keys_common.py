"""Shared pieces of the SO-101 narrated keyboard scripts: key vocabulary, episode scripts, song.txt, WAV I/O
and the session manifest.

Every time stored in a session is in seconds since the session's t0 on `time.perf_counter`, one clock for
the robot frames, the narration and the room microphone.
"""

from __future__ import annotations

import json
import os
import re
import wave
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

KEYS = (1, 2, 3, 4, 5, 6)

# Notes under the stickers, read from the tutorial photo; keys_map.py checks them by ear.
EXPECTED_NOTES = {1: "G#", 2: "C#", 3: "D#", 4: "E", 5: "G#", 6: "A"}

# Spoken words for the sticker numbers, including Whisper's usual mishearings of a lone number.
NUMBER_WORDS = {
    "one": 1,
    "won": 1,
    "two": 2,
    "to": 2,
    "too": 2,
    "three": 3,
    "four": 4,
    "for": 4,
    "five": 5,
    "six": 6,
}

# What to say for each key.
SPOKEN = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
# The same keys as a singing teacher names them, with stickers 1 to 6 on C D E F G A.
SOLFEGE = {1: "do", 2: "re", 3: "mi", 4: "fa", 5: "sol", 6: "la"}

# Every way a key is named out loud: numbers, solfege and letter names, with the usual mishearings.
# Many are ordinary words ("to", "so", "me", "a"), so use it only on what follows "play" or "press".
NOTE_WORDS = {
    **NUMBER_WORDS,
    **{str(key): key for key in KEYS},
    **dict.fromkeys(("do", "doe", "dough", "doh", "c", "see", "sea"), 1),
    **dict.fromkeys(("re", "ray", "rey", "d", "dee"), 2),
    **dict.fromkeys(("mi", "me", "mee", "e"), 3),
    **dict.fromkeys(("fa", "far", "fah", "f", "ef"), 4),
    **dict.fromkeys(("sol", "so", "soul", "sole", "g", "gee"), 5),
    **dict.fromkeys(("la", "lah", "a"), 6),
}

# The task of the frames where the arm holds still before a press (keys_scripted.py record --say).
WAIT_TASK = "wait"

# A + B + C contain every ordered pair of keys exactly once (30 jumps and 6 same-key repeats).
# S is the riff from song.txt plus the loop back to its start.
EPISODE_SCRIPTS = {
    "A": [1, 1, 2, 1, 3, 1, 4, 1, 5, 1, 6, 2, 2],
    "B": [2, 3, 2, 4, 2, 5, 2, 6, 3, 3, 4, 3, 5],
    "C": [5, 3, 6, 4, 4, 5, 4, 6, 5, 5, 6, 6, 1],
    "S": [1, 2, 5, 5, 5, 4, 6, 3, 3, 3, 1, 2, 5],
}

MANIFEST = "manifest.json"
TRANSCRIPT = "transcript.json"
PRESSES = "presses.json"


def task_for_key(key: int) -> str:
    return f"press key {key}"


def words_to_keys(text: str) -> list[int]:
    """Sticker numbers in one transcribed word: "Two," -> [2], "1-2" -> [1, 2], "hello" -> []."""
    token = re.sub(r"[^a-z0-9]", "", text.lower())
    if token in NUMBER_WORDS:
        return [NUMBER_WORDS[token]]
    if token and all(c in "123456" for c in token):
        return [int(c) for c in token]
    return []


def read_song(path: Path) -> list[list[int]]:
    """song.txt: one phrase per line, key numbers 1 to 6 separated by spaces, '#' starts a comment.

    '-' holds the previous key down one more beat and comes back as 0.
    """
    phrases: list[list[int]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        tokens = line.split("#", 1)[0].split()
        if not tokens:
            continue
        if not all(tok in {*(str(k) for k in KEYS), "-"} for tok in tokens):
            raise ValueError(f"{path}:{line_no}: expected key numbers 1 to 6 or '-', got {line.strip()!r}")
        if not phrases and tokens[0] == "-":
            raise ValueError(f"{path}:{line_no}: the song starts with '-', which has no key to hold")
        phrases.append([0 if tok == "-" else int(tok) for tok in tokens])
    return phrases


def read_wav(path: Path) -> tuple[NDArray[np.float32], int]:
    """First channel of a 16-bit PCM WAV as float32 in [-1, 1], and its sample rate."""
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        rate, channels = f.getframerate(), f.getnchannels()
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2")
    return pcm.reshape(-1, channels)[:, 0].astype(np.float32) / 32768.0, rate


def write_wav(path: Path, audio: NDArray[np.floating], rate: int) -> None:
    pcm = (np.clip(audio, -1.0, 32767 / 32768) * 32768).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(pcm.tobytes())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    """Write via a temporary file, so a crash never leaves half a manifest behind."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def audio_to_session_time(audio_meta: dict[str, Any], file_seconds: float) -> float:
    """Session time of a position (in seconds from the start of the file) in a recorded WAV."""
    sample = file_seconds * audio_meta["sample_rate"]
    return float(audio_meta["t_first_sample"] + sample * audio_meta["seconds_per_sample"])


def session_windows(manifest: dict[str, Any]) -> list[tuple[int | None, float, float]]:
    """(episode index, start, end) of every recorded episode. A --no-robot session has one window, index None."""
    if manifest["episodes"]:
        return [(ep["episode_index"], ep["start"], ep["end"]) for ep in manifest["episodes"]]
    window = manifest["no_robot"]
    return [(None, window["start"], window["end"])]
