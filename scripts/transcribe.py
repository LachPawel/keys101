"""Speech to text with word timestamps for a session's narration, keeping only the six key numbers.

Runs Whisper locally on the Apple Silicon GPU (mlx-whisper). Reads voice_vf.wav (Voice Focus output from
audio_hooks.py) when present, else voice.wav. Writes transcript.json with every recognized word and the key
numbers among them, all on the session clock, and prints what you said in each episode next to its script.

    uv run --with-requirements scripts/requirements-keys.txt scripts/transcribe.py outputs/keys_sessions/<stamp>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from keys_common import (
    MANIFEST,
    TRANSCRIPT,
    audio_to_session_time,
    load_json,
    session_windows,
    words_to_keys,
    write_json,
)

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"


def key_words(raw_words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Key numbers among the recognized words. A word holding several digits ("1-2") is split evenly."""
    out = []
    for word in raw_words:
        keys = words_to_keys(word["text"])
        step = (word["end"] - word["t"]) / max(len(keys), 1)
        for i, key in enumerate(keys):
            t = word["t"] + i * step
            out.append(
                {
                    "t": round(t, 4),
                    "end": round(t + step, 4),
                    "key": key,
                    "text": word["text"],
                    "prob": word["prob"],
                }
            )
    return out


def transcribe_session(session: Path, model: str, audio_file: str | None) -> dict[str, Any]:
    try:
        import mlx_whisper
    except ImportError:
        sys.exit("mlx-whisper is not installed: run with --with-requirements scripts/requirements-keys.txt")

    manifest = load_json(session / MANIFEST)
    voice_meta = manifest["audio"]["voice"]
    if audio_file is None:
        audio_file = "voice_vf.wav" if (session / "voice_vf.wav").exists() else voice_meta["file"]
    result = mlx_whisper.transcribe(
        str(session / audio_file),
        path_or_hf_repo=model,
        language="en",
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt="one, two, three, four, five, six",
    )
    raw = [
        {
            "t": round(audio_to_session_time(voice_meta, w["start"]), 4),
            "end": round(audio_to_session_time(voice_meta, w["end"]), 4),
            "text": w["word"].strip(),
            "prob": round(float(w.get("probability", 0.0)), 3),
        }
        for segment in result["segments"]
        for w in segment.get("words", [])
    ]
    return {"audio": audio_file, "model": model, "words": key_words(raw), "raw_words": raw}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sessions", type=Path, nargs="+")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--audio", help="Audio file in the session folder (default: voice_vf.wav, else voice.wav)"
    )
    args = parser.parse_args()

    for session in args.sessions:
        transcript = transcribe_session(session, args.model, args.audio)
        write_json(session / TRANSCRIPT, transcript)
        manifest = load_json(session / MANIFEST)
        scripts = {ep["episode_index"]: ep["keys"] for ep in manifest["episodes"]}
        print(f"{session.name}: {len(transcript['words'])} key words from {transcript['audio']}")
        for index, start, end in session_windows(manifest):
            said = [w["key"] for w in transcript["words"] if start <= w["t"] < end]
            label = "whole recording" if index is None else f"episode {index}"
            print(f"  {label}: said   {' '.join(map(str, said))}")
            if index in scripts:
                print(f"  {' ' * len(label)}  script {' '.join(map(str, scripts[index]))}")


if __name__ == "__main__":
    main()
