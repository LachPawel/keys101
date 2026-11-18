"""Find the key presses in a session's room.wav and write presses.json next to it.

Uses the fingerprints in keys.json (from keys_map.py). Times are converted to the session clock.

    uv run scripts/detect_presses.py outputs/keys_sessions/<stamp> [more sessions]
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import key_sounds as ks
import numpy as np
from keys_common import MANIFEST, PRESSES, audio_to_session_time, load_json, read_wav, write_json

# Two onsets of the same key closer than this are one press (a bounce of the rubber tip).
SAME_KEY_MERGE_S = 0.15


def detect_session(session: Path, keys: dict[str, Any], min_sim: float, margin: float) -> dict[str, Any]:
    manifest = load_json(session / MANIFEST)
    room = manifest["audio"]["room"]
    audio, rate = read_wav(session / room["file"])
    templates = {int(k): np.asarray(v["template"]) for k, v in keys["keys"].items()}

    onsets, _ = ks.detect_onsets(audio, rate)
    features = (
        np.stack([ks.note_feature(audio, rate, t) for t in onsets]) if len(onsets) else np.zeros((0, 1))
    )
    results = ks.classify(features, templates, min_sim, margin) if len(onsets) else []

    presses: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for t_file, (key, best, sim, second) in zip(onsets, results, strict=True):
        t = round(audio_to_session_time(room, float(t_file)), 4)
        if key is None:
            rejected.append(
                {"t": t, "closest_key": best, "sim": round(sim, 3), "runner_up": round(second, 3)}
            )
        elif presses and presses[-1]["key"] == key and t - presses[-1]["t"] < SAME_KEY_MERGE_S:
            continue
        else:
            presses.append({"t": t, "key": key, "sim": round(sim, 3), "runner_up": round(second, 3)})
    return {"min_sim": min_sim, "margin": margin, "presses": presses, "rejected": rejected}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sessions", type=Path, nargs="+")
    parser.add_argument("--keys", type=Path, default=Path("keys.json"))
    parser.add_argument(
        "--min-sim", type=float, help="Reject onsets below this similarity (default: keys.json)"
    )
    parser.add_argument("--margin", type=float, default=0.05, help="Required lead over the second-best key")
    args = parser.parse_args()

    keys = load_json(args.keys)
    min_sim = args.min_sim if args.min_sim is not None else keys["min_sim"]
    for session in args.sessions:
        result = detect_session(session, keys, min_sim, args.margin)
        write_json(session / PRESSES, result)
        counts = Counter(p["key"] for p in result["presses"])
        per_key = "  ".join(f"{k}:{counts.get(k, 0)}" for k in sorted({int(k) for k in keys["keys"]}))
        print(
            f"{session.name}: {len(result['presses'])} presses ({per_key}), {len(result['rejected'])} rejected"
        )


if __name__ == "__main__":
    main()
