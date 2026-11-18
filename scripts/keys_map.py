"""Record a sound fingerprint for each sticker key and write keys.json.

The keyboard plays through the other MacBook's speakers, so this Mac knows the keys only by ear. Press each
key by hand a few times when prompted; the MacBook mic (the same one the `ears` camera uses) hears the notes,
the script finds the presses, averages each key's spectrum into a fingerprint and estimates the note so you
can check the sticker table.

    uv run --with-requirements scripts/requirements-keys.txt scripts/keys_map.py
    uv run scripts/keys_map.py --wav calibration.wav   # offline: keys 1..6 in order, --presses each
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import key_sounds as ks
import numpy as np
from keys_common import EXPECTED_NOTES, KEYS, read_wav, write_json, write_wav
from numpy.typing import NDArray

RATE = 16000


def record_key(mic: str, key: int, presses: int, seconds: float) -> NDArray[np.float32]:
    import sounddevice as sd

    input(f"\nKey {key} ({EXPECTED_NOTES[key]}): press Enter, then press it {presses} times, ~1 s apart.")
    audio = sd.rec(round(seconds * RATE), samplerate=RATE, channels=1, dtype="float32", device=mic)
    sd.wait()
    return np.asarray(audio[:, 0])


def strongest_onsets(audio: NDArray[np.float32], count: int) -> NDArray[np.float64]:
    times, strength = ks.detect_onsets(audio, RATE)
    return np.sort(times[np.argsort(strength)[::-1][:count]])


def build_keys(takes: dict[int, tuple[NDArray[np.float32], NDArray[np.float64]]]) -> dict:
    """Fingerprint, heard note and self-check for every key, from (audio, onset times) per key."""
    features = {
        k: np.stack([ks.note_feature(audio, RATE, t) for t in times]) for k, (audio, times) in takes.items()
    }
    templates = {k: f.mean(axis=0) / np.linalg.norm(f.mean(axis=0)) for k, f in features.items()}

    keys: dict[str, dict] = {}
    worst_own, worst_other = 1.0, 0.0
    for key, (audio, times) in takes.items():
        notes = [ks.estimate_midi(audio, RATE, t) for t in times]
        midi = max(set(notes), key=notes.count)
        results = ks.classify(features[key], templates, min_sim=0.0, margin=0.0)
        own = min(r[2] for r in results)
        other = max(
            float(features[key][i] @ templates[k]) for i in range(len(times)) for k in templates if k != key
        )
        worst_own, worst_other = min(worst_own, own), max(worst_other, other)
        keys[str(key)] = {
            "note": ks.note_name(midi),
            "midi": midi,
            "expected": EXPECTED_NOTES[key],
            "presses": len(times),
            "self_similarity": round(own, 3),
            "template": [round(float(v), 5) for v in templates[key]],
        }
        flag = (
            ""
            if ks.note_name(midi).rstrip("0123456789") == EXPECTED_NOTES[key]
            else "  <- not the expected note"
        )
        confused = [r[1] for r in results if r[1] != key]
        if confused:
            flag += f"  <- some presses look like key {confused[0]}"
        print(
            f"key {key}: heard {ks.note_name(midi):4s} expected {EXPECTED_NOTES[key]:3s} self-match {own:.2f}{flag}"
        )

    # Not derived from the calibration: there the keys are clean, while in a session your voice often
    # overlaps a press and pulls its similarity well below 1. The runner-up margin keeps keys apart.
    min_sim = ks.DEFAULT_MIN_SIM
    print(f"\nWorst self-match {worst_own:.2f}, best wrong-key match {worst_other:.2f}, min_sim {min_sim}")
    if worst_own - worst_other < 0.2:
        print("Some keys sound alike. Re-run with more presses, or use a brighter synth patch.")
    return {
        "sample_rate": RATE,
        "feature": {
            "f_min": ks.F_MIN,
            "f_max": ks.F_MAX,
            "bins_per_semitone": ks.BINS_PER_SEMITONE,
            "win_s": ks.FEATURE_WIN_S,
        },
        "min_sim": min_sim,
        "keys": keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mic", default="MacBook Pro Microphone", help="Input device name, or part of it")
    parser.add_argument("--presses", type=int, default=3, help="Presses per key")
    parser.add_argument("--seconds", type=float, default=5.0, help="Recording length per key")
    parser.add_argument("--wav", type=Path, help="Use one recording of keys 1..6 in order instead of the mic")
    parser.add_argument("--save-wav", type=Path, help="Also save the live calibration audio here")
    parser.add_argument("--out", type=Path, default=Path("keys.json"))
    args = parser.parse_args()

    takes: dict[int, tuple[NDArray[np.float32], NDArray[np.float64]]] = {}
    if args.wav:
        audio, rate = read_wav(args.wav)
        if rate != RATE:
            sys.exit(f"{args.wav}: expected {RATE} Hz, got {rate}")
        times = strongest_onsets(audio, args.presses * len(KEYS))
        if len(times) < args.presses * len(KEYS):
            sys.exit(f"Heard {len(times)} presses, expected {args.presses * len(KEYS)}.")
        for i, key in enumerate(KEYS):
            takes[key] = (audio, times[i * args.presses : (i + 1) * args.presses])
    else:
        print(f"Listening on {args.mic!r}. Keep quiet and keep the arm still while recording.")
        for key in KEYS:
            while True:
                audio = record_key(args.mic, key, args.presses, args.seconds)
                times = strongest_onsets(audio, args.presses)
                if len(times) == args.presses:
                    break
                print(f"Heard {len(times)} presses, expected {args.presses}. Again.")
            takes[key] = (audio, times)
        if args.save_wav:
            write_wav(args.save_wav, np.concatenate([takes[k][0] for k in KEYS]), RATE)

    result = build_keys(takes)
    result["mic"] = args.mic
    write_json(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
