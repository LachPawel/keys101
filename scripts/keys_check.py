#!/usr/bin/env python
"""Checks for the scripted keyboard recording.

listen   No robot. Press the keys by hand: prints each MIDI note and how loud the microphone heard it,
         and writes ears_check.png, the picture the ears camera would show. Ctrl+C to stop.
session  After a recording: frame spacing, presses that sounded their key, MIDI against the frame clock,
         how long repeated notes take against the ears window, how loud the notes are in room.wav, and
         whether the joints give away how many times a key was already pressed.

uv run --with-requirements scripts/requirements-keys.txt scripts/keys_check.py listen
uv run --with-requirements scripts/requirements-keys.txt scripts/keys_check.py session outputs/keys_scripted/<stamp>_session
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from keys_common import load_json, read_wav

DEFAULT_CAMERAS = Path(__file__).with_name("keys_cameras.json")
NOTE_S = 0.3  # how much of a note is measured, from its note-on
LOUD_DBFS, ABOVE_ROOM_DB = -45.0, 15.0  # a note the ears camera shows clearly


def dbfs(audio: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def verdict(note_db: float, room_db: float) -> str:
    ok = note_db >= LOUD_DBFS and note_db - room_db >= ABOVE_ROOM_DB
    return "ok" if ok else "TOO QUIET: move the speaker closer or turn it up"


def ears_config(cameras: dict[str, dict]) -> dict[str, Any]:
    return next((cam for cam in cameras.values() if cam.get("type") == "spectrogram"), {})


def listen(args: argparse.Namespace) -> None:
    import sounddevice as sd
    from keys_scripted import open_midi
    from lerobot_camera_spectrogram.config_spectrogram import SpectrogramCameraConfig
    from lerobot_camera_spectrogram.spectrogram import mel_filterbank, spectrogram_image
    from PIL import Image

    ears = {k: v for k, v in ears_config(load_json(args.cameras)).items() if k != "type"}
    config = SpectrogramCameraConfig(**ears)
    rate, window = config.sample_rate, int(config.sample_rate * config.window_s)
    filters = mel_filterbank(rate, config.n_fft, config.n_mels, config.fmin, config.fmax or rate / 2)
    ring = np.zeros(max(window, 3 * rate), dtype=np.float32)
    end_time = [time.perf_counter()]

    def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        ring[:] = np.roll(ring, -frames)
        ring[-frames:] = indata[:, 0]
        end_time[0] = time.perf_counter()

    t0 = time.perf_counter()
    midi = open_midi(args.midi, t0)
    seen = 0
    with sd.InputStream(
        device=config.device, channels=1, samplerate=rate, dtype="float32", callback=callback
    ):
        print(f"Microphone: {config.device}, ears window {config.window_s} s. Press the keys. Ctrl+C stops.")
        try:
            while True:
                time.sleep(0.05)
                events = [e for e in (midi.events if midi else []) if e[2]]
                if len(events) == seen or time.perf_counter() - t0 - events[-1][0] < NOTE_S + 0.15:
                    continue
                for t, note, _ in events[seen:]:
                    start = len(ring) - int((end_time[0] - t0 - t) * rate)
                    note_db = dbfs(ring[max(start, 0) : start + int(NOTE_S * rate)])
                    room_db = dbfs(ring[max(start - int(0.5 * rate), 0) : max(start - int(0.1 * rate), 1)])
                    print(
                        f"note {note}: {note_db:.0f} dBFS, room {room_db:.0f} dBFS  {verdict(note_db, room_db)}"
                    )
                seen = len(events)
                Image.fromarray(spectrogram_image(ring[-window:], config, filters)).resize((512, 512)).save(
                    args.png
                )
                print(f"  wrote {args.png} (time left to right, newest on the right)")
        except KeyboardInterrupt:
            pass
    if midi:
        midi.close()


def session(args: argparse.Namespace) -> None:
    data = load_json(args.folder / "session.json")
    episodes = data["episodes"]
    presses = [(ep, p) for ep in episodes for p in ep["presses"]]
    print(f"{len(episodes)} episodes, {len(presses)} presses, mode {data['mode']}")

    gaps = np.concatenate([np.diff(ep["frame_times"]) for ep in episodes]) * 1000
    target = 1000 / data["fps"]
    print(
        f"Frames: median {np.median(gaps):.1f} ms, 99% under {np.percentile(gaps, 99):.1f} ms, "
        f"worst {gaps.max():.0f} ms (target {target:.1f})  {'ok' if abs(np.median(gaps) - target) < 2 else 'SLOW LOOP'}"
    )

    judged = [p["ok"] for _, p in presses if p["ok"] is not None]
    if judged:
        print(f"Presses that sounded their key: {sum(judged)}/{len(judged)}")
        for ep, p in presses:
            if p["ok"] is False:
                print(f"  episode {ep['episode_index']} key {p['key']}: heard {p['notes']}")
    notes = [json.loads(line) for line in (args.folder / "midi.jsonl").read_text().splitlines()]
    ons = [n for n in notes if n["velocity"]]
    if not ons:
        print("MIDI: no notes logged. Nothing else can be checked.")
        return

    def first_on(p: dict[str, Any]) -> float | None:
        return next((n["t"] for n in ons if p["start"] <= n["t"] < p["end"]), None)

    late = [
        (t - ep["frame_times"][min(p["frame_down"], len(ep["frame_times"]) - 1)]) * 1000
        for ep, p in presses
        if (t := first_on(p)) is not None
    ]
    print(
        f"Note-on against the frame where the arm is down: median {np.median(late):+.0f} ms, "
        f"from {min(late):+.0f} to {max(late):+.0f} ms  {'ok' if max(abs(x) for x in late) < 400 else 'CLOCKS DISAGREE'}"
    )

    periods = [
        b - a
        for ep in episodes
        for p, q in zip(ep["presses"], ep["presses"][1:], strict=False)
        if p["key"] == q["key"] and (a := first_on(p)) is not None and (b := first_on(q)) is not None
    ]
    window = ears_config(data["cameras"]).get("window_s", 1.0)
    if periods:
        # At the decision after the third press, the first note is two periods plus one lift and settle old.
        need = 2 * max(periods) + 1.0
        print(
            f"Repeated notes: {np.median(periods):.2f} s apart (longest {max(periods):.2f}). Three of them "
            f"need {need:.1f} s of sound, the ears show {window} s  {'ok' if window >= need else 'RAISE window_s'}"
        )

    audio_meta = data.get("audio", {}).get("room")
    if audio_meta and (args.folder / "room.wav").exists():
        audio, rate = read_wav(args.folder / "room.wav")

        def cut(t: float, seconds: float) -> np.ndarray:
            i = int((t - audio_meta["t_first_sample"]) / audio_meta["seconds_per_sample"])
            return audio[max(i, 0) : max(i + int(seconds * rate), 1)]

        note_db = float(np.median([dbfs(cut(n["t"], NOTE_S)) for n in ons]))
        room_db = float(np.median([dbfs(cut(n["t"] - 0.5, 0.4)) for n in ons]))
        print(
            f"room.wav: notes {note_db:.0f} dBFS, between notes {room_db:.0f} dBFS  {verdict(note_db, room_db)}"
        )
    else:
        print("room.wav: no clock fit yet (it is written when the recording ends).")

    leak_check(Path(data["dataset_root"]), episodes)


def leak_check(root: Path, episodes: list[dict[str, Any]]) -> None:
    """Joint readings at the last frame of each press, grouped by how many times the key was pressed so far."""
    import pandas as pd

    files = sorted(root.glob("data/**/*.parquet"))
    if not files:
        print(f"Leak check: no data under {root} yet (written when the recording ends).")
        return
    frames = pd.concat(
        pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]) for f in files
    )
    states = {(int(e), int(f)): np.asarray(s) for e, f, s in frames.itertuples(index=False)}
    by_count: dict[tuple[int, int], list[np.ndarray]] = {}
    for ep in episodes:
        count = 0
        for i, p in enumerate(ep["presses"]):
            count = count + 1 if i and ep["presses"][i - 1]["key"] == p["key"] else 1
            if (state := states.get((ep["episode_index"], p["frame_end"] - 1))) is not None:
                by_count.setdefault((p["key"], count), []).append(state)
    for key in sorted({k for k, c in by_count if c > 1}):
        groups = {c: np.stack(v) for (k, c), v in by_count.items() if k == key}
        means = np.stack([g.mean(axis=0) for g in groups.values()])
        apart = float(np.ptp(means, axis=0).max())
        spread = float(max(g.std(axis=0).max() for g in groups.values()))
        fewest = min(len(g) for g in groups.values())
        sizes = ", ".join(f"press {c}: {len(g)}" for c, g in sorted(groups.items()))
        if fewest < 2:
            flag = "too few presses to tell"
        else:  # the means of `fewest` jittered poses still differ by chance, by about spread / sqrt(fewest)
            flag = "ok" if apart <= max(0.5, 3 * spread / fewest**0.5) else "LEAK: the pose tells the count"
        print(
            f"Key {key} ({sizes}): joints differ by {apart:.2f} deg between counts, {spread:.2f} deg within  {flag}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("listen", help="Press keys by hand: MIDI notes, microphone level, ears picture")
    p.add_argument("--cameras", type=Path, default=DEFAULT_CAMERAS)
    p.add_argument("--midi", default="auto")
    p.add_argument("--png", type=Path, default=Path("ears_check.png"))
    p.set_defaults(run=listen)
    p = sub.add_parser("session", help="Check a recorded session folder")
    p.add_argument("folder", type=Path)
    p.set_defaults(run=session)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
