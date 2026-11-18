#!/usr/bin/env python
"""Check a teleop keyboard dataset by its ears, then build the two training sets.

The ears camera shows the notes straight from MIDI (keys_synth.py --ears), so every episode says which key
really sounded and when. This reads that back and compares it with the episode's task.

    uv run --with-requirements scripts/requirements-keys.txt scripts/keys_teleop_check.py <dataset folder>
    uv run --with-requirements scripts/requirements-keys.txt scripts/keys_teleop_check.py <dataset folder> --build

--build writes two datasets next to the recording, without the episodes that were silent or hit the wrong
key, and prints the push and training commands:

    <folder>_both   camera + ears, task text
    <folder>_text   camera only, task text
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from keys_common import load_json
from keys_synth import Piano, note_name
from numpy.typing import NDArray

EARS = "observation.images.ears"
# Pitch class of the note each task names.
TASK_NOTES = {"do": 0, "re": 2, "mi": 4, "fa": 5, "sol": 7, "la": 9}
NEWEST_COLUMNS = 4  # the right edge of the ears picture: the last 0.1 s
LOUD = 50  # grey level (0 to 255) that counts as sound; the cable is digitally silent otherwise
PROFILE_FRAMES = 12  # frames after the onset that describe the note
NOTE_RANGE = range(36, 97)


@dataclass
class Press:
    episode: int
    task: str
    length: int
    onset: int | None = None  # frame of the first sound
    presses: int = 0
    note: int | None = None

    @property
    def expected(self) -> int | None:
        words = self.task.replace(",", " ").split()
        return TASK_NOTES.get(words[1]) if len(words) > 1 else None

    @property
    def problem(self) -> str:
        if self.onset is None or self.note is None:
            return "silent"
        if self.expected is not None and self.note % 12 != self.expected:
            return f"wrong key ({note_name(self.note)})"
        return ""


def newest(frame: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Loudness per frequency row in the newest slice of one ears frame."""
    grey = frame[..., 0] if frame.ndim == 3 else frame
    return grey[:, -NEWEST_COLUMNS:].astype(np.float32).mean(axis=1)


def ears_config(info: dict) -> object:
    from lerobot_camera_spectrogram.config_spectrogram import SpectrogramCameraConfig

    shape = info["features"][EARS]["shape"]
    cameras = load_json(Path(__file__).with_name("keys_cameras_midi.json"))
    ears = {k: v for k, v in cameras["ears"].items() if k != "type"}
    return SpectrogramCameraConfig(**{**ears, "height": shape[0], "width": shape[1]})


def note_templates(config: object, fps: int) -> dict[int, NDArray[np.float32]]:
    """What each note looks like in the newest slice of the ears picture, computed with the same piano."""
    from lerobot_camera_spectrogram.spectrogram import mel_filterbank, spectrogram_image

    rate, window = config.sample_rate, int(config.sample_rate * config.window_s)  # type: ignore[attr-defined]
    filters = mel_filterbank(rate, config.n_fft, config.n_mels, config.fmin, config.fmax or rate / 2)  # type: ignore[attr-defined]
    step = rate // fps
    templates = {}
    for note in NOTE_RANGE:
        piano = Piano(rate=rate)
        piano.note_on(note, 90)
        audio = np.concatenate([np.zeros(window, dtype=np.float32), piano.render(step * PROFILE_FRAMES)])
        slices = [
            newest(spectrogram_image(audio[k * step : k * step + window], config, filters))
            for k in range(1, PROFILE_FRAMES + 1)
        ]
        templates[note] = np.max(slices, axis=0)
    return templates


def best_note(profile: NDArray[np.float32], templates: dict[int, NDArray[np.float32]]) -> int:
    def unit(x: NDArray[np.float32]) -> NDArray[np.float32]:
        x = x - x.mean()
        return x / (np.linalg.norm(x) + 1e-9)

    p = unit(profile)
    return max(templates, key=lambda note: float(p @ unit(templates[note])))


def read_presses(root: Path) -> tuple[list[Press], dict]:
    import av
    import pandas as pd

    info = load_json(root / "meta" / "info.json")
    fps = info["fps"]
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    tables = []
    for file in files:
        try:
            tables.append(pd.read_parquet(file))
        except Exception:  # noqa: BLE001  the recorder is still writing this one
            print(f"Skipping {file.name}: still being written, run again when the recording has stopped.")
    episodes = pd.concat(tables).sort_values("episode_index")
    templates = note_templates(ears_config(info), fps)

    slices: dict[tuple[int, int], NDArray[np.float32]] = {}
    presses = []
    for ep in episodes.to_dict("records"):
        key = (int(ep[f"videos/{EARS}/chunk_index"]), int(ep[f"videos/{EARS}/file_index"]))
        if key not in slices:
            video = root / "videos" / EARS / f"chunk-{key[0]:03d}" / f"file-{key[1]:03d}.mp4"
            with av.open(str(video)) as container:
                slices[key] = np.stack(
                    [newest(f.to_ndarray(format="gray")) for f in container.decode(video=0)]
                )
        start = round(float(ep[f"videos/{EARS}/from_timestamp"]) * fps)
        sound = slices[key][start : start + int(ep["length"])]
        press = Press(int(ep["episode_index"]), str(list(ep["tasks"])[0]), int(ep["length"]))
        loud = sound.max(axis=1) > LOUD
        if loud.any():
            press.onset = int(np.argmax(loud))
            press.presses = int(np.sum(loud[1:] & ~loud[:-1]) + loud[0])
            profile = sound[press.onset : press.onset + PROFILE_FRAMES].max(axis=0)
            press.note = best_note(profile, templates)
        presses.append(press)
    return presses, info


def report(presses: list[Press], fps: int) -> list[int]:
    bad = [p.episode for p in presses if p.problem]
    print(f"{len(presses)} episodes, {len(bad)} to drop.\n")
    for task in dict.fromkeys(p.task for p in presses):
        group = [p for p in presses if p.task == task]
        good = [p for p in group if not p.problem]
        notes = Counter(note_name(p.note) for p in group if p.note is not None)
        onsets = [p.onset / fps for p in good if p.onset is not None]
        timing = f", note at {np.median(onsets):.1f} s (median)" if onsets else ""
        print(f"{task!r}: {len(good)} good of {len(group)}, heard {dict(notes)}{timing}")
        for p in group:
            if p.problem:
                print(f"    episode {p.episode}: {p.problem}")
    return bad


def build(root: Path, repo_id: str, bad: list[int], tag: str = "") -> dict[str, Path]:
    from lerobot.datasets.dataset_tools import delete_episodes, remove_feature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    names = {name: f"{tag}_{name}" if tag else name for name in ("both", "text")}
    out = {name: root.with_name(f"{root.name}_{suffix}") for name, suffix in names.items()}
    if taken := [str(path) for path in out.values() if path.exists()]:
        raise SystemExit(f"Already there, delete first: {taken}")
    source = LeRobotDataset(repo_id, root=root)
    if bad:
        delete_episodes(source, bad, output_dir=out["both"], repo_id=f"{repo_id}_{names['both']}")
    else:
        shutil.copytree(root, out["both"])
    both = LeRobotDataset(f"{repo_id}_{names['both']}", root=out["both"])
    remove_feature(both, EARS, output_dir=out["text"], repo_id=f"{repo_id}_{names['text']}")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=Path, help="The recorded dataset folder")
    parser.add_argument("--repo-id", help="Default: PawelLach/<folder name>")
    parser.add_argument("--build", action="store_true", help="Write the cleaned _both and _text datasets")
    parser.add_argument("--tag", default="", help="Name for a second build after adding episodes, e.g. v2")
    args = parser.parse_args(argv)
    root = args.root.expanduser()
    repo_id = args.repo_id or f"PawelLach/{root.name}"

    presses, info = read_presses(root)
    bad = report(presses, info["fps"])
    if not args.build:
        return
    from keys_variants import commands

    out = build(root, repo_id, bad, args.tag)
    if args.tag:
        repo_id = f"{repo_id}_{args.tag}"
    print(f"\nWrote {out['both']} and {out['text']}.\nPush (private), then train:\n")
    print(commands(out, repo_id, None))


if __name__ == "__main__":
    main()
