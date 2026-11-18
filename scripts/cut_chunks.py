"""Match spoken key numbers to the key presses heard in the room, report how well they agree, and cut each
episode into one chunk per press, written as a new dataset where every chunk is an episode "press key N".

Matching: each spoken number goes to the first press after it (allowing --lookback for Whisper's late word
starts) within --max-gap. If that press is the spoken key the pair is a match; if it is another key the word
is a mislabel; if nothing is heard in time the word has no press. Presses no word matched are listed too.

Cuts: each press ends a chunk at its time + --hold; the chunk starts where the previous one ended (the
episode start for the first). `--cut-source notes` uses the heard press times and keys; `--cut-source voice`
uses the spoken numbers, shifted by the median voice-to-press gap.

    uv run scripts/cut_chunks.py outputs/keys_sessions/<stamp> [more sessions] --dry-run
    uv run scripts/cut_chunks.py outputs/keys_sessions/<stamp> --cut-source notes
"""

from __future__ import annotations

import argparse
import bisect
import difflib
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from keys_common import MANIFEST, PRESSES, TRANSCRIPT, load_json, session_windows, task_for_key


@dataclass(frozen=True)
class Event:
    t: float
    key: int


@dataclass(frozen=True)
class Match:
    word: Event
    press: Event | None
    status: str  # "ok", "mislabel" or "no_press"


@dataclass(frozen=True)
class Chunk:
    session: str
    episode: int
    key: int
    start: float
    end: float
    first_frame: int
    end_frame: int
    confirmed: bool


def match_words(
    words: list[Event], presses: list[Event], max_gap: float, lookback: float
) -> tuple[list[Match], list[Event]]:
    """Pair every spoken number with the next unused press; also return presses no word claimed."""
    presses = sorted(presses, key=lambda p: p.t)
    times = [p.t for p in presses]
    used = [False] * len(presses)
    matches = []
    for word in sorted(words, key=lambda w: w.t):
        i = bisect.bisect_left(times, word.t - lookback)
        while i < len(presses) and used[i]:
            i += 1
        if i == len(presses) or presses[i].t > word.t + max_gap:
            matches.append(Match(word, None, "no_press"))
        elif presses[i].key == word.key:
            used[i] = True
            matches.append(Match(word, presses[i], "ok"))
        else:
            matches.append(Match(word, presses[i], "mislabel"))
    return matches, [p for p, u in zip(presses, used, strict=True) if not u]


def report(name: str, matches: list[Match], unlabeled: list[Event]) -> None:
    ok = [m for m in matches if m.status == "ok"]
    mislabels = [m for m in matches if m.status == "mislabel"]
    no_press = [m for m in matches if m.status == "no_press"]
    rate = len(mislabels) / max(len(ok) + len(mislabels), 1)
    print(
        f"{name}: {len(matches)} spoken, {len(ok)} matched, {len(mislabels)} mislabeled ({rate:.1%}), "
        f"{len(no_press)} with no press, {len(unlabeled)} presses without a word"
    )
    gaps = [m.press.t - m.word.t for m in ok if m.press is not None]
    if gaps:
        print(
            f"  gap press - voice: median {statistics.median(gaps):.3f} s, mean {statistics.mean(gaps):.3f} s, "
            f"min {min(gaps):.3f}, max {max(gaps):.3f} (n={len(gaps)})"
        )
    for m in mislabels:
        assert m.press is not None
        print(
            f"  mislabel at {m.word.t:8.2f} s: said {m.word.key}, pressed {m.press.key} {m.press.t - m.word.t:+.2f} s later"
        )


def plan_episode(
    events: list[tuple[Event, bool]], start: float, end: float, hold: float, min_len: float
) -> tuple[list[tuple[float, float, int, bool]], list[tuple[float, float, int]]]:
    """Contiguous chunks ending `hold` after each event; returns (kept, too short)."""
    kept, dropped = [], []
    prev = start
    for event, confirmed in sorted(events, key=lambda e: e[0].t):
        if not start <= event.t < end:
            continue
        stop = min(event.t + hold, end)
        if stop - prev >= min_len:
            kept.append((prev, stop, event.key, confirmed))
        else:
            dropped.append((prev, stop, event.key))
        prev = stop
    return kept, dropped


def plan_words(
    words: list[Event],
    script: list[int],
    start: float,
    end: float,
    lead: float,
    last_len: float,
    min_len: float,
    max_len: float,
) -> tuple[list[tuple[float, float, int, bool]], list[tuple[float, float, int]]]:
    """One chunk per spoken number, from just before it to just before the next one.

    You say a number, move, press, then wait for the next prompt, so each press lies between its word and
    the next. Confirmed means the word agrees with the episode's script. Chunks outside [min_len, max_len]
    are dropped: a long one means Whisper missed a word in between, so it would hold several presses.
    """
    episode_words = sorted((w for w in words if start <= w.t < end), key=lambda w: w.t)
    matcher = difflib.SequenceMatcher(None, [w.key for w in episode_words], script, autojunk=False)
    agreed = {b.a + i for b in matcher.get_matching_blocks() for i in range(b.size)}
    kept, dropped = [], []
    for i, word in enumerate(episode_words):
        chunk_start = max(word.t - lead, start)
        if i + 1 < len(episode_words):
            chunk_end = episode_words[i + 1].t - lead
        else:
            chunk_end = min(word.t + last_len, end)
        if min_len <= chunk_end - chunk_start <= max_len:
            kept.append((chunk_start, chunk_end, word.key, i in agreed))
        else:
            dropped.append((chunk_start, chunk_end, word.key))
    return kept, dropped


def load_events(session: Path) -> tuple[dict[str, Any], list[Event], list[Event]]:
    for name, script in ((TRANSCRIPT, "transcribe.py"), (PRESSES, "detect_presses.py")):
        if not (session / name).exists():
            sys.exit(f"{session}: no {name}; run scripts/{script} on it first")
    manifest = load_json(session / MANIFEST)
    words = [Event(w["t"], w["key"]) for w in load_json(session / TRANSCRIPT)["words"]]
    presses = [Event(p["t"], p["key"]) for p in load_json(session / PRESSES)["presses"]]
    return manifest, words, presses


def in_windows(events: list[Event], windows: list[tuple[int | None, float, float]]) -> list[Event]:
    """Only events inside a recorded episode count; outside them you may talk freely."""
    return [e for e in events if any(start <= e.t < end for _, start, end in windows)]


def user_features(features: dict[str, Any]) -> dict[str, Any]:
    from lerobot.utils.constants import DEFAULT_FEATURES

    return {
        k: {"dtype": v["dtype"], "shape": tuple(v["shape"]), "names": v["names"]}
        for k, v in features.items()
        if k not in DEFAULT_FEATURES
    }


def to_frame_value(value: Any, feature: dict[str, Any]) -> np.ndarray:
    array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    if feature["dtype"] in ("image", "video") and array.dtype != np.uint8:
        # The reader returns channel-first float images in [0, 1].
        array = (np.clip(array.transpose(1, 2, 0), 0, 1) * 255).round().astype(np.uint8)
    return array


def write_chunks(
    chunks: list[Chunk], sessions: dict[str, dict[str, Any]], repo_id: str, root: Path | None
) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dst = None
    features: dict[str, Any] = {}
    for name in dict.fromkeys(c.session for c in chunks):
        info = sessions[name]["dataset"]
        src = LeRobotDataset(info["repo_id"], root=info["root"])
        src_features = user_features(src.meta.features)
        if dst is None:
            features = src_features
            dst = LeRobotDataset.create(
                repo_id, fps=src.fps, features=features, root=root, robot_type=src.meta.robot_type
            )
        elif src_features != features:
            sys.exit(f"Session {name} has different features than the first session; cut them separately")
        for chunk in (c for c in chunks if c.session == name):
            base = src.meta.episodes[chunk.episode]["dataset_from_index"]
            for frame_index in range(chunk.first_frame, chunk.end_frame):
                item = src[base + frame_index]
                frame = {k: to_frame_value(item[k], f) for k, f in features.items()}
                dst.add_frame({**frame, "task": task_for_key(chunk.key)})
            dst.save_episode()
    assert dst is not None
    dst.finalize()
    return Path(dst.root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sessions", type=Path, nargs="+")
    parser.add_argument("--dry-run", action="store_true", help="Print the report and planned cuts only")
    parser.add_argument("--cut-source", choices=("notes", "voice", "words"), default="notes")
    parser.add_argument(
        "--max-gap", type=float, default=3.0, help="Longest wait from a word to its press (s)"
    )
    parser.add_argument("--lookback", type=float, default=0.3, help="A press may come this early (s)")
    parser.add_argument("--hold", type=float, default=0.3, help="A chunk ends this long after its press (s)")
    parser.add_argument("--min-len", type=float, default=0.5, help="Drop chunks shorter than this (s)")
    parser.add_argument("--max-len", type=float, default=8.5, help="words: drop chunks longer than this (s)")
    parser.add_argument(
        "--lead", type=float, default=0.3, help="words: a chunk starts this long before its word"
    )
    parser.add_argument(
        "--last-len", type=float, default=6.0, help="words: length of an episode's last chunk"
    )
    parser.add_argument(
        "--confirmed-only", action="store_true", help="Write only chunks voice and sound agree on"
    )
    parser.add_argument("--repo-id", default="PawelLach/so101_keys_chunks")
    parser.add_argument("--root", type=Path, help="Output folder (default: HF_LEROBOT_HOME/<repo-id>)")
    args = parser.parse_args()

    loaded = {s.name: (s, *load_events(s)) for s in args.sessions}
    all_matches: list[Match] = []
    per_session: dict[str, tuple[list[Match], list[Event]]] = {}
    for name, (_, manifest, words, presses) in loaded.items():
        windows = session_windows(manifest)
        matches, unlabeled = match_words(
            in_windows(words, windows), in_windows(presses, windows), args.max_gap, args.lookback
        )
        per_session[name] = (matches, unlabeled)
        all_matches += matches
        report(name, matches, unlabeled)
    if len(loaded) > 1:
        report("all sessions", all_matches, [p for _, u in per_session.values() for p in u])

    gaps = [m.press.t - m.word.t for m in all_matches if m.status == "ok" and m.press is not None]
    median_gap = statistics.median(gaps) if gaps else None
    if args.cut_source == "voice" and median_gap is None:
        sys.exit("No matched words, so there is no median gap to shift the voice cuts by")

    chunks: list[Chunk] = []
    print(f"\nPlanned cuts (source: {args.cut_source}, hold {args.hold} s)")
    print(
        f"{'session':16s} {'ep':>3s} {'#':>3s} {'key':>4s} {'start':>7s} {'end':>7s} {'frames':>11s}  confirmed"
    )
    for name, (_, manifest, words, presses) in loaded.items():
        matches, _ = per_session[name]
        confirmed_presses = {m.press for m in matches if m.status == "ok"}
        if args.cut_source == "notes":
            events = [(p, p in confirmed_presses) for p in presses]
        elif args.cut_source == "voice":
            assert median_gap is not None
            events = [(Event(m.word.t + median_gap, m.word.key), m.status == "ok") for m in matches]
        for episode in manifest["episodes"]:
            frame_t = np.asarray(episode["frame_t"])
            if args.cut_source == "words":
                kept, dropped = plan_words(
                    words,
                    episode["keys"],
                    episode["start"],
                    episode["end"],
                    args.lead,
                    args.last_len,
                    args.min_len,
                    args.max_len,
                )
            else:
                kept, dropped = plan_episode(
                    events, episode["start"], episode["end"], args.hold, args.min_len
                )
            for i, (start, end, key, confirmed) in enumerate(kept):
                f0, f1 = int(np.searchsorted(frame_t, start)), int(np.searchsorted(frame_t, end))
                chunk = Chunk(name, episode["episode_index"], key, start, end, f0, f1, confirmed)
                rel0, rel1 = start - episode["start"], end - episode["start"]
                skip = "" if confirmed or not args.confirmed_only else "  (skipped)"
                print(
                    f"{name:16s} {chunk.episode:3d} {i:3d} {key:4d} {rel0:7.2f} {rel1:7.2f} "
                    f"{f'{f0}-{f1 - 1}':>11s}  {'yes' if confirmed else 'no'}{skip}"
                )
                if confirmed or not args.confirmed_only:
                    chunks.append(chunk)
            for start, end, key in dropped:
                print(
                    f"{name:16s} {episode['episode_index']:3d}   - {key:4d}  dropped: {end - start:.2f} s is outside the length limits"
                )
        if not manifest["episodes"]:
            print(f"{name:16s} recorded without the robot: nothing to cut")

    frames = sum(c.end_frame - c.first_frame for c in chunks)
    print(
        f"\n{len(chunks)} chunks, {frames} frames"
        + (f", median gap {median_gap:.3f} s" if median_gap else "")
    )
    if args.dry_run or not chunks:
        return
    sessions = {name: manifest for name, (_, manifest, _, _) in loaded.items()}
    out = write_chunks(chunks, sessions, args.repo_id, args.root)
    print(f"Wrote {args.repo_id} to {out}")


if __name__ == "__main__":
    main()
