#!/usr/bin/env python
"""Merge the note-by-note recording sessions into one training set.

    uv run --with-requirements scripts/requirements-keys.txt scripts/keys_flow_build.py

Takes every session under outputs/keys_flow that finished with at least --min-episodes episodes, and writes
the merged dataset to $HF_LEROBOT_HOME/PawelLach/so101_keys_flow_<n>ep.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from keys_common import load_json

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sessions", type=Path, default=ROOT / "outputs" / "keys_flow")
    parser.add_argument("--min-episodes", type=int, default=100, help="Leaves out short test sessions")
    parser.add_argument("--user", default="PawelLach")
    args = parser.parse_args()

    from lerobot.datasets.dataset_tools import merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    sessions = []
    for folder in sorted(args.sessions.iterdir()):
        info = folder / "session.json"
        if info.exists() and load_json(info)["episodes"] >= args.min_episodes:
            sessions.append(folder)
    if not sessions:
        raise SystemExit("No finished session with enough episodes.")
    total = sum(load_json(s / "session.json")["episodes"] for s in sessions)
    repo_id = f"{args.user}/so101_keys_flow_{total}ep"
    out = HOME / repo_id
    if out.exists():
        shutil.rmtree(out)
    datasets = [LeRobotDataset(f"{args.user}/so101_keys_flow", root=s / "dataset") for s in sessions]
    merged = merge_datasets(datasets, output_repo_id=repo_id, output_dir=out) if len(datasets) > 1 else None
    if merged is None:
        shutil.copytree(sessions[0] / "dataset", out)
        merged = LeRobotDataset(repo_id, root=out)
    print(f"MERGED {[s.name for s in sessions]} -> {repo_id}")
    print(f"MERGED {merged.num_episodes} episodes, {merged.num_frames} frames, at {out}")


if __name__ == "__main__":
    main()
