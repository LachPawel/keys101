#!/usr/bin/env python
"""Build the training variants from one `keys_scripted.py record --say` recording.

  both    The recording as it is: front and ears cameras, tasks "wait" and "press key N".
  sound   The same pictures with one task for every frame ("play the key you hear"): only the spoken
          word in the ears picture says which key.
  text    Without the ears camera: only the task text says which key.

  uv run scripts/keys_variants.py outputs/keys_scripted/<stamp> --repo-id PawelLach/so101_keys_say

Writes <stamp>_both, <stamp>_sound and <stamp>_text next to the recording and prints the push and
training commands. --only-keys 1 3 5 also prints the episodes to leave out for a three-key run.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from keys_common import load_json

SOUND_TASK = "play the key you hear"
EARS = "observation.images.ears"
FRONT_MAP = '"observation.images.front": "observation.images.camera1"'
EARS_MAP = '"observation.images.ears": "observation.images.camera2"'


def build(root: Path, repo_id: str, ears: str = EARS) -> dict[str, Path]:
    from lerobot.datasets.dataset_tools import modify_tasks, remove_feature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out = {name: root.with_name(f"{root.name}_{name}") for name in ("both", "sound", "text")}
    if taken := [str(path) for path in out.values() if path.exists()]:
        raise SystemExit(f"Already there, delete first: {taken}")
    shutil.copytree(root, out["both"])
    shutil.copytree(root, out["sound"])
    modify_tasks(LeRobotDataset(f"{repo_id}_sound", root=out["sound"]), new_task=SOUND_TASK)
    remove_feature(
        LeRobotDataset(repo_id, root=root), ears, output_dir=out["text"], repo_id=f"{repo_id}_text"
    )
    return out


def episodes_without(session: Path, keys: list[int]) -> list[int]:
    """Episodes that press any key outside `keys`, for --dataset.exclude_episodes."""
    episodes = load_json(session / "session.json")["episodes"]
    return [ep["episode_index"] for ep in episodes if not set(ep["keys"]) <= set(keys)]


def commands(out: dict[str, Path], repo_id: str, exclude: list[int] | None) -> str:
    lines = []
    for name, path in out.items():
        repo = f"{repo_id}_{name}"
        lines.append(
            f'uv run python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; '
            f"D('{repo}', root='{path}').push_to_hub(private=True)\""
        )
    lines.append("")
    for name in out:
        rename = FRONT_MAP if name == "text" else f"{FRONT_MAP}, {EARS_MAP}"
        left_out = f" --dataset.exclude_episodes='{exclude}'" if exclude else ""
        lines.append(
            f"lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id={repo_id}_{name}{left_out}"
            " --policy.freeze_vision_encoder=false --policy.train_expert_only=false"
            " --dataset.image_transforms.enable=true --dataset.eval_split=0.1 --eval_steps=1000"
            f" --rename_map='{{{rename}}}' --batch_size=64 --steps=20000 --save_freq=5000"
            f" --policy.repo_id={repo_id.split('/')[0]}/smolvla_keys_{name}"
            " --save_checkpoint_to_hub=true --job.target=a100-large --job.timeout=12h"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=Path, help="The recorded dataset folder, outputs/keys_scripted/<stamp>")
    parser.add_argument("--repo-id", required=True, help="The repo id it was recorded under")
    parser.add_argument("--ears", default=EARS, help="The feature the text variant drops")
    parser.add_argument("--only-keys", type=int, nargs="+", help="Also print the episodes outside these keys")
    args = parser.parse_args()
    out = build(args.root, args.repo_id, args.ears)
    exclude = None
    if args.only_keys:
        exclude = episodes_without(args.root.with_name(f"{args.root.name}_session"), args.only_keys)
    print("\nPush (private), then train. Time the first run before starting the others.\n")
    print(commands(out, args.repo_id, exclude))


if __name__ == "__main__":
    main()
