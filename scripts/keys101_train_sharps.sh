#!/bin/zsh
# The black keys join the white ones: one more SmolVLA training on every key the arm was shown.
#
#   scripts/keys101_train_sharps.sh 20261122_121000          # the sharps recording's folder name
#   DRY=1 scripts/keys101_train_sharps.sh 20261122_121000    # build the dataset only: no upload, no training
#
# What it does:
#   1. Drops the sharps episodes whose press did not sound its own key.
#   2. Merges them with the cleaned white-key recording (KEYS101_WHITE, made by keys101_train.sh).
#   3. Pushes the merged dataset to your Hugging Face account, private.
#   4. Starts one paid cloud job on a100-large (camera and ears) and returns at once.
#
# Cost: 6,000 steps, about 2 hours, about $5. The 3 hour limit caps it at $7.50.
set -e
set -o pipefail
cd "$(dirname "$0")/.."

STAMP=${1:?Give the sharps recording folder name, e.g. 20261122_121000}
WHITE=${KEYS101_WHITE:-20261122_103621}
DATA=${KEYS101_DATA:-PawelLach/so101-keys101-22keys}
MODEL=${KEYS101_MODEL:-PawelLach/smolvla-keys101-22keys}
STEPS=${KEYS101_STEPS:-6000}
[ -d "outputs/keys_scripted/$STAMP" ] || { echo "No outputs/keys_scripted/$STAMP"; exit 1; }
[ -d "outputs/keys_scripted/${WHITE}_clean" ] || { echo "No outputs/keys_scripted/${WHITE}_clean: run keys101_train.sh first"; exit 1; }

echo "== 1/2 Building the dataset"
LOG=outputs/keys_scripted/${STAMP}_build.log
trap 'echo "It stopped. The full output is in $LOG"' ERR
DRY=${DRY:-} uv run python - "$STAMP" "$WHITE" "$DATA" <<'EOF' 2>&1 | tee "$LOG" | grep -E "^(KEPT|DROPPED|BUILT|PUSHED|SKIP)"
import json, os, shutil, sys
from collections import Counter
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset

stamp, white, repo = sys.argv[1], sys.argv[2], sys.argv[3]
out = Path("outputs/keys_scripted")
root, clean, merged_dir = out / stamp, out / f"{stamp}_clean", out / f"{stamp}_all"
session = json.loads((out / f"{stamp}_session" / "session.json").read_text())
bad = [e["episode_index"] for e in session["episodes"] if any(p["ok"] is False for p in e["presses"])]
print(f"KEPT {len(session['episodes']) - len(bad)} sharps episodes")
print(f"DROPPED {len(bad)} whose press missed: {bad}")
for folder in (clean, merged_dir):
    if folder.exists():
        shutil.rmtree(folder)
if bad:
    delete_episodes(LeRobotDataset(f"{repo}-sharps", root=root), bad, output_dir=clean, repo_id=f"{repo}-sharps")
else:
    shutil.copytree(root, clean)
parts = [LeRobotDataset(f"{repo}-white", root=out / f"{white}_clean"), LeRobotDataset(f"{repo}-sharps", root=clean)]
merged = merge_datasets(parts, output_repo_id=repo, output_dir=merged_dir)
merged = LeRobotDataset(repo, root=merged_dir)
tasks = sorted(merged.meta.tasks.index) if hasattr(merged.meta.tasks, "index") else sorted(merged.meta.tasks)
print(f"BUILT {repo}: {merged.num_episodes} episodes, {merged.num_frames} frames, cameras {merged.meta.camera_keys}")
print(f"BUILT {len(tasks)} tasks: {tasks}")
if os.environ.get("DRY"):
    print("SKIP upload (DRY)")
else:
    merged.push_to_hub(private=True)
    print(f"PUSHED {repo}")
EOF

if [ -n "$DRY" ]; then
  echo "DRY: no upload, no training."
  exit 0
fi

echo "== 2/2 Starting the training on every key (${STEPS} steps)"
uv run lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=${DATA} \
  --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
  --dataset.image_transforms.enable=true --dataset.eval_split=0.1 --eval_steps=1000 \
  --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.ears": "observation.images.camera2"}' \
  --batch_size=64 --steps=${STEPS} --save_freq=2000 \
  --policy.repo_id=${MODEL} --policy.private=true --save_checkpoint_to_hub=true \
  --job.target=a100-large --job.timeout=3h --job.detach=true 2>&1 | grep -E "Job submitted|Job page|Model repo|Cancel|rror"

echo "The job runs in the cloud. Leave the rig as it is."
