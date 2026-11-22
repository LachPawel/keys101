#!/bin/zsh
# From a finished Keys101 recording to two SmolVLA trainings in the cloud.
#
#   scripts/keys101_train.sh 20261122_104500          # the folder name the recorder printed
#   DRY=1 scripts/keys101_train.sh 20261122_104500    # build the datasets only: no upload, no training
#
# What it does:
#   1. Drops the episodes whose press did not sound its own key (the recorder checked each one on MIDI).
#   2. Makes a second copy without the ears camera: camera only.
#   3. Pushes both to your Hugging Face account, private.
#   4. Starts two paid cloud jobs on a100-large, one per dataset, and returns at once.
#
# Both train from the SmolVLA base with the vision encoder unfrozen, on the same episodes: the ears are the
# only difference. Cost: 6,000 steps each, about 2 hours, about $5 each. The 3 hour limit caps each at $7.50.
# A checkpoint goes to the Hub every 2,000 steps.
set -e
set -o pipefail
cd "$(dirname "$0")/.."

STAMP=${1:?Give the recording folder name, e.g. 20261122_104500}
ROOT=outputs/keys_scripted/$STAMP
DATA=${KEYS101_DATA:-PawelLach/so101-keys101-12keys}
MODEL=${KEYS101_MODEL:-PawelLach/smolvla-keys101}
STEPS=${KEYS101_STEPS:-6000}
[ -d "$ROOT" ] || { echo "No $ROOT"; exit 1; }

echo "== 1/3 Building the datasets"
LOG=outputs/keys_scripted/${STAMP}_build.log
trap 'echo "It stopped. The full output is in $LOG"' ERR
DRY=${DRY:-} uv run python - "$STAMP" "$DATA" <<'EOF' 2>&1 | tee "$LOG" | grep -E "^(KEPT|DROPPED|BUILT|PUSHED|SKIP)"
import json, os, shutil, sys
from pathlib import Path

from lerobot.datasets.dataset_tools import delete_episodes, remove_feature
from lerobot.datasets.lerobot_dataset import LeRobotDataset

stamp, repo = sys.argv[1], sys.argv[2]
out = Path("outputs/keys_scripted")
root, clean, camera = out / stamp, out / f"{stamp}_clean", out / f"{stamp}_camera_only"
session = json.loads((out / f"{stamp}_session" / "session.json").read_text())
bad = [e["episode_index"] for e in session["episodes"] if any(p["ok"] is False for p in e["presses"])]
print(f"KEPT {len(session['episodes']) - len(bad)} episodes")
print(f"DROPPED {len(bad)} whose press missed: {bad}")
for folder in (clean, camera):
    if folder.exists():
        shutil.rmtree(folder)
if bad:
    delete_episodes(LeRobotDataset(repo, root=root), bad, output_dir=clean, repo_id=repo)
else:
    shutil.copytree(root, clean)
remove_feature(
    LeRobotDataset(repo, root=clean), "observation.images.ears", output_dir=camera, repo_id=f"{repo}-camera-only"
)
both, cam = LeRobotDataset(repo, root=clean), LeRobotDataset(f"{repo}-camera-only", root=camera)
print(f"BUILT {repo}: {both.num_episodes} episodes, {both.num_frames} frames, cameras {both.meta.camera_keys}")
print(f"BUILT {repo}-camera-only: {cam.num_episodes} episodes, cameras {cam.meta.camera_keys}")
if os.environ.get("DRY"):
    print("SKIP upload (DRY)")
else:
    both.push_to_hub(private=True)
    print(f"PUSHED {repo}")
    cam.push_to_hub(private=True)
    print(f"PUSHED {repo}-camera-only")
EOF

if [ -n "$DRY" ]; then
  echo "DRY: no upload, no training."
  exit 0
fi

train() {  # dataset, model repo, rename map
  uv run lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=$1 \
    --policy.freeze_vision_encoder=false --policy.train_expert_only=false \
    --dataset.image_transforms.enable=true --dataset.eval_split=0.1 --eval_steps=1000 \
    --rename_map="$3" \
    --batch_size=64 --steps=${STEPS} --save_freq=2000 \
    --policy.repo_id=$2 --policy.private=true --save_checkpoint_to_hub=true \
    --job.target=a100-large --job.timeout=3h --job.detach=true 2>&1 | grep -E "Job submitted|Job page|Model repo|Cancel|rror"
}

echo "== 2/3 Starting the training with camera and ears (${STEPS} steps)"
train ${DATA} ${MODEL} '{"observation.images.front": "observation.images.camera1", "observation.images.ears": "observation.images.camera2"}'

echo "== 3/3 Starting the training with the camera only (${STEPS} steps)"
train ${DATA}-camera-only ${MODEL}-camera-only '{"observation.images.front": "observation.images.camera1"}'

echo "Both jobs run in the cloud. Leave the rig as it is."
