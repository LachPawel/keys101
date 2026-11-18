#!/bin/zsh
# Teleop recording, one key per run: you drive the leader arm, the follower presses the key.
#
#   scripts/keys_teleop.sh do        # 50 episodes of C
#   scripts/keys_teleop.sh mi 30     # 30 episodes of E
#   scripts/keys_teleop.sh sol
#
# Notes: do re mi fa sol la (C D E F G A). Every run adds its episodes to the same dataset, so the result is
# one dataset and one policy; a run only decides which label its episodes get.
#
# lerobot-record names a new dataset <repo>_<date>_<time>. Later runs find the newest such dataset that has
# episodes and add to it. KEYS_NEW=1 scripts/keys_teleop.sh do starts a fresh one instead.
#
# The piano must be running in another tab:
#   uv run --with-requirements scripts/requirements-keys.txt scripts/keys_synth.py --ears "BlackHole 2ch"
#
# Keys while recording: right arrow ends the episode early, left arrow records it again, Esc stops.
set -e
cd "$(dirname "$0")/.."

REPO=${KEYS_REPO:-PawelLach/so101_keys_teleop}
EPISODES=${2:-50}

case "$1" in
  do)  TASK="play do, the first key" ;;
  re)  TASK="play re, the second key" ;;
  mi)  TASK="play mi, the third key" ;;
  fa)  TASK="play fa, the fourth key" ;;
  sol) TASK="play sol, the fifth key" ;;
  la)  TASK="play la, the sixth key" ;;
  *) echo "usage: scripts/keys_teleop.sh do|re|mi|fa|sol|la [episodes]"; exit 1 ;;
esac

HOME_DIR="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
EXISTING=""
EXISTING_DIR=""
if [[ -z "$KEYS_NEW" ]]; then
  # Only the recorder's own <repo>_<date>_<time> folders, not the _both/_text training sets built from them.
  for dir in "$HOME_DIR/$REPO"_<->_<->(N/On); do
    if grep -Eq '"total_episodes": *[1-9]' "$dir/meta/info.json" 2>/dev/null; then
      EXISTING="${REPO:h}/${dir:t}"
      EXISTING_DIR="$dir"
      break
    fi
  done
fi

if [[ -n "$EXISTING" ]]; then
  DATASET=(--dataset.repo_id="$EXISTING" --dataset.root="$EXISTING_DIR" --resume=true)
  echo "Adding to: $EXISTING"
else
  DATASET=(--dataset.repo_id="$REPO")
  echo "Starting a new dataset: ${REPO}_<date>_<time>"
fi

echo "Task: $TASK   ($EPISODES episodes, 6 s each, 3 s to get back to the start pose)"
uv run --with-requirements scripts/requirements-keys.txt lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/tty.usbmodem5B415331661 --robot.id=workshop_so101 \
  --robot.cameras="$(tr -d '\n' < scripts/keys_cameras_midi.json)" \
  --teleop.type=so101_leader --teleop.port=/dev/tty.usbmodem5B415332681 --teleop.id=workshop_so101 \
  "${DATASET[@]}" --dataset.single_task="$TASK" \
  --dataset.num_episodes="$EPISODES" --dataset.episode_time_s=6 --dataset.reset_time_s=3 \
  --dataset.fps=30 --dataset.push_to_hub=false
