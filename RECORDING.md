# Recording the Keys101 dataset and training SmolVLA

The arm records its own dataset: it presses twelve sticker keys from poses you teach once, and the script
labels every episode with its note ("press E3"). No leader arm, no narration. About 30 minutes of recording,
about 2 hours of training.

One policy learns all twelve keys:

- **Stickers 1 to 6, the bass:** C3 D3 E3 F3 G3 A3, the first six white keys from the left. The arm plays
  them with you in Play together, as your left hand.
- **Stickers 7 to 12, the melody:** C4 D4 E4 F4 G4 A4, the six white keys from middle C. The arm shows them
  to you in Learn and Watch and repeat.

The same recording trains two models at once, to compare them: one with the camera and the "ears" (the MIDI
notes drawn as a picture), one with the camera only.

`keys_layout.json` names the note under every sticker. The recorder and Keys101 both read it.
Every command runs from the repository root (`~/code/keys101`). Shortcut:

```bash
R="uv run --with-requirements scripts/requirements-keys.txt"
```

## Sharp keys: no

White keys only. The MicroLab's black keys are narrow and sit between two white ones, so a gripper tip a few
millimetres off plays the wrong note or two at once. No lesson needs the policy on a black key.

## 0. Once, before anything

- `hf auth login`, with a token that can write. Training runs on Hugging Face Jobs and needs billing set up.
- BlackHole 2ch installed (the robot's "ears" listen on it).
- The MicroLab's octave buttons at the default, so middle C is C4 (MIDI 60).
- Stickers 1 to 6 on C3 to A3 (white keys 1 to 6 from the left), stickers 7 to 12 on C4 to A4 (white keys
  8 to 13). The 7th white key, B3, and the last two get none.

## 1. Set up the rig, then do not move it

- Arm (the follower, port `usbmodem5B415331661`), keyboard and Reachy Mini in their demo positions. Tape down
  the keyboard, the arm's base and Reachy. Anything that moves after recording costs the policy accuracy.
- The arm must reach both C3 and A4 without stretching flat.

## 2. The piano, in its own terminal tab, running the whole time

```bash
$R scripts/keys_synth.py --ears "BlackHole 2ch"
```

It plays every key through the speakers and sends the same notes down the cable the ears camera listens on.

## 3. Reachy's camera

The view is set and saved: Reachy behind the keyboard on the window side, facing the player. The head pose is
in `reachy_pose.json`, the layout and camera settings in `rig/setup.json`, and the view the policy gets in
`rig/view_policy.jpg`. Reachy's camera is index 0, as `scripts/keys_cameras_midi.json` expects.

- Do not move the head from now until the demo is over. It has 5 to 10 degrees of play (most in roll), so
  any move and return lands tens of pixels off. `keys_reachy.py lock` leaves it alone while it is still there.
- Keep Reachy powered and the Reachy Mini Control app running the whole time, and do not unplug or move its
  USB cable: when the app restarts, the head drops and comes back somewhere else.
- Do not run the stage display (`keys_ui.py`) or the voice teacher while recording: they move the head.
- To look through the camera, run this and open http://127.0.0.1:8790/policy.jpg. Stop it (Ctrl+C) before
  recording: the recorder needs the camera.

```bash
uv run scripts/keys_camera_serve.py
```

## 4. Teach the poses (about 10 minutes)

Torque goes off and the arm goes limp. Move it by hand to each pose and press Enter: rest, home, then for each
of the twelve keys `down` (gripper closed, tip pressing the key until it sounds) and `above` (about 4 cm
straight up from there). The prompt names the note: "key 7 (C4)".

```bash
$R scripts/keys_scripted.py teach
```

## 5. Test every key

```bash
$R scripts/keys_scripted.py test --midi MicroLab
```

Each key must print its own note: 48, 50, 52, 53, 55, 57, then 60, 62, 64, 65, 67, 69. This also writes
`keys_midi.json`, so the recording can check every press. A key that misses: re-teach it
(`teach --only 7`) or nudge it (`keys_scripted.py adjust 7`), then test again.

## 6. Record (about 30 minutes)

Stop the camera server first. A one-minute trial comes first, to see that the presses land at this speed:

```bash
$R scripts/keys_scripted.py record --presses 36 --speed 90 --fast --jitter 1 --max-step 25 --midi MicroLab --cameras scripts/keys_cameras_midi.json --repo-id PawelLach/so101-keys101-trial
```

"Presses that sounded their key: 36/36" (or close): go on to the real one.

```bash
$R scripts/keys_scripted.py record --presses 600 --speed 90 --fast --jitter 1 --max-step 25 --midi MicroLab --cameras scripts/keys_cameras_midi.json --repo-id PawelLach/so101-keys101-12keys
```

- One episode per press, about 50 per key, labelled with the note: "press C3", "press E4". Each starts where
  the last press ended. The order is a tour that makes every jump between two keys equally often, bass to
  melody included.
- `--speed 90 --fast` records quick presses, so the policy learns to be quick. Long jumps rise in an arc and
  the arm stops for a moment above the key before pressing: without that it dragged over the keys and hit
  neighbours. `--max-step 25` lets the motors keep up with the plan.
- Hands on the keyboard are fine, and in Keys101 they are there anyway. Hold the keyboard still if it is not
  taped down. Never press a key while it records: the ears and the press check would hear it.
- A line that ends in `WRONG` or `NO NOTE` is a press that missed. A few are fine: they are dropped before
  training. Many: Ctrl+C, re-teach that key, start again.
- Ctrl+C keeps every finished episode. The last lines print the folder, `outputs/keys_scripted/<stamp>`.
- If it hangs while disconnecting the ears camera, close the tab: nothing is lost.

## 7. Push and train both models (one command, about 2 hours, about $5 each)

With the folder name the recorder printed:

```bash
scripts/keys101_train.sh 20261122_104500
```

It drops the episodes whose press missed, makes a second copy of the dataset without the ears (camera only),
pushes both to your Hugging Face account as private datasets, and starts two cloud trainings on an A100:
`PawelLach/smolvla-keys101` (camera and ears) and `PawelLach/smolvla-keys101-camera-only`. Both start from the
SmolVLA base with the vision encoder unfrozen, 6,000 steps, a checkpoint every 2,000 steps.
`DRY=1 scripts/keys101_train.sh <stamp>` builds the datasets without uploading or training.

Do not use `scripts/keys_flow_train.sh` for this: it pushes to the `flow` policy's repo, the public one.

While they train, leave the rig exactly as it is. The 4,000-step checkpoint (about 80 minutes in) is worth a
first try.

## 8. Use them in Keys101

Download a checkpoint of each and link it where Keys101 looks for the models `keys101` and `keys101-cam`:

```bash
STEP=006000
```

```bash
uv run hf download PawelLach/smolvla-keys101 --include "checkpoints/$STEP/pretrained_model/*" --local-dir outputs/keys_models/keys101_hub && ln -sfn keys101_hub/checkpoints/$STEP/pretrained_model outputs/keys_models/keys101
```

```bash
uv run hf download PawelLach/smolvla-keys101-camera-only --include "checkpoints/$STEP/pretrained_model/*" --local-dir outputs/keys_models/keys101_cam_hub && ln -sfn keys101_cam_hub/checkpoints/$STEP/pretrained_model outputs/keys_models/keys101-cam
```

With the piano tab still running:

```bash
$R scripts/keys_coach.py --arm policy
```

```bash
$R scripts/keys_coach.py --arm policy --model keys101-cam
```

Keys101 sends "press E3", the words this dataset was recorded with. The policy plays the bass in Play
together and the melody in Watch and repeat. `--bass poses` plays the bass from the taught poses instead:
instant, for when the policy is slow. (The older `flow` policy runs with `--model flow --prompts flow`.)

## If something goes wrong

- The model misses keys: the rig moved, or Reachy's head is not where it was (compare with
  `rig/view_policy.jpg`).
- No time to train: `keys_coach.py --arm scripted` runs every mode from the taught poses.
- Before you submit, make the datasets and the models public on Hugging Face.
