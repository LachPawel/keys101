# Recording the Keys101 dataset and training SmolVLA

The arm records its own dataset: it presses the six sticker keys from poses you teach once, and the script
labels every episode ("press key 3"). No leader arm, no narration. About 20 minutes of recording, about 2
hours of training.

The policy learns the six white keys C4 D4 E4 F4 G4 A4 (stickers 1 to 6). Keys101 uses it for the arm's own
notes in Watch and repeat and Jam. Pointing, Play together and every other key use the taught poses, so
the app works before the model is ready.

Every command runs from the repository root (`~/code/keys101`). Shortcut:

```bash
R="uv run --with-requirements scripts/requirements-keys.txt"
```

## Sharp keys: no

Train the six white keys only.

- The MicroLab has mini keys. The black keys are narrow and sit between two white ones, so a gripper tip that
  is a few millimetres off plays the wrong note or two notes at once.
- Keys101 never asks the policy for a black key. The arm's parts in the songs are all white keys, and black
  keys (the D sharp in Für Elise) are pointed at from taught poses.
- More keys means fewer episodes per key in the same 20 minutes. 50 per key is what worked for the `flow`
  policy.

If you have time for a second dataset, add more white keys (B4, C5), not sharps.

## 0. Once, before anything

- `hf auth login`, with a token that can write. Training runs on Hugging Face Jobs and needs billing set up.
- BlackHole 2ch installed (the robot's "ears" listen on it).
- The MicroLab's octave buttons at the default, so middle C is C4 (MIDI 60).
- Stickers 1 to 6 on the six white keys from middle C: C4 D4 E4 F4 G4 A4.
- The name of the keyboard's MIDI port:

```bash
uv run --with mido --with python-rtmidi python -c "import mido; print(mido.get_input_names())"
```

## 1. Set up the rig, then do not move it

- Arm, keyboard and Reachy Mini in their demo positions. Tape down the keyboard, the arm's base and Reachy.
  Anything that moves after recording costs the policy accuracy.
- Even light from the side. No bright window in the camera's view.

## 2. The piano, in its own terminal tab, running the whole time

```bash
$R scripts/keys_synth.py --ears "BlackHole 2ch"
```

It plays every key through the speakers and sends the same notes down the cable the ears camera listens on.

## 3. Reachy's camera

The view is set and saved: Reachy behind the keyboard on the window side, facing the player. The head pose is
in `reachy_pose.json`, the layout and camera settings in `rig/setup.json`, and the view the policy gets in
`rig/view_policy.jpg`. Reachy's camera is index 0, as `scripts/keys_cameras_midi.json` expects.

- Do not move the head from now until the demo is over. It has 2 to 3 degrees of play, so any move and
  return lands 10 to 50 pixels off. `keys_reachy.py lock` leaves it alone while it is still there.
- Do not run the stage display (`keys_ui.py`) or the voice teacher while recording: they move the head.
- To look through the camera, run this in your own Terminal app (it needs camera permission) and open
  http://127.0.0.1:8790/policy.jpg. Stop it (Ctrl+C) before recording: the recorder needs the camera.

```bash
uv run scripts/keys_camera_serve.py
```

- If the head was moved anyway (Reachy restarted): `$R scripts/keys_reachy.py lock` brings it back to within
  2 to 3 degrees. Compare with `rig/view_policy.jpg` before recording.

## 4. Teach the poses (5 minutes)

Torque goes off and the arm goes limp. Move it by hand to each pose and press Enter: rest, home, then for each
key `down` (tip pressing the key) and `above` (about 4 cm straight up from there).

```bash
$R scripts/keys_scripted.py teach
```

## 5. Test every key

```bash
$R scripts/keys_scripted.py test --midi MicroLab
```

Each key must print its own note: 60, 62, 64, 65, 67, 69. This also writes `keys_midi.json`, so the
recording can check every press. A key that misses: re-teach it (`teach --only 3`) or nudge it
(`keys_scripted.py adjust 3`), then test again.

## 6. Record (about 20 minutes)

```bash
$R scripts/keys_scripted.py record --presses 300 --speed 90 --midi MicroLab --cameras scripts/keys_cameras_midi.json --repo-id PawelLach/so101-keys101-300ep
```

- One episode per press, task "press key N". Each starts where the last press ended, the way Keys101 uses it.
  The episode scripts cover every pair of keys.
- 90 deg/s is faster than the default 60. Do not go higher: faster demos are harder to learn.
- For about a third of the episodes, rest a hand on the keys below C4 without pressing. In Keys101 your hands
  are on the keyboard while the arm plays, so the policy should have seen a hand. Never press a key while it
  records: the ears and the press check would hear it.
- Every line should end in `ok`. Many `WRONG` or `NO NOTE`: Ctrl+C, re-teach that key, start again.
- Ctrl+C keeps every finished episode. The last lines print the folder, `outputs/keys_scripted/<stamp>`.
- If it hangs while disconnecting the ears camera, close the tab: nothing is lost.

## 7. Push the dataset (private)

Put the folder name from the end of the recording in `STAMP`:

```bash
STAMP=20261122_101500
```

```bash
uv run python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; D('PawelLach/so101-keys101-300ep', root='outputs/keys_scripted/$STAMP').push_to_hub(private=True)"
```

## 8. Train (about 2 hours, about $5)

From the SmolVLA base, on an A100 on Hugging Face Jobs. It returns at once and runs in the cloud.
A checkpoint goes to the private model repo every 2,000 steps.

```bash
uv run lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=PawelLach/so101-keys101-300ep --policy.freeze_vision_encoder=false --policy.train_expert_only=false --dataset.image_transforms.enable=true --dataset.eval_split=0.1 --eval_steps=1000 --rename_map='{"observation.images.front": "observation.images.camera1", "observation.images.ears": "observation.images.camera2"}' --batch_size=64 --steps=6000 --save_freq=2000 --policy.repo_id=PawelLach/smolvla-keys101 --policy.private=true --save_checkpoint_to_hub=true --job.target=a100-large --job.timeout=3h --job.detach=true
```

Do not use `scripts/keys_flow_train.sh` for this: it pushes to the `flow` policy's repo, the public one.

While it trains, leave the rig exactly as it is. The 4,000-step checkpoint (about 80 minutes in) is worth a
first try.

## 9. Use it in Keys101

Download a checkpoint and link it where Keys101 looks for the `keys101` model:

```bash
STEP=006000
```

```bash
uv run hf download PawelLach/smolvla-keys101 --include "checkpoints/$STEP/pretrained_model/*" --local-dir outputs/keys_models/keys101_hub
```

```bash
ln -sfn keys101_hub/checkpoints/$STEP/pretrained_model outputs/keys_models/keys101
```

With the piano tab still running:

```bash
$R scripts/keys_coach.py --arm policy --model keys101 --prompts scripted
```

`--prompts scripted` sends "press key 3", the words this dataset was recorded with. (The older `flow` policy
was trained on "play mi, the third key" and runs with the default `--prompts flow`.)

## If something goes wrong

- The model misses keys: the rig moved, or Reachy's head is not at the saved pose (`keys_reachy.py lock`).
- No time to train: `keys_coach.py --arm scripted` runs every mode from the taught poses.
- Before you submit, make the dataset and the model public on Hugging Face.
