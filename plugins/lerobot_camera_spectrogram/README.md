# lerobot_camera_spectrogram

A LeRobot camera plugin that turns a microphone into a camera: each frame is a 128x128 mel
spectrogram of the last second of audio (time left to right, frequency bottom to top, loudness as
brightness). Record it next to your regular cameras and train policies that can hear.

## Install

Into an existing LeRobot environment, without touching its lerobot version:

```bash
pip install -e .            # or: uv pip install -e . --python /path/to/env/python
```

For LeLab, see LeLab's README ("Microphone as a camera").

## Use

```bash
python -c "from lerobot_camera_spectrogram import SpectrogramCamera; print(SpectrogramCamera.find_cameras())"

lerobot-record ... --robot.cameras="{front: {type: opencv, index_or_path: 0}, ears: {type: spectrogram, device: 1}}"
```

`device` is a sounddevice input index or part of a device name; leave it out for the default input.
Other options (sample rate, window length, dB range) are documented in `config_spectrogram.py`.
