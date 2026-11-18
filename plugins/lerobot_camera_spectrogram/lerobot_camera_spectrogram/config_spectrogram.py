"""Configuration for the "spectrogram" camera: a microphone seen as a mel spectrogram image."""

from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig, ColorMode


@CameraConfig.register_subclass("spectrogram")
@dataclass
class SpectrogramCameraConfig(CameraConfig):
    """A microphone rendered as a rolling mel spectrogram, shaped like a camera frame.

    Each frame shows the last `window_s` seconds of audio: time runs left to right (newest on the
    right), frequency bottom to top, loudness as brightness. Frames are grey, repeated over three
    channels so they drop into anything that expects an RGB camera.

    Example (LeRobot CLI):
        --robot.cameras="{ears: {type: spectrogram, device: 1}}"

    Attributes:
        device: sounddevice input index, or part of the device name. None uses the system default input.
        sample_rate: Capture rate in Hz.
        window_s: Seconds of audio shown in one frame.
        n_fft: FFT size of each spectrogram column.
        n_mels: Mel bands before resizing to `height`.
        fmin, fmax: Frequency range of the mel bands in Hz; fmax None means sample_rate / 2.
        min_db, max_db: Loudness mapped to black and white, in dB relative to full scale. Fixed rather
            than per-frame, so a quiet room stays dark and a loud shake stays bright.
        width, height, fps: Frame size and rate; default 128 x 128 at 30 fps.
    """

    device: int | str | None = None
    sample_rate: int = 16000
    window_s: float = 1.0
    n_fft: int = 512
    n_mels: int = 128
    fmin: float = 20.0
    fmax: float | None = None
    min_db: float = -80.0
    max_db: float = 0.0
    color_mode: ColorMode = ColorMode.RGB

    def __post_init__(self) -> None:
        if self.width is None:
            self.width = 128
        if self.height is None:
            self.height = 128
        if self.fps is None:
            self.fps = 30
        if self.sample_rate <= 0 or self.window_s <= 0:
            raise ValueError("sample_rate and window_s must be positive")
        if self.n_fft > self.sample_rate * self.window_s:
            raise ValueError(f"n_fft ({self.n_fft}) is longer than the window ({self.window_s} s)")
        fmax = self.fmax or self.sample_rate / 2
        if not 0 <= self.fmin < fmax <= self.sample_rate / 2:
            raise ValueError(f"Need 0 <= fmin < fmax <= sample_rate / 2, got fmin={self.fmin}, fmax={fmax}")
        if self.min_db >= self.max_db:
            raise ValueError("min_db must be below max_db")
        if self.color_mode != ColorMode.RGB:
            raise ValueError("Spectrogram frames are grey RGB; only color_mode=rgb is supported")
