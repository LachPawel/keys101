"""SpectrogramCamera: turns a microphone into 128x128 mel spectrogram frames for LeRobot."""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config_spectrogram import SpectrogramCameraConfig

logger = logging.getLogger(__name__)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> NDArray[np.float32]:
    """Triangular mel filters (HTK mel scale), shape (n_mels, n_fft // 2 + 1)."""

    def to_mel(f):
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def to_hz(m):
        return 700.0 * (10.0 ** (np.asarray(m) / 2595.0) - 1.0)

    edges = to_hz(np.linspace(to_mel(fmin), to_mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    filters = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for i in range(n_mels):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        rising = (bins - lo) / max(mid - lo, 1e-9)
        falling = (hi - bins) / max(hi - mid, 1e-9)
        filters[i] = np.clip(np.minimum(rising, falling), 0.0, None)
    return filters


def spectrogram_image(audio: NDArray[np.float32], config: SpectrogramCameraConfig, filters: NDArray) -> NDArray:
    """Render audio (the last window_s seconds) as a (height, width, 3) uint8 frame."""
    n_fft, width, height = config.n_fft, config.width, config.height
    starts = np.linspace(0, len(audio) - n_fft, width).astype(int)
    frames = audio.astype(np.float64)[starts[:, None] + np.arange(n_fft)] * np.hanning(n_fft)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2 / (n_fft / 2) ** 2  # 0 dB = full-scale sine
    # macOS Accelerate can raise spurious floating-point flags in matmul; the result is unaffected.
    with np.errstate(all="ignore"):
        mel_db = 10.0 * np.log10(filters.astype(np.float64) @ power.T + 1e-10)  # (n_mels, width)
    level = np.clip((mel_db - config.min_db) / (config.max_db - config.min_db), 0.0, 1.0)
    if level.shape[0] != height:  # resample the frequency axis to the frame height
        src = np.linspace(0, level.shape[0] - 1, height)
        level = np.stack([np.interp(src, np.arange(level.shape[0]), col) for col in level.T], axis=1)
    grey = (np.flipud(level) * 255).astype(np.uint8)  # low frequencies at the bottom
    return np.repeat(grey[:, :, None], 3, axis=2)


class SpectrogramCamera(Camera):
    """A microphone as a LeRobot camera. Audio streams into a ring buffer; a background thread renders
    a spectrogram frame `fps` times per second, served by `async_read` and `read_latest` like OpenCV."""

    def __init__(self, config: SpectrogramCameraConfig):
        super().__init__(config)
        self.config = config
        self.window = int(config.sample_rate * config.window_s)
        self.filters = mel_filterbank(
            config.sample_rate, config.n_fft, config.n_mels, config.fmin, config.fmax or config.sample_rate / 2
        )
        self.stream: sd.InputStream | None = None
        self.audio = np.zeros(self.window, dtype=np.float32)
        self.audio_lock = Lock()
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event = Event()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.config.device})"

    @property
    def is_connected(self) -> bool:
        return self.stream is not None and self.stream.active

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Input devices that can serve as ears, with their sounddevice index as `id`."""
        try:
            default_input = sd.default.device[0]
        except Exception:
            default_input = None
        found = []
        for index, device in enumerate(sd.query_devices()):
            if device["max_input_channels"] < 1:
                continue
            found.append(
                {
                    "name": device["name"],
                    "type": "Spectrogram",
                    "id": index,
                    "default_samplerate": device["default_samplerate"],
                    "max_input_channels": device["max_input_channels"],
                    "is_default": index == default_input,
                }
            )
        return found

    def _resolve_device(self) -> int | None:
        device = self.config.device
        if device is None or isinstance(device, int):
            return device
        if str(device).isdigit():
            return int(device)
        for entry in self.find_cameras():
            if str(device).lower() in entry["name"].lower():
                return entry["id"]
        raise ValueError(f"No input device matching {device!r}; found: {[d['name'] for d in self.find_cameras()]}")

    def _on_audio(self, indata, frames, time_info, status) -> None:
        chunk = indata[:, 0]
        with self.audio_lock:
            if len(chunk) >= self.window:
                self.audio[:] = chunk[-self.window :]
            else:
                self.audio = np.roll(self.audio, -len(chunk))
                self.audio[-len(chunk) :] = chunk

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")
        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            device=self._resolve_device(),
            callback=self._on_audio,
        )
        self.stream.start()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()
        if warmup:
            self.async_read(timeout_ms=2000)
        logger.info(f"{self} connected.")

    def read(self) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self.audio_lock:
            audio = self.audio.copy()
        return spectrogram_image(audio, self.config, self.filters)

    def _read_loop(self) -> None:
        period = 1.0 / self.config.fps
        while self.stop_event is not None and not self.stop_event.is_set():
            started = time.perf_counter()
            try:
                frame = self.read()
            except DeviceNotConnectedError:
                break
            except Exception as exc:  # keep serving the last frame rather than killing the loop
                logger.warning(f"{self} could not render a frame: {exc}")
            else:
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = time.perf_counter()
                self.new_frame_event.set()
            time.sleep(max(0.0, period - (time.perf_counter() - started)))

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"{self} produced no frame within {timeout_ms} ms.")
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()
        if frame is None:
            raise RuntimeError(f"{self} has no frame yet.")
        return frame

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        with self.frame_lock:
            frame, timestamp = self.latest_frame, self.latest_timestamp
        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not produced a frame yet.")
        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is {age_ms:.0f} ms old (max {max_age_ms} ms).")
        return frame

    def disconnect(self) -> None:
        if self.stream is None:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.thread = self.stop_event = None
        with self.frame_lock:
            self.latest_frame = self.latest_timestamp = None
            self.new_frame_event.clear()
        logger.info(f"{self} disconnected.")
