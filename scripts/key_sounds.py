"""Find key presses in a room recording.

Onsets come from spectral flux. Each onset is then identified by comparing what the note added to the
spectrum (post-onset minus pre-onset, on a quarter-tone grid) with one fingerprint per key that keys_map.py
recorded through the same speakers and microphone. Comparing against fingerprints instead of estimating
pitch keeps the two G sharps (keys 1 and 5, an octave apart) apart even when the laptop speakers barely play
the lower fundamental, and rejects onsets that are not a key at all, such as your voice.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from scipy.ndimage import maximum_filter1d, uniform_filter1d

F_MIN, F_MAX = 100.0, 4000.0
BINS_PER_SEMITONE = 4

HOP_S = 0.01
ONSET_FFT = 1024
# Flux peaks slightly before a sharp attack reaches the centre of the analysis window; measured on synthetic
# notes with a 5 ms attack.
ONSET_OFFSET_S = 0.008
# Onsets closer than this are one onset.
MIN_GAP_S = 0.1
# Flux must exceed its local mean by this many median absolute deviations.
FLUX_THRESHOLD = 8.0
# Level after the onset must be this far above the recording's noise floor.
MIN_LEVEL_DB = 12.0

# Similarity to accept an onset as a key. On synthetic tests a voice alone scores below 0.2 and a press with
# a word spoken over it about 0.65; clean presses score close to 1.
DEFAULT_MIN_SIM = 0.5

FEATURE_WIN_S = 0.2
FEATURE_PRE_GAP_S = 0.03
FEATURE_POST_GAP_S = 0.02
FEATURE_FFT = 16384

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def note_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_hz(midi: float | NDArray[np.floating]) -> NDArray[np.floating]:
    return np.asarray(440.0 * 2.0 ** ((np.asarray(midi) - 69) / 12))


def hz_to_midi(hz: float) -> float:
    return float(69 + 12 * np.log2(hz / 440.0))


def _band_log_magnitude(audio: NDArray[np.float32], rate: int, hop: int) -> NDArray[np.float32]:
    """Log magnitude STFT between F_MIN and F_MAX, (frames, bins). Computed in blocks to bound memory."""
    if len(audio) < ONSET_FFT:
        audio = np.pad(audio, (0, ONSET_FFT - len(audio)))
    freqs = np.fft.rfftfreq(ONSET_FFT, 1 / rate)
    band = (freqs >= F_MIN) & (freqs <= F_MAX)
    window = np.hanning(ONSET_FFT).astype(np.float32)
    frames = sliding_window_view(audio, ONSET_FFT)[::hop]
    out: NDArray[np.float32] = np.empty((len(frames), int(band.sum())), dtype=np.float32)
    for start in range(0, len(frames), 2048):
        block = frames[start : start + 2048] * window
        out[start : start + len(block)] = np.abs(np.fft.rfft(block, axis=1))[:, band]
    return out


def detect_onsets(
    audio: NDArray[np.float32], rate: int, threshold: float = FLUX_THRESHOLD
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Onset times in seconds from the start of `audio`, and their flux strength."""
    hop = round(HOP_S * rate)
    mag = _band_log_magnitude(audio, rate, hop)
    level_db = 10 * np.log10(np.mean(mag**2, axis=1) + 1e-20)
    logmag = np.log1p(1000 * mag / (mag.max() + 1e-12))

    # SuperFlux-style: compare each bin with the strongest neighbouring bin two frames earlier, so slow
    # changes and vibrato in the voice add little flux.
    reference = maximum_filter1d(logmag, size=3, axis=1)
    flux = np.zeros(len(logmag))
    flux[2:] = np.maximum(0.0, logmag[2:] - reference[:-2]).sum(axis=1)

    mad = np.median(np.abs(flux - np.median(flux))) + 1e-9
    local_mean = uniform_filter1d(flux, size=21)
    local_max = maximum_filter1d(flux, size=7)
    level_after = maximum_filter1d(level_db, size=11, origin=-5)
    noise_floor = np.percentile(level_db, 10)
    candidates = np.flatnonzero(
        (flux == local_max)
        & (flux >= local_mean + threshold * mad)
        & (level_after >= noise_floor + MIN_LEVEL_DB)
    )

    min_gap = round(MIN_GAP_S / HOP_S)
    kept: list[int] = []
    for idx in candidates:
        if kept and idx - kept[-1] < min_gap:
            if flux[idx] > flux[kept[-1]]:
                kept[-1] = idx
            continue
        kept.append(int(idx))
    frames = np.asarray(kept, dtype=np.int64)
    times = (frames * hop + ONSET_FFT / 2) / rate + ONSET_OFFSET_S
    return np.maximum(times, 0.0), flux[frames]


def _semitone_grid() -> NDArray[np.floating]:
    lo, hi = hz_to_midi(F_MIN), hz_to_midi(F_MAX)
    return np.arange(lo, hi, 1 / BINS_PER_SEMITONE)


def _segment(audio: NDArray[np.float32], start: int, length: int) -> NDArray[np.float32]:
    seg: NDArray[np.float32] = np.zeros(length, dtype=np.float32)
    lo, hi = max(start, 0), min(start + length, len(audio))
    if hi > lo:
        seg[lo - start : hi - start] = audio[lo:hi]
    return seg


def _grid_spectrum(segment: NDArray[np.float32], rate: int) -> NDArray[np.floating]:
    """Magnitude spectrum max-pooled into quarter-tone bands between F_MIN and F_MAX."""
    mag = np.abs(np.fft.rfft(segment * np.hanning(len(segment)), FEATURE_FFT))
    freqs = np.fft.rfftfreq(FEATURE_FFT, 1 / rate)
    edges = midi_to_hz(np.append(_semitone_grid(), hz_to_midi(F_MAX)) - 0.5 / BINS_PER_SEMITONE)
    starts = np.minimum(np.searchsorted(freqs, edges[:-1]), len(mag) - 1)
    return np.asarray(np.maximum.reduceat(mag, starts))


def note_feature(audio: NDArray[np.float32], rate: int, t: float) -> NDArray[np.float64]:
    """Unit-length fingerprint of what the sound at time `t` added to the spectrum."""
    win = round(FEATURE_WIN_S * rate)
    i = round(t * rate)
    post = _grid_spectrum(_segment(audio, i + round(FEATURE_POST_GAP_S * rate), win), rate)
    pre = _grid_spectrum(_segment(audio, i - round(FEATURE_PRE_GAP_S * rate) - win, win), rate)
    added = np.maximum(0.0, post - pre)
    feature = np.log1p(100 * added / (added.max() + 1e-12))
    return np.asarray(feature / (np.linalg.norm(feature) + 1e-12))


def classify(
    features: NDArray[np.float64], templates: dict[int, NDArray[np.float64]], min_sim: float, margin: float
) -> list[tuple[int | None, int, float, float]]:
    """(key or None if rejected, best key, best similarity, runner-up similarity) for each feature row."""
    keys = sorted(templates)
    # errstate: Accelerate's matmul on Apple Silicon raises spurious floating-point warnings.
    with np.errstate(all="ignore"):
        sims = np.atleast_2d(features) @ np.stack([templates[k] for k in keys]).T
    out: list[tuple[int | None, int, float, float]] = []
    for row in sims:
        order = np.argsort(row)[::-1]
        best, second = float(row[order[0]]), float(row[order[1]]) if len(order) > 1 else 0.0
        key = keys[order[0]] if best >= min_sim and best - second >= margin else None
        out.append((key, keys[order[0]], best, second))
    return out


def estimate_midi(audio: NDArray[np.float32], rate: int, t: float, lo: int = 40, hi: int = 90) -> int:
    """Rough pitch of the note starting at `t`, only to check the sticker table by ear.

    Scores every candidate by how many of its first 8 harmonics are present, then takes the lowest candidate
    close to the best score, since a note's octave-up candidate matches all of the note's even harmonics.
    """
    seg = _segment(audio, round((t + FEATURE_POST_GAP_S) * rate), round(FEATURE_WIN_S * rate))
    mag = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), FEATURE_FFT))
    freqs = np.fft.rfftfreq(FEATURE_FFT, 1 / rate)
    present = np.clip((20 * np.log10(mag / (mag.max() + 1e-12) + 1e-9) + 50) / 50, 0, 1)
    scores = []
    for midi in range(lo, hi + 1):
        harmonics = midi_to_hz(midi) * np.arange(1, 9)
        harmonics = harmonics[harmonics < F_MAX]
        values = [present[(freqs >= h * 0.985) & (freqs <= h * 1.015)].max() for h in harmonics]
        scores.append(float(np.mean(values)))
    best = max(scores)
    return lo + next(i for i, s in enumerate(scores) if s >= 0.8 * best)
