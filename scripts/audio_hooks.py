"""ai-coustics hooks for the narration track.

- tyto_score: Tyto (tyto-1.1-l-16khz) scores the raw voice.wav in 5 s windows. The docs say to score audio
  before enhancement, so this runs on the original.
- voice_focus: Quail Voice Focus (quail-vf-2.3-l-16khz) isolates your voice into voice_vf.wav, which
  transcribe.py then prefers. The output is delay-compensated, so its timestamps match voice.wav.

Needs the Python SDK (`aic-sdk`, 3.4.0) and a key in AIC_SDK_LICENSE; when the variable is unset the key is
read from ~/code/codex/.env. Models are downloaded once into outputs/aic_models.

    uv run --with aic-sdk==3.4.0 scripts/audio_hooks.py outputs/keys_sessions/<stamp>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from keys_common import MANIFEST, load_json, read_wav, write_json, write_wav

VOICE_FOCUS_MODEL = "quail-vf-2.3-l-16khz"
TYTO_MODEL = "tyto-1.1-l-16khz"
TYTO_DIMENSIONS = (
    "risk_score",
    "speaker_reverb",
    "speaker_loudness",
    "interfering_speech",
    "noise",
    "packet_loss",
    "codec_degradation",
)
DEFAULT_ENV_FILE = Path.home() / "code" / "codex" / ".env"
MODELS_DIR = Path("outputs/aic_models")


def license_key(env_file: Path = DEFAULT_ENV_FILE) -> str:
    if key := os.environ.get("AIC_SDK_LICENSE"):
        return key
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip().removeprefix("export ") == "AIC_SDK_LICENSE" and value.strip():
                return value.strip().strip("\"'")
    sys.exit(f"No ai-coustics key: set AIC_SDK_LICENSE or add it to {env_file}")


def _sdk() -> Any:
    try:
        import aic_sdk
    except ImportError:
        sys.exit("The ai-coustics SDK is not installed: run with `uv run --with aic-sdk==3.4.0 ...`")
    return aic_sdk


def _load_model(aic: Any, model_id: str) -> Any:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return aic.Model.from_file(aic.Model.download(model_id, MODELS_DIR))


def enhance(audio: np.ndarray, rate: int, key: str, model: Any = None) -> np.ndarray:
    """Voice Focus on an array: the primary speaker, same length and timing as the input."""
    aic = _sdk()
    model = model or _load_model(aic, VOICE_FOCUS_MODEL)
    config = aic.ProcessorConfig.optimal(model, sample_rate=rate)
    processor = aic.Processor(model, key)
    try:
        processor.initialize(config)
        delay = processor.get_context().get_audio_delay()
        block = config.block_size
        padded = np.zeros(((audio.size + delay + block - 1) // block) * block, dtype=np.float32)
        padded[: audio.size] = audio
        output = np.empty_like(padded)
        for start in range(0, padded.size, block):
            output[start : start + block] = processor.process(padded[start : start + block])
        enhanced = output[delay : delay + audio.size]
        if not np.isfinite(enhanced).all():
            raise RuntimeError("Voice Focus returned non-finite samples")
        return enhanced
    finally:
        processor.terminate_session()


def voice_focus(wav_in: Path, wav_out: Path, key: str) -> None:
    """Isolate the primary speaker; same length and timing as the input. Follows the SDK quickstart."""
    audio, rate = read_wav(wav_in)
    write_wav(wav_out, enhance(audio, rate, key), rate)


def tyto_score(wav: Path, key: str, step_s: float = 1.0) -> dict[str, Any]:
    """Tyto scores per 5 s window plus mean, 95th percentile and max per dimension. Lower is better."""
    aic = _sdk()
    audio, rate = read_wav(wav)
    if audio.size < 5 * rate:
        raise ValueError(f"{wav}: shorter than Tyto's 5 s window")
    analyzer = aic.FileAnalyzer(_load_model(aic, TYTO_MODEL), key)
    try:
        results = analyzer.analyze(np.ascontiguousarray(audio), rate, round(rate * step_s))
    finally:
        del analyzer
    windows = {name: [float(getattr(r, name)) for r in results] for name in TYTO_DIMENSIONS}
    summary = {
        name: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))}
        for name, v in windows.items()
    }
    return {"model": TYTO_MODEL, "file": wav.name, "step_s": step_s, "summary": summary, "windows": windows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sessions", type=Path, nargs="+")
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--skip-tyto", action="store_true")
    parser.add_argument("--skip-voice-focus", action="store_true")
    args = parser.parse_args()

    key = license_key(args.env_file)
    for session in args.sessions:
        voice = session / load_json(session / MANIFEST)["audio"]["voice"]["file"]
        if not args.skip_tyto:
            scores = tyto_score(voice, key)
            write_json(session / "tyto.json", scores)
            risk = scores["summary"]["risk_score"]
            print(
                f"{session.name}: Tyto risk mean {risk['mean']:.2f}, p95 {risk['p95']:.2f} (lower is better)"
            )
        if not args.skip_voice_focus:
            voice_focus(voice, session / "voice_vf.wav", key)
            print(f"{session.name}: wrote voice_vf.wav")


if __name__ == "__main__":
    main()
