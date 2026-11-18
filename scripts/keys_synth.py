"""A small piano for a USB MIDI keyboard that has no sound of its own.

Run it in its own terminal tab and leave it running: it plays every key through the speakers. The recorder
and the checkers can read the same keyboard at the same time.

    uv run --with-requirements scripts/requirements-keys.txt scripts/keys_synth.py
    uv run --with-requirements scripts/requirements-keys.txt scripts/keys_synth.py --demo   # no keyboard

With --ears "BlackHole 2ch" the same notes also go down a virtual cable. An `ears` camera listening on that
cable then sees the notes alone, straight from MIDI: no room, no voice, no speaker in between.

The tone is computed, not sampled, so it is the same on every machine and at every venue.
"""

from __future__ import annotations

import argparse
import contextlib
import queue
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

RATE = 48000
BLOCK = 256

# Relative level of harmonics 1, 2, 3, ... of a struck string.
HARMONICS = (1.0, 0.5, 0.33, 0.2, 0.12, 0.07)
ATTACK_S = 0.004
# The level halves this often while the key is held, and this often once it is released.
HOLD_HALF_S = 0.45
RELEASE_HALF_S = 0.04
# A voice quieter than this is dropped.
SILENT = 1e-4

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_hz(note: int) -> float:
    return 440.0 * 2.0 ** ((note - 69) / 12.0)


def note_name(note: int) -> str:
    return f"{NOTE_NAMES[note % 12]}{note // 12 - 1}"


@dataclass
class Voice:
    note: int
    level: float
    age: int = 0
    gain: float = 1.0
    released: bool = False


class Piano:
    """Turns note-ons and note-offs into audio, one block at a time."""

    def __init__(self, rate: int = RATE, volume: float = 0.5) -> None:
        self.rate = rate
        self.volume = volume
        self.voices: list[Voice] = []

    def note_on(self, note: int, velocity: int) -> None:
        self.note_off(note)
        self.voices.append(Voice(note, (velocity / 127.0) ** 1.5))

    def note_off(self, note: int) -> None:
        for voice in self.voices:
            if voice.note == note:
                voice.released = True

    def render(self, frames: int) -> NDArray[np.float32]:
        out = np.zeros(frames, dtype=np.float64)
        for voice in self.voices:
            t = (voice.age + np.arange(frames)) / self.rate
            hz = note_hz(voice.note)
            tone = np.zeros(frames, dtype=np.float64)
            for k, amp in enumerate(HARMONICS, start=1):
                if hz * k < self.rate / 2:
                    # Higher harmonics die away faster, as on a real string.
                    tone += amp * np.exp(-t * (k - 1) * 1.5) * np.sin(2 * np.pi * hz * k * t)
            half = RELEASE_HALF_S if voice.released else HOLD_HALF_S
            decay = 0.5 ** (np.arange(1, frames + 1) / (half * self.rate))
            attack = np.minimum(1.0, t / ATTACK_S)
            out += voice.level * voice.gain * decay * attack * tone
            voice.gain *= float(decay[-1])
            voice.age += frames
        self.voices = [v for v in self.voices if v.gain > SILENT]
        return np.tanh(self.volume * out).astype(np.float32)


def find_port(name: str) -> str:
    import mido

    ports = mido.get_input_names()
    if not ports:
        sys.exit("No MIDI keyboard found. Plug it in, or try --demo.")
    if name == "auto":
        return str(ports[0])
    for port in ports:
        if name.lower() in port.lower():
            return str(port)
    sys.exit(f"No MIDI input matching {name!r}. Found: {ports}")


def run(args: argparse.Namespace) -> None:
    import sounddevice as sd

    # One piano per output, so the speakers and the ears each run on their own audio clock.
    outputs = [args.device] + ([args.ears] if args.ears else [])
    pianos = [Piano(volume=args.volume) for _ in outputs]
    queues: list[queue.SimpleQueue[tuple[int, int]]] = [queue.SimpleQueue() for _ in outputs]

    def send(note: int, velocity: int) -> None:
        for events in queues:
            events.put((note, velocity))

    def player(piano: Piano, events: queue.SimpleQueue[tuple[int, int]]) -> Any:
        def audio(outdata: NDArray[np.float32], frames: int, _time: Any, _status: Any) -> None:
            while True:
                try:
                    note, velocity = events.get_nowait()
                except queue.Empty:
                    break
                if velocity > 0:
                    piano.note_on(note, velocity)
                else:
                    piano.note_off(note)
            outdata[:] = piano.render(frames)[:, None]

        return audio

    with contextlib.ExitStack() as stack:
        names = []
        for device, piano, events in zip(outputs, pianos, queues, strict=True):
            stream = sd.OutputStream(
                samplerate=RATE, blocksize=BLOCK, channels=1, dtype="float32", latency="low",
                device=device, callback=player(piano, events),
            )  # fmt: skip
            stack.enter_context(stream)
            names.append(sd.query_devices(stream.device, "output")["name"])
        device = " + ".join(names)
        if args.demo:
            print(f"Playing C D E F G A through: {device}")
            for note in (60, 62, 64, 65, 67, 69):
                send(note, 90)
                time.sleep(0.4)
                send(note, 0)
                time.sleep(0.1)
            time.sleep(0.5)
            return

        import mido

        def on_message(msg: Any) -> None:
            if msg.type == "note_on":
                send(msg.note, msg.velocity)
                if msg.velocity > 0:
                    print(f"  {note_name(msg.note):<4} velocity {msg.velocity}", flush=True)
            elif msg.type == "note_off":
                send(msg.note, 0)

        port = find_port(args.midi)
        with mido.open_input(port, callback=on_message):
            print(f"Piano on: {port} -> {device}. Leave this running. Ctrl+C stops.")
            try:
                while True:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                print()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--midi", default="auto", help="Part of the keyboard's name; auto takes the first input")
    p.add_argument("--device", default=None, help="Output device name or number; default is the Mac's output")
    p.add_argument(
        "--ears", default=None, help='A second output for the robot\'s ears, a virtual cable: "BlackHole 2ch"'
    )
    p.add_argument("--volume", type=float, default=0.5)
    p.add_argument("--demo", action="store_true", help="Play six notes without a keyboard, then stop")
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
