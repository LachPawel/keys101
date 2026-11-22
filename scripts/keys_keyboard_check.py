#!/usr/bin/env python
"""Is the keyboard where it was when the dataset was recorded? A live meter, to slide it back by hand.

    uv run scripts/keys_camera_serve.py          # in one tab: Reachy's camera
    uv run scripts/keys_keyboard_check.py        # in another: one line a second, until Ctrl+C

It compares Reachy's view with rig/keyboard_reference.png (a frame of the recording) at the keyboard's two
ends and its middle, and says which way to move each end, seen from the player's side. "ok" is within 3
pixels, about 4 mm: as much as the keyboard drifted during the recording itself. Keep your hands and the arm out of the picture while you read it.

The trained policies and the taught poses both expect the keyboard exactly there. Reachy's head must not
have moved either: this only measures the keyboard against the camera.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "rig" / "keyboard_reference.png"
URL = "http://127.0.0.1:8790/policy.jpg"
OK_PX = 3.0  # the keyboard drifted this much during the recording itself, and 99% of presses still landed
# Windows on the keyboard's near face and key tops (x0, y0, x1, y1), in Reachy's 640x360 view. Reachy looks
# at the keyboard from behind, so the picture's right side is the player's left: the low notes.
WINDOWS = {
    "low end (your left)": (440, 170, 630, 290),
    "middle": (230, 250, 430, 340),
    "high end (your right)": (20, 230, 200, 350),
}


def shifts(reference: np.ndarray, live: np.ndarray) -> dict[str, tuple[float, float, float]]:
    """For each window: how far the live picture is right of and below the reference, and how sure."""
    out = {}
    for name, (x0, y0, x1, y1) in WINDOWS.items():
        window = cv2.createHanningWindow((x1 - x0, y1 - y0), cv2.CV_32F)
        (dx, dy), sure = cv2.phaseCorrelate(reference[y0:y1, x0:x1], live[y0:y1, x0:x1], window)
        out[name] = (dx, dy, sure)
    return out


def advice(dx: float, dy: float) -> str:
    """What to do with this end, from the player's side. In the picture, right is the player's left and up is
    toward the player."""
    steps = []
    if abs(dy) > OK_PX:
        steps.append(f"{'push away from you' if dy < 0 else 'pull toward you'} {abs(dy):.0f} px")
    if abs(dx) > OK_PX:
        steps.append(f"slide {'right' if dx > 0 else 'left'} {abs(dx):.0f} px")
    return ", ".join(steps) or "ok"


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--once", action="store_true", help="One reading, then stop")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    args = parser.parse_args()
    reference = gray(cv2.imread(str(args.reference)))
    while True:
        try:
            data = urllib.request.urlopen(URL, timeout=3).read()
        except OSError:
            raise SystemExit("No camera: start it first with  uv run scripts/keys_camera_serve.py") from None
        live = gray(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR))
        parts = []
        for name, (dx, dy, sure) in shifts(reference, live).items():
            parts.append(f"{name}: {advice(dx, dy) if sure > 0.3 else 'cannot see it (hands away?)'}")
        print("   |   ".join(parts), flush=True)
        if args.once:
            return
        time.sleep(1.0)


if __name__ == "__main__":
    main()
