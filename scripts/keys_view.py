#!/usr/bin/env python
"""Line the camera up with the view the policy was trained on.

    uv run --with-requirements scripts/requirements-keys.txt scripts/keys_view.py            # live window
    uv run --with-requirements scripts/requirements-keys.txt scripts/keys_view.py --ref night
    uv run --with-requirements scripts/requirements-keys.txt scripts/keys_view.py --snapshot  # one picture

The live window shows the camera with the outlines of the training view drawn on top in green. Move Reachy
until the green outlines of the keys sit on the real keys. Press q to close.

References, taken from the recordings: `noon` (17 November, 134 episodes) and `night` (16 November, 134).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import draccus
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
REFS = ROOT / "outputs" / "keys_refs"


def outlines(image: np.ndarray) -> np.ndarray:
    """Strong edges only (keys, keyboard case, arm), so the wood grain of the table does not clutter the view."""
    grey = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (0, 0), 1.5)
    return cv2.dilate(cv2.Canny(grey, 60, 150), np.ones((2, 2), np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ref", choices=("noon", "night"), default="noon")
    parser.add_argument(
        "--snapshot", action="store_true", help="Save one overlay picture instead of a window"
    )
    args = parser.parse_args()

    from lerobot.cameras.configs import CameraConfig
    from lerobot.cameras.utils import make_cameras_from_configs
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    raw = json.loads((ROOT / "scripts" / "keys_cameras_front.json").read_text())
    camera = make_cameras_from_configs({k: draccus.decode(CameraConfig, v) for k, v in raw.items()})["front"]
    reference = cv2.imread(str(REFS / f"{args.ref}.png"))
    if reference is None:
        raise SystemExit(f"No reference picture {REFS / args.ref}.png")
    lines = outlines(reference) > 0

    def overlay() -> np.ndarray:
        live = cv2.cvtColor(camera.async_read(timeout_ms=1000), cv2.COLOR_RGB2BGR)
        live[lines] = (0, 255, 0)
        return live

    camera.connect()
    try:
        if args.snapshot:
            for _ in range(15):
                picture = overlay()
            out = ROOT / "outputs" / "captured_images" / "front_overlay.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), picture)
            print(f"VIEW saved {out}")
            return
        print("Move Reachy until the green outlines sit on the real keys. Press q in the window to close.")
        while True:
            cv2.imshow(
                f"camera, with the '{args.ref}' training view in green (q closes)",
                cv2.resize(overlay(), (1280, 720)),
            )
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        camera.disconnect()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
