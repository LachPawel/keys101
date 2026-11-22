#!/usr/bin/env python
"""Reachy's head in Keys101: alive on the roadmap, locked on the piano while anything plays.

On the roadmap it sways, nods while it talks and turns toward whoever speaks: its microphone array says
where speech comes from, and it turns its body too when that is far to the side (keys_reachy.follow).

The moment a lesson or a mode starts it goes back to the saved pose and stays there: the policy's camera
is in that head. Only the antennas keep moving. A head sent back to a pose lands a few degrees off, so the
view is then matched by eye: the policy's own camera picture is compared with a frame of the recording
(rig/keyboard_reference.png) and the head is nudged until the keyboard sits where it sat in training.

Used by keys_coach.py (--head alive, the default with --voice). Nothing here can stop the game: with no
Reachy Mini daemon, or no saved pose, the head simply stays where it is.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections.abc import Callable
from typing import Any

RAD_PER_PX = (
    0.0023  # how far the head turns for one pixel in the policy's 640x360 view (measured: 26 px, 3.5 deg)
)
VIEW_OK_PX = 2.5  # the keyboard within this many pixels of the recording: good
MATCH_TRIES = 5


class Head:
    def __init__(self, arm: Any, spoken: Callable[[], str] = lambda: "") -> None:
        import keys_reachy

        self.kr = keys_reachy
        self.arm, self.spoken = arm, spoken
        self.busy = False  # something plays: hold the saved pose
        self.matching = False
        self.results: list[dict[str, bool]] = []  # two right answers make keys_reachy.follow bounce for joy
        self.stop = threading.Event()
        self.ok = keys_reachy.load() is not None
        if not self.ok:
            print(f"Reachy's head stays still: no saved pose ({keys_reachy.POSE_FILE}).", flush=True)
            return
        threading.Thread(
            target=keys_reachy.follow, args=(self, self.stop), kwargs={"look": True}, daemon=True
        ).start()
        threading.Thread(target=self._antennas, daemon=True).start()

    # keys_reachy.follow reads this as it reads the stage display's picture of the rig.
    @property
    def state(self) -> dict[str, Any]:
        return {
            "arm_busy": self.busy,
            "teacher": self.spoken(),
            "phase": "idle",
            "results": self.results,
            "sung": None,
        }

    def to_keys(self) -> None:
        """Back to the piano, and wait there until the view is the one the policy learned from."""
        if not self.ok:
            return
        self.results = []
        self.busy = True
        daemon = self.kr.Daemon()
        started, last = time.monotonic(), None
        try:
            while time.monotonic() - started < 4.0:
                time.sleep(0.15)
                now = daemon.pose()
                if time.monotonic() - started > 1.0 and last is not None and self.kr.still(last, now):
                    break
                last = now
            self.matching = True
            self._match_view(daemon)
        except OSError as error:
            print(f"Reachy's head: {error}", flush=True)
        finally:
            self.matching = False

    def release(self) -> None:
        """The playing is over: the head may look around again."""
        self.busy = False

    def cheer(self) -> None:
        self.results = [{"ok": True}, {"ok": True}]

    def close(self) -> None:
        self.stop.set()

    def _match_view(self, daemon: Any) -> None:
        """Nudge the head until the keyboard sits where it sat in the recording, judged from the policy's own
        camera. Needs an arm that can hand over that picture (the policy arm); else the saved pose is it."""
        snap = getattr(self.arm, "snap", None)
        if snap is None:
            return
        import cv2
        import keys_keyboard_check as check
        import numpy as np

        if not check.REFERENCE.exists():
            return
        reference = check.gray(cv2.imread(str(check.REFERENCE)))
        pose = self.kr.load()
        command = dict(pose["head"])
        yaw_sign, last_dx = -1.0, None  # a head turned left moves the picture right; flipped if that is wrong
        for attempt in range(MATCH_TRIES + 1):
            picture = snap()
            if picture is None:
                return
            seen = [
                (dx, dy)
                for dx, dy, sure in check.shifts(reference, check.gray(picture)).values()
                if sure > 0.3
            ]
            if len(seen) < 2:
                print("Reachy's head: the keyboard is hidden (hands?), the view is not checked.", flush=True)
                return
            dx, dy = float(np.median([s[0] for s in seen])), float(np.median([s[1] for s in seen]))
            if abs(dx) <= VIEW_OK_PX and abs(dy) <= VIEW_OK_PX:
                print(
                    f"Reachy's head: on the piano, the view matches within {max(abs(dx), abs(dy)):.0f} px.",
                    flush=True,
                )
                return
            if attempt == MATCH_TRIES:
                print(f"Reachy's head: on the piano, the view is {dx:+.0f}, {dy:+.0f} px off.", flush=True)
                return
            if last_dx is not None and abs(dx) > abs(last_dx) + 1.0:
                yaw_sign = -yaw_sign  # the last sideways nudge made it worse
            last_dx = dx
            step = lambda px: max(-0.06, min(0.06, 0.8 * px * RAD_PER_PX))
            command["pitch"] += step(dy)  # the picture too high: the head looks too far down
            command["yaw"] += yaw_sign * step(dx)
            daemon.goto({**pose, "head": command}, 0.35)
            time.sleep(1.0)

    def _antennas(self) -> None:
        """While the head is locked, the antennas keep a slow, small sway: Reachy is still watching."""
        daemon = self.kr.Daemon()
        works = True
        while not self.stop.is_set():
            time.sleep(0.1)
            if not (self.busy and works) or self.matching:
                continue
            t = time.time()
            sway = 0.25 * math.sin(2 * math.pi * t / 3.1)
            with contextlib.suppress(OSError):
                try:
                    daemon.call(
                        "POST", "/api/move/set_target", {"target_antennas": [0.2 + sway, -0.2 + sway]}
                    )
                except OSError as error:
                    if "HTTP 4" in str(error):
                        works = False  # this daemon wants the head in every target: leave the antennas be
                    raise
