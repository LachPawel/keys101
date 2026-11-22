#!/usr/bin/env python
"""Reachy's head in Keys101: alive on the roadmap, locked on the piano while anything plays.

On the roadmap it sways, nods while it talks and turns toward whoever speaks: its microphone array says
where speech comes from, and it turns its body too when that is far to the side (keys_reachy.follow).

The moment a lesson or a mode starts it goes back to the pose it held during the recording and stays
there: the policy's camera is in that head. Only the antennas keep moving. This head lands up to ten degrees
from where it is sent, and differently from every side, so it is always brought in the same way (from above
and from one side in roll), its measured pose is compared with the recording's (`held` in reachy_pose.json)
and the command is corrected until they agree within a degree. The command that worked is remembered for
the next time. The policy's own camera picture is then compared with a frame of the recording
(rig/keyboard_reference.png), and the offset is printed.

Used by keys_coach.py (--head alive, the default with --voice). Nothing here can stop the game: with no
Reachy Mini daemon, or no saved pose, the head simply stays where it is.
"""

from __future__ import annotations

import contextlib
import json
import math
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

AXES = ("roll", "pitch", "yaw")
APPROACH = {"roll": 0.15, "pitch": -0.25}  # every return comes through this offset: the play is then the same
# Good enough: about two degrees, some fifteen pixels in the policy's view. Measured with the head brought
# back this way, the camera-only policy pressed 24 of 24 keys right. Closer is beyond this neck's play.
POSE_OK_RAD = 0.04
RETURN_TRIES = 3
COMMAND = Path(__file__).resolve().parent.parent / "outputs" / "reachy_keys_command.json"  # what worked last


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
            while time.monotonic() - started < 3.0:  # keys_reachy.follow sends it back first: let that end
                time.sleep(0.15)
                now = daemon.pose()
                if time.monotonic() - started > 0.6 and last is not None and self.kr.still(last, now):
                    break
                last = now
            self.matching = True
            self._return(daemon)
            self._report_view()
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

    def _return(self, daemon: Any) -> None:
        """Bring the head to the pose it held in the recording, and correct the command until its measured
        pose agrees. Every try comes in through the same approach pose, so the result can be repeated."""
        pose = self.kr.load()
        held = pose.get("held")
        if not held:
            return

        def off() -> dict[str, float]:
            now = daemon.pose()["head"]
            return {a: float(held[a]) - float(now[a]) for a in AXES}

        def go(command: dict[str, float], seconds: float, settle: float) -> None:
            daemon.goto({**pose, "head": {**pose["head"], **command}}, seconds)
            time.sleep(seconds + settle)

        if max(abs(v) for v in off().values()) < POSE_OK_RAD:
            return  # nobody moved it
        command = {a: float(held[a]) for a in AXES}
        with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
            saved = json.loads(COMMAND.read_text())
            if saved.get("held") == {a: held[a] for a in AXES}:
                command = {a: float(saved["command"][a]) for a in AXES}
        worst = 9.0
        for attempt in range(RETURN_TRIES):
            go({a: command[a] + APPROACH.get(a, 0.0) for a in AXES}, 0.5, 0.4)
            go(command, 0.5, 0.9)
            error = off()
            worst = max(abs(v) for v in error.values())
            if worst < POSE_OK_RAD or attempt == RETURN_TRIES - 1:
                break
            for a in AXES:  # never far from the pose itself: a wild command could strain the neck
                command[a] = max(float(held[a]) - 0.4, min(float(held[a]) + 0.4, command[a] + 0.7 * error[a]))
        if worst < POSE_OK_RAD:
            with contextlib.suppress(OSError):
                COMMAND.parent.mkdir(parents=True, exist_ok=True)
                COMMAND.write_text(
                    json.dumps({"held": {a: held[a] for a in AXES}, "command": command}) + "\n"
                )
        print(
            f"Reachy's head: on the piano, {math.degrees(worst):.1f} degrees from the recording's pose.",
            flush=True,
        )

    def _report_view(self) -> None:
        """How far the keyboard sits from where it sat in the recording, in the policy's own camera picture.
        Needs an arm that can hand over that picture (the policy arm)."""
        snap = getattr(self.arm, "snap", None)
        if snap is None:
            return
        import cv2
        import keys_keyboard_check as check
        import numpy as np

        picture = snap() if check.REFERENCE.exists() else None
        if picture is None:
            return
        reference = check.gray(cv2.imread(str(check.REFERENCE)))
        seen = [
            (dx, dy) for dx, dy, sure in check.shifts(reference, check.gray(picture)).values() if sure > 0.3
        ]
        if not seen:
            return  # hands or the arm are in the way: nothing to say
        dx, dy = float(np.median([s[0] for s in seen])), float(np.median([s[1] for s in seen]))
        print(
            f"Reachy's head: the keyboard is {dx:+.0f}, {dy:+.0f} px from where the policy learned it.",
            flush=True,
        )

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
