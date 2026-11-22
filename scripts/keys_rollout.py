#!/usr/bin/env python
"""`lerobot-rollout`, with two more chat commands for the interactive session: /pause and /goto.

    /pause        stop the policy and hold the arm exactly where it is. /start carries on from there.
    /pause 8      the same, then raise the arm 8 degrees at its second servo (shoulder lift)
    /goto {...} 0.6   pause, then move to a pose (joint -> degrees) in 0.6 s. keys_coach.py points with it.

The stock session only has /reset, which sends the arm back to the pose it had at launch. For the keyboard
that is the wrong place to wait: the policy was trained going from key to key, and its first press after a
trip home is its weakest (do and re land one key to the right). With /pause the arm waits in the air above
the key it just played, which is how almost every training episode begins.

The raise is there because the policy only learned presses: after a note it leaves the finger on the key.
Lifting the shoulder a few degrees takes the finger off, so nothing else gets played while the arm waits.

Takes the same arguments as lerobot-rollout. Nothing in LeRobot itself is changed: the command is added to
the session when it is created.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from lerobot.rollout import interactive
from lerobot.scripts import lerobot_rollout


def pause(controller: Any) -> bool:
    """End the running segment without the trip home: `RolloutController.reset` minus the reset.

    The serve loop then reports "Rollout run ended. Robot is holding position", and start() runs again.
    """
    with controller._control_lock:
        if controller._stopped.is_set() or controller._stop_requested.is_set():
            return False
        controller._start_requested.clear()  # last command wins: cancel a pending start
        controller._ctx.policy.inference.pause()
        controller._segment_stop.set()
        controller._wake.set()
        return True


def raise_arm(
    controller: Any, degrees: float, joint: str = "shoulder_lift.pos", seconds: float = 0.4, fps: int = 50
) -> bool:
    """Turn one joint to lift the fingertip, once the segment has ended. Up is the negative direction of the
    shoulder lift on this arm: the taught "above" poses sit 19 to 41 degrees below their "down" poses."""
    degrees = max(0.0, min(float(degrees), 25.0))
    deadline = time.perf_counter() + 2.0
    while controller.running and time.perf_counter() < deadline:
        time.sleep(0.01)
    if controller.running or not degrees:
        return False
    hardware = controller._ctx.hardware
    robot = hardware.robot_wrapper
    observation = robot.get_observation()
    start = {key: float(observation[key]) for key in hardware.initial_position}
    steps = max(int(seconds * fps), 1)
    for step in range(1, steps + 1):
        robot.send_action({**start, joint: start[joint] - degrees * step / steps})
        time.sleep(1 / fps)
    return True


def goto(controller: Any, pose: dict[str, float], seconds: float, fps: int = 50) -> bool:
    """Move the arm to a pose (joint name -> degrees, e.g. a taught "above" pose) along a smooth curve,
    with the policy paused. For keys_coach.py: pointing at a key, and keys the policy never learned."""
    if controller.running and not pause(controller):
        return False
    deadline = time.perf_counter() + 2.0
    while controller.running and time.perf_counter() < deadline:
        time.sleep(0.01)
    if controller.running:
        return False
    hardware = controller._ctx.hardware
    robot = hardware.robot_wrapper
    observation = robot.get_observation()
    start = {key: float(observation[key]) for key in hardware.initial_position}
    end = {**start, **{k: float(v) for k, v in pose.items() if k in start}}
    steps = max(int(max(seconds, 0.05) * fps), 1)
    t0 = time.perf_counter()
    for step in range(1, steps + 1):
        s = step / steps
        a = s**3 * (10 - 15 * s + 6 * s * s)  # min-jerk: still at both ends
        robot.send_action({k: start[k] + a * (end[k] - start[k]) for k in start})
        time.sleep(max(0.0, t0 + step / fps - time.perf_counter()))
    return True


def snap(controller: Any, path: str, camera: str = "front") -> bool:
    """Save what the policy's camera sees right now as a JPEG, with the policy paused. For keys_coach.py:
    to check that Reachy's head is back on the view the policy learned from."""
    import cv2

    if controller.running:
        return False
    observation = controller._ctx.hardware.robot_wrapper.get_observation()
    image = observation.get(camera)
    if image is None:
        return False
    tmp = f"{path}.tmp.jpg"
    cv2.imwrite(tmp, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    os.replace(tmp, path)  # the reader never sees half a file
    return True


def add_snap_command() -> None:
    """/snap /path/to/view.jpg   -> the camera's picture, saved to that file."""
    original = interactive.InteractiveSession.__init__

    def with_snap(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)

        def command(cmd: Any) -> None:
            path = (cmd.args or "").strip()
            if not path or not snap(self.controller, path):
                self._print("No picture: give a file path, and pause the policy first.")

        self._commands["snap"] = (command, " <file>", "save the camera's picture (policy paused)")

    interactive.InteractiveSession.__init__ = with_snap  # type: ignore[method-assign]


def add_goto_command() -> None:
    """/goto {"shoulder_pan.pos": 3.1, ...} 0.6   -> pause the policy, move there in 0.6 s."""
    original = interactive.InteractiveSession.__init__

    def with_goto(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)

        def command(cmd: Any) -> None:
            text = (cmd.args or "").strip()
            body, _, last = text.rpartition(" ")
            try:
                seconds = float(last)
            except ValueError:
                body, seconds = text, 0.6
            try:
                pose = json.loads(body)
            except json.JSONDecodeError:
                self._print("/goto takes a JSON pose and an optional number of seconds.")
                return
            if not goto(self.controller, pose, seconds):
                self._print("Can't move: the session is busy or stopping.")

        self._commands["goto"] = (command, " <json pose> [seconds]", "pause the policy and move to a pose")

    interactive.InteractiveSession.__init__ = with_goto  # type: ignore[method-assign]


def add_pause_command() -> None:
    original = interactive.InteractiveSession.__init__

    def with_pause(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)

        def command(cmd: Any) -> None:
            if not pause(self.controller):
                self._print("Can't pause: the session is stopping.")
                return
            try:
                degrees = float(cmd.args) if cmd.args else 0.0
            except ValueError:
                self._print(f"/pause takes a number of degrees, not {cmd.args!r}.")
                return
            if degrees and not raise_arm(self.controller, degrees):
                self._print("Paused, but the arm was not raised.")

        self._commands["pause"] = (
            command,
            " [degrees]",
            "stop movement, hold the pose, optionally raise the arm",
        )

    interactive.InteractiveSession.__init__ = with_pause  # type: ignore[method-assign]


if __name__ == "__main__":
    add_pause_command()
    add_goto_command()
    add_snap_command()
    lerobot_rollout.main()
