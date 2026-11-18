"""LeRobot camera plugin: an OpenCV camera that only offers large frames, delivered small (type "opencv_scaled").

The Reachy Mini camera has no 640x480 mode: it captures 1920x1080 at 60 fps. This camera opens it in that
mode and shrinks every frame to `width` x `height`, so the dataset stays small and the control loop fast.

    {"type": "opencv_scaled", "index_or_path": 0, "width": 640, "height": 360, "fps": 30,
     "capture_width": 1920, "capture_height": 1080, "capture_fps": 60}

`width`, `height` and `fps` are what the rest of LeRobot sees; the capture_* values are what the camera is
asked for, and they must be a mode it really has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
from numpy.typing import NDArray

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


@CameraConfig.register_subclass("opencv_scaled")
@dataclass
class ScaledOpenCVCameraConfig(OpenCVCameraConfig):
    capture_width: int = 1920
    capture_height: int = 1080
    capture_fps: float = 60.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.width is None or self.height is None:
            raise ValueError("opencv_scaled needs width and height: the size of the delivered frames")


class ScaledOpenCVCamera(OpenCVCamera):
    config: ScaledOpenCVCameraConfig

    def _reset_connection_settings(self) -> None:
        super()._reset_connection_settings()
        # What the camera is asked for; self.width and self.height stay the delivered size.
        self.fps = self.config.capture_fps
        self.capture_width, self.capture_height = self.config.capture_width, self.config.capture_height

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        full = super()._postprocess_image(image)
        return cv2.resize(full, (self.config.width, self.config.height), interpolation=cv2.INTER_AREA)


__all__ = ["ScaledOpenCVCamera", "ScaledOpenCVCameraConfig"]
