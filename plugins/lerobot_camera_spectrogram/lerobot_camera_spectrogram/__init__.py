"""LeRobot camera plugin: a microphone as a mel spectrogram camera (type "spectrogram").

Importing this package registers the "spectrogram" camera type. LeRobot's scripts do that
automatically through register_third_party_plugins() once the package is installed.
"""

from .config_spectrogram import SpectrogramCameraConfig
from .spectrogram import SpectrogramCamera

__all__ = ["SpectrogramCamera", "SpectrogramCameraConfig"]
