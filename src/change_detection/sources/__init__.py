"""Frame source adapters for files, image sequences, and webcams."""

from .base import (
    BaseFrameSource,
    FrameSource,
    SourceClosedError,
    SourceError,
    SourceOpenError,
    SourceReadError,
    SourceSeekError,
)
from .image_sequence import ImageSequenceSource
from .video import VideoSource
from .webcam import WebcamSource

__all__ = [
    "BaseFrameSource",
    "FrameSource",
    "ImageSequenceSource",
    "SourceClosedError",
    "SourceError",
    "SourceOpenError",
    "SourceReadError",
    "SourceSeekError",
    "VideoSource",
    "WebcamSource",
]
