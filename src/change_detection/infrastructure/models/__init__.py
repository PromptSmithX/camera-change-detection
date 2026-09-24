"""Detector model adapters."""

from .yolo import YoloDetector, YoloDependencyError
from .dinov2 import DinoV2DependencyError, DinoV2Encoder

__all__ = ["DinoV2DependencyError", "DinoV2Encoder", "YoloDependencyError", "YoloDetector"]
