"""Detector model adapters."""

from .yolo import YoloDetector, YoloDependencyError

__all__ = ["YoloDependencyError", "YoloDetector"]
