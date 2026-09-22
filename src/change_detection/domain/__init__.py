"""Framework-independent domain contracts."""

from .contracts import (
    AnnotationStatus,
    BBox,
    Detection,
    EventAnnotation,
    EventType,
    MovementOutcome,
    Observation,
    Point,
    SampleAnnotation,
    Split,
    Track,
    can_transition_quality,
)
from .frames import FrameContext, SourceMetadata

__all__ = [
    "AnnotationStatus",
    "BBox",
    "Detection",
    "EventAnnotation",
    "EventType",
    "MovementOutcome",
    "Observation",
    "Point",
    "SampleAnnotation",
    "Split",
    "Track",
    "can_transition_quality",
    "FrameContext",
    "SourceMetadata",
]
