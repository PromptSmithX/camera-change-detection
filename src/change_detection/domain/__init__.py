"""Framework-independent domain contracts."""

from .contracts import (
    AnnotationStatus,
    BBox,
    EventAnnotation,
    EventType,
    MovementOutcome,
    SampleAnnotation,
    Split,
    can_transition_quality,
)
from .frames import FrameContext, SourceMetadata

__all__ = [
    "AnnotationStatus",
    "BBox",
    "EventAnnotation",
    "EventType",
    "MovementOutcome",
    "SampleAnnotation",
    "Split",
    "can_transition_quality",
    "FrameContext",
    "SourceMetadata",
]
