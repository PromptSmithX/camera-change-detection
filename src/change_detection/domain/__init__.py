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

__all__ = [
    "AnnotationStatus",
    "BBox",
    "EventAnnotation",
    "EventType",
    "MovementOutcome",
    "SampleAnnotation",
    "Split",
    "can_transition_quality",
]
