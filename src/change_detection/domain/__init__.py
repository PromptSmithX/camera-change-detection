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
from .memory import BaselineObject, Embedding, MemoryObject, ObjectState, embedding_from_value

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
    "BaselineObject",
    "Embedding",
    "MemoryObject",
    "ObjectState",
    "embedding_from_value",
]
