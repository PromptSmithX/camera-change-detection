"""Runtime event contracts shared by the event engine and output writers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .contracts import BBox, EventType, MovementOutcome


class EventLifecycle(StrEnum):
    """Public lifecycle labels for runtime event records."""

    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"


class EventActionType(StrEnum):
    """Internal actions emitted by the deterministic event engine."""

    CREATED = "created"
    CONFIRMED = "confirmed"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class EventRecord:
    """A public, evaluator-compatible event record.

    ``before_bbox``/``after_bbox`` are the internal canonical names.  The
    serialized form also includes the dataset-compatible aliases
    ``baseline_bbox``/``new_bbox`` and ``from_bbox``/``to_bbox``.
    """

    event_id: str
    event_type: EventType
    object_id: str
    started_at_sec: float
    confirmed_at_sec: float | None = None
    ended_at_sec: float | None = None
    confidence: float = 0.0
    before_bbox: BBox | None = None
    after_bbox: BBox | None = None
    movement_outcome: MovementOutcome | None = None
    lifecycle: EventLifecycle = EventLifecycle.CREATED

    def to_dict(self) -> dict[str, Any]:
        before = self.before_bbox.to_list() if self.before_bbox is not None else None
        after = self.after_bbox.to_list() if self.after_bbox is not None else None
        return {
            "event_id": self.event_id,
            "type": self.event_type.value,
            "object_id": self.object_id,
            "started_at_sec": float(self.started_at_sec),
            "confirmed_at_sec": (
                float(self.confirmed_at_sec) if self.confirmed_at_sec is not None else None
            ),
            "ended_at_sec": float(self.ended_at_sec) if self.ended_at_sec is not None else None,
            "confidence": float(self.confidence),
            "lifecycle": self.lifecycle.value,
            "bbox": after if self.event_type == EventType.FORGOTTEN_OBJECT else None,
            "after_bbox": after,
            "before_bbox": before,
            "baseline_bbox": before,
            "new_bbox": after,
            "from_bbox": before,
            "to_bbox": after,
            "movement_outcome": (
                self.movement_outcome.value if self.movement_outcome is not None else None
            ),
        }


@dataclass(frozen=True, slots=True)
class EventAction:
    """One state transition emitted by ``EventEngine``."""

    action: EventActionType
    event: EventRecord
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "event": self.event.to_dict(),
        }

