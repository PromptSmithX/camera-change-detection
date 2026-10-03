"""Event lifecycle storage and duplicate prevention."""

from __future__ import annotations

from dataclasses import dataclass, replace

from change_detection.domain import EventAction, EventActionType, EventRecord


@dataclass(frozen=True, slots=True)
class EventStoreResult:
    """Result of applying one event action."""

    action: EventAction
    public_event: EventRecord | None = None


class EventStore:
    """Keep one active event per type/object and finalized public records."""

    def __init__(self) -> None:
        self._active: dict[tuple[str, str], EventRecord] = {}
        self._confirmed: dict[str, EventRecord] = {}
        self._lifecycle: list[EventAction] = []

    @staticmethod
    def _key(event: EventRecord) -> tuple[str, str]:
        return event.event_type.value, event.object_id

    @property
    def active_events(self) -> tuple[EventRecord, ...]:
        return tuple(sorted(self._active.values(), key=lambda item: item.event_id))

    @property
    def confirmed_events(self) -> tuple[EventRecord, ...]:
        return tuple(
            sorted(self._confirmed.values(), key=lambda item: (item.started_at_sec, item.event_id))
        )

    @property
    def lifecycle_actions(self) -> tuple[EventAction, ...]:
        return tuple(self._lifecycle)

    def reset(self) -> None:
        self._active.clear()
        self._confirmed.clear()
        self._lifecycle.clear()

    def apply(self, action: EventAction) -> EventStoreResult | None:
        """Apply an engine action, suppressing duplicate lifecycle transitions."""

        key = self._key(action.event)
        current = self._active.get(key)
        if action.action == EventActionType.CREATED:
            if current is not None:
                return None
            event = replace(action.event, lifecycle=action.event.lifecycle)
            self._active[key] = event
            stored_action = EventAction(action.action, event, action.reason)
            self._lifecycle.append(stored_action)
            return EventStoreResult(stored_action)

        if action.action == EventActionType.CONFIRMED:
            if current is not None and current.confirmed_at_sec is not None:
                return None
            event = replace(
                current or action.event,
                confirmed_at_sec=action.event.confirmed_at_sec,
                confidence=action.event.confidence,
                before_bbox=action.event.before_bbox,
                after_bbox=action.event.after_bbox,
                movement_outcome=action.event.movement_outcome,
                evidence_sources=action.event.evidence_sources,
                lifecycle=action.event.lifecycle,
            )
            self._active[key] = event
            self._confirmed[event.event_id] = event
            stored_action = EventAction(action.action, event, action.reason)
            self._lifecycle.append(stored_action)
            return EventStoreResult(stored_action, replace(event))

        if action.action == EventActionType.CLOSED:
            if current is None:
                return None
            event = replace(
                current,
                ended_at_sec=action.event.ended_at_sec,
                evidence_sources=action.event.evidence_sources,
                lifecycle=action.event.lifecycle,
            )
            self._active.pop(key, None)
            if event.confirmed_at_sec is not None:
                self._confirmed[event.event_id] = event
            stored_action = EventAction(action.action, event, action.reason)
            self._lifecycle.append(stored_action)
            return EventStoreResult(stored_action)

        if action.action == EventActionType.CANCELLED:
            if current is None:
                # A short-lived candidate can be cancelled before its
                # ``CREATED`` threshold.  Preserve that decision in the
                # lifecycle audit without promoting it to a public event.
                stored_action = EventAction(action.action, action.event, action.reason)
                self._lifecycle.append(stored_action)
                return EventStoreResult(stored_action)
            self._active.pop(key, None)
            stored_action = EventAction(action.action, current, action.reason)
            self._lifecycle.append(stored_action)
            return EventStoreResult(stored_action)

        raise ValueError(f"Unsupported event action: {action.action}")
