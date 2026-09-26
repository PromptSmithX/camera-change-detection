"""Public event, lifecycle, and snapshot artifact writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from change_detection.domain import EventAction, EventRecord

from .store import EventStore, EventStoreResult


class EventOutputError(RuntimeError):
    """Raised when event artifacts cannot be written."""


class EventArtifactWriter:
    """Write one final public record per event plus inspectable lifecycle logs."""

    def __init__(self, output_dir: str | Path, *, reference_image: Any | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"
        self.lifecycle_path = self.output_dir / "event_lifecycle.jsonl"
        self.snapshots_dir = self.output_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.reference_image = reference_image
        self._lifecycle = self.lifecycle_path.open("w", encoding="utf-8", newline="\n")
        self._snapshots_written: set[str] = set()

    def write_lifecycle(self, result: EventStoreResult) -> None:
        self._lifecycle.write(
            json.dumps(result.action.to_dict(), ensure_ascii=False) + "\n"
        )
        self._lifecycle.flush()

    def write_confirmation_snapshot(self, event: EventRecord, frame_image: Any) -> None:
        if event.event_id in self._snapshots_written:
            return
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - runtime dependency boundary
            raise EventOutputError("Snapshots require OpenCV") from exc
        before = self.reference_image if self.reference_image is not None else frame_image
        before_path = self.snapshots_dir / f"{event.event_id}_before.png"
        after_path = self.snapshots_dir / f"{event.event_id}_after.png"
        if not cv2.imwrite(str(before_path), before):
            raise EventOutputError(f"Could not write event snapshot: {before_path}")
        if not cv2.imwrite(str(after_path), frame_image):
            raise EventOutputError(f"Could not write event snapshot: {after_path}")
        self._snapshots_written.add(event.event_id)

    def finalize(self, store: EventStore) -> None:
        try:
            with self.events_path.open("w", encoding="utf-8", newline="\n") as records:
                for event in store.confirmed_events:
                    records.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        finally:
            self._lifecycle.close()

    def close(self) -> None:
        if not self._lifecycle.closed:
            self._lifecycle.close()
