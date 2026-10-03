"""Public event, lifecycle, and snapshot artifact writers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from change_detection.domain import EventAction, EventRecord
from change_detection.scene import RegionChangeEvidence

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

    def write_confirmation_snapshot(
        self,
        event: EventRecord,
        frame_image: Any,
        *,
        region_evidence: RegionChangeEvidence | None = None,
    ) -> None:
        if event.event_id in self._snapshots_written:
            return
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - runtime dependency boundary
            raise EventOutputError("Snapshots require OpenCV") from exc
        before = (self.reference_image if self.reference_image is not None else frame_image).copy()
        after = frame_image.copy()
        bbox = event.after_bbox or event.before_bbox
        if bbox is not None:
            for image, color in ((before, (0, 0, 255)), (after, (0, 220, 0))):
                cv2.rectangle(
                    image,
                    (bbox.x1, bbox.y1),
                    (bbox.x2 - 1, bbox.y2 - 1),
                    color,
                    2,
                )
                cv2.putText(
                    image,
                    f"{event.event_id[:12]} oid={event.object_id[:8]}",
                    (bbox.x1, max(16, bbox.y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        before_path = self.snapshots_dir / f"{event.event_id}_before.png"
        after_path = self.snapshots_dir / f"{event.event_id}_after.png"
        if not cv2.imwrite(str(before_path), before):
            raise EventOutputError(f"Could not write event snapshot: {before_path}")
        if not cv2.imwrite(str(after_path), after):
            raise EventOutputError(f"Could not write event snapshot: {after_path}")
        debug_path = self.snapshots_dir / f"{event.event_id}_debug.json"
        debug_path.write_text(
            json.dumps(
                {
                    "event_id": event.event_id,
                    "object_id": event.object_id,
                    "confirmed_at_sec": event.confirmed_at_sec,
                    "bbox": bbox.to_list() if bbox is not None else None,
                    "region_evidence": (
                        region_evidence.to_dict() if region_evidence is not None else None
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
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
