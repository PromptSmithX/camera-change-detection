"""Supervision ByteTrack adapter with domain-only output."""

from __future__ import annotations

from typing import Any, Callable

from change_detection.domain import BBox, Detection, Track
from change_detection.perception.contracts import PerceptionDependencyError


class ByteTrackDependencyError(PerceptionDependencyError):
    """Raised when the optional Supervision ByteTrack runtime is unavailable."""


def _iou(left: BBox, right: BBox) -> float:
    return left.iou(right)


class ByteTrackAdapter:
    """Keep short-term IDs using Supervision's ByteTrack implementation."""

    def __init__(
        self,
        *,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
        tracker_instance: Any | None = None,
        tracker_factory: Callable[..., Any] | None = None,
        detections_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not 0.0 <= float(track_activation_threshold) <= 1.0:
            raise ValueError("track_activation_threshold must be between 0 and 1")
        if not 0.0 <= float(minimum_matching_threshold) <= 1.0:
            raise ValueError("minimum_matching_threshold must be between 0 and 1")
        if int(lost_track_buffer) < 1 or int(minimum_consecutive_frames) < 1:
            raise ValueError("ByteTrack frame settings must be positive")
        self.track_activation_threshold = float(track_activation_threshold)
        self.lost_track_buffer = int(lost_track_buffer)
        self.minimum_matching_threshold = float(minimum_matching_threshold)
        self.minimum_consecutive_frames = int(minimum_consecutive_frames)
        self._tracker_factory = tracker_factory
        self._detections_factory = detections_factory
        self._tracker = tracker_instance or self._create_tracker()

    def _create_tracker(self) -> Any:
        if self._tracker_factory is not None:
            return self._tracker_factory(
                track_activation_threshold=self.track_activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.minimum_matching_threshold,
                minimum_consecutive_frames=self.minimum_consecutive_frames,
            )
        try:
            import supervision as sv
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ByteTrackDependencyError(
                "ByteTrack requires the perception extra; install "
                "with `pip install -e .[perception]`"
            ) from exc
        try:
            return sv.ByteTrack(
                track_activation_threshold=self.track_activation_threshold,
                lost_track_buffer=self.lost_track_buffer,
                minimum_matching_threshold=self.minimum_matching_threshold,
                minimum_consecutive_frames=self.minimum_consecutive_frames,
            )
        except Exception as exc:  # pragma: no cover - dependency API specific
            raise ByteTrackDependencyError(f"Could not create ByteTrack: {exc}") from exc

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "bytetrack",
            "framework": "supervision",
            "track_activation_threshold": self.track_activation_threshold,
            "lost_track_buffer": self.lost_track_buffer,
            "minimum_matching_threshold": self.minimum_matching_threshold,
            "minimum_consecutive_frames": self.minimum_consecutive_frames,
        }

    def reset(self) -> None:
        reset = getattr(self._tracker, "reset", None)
        if callable(reset):
            reset()
        else:
            self._tracker = self._create_tracker()

    def update(self, detections: list[Detection], frame: Any) -> list[Track]:
        del frame  # ByteTrack only consumes detections; frame remains in the protocol.
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - runtime extra
            raise ByteTrackDependencyError("ByteTrack requires NumPy") from exc
        detections_factory = self._detections_factory
        if detections_factory is None:
            try:
                import supervision as sv
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ByteTrackDependencyError(
                    "ByteTrack requires the perception extra; install "
                    "with `pip install -e .[perception]`"
                ) from exc
            detections_factory = sv.Detections

        if detections:
            native = detections_factory(
                xyxy=np.asarray([item.bbox.to_list() for item in detections], dtype=float),
                confidence=np.asarray([item.confidence for item in detections], dtype=float),
                class_id=np.asarray([item.class_id for item in detections], dtype=int),
            )
        else:
            native = detections_factory(
                xyxy=np.empty((0, 4), dtype=float),
                confidence=np.empty((0,), dtype=float),
                class_id=np.empty((0,), dtype=int),
            )
        try:
            tracked = self._tracker.update_with_detections(native)
        except Exception as exc:  # pragma: no cover - dependency API specific
            raise ByteTrackDependencyError(f"ByteTrack update failed: {exc}") from exc

        tracker_ids = getattr(tracked, "tracker_id", None)
        if tracker_ids is None:
            return []
        xyxy = getattr(tracked, "xyxy", [])
        confidence_values = getattr(tracked, "confidence", None)
        class_values = getattr(tracked, "class_id", None)
        result: list[Track] = []
        used_detection_indices: set[int] = set()
        for row, tracker_id in enumerate(list(tracker_ids)):
            if tracker_id is None:
                continue
            coordinates = list(xyxy[row])
            bbox = BBox(
                int(coordinates[0]),
                int(coordinates[1]),
                int(coordinates[2]),
                int(coordinates[3]),
            )
            if bbox.area <= 0:
                continue
            class_id = int(class_values[row]) if class_values is not None else 0
            confidence = (
                float(confidence_values[row])
                if confidence_values is not None
                else 0.0
            )
            best_index: int | None = None
            best_iou = 0.0
            for index, detection in enumerate(detections):
                if index in used_detection_indices or detection.class_id != class_id:
                    continue
                score = _iou(bbox, detection.bbox)
                if score > best_iou:
                    best_iou = score
                    best_index = index
            if best_index is not None and best_iou >= 0.5:
                used_detection_indices.add(best_index)
                source = detections[best_index]
                class_name = source.class_name
                confidence = source.confidence
            else:
                class_name = str(class_id)
                best_index = None
            result.append(
                Track(
                    tracker_id=int(tracker_id),
                    bbox=bbox,
                    confidence=max(0.0, min(1.0, confidence)),
                    class_id=class_id,
                    class_name=class_name,
                    detection_index=best_index,
                )
            )
        return result
