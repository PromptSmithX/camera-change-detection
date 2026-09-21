"""Pure-Python contracts used by dataset and evaluation code.

The domain layer intentionally has no dependency on OpenCV, torch, or a model
framework.  Media adapters may convert their native objects into these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


class EventType(StrEnum):
    FORGOTTEN_OBJECT = "FORGOTTEN_OBJECT"
    MOVED_OBJECT = "MOVED_OBJECT"


class Split(StrEnum):
    VALIDATION = "validation"
    TEST = "test"


class AnnotationStatus(StrEnum):
    PROVISIONAL = "provisional"
    VERIFIED = "verified"
    EXCLUDED = "excluded"


def can_transition_quality(old: AnnotationStatus | str, new: AnnotationStatus | str) -> bool:
    """Return whether a review status transition is allowed within a version."""

    previous = AnnotationStatus(old)
    target = AnnotationStatus(new)
    allowed = {
        AnnotationStatus.PROVISIONAL: {
            AnnotationStatus.PROVISIONAL,
            AnnotationStatus.VERIFIED,
            AnnotationStatus.EXCLUDED,
        },
        AnnotationStatus.VERIFIED: {AnnotationStatus.VERIFIED, AnnotationStatus.EXCLUDED},
        AnnotationStatus.EXCLUDED: {
            AnnotationStatus.EXCLUDED,
            AnnotationStatus.PROVISIONAL,
            AnnotationStatus.VERIFIED,
        },
    }
    return target in allowed[previous]


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"Expected a finite number, got {value!r}")
    return result


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


@dataclass(frozen=True, slots=True)
class BBox:
    """Integer half-open image rectangle ``[x1, y1, x2, y2]``."""

    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_value(cls, value: Sequence[Any] | None) -> "BBox | None":
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError(f"BBox must have four values, got {value!r}")
        return cls(*(int(item) for item in value))

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def is_valid_for(self, width: int, height: int) -> bool:
        return (
            0 <= self.x1 < self.x2 <= width
            and 0 <= self.y1 < self.y2 <= height
        )

    def is_inside(self, other: "BBox") -> bool:
        return (
            other.x1 <= self.x1
            and other.y1 <= self.y1
            and self.x2 <= other.x2
            and self.y2 <= other.y2
        )

    def iou(self, other: "BBox") -> float:
        ix1 = max(self.x1, other.x1)
        iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2)
        iy2 = min(self.y2, other.y2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0


def _intervals(values: Iterable[Sequence[Any]] | None) -> tuple[tuple[float, float], ...]:
    if values is None:
        return ()
    result: list[tuple[float, float]] = []
    for value in values:
        if len(value) != 2:
            raise ValueError(f"An interval must have two values, got {value!r}")
        start, end = float(value[0]), float(value[1])
        if not isfinite(start) or not isfinite(end) or start < 0 or end < start:
            raise ValueError(f"Invalid interval: {value!r}")
        result.append((start, end))
    return tuple(result)


@dataclass(slots=True)
class EventAnnotation:
    event_id: str
    event_type: EventType
    object_id: str
    start_time_sec: float
    confirmation_time_sec: float | None
    end_time_sec: float | None
    bbox: BBox | None = None
    baseline_bbox: BBox | None = None
    new_bbox: BBox | None = None
    object_class: str | None = None
    start_frame: int | None = None
    confirmation_frame: int | None = None
    end_frame: int | None = None
    occlusion_intervals: tuple[tuple[float, float], ...] = ()
    lighting_change_intervals: tuple[tuple[float, float], ...] = ()
    notes: str | None = None
    difficult: bool = False
    ambiguous: bool = False
    quality_status: AnnotationStatus = AnnotationStatus.PROVISIONAL

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EventAnnotation":
        raw_type = data.get("type", data.get("event_type"))
        if raw_type is None:
            raise ValueError("Event is missing type")
        raw_event_id = data.get("event_id", data.get("id"))
        raw_object_id = data.get("object_id")
        return cls(
            event_id="" if raw_event_id in (None, "") else str(raw_event_id),
            event_type=EventType(str(raw_type)),
            object_id="" if raw_object_id in (None, "") else str(raw_object_id),
            start_time_sec=float(data.get("start_time_sec", data.get("started_at_sec", 0.0))),
            confirmation_time_sec=_as_optional_float(
                data.get("confirmation_time_sec", data.get("confirmed_at_sec"))
            ),
            end_time_sec=_as_optional_float(data.get("end_time_sec", data.get("ended_at_sec"))),
            bbox=BBox.from_value(data.get("bbox", data.get("after_bbox"))),
            baseline_bbox=BBox.from_value(
                data.get("baseline_bbox", data.get("before_bbox", data.get("from_bbox")))
            ),
            new_bbox=BBox.from_value(data.get("new_bbox", data.get("to_bbox"))),
            object_class=data.get("object_class", data.get("class")),
            start_frame=_as_optional_int(data.get("start_frame")),
            confirmation_frame=_as_optional_int(data.get("confirmation_frame")),
            end_frame=_as_optional_int(data.get("end_frame")),
            occlusion_intervals=_intervals(data.get("occlusion_intervals")),
            lighting_change_intervals=_intervals(data.get("lighting_change_intervals")),
            notes=data.get("notes"),
            difficult=bool(data.get("difficult", False)),
            ambiguous=bool(data.get("ambiguous", False)),
            quality_status=AnnotationStatus(
                data.get("quality_status", AnnotationStatus.PROVISIONAL.value)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "event_id": self.event_id,
            "type": self.event_type.value,
            "object_id": self.object_id,
            "start_time_sec": self.start_time_sec,
            "confirmation_time_sec": self.confirmation_time_sec,
            "end_time_sec": self.end_time_sec,
            "bbox": self.bbox.to_list() if self.bbox else None,
            "baseline_bbox": self.baseline_bbox.to_list() if self.baseline_bbox else None,
            "new_bbox": self.new_bbox.to_list() if self.new_bbox else None,
            "object_class": self.object_class,
            "start_frame": self.start_frame,
            "confirmation_frame": self.confirmation_frame,
            "end_frame": self.end_frame,
            "occlusion_intervals": [list(item) for item in self.occlusion_intervals],
            "lighting_change_intervals": [list(item) for item in self.lighting_change_intervals],
            "notes": self.notes,
            "difficult": self.difficult,
            "ambiguous": self.ambiguous,
            "quality_status": self.quality_status.value,
        }
        return result


@dataclass(slots=True)
class SampleAnnotation:
    video_id: str
    camera_id: str
    path: str
    annotation_path: str
    roi_path: str
    split: Split
    scenario_tags: list[str]
    fps: float
    width: int
    height: int
    frame_count: int
    duration_sec: float
    events: list[EventAnnotation] = field(default_factory=list)
    roi_bbox: BBox | None = None
    reference_frame_range: tuple[int, int] | None = None
    quality_status: AnnotationStatus = AnnotationStatus.PROVISIONAL
    reviewer: str | None = None
    review_notes: str | None = None
    group_id: str | None = None
    media_type: str = "video"
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    source_video_id: str | None = None
    frame_pattern: str | None = None

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        manifest_sample: Mapping[str, Any] | None = None,
    ) -> "SampleAnnotation":
        sample = manifest_sample or data
        roi = data.get("roi", {})
        reference = data.get("reference", {})
        reference_range = reference.get("frame_range")
        if reference_range is not None:
            reference_range = (int(reference_range[0]), int(reference_range[1]))
        return cls(
            video_id=str(data.get("video_id", data.get("id", sample.get("id", "")))),
            camera_id=str(data.get("camera_id", sample.get("camera_id", ""))),
            path=str(data.get("path", data.get("video_path", sample.get("path", "")))),
            annotation_path=str(data.get("annotation_path", sample.get("annotation_path", ""))),
            roi_path=str(data.get("roi_path", sample.get("roi_path", ""))),
            split=Split(str(data.get("split", sample.get("split", "")))),
            scenario_tags=list(data.get("scenario_tags", sample.get("scenario_tags", []))),
            fps=float(data["fps"]),
            width=int(data["width"]),
            height=int(data["height"]),
            frame_count=int(data.get("frame_count", data.get("num_frames", 0))),
            duration_sec=float(data.get("duration_sec", 0.0)),
            events=[EventAnnotation.from_dict(item) for item in data.get("events", [])],
            roi_bbox=BBox.from_value(roi.get("bbox", data.get("roi_bbox"))),
            reference_frame_range=reference_range,
            quality_status=AnnotationStatus(
                data.get("quality_status", sample.get("quality_status", AnnotationStatus.PROVISIONAL.value))
            ),
            reviewer=data.get("reviewer", sample.get("reviewer")),
            review_notes=data.get("review_notes", sample.get("review_notes")),
            group_id=data.get("group_id", sample.get("group_id")),
            media_type=str(data.get("media_type", sample.get("media_type", "video"))),
            warnings=list(data.get("warnings", sample.get("warnings", []))),
            provenance=dict(data.get("provenance", sample.get("provenance", {}))),
            source_video_id=data.get("source_video_id", sample.get("source_video_id")),
            frame_pattern=data.get("frame_pattern", sample.get("frame_pattern")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "camera_id": self.camera_id,
            "path": self.path,
            "annotation_path": self.annotation_path,
            "roi_path": self.roi_path,
            "split": self.split.value,
            "scenario_tags": list(self.scenario_tags),
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "duration_sec": self.duration_sec,
            "media_type": self.media_type,
            "events": [event.to_dict() for event in self.events],
            "roi": {"bbox": self.roi_bbox.to_list() if self.roi_bbox else None},
            "reference": {
                "frame_range": list(self.reference_frame_range)
                if self.reference_frame_range
                else None
            },
            "quality_status": self.quality_status.value,
            "reviewer": self.reviewer,
            "review_notes": self.review_notes,
            "group_id": self.group_id,
            "warnings": list(self.warnings),
            "provenance": self.provenance,
            "source_video_id": self.source_video_id,
            "frame_pattern": self.frame_pattern,
        }
