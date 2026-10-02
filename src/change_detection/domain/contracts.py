"""Pure-Python contracts used by dataset and evaluation code.

The domain layer intentionally has no dependency on OpenCV, torch, or a model
framework.  Media adapters may convert their native objects into these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence


SCENE_CHANGE_CLASS_ID = 1000


class EventType(StrEnum):
    FORGOTTEN_OBJECT = "FORGOTTEN_OBJECT"
    MOVED_OBJECT = "MOVED_OBJECT"


class MovementOutcome(StrEnum):
    """How a moved baseline object is observed after leaving its baseline."""

    RELOCATED = "relocated"
    LEFT_SCENE = "left_scene"


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


@dataclass(frozen=True, slots=True)
class Point:
    """A framework-independent 2D point in source-image coordinates."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not isfinite(float(self.x)) or not isfinite(float(self.y)):
            raise ValueError("Point coordinates must be finite")

    def to_list(self) -> list[float]:
        return [float(self.x), float(self.y)]


@dataclass(frozen=True, slots=True)
class Detection:
    """One model-independent detector result.

    ``mask`` is intentionally opaque.  M2 only uses bounding boxes, while a
    future segmentation adapter may populate it without changing the public
    detection fields.
    """

    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    mask: Any = field(default=None, repr=False, compare=False)
    proposal_source: str = "yolo"
    reference_change_score: float | None = None
    event_candidate: bool = True
    change_bbox: BBox | None = None
    evidence_sources: tuple[str, ...] = ()
    semantic_score: float | None = None
    change_score: float | None = None
    alignment_score: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Detection confidence must be finite and between 0 and 1")
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ValueError("Detection class_id must be a non-negative integer")
        if not str(self.class_name).strip():
            raise ValueError("Detection class_name must not be empty")
        if self.bbox.area <= 0:
            raise ValueError("Detection bbox must have positive area")
        if not str(self.proposal_source).strip():
            raise ValueError("Detection proposal_source must not be empty")
        if self.reference_change_score is not None and (
            not isfinite(float(self.reference_change_score))
            or not 0.0 <= float(self.reference_change_score) <= 1.0
        ):
            raise ValueError("Detection reference_change_score must be between 0 and 1")
        if not isinstance(self.event_candidate, bool):
            raise ValueError("Detection event_candidate must be a boolean")
        if self.change_bbox is not None and self.change_bbox.area <= 0:
            raise ValueError("Detection change_bbox must have positive area")
        sources = tuple(dict.fromkeys(str(item).strip() for item in self.evidence_sources))
        if not sources:
            sources = (str(self.proposal_source).strip(),)
        if any(not item for item in sources):
            raise ValueError("Detection evidence_sources must not contain empty values")
        object.__setattr__(self, "evidence_sources", sources)
        for name, value in (
            ("semantic_score", self.semantic_score),
            ("change_score", self.change_score),
            ("alignment_score", self.alignment_score),
        ):
            if value is not None and (
                not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"Detection {name} must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": self.bbox.to_list(),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "proposal_source": self.proposal_source,
            "reference_change_score": self.reference_change_score,
            "event_candidate": self.event_candidate,
            "change_bbox": self.change_bbox.to_list() if self.change_bbox is not None else None,
            "evidence_sources": list(self.evidence_sources),
            "semantic_score": self.semantic_score,
            "change_score": self.change_score,
            "alignment_score": self.alignment_score,
        }


@dataclass(frozen=True, slots=True)
class Track:
    """Short-term tracker output.

    ``tracker_id`` is deliberately scoped to one runtime stream.  It must not
    be used as the persistent identity required by M3 and later milestones.
    """

    tracker_id: int
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    detection_index: int | None = None
    age_frames: int = 1
    hits: int = 1
    time_since_update: int = 0
    is_confirmed: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.tracker_id, bool) or int(self.tracker_id) < 0:
            raise ValueError("Track tracker_id must be a non-negative integer")
        if not isfinite(float(self.confidence)) or not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Track confidence must be finite and between 0 and 1")
        if isinstance(self.class_id, bool) or int(self.class_id) < 0:
            raise ValueError("Track class_id must be a non-negative integer")
        if not str(self.class_name).strip():
            raise ValueError("Track class_name must not be empty")
        if self.bbox.area <= 0:
            raise ValueError("Track bbox must have positive area")
        if self.detection_index is not None and (
            isinstance(self.detection_index, bool) or int(self.detection_index) < 0
        ):
            raise ValueError("Track detection_index must be a non-negative integer")
        if self.age_frames < 1 or self.hits < 1 or self.time_since_update < 0:
            raise ValueError("Track frame counters are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracker_id": int(self.tracker_id),
            "bbox": self.bbox.to_list(),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
            "detection_index": self.detection_index,
            "age_frames": int(self.age_frames),
            "hits": int(self.hits),
            "time_since_update": int(self.time_since_update),
            "is_confirmed": bool(self.is_confirmed),
        }


@dataclass(slots=True)
class Observation:
    """A source-timestamped, ROI-valid short-term observation."""

    frame_index: int
    timestamp_sec: float
    bbox: BBox
    centroid: Point
    detector_class: str
    detector_class_id: int
    detector_confidence: float
    tracker_id: int | None
    embedding: Any = field(default=None, repr=False, compare=False)
    occluded: bool = False
    proposal_source: str = "yolo"
    reference_change_score: float | None = None
    event_candidate: bool = True
    change_bbox: BBox | None = None
    evidence_sources: tuple[str, ...] = ()
    semantic_score: float | None = None
    change_score: float | None = None
    alignment_score: float | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("Observation frame_index must be non-negative")
        if not isfinite(float(self.timestamp_sec)) or float(self.timestamp_sec) < 0:
            raise ValueError("Observation timestamp_sec must be finite and non-negative")
        if not str(self.detector_class).strip():
            raise ValueError("Observation detector_class must not be empty")
        if isinstance(self.detector_class_id, bool) or int(self.detector_class_id) < 0:
            raise ValueError("Observation detector_class_id must be non-negative")
        if not isfinite(float(self.detector_confidence)) or not 0.0 <= float(self.detector_confidence) <= 1.0:
            raise ValueError("Observation detector_confidence must be between 0 and 1")
        if self.tracker_id is not None and (
            isinstance(self.tracker_id, bool) or int(self.tracker_id) < 0
        ):
            raise ValueError("Observation tracker_id must be a non-negative integer")
        if not str(self.proposal_source).strip():
            raise ValueError("Observation proposal_source must not be empty")
        if self.reference_change_score is not None and (
            not isfinite(float(self.reference_change_score))
            or not 0.0 <= float(self.reference_change_score) <= 1.0
        ):
            raise ValueError("Observation reference_change_score must be between 0 and 1")
        if not isinstance(self.event_candidate, bool):
            raise ValueError("Observation event_candidate must be a boolean")
        if self.change_bbox is not None and self.change_bbox.area <= 0:
            raise ValueError("Observation change_bbox must have positive area")
        sources = tuple(dict.fromkeys(str(item).strip() for item in self.evidence_sources))
        if not sources:
            sources = (str(self.proposal_source).strip(),)
        if any(not item for item in sources):
            raise ValueError("Observation evidence_sources must not contain empty values")
        self.evidence_sources = sources
        for name, value in (
            ("semantic_score", self.semantic_score),
            ("change_score", self.change_score),
            ("alignment_score", self.alignment_score),
        ):
            if value is not None and (
                not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"Observation {name} must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "timestamp_sec": float(self.timestamp_sec),
            "bbox": self.bbox.to_list(),
            "centroid": self.centroid.to_list(),
            "detector_class": self.detector_class,
            "detector_class_id": int(self.detector_class_id),
            "detector_confidence": float(self.detector_confidence),
            "tracker_id": self.tracker_id,
            "occluded": bool(self.occluded),
            "proposal_source": self.proposal_source,
            "reference_change_score": self.reference_change_score,
            "event_candidate": self.event_candidate,
            "change_bbox": self.change_bbox.to_list() if self.change_bbox is not None else None,
            "evidence_sources": list(self.evidence_sources),
            "semantic_score": self.semantic_score,
            "change_score": self.change_score,
            "alignment_score": self.alignment_score,
        }


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
    movement_outcome: MovementOutcome | None = None
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
        event_type = EventType(str(raw_type))
        raw_event_id = data.get("event_id", data.get("id"))
        raw_object_id = data.get("object_id")
        new_bbox = BBox.from_value(data.get("new_bbox", data.get("to_bbox")))
        raw_outcome = data.get("movement_outcome")
        movement_outcome = (
            MovementOutcome(str(raw_outcome))
            if raw_outcome not in (None, "")
            else (
                MovementOutcome.RELOCATED
                if event_type == EventType.MOVED_OBJECT and new_bbox is not None
                else None
            )
        )
        return cls(
            event_id="" if raw_event_id in (None, "") else str(raw_event_id),
            event_type=event_type,
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
            new_bbox=new_bbox,
            movement_outcome=movement_outcome,
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
            "movement_outcome": self.movement_outcome.value if self.movement_outcome else None,
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
