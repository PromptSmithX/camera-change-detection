"""Model-independent configuration contracts and loading helpers."""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class ConfigValidationError(ValueError):
    """Raised when a system configuration violates the public contract."""


def _finite_number(value: Any, *, name: str, minimum: float = 0.0, maximum: float | None = None) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{name} must be a number") from exc
    if not math.isfinite(number) or number < minimum or (maximum is not None and number > maximum):
        upper = f" and <= {maximum}" if maximum is not None else ""
        raise ConfigValidationError(f"{name} must be finite, >= {minimum}{upper}")
    return number


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{name} must be an object")
    return value


def _reject_unknown(value: Mapping[str, Any], allowed: set[str], *, name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigValidationError(f"Unknown {name} field(s): {', '.join(unknown)}")


@dataclass(frozen=True, slots=True)
class EventTimingConfig:
    candidate_seconds: float = 1.0
    confirm_seconds: float = 4.0
    disappear_grace_seconds: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, name: str) -> "EventTimingConfig":
        _reject_unknown(
            value,
            {"candidate_seconds", "confirm_seconds", "disappear_grace_seconds"},
            name=name,
        )
        candidate = _finite_number(value.get("candidate_seconds", 1.0), name=f"{name}.candidate_seconds")
        confirm = _finite_number(value.get("confirm_seconds", 4.0), name=f"{name}.confirm_seconds")
        grace = _finite_number(value.get("disappear_grace_seconds", 1.0), name=f"{name}.disappear_grace_seconds")
        if confirm < candidate:
            raise ConfigValidationError(f"{name}.confirm_seconds must be >= candidate_seconds")
        return cls(candidate, confirm, grace)


@dataclass(frozen=True, slots=True)
class ForgottenEventConfig:
    """Temporal and stability guards for ``FORGOTTEN_OBJECT``."""

    candidate_seconds: float = 1.0
    confirm_seconds: float = 4.0
    disappear_grace_seconds: float = 1.0
    min_confidence: float = 0.35
    max_centroid_jitter_ratio: float = 0.03
    max_area_change_ratio: float = 0.35
    resolution_confirm_seconds: float = 5.0
    occlusion_overlap_threshold: float = 0.20
    reid_iou_threshold: float = 0.50

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, name: str) -> "ForgottenEventConfig":
        _reject_unknown(
            value,
            {
                "candidate_seconds",
                "confirm_seconds",
                "disappear_grace_seconds",
                "min_confidence",
                "max_centroid_jitter_ratio",
                "max_area_change_ratio",
                "resolution_confirm_seconds",
                "occlusion_overlap_threshold",
                "reid_iou_threshold",
            },
            name=name,
        )
        candidate = _finite_number(value.get("candidate_seconds", 1.0), name=f"{name}.candidate_seconds")
        confirm = _finite_number(value.get("confirm_seconds", 4.0), name=f"{name}.confirm_seconds")
        grace = _finite_number(
            value.get("disappear_grace_seconds", 1.0),
            name=f"{name}.disappear_grace_seconds",
        )
        if confirm < candidate:
            raise ConfigValidationError(f"{name}.confirm_seconds must be >= candidate_seconds")
        return cls(
            candidate_seconds=candidate,
            confirm_seconds=confirm,
            disappear_grace_seconds=grace,
            min_confidence=_finite_number(
                value.get("min_confidence", 0.35),
                name=f"{name}.min_confidence",
                maximum=1.0,
            ),
            max_centroid_jitter_ratio=_finite_number(
                value.get("max_centroid_jitter_ratio", 0.03),
                name=f"{name}.max_centroid_jitter_ratio",
                maximum=1.0,
            ),
            max_area_change_ratio=_finite_number(
                value.get("max_area_change_ratio", 0.35),
                name=f"{name}.max_area_change_ratio",
                maximum=1.0,
            ),
            resolution_confirm_seconds=_finite_number(
                value.get("resolution_confirm_seconds", 5.0),
                name=f"{name}.resolution_confirm_seconds",
                minimum=0.000001,
            ),
            occlusion_overlap_threshold=_finite_number(
                value.get("occlusion_overlap_threshold", 0.20),
                name=f"{name}.occlusion_overlap_threshold",
                maximum=1.0,
            ),
            reid_iou_threshold=_finite_number(
                value.get("reid_iou_threshold", 0.50),
                name=f"{name}.reid_iou_threshold",
                maximum=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class MovedEventConfig:
    """Identity, displacement, and departure guards for ``MOVED_OBJECT``."""

    missing_grace_seconds: float = 1.0
    confirm_seconds: float = 2.0
    min_identity_score: float = 0.75
    min_displacement_ratio: float = 0.05
    old_location_iou_threshold: float = 0.5
    egress_edge_ratio: float = 0.10
    min_outward_speed_px_per_sec: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, name: str) -> "MovedEventConfig":
        _reject_unknown(
            value,
            {
                "missing_grace_seconds",
                "confirm_seconds",
                "min_identity_score",
                "min_displacement_ratio",
                "old_location_iou_threshold",
                "egress_edge_ratio",
                "min_outward_speed_px_per_sec",
            },
            name=name,
        )
        return cls(
            missing_grace_seconds=_finite_number(
                value.get("missing_grace_seconds", 1.0),
                name=f"{name}.missing_grace_seconds",
                minimum=0.000001,
            ),
            confirm_seconds=_finite_number(
                value.get("confirm_seconds", 2.0),
                name=f"{name}.confirm_seconds",
                minimum=0.000001,
            ),
            min_identity_score=_finite_number(
                value.get("min_identity_score", 0.75),
                name=f"{name}.min_identity_score",
                maximum=1.0,
            ),
            min_displacement_ratio=_finite_number(
                value.get("min_displacement_ratio", 0.05),
                name=f"{name}.min_displacement_ratio",
                maximum=1.0,
            ),
            old_location_iou_threshold=_finite_number(
                value.get("old_location_iou_threshold", 0.5),
                name=f"{name}.old_location_iou_threshold",
                maximum=1.0,
            ),
            egress_edge_ratio=_finite_number(
                value.get("egress_edge_ratio", 0.10),
                name=f"{name}.egress_edge_ratio",
                maximum=1.0,
            ),
            min_outward_speed_px_per_sec=_finite_number(
                value.get("min_outward_speed_px_per_sec", 1.0),
                name=f"{name}.min_outward_speed_px_per_sec",
                minimum=0.000001,
            ),
        )


@dataclass(frozen=True, slots=True)
class EvaluationThresholdConfig:
    event_boundary_tolerance_seconds: float = 1.0
    forgotten_iou_threshold: float = 0.3
    moved_iou_threshold: float = 0.3
    require_prediction_object_id: bool = False
    latency_deadlines_seconds: tuple[float, ...] = (3.0, 5.0, 10.0)
    # Deprecated compatibility fields. The correctness evaluator no longer
    # uses annotation/prediction start-time proximity for event matching.
    start_tolerance_seconds: float = 3.0
    min_temporal_overlap: float = 0.1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationThresholdConfig":
        _reject_unknown(
            value,
            {
                "event_boundary_tolerance_seconds",
                "start_tolerance_seconds",
                "min_temporal_overlap",
                "forgotten_iou_threshold",
                "moved_iou_threshold",
                "require_prediction_object_id",
                "latency_deadlines_seconds",
            },
            name="evaluation",
        )
        require_id = value.get("require_prediction_object_id", False)
        if not isinstance(require_id, bool):
            raise ConfigValidationError("evaluation.require_prediction_object_id must be boolean")
        raw_deadlines = value.get("latency_deadlines_seconds", (3.0, 5.0, 10.0))
        if not isinstance(raw_deadlines, (list, tuple)) or not raw_deadlines:
            raise ConfigValidationError(
                "evaluation.latency_deadlines_seconds must be a non-empty array"
            )
        deadlines = tuple(
            sorted(
                {
                    _finite_number(
                        item,
                        name="evaluation.latency_deadlines_seconds[]",
                    )
                    for item in raw_deadlines
                }
            )
        )
        return cls(
            event_boundary_tolerance_seconds=_finite_number(
                value.get("event_boundary_tolerance_seconds", 1.0),
                name="evaluation.event_boundary_tolerance_seconds",
            ),
            forgotten_iou_threshold=_finite_number(
                value.get("forgotten_iou_threshold", 0.3),
                name="evaluation.forgotten_iou_threshold",
                maximum=1.0,
            ),
            moved_iou_threshold=_finite_number(
                value.get("moved_iou_threshold", 0.3),
                name="evaluation.moved_iou_threshold",
                maximum=1.0,
            ),
            require_prediction_object_id=require_id,
            latency_deadlines_seconds=deadlines,
            start_tolerance_seconds=_finite_number(
                value.get("start_tolerance_seconds", 3.0),
                name="evaluation.start_tolerance_seconds",
            ),
            min_temporal_overlap=_finite_number(
                value.get("min_temporal_overlap", 0.1),
                name="evaluation.min_temporal_overlap",
                maximum=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class SourceConfig:
    """Configuration needed to construct an M1 frame source."""

    type: str = "video"
    path: str | None = None
    source_id: str | None = None
    frame_pattern: str = "*"
    fps: float | None = None
    device_index: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceConfig":
        if not value:
            return cls()
        _reject_unknown(
            value,
            {"type", "path", "source_id", "frame_pattern", "fps", "device_index"},
            name="source",
        )
        source_type = str(value.get("type", "video")).strip()
        if source_type not in {"video", "image_sequence", "webcam"}:
            raise ConfigValidationError(
                "source.type must be one of: video, image_sequence, webcam"
            )
        path_value = value.get("path")
        path = str(path_value).strip() if path_value not in (None, "") else None
        if source_type in {"video", "image_sequence"} and not path:
            raise ConfigValidationError(f"source.path is required for {source_type}")
        frame_pattern = str(value.get("frame_pattern", "*")).strip()
        if source_type == "image_sequence" and not frame_pattern:
            raise ConfigValidationError("source.frame_pattern must not be empty")
        fps_value = value.get("fps")
        fps = (
            _finite_number(fps_value, name="source.fps", minimum=0.000001)
            if fps_value not in (None, "")
            else None
        )
        if source_type == "image_sequence" and fps is None:
            raise ConfigValidationError("source.fps is required for image_sequence")
        device_index = value.get("device_index", 0)
        if isinstance(device_index, bool):
            raise ConfigValidationError("source.device_index must be a non-negative integer")
        try:
            device_index = int(device_index)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                "source.device_index must be a non-negative integer"
            ) from exc
        if device_index < 0:
            raise ConfigValidationError("source.device_index must be a non-negative integer")
        source_id = value.get("source_id")
        source_id = str(source_id).strip() if source_id not in (None, "") else None
        return cls(source_type, path, source_id, frame_pattern, fps, device_index)


@dataclass(frozen=True, slots=True)
class ROIConfig:
    """M1 ROI input; coordinates and path are mutually exclusive."""

    type: str = "bbox"
    coordinates: tuple[int, int, int, int] | None = None
    path: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ROIConfig":
        _reject_unknown(value, {"type", "coordinates", "path"}, name="roi")
        roi_type = str(value.get("type", "bbox")).strip()
        if roi_type != "bbox":
            raise ConfigValidationError("roi.type must be 'bbox' in M1")
        raw_coordinates = value.get("coordinates")
        coordinates: tuple[int, int, int, int] | None = None
        if raw_coordinates is not None:
            if not isinstance(raw_coordinates, (list, tuple)) or len(raw_coordinates) != 4:
                raise ConfigValidationError("roi.coordinates must contain four integers")
            try:
                coordinates = tuple(int(item) for item in raw_coordinates)  # type: ignore[assignment]
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError("roi.coordinates must contain four integers") from exc
        path_value = value.get("path")
        path = str(path_value).strip() if path_value not in (None, "") else None
        if coordinates is not None and path is not None:
            raise ConfigValidationError("roi.coordinates and roi.path are mutually exclusive")
        if coordinates is not None and (
            coordinates[0] == coordinates[2] or coordinates[1] == coordinates[3]
        ):
            raise ConfigValidationError("roi.coordinates must define a non-empty box")
        return cls(roi_type, coordinates, path)


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    sample_count: int = 30
    sample_interval_seconds: float = 0.2
    start_frame: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CalibrationConfig":
        _reject_unknown(
            value,
            {"sample_count", "sample_interval_seconds", "start_frame"},
            name="calibration",
        )
        sample_count = value.get("sample_count", 30)
        start_frame = value.get("start_frame", 0)
        if isinstance(sample_count, bool) or isinstance(start_frame, bool):
            raise ConfigValidationError("calibration frame values must be integers")
        try:
            sample_count = int(sample_count)
            start_frame = int(start_frame)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError("calibration frame values must be integers") from exc
        if sample_count < 2:
            raise ConfigValidationError("calibration.sample_count must be >= 2")
        if start_frame < 0:
            raise ConfigValidationError("calibration.start_frame must be >= 0")
        interval = _finite_number(
            value.get("sample_interval_seconds", 0.2),
            name="calibration.sample_interval_seconds",
            minimum=0.000001,
        )
        return cls(sample_count, interval, start_frame)


@dataclass(frozen=True, slots=True)
class StabilityConfig:
    resize_width: int = 160
    resize_height: int = 90
    max_change_ratio: float = 0.02
    required_stable_fraction: float = 0.9

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StabilityConfig":
        _reject_unknown(
            value,
            {"resize_width", "resize_height", "max_change_ratio", "required_stable_fraction"},
            name="stability",
        )
        width = value.get("resize_width", 160)
        height = value.get("resize_height", 90)
        if isinstance(width, bool) or isinstance(height, bool):
            raise ConfigValidationError("stability resize dimensions must be integers")
        try:
            width = int(width)
            height = int(height)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError("stability resize dimensions must be integers") from exc
        if width <= 0 or height <= 0:
            raise ConfigValidationError("stability resize dimensions must be positive")
        max_change_ratio = _finite_number(
            value.get("max_change_ratio", 0.02),
            name="stability.max_change_ratio",
            maximum=1.0,
        )
        stable_fraction = _finite_number(
            value.get("required_stable_fraction", 0.9),
            name="stability.required_stable_fraction",
            maximum=1.0,
        )
        return cls(width, height, max_change_ratio, stable_fraction)


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Runtime sampling settings shared by perception runners."""

    processing_fps: float = 5.0
    start_frame: int | None = None
    warmup_seconds: float = 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RuntimeConfig":
        _reject_unknown(value, {"processing_fps", "start_frame", "warmup_seconds"}, name="runtime")
        raw_start_frame = value.get("start_frame")
        if raw_start_frame in (None, ""):
            start_frame = None
        else:
            if isinstance(raw_start_frame, bool):
                raise ConfigValidationError("runtime.start_frame must be a non-negative integer")
            try:
                start_frame = int(raw_start_frame)
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError("runtime.start_frame must be a non-negative integer") from exc
            if start_frame < 0:
                raise ConfigValidationError("runtime.start_frame must be a non-negative integer")
        return cls(
            processing_fps=_finite_number(
                value.get("processing_fps", 5.0),
                name="runtime.processing_fps",
                minimum=0.000001,
            ),
            start_frame=start_frame,
            warmup_seconds=_finite_number(
                value.get("warmup_seconds", 0.0),
                name="runtime.warmup_seconds",
            ),
        )


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """Model-independent settings for the M2 detector adapter."""

    model: str = "yolov8s.pt"
    confidence: float = 0.35
    iou: float = 0.7
    device: str | None = None
    classes: tuple[int, ...] | None = None
    max_detections: int = 300

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DetectorConfig":
        _reject_unknown(
            value,
            {"model", "confidence", "iou", "device", "classes", "max_detections"},
            name="perception.detector",
        )
        model = str(value.get("model", "yolov8s.pt")).strip()
        if not model:
            raise ConfigValidationError("perception.detector.model must not be empty")
        device_value = value.get("device")
        device = str(device_value).strip() if device_value not in (None, "") else None
        raw_classes = value.get("classes")
        classes: tuple[int, ...] | None = None
        if raw_classes is not None:
            if not isinstance(raw_classes, (list, tuple)):
                raise ConfigValidationError("perception.detector.classes must be an array")
            parsed: list[int] = []
            for item in raw_classes:
                if isinstance(item, bool):
                    raise ConfigValidationError(
                        "perception.detector.classes must contain non-negative integers"
                    )
                try:
                    class_id = int(item)
                except (TypeError, ValueError) as exc:
                    raise ConfigValidationError(
                        "perception.detector.classes must contain non-negative integers"
                    ) from exc
                if class_id < 0:
                    raise ConfigValidationError(
                        "perception.detector.classes must contain non-negative integers"
                    )
                parsed.append(class_id)
            classes = tuple(dict.fromkeys(parsed))
        max_detections = value.get("max_detections", 300)
        if isinstance(max_detections, bool):
            raise ConfigValidationError("perception.detector.max_detections must be a positive integer")
        try:
            max_detections = int(max_detections)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                "perception.detector.max_detections must be a positive integer"
            ) from exc
        if max_detections < 1:
            raise ConfigValidationError("perception.detector.max_detections must be a positive integer")
        return cls(
            model=model,
            confidence=_finite_number(
                value.get("confidence", 0.35),
                name="perception.detector.confidence",
                maximum=1.0,
            ),
            iou=_finite_number(
                value.get("iou", 0.7),
                name="perception.detector.iou",
                maximum=1.0,
            ),
            device=device,
            classes=classes,
            max_detections=max_detections,
        )


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    """ByteTrack settings exposed without importing the tracker library."""

    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.8
    minimum_consecutive_frames: int = 1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrackerConfig":
        _reject_unknown(
            value,
            {
                "track_activation_threshold",
                "lost_track_buffer",
                "minimum_matching_threshold",
                "minimum_consecutive_frames",
            },
            name="perception.tracker",
        )
        buffer_size = value.get("lost_track_buffer", 30)
        consecutive = value.get("minimum_consecutive_frames", 1)
        if isinstance(buffer_size, bool) or isinstance(consecutive, bool):
            raise ConfigValidationError("perception.tracker frame settings must be integers")
        try:
            buffer_size = int(buffer_size)
            consecutive = int(consecutive)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                "perception.tracker frame settings must be integers"
            ) from exc
        if buffer_size < 1 or consecutive < 1:
            raise ConfigValidationError("perception.tracker frame settings must be positive")
        return cls(
            track_activation_threshold=_finite_number(
                value.get("track_activation_threshold", 0.25),
                name="perception.tracker.track_activation_threshold",
                maximum=1.0,
            ),
            lost_track_buffer=buffer_size,
            minimum_matching_threshold=_finite_number(
                value.get("minimum_matching_threshold", 0.8),
                name="perception.tracker.minimum_matching_threshold",
                maximum=1.0,
            ),
            minimum_consecutive_frames=consecutive,
        )


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    """Configuration for the optional M3 appearance encoder."""

    enabled: bool = False
    name: str = "dinov2"
    model: str = "dinov2_vits14"
    repository: str = "facebookresearch/dinov2"
    device: str | None = None
    input_size: int = 224
    batch_size: int = 8
    semantic_refresh_fps: float = 2.5
    crop_padding_ratio: float = 0.10

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EncoderConfig":
        _reject_unknown(
            value,
            {
                "enabled",
                "name",
                "model",
                "repository",
                "device",
                "input_size",
                "batch_size",
                "semantic_refresh_fps",
                "crop_padding_ratio",
            },
            name="perception.encoder",
        )
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigValidationError("perception.encoder.enabled must be boolean")
        name = str(value.get("name", "dinov2")).strip()
        if name != "dinov2":
            raise ConfigValidationError("perception.encoder.name must be 'dinov2' in M3")
        model = str(value.get("model", "dinov2_vits14")).strip()
        if not model:
            raise ConfigValidationError("perception.encoder.model must not be empty")
        repository = str(value.get("repository", "facebookresearch/dinov2")).strip()
        if not repository:
            raise ConfigValidationError("perception.encoder.repository must not be empty")
        device_value = value.get("device")
        device = str(device_value).strip() if device_value not in (None, "") else None
        integer_values = {"input_size": value.get("input_size", 224), "batch_size": value.get("batch_size", 8)}
        parsed: dict[str, int] = {}
        for field_name, raw_value in integer_values.items():
            if isinstance(raw_value, bool):
                raise ConfigValidationError(f"perception.encoder.{field_name} must be a positive integer")
            try:
                parsed[field_name] = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError(
                    f"perception.encoder.{field_name} must be a positive integer"
                ) from exc
            if parsed[field_name] < 1:
                raise ConfigValidationError(f"perception.encoder.{field_name} must be a positive integer")
        padding = _finite_number(
            value.get("crop_padding_ratio", 0.10),
            name="perception.encoder.crop_padding_ratio",
            maximum=1.0,
        )
        return cls(
            enabled=enabled,
            name=name,
            model=model,
            repository=repository,
            device=device,
            input_size=parsed["input_size"],
            batch_size=parsed["batch_size"],
            semantic_refresh_fps=_finite_number(
                value.get("semantic_refresh_fps", 2.5),
                name="perception.encoder.semantic_refresh_fps",
                minimum=0.000001,
            ),
            crop_padding_ratio=padding,
        )


@dataclass(frozen=True, slots=True)
class AssociationConfig:
    """Model-independent M3 identity-association settings."""

    appearance_weight: float = 0.50
    spatial_weight: float = 0.20
    size_weight: float = 0.15
    class_weight: float = 0.15
    min_appearance_similarity: float = 0.65
    min_total_score: float = 0.70
    ambiguity_margin: float = 0.05
    require_same_class: bool = True
    max_reid_seconds: float = 3.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AssociationConfig":
        _reject_unknown(
            value,
            {
                "appearance_weight",
                "spatial_weight",
                "size_weight",
                "class_weight",
                "min_appearance_similarity",
                "min_total_score",
                "ambiguity_margin",
                "require_same_class",
                "max_reid_seconds",
            },
            name="association",
        )
        require_same_class = value.get("require_same_class", True)
        if not isinstance(require_same_class, bool):
            raise ConfigValidationError("association.require_same_class must be boolean")
        weights = {
            field_name: _finite_number(value.get(field_name, default), name=f"association.{field_name}", maximum=1.0)
            for field_name, default in (
                ("appearance_weight", 0.50),
                ("spatial_weight", 0.20),
                ("size_weight", 0.15),
                ("class_weight", 0.15),
            )
        }
        if abs(sum(weights.values()) - 1.0) > 1e-6:
            raise ConfigValidationError("association weights must sum to 1.0")
        return cls(
            **weights,
            min_appearance_similarity=_finite_number(
                value.get("min_appearance_similarity", 0.65),
                name="association.min_appearance_similarity",
                maximum=1.0,
            ),
            min_total_score=_finite_number(
                value.get("min_total_score", 0.70),
                name="association.min_total_score",
                maximum=1.0,
            ),
            ambiguity_margin=_finite_number(
                value.get("ambiguity_margin", 0.05),
                name="association.ambiguity_margin",
                maximum=1.0,
            ),
            require_same_class=require_same_class,
            max_reid_seconds=_finite_number(
                value.get("max_reid_seconds", 3.0),
                name="association.max_reid_seconds",
                minimum=0.000001,
            ),
        )


@dataclass(frozen=True, slots=True)
class MemoryConfig:
    missing_grace_seconds: float = 1.0
    observation_history_size: int = 30
    embedding_history_size: int = 10

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MemoryConfig":
        _reject_unknown(
            value,
            {"missing_grace_seconds", "observation_history_size", "embedding_history_size"},
            name="memory",
        )
        parsed: dict[str, int] = {}
        for field_name, default in (("observation_history_size", 30), ("embedding_history_size", 10)):
            raw_value = value.get(field_name, default)
            if isinstance(raw_value, bool):
                raise ConfigValidationError(f"memory.{field_name} must be a positive integer")
            try:
                parsed[field_name] = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ConfigValidationError(f"memory.{field_name} must be a positive integer") from exc
            if parsed[field_name] < 1:
                raise ConfigValidationError(f"memory.{field_name} must be a positive integer")
        return cls(
            missing_grace_seconds=_finite_number(
                value.get("missing_grace_seconds", 1.0),
                name="memory.missing_grace_seconds",
                minimum=0.000001,
            ),
            **parsed,
        )


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Optional baseline reference used by the M3 runtime."""

    path: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BaselineConfig":
        _reject_unknown(value, {"path"}, name="baseline")
        raw_path = value.get("path")
        path = str(raw_path).strip() if raw_path not in (None, "") else None
        return cls(path=path)


@dataclass(frozen=True, slots=True)
class MaskedOverlapConfig:
    """Recover new detections that cover a masked baseline object."""

    enabled: bool = True
    min_overlap_ratio: float = 0.25
    min_changed_fraction: float = 0.5
    max_person_overlap_ratio: float = 0.1

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MaskedOverlapConfig":
        name = "perception.reference_change.masked_overlap"
        _reject_unknown(
            value,
            {"enabled", "min_overlap_ratio", "min_changed_fraction", "max_person_overlap_ratio"},
            name=name,
        )
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigValidationError(f"{name}.enabled must be boolean")
        return cls(
            enabled=enabled,
            min_overlap_ratio=_finite_number(
                value.get("min_overlap_ratio", 0.25),
                name=f"{name}.min_overlap_ratio",
                maximum=1.0,
            ),
            min_changed_fraction=_finite_number(
                value.get("min_changed_fraction", 0.5),
                name=f"{name}.min_changed_fraction",
                maximum=1.0,
            ),
            max_person_overlap_ratio=_finite_number(
                value.get("max_person_overlap_ratio", 0.1),
                name=f"{name}.max_person_overlap_ratio",
                maximum=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class SmallComponentConfig:
    """Conservative fallback for persistent changes below the normal area floor."""

    enabled: bool = False
    min_component_ratio: float = 0.00075
    min_component_area_px: int = 48
    stable_seconds: float = 2.0
    min_fill_ratio: float = 0.55
    min_aspect_ratio: float = 0.50

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SmallComponentConfig":
        name = "perception.reference_change.small_component"
        _reject_unknown(
            value,
            {
                "enabled",
                "min_component_ratio",
                "min_component_area_px",
                "stable_seconds",
                "min_fill_ratio",
                "min_aspect_ratio",
            },
            name=name,
        )
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigValidationError(f"{name}.enabled must be boolean")
        raw_min_area = value.get("min_component_area_px", 48)
        if isinstance(raw_min_area, bool):
            raise ConfigValidationError(
                f"{name}.min_component_area_px must be a positive integer"
            )
        try:
            min_component_area_px = int(raw_min_area)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                f"{name}.min_component_area_px must be a positive integer"
            ) from exc
        if min_component_area_px < 1:
            raise ConfigValidationError(
                f"{name}.min_component_area_px must be a positive integer"
            )
        return cls(
            enabled=enabled,
            min_component_ratio=_finite_number(
                value.get("min_component_ratio", 0.00075),
                name=f"{name}.min_component_ratio",
                minimum=0.000001,
                maximum=1.0,
            ),
            min_component_area_px=min_component_area_px,
            stable_seconds=_finite_number(
                value.get("stable_seconds", 2.0),
                name=f"{name}.stable_seconds",
            ),
            min_fill_ratio=_finite_number(
                value.get("min_fill_ratio", 0.55),
                name=f"{name}.min_fill_ratio",
                maximum=1.0,
            ),
            min_aspect_ratio=_finite_number(
                value.get("min_aspect_ratio", 0.50),
                name=f"{name}.min_aspect_ratio",
                maximum=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class AlignmentConfig:
    """Registration limits used by baseline-residual extraction."""

    min_response: float = 0.02
    max_translation_ratio: float = 0.03
    max_unfused_translation_ratio: float = 0.003

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AlignmentConfig":
        name = "perception.reference_change.alignment"
        _reject_unknown(
            value,
            {"min_response", "max_translation_ratio", "max_unfused_translation_ratio"},
            name=name,
        )
        parsed = cls(
            min_response=_finite_number(
                value.get("min_response", 0.02),
                name=f"{name}.min_response",
                maximum=1.0,
            ),
            max_translation_ratio=_finite_number(
                value.get("max_translation_ratio", 0.03),
                name=f"{name}.max_translation_ratio",
                maximum=1.0,
            ),
            max_unfused_translation_ratio=_finite_number(
                value.get("max_unfused_translation_ratio", 0.003),
                name=f"{name}.max_unfused_translation_ratio",
                maximum=1.0,
            ),
        )
        if parsed.max_unfused_translation_ratio > parsed.max_translation_ratio:
            raise ConfigValidationError(
                f"{name}.max_unfused_translation_ratio must not exceed "
                f"{name}.max_translation_ratio"
            )
        return parsed


@dataclass(frozen=True, slots=True)
class BaselineResidualConfig:
    """Detect a new object inside an otherwise masked baseline object."""

    enabled: bool = True
    min_component_ratio: float = 0.00075
    min_component_area_px: int = 48
    stable_seconds: float = 2.0
    min_fill_ratio: float = 0.35
    max_person_overlap_ratio: float = 0.10
    max_baseline_coverage_ratio: float = 0.50
    min_current_edge_ratio: float = 0.35

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BaselineResidualConfig":
        name = "perception.reference_change.baseline_residual"
        _reject_unknown(
            value,
            {
                "enabled",
                "min_component_ratio",
                "min_component_area_px",
                "stable_seconds",
                "min_fill_ratio",
                "max_person_overlap_ratio",
                "max_baseline_coverage_ratio",
                "min_current_edge_ratio",
            },
            name=name,
        )
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ConfigValidationError(f"{name}.enabled must be boolean")
        raw_min_area = value.get("min_component_area_px", 48)
        if isinstance(raw_min_area, bool):
            raise ConfigValidationError(f"{name}.min_component_area_px must be a positive integer")
        try:
            min_component_area_px = int(raw_min_area)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                f"{name}.min_component_area_px must be a positive integer"
            ) from exc
        if min_component_area_px < 1:
            raise ConfigValidationError(f"{name}.min_component_area_px must be a positive integer")
        return cls(
            enabled=enabled,
            min_component_ratio=_finite_number(
                value.get("min_component_ratio", 0.00075),
                name=f"{name}.min_component_ratio",
                minimum=0.000001,
                maximum=1.0,
            ),
            min_component_area_px=min_component_area_px,
            stable_seconds=_finite_number(
                value.get("stable_seconds", 2.0),
                name=f"{name}.stable_seconds",
            ),
            min_fill_ratio=_finite_number(
                value.get("min_fill_ratio", 0.35),
                name=f"{name}.min_fill_ratio",
                maximum=1.0,
            ),
            max_person_overlap_ratio=_finite_number(
                value.get("max_person_overlap_ratio", 0.10),
                name=f"{name}.max_person_overlap_ratio",
                maximum=1.0,
            ),
            max_baseline_coverage_ratio=_finite_number(
                value.get("max_baseline_coverage_ratio", 0.50),
                name=f"{name}.max_baseline_coverage_ratio",
                maximum=1.0,
            ),
            min_current_edge_ratio=_finite_number(
                value.get("min_current_edge_ratio", 0.35),
                name=f"{name}.min_current_edge_ratio",
                maximum=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class ReferenceChangeConfig:
    min_component_ratio: float = 0.0025
    min_component_area_px: int = 64
    stable_seconds: float = 1.0
    min_current_edge_ratio: float = 0.0
    fusion_min_overlap_ratio: float = 0.25
    fusion_min_iou: float = 0.10
    small_component: SmallComponentConfig = SmallComponentConfig()
    masked_overlap: MaskedOverlapConfig = MaskedOverlapConfig()
    alignment: AlignmentConfig = AlignmentConfig()
    baseline_residual: BaselineResidualConfig = BaselineResidualConfig()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReferenceChangeConfig":
        name = "perception.reference_change"
        _reject_unknown(
            value,
            {
                "min_component_ratio",
                "min_component_area_px",
                "stable_seconds",
                "min_current_edge_ratio",
                "fusion_min_overlap_ratio",
                "fusion_min_iou",
                "small_component",
                "masked_overlap",
                "alignment",
                "baseline_residual",
            },
            name=name,
        )
        raw_min_area = value.get("min_component_area_px", 64)
        if isinstance(raw_min_area, bool):
            raise ConfigValidationError(
                f"{name}.min_component_area_px must be a positive integer"
            )
        try:
            min_component_area_px = int(raw_min_area)
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                f"{name}.min_component_area_px must be a positive integer"
            ) from exc
        if min_component_area_px < 1:
            raise ConfigValidationError(
                f"{name}.min_component_area_px must be a positive integer"
            )
        return cls(
            min_component_ratio=_finite_number(
                value.get("min_component_ratio", 0.0025),
                name=f"{name}.min_component_ratio",
                minimum=0.000001,
                maximum=1.0,
            ),
            min_component_area_px=min_component_area_px,
            stable_seconds=_finite_number(
                value.get("stable_seconds", 1.0),
                name=f"{name}.stable_seconds",
            ),
            min_current_edge_ratio=_finite_number(
                value.get("min_current_edge_ratio", 0.0),
                name=f"{name}.min_current_edge_ratio",
                maximum=1.0,
            ),
            fusion_min_overlap_ratio=_finite_number(
                value.get("fusion_min_overlap_ratio", 0.25),
                name=f"{name}.fusion_min_overlap_ratio",
                maximum=1.0,
            ),
            fusion_min_iou=_finite_number(
                value.get("fusion_min_iou", 0.10),
                name=f"{name}.fusion_min_iou",
                maximum=1.0,
            ),
            small_component=SmallComponentConfig.from_mapping(
                _mapping(value.get("small_component", {}), name=f"{name}.small_component")
            ),
            masked_overlap=MaskedOverlapConfig.from_mapping(
                _mapping(value.get("masked_overlap", {}), name=f"{name}.masked_overlap")
            ),
            alignment=AlignmentConfig.from_mapping(
                _mapping(value.get("alignment", {}), name=f"{name}.alignment")
            ),
            baseline_residual=BaselineResidualConfig.from_mapping(
                _mapping(value.get("baseline_residual", {}), name=f"{name}.baseline_residual")
            ),
        )


@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    """M2 perception configuration.

    It is disabled by default so existing M1 calibration configs do not
    suddenly require ML dependencies or model weights.
    """

    enabled: bool = False
    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    encoder: EncoderConfig = EncoderConfig()
    reference_change: ReferenceChangeConfig = ReferenceChangeConfig()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PerceptionConfig":
        _reject_unknown(value, {"enabled", "detector", "tracker", "encoder", "reference_change"}, name="perception")
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ConfigValidationError("perception.enabled must be boolean")
        return cls(
            enabled=enabled,
            detector=DetectorConfig.from_mapping(
                _mapping(value.get("detector", {}), name="perception.detector")
            ),
            tracker=TrackerConfig.from_mapping(
                _mapping(value.get("tracker", {}), name="perception.tracker")
            ),
            encoder=EncoderConfig.from_mapping(
                _mapping(value.get("encoder", {}), name="perception.encoder")
            ),
            reference_change=ReferenceChangeConfig.from_mapping(
                _mapping(value.get("reference_change", {}), name="perception.reference_change")
            ),
        )


@dataclass(frozen=True, slots=True)
class SystemConfig:
    dataset_version: str = "development"
    forgotten: ForgottenEventConfig = ForgottenEventConfig()
    moved: MovedEventConfig = MovedEventConfig()
    evaluation: EvaluationThresholdConfig = EvaluationThresholdConfig()
    source: SourceConfig = SourceConfig()
    roi: ROIConfig = ROIConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    stability: StabilityConfig = StabilityConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    perception: PerceptionConfig = PerceptionConfig()
    association: AssociationConfig = AssociationConfig()
    memory: MemoryConfig = MemoryConfig()
    baseline: BaselineConfig = BaselineConfig()

    def to_dict(self) -> dict[str, Any]:
        result = {
            "dataset_version": self.dataset_version,
            "events": {
                "forgotten": asdict(self.forgotten),
                "moved": asdict(self.moved),
            },
            "evaluation": asdict(self.evaluation),
            "source": asdict(self.source),
            "roi": asdict(self.roi),
            "calibration": asdict(self.calibration),
            "stability": asdict(self.stability),
            "runtime": asdict(self.runtime),
            "perception": asdict(self.perception),
            "association": asdict(self.association),
            "memory": asdict(self.memory),
            "baseline": asdict(self.baseline),
        }
        # An empty source section is the serializable representation of the
        # intentionally unconfigured default SourceConfig.  This keeps
        # ``validate_config(config.to_dict())`` round-trippable without making
        # a missing video path look like a runnable source.
        if self.source == SourceConfig():
            result["source"] = {}
        return result


def validate_config(value: Mapping[str, Any]) -> SystemConfig:
    """Validate a config mapping without importing media/model libraries."""

    _reject_unknown(
        value,
        {
            "dataset_version",
            "events",
            "evaluation",
            "source",
            "roi",
            "calibration",
            "stability",
            "runtime",
            "perception",
            "association",
            "memory",
            "baseline",
        },
        name="root",
    )
    dataset_version = str(value.get("dataset_version", "development")).strip()
    if not dataset_version:
        raise ConfigValidationError("dataset_version must not be empty")
    events = _mapping(value.get("events", {}), name="events")
    _reject_unknown(events, {"forgotten", "moved"}, name="events")
    forgotten = ForgottenEventConfig.from_mapping(
        _mapping(events.get("forgotten", {}), name="events.forgotten"),
        name="events.forgotten",
    )
    moved = MovedEventConfig.from_mapping(
        _mapping(events.get("moved", {}), name="events.moved"),
        name="events.moved",
    )
    evaluation = EvaluationThresholdConfig.from_mapping(
        _mapping(value.get("evaluation", {}), name="evaluation")
    )
    source = (
        SourceConfig.from_mapping(_mapping(value["source"], name="source"))
        if "source" in value
        else SourceConfig()
    )
    roi = ROIConfig.from_mapping(_mapping(value.get("roi", {}), name="roi"))
    calibration = CalibrationConfig.from_mapping(
        _mapping(value.get("calibration", {}), name="calibration")
    )
    stability = StabilityConfig.from_mapping(
        _mapping(value.get("stability", {}), name="stability")
    )
    runtime = RuntimeConfig.from_mapping(
        _mapping(value.get("runtime", {}), name="runtime")
    )
    perception = PerceptionConfig.from_mapping(
        _mapping(value.get("perception", {}), name="perception")
    )
    if perception.encoder.enabled and not perception.enabled:
        raise ConfigValidationError("perception.encoder.enabled requires perception.enabled=true")
    association = AssociationConfig.from_mapping(
        _mapping(value.get("association", {}), name="association")
    )
    memory = MemoryConfig.from_mapping(_mapping(value.get("memory", {}), name="memory"))
    baseline = BaselineConfig.from_mapping(_mapping(value.get("baseline", {}), name="baseline"))
    return SystemConfig(
        dataset_version=dataset_version,
        forgotten=forgotten,
        moved=moved,
        evaluation=evaluation,
        source=source,
        roi=roi,
        calibration=calibration,
        stability=stability,
        runtime=runtime,
        perception=perception,
        association=association,
        memory=memory,
        baseline=baseline,
    )


def load_config(path: Path) -> SystemConfig:
    """Load JSON or TOML config and validate it into ``SystemConfig``."""

    with path.open("rb") as handle:
        if path.suffix.lower() == ".toml":
            value = tomllib.load(handle)
        else:
            value = json.load(handle)
    return validate_config(_mapping(value, name="config"))
