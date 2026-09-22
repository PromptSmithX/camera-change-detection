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
class EvaluationThresholdConfig:
    start_tolerance_seconds: float = 3.0
    min_temporal_overlap: float = 0.1
    forgotten_iou_threshold: float = 0.3
    moved_iou_threshold: float = 0.3
    require_prediction_object_id: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationThresholdConfig":
        _reject_unknown(
            value,
            {
                "start_tolerance_seconds",
                "min_temporal_overlap",
                "forgotten_iou_threshold",
                "moved_iou_threshold",
                "require_prediction_object_id",
            },
            name="evaluation",
        )
        require_id = value.get("require_prediction_object_id", False)
        if not isinstance(require_id, bool):
            raise ConfigValidationError("evaluation.require_prediction_object_id must be boolean")
        return cls(
            start_tolerance_seconds=_finite_number(
                value.get("start_tolerance_seconds", 3.0),
                name="evaluation.start_tolerance_seconds",
            ),
            min_temporal_overlap=_finite_number(
                value.get("min_temporal_overlap", 0.1),
                name="evaluation.min_temporal_overlap",
                maximum=1.0,
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
class SystemConfig:
    dataset_version: str = "development"
    forgotten: EventTimingConfig = EventTimingConfig()
    moved: EventTimingConfig = EventTimingConfig()
    evaluation: EvaluationThresholdConfig = EvaluationThresholdConfig()
    source: SourceConfig = SourceConfig()
    roi: ROIConfig = ROIConfig()
    calibration: CalibrationConfig = CalibrationConfig()
    stability: StabilityConfig = StabilityConfig()

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
        {"dataset_version", "events", "evaluation", "source", "roi", "calibration", "stability"},
        name="root",
    )
    dataset_version = str(value.get("dataset_version", "development")).strip()
    if not dataset_version:
        raise ConfigValidationError("dataset_version must not be empty")
    events = _mapping(value.get("events", {}), name="events")
    _reject_unknown(events, {"forgotten", "moved"}, name="events")
    forgotten = EventTimingConfig.from_mapping(
        _mapping(events.get("forgotten", {}), name="events.forgotten"),
        name="events.forgotten",
    )
    moved = EventTimingConfig.from_mapping(
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
    return SystemConfig(
        dataset_version=dataset_version,
        forgotten=forgotten,
        moved=moved,
        evaluation=evaluation,
        source=source,
        roi=roi,
        calibration=calibration,
        stability=stability,
    )


def load_config(path: Path) -> SystemConfig:
    """Load JSON or TOML config and validate it into ``SystemConfig``."""

    with path.open("rb") as handle:
        if path.suffix.lower() == ".toml":
            value = tomllib.load(handle)
        else:
            value = json.load(handle)
    return validate_config(_mapping(value, name="config"))
