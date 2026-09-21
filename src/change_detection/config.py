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
class SystemConfig:
    dataset_version: str = "development"
    forgotten: EventTimingConfig = EventTimingConfig()
    moved: EventTimingConfig = EventTimingConfig()
    evaluation: EvaluationThresholdConfig = EvaluationThresholdConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_config(value: Mapping[str, Any]) -> SystemConfig:
    """Validate a config mapping without importing media/model libraries."""

    _reject_unknown(value, {"dataset_version", "events", "evaluation"}, name="root")
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
    return SystemConfig(dataset_version=dataset_version, forgotten=forgotten, moved=moved, evaluation=evaluation)


def load_config(path: Path) -> SystemConfig:
    """Load JSON or TOML config and validate it into ``SystemConfig``."""

    with path.open("rb") as handle:
        if path.suffix.lower() == ".toml":
            value = tomllib.load(handle)
        else:
            value = json.load(handle)
    return validate_config(_mapping(value, name="config"))
