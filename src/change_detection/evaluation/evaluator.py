"""Deterministic event-level evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite
from typing import Any, Iterable, Mapping

from change_detection.domain import BBox, EventAnnotation, EventType, MovementOutcome


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Thresholds for correctness matching and separate latency reporting.

    ``start_tolerance_seconds`` and ``min_temporal_overlap`` remain accepted so
    older configuration files can still be loaded. Correctness matching no
    longer uses them because annotation start time and detector start time have
    different semantics.
    """

    event_boundary_tolerance_seconds: float = 1.0
    forgotten_iou_threshold: float = 0.3
    moved_iou_threshold: float = 0.3
    require_prediction_object_id: bool = False
    latency_deadlines_seconds: tuple[float, ...] = (3.0, 5.0, 10.0)
    start_tolerance_seconds: float = 3.0
    min_temporal_overlap: float = 0.1

    def __post_init__(self) -> None:
        numeric_fields = {
            "event_boundary_tolerance_seconds": self.event_boundary_tolerance_seconds,
            "start_tolerance_seconds": self.start_tolerance_seconds,
        }
        for name, value in numeric_fields.items():
            if not isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        unit_fields = {
            "forgotten_iou_threshold": self.forgotten_iou_threshold,
            "moved_iou_threshold": self.moved_iou_threshold,
            "min_temporal_overlap": self.min_temporal_overlap,
        }
        for name, value in unit_fields.items():
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        deadlines = tuple(sorted({float(value) for value in self.latency_deadlines_seconds}))
        if not deadlines or any(not isfinite(value) or value < 0 for value in deadlines):
            raise ValueError("latency_deadlines_seconds must contain finite non-negative values")
        object.__setattr__(self, "latency_deadlines_seconds", deadlines)


@dataclass(slots=True)
class Match:
    prediction: EventAnnotation
    ground_truth: EventAnnotation
    spatial_score: float
    latency_sec: float | None
    start_delay_sec: float


def _prediction_bbox(event: EventAnnotation, *, moved: bool) -> BBox | None:
    if moved:
        return event.new_bbox or event.bbox
    return event.bbox or event.new_bbox


def _spatial_score(prediction: EventAnnotation, ground_truth: EventAnnotation) -> float | None:
    if ground_truth.event_type == EventType.MOVED_OBJECT:
        if ground_truth.movement_outcome == MovementOutcome.LEFT_SCENE:
            if ground_truth.baseline_bbox is None or prediction.baseline_bbox is None:
                return None
            return ground_truth.baseline_bbox.iou(prediction.baseline_bbox)
        gt_pairs = (ground_truth.baseline_bbox, ground_truth.new_bbox)
        pred_pairs = (prediction.baseline_bbox, prediction.new_bbox)
        if any(item is None for item in gt_pairs + pred_pairs):
            return None
        assert gt_pairs[0] is not None and gt_pairs[1] is not None
        assert pred_pairs[0] is not None and pred_pairs[1] is not None
        return (gt_pairs[0].iou(pred_pairs[0]) + gt_pairs[1].iou(pred_pairs[1])) / 2.0
    gt_bbox = ground_truth.bbox
    pred_bbox = _prediction_bbox(prediction, moved=False)
    if gt_bbox is None or pred_bbox is None:
        return None
    return gt_bbox.iou(pred_bbox)


def _spatial_threshold(ground_truth: EventAnnotation, config: EvaluationConfig) -> float:
    if ground_truth.event_type == EventType.MOVED_OBJECT:
        return config.moved_iou_threshold
    return config.forgotten_iou_threshold


def _identity_match(
    prediction: EventAnnotation,
    ground_truth: EventAnnotation,
    config: EvaluationConfig,
) -> bool:
    if not config.require_prediction_object_id:
        return True
    return bool(
        prediction.object_id
        and ground_truth.object_id
        and prediction.object_id == ground_truth.object_id
    )


def _event_window_match(
    prediction: EventAnnotation,
    ground_truth: EventAnnotation,
    config: EvaluationConfig,
) -> bool:
    confirmed_at = prediction.confirmation_time_sec
    if confirmed_at is None:
        return False
    tolerance = config.event_boundary_tolerance_seconds
    if confirmed_at < ground_truth.start_time_sec - tolerance:
        return False
    return ground_truth.end_time_sec is None or confirmed_at <= ground_truth.end_time_sec + tolerance


def _candidate_spatial_score(
    prediction: EventAnnotation,
    ground_truth: EventAnnotation,
    config: EvaluationConfig,
) -> float | None:
    if prediction.event_type != ground_truth.event_type:
        return None
    if (
        ground_truth.event_type == EventType.MOVED_OBJECT
        and prediction.movement_outcome != ground_truth.movement_outcome
    ):
        return None
    if not _event_window_match(prediction, ground_truth, config):
        return None
    if not _identity_match(prediction, ground_truth, config):
        return None
    spatial = _spatial_score(prediction, ground_truth)
    if spatial is None or spatial < _spatial_threshold(ground_truth, config):
        return None
    return spatial


def _confirmation_distance(prediction: EventAnnotation, ground_truth: EventAnnotation) -> float:
    confirmed_at = prediction.confirmation_time_sec
    if confirmed_at is None:
        return float("inf")
    target = ground_truth.confirmation_time_sec
    if target is None:
        target = ground_truth.start_time_sec
    return abs(confirmed_at - target)


def _as_event(value: EventAnnotation | Mapping[str, Any]) -> EventAnnotation:
    if isinstance(value, EventAnnotation):
        return value
    return EventAnnotation.from_dict(value)


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    items = [float(value) for value in values]
    return {
        "count": len(items),
        "mean": sum(items) / len(items) if items else None,
        "median": _percentile(items, 0.5),
        "p95": _percentile(items, 0.95),
        "min": min(items) if items else None,
        "max": max(items) if items else None,
    }


def summarize_timeliness(
    *,
    confirmation_latencies: Iterable[float],
    start_delays: Iterable[float],
    deadlines_seconds: Iterable[float],
    ground_truth_count: int,
) -> dict[str, Any]:
    """Summarize alert latency without changing correctness TP/FP/FN."""

    latencies = [float(value) for value in confirmation_latencies]
    delays = [float(value) for value in start_delays]
    deadlines = tuple(sorted({float(value) for value in deadlines_seconds}))
    deadline_results: dict[str, Any] = {}
    for deadline in deadlines:
        on_time = sum(value <= deadline + 1e-9 for value in latencies)
        deadline_results[f"{deadline:g}s"] = {
            "threshold_sec": deadline,
            "on_time_count": on_time,
            "late_count": len(latencies) - on_time,
            "on_time_rate_of_matches": on_time / len(latencies) if latencies else 0.0,
            "recall": on_time / ground_truth_count if ground_truth_count else 0.0,
        }
    return {
        "matched_events": len(latencies),
        "early_count": sum(value < 0 for value in latencies),
        "confirmation_latency_sec": _distribution(latencies),
        "start_delay_sec": _distribution(delays),
        "deadlines": deadline_results,
    }


def _window_distance(prediction: EventAnnotation, ground_truth: EventAnnotation) -> float:
    confirmed_at = prediction.confirmation_time_sec
    if confirmed_at is None:
        return float("inf")
    if confirmed_at < ground_truth.start_time_sec:
        return ground_truth.start_time_sec - confirmed_at
    if ground_truth.end_time_sec is not None and confirmed_at > ground_truth.end_time_sec:
        return confirmed_at - ground_truth.end_time_sec
    return 0.0


def _closest_ground_truth(
    prediction: EventAnnotation,
    ground_truth: list[EventAnnotation],
) -> EventAnnotation | None:
    if not ground_truth:
        return None
    same_type = [item for item in ground_truth if item.event_type == prediction.event_type]
    candidates = same_type or ground_truth
    if prediction.event_type == EventType.MOVED_OBJECT:
        same_outcome = [
            item for item in candidates if item.movement_outcome == prediction.movement_outcome
        ]
        candidates = same_outcome or candidates

    def key(item: EventAnnotation) -> tuple[float, float, float, str]:
        spatial = _spatial_score(prediction, item)
        return (
            _window_distance(prediction, item),
            -(spatial if spatial is not None else -1.0),
            _confirmation_distance(prediction, item),
            item.event_id,
        )

    return min(candidates, key=key)


def _error_category(
    prediction: EventAnnotation,
    ground_truth: EventAnnotation | None,
    config: EvaluationConfig,
) -> str:
    if ground_truth is None:
        return "unknown_fp"
    if prediction.event_type != ground_truth.event_type:
        return "wrong_type"
    if (
        ground_truth.event_type == EventType.MOVED_OBJECT
        and prediction.movement_outcome != ground_truth.movement_outcome
    ):
        return "movement_outcome_mismatch"
    if prediction.confirmation_time_sec is None:
        return "missing_confirmation_time"
    if not _event_window_match(prediction, ground_truth, config):
        return "outside_event_window"
    if not _identity_match(prediction, ground_truth, config):
        return "identity_mismatch"
    spatial = _spatial_score(prediction, ground_truth)
    if spatial is None or spatial < _spatial_threshold(ground_truth, config):
        return "spatial_mismatch"
    return "unknown_fp"


def evaluate_events(
    ground_truth: Iterable[EventAnnotation | Mapping[str, Any]],
    predictions: Iterable[EventAnnotation | Mapping[str, Any]],
    *,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    """Match event correctness once, then report alert timeliness separately."""

    config = config or EvaluationConfig()
    gt = [_as_event(item) for item in ground_truth]
    pred = [_as_event(item) for item in predictions]
    candidates: list[tuple[float, float, int, int]] = []
    for prediction_index, prediction in enumerate(pred):
        for ground_truth_index, ground_truth_item in enumerate(gt):
            spatial = _candidate_spatial_score(prediction, ground_truth_item, config)
            if spatial is not None:
                candidates.append(
                    (
                        _confirmation_distance(prediction, ground_truth_item),
                        -spatial,
                        prediction_index,
                        ground_truth_index,
                    )
                )
    candidates.sort()

    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()
    matches: list[Match] = []
    for _, negative_spatial, prediction_index, ground_truth_index in candidates:
        if prediction_index in used_predictions or ground_truth_index in used_ground_truth:
            continue
        used_predictions.add(prediction_index)
        used_ground_truth.add(ground_truth_index)
        prediction_item = pred[prediction_index]
        ground_truth_item = gt[ground_truth_index]
        latency = (
            prediction_item.confirmation_time_sec - ground_truth_item.confirmation_time_sec
            if prediction_item.confirmation_time_sec is not None
            and ground_truth_item.confirmation_time_sec is not None
            else None
        )
        matches.append(
            Match(
                prediction=prediction_item,
                ground_truth=ground_truth_item,
                spatial_score=-negative_spatial,
                latency_sec=latency,
                start_delay_sec=(
                    prediction_item.start_time_sec - ground_truth_item.start_time_sec
                ),
            )
        )

    unmatched_predictions = [
        item for index, item in enumerate(pred) if index not in used_predictions
    ]
    unmatched_ground_truth = [
        item for index, item in enumerate(gt) if index not in used_ground_truth
    ]
    result: dict[str, Any] = {
        "overall": _metrics(len(matches), len(unmatched_predictions), len(unmatched_ground_truth)),
        "FORGOTTEN_OBJECT": {},
        "MOVED_OBJECT": {},
        "MOVED_OBJECT_OUTCOMES": {
            MovementOutcome.RELOCATED.value: {},
            MovementOutcome.LEFT_SCENE.value: {},
        },
        "matches": [
            {
                "prediction_event_id": item.prediction.event_id,
                "ground_truth_event_id": item.ground_truth.event_id,
                "score": item.spatial_score,
                "spatial_score": item.spatial_score,
                "latency_sec": item.latency_sec,
                "confirmation_latency_sec": item.latency_sec,
                "start_delay_sec": item.start_delay_sec,
                "movement_outcome": (
                    item.ground_truth.movement_outcome.value
                    if item.ground_truth.movement_outcome is not None
                    else None
                ),
            }
            for item in matches
        ],
        "error_breakdown": {},
    }

    for event_type in (EventType.FORGOTTEN_OBJECT, EventType.MOVED_OBJECT):
        type_matches = [item for item in matches if item.ground_truth.event_type == event_type]
        type_gt = [item for item in gt if item.event_type == event_type]
        type_pred = [item for item in pred if item.event_type == event_type]
        type_prediction_ids = {item.prediction.event_id for item in type_matches}
        type_gt_ids = {item.ground_truth.event_id for item in type_matches}
        result[event_type.value] = _metrics(
            len(type_matches),
            len([item for item in type_pred if item.event_id not in type_prediction_ids]),
            len([item for item in type_gt if item.event_id not in type_gt_ids]),
        )

    for outcome in MovementOutcome:
        outcome_matches = [
            item
            for item in matches
            if item.ground_truth.event_type == EventType.MOVED_OBJECT
            and item.ground_truth.movement_outcome == outcome
        ]
        outcome_gt = [
            item
            for item in gt
            if item.event_type == EventType.MOVED_OBJECT and item.movement_outcome == outcome
        ]
        outcome_pred = [
            item
            for item in pred
            if item.event_type == EventType.MOVED_OBJECT and item.movement_outcome == outcome
        ]
        outcome_prediction_ids = {item.prediction.event_id for item in outcome_matches}
        outcome_gt_ids = {item.ground_truth.event_id for item in outcome_matches}
        result["MOVED_OBJECT_OUTCOMES"][outcome.value] = _metrics(
            len(outcome_matches),
            len(
                [item for item in outcome_pred if item.event_id not in outcome_prediction_ids]
            ),
            len([item for item in outcome_gt if item.event_id not in outcome_gt_ids]),
        )

    for item in unmatched_predictions:
        duplicate = any(
            _candidate_spatial_score(item, candidate.ground_truth, config) is not None
            for candidate in matches
        )
        category = (
            "duplicate_event"
            if duplicate
            else _error_category(item, _closest_ground_truth(item, gt), config)
        )
        result["error_breakdown"][category] = (
            result["error_breakdown"].get(category, 0) + 1
        )
    for item in unmatched_ground_truth:
        key = "missed_" + item.event_type.value.lower()
        result["error_breakdown"][key] = result["error_breakdown"].get(key, 0) + 1

    latencies = [item.latency_sec for item in matches if item.latency_sec is not None]
    start_delays = [item.start_delay_sec for item in matches]
    result["timeliness"] = summarize_timeliness(
        confirmation_latencies=latencies,
        start_delays=start_delays,
        deadlines_seconds=config.latency_deadlines_seconds,
        ground_truth_count=len(gt),
    )
    result["latency"] = {
        "matched_events": len(latencies),
        "mean_confirmation_delta_sec": (
            sum(latencies) / len(latencies) if latencies else None
        ),
    }
    return result
