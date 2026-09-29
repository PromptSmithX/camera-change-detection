"""Deterministic event-level evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Mapping

from change_detection.domain import BBox, EventAnnotation, EventType, MovementOutcome


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    start_tolerance_seconds: float = 3.0
    min_temporal_overlap: float = 0.1
    forgotten_iou_threshold: float = 0.3
    moved_iou_threshold: float = 0.3
    require_prediction_object_id: bool = False
    max_confirmation_delay_seconds: float | None = None

    def __post_init__(self) -> None:
        delay = self.max_confirmation_delay_seconds
        if delay is not None and (not isfinite(float(delay)) or float(delay) < 0):
            raise ValueError("max_confirmation_delay_seconds must be finite and non-negative")


@dataclass(slots=True)
class Match:
    prediction: EventAnnotation
    ground_truth: EventAnnotation
    score: float
    latency_sec: float | None


def _end(event: EventAnnotation) -> float | None:
    return event.end_time_sec


def _temporal_overlap(prediction: EventAnnotation, ground_truth: EventAnnotation) -> float:
    prediction_end = _end(prediction)
    ground_truth_end = _end(ground_truth)
    if prediction_end is None or ground_truth_end is None:
        return 0.0
    intersection = max(0.0, min(prediction_end, ground_truth_end) - max(prediction.start_time_sec, ground_truth.start_time_sec))
    union = max(prediction_end, ground_truth_end) - min(prediction.start_time_sec, ground_truth.start_time_sec)
    return intersection / union if union > 0 else 0.0


def _temporal_match(prediction: EventAnnotation, ground_truth: EventAnnotation, config: EvaluationConfig) -> bool:
    if abs(prediction.start_time_sec - ground_truth.start_time_sec) <= config.start_tolerance_seconds:
        return True
    return _temporal_overlap(prediction, ground_truth) >= config.min_temporal_overlap


def _prediction_bbox(event: EventAnnotation, *, moved: bool) -> BBox | None:
    if moved:
        return event.new_bbox or event.bbox
    return event.bbox or event.new_bbox


def _spatial_score(prediction: EventAnnotation, ground_truth: EventAnnotation, config: EvaluationConfig) -> float | None:
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


def _identity_match(prediction: EventAnnotation, ground_truth: EventAnnotation, config: EvaluationConfig) -> bool:
    if not config.require_prediction_object_id:
        return True
    return bool(prediction.object_id and ground_truth.object_id and prediction.object_id == ground_truth.object_id)


def _confirmation_match(
    prediction: EventAnnotation,
    ground_truth: EventAnnotation,
    config: EvaluationConfig,
) -> bool:
    max_delay = config.max_confirmation_delay_seconds
    if max_delay is None:
        return True
    if prediction.confirmation_time_sec is None or ground_truth.confirmation_time_sec is None:
        return False
    delay = prediction.confirmation_time_sec - ground_truth.confirmation_time_sec
    return delay <= max_delay + 1e-9


def _match_score(prediction: EventAnnotation, ground_truth: EventAnnotation, config: EvaluationConfig) -> float | None:
    if prediction.event_type != ground_truth.event_type:
        return None
    if (
        ground_truth.event_type == EventType.MOVED_OBJECT
        and prediction.movement_outcome != ground_truth.movement_outcome
    ):
        return None
    if not _temporal_match(prediction, ground_truth, config):
        return None
    if not _identity_match(prediction, ground_truth, config):
        return None
    if not _confirmation_match(prediction, ground_truth, config):
        return None
    spatial = _spatial_score(prediction, ground_truth, config)
    threshold = config.moved_iou_threshold if ground_truth.event_type == EventType.MOVED_OBJECT else config.forgotten_iou_threshold
    if spatial is None or spatial < threshold:
        return None
    temporal = max(
        0.0,
        1.0 - min(abs(prediction.start_time_sec - ground_truth.start_time_sec) / max(config.start_tolerance_seconds, 1e-9), 1.0),
    )
    return 0.7 * spatial + 0.3 * temporal


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
    if not _temporal_match(prediction, ground_truth, config):
        return "temporal_mismatch"
    spatial = _spatial_score(prediction, ground_truth, config)
    if spatial is None or spatial < (config.moved_iou_threshold if ground_truth.event_type == EventType.MOVED_OBJECT else config.forgotten_iou_threshold):
        return "spatial_mismatch"
    if config.max_confirmation_delay_seconds is not None:
        if prediction.confirmation_time_sec is None or ground_truth.confirmation_time_sec is None:
            return "missing_confirmation_time"
        if not _confirmation_match(prediction, ground_truth, config):
            return "late_confirmation"
    return "identity_mismatch"


def _closest_ground_truth(
    prediction: EventAnnotation,
    ground_truth: list[EventAnnotation],
) -> EventAnnotation | None:
    if not ground_truth:
        return None
    return min(
        ground_truth,
        key=lambda item: (abs(prediction.start_time_sec - item.start_time_sec), item.event_id),
    )


def evaluate_events(
    ground_truth: Iterable[EventAnnotation | Mapping[str, Any]],
    predictions: Iterable[EventAnnotation | Mapping[str, Any]],
    *,
    config: EvaluationConfig | None = None,
) -> dict[str, Any]:
    """Match events one-to-one and return overall/type-specific metrics.

    Matching is deterministic greedy assignment over descending match score.
    A production experiment can replace this with Hungarian assignment without
    changing the public result contract.
    """

    config = config or EvaluationConfig()
    gt = [_as_event(item) for item in ground_truth]
    pred = [_as_event(item) for item in predictions]
    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(pred):
        for ground_truth_index, ground_truth_item in enumerate(gt):
            score = _match_score(prediction, ground_truth_item, config)
            if score is not None:
                candidates.append((score, prediction_index, ground_truth_index))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_predictions: set[int] = set()
    used_ground_truth: set[int] = set()
    matches: list[Match] = []
    for score, prediction_index, ground_truth_index in candidates:
        if prediction_index in used_predictions or ground_truth_index in used_ground_truth:
            continue
        used_predictions.add(prediction_index)
        used_ground_truth.add(ground_truth_index)
        prediction_item = pred[prediction_index]
        ground_truth_item = gt[ground_truth_index]
        matches.append(
            Match(
                prediction=prediction_item,
                ground_truth=ground_truth_item,
                score=score,
                latency_sec=(
                    prediction_item.confirmation_time_sec - ground_truth_item.confirmation_time_sec
                    if prediction_item.confirmation_time_sec is not None and ground_truth_item.confirmation_time_sec is not None
                    else None
                ),
            )
        )

    unmatched_predictions = [item for index, item in enumerate(pred) if index not in used_predictions]
    unmatched_ground_truth = [item for index, item in enumerate(gt) if index not in used_ground_truth]
    types = [EventType.FORGOTTEN_OBJECT, EventType.MOVED_OBJECT]
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
                "score": item.score,
                "latency_sec": item.latency_sec,
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
    for event_type in types:
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
            len([item for item in outcome_pred if item.event_id not in outcome_prediction_ids]),
            len([item for item in outcome_gt if item.event_id not in outcome_gt_ids]),
        )

    for item in unmatched_predictions:
        category = "duplicate_event" if any(
            candidate.prediction.event_type == item.event_type and _temporal_match(item, candidate.ground_truth, config)
            for candidate in matches
        ) else _error_category(item, _closest_ground_truth(item, gt), config)
        result["error_breakdown"][category] = result["error_breakdown"].get(category, 0) + 1
    for item in unmatched_ground_truth:
        result["error_breakdown"]["missed_" + item.event_type.value.lower()] = result["error_breakdown"].get("missed_" + item.event_type.value.lower(), 0) + 1

    latencies = [item.latency_sec for item in matches if item.latency_sec is not None]
    result["latency"] = {
        "matched_events": len(latencies),
        "mean_confirmation_delta_sec": sum(latencies) / len(latencies) if latencies else None,
    }
    return result
