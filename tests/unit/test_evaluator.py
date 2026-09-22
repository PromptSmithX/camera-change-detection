from change_detection.domain import EventAnnotation, EventType
from change_detection.evaluation import EvaluationConfig, evaluate_events


def forgotten(event_id: str, start: float = 10.0, confirmation: float | None = 14.0) -> EventAnnotation:
    return EventAnnotation.from_dict(
        {
            "event_id": event_id,
            "type": "FORGOTTEN_OBJECT",
            "object_id": "obj-1",
            "start_time_sec": start,
            "confirmation_time_sec": confirmation,
            "end_time_sec": None,
            "bbox": [100, 100, 160, 160],
        }
    )


def moved(event_id: str, start: float = 10.0) -> EventAnnotation:
    return EventAnnotation.from_dict(
        {
            "event_id": event_id,
            "type": "MOVED_OBJECT",
            "object_id": "obj-1",
            "start_time_sec": start,
            "confirmation_time_sec": 14.0,
            "end_time_sec": 20.0,
            "baseline_bbox": [10, 10, 50, 50],
            "new_bbox": [100, 100, 150, 150],
        }
    )


def moved_left_scene(event_id: str, start: float = 10.0) -> EventAnnotation:
    return EventAnnotation.from_dict(
        {
            "event_id": event_id,
            "type": "MOVED_OBJECT",
            "object_id": "obj-1",
            "start_time_sec": start,
            "confirmation_time_sec": 14.0,
            "end_time_sec": 20.0,
            "baseline_bbox": [10, 10, 50, 50],
            "new_bbox": None,
            "movement_outcome": "left_scene",
        }
    )


def test_event_evaluator_matches_one_event_and_reports_latency():
    result = evaluate_events(
        [forgotten("gt-1")],
        [forgotten("pred-1", start=10.5, confirmation=15.0)],
    )
    assert result["overall"]["tp"] == 1
    assert result["overall"]["fp"] == 0
    assert result["overall"]["fn"] == 0
    assert result["FORGOTTEN_OBJECT"]["f1"] == 1.0
    assert result["latency"]["mean_confirmation_delta_sec"] == 1.0


def test_duplicate_prediction_is_one_tp_and_one_fp():
    result = evaluate_events(
        [forgotten("gt-1")],
        [forgotten("pred-1"), forgotten("pred-2")],
    )
    assert result["overall"]["tp"] == 1
    assert result["overall"]["fp"] == 1
    assert result["error_breakdown"]["duplicate_event"] == 1


def test_moved_event_requires_both_locations():
    result = evaluate_events([moved("gt-1")], [moved("pred-1")])
    assert result["MOVED_OBJECT"]["tp"] == 1

    incomplete = moved("pred-2")
    incomplete.new_bbox = None
    result = evaluate_events([moved("gt-1")], [incomplete])
    assert result["MOVED_OBJECT"]["tp"] == 0
    assert result["MOVED_OBJECT"]["fp"] == 1
    assert result["MOVED_OBJECT"]["fn"] == 1


def test_left_scene_moved_event_matches_on_baseline_and_reports_outcome():
    result = evaluate_events([moved_left_scene("gt-1")], [moved_left_scene("pred-1")])
    assert result["MOVED_OBJECT"]["tp"] == 1
    assert result["MOVED_OBJECT_OUTCOMES"]["left_scene"]["tp"] == 1


def test_moved_outcomes_do_not_match_each_other():
    result = evaluate_events([moved_left_scene("gt-1")], [moved("pred-1")])
    assert result["MOVED_OBJECT"]["tp"] == 0
    assert result["error_breakdown"]["movement_outcome_mismatch"] == 1


def test_wrong_type_does_not_match():
    result = evaluate_events([forgotten("gt-1")], [moved("pred-1")])
    assert result["overall"] == {"tp": 0, "fp": 1, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert result["error_breakdown"]["wrong_type"] == 1


def test_open_ended_event_can_match_by_start_time_and_spatial_evidence():
    ground_truth = forgotten("gt-1", start=10.0, confirmation=None)
    prediction = forgotten("pred-1", start=12.5, confirmation=None)
    result = evaluate_events([ground_truth], [prediction])
    assert result["overall"]["tp"] == 1
    assert result["latency"]["matched_events"] == 0


def test_spatial_mismatch_is_reported():
    prediction = forgotten("pred-1")
    prediction.bbox = type(prediction.bbox)(300, 300, 360, 360)
    result = evaluate_events([forgotten("gt-1")], [prediction])
    assert result["overall"]["fp"] == 1
    assert result["error_breakdown"]["spatial_mismatch"] == 1
