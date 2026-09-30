from change_detection.domain import EventAnnotation, EventType
from change_detection.evaluation import EvaluationConfig, evaluate_events


def forgotten(
    event_id: str,
    start: float = 10.0,
    confirmation: float | None = 14.0,
    end: float | None = 30.0,
) -> EventAnnotation:
    return EventAnnotation.from_dict(
        {
            "event_id": event_id,
            "type": "FORGOTTEN_OBJECT",
            "object_id": "obj-1",
            "start_time_sec": start,
            "confirmation_time_sec": confirmation,
            "end_time_sec": end,
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


def test_event_evaluator_matches_one_event_and_reports_timeliness():
    result = evaluate_events(
        [forgotten("gt-1")],
        [forgotten("pred-1", start=10.5, confirmation=15.0)],
    )
    assert result["overall"]["tp"] == 1
    assert result["overall"]["fp"] == 0
    assert result["overall"]["fn"] == 0
    assert result["FORGOTTEN_OBJECT"]["f1"] == 1.0
    assert result["latency"]["mean_confirmation_delta_sec"] == 1.0
    assert result["timeliness"]["deadlines"]["3s"]["on_time_count"] == 1


def test_late_start_still_matches_when_confirmation_is_in_ground_truth_window():
    prediction = forgotten("pred-1", start=24.0, confirmation=28.0)
    result = evaluate_events([forgotten("gt-1")], [prediction])

    assert result["overall"]["tp"] == 1
    assert result["matches"][0]["start_delay_sec"] == 14.0
    assert result["matches"][0]["confirmation_latency_sec"] == 14.0
    assert result["timeliness"]["deadlines"]["10s"]["late_count"] == 1


def test_active_prediction_without_end_time_can_match():
    prediction = forgotten("pred-1", confirmation=16.0, end=None)
    result = evaluate_events([forgotten("gt-1")], [prediction])
    assert result["overall"]["tp"] == 1


def test_prediction_requires_confirmation_and_must_be_inside_event_window():
    unconfirmed = forgotten("pred-1", confirmation=None)
    result = evaluate_events([forgotten("gt-1")], [unconfirmed])
    assert result["overall"]["tp"] == 0
    assert result["error_breakdown"]["missing_confirmation_time"] == 1

    after_removal = forgotten("pred-2", start=31.0, confirmation=32.0, end=None)
    result = evaluate_events([forgotten("gt-1")], [after_removal])
    assert result["overall"]["tp"] == 0
    assert result["error_breakdown"]["outside_event_window"] == 1


def test_duplicate_prediction_is_one_tp_and_one_fp():
    result = evaluate_events(
        [forgotten("gt-1")],
        [forgotten("pred-1"), forgotten("pred-2")],
    )
    assert result["overall"]["tp"] == 1
    assert result["overall"]["fp"] == 1
    assert result["error_breakdown"]["duplicate_event"] == 1


def test_nearest_confirmation_wins_before_higher_iou_duplicate():
    ground_truth = forgotten("gt-1", confirmation=20.0, end=100.0)
    late = forgotten("pred-late", start=55.0, confirmation=60.0, end=None)
    near = forgotten("pred-near", start=20.0, confirmation=24.0, end=None)
    near.bbox = type(near.bbox)(102, 102, 162, 162)

    result = evaluate_events([ground_truth], [late, near])

    assert result["matches"][0]["prediction_event_id"] == "pred-near"
    assert result["matches"][0]["confirmation_latency_sec"] == 4.0
    assert result["error_breakdown"]["duplicate_event"] == 1


def test_early_confirmation_after_event_start_is_valid():
    prediction = forgotten("pred-1", start=11.0, confirmation=12.0)
    result = evaluate_events([forgotten("gt-1", confirmation=14.0)], [prediction])
    assert result["overall"]["tp"] == 1
    assert result["timeliness"]["early_count"] == 1


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
    assert result["overall"] == {
        "tp": 0,
        "fp": 1,
        "fn": 1,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }
    assert result["error_breakdown"]["wrong_type"] == 1


def test_spatial_mismatch_is_reported():
    prediction = forgotten("pred-1")
    prediction.bbox = type(prediction.bbox)(300, 300, 360, 360)
    result = evaluate_events([forgotten("gt-1")], [prediction])
    assert result["overall"]["fp"] == 1
    assert result["error_breakdown"]["spatial_mismatch"] == 1


def test_custom_boundary_tolerance_and_deadlines_are_applied():
    prediction = forgotten("pred-1", start=30.0, confirmation=31.5, end=None)
    result = evaluate_events(
        [forgotten("gt-1", end=30.0)],
        [prediction],
        config=EvaluationConfig(
            event_boundary_tolerance_seconds=2.0,
            latency_deadlines_seconds=(5.0, 20.0),
        ),
    )
    assert result["overall"]["tp"] == 1
    assert set(result["timeliness"]["deadlines"]) == {"5s", "20s"}
