from change_detection.domain import AnnotationStatus, BBox, EventAnnotation, EventType, MovementOutcome, can_transition_quality


def test_bbox_is_integer_half_open_and_iou_is_deterministic():
    box = BBox.from_value([10, 20, 30, 40])
    assert box is not None
    assert box.width == 20
    assert box.height == 20
    assert box.area == 400
    assert box.is_valid_for(100, 100)
    assert box.iou(BBox(20, 20, 40, 40)) == 1 / 3


def test_event_round_trip_preserves_contract_fields():
    event = EventAnnotation.from_dict(
        {
            "event_id": "evt-1",
            "type": "MOVED_OBJECT",
            "object_id": "obj-1",
            "start_time_sec": 12.0,
            "confirmation_time_sec": 15.0,
            "end_time_sec": None,
            "baseline_bbox": [10, 20, 30, 40],
            "new_bbox": [50, 60, 80, 100],
            "start_frame": 300,
            "confirmation_frame": 375,
        }
    )
    value = event.to_dict()
    assert value["event_id"] == "evt-1"
    assert value["type"] == EventType.MOVED_OBJECT.value
    assert value["baseline_bbox"] == [10, 20, 30, 40]
    assert value["new_bbox"] == [50, 60, 80, 100]
    assert value["movement_outcome"] == MovementOutcome.RELOCATED.value
    assert value["confirmation_frame"] == 375


def test_left_scene_outcome_round_trips_without_new_bbox():
    event = EventAnnotation.from_dict(
        {
            "event_id": "evt-left-scene",
            "type": "MOVED_OBJECT",
            "object_id": "obj-1",
            "start_time_sec": 12.0,
            "confirmation_time_sec": 15.0,
            "end_time_sec": None,
            "baseline_bbox": [10, 20, 30, 40],
            "new_bbox": None,
            "movement_outcome": "left_scene",
        }
    )
    assert event.movement_outcome == MovementOutcome.LEFT_SCENE
    assert event.to_dict()["new_bbox"] is None


def test_quality_state_transitions_are_explicit():
    assert can_transition_quality(AnnotationStatus.PROVISIONAL, AnnotationStatus.VERIFIED)
    assert can_transition_quality("excluded", "verified")
    assert not can_transition_quality(AnnotationStatus.VERIFIED, AnnotationStatus.PROVISIONAL)
