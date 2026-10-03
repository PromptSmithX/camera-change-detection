from __future__ import annotations

from dataclasses import replace

from change_detection.association import AssociationEngine
from change_detection.config import (
    AssociationConfig,
    ForgottenEventConfig,
    MemoryConfig,
    MovedEventConfig,
)
from change_detection.domain import (
    BaselineObject,
    BBox,
    EventType,
    MovementOutcome,
    Observation,
    Point,
)
from change_detection.events import EventEngine, EventStore
from change_detection.memory import ObjectMemory
from change_detection.scene import RegionChangeEvidence, SceneStatus


ROI = BBox(0, 0, 100, 100)


def _observation(
    tracker_id: int | None,
    bbox: BBox,
    timestamp: float,
    *,
    embedding: tuple[float, ...] = (1.0, 0.0),
    confidence: float = 0.95,
) -> Observation:
    return Observation(
        frame_index=round(timestamp * 10),
        timestamp_sec=timestamp,
        bbox=bbox,
        centroid=Point(*((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)),
        detector_class="bag",
        detector_class_id=24,
        detector_confidence=confidence,
        tracker_id=tracker_id,
        embedding=embedding,
    )


def _step(
    memory: ObjectMemory,
    engine: EventEngine,
    store: EventStore,
    observations: list[Observation],
    timestamp: float,
    *,
    scene_status: SceneStatus | None = None,
    region_evidence: RegionChangeEvidence | None = None,
):
    associations = AssociationEngine(AssociationConfig()).match(
        memory.objects,
        observations,
        roi_bbox=ROI,
        timestamp_sec=timestamp,
    )
    memory_update = memory.update(observations, associations, timestamp_sec=timestamp)
    actions = engine.update(
        memory,
        memory_update,
        observations,
        timestamp_sec=timestamp,
        scene_status=scene_status or SceneStatus(),
        roi_bbox=ROI,
        region_evidence_provider=(
            (lambda _bbox: region_evidence) if region_evidence is not None else None
        ),
    )
    results = [store.apply(action) for action in actions]
    return actions, [item for item in results if item is not None]


def _new_object_memory() -> ObjectMemory:
    ids = iter(("new-1", "new-2", "new-3"))
    return ObjectMemory(
        config=MemoryConfig(missing_grace_seconds=0.4),
        id_factory=lambda: next(ids),
    )


def _engine() -> EventEngine:
    ids = iter(("evt-1", "evt-2", "evt-3"))
    return EventEngine(
        forgotten=ForgottenEventConfig(
            candidate_seconds=0.5,
            confirm_seconds=1.0,
            disappear_grace_seconds=0.4,
            min_confidence=0.5,
            max_centroid_jitter_ratio=0.05,
            max_area_change_ratio=0.5,
        ),
        moved=MovedEventConfig(
            missing_grace_seconds=0.4,
            confirm_seconds=1.0,
            min_identity_score=0.75,
            min_displacement_ratio=0.05,
            old_location_iou_threshold=0.5,
            egress_edge_ratio=0.10,
            min_outward_speed_px_per_sec=1.0,
        ),
        event_id_factory=lambda: next(ids),
    )


def _baseline_memory() -> ObjectMemory:
    return ObjectMemory(
        [BaselineObject("base-1", BBox(10, 10, 20, 20), 24, "bag", 0.95, (1.0, 0.0))],
        config=MemoryConfig(missing_grace_seconds=0.4),
    )


def test_forgotten_stable_object_confirms_once_and_writes_public_record():
    memory = _new_object_memory()
    engine = _engine()
    store = EventStore()
    bbox = BBox(40, 40, 55, 55)

    _step(memory, engine, store, [_observation(1, bbox, 0.0)], 0.0)
    _step(memory, engine, store, [_observation(21, bbox, 0.5)], 0.5)
    _step(memory, engine, store, [_observation(21, bbox, 1.0)], 1.0)
    _step(memory, engine, store, [_observation(21, bbox, 1.5)], 1.5)

    assert len(store.confirmed_events) == 1
    event = store.confirmed_events[0]
    assert event.event_type == EventType.FORGOTTEN_OBJECT
    assert event.object_id == "new-1"
    assert event.after_bbox == bbox
    assert event.confirmed_at_sec == 1.0


def test_forgotten_short_lived_object_is_cancelled():
    memory = _new_object_memory()
    engine = _engine()
    store = EventStore()
    bbox = BBox(40, 40, 55, 55)

    _step(memory, engine, store, [_observation(1, bbox, 0.0)], 0.0)
    _step(memory, engine, store, [_observation(1, bbox, 0.2)], 0.2)
    _step(memory, engine, store, [], 1.0)

    assert store.confirmed_events == ()
    assert any(action.action.value == "cancelled" for action in store.lifecycle_actions)


def test_unconfirmed_forgotten_candidate_survives_occlusion_or_residual_dropout():
    memory = _new_object_memory()
    engine = _engine()
    store = EventStore()
    bbox = BBox(40, 40, 55, 55)

    first = replace(
        _observation(1, bbox, 0.0),
        proposal_source="baseline_residual",
    )
    second = replace(
        _observation(1, bbox, 0.5),
        proposal_source="baseline_residual",
    )
    resumed = replace(
        _observation(1, bbox, 1.5),
        proposal_source="baseline_residual",
    )
    _step(memory, engine, store, [first], 0.0)
    _step(memory, engine, store, [second], 0.5)
    _step(
        memory,
        engine,
        store,
        [],
        1.0,
        region_evidence=RegionChangeEvidence(0.8, 0.0, False),
    )
    _step(memory, engine, store, [resumed], 1.5)

    assert len(store.confirmed_events) == 1
    assert not any(
        action.reason == "short_lived_object" for action in store.lifecycle_actions
    )


def test_forgotten_tracker_id_switch_does_not_duplicate_event():
    memory = _new_object_memory()
    engine = _engine()
    store = EventStore()
    bbox = BBox(40, 40, 55, 55)

    for timestamp, tracker_id in ((0.0, 1), (0.5, 21), (1.0, 21), (1.5, 21)):
        _step(memory, engine, store, [_observation(tracker_id, bbox, timestamp)], timestamp)

    assert len(store.confirmed_events) == 1


def test_confirmed_forgotten_event_survives_occlusion_and_fragmented_identity():
    memory = _new_object_memory()
    engine = _engine()
    store = EventStore()
    bbox = BBox(40, 40, 55, 55)

    for timestamp in (0.0, 0.5, 1.0):
        _step(memory, engine, store, [_observation(1, bbox, timestamp)], timestamp)
    occluded = RegionChangeEvidence(0.0, 0.8, False)
    _step(memory, engine, store, [], 1.5, region_evidence=occluded)
    _step(memory, engine, store, [], 5.5, region_evidence=occluded)
    _step(
        memory,
        engine,
        store,
        [_observation(21, bbox, 6.0, embedding=(0.0, 1.0))],
        6.0,
        region_evidence=RegionChangeEvidence(0.8, 0.0, False),
    )

    assert len(store.confirmed_events) == 1
    assert len(store.active_events) == 1
    assert [item.action.value for item in store.lifecycle_actions].count("confirmed") == 1


def test_confirmed_forgotten_event_closes_only_after_baseline_is_restored():
    memory = _new_object_memory()
    engine = _engine()
    store = EventStore()
    bbox = BBox(40, 40, 55, 55)

    for timestamp in (0.0, 0.5, 1.0):
        _step(memory, engine, store, [_observation(1, bbox, timestamp)], timestamp)
    restored = RegionChangeEvidence(0.0, 0.0, True)
    _step(memory, engine, store, [], 1.5, region_evidence=restored)
    _step(memory, engine, store, [], 6.4, region_evidence=restored)
    assert len(store.active_events) == 1

    _step(memory, engine, store, [], 6.5, region_evidence=restored)
    assert store.active_events == ()
    assert store.confirmed_events[0].ended_at_sec == 6.5
    assert store.lifecycle_actions[-1].reason == "baseline_restored"


def test_forgotten_low_confidence_interval_does_not_advance_timer():
    memory = _new_object_memory()
    engine = _engine()
    store = EventStore()
    bbox = BBox(40, 40, 55, 55)

    _step(memory, engine, store, [_observation(1, bbox, 0.0)], 0.0)
    _step(memory, engine, store, [_observation(1, bbox, 0.5, confidence=0.2)], 0.5)
    _step(memory, engine, store, [_observation(1, bbox, 1.0)], 1.0)

    assert store.confirmed_events == ()


def test_masked_overlap_evidence_on_baseline_does_not_confirm_relocation():
    memory = _baseline_memory()
    engine = _engine()
    store = EventStore()
    old_bbox = BBox(10, 10, 20, 20)
    displaced_bbox = BBox(20, 10, 30, 20)

    _step(memory, engine, store, [_observation(1, old_bbox, 0.0)], 0.0)
    for timestamp in (0.5, 1.0, 1.5):
        observation = replace(
            _observation(1, displaced_bbox, timestamp),
            proposal_source="yolo_reference_change",
            event_candidate=True,
        )
        _step(memory, engine, store, [observation], timestamp)

    assert not any(event.event_type == EventType.MOVED_OBJECT for event in store.confirmed_events)


def test_forgotten_timer_freezes_during_scene_anomaly():
    memory = _new_object_memory()
    engine = _engine()
    store = EventStore()
    bbox = BBox(40, 40, 55, 55)

    _step(memory, engine, store, [_observation(1, bbox, 0.0)], 0.0)
    _step(
        memory,
        engine,
        store,
        [_observation(1, bbox, 0.5)],
        0.5,
        scene_status=SceneStatus(stable=False, lighting_change=True, reason="lighting"),
    )
    _step(memory, engine, store, [_observation(1, bbox, 1.0)], 1.0)
    assert store.confirmed_events == ()

    _step(memory, engine, store, [_observation(1, bbox, 1.5)], 1.5)
    assert len(store.confirmed_events) == 1


def test_moved_relocated_requires_identity_and_confirms_with_from_to_boxes():
    memory = _baseline_memory()
    engine = _engine()
    store = EventStore()
    old_bbox = BBox(10, 10, 20, 20)
    new_bbox = BBox(60, 60, 70, 70)

    _step(memory, engine, store, [_observation(1, old_bbox, 0.0)], 0.0)
    _step(memory, engine, store, [_observation(2, new_bbox, 0.5)], 0.5)
    _step(memory, engine, store, [_observation(2, new_bbox, 1.0)], 1.0)
    _step(memory, engine, store, [_observation(2, new_bbox, 1.5)], 1.5)

    assert len(store.confirmed_events) == 1
    event = store.confirmed_events[0]
    assert event.event_type == EventType.MOVED_OBJECT
    assert event.movement_outcome == MovementOutcome.RELOCATED
    assert event.before_bbox == old_bbox
    assert event.after_bbox == new_bbox


def test_moved_unrelated_object_does_not_create_event():
    memory = _baseline_memory()
    engine = _engine()
    store = EventStore()
    old_bbox = BBox(10, 10, 20, 20)
    unrelated_bbox = BBox(60, 60, 70, 70)

    _step(memory, engine, store, [_observation(1, old_bbox, 0.0)], 0.0)
    _step(
        memory,
        engine,
        store,
        [_observation(2, unrelated_bbox, 0.5, embedding=(0.0, 1.0))],
        0.5,
    )
    _step(
        memory,
        engine,
        store,
        [_observation(2, unrelated_bbox, 1.5, embedding=(0.0, 1.0))],
        1.5,
    )

    assert not any(item.event_type == EventType.MOVED_OBJECT for item in store.confirmed_events)


def test_moved_candidate_is_cancelled_when_object_returns_to_baseline():
    memory = _baseline_memory()
    engine = _engine()
    store = EventStore()
    old_bbox = BBox(10, 10, 20, 20)
    new_bbox = BBox(60, 60, 70, 70)

    _step(memory, engine, store, [_observation(1, old_bbox, 0.0)], 0.0)
    _step(memory, engine, store, [_observation(2, new_bbox, 0.5)], 0.5)
    _step(memory, engine, store, [_observation(2, old_bbox, 1.0)], 1.0)

    assert store.confirmed_events == ()
    assert any(action.action.value == "cancelled" for action in store.lifecycle_actions)


def test_moved_temporary_disappearance_without_egress_is_not_an_event():
    memory = _baseline_memory()
    engine = _engine()
    store = EventStore()
    old_bbox = BBox(10, 10, 20, 20)

    _step(memory, engine, store, [_observation(1, old_bbox, 0.0)], 0.0)
    _step(memory, engine, store, [], 0.5)
    _step(memory, engine, store, [], 1.0)

    assert store.confirmed_events == ()
    assert not any(item.event.event_type == EventType.MOVED_OBJECT for item in store.lifecycle_actions)


def test_left_scene_requires_visible_outward_egress_evidence():
    memory = _baseline_memory()
    engine = _engine()
    store = EventStore()
    old_bbox = BBox(10, 10, 20, 20)

    _step(memory, engine, store, [_observation(1, old_bbox, 0.4)], 0.4)
    _step(memory, engine, store, [_observation(1, BBox(35, 20, 45, 30), 0.5)], 0.5)
    _step(memory, engine, store, [_observation(1, BBox(60, 30, 70, 40), 0.6)], 0.6)
    _step(memory, engine, store, [_observation(1, BBox(85, 40, 95, 50), 0.7)], 0.7)
    _step(memory, engine, store, [], 0.8)
    _step(memory, engine, store, [], 1.4)
    _step(memory, engine, store, [], 2.4)

    assert len(store.confirmed_events) == 1
    assert store.confirmed_events[0].movement_outcome == MovementOutcome.LEFT_SCENE
    assert store.confirmed_events[0].after_bbox is None
