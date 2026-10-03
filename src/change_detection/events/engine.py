"""Pure deterministic FSMs for forgotten and moved-object events."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Callable
from uuid import uuid4

from change_detection.config import ForgottenEventConfig, MovedEventConfig
from change_detection.domain import (
    BBox,
    EventAction,
    EventActionType,
    EventLifecycle,
    EventRecord,
    EventType,
    MemoryObject,
    MovementOutcome,
    Observation,
    ObjectState,
)
from change_detection.memory import MemoryUpdateResult, ObjectMemory
from change_detection.scene import RegionChangeEvidence, SceneStatus


def _center(bbox: BBox) -> tuple[float, float]:
    return (float(bbox.x1 + bbox.x2) / 2.0, float(bbox.y1 + bbox.y2) / 2.0)


def _area_change(left: BBox, right: BBox) -> float:
    return abs(float(left.area - right.area)) / max(float(left.area), 1.0)


def _displacement_ratio(left: BBox, right: BBox, roi_bbox: BBox) -> float:
    x1, y1 = _center(left)
    x2, y2 = _center(right)
    diagonal = max(1.0, hypot(float(roi_bbox.width), float(roi_bbox.height)))
    return hypot(x2 - x1, y2 - y1) / diagonal


def _history_bbox(item: object) -> BBox | None:
    if not isinstance(item, dict):
        return None
    try:
        return BBox.from_value(item.get("bbox"))
    except (TypeError, ValueError):
        return None


def _history_time(item: object) -> float | None:
    if not isinstance(item, dict):
        return None
    value = item.get("timestamp_sec")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _has_egress_evidence(
    memory: MemoryObject,
    roi_bbox: BBox,
    config: MovedEventConfig,
    *,
    timestamp_sec: float,
) -> bool:
    """Require sustained, directional travel to an ROI edge before left_scene."""

    history = memory.observation_history
    parsed: list[tuple[BBox, float]] = []
    for item in history:
        bbox = _history_bbox(item)
        timestamp = _history_time(item)
        if bbox is not None and timestamp is not None:
            parsed.append((bbox, timestamp))
    if len(parsed) < 4:
        return False

    if timestamp_sec - parsed[-1][1] > 2.0:
        return False
    parsed = [item for item in parsed if 0.0 <= timestamp_sec - item[1] <= 2.0]
    if len(parsed) < 4 or parsed[-1][1] - parsed[0][1] <= 0:
        return False

    edge_x = max(1.0, roi_bbox.width * config.egress_edge_ratio)
    edge_y = max(1.0, roi_bbox.height * config.egress_edge_ratio)
    end_bbox = parsed[-1][0]
    edge_gaps = {
        "left": end_bbox.x1 - roi_bbox.x1,
        "right": roi_bbox.x2 - end_bbox.x2,
        "top": end_bbox.y1 - roi_bbox.y1,
        "bottom": roi_bbox.y2 - end_bbox.y2,
    }
    eligible_edges = {
        "left": edge_x,
        "right": edge_x,
        "top": edge_y,
        "bottom": edge_y,
    }
    edge = min(edge_gaps, key=lambda name: edge_gaps[name] / eligible_edges[name])
    if edge_gaps[edge] > eligible_edges[edge]:
        return False

    centers = [_center(item[0]) for item in parsed]
    if edge == "left":
        progress = [left[0] - right[0] for left, right in zip(centers[:-1], centers[1:], strict=False)]
        net_outward = centers[0][0] - centers[-1][0]
    elif edge == "right":
        progress = [right[0] - left[0] for left, right in zip(centers[:-1], centers[1:], strict=False)]
        net_outward = centers[-1][0] - centers[0][0]
    elif edge == "top":
        progress = [left[1] - right[1] for left, right in zip(centers[:-1], centers[1:], strict=False)]
        net_outward = centers[0][1] - centers[-1][1]
    else:
        progress = [right[1] - left[1] for left, right in zip(centers[:-1], centers[1:], strict=False)]
        net_outward = centers[-1][1] - centers[0][1]

    elapsed = parsed[-1][1] - parsed[0][1]
    directional_steps = sum(delta >= 0.5 for delta in progress)
    required_displacement = max(4.0, 0.03 * hypot(float(roi_bbox.width), float(roi_bbox.height)))
    return (
        net_outward >= required_displacement
        and directional_steps / max(1, len(progress)) >= 0.70
        and net_outward / elapsed >= config.min_outward_speed_px_per_sec
    )


@dataclass(slots=True)
class _ForgottenState:
    event: EventRecord
    anchor_bbox: BBox
    last_observed_sec: float
    eligible_seconds: float = 0.0
    candidate_created: bool = False
    confirmed: bool = False
    confidence: float = 0.0
    resolution_started_sec: float | None = None
    proposal_source: str = "yolo"


@dataclass(slots=True)
class _MovedState:
    event: EventRecord
    last_evidence_sec: float
    eligible_seconds: float = 0.0
    candidate_created: bool = False
    confirmed: bool = False


class EventEngine:
    """Coordinate independent M4/M5 state machines over ``ObjectMemory``."""

    def __init__(
        self,
        *,
        forgotten: ForgottenEventConfig | None = None,
        moved: MovedEventConfig | None = None,
        event_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.forgotten = forgotten or ForgottenEventConfig()
        self.moved = moved or MovedEventConfig()
        self._event_id_factory = event_id_factory or (lambda: f"evt_{uuid4().hex}")
        self._forgotten_states: dict[str, _ForgottenState] = {}
        self._moved_states: dict[str, _MovedState] = {}
        self._last_timestamp_sec: float | None = None

    def reset(self) -> None:
        self._forgotten_states.clear()
        self._moved_states.clear()
        self._last_timestamp_sec = None

    @staticmethod
    def _action(
        action: EventActionType,
        event: EventRecord,
        reason: str | None = None,
    ) -> EventAction:
        return EventAction(action, event, reason)

    def _stable_new_object(
        self,
        state: _ForgottenState,
        observation: Observation,
        roi_bbox: BBox,
    ) -> bool:
        anchor_x, anchor_y = _center(state.anchor_bbox)
        current_x, current_y = _center(observation.bbox)
        diagonal = max(1.0, hypot(float(roi_bbox.width), float(roi_bbox.height)))
        jitter = hypot(current_x - anchor_x, current_y - anchor_y) / diagonal
        return (
            jitter <= self.forgotten.max_centroid_jitter_ratio
            and _area_change(state.anchor_bbox, observation.bbox)
            <= self.forgotten.max_area_change_ratio
        )

    def _rebind_confirmed_forgotten_states(
        self,
        assigned: dict[str, tuple[Observation, float | None]],
    ) -> None:
        """Attach a fragmented identity to an unresolved event in the same region."""

        claimed: set[int] = set()
        for object_id, (observation, _score) in assigned.items():
            if object_id in self._forgotten_states or not observation.event_candidate:
                continue
            candidates: list[tuple[float, str, _ForgottenState]] = []
            for previous_id, state in self._forgotten_states.items():
                if not state.confirmed or id(state) in claimed:
                    continue
                bbox = state.event.after_bbox or state.anchor_bbox
                overlap = bbox.iou(observation.bbox)
                if overlap >= self.forgotten.reid_iou_threshold:
                    candidates.append((overlap, previous_id, state))
            if not candidates:
                continue
            _overlap, previous_id, state = max(candidates, key=lambda item: item[0])
            self._forgotten_states.pop(previous_id, None)
            self._forgotten_states[object_id] = state
            state.last_observed_sec = observation.timestamp_sec
            state.resolution_started_sec = None
            claimed.add(id(state))

    def _update_forgotten(
        self,
        memory: MemoryObject,
        observation: Observation | None,
        *,
        timestamp_sec: float,
        scene_status: SceneStatus,
        roi_bbox: BBox,
        region_evidence_provider: Callable[[BBox], RegionChangeEvidence | None] | None,
    ) -> list[EventAction]:
        actions: list[EventAction] = []
        state = self._forgotten_states.get(memory.object_id)
        if observation is not None and not observation.event_candidate:
            # A raw YOLO box (or a tracker prediction) is useful for keeping
            # identity alive, but it is not evidence for a change event and
            # must not be mistaken for disappearance either.
            if state is not None and not state.confirmed:
                state.last_observed_sec = timestamp_sec
                state.event.after_bbox = observation.bbox
            return actions
        if observation is not None:
            if observation.detector_confidence < self.forgotten.min_confidence:
                # It is still an observation for disappearance purposes, but
                # it must not contribute the low-confidence interval to the
                # stability timer when a later high-confidence detection
                # arrives.
                if state is not None:
                    state.last_observed_sec = timestamp_sec
                return actions
            if not scene_status.event_logic_enabled:
                if state is not None:
                    state.last_observed_sec = timestamp_sec
                    state.event.after_bbox = observation.bbox
                return actions
            if state is None:
                event = EventRecord(
                    event_id=self._event_id_factory(),
                    event_type=EventType.FORGOTTEN_OBJECT,
                    object_id=memory.object_id,
                    started_at_sec=timestamp_sec,
                    confidence=float(observation.detector_confidence),
                    after_bbox=observation.bbox,
                )
                state = _ForgottenState(
                    event=event,
                    anchor_bbox=observation.bbox,
                    last_observed_sec=timestamp_sec,
                    confidence=float(observation.detector_confidence),
                    proposal_source=observation.proposal_source,
                )
                self._forgotten_states[memory.object_id] = state
            elif not self._stable_new_object(state, observation, roi_bbox) and not state.confirmed:
                if state.candidate_created:
                    actions.append(self._action(EventActionType.CANCELLED, state.event, "unstable_object"))
                self._forgotten_states.pop(memory.object_id, None)
                return actions

            gap = max(0.0, timestamp_sec - state.last_observed_sec)
            # A timestamp gap between two observed frames is still positive
            # evidence that the object remained visible.  ``disappear_grace``
            # is only the no-observation grace period below; using it here
            # would freeze the normal timer whenever processing FPS is lower
            # than the configured grace window.
            if scene_status.event_logic_enabled:
                state.eligible_seconds += gap
            state.last_observed_sec = timestamp_sec
            state.resolution_started_sec = None
            state.confidence = max(state.confidence, float(observation.detector_confidence))
            state.event.after_bbox = observation.bbox
            state.event.confidence = state.confidence

            if not state.candidate_created and state.eligible_seconds >= self.forgotten.candidate_seconds:
                state.candidate_created = True
                state.event.lifecycle = EventLifecycle.CREATED
                actions.append(self._action(EventActionType.CREATED, state.event, "stable_candidate"))
            if not state.confirmed and state.eligible_seconds >= self.forgotten.confirm_seconds:
                state.confirmed = True
                state.event.confirmed_at_sec = timestamp_sec
                state.event.lifecycle = EventLifecycle.ACTIVE
                actions.append(self._action(EventActionType.CONFIRMED, state.event, "stable_confirmed"))
            return actions

        if state is None:
            return actions
        if not scene_status.event_logic_enabled:
            # Do not let a long anomaly interval look like a disappearance
            # when normal event logic resumes.
            state.last_observed_sec = timestamp_sec
            return actions
        missing_duration = max(0.0, timestamp_sec - state.last_observed_sec)
        if missing_duration <= self.forgotten.disappear_grace_seconds:
            return actions
        if state.confirmed:
            evidence_bbox = state.event.after_bbox or state.anchor_bbox
            evidence = (
                region_evidence_provider(evidence_bbox)
                if region_evidence_provider is not None
                else None
            )
            if evidence is None:
                state.resolution_started_sec = None
                return actions
            if (
                evidence.person_overlap_ratio
                >= self.forgotten.occlusion_overlap_threshold
                or not evidence.baseline_restored
            ):
                state.resolution_started_sec = None
                return actions
            if state.resolution_started_sec is None:
                state.resolution_started_sec = timestamp_sec
                return actions
            if (
                timestamp_sec - state.resolution_started_sec
                < self.forgotten.resolution_confirm_seconds
            ):
                return actions
            state.event.ended_at_sec = timestamp_sec
            state.event.lifecycle = EventLifecycle.CLOSED
            actions.append(
                self._action(EventActionType.CLOSED, state.event, "baseline_restored")
            )
        else:
            evidence_bbox = state.event.after_bbox or state.anchor_bbox
            evidence = (
                region_evidence_provider(evidence_bbox)
                if region_evidence_provider is not None
                else None
            )
            if (
                state.proposal_source
                in {"baseline_residual", "fused_baseline_residual"}
                and evidence is not None
                and (
                    evidence.person_overlap_ratio
                    >= self.forgotten.occlusion_overlap_threshold
                    or not evidence.baseline_restored
                )
            ):
                # Freeze an unconfirmed candidate while the pixels still show
                # change or a person hides the region.  Missing detector output
                # alone is not proof that a persistent residual disappeared.
                state.last_observed_sec = timestamp_sec
                return actions
            actions.append(self._action(EventActionType.CANCELLED, state.event, "short_lived_object"))
        self._forgotten_states.pop(memory.object_id, None)
        return actions

    def _update_moved(
        self,
        memory: MemoryObject,
        observation: Observation | None,
        assignment_score: float | None,
        current_observations: list[tuple[str, Observation]],
        *,
        timestamp_sec: float,
        scene_status: SceneStatus,
        roi_bbox: BBox,
    ) -> list[EventAction]:
        actions: list[EventAction] = []
        state = self._moved_states.get(memory.object_id)
        baseline_bbox = memory.baseline_bbox
        if baseline_bbox is None:
            return actions
        if not scene_status.event_logic_enabled:
            if state is not None:
                state.last_evidence_sec = timestamp_sec
            return actions
        if observation is not None and (
            not observation.event_candidate
            or observation.proposal_source == "yolo_reference_change"
        ):
            # Masked-overlap evidence can support a new forgotten object, but
            # it does not establish that an existing baseline object moved.
            # Keep these observations in memory for trajectory/re-ID while
            # requiring a contour-based proposal for relocation evidence.
            displaced = (
                _displacement_ratio(baseline_bbox, observation.bbox, roi_bbox)
                >= self.moved.min_displacement_ratio
            )
            if not displaced:
                if state is not None and not state.confirmed:
                    actions.append(
                        self._action(EventActionType.CANCELLED, state.event, "returned_or_invalid_move")
                    )
                    self._moved_states.pop(memory.object_id, None)
                elif state is not None and state.confirmed:
                    state.event.ended_at_sec = timestamp_sec
                    state.event.lifecycle = EventLifecycle.CLOSED
                    actions.append(
                        self._action(EventActionType.CLOSED, state.event, "returned_to_baseline")
                    )
                    self._moved_states.pop(memory.object_id, None)
                return actions
            observation = None
            assignment_score = None

        if observation is not None and assignment_score is not None:
            displaced = _displacement_ratio(baseline_bbox, observation.bbox, roi_bbox) >= self.moved.min_displacement_ratio
            old_location_occupied = any(
                object_id != memory.object_id
                and candidate_observation.bbox.iou(baseline_bbox) >= self.moved.old_location_iou_threshold
                for object_id, candidate_observation in current_observations
            )
            valid_candidate = (
                displaced
                and assignment_score >= self.moved.min_identity_score
                and not old_location_occupied
            )
            if not valid_candidate:
                if state is not None and not state.confirmed:
                    actions.append(self._action(EventActionType.CANCELLED, state.event, "returned_or_invalid_move"))
                    self._moved_states.pop(memory.object_id, None)
                elif state is not None and state.confirmed and not displaced:
                    state.event.ended_at_sec = timestamp_sec
                    state.event.lifecycle = EventLifecycle.CLOSED
                    actions.append(self._action(EventActionType.CLOSED, state.event, "returned_to_baseline"))
                    self._moved_states.pop(memory.object_id, None)
                return actions

            if state is None:
                event = EventRecord(
                    event_id=self._event_id_factory(),
                    event_type=EventType.MOVED_OBJECT,
                    object_id=memory.object_id,
                    started_at_sec=timestamp_sec,
                    confidence=float(assignment_score),
                    before_bbox=baseline_bbox,
                    after_bbox=observation.bbox,
                    movement_outcome=MovementOutcome.RELOCATED,
                )
                state = _MovedState(event=event, last_evidence_sec=timestamp_sec)
                self._moved_states[memory.object_id] = state
            else:
                state.event.after_bbox = observation.bbox
                state.event.confidence = max(state.event.confidence, float(assignment_score))

            gap = max(0.0, timestamp_sec - state.last_evidence_sec)
            if scene_status.event_logic_enabled:
                state.eligible_seconds += gap
            state.last_evidence_sec = timestamp_sec
            if not state.candidate_created:
                state.candidate_created = True
                actions.append(self._action(EventActionType.CREATED, state.event, "identity_displaced"))
            if not state.confirmed and state.eligible_seconds >= self.moved.confirm_seconds:
                state.confirmed = True
                state.event.confirmed_at_sec = timestamp_sec
                state.event.lifecycle = EventLifecycle.ACTIVE
                actions.append(self._action(EventActionType.CONFIRMED, state.event, "relocated_confirmed"))
            return actions

        if state is not None and not state.confirmed:
            if memory.state == ObjectState.PRESENT:
                return actions
            if not _has_egress_evidence(
                memory,
                roi_bbox,
                self.moved,
                timestamp_sec=timestamp_sec,
            ):
                if memory.missing_since_sec is not None and timestamp_sec - memory.missing_since_sec > self.moved.missing_grace_seconds:
                    actions.append(self._action(EventActionType.CANCELLED, state.event, "missing_without_egress"))
                    self._moved_states.pop(memory.object_id, None)
                return actions

        if state is not None and state.confirmed:
            return actions

        if memory.state in {ObjectState.TEMP_MISSING, ObjectState.STALE} and _has_egress_evidence(
            memory,
            roi_bbox,
            self.moved,
            timestamp_sec=timestamp_sec,
        ):
            missing_since = memory.missing_since_sec
            if missing_since is None or timestamp_sec - missing_since < self.moved.missing_grace_seconds:
                return actions
            if state is None or state.event.movement_outcome != MovementOutcome.LEFT_SCENE:
                event = EventRecord(
                    event_id=self._event_id_factory(),
                    event_type=EventType.MOVED_OBJECT,
                    object_id=memory.object_id,
                    started_at_sec=missing_since,
                    confidence=1.0,
                    before_bbox=baseline_bbox,
                    movement_outcome=MovementOutcome.LEFT_SCENE,
                )
                state = _MovedState(event=event, last_evidence_sec=timestamp_sec)
                self._moved_states[memory.object_id] = state
                actions.append(self._action(EventActionType.CREATED, state.event, "visible_egress"))
            else:
                gap = max(0.0, timestamp_sec - state.last_evidence_sec)
                if scene_status.event_logic_enabled:
                    state.eligible_seconds += gap
                state.last_evidence_sec = timestamp_sec
            if state is not None and not state.confirmed and state.eligible_seconds >= self.moved.confirm_seconds:
                state.confirmed = True
                state.event.confirmed_at_sec = timestamp_sec
                state.event.lifecycle = EventLifecycle.ACTIVE
                actions.append(self._action(EventActionType.CONFIRMED, state.event, "left_scene_confirmed"))
        return actions

    def update(
        self,
        memory: ObjectMemory,
        memory_update: MemoryUpdateResult,
        observations: list[Observation],
        *,
        timestamp_sec: float,
        scene_status: SceneStatus,
        roi_bbox: BBox,
        region_evidence_provider: Callable[[BBox], RegionChangeEvidence | None] | None = None,
    ) -> tuple[EventAction, ...]:
        """Advance both FSMs using one completed memory update."""

        if self._last_timestamp_sec is not None and timestamp_sec < self._last_timestamp_sec:
            raise ValueError("EventEngine timestamps must be monotonic")
        assigned: dict[str, tuple[Observation, float | None]] = {}
        for assignment in memory_update.assignments:
            if 0 <= assignment.observation_index < len(observations):
                assigned[assignment.object_id] = (
                    observations[assignment.observation_index],
                    assignment.score,
                )
        current_observations = [
            (object_id, observation)
            for object_id, (observation, _score) in assigned.items()
        ]
        self._rebind_confirmed_forgotten_states(assigned)
        actions: list[EventAction] = []
        for object_item in memory.objects:
            observation, score = assigned.get(object_item.object_id, (None, None))
            if object_item.is_baseline:
                actions.extend(
                    self._update_moved(
                        object_item,
                        observation,
                        score,
                        current_observations,
                        timestamp_sec=timestamp_sec,
                        scene_status=scene_status,
                        roi_bbox=roi_bbox,
                    )
                )
            else:
                actions.extend(
                    self._update_forgotten(
                        object_item,
                        observation,
                        timestamp_sec=timestamp_sec,
                        scene_status=scene_status,
                        roi_bbox=roi_bbox,
                        region_evidence_provider=region_evidence_provider,
                    )
                )
        self._last_timestamp_sec = timestamp_sec
        return tuple(actions)
