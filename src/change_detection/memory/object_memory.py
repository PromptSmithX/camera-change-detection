"""M3 object-memory state transitions independent of CV/model frameworks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
from uuid import uuid4

from change_detection.association import AssociationResult
from change_detection.config import MemoryConfig
from change_detection.domain import BaselineObject, MemoryObject, ObjectState, Observation


@dataclass(frozen=True, slots=True)
class IdentityAssignment:
    observation_index: int
    object_id: str
    match_kind: str
    score: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_index": self.observation_index,
            "object_id": self.object_id,
            "match_kind": self.match_kind,
            "association_score": self.score,
        }


@dataclass(frozen=True, slots=True)
class MemoryUpdateResult:
    assignments: tuple[IdentityAssignment, ...]
    transitions: tuple[dict[str, str], ...]


class ObjectMemory:
    """Owns persistent IDs, bounded history, and missing-state transitions."""

    def __init__(
        self,
        baseline_objects: Iterable[BaselineObject] = (),
        *,
        config: MemoryConfig | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.config = config or MemoryConfig()
        self._baseline = tuple(baseline_objects)
        self._id_factory = id_factory or (lambda: str(uuid4()))
        self._objects: dict[str, MemoryObject] = {}
        self.reset()

    @property
    def objects(self) -> tuple[MemoryObject, ...]:
        return tuple(self._objects[key] for key in sorted(self._objects))

    def reset(self) -> None:
        self._objects = {
            item.object_id: MemoryObject.from_baseline(item, timestamp_sec=0.0)
            for item in self._baseline
        }

    def _transition(self, memory: MemoryObject, target: ObjectState, transitions: list[dict[str, str]]) -> None:
        if memory.state != target:
            transitions.append(
                {
                    "object_id": memory.object_id,
                    "state_before": memory.state.value,
                    "state_after": target.value,
                }
            )
            memory.state = target

    def _clear_duplicate_tracker_binding(self, object_id: str, tracker_id: int | None) -> None:
        if tracker_id is None:
            return
        for candidate in self._objects.values():
            if candidate.object_id != object_id and candidate.last_tracker_id == tracker_id:
                candidate.last_tracker_id = None

    def update(
        self,
        observations: list[Observation],
        associations: AssociationResult,
        *,
        timestamp_sec: float,
    ) -> MemoryUpdateResult:
        transitions: list[dict[str, str]] = []
        assignments: list[IdentityAssignment] = []
        matched_ids = {item.object_id for item in associations.matches}

        for object_id in associations.unmatched_object_ids:
            memory = self._objects[object_id]
            if memory.state == ObjectState.PRESENT:
                memory.missing_since_sec = timestamp_sec
                self._transition(memory, ObjectState.TEMP_MISSING, transitions)
            elif (
                memory.state == ObjectState.TEMP_MISSING
                and memory.missing_since_sec is not None
                and timestamp_sec - memory.missing_since_sec >= self.config.missing_grace_seconds
            ):
                self._transition(memory, ObjectState.STALE, transitions)

        for match in associations.matches:
            observation = observations[match.observation_index]
            memory = self._objects[match.object_id]
            self._clear_duplicate_tracker_binding(memory.object_id, observation.tracker_id)
            self._transition(memory, ObjectState.PRESENT, transitions)
            memory.add_observation(
                observation,
                observation_history_size=self.config.observation_history_size,
                embedding_history_size=self.config.embedding_history_size,
            )
            assignments.append(
                IdentityAssignment(
                    match.observation_index,
                    memory.object_id,
                    match.match_kind,
                    match.score.total,
                )
            )

        for observation_index in associations.unmatched_observation_indices:
            observation = observations[observation_index]
            object_id = self._id_factory()
            if object_id in self._objects:
                raise ValueError(f"Object ID factory returned a duplicate ID: {object_id}")
            memory = MemoryObject(
                object_id=object_id,
                baseline_bbox=None,
                baseline_embedding=None,
                last_bbox=observation.bbox,
                last_embedding=None,
                detector_class_id=observation.detector_class_id,
                detector_class=observation.detector_class,
                first_seen_sec=timestamp_sec,
                last_seen_sec=timestamp_sec,
                last_tracker_id=observation.tracker_id,
            )
            self._objects[object_id] = memory
            self._clear_duplicate_tracker_binding(object_id, observation.tracker_id)
            memory.add_observation(
                observation,
                observation_history_size=self.config.observation_history_size,
                embedding_history_size=self.config.embedding_history_size,
            )
            assignments.append(IdentityAssignment(observation_index, object_id, "new", None))
            transitions.append(
                {
                    "object_id": object_id,
                    "state_before": "NEW",
                    "state_after": ObjectState.PRESENT.value,
                }
            )

        return MemoryUpdateResult(
            assignments=tuple(sorted(assignments, key=lambda item: item.observation_index)),
            transitions=tuple(transitions),
        )
