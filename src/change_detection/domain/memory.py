"""Framework-independent persistent-identity contracts for M3."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isfinite, sqrt
from typing import Any, Iterable, Sequence

from .contracts import BBox, Observation


Embedding = tuple[float, ...]


class ObjectState(StrEnum):
    PRESENT = "PRESENT"
    TEMP_MISSING = "TEMP_MISSING"
    STALE = "STALE"


def embedding_from_value(value: Iterable[float] | None) -> Embedding | None:
    if value is None:
        return None
    result = tuple(float(item) for item in value)
    if not result or not all(isfinite(item) for item in result):
        raise ValueError("Embedding must contain finite values")
    norm = sqrt(sum(item * item for item in result))
    if norm <= 0:
        raise ValueError("Embedding norm must be positive")
    return tuple(item / norm for item in result)


@dataclass(frozen=True, slots=True)
class BaselineObject:
    """Serializable M3 identity seed stored in a SceneBaseline."""

    object_id: str
    bbox: BBox
    class_id: int
    class_name: str
    confidence: float
    embedding: Embedding

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("BaselineObject object_id must not be empty")
        if self.bbox.area <= 0:
            raise ValueError("BaselineObject bbox must be non-empty")
        if self.class_id < 0 or not self.class_name:
            raise ValueError("BaselineObject class is invalid")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("BaselineObject confidence must be between 0 and 1")
        if not self.embedding:
            raise ValueError("BaselineObject embedding is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "bbox": self.bbox.to_list(),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "embedding": list(self.embedding),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BaselineObject":
        bbox = BBox.from_value(value.get("bbox"))
        embedding = embedding_from_value(value.get("embedding"))
        if bbox is None or embedding is None:
            raise ValueError("BaselineObject requires bbox and embedding")
        return cls(
            object_id=str(value.get("object_id", "")),
            bbox=bbox,
            class_id=int(value.get("class_id", -1)),
            class_name=str(value.get("class_name", "")),
            confidence=float(value.get("confidence", 0.0)),
            embedding=embedding,
        )


@dataclass(slots=True)
class MemoryObject:
    """Mutable, framework-neutral source of truth for a persistent object."""

    object_id: str
    baseline_bbox: BBox | None
    baseline_embedding: Embedding | None
    last_bbox: BBox
    last_embedding: Embedding | None
    detector_class_id: int
    detector_class: str
    first_seen_sec: float
    last_seen_sec: float
    state: ObjectState = ObjectState.PRESENT
    last_tracker_id: int | None = None
    missing_since_sec: float | None = None
    is_baseline: bool = False
    event_candidate_bbox: BBox | None = None
    event_candidate_embedding: Embedding | None = None
    event_candidate_seen_sec: float | None = None
    observation_history: list[dict[str, Any]] = field(default_factory=list, repr=False)
    embedding_history: list[Embedding] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if not self.object_id:
            raise ValueError("MemoryObject object_id must not be empty")
        if self.last_bbox.area <= 0:
            raise ValueError("MemoryObject last_bbox must be non-empty")
        if self.detector_class_id < 0 or not self.detector_class:
            raise ValueError("MemoryObject class is invalid")
        if self.first_seen_sec < 0 or self.last_seen_sec < self.first_seen_sec:
            raise ValueError("MemoryObject timestamps are invalid")

    @classmethod
    def from_baseline(cls, baseline: BaselineObject, *, timestamp_sec: float) -> "MemoryObject":
        return cls(
            object_id=baseline.object_id,
            baseline_bbox=baseline.bbox,
            baseline_embedding=baseline.embedding,
            last_bbox=baseline.bbox,
            last_embedding=baseline.embedding,
            detector_class_id=baseline.class_id,
            detector_class=baseline.class_name,
            first_seen_sec=timestamp_sec,
            last_seen_sec=timestamp_sec,
            state=ObjectState.PRESENT,
            is_baseline=True,
            embedding_history=[baseline.embedding],
        )

    def add_observation(
        self,
        observation: Observation,
        *,
        observation_history_size: int,
        embedding_history_size: int,
    ) -> None:
        is_person = (
            observation.detector_class_id == 0
            or observation.detector_class.casefold() == "person"
        )
        protect_candidate_identity = (
            self.event_candidate_bbox is not None
            and not observation.event_candidate
            and is_person
        )
        if not protect_candidate_identity:
            self.last_bbox = observation.bbox
            self.detector_class_id = observation.detector_class_id
            self.detector_class = observation.detector_class
        self.last_seen_sec = observation.timestamp_sec
        self.last_tracker_id = observation.tracker_id
        self.state = ObjectState.PRESENT
        self.missing_since_sec = None
        self.observation_history.append(observation.to_dict())
        del self.observation_history[:-observation_history_size]
        embedding = embedding_from_value(observation.embedding)
        if embedding is not None and not protect_candidate_identity:
            self.last_embedding = embedding
            self.embedding_history.append(embedding)
            del self.embedding_history[:-embedding_history_size]
        if observation.event_candidate:
            self.event_candidate_bbox = observation.bbox
            self.event_candidate_seen_sec = observation.timestamp_sec
            if embedding is not None:
                self.event_candidate_embedding = embedding

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "state": self.state.value,
            "is_baseline": self.is_baseline,
            "last_bbox": self.last_bbox.to_list(),
            "last_seen_sec": self.last_seen_sec,
            "last_tracker_id": self.last_tracker_id,
            "class_id": self.detector_class_id,
            "class_name": self.detector_class,
            "event_candidate_bbox": (
                self.event_candidate_bbox.to_list()
                if self.event_candidate_bbox is not None
                else None
            ),
            "event_candidate_seen_sec": self.event_candidate_seen_sec,
        }
