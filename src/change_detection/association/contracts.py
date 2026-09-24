"""Serializable M3 association results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AssociationScore:
    appearance: float
    spatial: float
    size: float
    class_compatibility: float
    total: float

    def to_dict(self) -> dict[str, float]:
        return {
            "appearance": self.appearance,
            "spatial": self.spatial,
            "size": self.size,
            "class_compatibility": self.class_compatibility,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class AssociationCandidate:
    object_id: str
    observation_index: int
    score: AssociationScore
    gated: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "observation_index": self.observation_index,
            "score": self.score.to_dict(),
            "gated": self.gated,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class AssociationMatch:
    object_id: str
    observation_index: int
    score: AssociationScore
    match_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "observation_index": self.observation_index,
            "score": self.score.to_dict(),
            "match_kind": self.match_kind,
        }


@dataclass(frozen=True, slots=True)
class AssociationResult:
    matches: tuple[AssociationMatch, ...]
    unmatched_observation_indices: tuple[int, ...]
    unmatched_object_ids: tuple[str, ...]
    candidates: tuple[AssociationCandidate, ...]
