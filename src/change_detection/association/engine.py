"""Deterministic weighted association for persistent identity."""

from __future__ import annotations

from math import sqrt
from typing import Iterable

from change_detection.config import AssociationConfig
from change_detection.domain import (
    BBox,
    MemoryObject,
    Observation,
    SCENE_CHANGE_CLASS_ID,
    embedding_from_value,
)

from .contracts import AssociationCandidate, AssociationMatch, AssociationResult, AssociationScore


class AssociationDependencyError(RuntimeError):
    """Raised when the optional Hungarian assignment dependency is missing."""


def _cosine_similarity(left: tuple[float, ...] | None, right: tuple[float, ...] | None) -> float:
    if left is None or right is None or len(left) != len(right):
        return 0.0
    return max(0.0, min(1.0, sum(a * b for a, b in zip(left, right, strict=True))))


class AssociationEngine:
    """Associate observations with eligible memory objects using Hungarian matching."""

    def __init__(self, config: AssociationConfig | None = None) -> None:
        self.config = config or AssociationConfig()

    def _score(self, memory: MemoryObject, observation: Observation, roi_bbox: BBox) -> AssociationScore:
        appearance = _cosine_similarity(memory.last_embedding, embedding_from_value(observation.embedding))
        diagonal = max(1.0, sqrt(float(roi_bbox.width**2 + roi_bbox.height**2)))
        dx = observation.centroid.x - (memory.last_bbox.x1 + memory.last_bbox.x2) / 2.0
        dy = observation.centroid.y - (memory.last_bbox.y1 + memory.last_bbox.y2) / 2.0
        spatial = max(0.0, 1.0 - sqrt(dx * dx + dy * dy) / diagonal)
        size = min(memory.last_bbox.area, observation.bbox.area) / max(memory.last_bbox.area, observation.bbox.area)
        class_compatibility = (
            1.0
            if memory.detector_class_id == observation.detector_class_id
            or memory.detector_class_id == SCENE_CHANGE_CLASS_ID
            or observation.detector_class_id == SCENE_CHANGE_CLASS_ID
            or observation.proposal_source == "fused"
            else 0.0
        )
        total = (
            self.config.appearance_weight * appearance
            + self.config.spatial_weight * spatial
            + self.config.size_weight * size
            + self.config.class_weight * class_compatibility
        )
        return AssociationScore(appearance, spatial, size, class_compatibility, total)

    def _candidate(self, memory: MemoryObject, observation: Observation, index: int, roi_bbox: BBox, timestamp_sec: float) -> AssociationCandidate:
        score = self._score(memory, observation, roi_bbox)
        # Baseline objects are seeded at calibration time (timestamp 0 in
        # memory) while a runtime can legitimately seek to the calibration
        # reference frame or later.  The first runtime observation must not
        # be rejected merely because of that absolute timestamp gap.  Once a
        # baseline object has one runtime observation, the normal re-ID window
        # protects against stale matches again.
        baseline_seed = memory.is_baseline and not memory.observation_history
        age = timestamp_sec - memory.last_seen_sec
        contour_candidate = (
            observation.event_candidate
            and observation.proposal_source in {"reference_change", "fused"}
        )
        baseline_reidentified = False
        baseline_candidate_reidentified = False
        if memory.is_baseline and memory.baseline_bbox is not None and age > self.config.max_reid_seconds:
            baseline_appearance = _cosine_similarity(
                memory.baseline_embedding,
                embedding_from_value(observation.embedding),
            )
            baseline_candidate_reidentified = (
                contour_candidate and baseline_appearance >= 0.85
            )
            baseline_reidentified = (
                memory.baseline_bbox.iou(observation.bbox) >= 0.5
                and baseline_appearance >= 0.85
            )
            if baseline_reidentified or baseline_candidate_reidentified:
                baseline_center_x = (memory.baseline_bbox.x1 + memory.baseline_bbox.x2) / 2.0
                baseline_center_y = (memory.baseline_bbox.y1 + memory.baseline_bbox.y2) / 2.0
                dx = observation.centroid.x - baseline_center_x
                dy = observation.centroid.y - baseline_center_y
                diagonal = max(1.0, sqrt(float(roi_bbox.width**2 + roi_bbox.height**2)))
                spatial = max(0.0, 1.0 - sqrt(dx * dx + dy * dy) / diagonal)
                size = min(memory.baseline_bbox.area, observation.bbox.area) / max(
                    memory.baseline_bbox.area,
                    observation.bbox.area,
                )
                total = (
                    self.config.appearance_weight * baseline_appearance
                    + self.config.spatial_weight * spatial
                    + self.config.size_weight * size
                    + self.config.class_weight * score.class_compatibility
                )
                if baseline_candidate_reidentified:
                    total = max(total, baseline_appearance)
                score = AssociationScore(
                    appearance=baseline_appearance,
                    spatial=spatial,
                    size=size,
                    class_compatibility=score.class_compatibility,
                    total=total,
                )
        if (
            not baseline_seed
            and age > self.config.max_reid_seconds
            and not baseline_reidentified
            and not baseline_candidate_reidentified
        ):
            return AssociationCandidate(memory.object_id, index, score, True, "reid_window_expired")
        if (
            self.config.require_same_class
            and score.class_compatibility == 0.0
            and not (contour_candidate and score.appearance >= 0.85)
        ):
            return AssociationCandidate(memory.object_id, index, score, True, "class_mismatch")
        if embedding_from_value(observation.embedding) is None or memory.last_embedding is None:
            return AssociationCandidate(memory.object_id, index, score, True, "missing_embedding")
        if score.appearance < self.config.min_appearance_similarity:
            return AssociationCandidate(memory.object_id, index, score, True, "appearance_below_threshold")
        if score.total < self.config.min_total_score:
            return AssociationCandidate(memory.object_id, index, score, True, "total_below_threshold")
        return AssociationCandidate(memory.object_id, index, score, False)

    def match(
        self,
        objects: Iterable[MemoryObject],
        observations: list[Observation],
        *,
        roi_bbox: BBox,
        timestamp_sec: float,
    ) -> AssociationResult:
        ordered_objects = sorted(objects, key=lambda item: item.object_id)
        candidates = [
            self._candidate(memory, observation, index, roi_bbox, timestamp_sec)
            for memory in ordered_objects
            for index, observation in enumerate(observations)
        ]
        candidate_by_pair = {(item.object_id, item.observation_index): item for item in candidates}
        matches: list[AssociationMatch] = []
        used_objects: set[str] = set()
        used_observations: set[int] = set()

        # Tracker IDs are useful short-term evidence, but only while the memory
        # object remains inside the same re-identification window.
        for observation_index, observation in enumerate(observations):
            if observation.tracker_id is None:
                continue
            same_tracker = [
                item
                for item in ordered_objects
                if item.last_tracker_id == observation.tracker_id
                and timestamp_sec - item.last_seen_sec <= self.config.max_reid_seconds
            ]
            if len(same_tracker) != 1:
                continue
            memory = same_tracker[0]
            candidate = candidate_by_pair[(memory.object_id, observation_index)]
            if candidate.gated and candidate.reason not in {"appearance_below_threshold", "total_below_threshold"}:
                continue
            if memory.object_id not in used_objects:
                matches.append(AssociationMatch(memory.object_id, observation_index, candidate.score, "tracker_continuity"))
                used_objects.add(memory.object_id)
                used_observations.add(observation_index)

        remaining_objects = [item for item in ordered_objects if item.object_id not in used_objects]
        remaining_observations = [index for index in range(len(observations)) if index not in used_observations]

        # Reject an observation when its two best viable candidates are too close.
        ambiguous_indices: set[int] = set()
        for index in remaining_observations:
            scores = sorted(
                (
                    candidate.score.total
                    for candidate in candidates
                    if candidate.observation_index == index and not candidate.gated
                    and candidate.object_id not in used_objects
                ),
                reverse=True,
            )
            if len(scores) > 1 and scores[0] - scores[1] < self.config.ambiguity_margin:
                ambiguous_indices.add(index)

        if remaining_objects and remaining_observations:
            try:
                from scipy.optimize import linear_sum_assignment
            except ImportError as exc:  # pragma: no cover - dependency boundary
                raise AssociationDependencyError(
                    "AssociationEngine requires SciPy; install the identity extra"
                ) from exc
            cost: list[list[float]] = []
            for memory in remaining_objects:
                row: list[float] = []
                for index in remaining_observations:
                    candidate = candidate_by_pair[(memory.object_id, index)]
                    valid = not candidate.gated and index not in ambiguous_indices
                    row.append(1.0 - candidate.score.total if valid else 1_000_000.0)
                cost.append(row)
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
                memory = remaining_objects[row]
                observation_index = remaining_observations[column]
                candidate = candidate_by_pair[(memory.object_id, observation_index)]
                if candidate.gated or observation_index in ambiguous_indices:
                    continue
                matches.append(AssociationMatch(memory.object_id, observation_index, candidate.score, "association"))
                used_objects.add(memory.object_id)
                used_observations.add(observation_index)

        adjusted_candidates = tuple(
            AssociationCandidate(
                item.object_id,
                item.observation_index,
                item.score,
                True,
                "ambiguous_match",
            )
            if item.observation_index in ambiguous_indices and not item.gated
            else item
            for item in candidates
        )
        return AssociationResult(
            matches=tuple(sorted(matches, key=lambda item: item.observation_index)),
            unmatched_observation_indices=tuple(index for index in range(len(observations)) if index not in used_observations),
            unmatched_object_ids=tuple(item.object_id for item in ordered_objects if item.object_id not in used_objects),
            candidates=adjusted_candidates,
        )
