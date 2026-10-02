"""Deterministic multi-source proposal consolidation before tracking."""

from __future__ import annotations

from math import sqrt
from typing import Iterable

from change_detection.domain import BBox, Detection, SCENE_CHANGE_CLASS_ID


_SEMANTIC_SOURCES = frozenset({"yolo", "fused", "yolo_reference_change"})
_CHANGE_SOURCES = frozenset(
    {"reference_change", "baseline_residual", "masked_overlap", "fused", "yolo_reference_change"}
)
_SOURCE_ORDER = {
    "yolo": 0,
    "reference_change": 1,
    "baseline_residual": 2,
    "masked_overlap": 3,
}


def has_semantic_evidence(detection: Detection) -> bool:
    return bool(_SEMANTIC_SOURCES.intersection(detection.evidence_sources))


def has_change_evidence(detection: Detection) -> bool:
    return bool(_CHANGE_SOURCES.intersection(detection.evidence_sources))


class ProposalConsolidator:
    """Emit one canonical proposal per cross-source spatial cluster.

    The overlap thresholds are the existing HybridDetector fusion criteria.
    Same-source proposals are not directly joined, while transitive evidence
    from a second source can still establish that duplicate semantic boxes
    describe the same physical object.
    """

    min_overlap_ratio = 0.25
    min_iou = 0.10

    @property
    def metadata(self) -> dict[str, object]:
        return {
            "name": "proposal_consolidator",
            "enabled": True,
            "min_overlap_ratio": self.min_overlap_ratio,
            "min_iou": self.min_iou,
        }

    @staticmethod
    def _person(detection: Detection) -> bool:
        return detection.class_id == 0 or detection.class_name.casefold() == "person"

    @staticmethod
    def _overlap_over_smaller(left: BBox, right: BBox) -> float:
        x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
        x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        return intersection / max(1, min(left.area, right.area))

    def _matches(self, left: Detection, right: Detection) -> bool:
        if left.evidence_sources == right.evidence_sources:
            return False
        return (
            self._overlap_over_smaller(left.bbox, right.bbox) >= self.min_overlap_ratio
            and left.bbox.iou(right.bbox) >= self.min_iou
        )

    @staticmethod
    def _semantic_score(detection: Detection) -> float:
        return float(
            detection.semantic_score
            if detection.semantic_score is not None
            else detection.confidence
        )

    @staticmethod
    def _change_score(detection: Detection) -> float:
        if detection.change_score is not None:
            return float(detection.change_score)
        if detection.reference_change_score is not None:
            return float(detection.reference_change_score)
        return float(detection.confidence)

    @staticmethod
    def _ordered_sources(detections: Iterable[Detection]) -> tuple[str, ...]:
        sources = {source for detection in detections for source in detection.evidence_sources}
        return tuple(sorted(sources, key=lambda item: (_SOURCE_ORDER.get(item, 100), item)))

    def _canonical(self, cluster: list[Detection]) -> Detection:
        semantic = [item for item in cluster if has_semantic_evidence(item)]
        changes = [item for item in cluster if has_change_evidence(item)]
        best_semantic = max(
            semantic,
            key=lambda item: (self._semantic_score(item), item.confidence, item.bbox.area),
            default=None,
        )
        best_change = max(
            changes,
            key=lambda item: (self._change_score(item), item.confidence, item.bbox.area),
            default=None,
        )

        semantic_score = self._semantic_score(best_semantic) if best_semantic is not None else None
        change_score = self._change_score(best_change) if best_change is not None else None
        alignment_score = (
            float(best_change.alignment_score)
            if best_change is not None and best_change.alignment_score is not None
            else (1.0 if best_change is not None else None)
        )
        if semantic_score is not None and change_score is not None:
            confidence = sqrt(semantic_score * change_score * float(alignment_score))
        elif change_score is not None:
            confidence = change_score * float(alignment_score)
        elif semantic_score is not None:
            confidence = semantic_score
        else:  # Defensive fallback for externally supplied detections.
            confidence = max(item.confidence for item in cluster)

        selected = best_semantic or best_change or max(cluster, key=lambda item: item.confidence)
        change_bbox = None
        if best_change is not None:
            change_bbox = best_change.change_bbox or best_change.bbox
        event_candidate = bool(
            best_change is not None
            and any(item.event_candidate and has_change_evidence(item) for item in cluster)
        )
        return Detection(
            bbox=selected.bbox,
            confidence=max(0.0, min(1.0, confidence)),
            class_id=selected.class_id if best_semantic is not None else SCENE_CHANGE_CLASS_ID,
            class_name=selected.class_name if best_semantic is not None else "scene_change",
            mask=selected.mask,
            proposal_source="canonical",
            reference_change_score=change_score,
            event_candidate=event_candidate,
            change_bbox=change_bbox,
            evidence_sources=self._ordered_sources(cluster),
            semantic_score=semantic_score,
            change_score=change_score,
            alignment_score=alignment_score,
        )

    def consolidate(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []
        object_indices = [index for index, item in enumerate(detections) if not self._person(item)]
        parents = {index: index for index in object_indices}

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[max(left_root, right_root)] = min(left_root, right_root)

        for offset, left_index in enumerate(object_indices):
            for right_index in object_indices[offset + 1 :]:
                if self._matches(detections[left_index], detections[right_index]):
                    union(left_index, right_index)

        clusters: dict[int, list[tuple[int, Detection]]] = {}
        for index in object_indices:
            clusters.setdefault(find(index), []).append((index, detections[index]))

        output: list[tuple[int, Detection]] = [
            (min(index for index, _ in cluster), self._canonical([item for _, item in cluster]))
            for cluster in clusters.values()
        ]
        for index, detection in enumerate(detections):
            if not self._person(detection):
                continue
            output.append(
                (
                    index,
                    Detection(
                        bbox=detection.bbox,
                        confidence=detection.confidence,
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                        mask=detection.mask,
                        proposal_source="canonical",
                        event_candidate=False,
                        evidence_sources=detection.evidence_sources,
                        semantic_score=(
                            detection.semantic_score
                            if detection.semantic_score is not None
                            else detection.confidence
                        ),
                    ),
                )
            )
        return [item for _, item in sorted(output, key=lambda pair: pair[0])]
