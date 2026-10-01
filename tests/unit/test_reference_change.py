from __future__ import annotations

import numpy as np
import pytest

from change_detection.config import MaskedOverlapConfig, SmallComponentConfig
from change_detection.domain import BBox, Detection, SCENE_CHANGE_CLASS_ID
from change_detection.infrastructure.models import HybridDetector, ReferenceChangeDetector
from change_detection.scene import BBoxROI


ROI = BBoxROI.from_coordinates([0, 0, 160, 120], source_width=160, source_height=120)
BASELINE_BOX = BBox(30, 20, 110, 100)
NEW_BOX = BBox(50, 40, 90, 80)


class _Detector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect(self, frame: np.ndarray, roi: BBoxROI) -> list[Detection]:
        del frame, roi
        return self.detections


def _frames() -> tuple[np.ndarray, np.ndarray]:
    # Background texture makes alignment deterministic while the new object
    # supplies a strong foreground change inside the baseline box.
    reference = np.random.default_rng(7).integers(0, 256, (120, 160, 3), dtype=np.uint8)
    changed = reference.copy()
    changed[NEW_BOX.y1 : NEW_BOX.y2, NEW_BOX.x1 : NEW_BOX.x2] = 255
    return reference, changed


def _hybrid(
    reference: np.ndarray,
    detections: list[Detection],
    *,
    masked_overlap: MaskedOverlapConfig | None = None,
) -> HybridDetector:
    return HybridDetector(
        _Detector(detections),
        ReferenceChangeDetector(
            reference,
            ROI,
            baseline_boxes=[BASELINE_BOX],
            masked_overlap=masked_overlap,
        ),
    )


@pytest.mark.parametrize("class_id,class_name", [(26, "handbag"), (56, "chair")])
def test_new_object_on_baseline_becomes_candidate_on_first_changed_frame(
    class_id: int, class_name: str
) -> None:
    reference, changed = _frames()
    detection = Detection(NEW_BOX, 0.9, class_id, class_name)
    hybrid = _hybrid(reference, [detection])

    before = hybrid.detect_at(reference, ROI, timestamp_sec=0.0)
    after = hybrid.detect_at(changed, ROI, timestamp_sec=0.2)

    assert len(before) == len(after) == 1
    assert before[0].proposal_source == "yolo"
    assert before[0].event_candidate is False
    assert after[0].proposal_source == "yolo_reference_change"
    assert after[0].event_candidate is True
    assert after[0].bbox == NEW_BOX
    assert (after[0].class_id, after[0].class_name) == (class_id, class_name)
    assert after[0].reference_change_score is not None
    assert after[0].reference_change_score >= 0.5


def test_person_coverage_blocks_masked_overlap_candidate() -> None:
    reference, changed = _frames()
    bag = Detection(NEW_BOX, 0.9, 26, "handbag")
    person = Detection(BBox(45, 35, 95, 85), 0.9, 0, "person")
    hybrid = _hybrid(reference, [bag, person])

    output = hybrid.detect_at(changed, ROI, timestamp_sec=0.0)

    assert all(not item.event_candidate for item in output)


def test_masked_overlap_can_be_disabled_without_changing_raw_yolo_output() -> None:
    reference, changed = _frames()
    detection = Detection(NEW_BOX, 0.9, 26, "handbag")
    hybrid = _hybrid(reference, [detection], masked_overlap=MaskedOverlapConfig(enabled=False))

    output = hybrid.detect_at(changed, ROI, timestamp_sec=0.0)

    assert len(output) == 1
    assert output[0].proposal_source == "yolo"
    assert output[0].event_candidate is False


def test_small_change_inside_baseline_does_not_promote_detection() -> None:
    reference, changed = _frames()
    changed = reference.copy()
    changed[55:59, 65:69] = 255
    hybrid = _hybrid(reference, [Detection(NEW_BOX, 0.9, 26, "handbag")])

    output = hybrid.detect_at(changed, ROI, timestamp_sec=0.0)

    assert len(output) == 1
    assert output[0].event_candidate is False


def test_masked_overlap_does_not_duplicate_existing_contour_proposal() -> None:
    class _Changes:
        def detect(self, frame, detections, *, timestamp_sec):
            del frame, detections, timestamp_sec
            return [Detection(BBox(40, 40, 66, 56), 0.8, SCENE_CHANGE_CLASS_ID, "scene_change", proposal_source="reference_change")]

        def masked_overlap_score(self, detection):
            del detection
            return 0.9

    overlapping = BBox(38, 38, 88, 57)
    hybrid = HybridDetector(
        _Detector([Detection(overlapping, 0.9, 7, "truck"), Detection(overlapping, 0.9, 2, "car")]),
        _Changes(),
    )

    output = hybrid.detect_at(np.zeros((120, 160, 3), dtype=np.uint8), ROI, timestamp_sec=0.0)

    assert len(output) == 2
    assert sum(item.event_candidate for item in output) == 1
    assert {item.proposal_source for item in output} == {"fused", "yolo"}
    fused = next(item for item in output if item.proposal_source == "fused")
    assert fused.bbox == overlapping
    assert fused.change_bbox == BBox(40, 40, 66, 56)
    assert (fused.class_id, fused.class_name) == (7, "truck")


def test_fusion_rejects_tiny_change_inside_unrelated_large_yolo_box() -> None:
    class _Changes:
        def detect(self, frame, detections, *, timestamp_sec):
            del frame, detections, timestamp_sec
            box = BBox(50, 50, 60, 60)
            return [
                Detection(
                    box,
                    0.8,
                    SCENE_CHANGE_CLASS_ID,
                    "scene_change",
                    proposal_source="reference_change",
                    change_bbox=box,
                )
            ]

        def masked_overlap_score(self, detection):
            del detection
            return None

    large_box = BBox(0, 0, 160, 120)
    hybrid = HybridDetector(
        _Detector([Detection(large_box, 0.9, 7, "truck")]),
        _Changes(),
    )

    output = hybrid.detect_at(
        np.zeros((120, 160, 3), dtype=np.uint8),
        ROI,
        timestamp_sec=0.0,
    )

    assert {item.proposal_source for item in output} == {"reference_change", "yolo"}
    assert next(item for item in output if item.proposal_source == "reference_change").bbox == BBox(
        50, 50, 60, 60
    )


def test_fusion_thresholds_are_validated() -> None:
    reference, _changed = _frames()
    change_detector = ReferenceChangeDetector(reference, ROI)

    with pytest.raises(ValueError, match="fusion_min_overlap_ratio"):
        HybridDetector(_Detector([]), change_detector, fusion_min_overlap_ratio=1.1)
    with pytest.raises(ValueError, match="fusion_min_iou"):
        HybridDetector(_Detector([]), change_detector, fusion_min_iou=-0.1)


def test_small_stationary_change_is_exposed_only_after_extended_stability() -> None:
    reference = np.random.default_rng(19).integers(0, 32, (120, 160, 3), dtype=np.uint8)
    changed = reference.copy()
    changed[55:65, 65:76] = 255
    detector = ReferenceChangeDetector(
        reference,
        ROI,
        min_component_ratio=0.01,
        small_component=SmallComponentConfig(enabled=True),
    )

    assert detector.detect(reference, [], timestamp_sec=0.0) == []
    proposals = []
    for step in range(1, 12):
        proposals = detector.detect(changed, [], timestamp_sec=step * 0.2)
        if step < 11:
            assert proposals == []

    assert len(proposals) == 1
    assert proposals[0].proposal_source == "reference_change"
    assert proposals[0].event_candidate is True
    assert proposals[0].bbox.iou(BBox(65, 55, 76, 65)) >= 0.5


def test_small_transient_change_does_not_survive_stability_gate() -> None:
    reference = np.random.default_rng(23).integers(0, 32, (120, 160, 3), dtype=np.uint8)
    changed = reference.copy()
    changed[55:65, 65:76] = 255
    detector = ReferenceChangeDetector(
        reference,
        ROI,
        min_component_ratio=0.01,
        small_component=SmallComponentConfig(enabled=True),
    )

    assert detector.detect(changed, [], timestamp_sec=0.0) == []
    assert detector.detect(reference, [], timestamp_sec=0.8) == []
    assert detector.detect(reference, [], timestamp_sec=1.6) == []
    assert detector.detect(changed, [], timestamp_sec=2.4) == []


def test_small_elongated_or_sparse_change_is_rejected() -> None:
    reference = np.random.default_rng(29).integers(0, 32, (120, 160, 3), dtype=np.uint8)
    elongated = reference.copy()
    elongated[55:61, 60:82] = 255
    sparse = reference.copy()
    sparse[55:63, 65:69] = 255
    sparse[62:70, 76:80] = 255
    small_component = SmallComponentConfig(enabled=True)

    elongated_detector = ReferenceChangeDetector(
        reference,
        ROI,
        min_component_ratio=0.01,
        small_component=small_component,
    )
    sparse_detector = ReferenceChangeDetector(
        reference,
        ROI,
        min_component_ratio=0.01,
        small_component=small_component,
    )

    for step in range(12):
        timestamp = step * 0.2
        assert elongated_detector.detect(elongated, [], timestamp_sec=timestamp) == []
        assert sparse_detector.detect(sparse, [], timestamp_sec=timestamp) == []
