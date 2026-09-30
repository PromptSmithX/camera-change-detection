from __future__ import annotations

import numpy as np
import pytest

from change_detection.config import MaskedOverlapConfig
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
