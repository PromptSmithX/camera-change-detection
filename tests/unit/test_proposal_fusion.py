from __future__ import annotations

from math import isclose, sqrt

from change_detection.domain import BBox, Detection, FrameContext, SCENE_CHANGE_CLASS_ID, SourceMetadata, Track
from change_detection.perception import ObservationBuilder, ProposalConsolidator
from change_detection.scene import BBoxROI


def _change(
    bbox: BBox,
    source: str,
    *,
    score: float,
    alignment: float = 0.8,
) -> Detection:
    return Detection(
        bbox,
        score,
        SCENE_CHANGE_CLASS_ID,
        "scene_change",
        proposal_source=source,
        reference_change_score=score,
        change_bbox=bbox,
        evidence_sources=(source,),
        change_score=score,
        alignment_score=alignment,
    )


def _yolo(bbox: BBox, *, confidence: float = 0.9, person: bool = False) -> Detection:
    return Detection(
        bbox,
        confidence,
        0 if person else 26,
        "person" if person else "handbag",
        evidence_sources=("yolo",),
        semantic_score=confidence,
        event_candidate=False,
    )


def test_yolo_and_residual_emit_one_canonical_detection() -> None:
    bbox = BBox(10, 10, 30, 30)
    output = ProposalConsolidator().consolidate(
        [_yolo(bbox), _change(BBox(11, 11, 29, 29), "baseline_residual", score=0.64)]
    )

    assert len(output) == 1
    assert output[0].proposal_source == "canonical"
    assert output[0].bbox == bbox
    assert output[0].class_name == "handbag"
    assert output[0].change_bbox == BBox(11, 11, 29, 29)
    assert output[0].evidence_sources == ("yolo", "baseline_residual")
    assert isclose(output[0].confidence, sqrt(0.9 * 0.64 * 0.8))


def test_reference_change_and_residual_emit_one_change_only_proposal() -> None:
    output = ProposalConsolidator().consolidate(
        [
            _change(BBox(10, 10, 30, 30), "reference_change", score=0.7),
            _change(BBox(11, 11, 31, 31), "baseline_residual", score=0.8),
        ]
    )

    assert len(output) == 1
    assert output[0].class_id == SCENE_CHANGE_CLASS_ID
    assert output[0].evidence_sources == ("reference_change", "baseline_residual")
    assert output[0].change_score == 0.8
    assert isclose(output[0].confidence, 0.8 * 0.8)


def test_nearby_objects_without_required_overlap_stay_separate() -> None:
    output = ProposalConsolidator().consolidate(
        [
            _yolo(BBox(10, 10, 30, 30)),
            _change(BBox(28, 10, 48, 30), "reference_change", score=0.8),
        ]
    )

    assert len(output) == 2


def test_person_is_never_merged_or_exposed_as_event_candidate() -> None:
    bbox = BBox(10, 10, 30, 40)
    output = ProposalConsolidator().consolidate(
        [_yolo(bbox, person=True), _change(bbox, "reference_change", score=0.8)]
    )

    assert len(output) == 2
    person = next(item for item in output if item.class_name == "person")
    assert person.event_candidate is False
    assert person.evidence_sources == ("yolo",)


def test_yolo_dropout_preserves_change_candidate() -> None:
    output = ProposalConsolidator().consolidate(
        [_change(BBox(10, 10, 30, 30), "baseline_residual", score=0.75)]
    )

    assert len(output) == 1
    assert output[0].event_candidate is True
    assert output[0].semantic_score is None


def test_lineage_and_scores_reach_observation_through_track_mapping() -> None:
    bbox = BBox(12, 8, 22, 18)
    detection = ProposalConsolidator().consolidate(
        [_yolo(bbox), _change(bbox, "reference_change", score=0.64)]
    )[0]
    track = Track(7, bbox, detection.confidence, detection.class_id, detection.class_name, 0)
    frame = FrameContext(
        "fake",
        1,
        0.1,
        object(),
        SourceMetadata("fake", "video", 64, 48, 10.0, 2),
    )
    roi = BBoxROI.from_coordinates([0, 0, 64, 48], source_width=64, source_height=48)

    observation = ObservationBuilder().build(frame, [detection], [track], roi)[0]

    assert observation.proposal_source == "canonical"
    assert observation.evidence_sources == ("yolo", "reference_change")
    assert observation.semantic_score == 0.9
    assert observation.change_score == 0.64
    assert observation.alignment_score == 0.8
    assert observation.to_dict()["evidence_sources"] == ["yolo", "reference_change"]
