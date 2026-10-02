"""Build timestamped observations from detector and short-term tracker output."""

from __future__ import annotations

from typing import Any

from change_detection.domain import Detection, FrameContext, Observation, Point, Track


class ObservationBuildError(ValueError):
    """Raised when perception results cannot be converted to observations."""


class ObservationBuilder:
    """Convert model-independent tracks into ROI-valid observations.

    The builder is intentionally stateless.  Persistent identity and history
    are responsibilities of M3 ``ObjectMemory``.
    """

    def build(
        self,
        frame: FrameContext,
        detections: list[Detection],
        tracks: list[Track],
        roi: Any,
    ) -> list[Observation]:
        observations: list[Observation] = []
        seen_tracker_ids: set[int] = set()
        for track in tracks:
            if track.tracker_id in seen_tracker_ids:
                raise ObservationBuildError(
                    f"Duplicate tracker_id in one frame: {track.tracker_id}"
                )
            seen_tracker_ids.add(track.tracker_id)
            if not roi.contains_bbox(track.bbox):
                continue

            detection = None
            if track.detection_index is not None:
                if track.detection_index >= len(detections):
                    raise ObservationBuildError(
                        f"Track detection_index is outside detections: {track.detection_index}"
                    )
                detection = detections[track.detection_index]

            class_id = detection.class_id if detection is not None else track.class_id
            class_name = detection.class_name if detection is not None else track.class_name
            confidence = (
                detection.confidence if detection is not None else track.confidence
            )
            observations.append(
                Observation(
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    bbox=track.bbox,
                    centroid=Point(
                        (track.bbox.x1 + track.bbox.x2) / 2.0,
                        (track.bbox.y1 + track.bbox.y2) / 2.0,
                    ),
                    detector_class=class_name,
                    detector_class_id=class_id,
                    detector_confidence=confidence,
                    tracker_id=track.tracker_id,
                    proposal_source=(
                        detection.proposal_source if detection is not None else "tracker_prediction"
                    ),
                    reference_change_score=(
                        detection.reference_change_score if detection is not None else None
                    ),
                    event_candidate=(
                        detection.event_candidate if detection is not None else False
                    ),
                    change_bbox=(
                        detection.change_bbox if detection is not None else None
                    ),
                    evidence_sources=(
                        detection.evidence_sources
                        if detection is not None
                        else ("tracker_prediction",)
                    ),
                    semantic_score=(
                        detection.semantic_score if detection is not None else None
                    ),
                    change_score=(
                        detection.change_score if detection is not None else None
                    ),
                    alignment_score=(
                        detection.alignment_score if detection is not None else None
                    ),
                )
            )
        return observations
