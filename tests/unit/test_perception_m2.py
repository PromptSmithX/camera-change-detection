from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from change_detection.domain import BBox, Detection, FrameContext, SourceMetadata, Track
from change_detection.infrastructure.models import YoloDetector
from change_detection.infrastructure.tracking import ByteTrackAdapter
from change_detection.perception import ObservationBuilder
from change_detection.pipeline import PerceptionRunner
from change_detection.scene import BBoxROI
from change_detection.sources import BaseFrameSource


def _roi() -> BBoxROI:
    return BBoxROI.from_coordinates([10, 5, 50, 35], source_width=64, source_height=48)


def _frame(index: int = 0) -> FrameContext:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    metadata = SourceMetadata("fake", "video", 64, 48, 10.0, 2)
    return FrameContext("fake", index, index / 10.0, image, metadata)


def test_detection_track_and_observation_serialize_without_framework_types():
    change_bbox = BBox(14, 10, 20, 16)
    detection = Detection(
        BBox(12, 8, 22, 18),
        0.8,
        2,
        "car",
        proposal_source="fused",
        change_bbox=change_bbox,
    )
    track = Track(7, detection.bbox, detection.confidence, detection.class_id, detection.class_name, 0)
    observation = ObservationBuilder().build(_frame(), [detection], [track], _roi())[0]

    assert detection.to_dict()["bbox"] == [12, 8, 22, 18]
    assert detection.to_dict()["change_bbox"] == [14, 10, 20, 16]
    assert track.to_dict()["tracker_id"] == 7
    assert observation.to_dict()["centroid"] == [17.0, 13.0]
    assert observation.to_dict()["tracker_id"] == 7
    assert observation.to_dict()["change_bbox"] == [14, 10, 20, 16]


class _FakeBoxes:
    xyxy = [[2.2, 3.1, 12.1, 13.9], [35.0, 1.0, 45.0, 12.0]]
    conf = [0.91, 0.88]
    cls = [2, 0]


class _FakeResult:
    boxes = _FakeBoxes()
    names = {0: "person", 2: "car"}


class _FakeYolo:
    names = {0: "person", 2: "car"}

    def __init__(self):
        self.sources = []

    def predict(self, *, source, **kwargs):
        self.sources.append((source, kwargs))
        return [_FakeResult()]


def test_yolo_adapter_offsets_roi_coordinates_and_filters_outside_boxes():
    model = _FakeYolo()
    detector = YoloDetector(model_instance=model, confidence=0.3)
    detections = detector.detect(_frame().image, _roi())

    assert len(detections) == 1
    assert detections[0].class_name == "car"
    assert detections[0].bbox.to_list() == [12, 8, 23, 19]
    assert model.sources[0][0].shape[:2] == (30, 40)


class _NativeDetections:
    def __init__(self, *, xyxy, confidence, class_id):
        self.xyxy = xyxy
        self.confidence = confidence
        self.class_id = class_id


class _FakeByteTrack:
    def __init__(self):
        self.reset_count = 0

    def update_with_detections(self, detections):
        class Result:
            xyxy = np.asarray([[12, 8, 22, 18]], dtype=float)
            confidence = np.asarray([0.8], dtype=float)
            class_id = np.asarray([2], dtype=int)
            tracker_id = np.asarray([41], dtype=int)

        assert len(detections.xyxy) == 1
        return Result()

    def reset(self):
        self.reset_count += 1


def test_bytetrack_adapter_maps_native_track_to_domain_track_without_importing_supervision():
    native_tracker = _FakeByteTrack()
    adapter = ByteTrackAdapter(
        tracker_instance=native_tracker,
        detections_factory=_NativeDetections,
    )
    detection = Detection(BBox(12, 8, 22, 18), 0.8, 2, "car")
    tracks = adapter.update([detection], _frame().image)

    assert len(tracks) == 1
    assert tracks[0].tracker_id == 41
    assert tracks[0].detection_index == 0
    adapter.reset()
    assert native_tracker.reset_count == 1


class _FakeSource(BaseFrameSource):
    def __init__(self):
        metadata = SourceMetadata("fake", "video", 64, 48, 10.0, 2)
        self.frames = [_frame(0), _frame(1)]
        self.cursor = 0
        super().__init__(metadata)

    def read(self):
        self._ensure_open()
        if self.cursor == len(self.frames):
            return None
        frame = self.frames[self.cursor]
        self.cursor += 1
        return frame

    def seek(self, frame_index: int) -> None:
        self.cursor = frame_index


class _FakeDetector:
    metadata = {"name": "fake-detector"}

    def detect(self, frame, roi):
        return [Detection(BBox(12, 8, 22, 18), 0.8, 2, "car")]


class _FakeTracker:
    metadata = {"name": "fake-tracker"}

    def reset(self):
        pass

    def update(self, detections, frame):
        return [Track(1, detections[0].bbox, detections[0].confidence, 2, "car", 0)]


def test_perception_runner_writes_jsonl_video_and_metadata(tmp_path: Path):
    result = PerceptionRunner(
        source=_FakeSource(),
        roi=_roi(),
        detector=_FakeDetector(),
        tracker=_FakeTracker(),
        processing_fps=10.0,
    ).run(tmp_path)

    assert result.frames_processed == 2
    assert result.observation_count == 2
    assert result.video_path.is_file()
    records = [json.loads(line) for line in result.observations_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["observations"][0]["tracker_id"] == 1
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["frames_processed"] == 2
    assert metadata["detector"]["name"] == "fake-detector"
