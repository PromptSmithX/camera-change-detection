from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from change_detection.association import AssociationEngine
from change_detection.config import AssociationConfig, MemoryConfig
from change_detection.domain import BaselineObject, BBox, Detection, FrameContext, Observation, Point, SourceMetadata, Track
from change_detection.infrastructure.models import DinoV2Encoder
from change_detection.memory import ObjectMemory
from change_detection.perception import EmbeddingRefresher
from change_detection.pipeline import PerceptionRunner
from change_detection.scene import BBoxROI
from change_detection.sources import BaseFrameSource


def _observation(
    tracker_id: int | None,
    embedding: tuple[float, ...],
    *,
    timestamp: float,
    bbox: BBox | None = None,
) -> Observation:
    bbox = bbox or BBox(10, 10, 30, 30)
    return Observation(
        frame_index=round(timestamp * 10),
        timestamp_sec=timestamp,
        bbox=bbox,
        centroid=Point((bbox.x1 + bbox.x2) / 2, (bbox.y1 + bbox.y2) / 2),
        detector_class="bag",
        detector_class_id=24,
        detector_confidence=0.9,
        tracker_id=tracker_id,
        embedding=embedding,
    )


def _memory(*, id_factory=lambda: "new-1") -> ObjectMemory:
    baseline = BaselineObject("baseline-1", BBox(10, 10, 30, 30), 24, "bag", 0.9, (1.0, 0.0))
    return ObjectMemory(
        [baseline],
        config=MemoryConfig(missing_grace_seconds=1.0, observation_history_size=3, embedding_history_size=2),
        id_factory=id_factory,
    )


def _update(memory: ObjectMemory, observations: list[Observation], timestamp: float):
    result = AssociationEngine(AssociationConfig()).match(
        memory.objects,
        observations,
        roi_bbox=BBox(0, 0, 100, 100),
        timestamp_sec=timestamp,
    )
    return memory.update(observations, result, timestamp_sec=timestamp)


def test_id_switch_and_short_dropout_keep_the_baseline_object_id():
    memory = _memory()
    first = _update(memory, [_observation(1, (1.0, 0.0), timestamp=0.0)], 0.0)
    assert first.assignments[0].object_id == "baseline-1"
    _update(memory, [], 0.2)
    second = _update(memory, [_observation(21, (1.0, 0.0), timestamp=0.5)], 0.5)
    assert second.assignments[0].object_id == "baseline-1"
    assert memory.objects[0].last_tracker_id == 21
    assert memory.objects[0].state.value == "PRESENT"


def test_unrelated_appearance_creates_new_object_and_missing_becomes_stale():
    ids = iter(["new-1"])
    memory = _memory(id_factory=lambda: next(ids))
    _update(memory, [], 0.0)
    _update(memory, [], 1.1)
    assert memory.objects[0].state.value == "STALE"
    update = _update(memory, [_observation(2, (0.0, 1.0), timestamp=1.2)], 1.2)
    assert update.assignments[0].object_id == "new-1"
    assert {item.object_id for item in memory.objects} == {"baseline-1", "new-1"}


def test_first_baseline_observation_after_runtime_seek_is_not_reid_expired():
    memory = _memory()
    observation = _observation(1, (1.0, 0.0), timestamp=12.0)
    associations = AssociationEngine(AssociationConfig()).match(
        memory.objects,
        [observation],
        roi_bbox=BBox(0, 0, 100, 100),
        timestamp_sec=12.0,
    )

    assert associations.matches[0].object_id == "baseline-1"


class _FakeDino:
    def to(self, _device):
        return self

    def eval(self):
        return self

    def __call__(self, batch):
        import torch

        return torch.ones((len(batch), 384), device=batch.device)


def test_dinov2_adapter_prepares_roi_clipped_crops_without_hub_download():
    encoder = DinoV2Encoder(model_instance=_FakeDino(), device="cpu", input_size=28)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    embeddings = encoder.encode(frame, [_observation(1, (1.0, 0.0), timestamp=0.0)], BBox(0, 0, 64, 48))
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 384
    assert abs(sum(value * value for value in embeddings[0]) - 1.0) < 1e-6


def test_dinov2_adapter_accepts_ultralytics_device_index():
    encoder = DinoV2Encoder(model_instance=_FakeDino(), device="0")

    assert encoder.metadata["device"] == "cuda:0"


class _Source(BaseFrameSource):
    def __init__(self):
        metadata = SourceMetadata("fake", "video", 64, 48, 10.0, 1)
        self._frames = [FrameContext("fake", 0, 0.0, np.zeros((48, 64, 3), dtype=np.uint8), metadata)]
        super().__init__(metadata)

    def read(self):
        self._ensure_open()
        return self._frames.pop(0) if self._frames else None

    def seek(self, frame_index: int) -> None:
        del frame_index


class _Detector:
    metadata = {"name": "fake"}

    def detect(self, frame, roi):
        del frame, roi
        return [Detection(BBox(10, 10, 30, 30), 0.9, 24, "bag")]


class _Tracker:
    metadata = {"name": "fake"}

    def reset(self):
        pass

    def update(self, detections, frame):
        del frame
        detection = detections[0]
        return [Track(1, detection.bbox, detection.confidence, detection.class_id, detection.class_name, 0)]


class _Encoder:
    metadata = {"name": "fake-encoder"}

    def encode(self, frame, observations, roi_bbox):
        del frame, roi_bbox
        return [(1.0, 0.0) for _ in observations]


def test_runner_keeps_m2_observation_schema_and_writes_m3_identity_debug(tmp_path: Path):
    roi = BBoxROI.from_coordinates([0, 0, 64, 48], source_width=64, source_height=48)
    runner = PerceptionRunner(
        source=_Source(),
        roi=roi,
        detector=_Detector(),
        tracker=_Tracker(),
        processing_fps=10.0,
        embedding_refresher=EmbeddingRefresher(_Encoder(), semantic_refresh_fps=2.5),
        association_engine=AssociationEngine(),
        object_memory=ObjectMemory((), id_factory=lambda: "runtime-1"),
    )
    result = runner.run(tmp_path)
    observation = json.loads(result.observations_path.read_text(encoding="utf-8").strip())
    identity = json.loads(result.identity_path.read_text(encoding="utf-8").strip())
    assert "identity" not in observation
    assert identity["assignments"][0]["object_id"] == "runtime-1"
    assert result.identity_assignment_count == 1
