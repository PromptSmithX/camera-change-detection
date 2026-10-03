from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from change_detection.association import AssociationEngine
from change_detection.config import ForgottenEventConfig, MemoryConfig, MovedEventConfig
from change_detection.domain import BBox, Detection, FrameContext, SourceMetadata, Track
from change_detection.events import EventEngine, EventStore
from change_detection.memory import ObjectMemory
from change_detection.perception import EmbeddingRefresher
from change_detection.pipeline import PerceptionRunner
from change_detection.scene import BBoxROI
from change_detection.sources import BaseFrameSource


class _Source(BaseFrameSource):
    def __init__(self) -> None:
        metadata = SourceMetadata("event-output", "video", 64, 48, 2.0, 4)
        self._frames = [
            FrameContext(
                "event-output",
                index,
                index / 2.0,
                np.zeros((48, 64, 3), dtype=np.uint8),
                metadata,
            )
            for index in range(4)
        ]
        super().__init__(metadata)

    def read(self):
        self._ensure_open()
        return self._frames.pop(0) if self._frames else None

    def seek(self, frame_index: int) -> None:
        del frame_index


class _Detector:
    metadata = {"name": "event-detector"}

    def detect(self, frame, roi):
        del frame, roi
        return [Detection(BBox(20, 15, 32, 27), 0.95, 24, "bag")]


class _Tracker:
    metadata = {"name": "event-tracker"}

    def reset(self):
        return None

    def update(self, detections, frame):
        del frame
        detection = detections[0]
        return [Track(1, detection.bbox, detection.confidence, 24, "bag", 0)]


class _Encoder:
    metadata = {"name": "event-encoder"}

    def encode(self, frame, observations, roi_bbox):
        del frame, roi_bbox
        return [(1.0, 0.0) for _ in observations]


def test_runner_writes_public_event_lifecycle_and_snapshots(tmp_path: Path):
    roi = BBoxROI.from_coordinates([0, 0, 64, 48], source_width=64, source_height=48)
    ids = iter(("new-1",))
    store = EventStore()
    runner = PerceptionRunner(
        source=_Source(),
        roi=roi,
        detector=_Detector(),
        tracker=_Tracker(),
        processing_fps=2.0,
        embedding_refresher=EmbeddingRefresher(_Encoder(), semantic_refresh_fps=2.0),
        association_engine=AssociationEngine(),
        object_memory=ObjectMemory(
            config=MemoryConfig(missing_grace_seconds=0.5),
            id_factory=lambda: next(ids),
        ),
        event_engine=EventEngine(
            forgotten=ForgottenEventConfig(
                candidate_seconds=0.5,
                confirm_seconds=1.0,
                disappear_grace_seconds=0.5,
                min_confidence=0.5,
                max_centroid_jitter_ratio=0.05,
                max_area_change_ratio=0.5,
            ),
            moved=MovedEventConfig(),
            event_id_factory=lambda: "evt-output-1",
        ),
        event_store=store,
        reference_image=np.zeros((48, 64, 3), dtype=np.uint8),
    )

    result = runner.run(tmp_path)

    events = [json.loads(line) for line in result.events_path.read_text().splitlines()]
    lifecycle = [json.loads(line) for line in result.event_lifecycle_path.read_text().splitlines()]
    assert result.event_count == 1
    assert len(events) == 1
    assert events[0]["type"] == "FORGOTTEN_OBJECT"
    assert events[0]["bbox"] == [20, 15, 32, 27]
    assert [item["action"] for item in lifecycle] == ["created", "confirmed"]
    assert (result.snapshots_dir / "evt-output-1_before.png").is_file()
    assert (result.snapshots_dir / "evt-output-1_after.png").is_file()
    debug = json.loads(
        (result.snapshots_dir / "evt-output-1_debug.json").read_text(encoding="utf-8")
    )
    assert debug["object_id"] == "new-1"
    assert debug["bbox"] == [20, 15, 32, 27]
    metadata = json.loads(result.metadata_path.read_text())
    assert metadata["events"]["confirmed_count"] == 1
