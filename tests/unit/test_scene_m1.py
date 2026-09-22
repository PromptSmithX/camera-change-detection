from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from change_detection.config import CalibrationConfig, StabilityConfig, validate_config
from change_detection.domain import FrameContext, SourceMetadata
from change_detection.scene import (
    BBoxROI,
    BaselineIntegrityError,
    CalibrationError,
    CalibrationService,
    SceneBaseline,
    SceneStabilityMonitor,
    StabilityState,
    load_roi_file,
)
from change_detection.sources import BaseFrameSource, SourceSeekError


def _roi() -> BBoxROI:
    return BBoxROI.from_coordinates(
        [-5, -5, 40, 30],
        source_width=32,
        source_height=24,
        clip=True,
    )


def test_bbox_roi_clips_and_legacy_json_loads(tmp_path: Path):
    roi = _roi()
    assert roi.bbox.to_list() == [0, 0, 32, 24]
    assert roi.to_dict() == {
        "type": "bbox",
        "coordinates": [0, 0, 32, 24],
        "source_width": 32,
        "source_height": 24,
    }
    legacy = tmp_path / "roi.json"
    legacy.write_text(
        json.dumps({"bbox": [1, 2, 20, 22], "width": 32, "height": 24}),
        encoding="utf-8",
    )
    loaded = load_roi_file(legacy)
    assert loaded.bbox.to_list() == [1, 2, 20, 22]
    with pytest.raises(ValueError, match="non-empty"):
        BBoxROI.from_coordinates([2, 2, 2, 4], source_width=32, source_height=24)


def test_scene_stability_distinguishes_static_and_moving_frames():
    roi = BBoxROI.from_coordinates([0, 0, 32, 24], source_width=32, source_height=24)
    static_monitor = SceneStabilityMonitor(
        roi,
        resize_width=16,
        resize_height=12,
        max_change_ratio=0.02,
        required_stable_fraction=0.9,
    )
    static = np.full((24, 32, 3), 100, dtype=np.uint8)
    assert static_monitor.update(static).state == StabilityState.UNKNOWN
    for _ in range(4):
        static_monitor.update(static)
    assert static_monitor.summary().state == StabilityState.STABLE

    moving_monitor = SceneStabilityMonitor(
        roi,
        resize_width=16,
        resize_height=12,
        max_change_ratio=0.02,
        required_stable_fraction=0.9,
    )
    moving_monitor.update(np.zeros((24, 32, 3), dtype=np.uint8))
    for _ in range(4):
        moving_monitor.update(np.full((24, 32, 3), 255, dtype=np.uint8))
    assert moving_monitor.summary().state == StabilityState.UNSTABLE


class _FakeSource(BaseFrameSource):
    def __init__(self, images: list[np.ndarray], *, fps: float = 10.0):
        self._frames = [
            FrameContext(
                source_id="fake",
                frame_index=index,
                timestamp_sec=index / fps,
                image=image,
                metadata=SourceMetadata("fake", "video", 32, 24, fps, len(images)),
            )
            for index, image in enumerate(images)
        ]
        self._cursor = 0
        super().__init__(self._frames[0].metadata)

    def read(self):
        self._ensure_open()
        if self._cursor >= len(self._frames):
            return None
        frame = self._frames[self._cursor]
        self._cursor += 1
        return frame

    def seek(self, frame_index: int) -> None:
        self._ensure_open()
        if frame_index < 0 or frame_index >= len(self._frames):
            raise SourceSeekError("invalid fake seek")
        self._cursor = frame_index


def _calibration_service() -> CalibrationService:
    return CalibrationService(
        CalibrationConfig(sample_count=5, sample_interval_seconds=0.2, start_frame=0),
        StabilityConfig(
            resize_width=16,
            resize_height=12,
            max_change_ratio=0.02,
            required_stable_fraction=0.9,
        ),
    )


def test_calibration_selects_middle_stable_frame_and_persists(tmp_path: Path):
    image = np.full((24, 32, 3), 100, dtype=np.uint8)
    source = _FakeSource([image.copy() for _ in range(12)])
    result = _calibration_service().calibrate(source, _roi())
    baseline = result.baseline
    assert baseline.reference_frame_index == 4
    assert baseline.baseline_objects == ()
    metadata_path = baseline.save(tmp_path / "baseline")
    loaded = SceneBaseline.load(metadata_path)
    assert SceneBaseline.load(metadata_path.parent).reference_frame_index == 4
    assert loaded.reference_frame_index == baseline.reference_frame_index
    assert loaded.reference_image_sha256 == baseline.reference_image_sha256
    reference = loaded.load_reference_image(metadata_path)
    assert reference.shape[:2] == (24, 32)

    reference_path = metadata_path.parent / loaded.reference_image_path
    reference_path.write_bytes(b"corrupted")
    with pytest.raises(BaselineIntegrityError, match="SHA256 mismatch"):
        SceneBaseline.load(metadata_path)


def test_calibration_rejects_moving_scene_with_diagnostics():
    black = np.zeros((24, 32, 3), dtype=np.uint8)
    white = np.full((24, 32, 3), 255, dtype=np.uint8)
    source = _FakeSource([black, white] * 6)
    service = CalibrationService(
        CalibrationConfig(sample_count=5, sample_interval_seconds=0.1, start_frame=0),
        StabilityConfig(
            resize_width=16,
            resize_height=12,
            max_change_ratio=0.02,
            required_stable_fraction=0.9,
        ),
    )
    with pytest.raises(CalibrationError, match="not stable") as error:
        service.calibrate(source, _roi())
    assert error.value.diagnostics["reason"] == "unstable_scene"


def test_m1_config_accepts_json_shape_and_rejects_invalid_source():
    config = validate_config(
        {
            "source": {
                "type": "image_sequence",
                "path": "frames",
                "frame_pattern": "*.bmp",
                "fps": 25,
            },
            "roi": {"coordinates": [0, 0, 32, 24]},
            "calibration": {"sample_count": 4},
            "stability": {"max_change_ratio": 0.1},
        }
    )
    assert config.source.type == "image_sequence"
    assert config.roi.coordinates == (0, 0, 32, 24)
    assert config.calibration.sample_count == 4
    with pytest.raises(ValueError, match="source.path"):
        validate_config({"source": {"type": "video"}})
