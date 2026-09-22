from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from change_detection.domain import FrameContext
from change_detection.sources import (
    ImageSequenceSource,
    SourceClosedError,
    SourceReadError,
    SourceSeekError,
    VideoSource,
    WebcamSource,
)


def _write_image(path: Path, value: int, *, width: int = 32, height: int = 24) -> None:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_image_sequence_is_naturally_sorted_and_seekable(tmp_path: Path):
    _write_image(tmp_path / "frame10.png", 10)
    _write_image(tmp_path / "frame2.png", 2)
    _write_image(tmp_path / "frame1.png", 1)

    source = ImageSequenceSource(tmp_path, fps=10.0, frame_pattern="*.png", source_id="seq")
    try:
        assert [path.name for path in source.frame_paths] == [
            "frame1.png",
            "frame2.png",
            "frame10.png",
        ]
        frames = [source.read(), source.read(), source.read(), source.read()]
        assert [frame.frame_index for frame in frames[:3] if frame is not None] == [0, 1, 2]
        assert frames[0] is not None and frames[0].timestamp_sec == 0.0
        assert frames[2] is not None and frames[2].timestamp_sec == 0.2
        assert frames[3] is None
        source.seek(1)
        frame = source.read()
        assert frame is not None and frame.frame_index == 1
        assert frame.image[0, 0, 0] == 2
    finally:
        source.close()
    with pytest.raises(SourceClosedError):
        source.read()


def test_image_sequence_rejects_dimension_change(tmp_path: Path):
    _write_image(tmp_path / "frame1.png", 1, width=32, height=24)
    _write_image(tmp_path / "frame2.png", 2, width=40, height=24)
    source = ImageSequenceSource(tmp_path, fps=10.0, frame_pattern="*.png")
    try:
        assert source.read() is not None
        with pytest.raises(SourceReadError, match="dimensions changed"):
            source.read()
    finally:
        source.close()


def test_video_source_reads_timestamp_and_supports_seek(tmp_path: Path):
    path = tmp_path / "fixture.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    for value in (1, 2, 3):
        writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
    writer.release()

    source = VideoSource(path, source_id="fixture")
    try:
        assert source.metadata.source_id == "fixture"
        first = source.read()
        assert first is not None and first.frame_index == 0
        assert first.timestamp_sec == 0.0
        source.seek(2)
        last = source.read()
        assert last is not None and last.frame_index == 2
        assert source.read() is None
    finally:
        source.close()


class _FakeCapture:
    def __init__(self):
        self.frames = [np.zeros((24, 32, 3), dtype=np.uint8)] * 3
        self.released = False

    def isOpened(self):
        return True

    def set(self, *_args):
        return True

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 32
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 24
        if property_id == cv2.CAP_PROP_FPS:
            return 10
        return 0

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


def test_webcam_source_uses_monotonic_timestamps_and_rejects_seek(monkeypatch):
    fake = _FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _index: fake)
    source = WebcamSource(0, source_id="fake-webcam")
    try:
        frame = source.read()
        assert isinstance(frame, FrameContext)
        assert frame is not None and frame.frame_index == 0
        assert frame.timestamp_sec >= 0
        with pytest.raises(SourceSeekError, match="does not support seek"):
            source.seek(0)
        source.reset()
        assert source.read() is not None
    finally:
        source.close()
    assert fake.released
