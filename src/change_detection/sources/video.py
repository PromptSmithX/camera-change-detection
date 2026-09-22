"""OpenCV-backed finite video source."""

from __future__ import annotations

import math
from pathlib import Path

from change_detection.domain import FrameContext, SourceMetadata

from .base import BaseFrameSource, SourceOpenError, SourceReadError, SourceSeekError


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SourceOpenError(
            "VideoSource requires OpenCV; install the runtime extra"
        ) from exc
    return cv2


class VideoSource(BaseFrameSource):
    """Read a video file with deterministic frame-index timestamps."""

    def __init__(self, path: str | Path, *, source_id: str | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise SourceOpenError(f"Video source does not exist: {self.path}")
        cv2 = _cv2()
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            capture.release()
            raise SourceOpenError(f"Could not open video source: {self.path}")
        try:
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0:
                raise SourceOpenError(f"Video has invalid dimensions: {self.path}")
            if not math.isfinite(fps) or fps <= 0:
                raise SourceOpenError(f"Video has invalid FPS: {self.path}")
            frame_count = int(round(raw_count)) if math.isfinite(raw_count) and raw_count > 0 else None
            metadata = SourceMetadata(
                source_id=source_id or self.path.stem,
                media_type="video",
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count,
            )
        except Exception:
            capture.release()
            raise
        self._capture = capture
        self._cv2 = cv2
        self._next_frame_index = 0
        super().__init__(metadata)

    def read(self) -> FrameContext | None:
        self._ensure_open()
        ok, image = self._capture.read()
        if not ok:
            return None
        frame_index = self._next_frame_index
        self._next_frame_index += 1
        assert self.metadata.fps is not None
        return FrameContext(
            source_id=self.metadata.source_id,
            frame_index=frame_index,
            timestamp_sec=frame_index / self.metadata.fps,
            image=image,
            metadata=self.metadata,
        )

    def seek(self, frame_index: int) -> None:
        self._ensure_open()
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise SourceSeekError("frame_index must be a non-negative integer")
        if self.metadata.frame_count is not None and frame_index >= self.metadata.frame_count:
            raise SourceSeekError(
                f"frame_index {frame_index} is outside video ({self.metadata.frame_count} frames)"
            )
        if not self._capture.set(self._cv2.CAP_PROP_POS_FRAMES, int(frame_index)):
            raise SourceSeekError(f"Could not seek video to frame {frame_index}")
        self._next_frame_index = int(frame_index)

    def close(self) -> None:
        if not self.closed:
            self._capture.release()
        super().close()
