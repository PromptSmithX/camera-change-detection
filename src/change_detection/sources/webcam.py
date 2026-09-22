"""OpenCV-backed webcam source."""

from __future__ import annotations

import math
import time

from change_detection.domain import FrameContext, SourceMetadata

from .base import BaseFrameSource, SourceOpenError, SourceReadError, SourceSeekError


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SourceOpenError(
            "WebcamSource requires OpenCV; install the runtime extra"
        ) from exc
    return cv2


class WebcamSource(BaseFrameSource):
    """Read a live webcam with monotonic elapsed timestamps."""

    def __init__(
        self,
        device_index: int = 0,
        *,
        source_id: str | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
    ) -> None:
        if isinstance(device_index, bool) or device_index < 0:
            raise SourceOpenError("device_index must be a non-negative integer")
        cv2 = _cv2()
        capture = cv2.VideoCapture(int(device_index))
        if not capture.isOpened():
            capture.release()
            raise SourceOpenError(f"Could not open webcam device {device_index}")
        try:
            if width is not None:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            if height is not None:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
            if fps is not None:
                if not math.isfinite(float(fps)) or float(fps) <= 0:
                    raise SourceOpenError("webcam fps must be finite and positive")
                capture.set(cv2.CAP_PROP_FPS, float(fps))
            actual_width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            actual_height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
            pending_frame = None
            if actual_width <= 0 or actual_height <= 0:
                ok, pending_frame = capture.read()
                if not ok or pending_frame is None:
                    raise SourceOpenError(f"Webcam did not return a valid frame: {device_index}")
                actual_height, actual_width = pending_frame.shape[:2]
            metadata = SourceMetadata(
                source_id=source_id or f"webcam:{device_index}",
                media_type="webcam",
                width=actual_width,
                height=actual_height,
                fps=(actual_fps if math.isfinite(actual_fps) and actual_fps > 0 else fps),
                frame_count=None,
            )
        except Exception:
            capture.release()
            raise
        self._capture = capture
        self._cv2 = cv2
        self._next_frame_index = 0
        self._time_origin = time.monotonic()
        self._pending_frame = pending_frame
        super().__init__(metadata)

    def read(self) -> FrameContext | None:
        self._ensure_open()
        if self._pending_frame is not None:
            image = self._pending_frame
            self._pending_frame = None
        else:
            ok, image = self._capture.read()
            if not ok or image is None:
                raise SourceReadError(f"Could not read webcam frame: {self.metadata.source_id}")
        height, width = image.shape[:2]
        if width != self.metadata.width or height != self.metadata.height:
            raise SourceReadError(
                f"Webcam dimensions changed: expected "
                f"{self.metadata.width}x{self.metadata.height}, got {width}x{height}"
            )
        frame_index = self._next_frame_index
        self._next_frame_index += 1
        return FrameContext(
            source_id=self.metadata.source_id,
            frame_index=frame_index,
            timestamp_sec=max(0.0, time.monotonic() - self._time_origin),
            image=image,
            metadata=self.metadata,
        )

    def seek(self, frame_index: int) -> None:
        self._ensure_open()
        raise SourceSeekError("WebcamSource does not support seek")

    def reset(self) -> None:
        self._ensure_open()
        self._next_frame_index = 0
        self._time_origin = time.monotonic()

    def close(self) -> None:
        if not self.closed:
            self._capture.release()
        super().close()

