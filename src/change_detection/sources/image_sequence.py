"""OpenCV-backed image sequence source."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from change_detection.domain import FrameContext, SourceMetadata

from .base import BaseFrameSource, SourceOpenError, SourceReadError, SourceSeekError


_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def natural_key(path: Path) -> list[Any]:
    """Sort ``frame2`` before ``frame10`` while keeping names deterministic."""

    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SourceOpenError(
            "ImageSequenceSource requires OpenCV; install the runtime extra"
        ) from exc
    return cv2


class ImageSequenceSource(BaseFrameSource):
    """Read a naturally sorted directory of image frames."""

    def __init__(
        self,
        path: str | Path,
        *,
        fps: float,
        frame_pattern: str = "*",
        source_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_dir():
            raise SourceOpenError(f"Image sequence directory does not exist: {self.path}")
        if not math.isfinite(float(fps)) or float(fps) <= 0:
            raise SourceOpenError("Image sequence FPS must be finite and positive")
        if not frame_pattern.strip():
            raise SourceOpenError("Image sequence frame_pattern must not be empty")
        files = [
            item
            for item in self.path.glob(frame_pattern)
            if item.is_file() and item.suffix.lower() in _IMAGE_EXTENSIONS
        ]
        self._files = sorted(files, key=natural_key)
        if not self._files:
            raise SourceOpenError(
                f"No image frames matched {frame_pattern!r} in {self.path}"
            )
        cv2 = _cv2()
        first = cv2.imread(str(self._files[0]), cv2.IMREAD_COLOR)
        if first is None:
            raise SourceOpenError(f"Could not decode image frame: {self._files[0]}")
        height, width = first.shape[:2]
        if width <= 0 or height <= 0:
            raise SourceOpenError(f"Image frame has invalid dimensions: {self._files[0]}")
        self._cv2 = cv2
        self._fps = float(fps)
        self._next_frame_index = 0
        metadata = SourceMetadata(
            source_id=source_id or self.path.name,
            media_type="image_sequence",
            width=int(width),
            height=int(height),
            fps=self._fps,
            frame_count=len(self._files),
        )
        super().__init__(metadata)

    @property
    def frame_paths(self) -> tuple[Path, ...]:
        return tuple(self._files)

    def read(self) -> FrameContext | None:
        self._ensure_open()
        if self._next_frame_index >= len(self._files):
            return None
        frame_index = self._next_frame_index
        path = self._files[frame_index]
        image = self._cv2.imread(str(path), self._cv2.IMREAD_COLOR)
        if image is None:
            raise SourceReadError(f"Could not decode image frame: {path}")
        height, width = image.shape[:2]
        if width != self.metadata.width or height != self.metadata.height:
            raise SourceReadError(
                f"Image dimensions changed at {path}: expected "
                f"{self.metadata.width}x{self.metadata.height}, got {width}x{height}"
            )
        self._next_frame_index += 1
        return FrameContext(
            source_id=self.metadata.source_id,
            frame_index=frame_index,
            timestamp_sec=frame_index / self._fps,
            image=image,
            metadata=self.metadata,
        )

    def seek(self, frame_index: int) -> None:
        self._ensure_open()
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise SourceSeekError("frame_index must be a non-negative integer")
        if frame_index >= len(self._files):
            raise SourceSeekError(
                f"frame_index {frame_index} is outside image sequence ({len(self._files)} frames)"
            )
        self._next_frame_index = int(frame_index)

    def close(self) -> None:
        super().close()
