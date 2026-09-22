"""Framework-independent frame and source metadata contracts.

The ``image`` field intentionally remains opaque here.  Runtime adapters may
store a NumPy/OpenCV array in it without making the domain package import
those libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Static metadata discovered when a frame source is opened."""

    source_id: str
    media_type: str
    width: int
    height: int
    fps: float | None = None
    frame_count: int | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        if self.media_type not in {"video", "image_sequence", "webcam"}:
            raise ValueError(f"Unsupported media_type: {self.media_type}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("source dimensions must be positive")
        if self.fps is not None and (not isfinite(self.fps) or self.fps <= 0):
            raise ValueError("fps must be finite and positive when provided")
        if self.frame_count is not None and self.frame_count < 0:
            raise ValueError("frame_count must be non-negative when provided")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "media_type": self.media_type,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceMetadata":
        return cls(
            source_id=str(value["source_id"]),
            media_type=str(value["media_type"]),
            width=int(value["width"]),
            height=int(value["height"]),
            fps=float(value["fps"]) if value.get("fps") is not None else None,
            frame_count=(
                int(value["frame_count"])
                if value.get("frame_count") is not None
                else None
            ),
        )


@dataclass(slots=True)
class FrameContext:
    """One decoded frame with a source-independent time coordinate."""

    source_id: str
    frame_index: int
    timestamp_sec: float
    image: Any
    metadata: SourceMetadata

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not isfinite(self.timestamp_sec) or self.timestamp_sec < 0:
            raise ValueError("timestamp_sec must be finite and non-negative")
        if self.source_id != self.metadata.source_id:
            raise ValueError("Frame source_id must match metadata.source_id")

