"""Framework-neutral frame source interfaces and errors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from change_detection.domain import FrameContext, SourceMetadata


class SourceError(RuntimeError):
    """Base class for source lifecycle and decoding errors."""


class SourceOpenError(SourceError):
    """The source could not be opened or did not expose valid metadata."""


class SourceReadError(SourceError):
    """A source frame could not be decoded or violated source metadata."""


class SourceSeekError(SourceError):
    """A source does not support a requested seek operation."""


class SourceClosedError(SourceError):
    """An operation was requested after the source was closed."""


class FrameSource(Protocol):
    """Public protocol shared by all M1 frame sources."""

    @property
    def metadata(self) -> SourceMetadata: ...

    def read(self) -> FrameContext | None: ...

    def seek(self, frame_index: int) -> None: ...

    def reset(self) -> None: ...

    def close(self) -> None: ...

    def __enter__(self) -> "FrameSource": ...

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None: ...


class BaseFrameSource(ABC):
    """Small lifecycle implementation used by concrete adapters."""

    def __init__(self, metadata: SourceMetadata) -> None:
        self._metadata = metadata
        self._closed = False

    @property
    def metadata(self) -> SourceMetadata:
        return self._metadata

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise SourceClosedError(f"Source is closed: {self._metadata.source_id}")

    @abstractmethod
    def read(self) -> FrameContext | None:
        """Read the next frame or return ``None`` at finite-source EOF."""

    @abstractmethod
    def seek(self, frame_index: int) -> None:
        """Move the next read position to a zero-based frame index."""

    def reset(self) -> None:
        self.seek(0)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "BaseFrameSource":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

