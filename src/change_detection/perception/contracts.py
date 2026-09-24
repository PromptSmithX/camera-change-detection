"""Interfaces between the domain and concrete detector/tracker adapters."""

from __future__ import annotations

from typing import Any, Protocol

from change_detection.domain import Detection, Embedding, Observation, Track


class PerceptionDependencyError(RuntimeError):
    """Raised when an optional detector/tracker runtime is unavailable."""


class Detector(Protocol):
    """Return domain detections in original source-image coordinates."""

    def detect(self, frame: Any, roi: Any) -> list[Detection]: ...


class Tracker(Protocol):
    """Link detections for short-term continuity within one source stream."""

    def update(self, detections: list[Detection], frame: Any) -> list[Track]: ...

    def reset(self) -> None: ...


class FeatureEncoder(Protocol):
    """Enrich observations with appearance embeddings without leaking a model."""

    def encode(self, frame: Any, observations: list[Observation], roi_bbox: Any) -> list[Embedding]: ...
