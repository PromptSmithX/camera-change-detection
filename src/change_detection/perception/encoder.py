"""M3 semantic-refresh scheduling independent of a concrete embedding model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from change_detection.domain import Embedding, Observation

from .contracts import FeatureEncoder


@dataclass(frozen=True, slots=True)
class EncoderRefreshResult:
    encoded_count: int
    reused_count: int


class EmbeddingRefresher:
    """Reuse a track embedding until the configured semantic refresh interval."""

    def __init__(self, encoder: FeatureEncoder, *, semantic_refresh_fps: float) -> None:
        if semantic_refresh_fps <= 0:
            raise ValueError("semantic_refresh_fps must be positive")
        self.encoder = encoder
        self.interval_sec = 1.0 / float(semantic_refresh_fps)
        self._cache: dict[int, tuple[float, Embedding]] = {}

    def reset(self) -> None:
        self._cache.clear()

    def enrich(self, frame: Any, observations: list[Observation], roi_bbox: Any) -> EncoderRefreshResult:
        selected: list[Observation] = []
        selected_indices: list[int] = []
        reused = 0
        for index, observation in enumerate(observations):
            cached = self._cache.get(observation.tracker_id) if observation.tracker_id is not None else None
            if cached is not None and observation.timestamp_sec - cached[0] < self.interval_sec:
                observation.embedding = cached[1]
                reused += 1
                continue
            selected.append(observation)
            selected_indices.append(index)
        if not selected:
            return EncoderRefreshResult(encoded_count=0, reused_count=reused)
        embeddings = self.encoder.encode(frame, selected, roi_bbox)
        if len(embeddings) != len(selected):
            raise ValueError("FeatureEncoder returned a different number of embeddings")
        for index, observation, embedding in zip(selected_indices, selected, embeddings, strict=True):
            observations[index].embedding = embedding
            if observation.tracker_id is not None:
                self._cache[observation.tracker_id] = (observation.timestamp_sec, embedding)
        return EncoderRefreshResult(encoded_count=len(selected), reused_count=reused)
