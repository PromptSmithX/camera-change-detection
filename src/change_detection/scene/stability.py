"""Lightweight deterministic scene stability measurement."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Any

from .roi import ROI


class StabilityState(StrEnum):
    UNKNOWN = "UNKNOWN"
    STABLE = "STABLE"
    UNSTABLE = "UNSTABLE"


@dataclass(frozen=True, slots=True)
class StabilityObservation:
    state: StabilityState
    change_ratio: float | None
    comparison_count: int
    stable_comparison_count: int
    stable_fraction: float


@dataclass(frozen=True, slots=True)
class StabilitySummary:
    state: StabilityState
    comparison_count: int
    stable_comparison_count: int
    stable_fraction: float
    mean_change_ratio: float | None
    max_change_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "comparison_count": self.comparison_count,
            "stable_comparison_count": self.stable_comparison_count,
            "stable_fraction": self.stable_fraction,
            "mean_change_ratio": self.mean_change_ratio,
            "max_change_ratio": self.max_change_ratio,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StabilitySummary":
        return cls(
            state=StabilityState(str(value["state"])),
            comparison_count=int(value["comparison_count"]),
            stable_comparison_count=int(value["stable_comparison_count"]),
            stable_fraction=float(value["stable_fraction"]),
            mean_change_ratio=(
                float(value["mean_change_ratio"])
                if value.get("mean_change_ratio") is not None
                else None
            ),
            max_change_ratio=(
                float(value["max_change_ratio"])
                if value.get("max_change_ratio") is not None
                else None
            ),
        )


def _cv2_and_numpy():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional runtime extra
        raise RuntimeError(
            "Scene stability requires OpenCV and NumPy; install the runtime extra"
        ) from exc
    return cv2, np


class SceneStabilityMonitor:
    """Compare consecutive ROI samples using normalized grayscale MAD."""

    def __init__(
        self,
        roi: ROI,
        *,
        resize_width: int = 160,
        resize_height: int = 90,
        max_change_ratio: float = 0.02,
        required_stable_fraction: float = 0.9,
    ) -> None:
        if resize_width <= 0 or resize_height <= 0:
            raise ValueError("stability resize dimensions must be positive")
        if not isfinite(max_change_ratio) or not 0 <= max_change_ratio <= 1:
            raise ValueError("max_change_ratio must be between 0 and 1")
        if not isfinite(required_stable_fraction) or not 0 <= required_stable_fraction <= 1:
            raise ValueError("required_stable_fraction must be between 0 and 1")
        self.roi = roi
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.max_change_ratio = max_change_ratio
        self.required_stable_fraction = required_stable_fraction
        self._previous: Any = None
        self._ratios: list[float] = []
        self._stable_count = 0

    def reset(self) -> None:
        self._previous = None
        self._ratios.clear()
        self._stable_count = 0

    def _prepare(self, image: Any) -> Any:
        cv2, np = _cv2_and_numpy()
        cropped = self.roi.crop(image)
        if getattr(cropped, "ndim", 0) == 2:
            gray = cropped
        elif getattr(cropped, "ndim", 0) == 3 and cropped.shape[2] >= 3:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Scene stability requires a grayscale or BGR image")
        resized = cv2.resize(
            gray,
            (self.resize_width, self.resize_height),
            interpolation=cv2.INTER_AREA,
        )
        return np.asarray(resized, dtype=np.float32) / 255.0

    def update(self, frame_or_image: Any) -> StabilityObservation:
        image = getattr(frame_or_image, "image", frame_or_image)
        current = self._prepare(image)
        if self._previous is None:
            self._previous = current
            return StabilityObservation(
                StabilityState.UNKNOWN,
                None,
                0,
                0,
                0.0,
            )
        _, np = _cv2_and_numpy()
        change_ratio = float(np.mean(np.abs(current - self._previous)))
        self._previous = current
        self._ratios.append(change_ratio)
        if change_ratio <= self.max_change_ratio:
            self._stable_count += 1
        comparison_count = len(self._ratios)
        stable_fraction = self._stable_count / comparison_count
        state = (
            StabilityState.STABLE
            if stable_fraction >= self.required_stable_fraction
            else StabilityState.UNSTABLE
        )
        return StabilityObservation(
            state,
            change_ratio,
            comparison_count,
            self._stable_count,
            stable_fraction,
        )

    def summary(self) -> StabilitySummary:
        _, np = _cv2_and_numpy()
        comparison_count = len(self._ratios)
        stable_fraction = (
            self._stable_count / comparison_count if comparison_count else 0.0
        )
        state = StabilityState.UNKNOWN
        if comparison_count:
            state = (
                StabilityState.STABLE
                if stable_fraction >= self.required_stable_fraction
                else StabilityState.UNSTABLE
            )
        return StabilitySummary(
            state=state,
            comparison_count=comparison_count,
            stable_comparison_count=self._stable_count,
            stable_fraction=stable_fraction,
            mean_change_ratio=(float(np.mean(self._ratios)) if self._ratios else None),
            max_change_ratio=(float(max(self._ratios)) if self._ratios else None),
        )

