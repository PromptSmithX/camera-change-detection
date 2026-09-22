"""Deterministic scene calibration service for M1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from change_detection.config import CalibrationConfig, StabilityConfig
from change_detection.domain import FrameContext
from change_detection.sources import FrameSource, SourceSeekError

from .baseline import SceneBaseline
from .roi import ROI
from .stability import SceneStabilityMonitor, StabilitySummary


class CalibrationError(RuntimeError):
    """Calibration failed with structured diagnostics."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Calibration output before it is persisted to disk."""

    baseline: SceneBaseline
    stability: StabilitySummary


class CalibrationService:
    """Collect samples, gate on stability, and create a SceneBaseline."""

    def __init__(
        self,
        calibration: CalibrationConfig | None = None,
        stability: StabilityConfig | None = None,
    ) -> None:
        self.calibration = calibration or CalibrationConfig()
        self.stability = stability or StabilityConfig()

    def calibrate(self, source: FrameSource, roi: ROI) -> CalibrationResult:
        if roi.source_width != source.metadata.width or roi.source_height != source.metadata.height:
            raise CalibrationError(
                "ROI source dimensions do not match source metadata",
                diagnostics={
                    "roi": [roi.source_width, roi.source_height],
                    "source": [source.metadata.width, source.metadata.height],
                },
            )
        start_frame = self.calibration.start_frame
        try:
            if source.metadata.media_type == "webcam":
                if start_frame != 0:
                    raise CalibrationError(
                        "Webcam calibration only supports start_frame=0",
                        diagnostics={"start_frame": start_frame},
                    )
                source.reset()
            else:
                source.seek(start_frame)
        except SourceSeekError as exc:
            raise CalibrationError(
                f"Could not seek source to calibration start: {exc}",
                diagnostics={"start_frame": start_frame},
            ) from exc

        monitor = SceneStabilityMonitor(
            roi,
            resize_width=self.stability.resize_width,
            resize_height=self.stability.resize_height,
            max_change_ratio=self.stability.max_change_ratio,
            required_stable_fraction=self.stability.required_stable_fraction,
        )
        frames: list[FrameContext] = []
        stable_frame_indices: list[int] = []
        next_sample_time: float | None = None
        while len(frames) < self.calibration.sample_count:
            frame = source.read()
            if frame is None:
                break
            if next_sample_time is None:
                next_sample_time = frame.timestamp_sec
            if frame.timestamp_sec + 1e-9 < next_sample_time:
                continue
            frames.append(frame)
            observation = monitor.update(frame)
            if observation.change_ratio is None or observation.change_ratio <= self.stability.max_change_ratio:
                stable_frame_indices.append(len(frames) - 1)
            next_sample_time += self.calibration.sample_interval_seconds

        if len(frames) < self.calibration.sample_count:
            raise CalibrationError(
                "Source ended before calibration collected enough samples",
                diagnostics={
                    "reason": "insufficient_frames",
                    "required_samples": self.calibration.sample_count,
                    "collected_samples": len(frames),
                },
            )
        summary = monitor.summary()
        if summary.state.value != "STABLE":
            raise CalibrationError(
                "Scene is not stable enough for calibration",
                diagnostics={"reason": "unstable_scene", "stability": summary.to_dict()},
            )
        if not stable_frame_indices:
            raise CalibrationError(
                "Calibration did not produce a stable reference frame",
                diagnostics={"reason": "no_stable_reference", "stability": summary.to_dict()},
            )
        reference_position = stable_frame_indices[(len(stable_frame_indices) - 1) // 2]
        reference_frame = frames[reference_position]
        baseline = SceneBaseline.from_calibration(
            source=source.metadata,
            roi=roi,  # type: ignore[arg-type]
            sample_interval_seconds=self.calibration.sample_interval_seconds,
            frames=tuple(frames),
            stable_frame_indices=tuple(frames[position].frame_index for position in stable_frame_indices),
            reference_frame=reference_frame,
            stability=summary,
        )
        return CalibrationResult(baseline=baseline, stability=summary)

