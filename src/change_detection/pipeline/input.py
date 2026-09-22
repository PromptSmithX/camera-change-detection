"""Reusable source and ROI construction shared by M1/M2 tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from change_detection.config import SystemConfig
from change_detection.scene import BBoxROI, ROIError, load_roi_file
from change_detection.sources import ImageSequenceSource, VideoSource, WebcamSource


def resolve_input(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def build_source(config: SystemConfig, root: Path) -> Any:
    source_config = config.source
    if source_config.type == "video":
        if not source_config.path:
            raise ValueError("source.path is required for video")
        return VideoSource(
            resolve_input(root, source_config.path),
            source_id=source_config.source_id,
        )
    if source_config.type == "image_sequence":
        if not source_config.path or source_config.fps is None:
            raise ValueError("source.path and source.fps are required for image_sequence")
        return ImageSequenceSource(
            resolve_input(root, source_config.path),
            fps=source_config.fps,
            frame_pattern=source_config.frame_pattern,
            source_id=source_config.source_id,
        )
    return WebcamSource(
        source_config.device_index,
        source_id=source_config.source_id,
        fps=source_config.fps,
    )


def build_roi(config: SystemConfig, root: Path, source: Any) -> BBoxROI:
    roi_config = config.roi
    if roi_config.path:
        return load_roi_file(
            resolve_input(root, roi_config.path),
            source_width=source.metadata.width,
            source_height=source.metadata.height,
            clip=True,
        )
    if roi_config.coordinates is None:
        raise ROIError("Config must provide roi.coordinates or roi.path")
    return BBoxROI.from_coordinates(
        roi_config.coordinates,
        source_width=source.metadata.width,
        source_height=source.metadata.height,
        clip=True,
    )
