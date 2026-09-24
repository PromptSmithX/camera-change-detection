"""Serializable scene baseline produced by M1 calibration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from change_detection.dataset.io import read_json, sha256_file, write_json
from change_detection.domain import BaselineObject, FrameContext, SourceMetadata

from .roi import BBoxROI, roi_from_mapping
from .stability import StabilitySummary


class BaselineIntegrityError(ValueError):
    """Raised when a persisted baseline or reference image is invalid."""


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional runtime extra
        raise BaselineIntegrityError(
            "SceneBaseline persistence requires OpenCV; install the runtime extra"
        ) from exc
    return cv2


def _safe_reference_path(metadata_path: Path, relative_path: str) -> Path:
    candidate = (metadata_path.parent / relative_path).resolve()
    parent = metadata_path.parent.resolve()
    if candidate != parent and parent not in candidate.parents:
        raise BaselineIntegrityError("reference image path escapes baseline directory")
    return candidate


@dataclass(slots=True)
class SceneBaseline:
    """Reference scene plus optional detector or M3 identity records."""

    source: SourceMetadata
    roi: BBoxROI
    sample_count: int
    sample_interval_seconds: float
    start_frame: int
    sampled_frame_indices: tuple[int, ...]
    stable_frame_indices: tuple[int, ...]
    reference_frame_index: int
    reference_timestamp_sec: float
    stability: StabilitySummary
    reference_image: Any = field(default=None, repr=False, compare=False)
    schema_version: int = 1
    reference_image_path: str = "reference.png"
    reference_image_sha256: str | None = None
    baseline_objects: tuple[BaselineObject | dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise BaselineIntegrityError(
                f"Unsupported SceneBaseline schema version: {self.schema_version}"
            )
        if self.sample_count < 2:
            raise BaselineIntegrityError("SceneBaseline sample_count must be >= 2")
        if len(self.sampled_frame_indices) != self.sample_count:
            raise BaselineIntegrityError("sampled_frame_indices length must equal sample_count")
        if self.reference_frame_index not in self.sampled_frame_indices:
            raise BaselineIntegrityError("reference frame must be one of sampled frames")
        if self.roi.source_width != self.source.width or self.roi.source_height != self.source.height:
            raise BaselineIntegrityError("ROI dimensions do not match source metadata")
        if self.schema_version == 2 and not all(
            isinstance(item, BaselineObject) for item in self.baseline_objects
        ):
            raise BaselineIntegrityError("SceneBaseline schema v2 requires identity baseline objects")

    @classmethod
    def from_calibration(
        cls,
        *,
        source: SourceMetadata,
        roi: BBoxROI,
        sample_interval_seconds: float,
        frames: tuple[FrameContext, ...],
        stable_frame_indices: tuple[int, ...],
        reference_frame: FrameContext,
        stability: StabilitySummary,
    ) -> "SceneBaseline":
        if not frames:
            raise BaselineIntegrityError("Calibration produced no frames")
        copied_image = (
            reference_frame.image.copy()
            if hasattr(reference_frame.image, "copy")
            else reference_frame.image
        )
        return cls(
            source=source,
            roi=roi,
            sample_count=len(frames),
            sample_interval_seconds=sample_interval_seconds,
            start_frame=frames[0].frame_index,
            sampled_frame_indices=tuple(frame.frame_index for frame in frames),
            stable_frame_indices=stable_frame_indices,
            reference_frame_index=reference_frame.frame_index,
            reference_timestamp_sec=reference_frame.timestamp_sec,
            stability=stability,
            reference_image=copied_image,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "roi": self.roi.to_dict(),
            "calibration": {
                "sample_count": self.sample_count,
                "sample_interval_seconds": self.sample_interval_seconds,
                "start_frame": self.start_frame,
                "sampled_frame_indices": list(self.sampled_frame_indices),
                "stable_frame_indices": list(self.stable_frame_indices),
            },
            "reference": {
                "frame_index": self.reference_frame_index,
                "timestamp_sec": self.reference_timestamp_sec,
                "image_path": self.reference_image_path,
                "image_sha256": self.reference_image_sha256,
            },
            "stability": self.stability.to_dict(),
            "baseline_objects": [
                item.to_dict() if isinstance(item, BaselineObject) else dict(item)
                for item in self.baseline_objects
            ],
        }

    def set_identity_baseline(self, objects: tuple[BaselineObject, ...]) -> None:
        """Upgrade this in-memory calibration result to the M3 baseline schema."""

        self.schema_version = 2
        self.baseline_objects = objects

    def identity_objects(self) -> tuple[BaselineObject, ...]:
        if self.schema_version != 2:
            raise BaselineIntegrityError(
                "M3 requires SceneBaseline schema v2; recalibrate with perception.encoder.enabled=true"
            )
        if not all(isinstance(item, BaselineObject) for item in self.baseline_objects):
            raise BaselineIntegrityError("SceneBaseline schema v2 has invalid identity objects")
        return tuple(self.baseline_objects)  # type: ignore[return-value]

    def save(self, directory: str | Path) -> Path:
        """Write ``baseline.json`` and a lossless ``reference.png``."""

        if self.reference_image is None:
            raise BaselineIntegrityError("Cannot save a baseline without reference_image")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        reference_path = directory / self.reference_image_path
        cv2 = _cv2()
        if not cv2.imwrite(str(reference_path), self.reference_image):
            raise BaselineIntegrityError(f"Could not write reference image: {reference_path}")
        self.reference_image_sha256 = sha256_file(reference_path)
        metadata_path = directory / "baseline.json"
        write_json(metadata_path, self.to_dict())
        return metadata_path

    @classmethod
    def load(cls, metadata_path: str | Path) -> "SceneBaseline":
        metadata_path = Path(metadata_path).resolve()
        if metadata_path.is_dir():
            metadata_path = metadata_path / "baseline.json"
        payload = read_json(metadata_path)
        schema_version = int(payload.get("schema_version", -1))
        if schema_version not in {1, 2}:
            raise BaselineIntegrityError("Unsupported or missing SceneBaseline schema_version")
        source = SourceMetadata.from_dict(payload["source"])
        roi = roi_from_mapping(
            payload["roi"],
            source_width=source.width,
            source_height=source.height,
            clip=False,
        )
        calibration = payload["calibration"]
        reference = payload["reference"]
        reference_path = str(reference["image_path"])
        resolved_reference = _safe_reference_path(metadata_path, reference_path)
        if not resolved_reference.is_file():
            raise BaselineIntegrityError(f"Reference image does not exist: {resolved_reference}")
        if not reference.get("image_sha256"):
            raise BaselineIntegrityError("Reference image SHA256 is missing")
        expected_hash = str(reference["image_sha256"])
        actual_hash = sha256_file(resolved_reference)
        if actual_hash != expected_hash:
            raise BaselineIntegrityError(
                f"Reference image SHA256 mismatch: expected {expected_hash}, got {actual_hash}"
            )
        sampled_indices = tuple(int(item) for item in calibration["sampled_frame_indices"])
        stable_indices = tuple(int(item) for item in calibration["stable_frame_indices"])
        baseline = cls(
            source=source,
            roi=roi,
            sample_count=int(calibration["sample_count"]),
            sample_interval_seconds=float(calibration["sample_interval_seconds"]),
            start_frame=int(calibration["start_frame"]),
            sampled_frame_indices=sampled_indices,
            stable_frame_indices=stable_indices,
            reference_frame_index=int(reference["frame_index"]),
            reference_timestamp_sec=float(reference["timestamp_sec"]),
            stability=StabilitySummary.from_dict(payload["stability"]),
            reference_image=None,
            reference_image_path=reference_path,
            reference_image_sha256=expected_hash,
            schema_version=schema_version,
            baseline_objects=(
                tuple(BaselineObject.from_dict(dict(item)) for item in payload.get("baseline_objects", []))
                if schema_version == 2
                else tuple(dict(item) for item in payload.get("baseline_objects", []))
            ),
        )
        return baseline

    def load_reference_image(self, metadata_path: str | Path) -> Any:
        """Decode and dimension-check the persisted reference image."""

        metadata_path = Path(metadata_path).resolve()
        if metadata_path.is_dir():
            metadata_path = metadata_path / "baseline.json"
        path = _safe_reference_path(metadata_path, self.reference_image_path)
        cv2 = _cv2()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise BaselineIntegrityError(f"Could not decode reference image: {path}")
        height, width = image.shape[:2]
        if width != self.source.width or height != self.source.height:
            raise BaselineIntegrityError(
                f"Reference image dimensions {width}x{height} do not match "
                f"source {self.source.width}x{self.source.height}"
            )
        return image
