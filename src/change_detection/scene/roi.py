"""ROI abstractions for the M1 scene pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from change_detection.dataset.io import read_json
from change_detection.domain import BBox


class ROIError(ValueError):
    """Raised when an ROI cannot be represented for a source."""


class ROI(Protocol):
    """Minimal ROI contract consumed by stability/calibration code."""

    @property
    def type(self) -> str: ...

    @property
    def bbox(self) -> BBox: ...

    @property
    def source_width(self) -> int: ...

    @property
    def source_height(self) -> int: ...

    def crop(self, image: Any) -> Any: ...

    def contains_bbox(self, bbox: BBox) -> bool: ...

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class BBoxROI:
    """Axis-aligned integer half-open ROI in original source coordinates."""

    bbox: BBox
    source_width: int
    source_height: int
    type: str = field(default="bbox", init=False)

    def __post_init__(self) -> None:
        if self.source_width <= 0 or self.source_height <= 0:
            raise ROIError("ROI source dimensions must be positive")
        if self.bbox.width <= 0 or self.bbox.height <= 0:
            raise ROIError("ROI bbox must define a non-empty area")
        if not self.bbox.is_valid_for(self.source_width, self.source_height):
            raise ROIError(
                f"ROI bbox is outside {self.source_width}x{self.source_height}: "
                f"{self.bbox.to_list()}"
            )

    @classmethod
    def from_coordinates(
        cls,
        coordinates: list[int] | tuple[int, int, int, int] | BBox,
        *,
        source_width: int,
        source_height: int,
        clip: bool = True,
    ) -> "BBoxROI":
        if isinstance(coordinates, BBox):
            value = coordinates
        else:
            value = BBox.from_value(coordinates)
        if value is None:
            raise ROIError("ROI coordinates are required")
        if clip:
            value = BBox(
                max(0, value.x1),
                max(0, value.y1),
                min(source_width, value.x2),
                min(source_height, value.y2),
            )
        return cls(value, source_width, source_height)

    @classmethod
    def from_bbox(
        cls,
        bbox: BBox,
        *,
        source_width: int,
        source_height: int,
        clip: bool = True,
    ) -> "BBoxROI":
        return cls.from_coordinates(
            bbox,
            source_width=source_width,
            source_height=source_height,
            clip=clip,
        )

    @property
    def coordinates(self) -> tuple[int, int, int, int]:
        return (self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        source_width: int | None = None,
        source_height: int | None = None,
        clip: bool = True,
    ) -> "BBoxROI":
        roi_type = str(value.get("type", "bbox"))
        if roi_type != "bbox":
            raise ROIError(f"Unsupported ROI type in M1: {roi_type}")
        coordinates = value.get("coordinates", value.get("bbox", value.get("roi_bbox")))
        width = value.get("source_width", value.get("width", source_width))
        height = value.get("source_height", value.get("height", source_height))
        if width is None or height is None:
            raise ROIError("ROI source_width and source_height are required")
        return cls.from_coordinates(
            coordinates,
            source_width=int(width),
            source_height=int(height),
            clip=clip,
        )

    def contains_bbox(self, bbox: BBox) -> bool:
        return bbox.is_inside(self.bbox)

    def crop(self, image: Any) -> Any:
        """Return a view of the ROI using NumPy-style image slicing."""

        shape = getattr(image, "shape", None)
        if shape is None or len(shape) < 2:
            raise ROIError("ROI crop requires an image-like value with shape")
        height, width = int(shape[0]), int(shape[1])
        if width < self.source_width or height < self.source_height:
            raise ROIError(
                f"Image dimensions {width}x{height} are smaller than ROI source "
                f"dimensions {self.source_width}x{self.source_height}"
            )
        return image[self.bbox.y1 : self.bbox.y2, self.bbox.x1 : self.bbox.x2]

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "coordinates": self.bbox.to_list(),
            "source_width": self.source_width,
            "source_height": self.source_height,
        }


def roi_from_mapping(
    value: Mapping[str, Any],
    *,
    source_width: int | None = None,
    source_height: int | None = None,
    clip: bool = True,
) -> BBoxROI:
    """Load both the M1 typed format and existing legacy ``bbox`` records."""

    return BBoxROI.from_mapping(
        value,
        source_width=source_width,
        source_height=source_height,
        clip=clip,
    )


def load_roi_file(
    path: str | Path,
    *,
    source_width: int | None = None,
    source_height: int | None = None,
    clip: bool = True,
) -> BBoxROI:
    """Read an ROI JSON file without changing the canonical dataset."""

    return roi_from_mapping(
        read_json(Path(path)),
        source_width=source_width,
        source_height=source_height,
        clip=clip,
    )
