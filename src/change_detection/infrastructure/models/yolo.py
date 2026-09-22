"""Ultralytics YOLO adapter.

The rest of the application only sees ``change_detection.domain.Detection``;
Ultralytics result objects and tensors stop at this module boundary.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from change_detection.domain import BBox, Detection
from change_detection.perception.contracts import PerceptionDependencyError


class YoloDependencyError(PerceptionDependencyError):
    """Raised when Ultralytics or a requested YOLO model is unavailable."""


def _as_list(value: Any) -> list[Any]:
    """Convert a tensor/array/list-like value without importing torch/numpy."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, names.get(str(class_id), class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def _integer_bbox(values: Any, *, width: int, height: int, offset_x: int, offset_y: int) -> BBox | None:
    coordinates = _as_list(values)
    if len(coordinates) != 4:
        return None
    x1, y1, x2, y2 = (float(item) for item in coordinates)
    if not all(math.isfinite(item) for item in (x1, y1, x2, y2)):
        return None
    # BBox is half-open.  Floor the beginning and ceil the end so a fractional
    # model result is never accidentally shrunk to an empty integer box.
    box = BBox(
        max(0, math.floor(x1) + offset_x),
        max(0, math.floor(y1) + offset_y),
        min(width, math.ceil(x2) + offset_x),
        min(height, math.ceil(y2) + offset_y),
    )
    return box if box.area > 0 else None


class YoloDetector:
    """Run a pretrained Ultralytics YOLO model over an M1 ROI."""

    def __init__(
        self,
        model: str | Path = "yolov8s.pt",
        *,
        confidence: float = 0.35,
        iou: float = 0.7,
        device: str | None = None,
        classes: tuple[int, ...] | list[int] | None = None,
        max_detections: int = 300,
        model_instance: Any | None = None,
    ) -> None:
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not 0.0 <= float(iou) <= 1.0:
            raise ValueError("iou must be between 0 and 1")
        if int(max_detections) < 1:
            raise ValueError("max_detections must be positive")
        self.model_path = str(model)
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = device
        self.classes = tuple(classes) if classes is not None else None
        self.max_detections = int(max_detections)
        if model_instance is not None:
            self._model = model_instance
        else:
            try:
                from ultralytics import YOLO
            except Exception as exc:  # pragma: no cover - model/runtime specific
                raise YoloDependencyError(
                    "Could not import Ultralytics. Install the perception extra and "
                    f"ensure its config directory is writable: {exc}"
                ) from exc
            try:
                self._model = YOLO(self.model_path)
            except Exception as exc:  # pragma: no cover - model/runtime specific
                raise YoloDependencyError(
                    f"Could not load YOLO model {self.model_path!r}: {exc}"
                ) from exc

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "yolo",
            "framework": "ultralytics",
            "model": self.model_path,
            "confidence": self.confidence,
            "iou": self.iou,
            "device": self.device,
            "classes": list(self.classes) if self.classes is not None else None,
            "max_detections": self.max_detections,
        }

    def detect(self, frame: Any, roi: Any) -> list[Detection]:
        crop = roi.crop(frame)
        kwargs: dict[str, Any] = {
            "conf": self.confidence,
            "iou": self.iou,
            "classes": list(self.classes) if self.classes is not None else None,
            "max_det": self.max_detections,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        try:
            results = self._model.predict(source=crop, **kwargs)
        except Exception as exc:  # pragma: no cover - runtime/model specific
            raise YoloDependencyError(f"YOLO inference failed: {exc}") from exc
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        xyxy = _as_list(getattr(boxes, "xyxy", []))
        confidences = _as_list(getattr(boxes, "conf", []))
        class_ids = _as_list(getattr(boxes, "cls", []))
        names = getattr(result, "names", getattr(self._model, "names", {}))
        source_height, source_width = int(frame.shape[0]), int(frame.shape[1])
        detections: list[Detection] = []
        for index, coordinates in enumerate(xyxy):
            if index >= len(confidences) or index >= len(class_ids):
                break
            class_id = int(class_ids[index])
            bbox = _integer_bbox(
                coordinates,
                width=source_width,
                height=source_height,
                offset_x=int(roi.bbox.x1),
                offset_y=int(roi.bbox.y1),
            )
            if bbox is None or not roi.contains_bbox(bbox):
                continue
            confidence = float(confidences[index])
            if not math.isfinite(confidence):
                continue
            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=max(0.0, min(1.0, confidence)),
                    class_id=class_id,
                    class_name=_class_name(names, class_id),
                )
            )
        return detections
