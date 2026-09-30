"""Reference-image change proposals for fixed-camera scenes."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Iterable

from change_detection.config import MaskedOverlapConfig
from change_detection.domain import BBox, Detection, SCENE_CHANGE_CLASS_ID


@dataclass(slots=True)
class _PersistentRegion:
    bbox: BBox
    anchor_bbox: BBox
    first_seen_sec: float
    last_seen_sec: float
    score: float


@dataclass(slots=True)
class _FrameChangeEvidence:
    changed_before_baseline_mask: Any
    baseline_mask: Any
    person_mask: Any


class ReferenceChangeDetector:
    """Propose persistent regions that differ from a calibrated reference.

    The detector is deliberately conservative: people and known baseline
    objects are masked before change components are extracted, camera shifts
    above three pixels are rejected, and a component must persist before it is
    exposed as an event candidate.
    """

    def __init__(
        self,
        reference_image: Any,
        roi: Any,
        *,
        baseline_boxes: Iterable[BBox] = (),
        pixel_threshold: int = 25,
        min_component_ratio: float = 0.0025,
        max_component_ratio: float = 0.25,
        global_change_ratio: float = 0.35,
        stable_seconds: float = 1.0,
        stationary_tolerance_px: float = 6.0,
        edge_extra_seconds: float = 2.0,
        max_gap_seconds: float = 0.6,
        max_camera_shift_px: float = 3.0,
        masked_overlap: MaskedOverlapConfig | None = None,
    ) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - runtime dependency boundary
            raise RuntimeError(
                "Reference change detection requires OpenCV and NumPy"
            ) from exc

        if not 0 <= int(pixel_threshold) <= 255:
            raise ValueError("pixel_threshold must be between 0 and 255")
        if not 0.0 < float(min_component_ratio) < float(max_component_ratio) <= 1.0:
            raise ValueError("component ratios must satisfy 0 < min < max <= 1")
        if not 0.0 < float(global_change_ratio) <= 1.0:
            raise ValueError("global_change_ratio must be in (0, 1]")
        if min(
            float(stable_seconds),
            float(stationary_tolerance_px),
            float(edge_extra_seconds),
            float(max_gap_seconds),
        ) < 0:
            raise ValueError("change timing settings must be non-negative")
        if float(max_camera_shift_px) < 0:
            raise ValueError("max_camera_shift_px must be non-negative")

        self._cv2 = cv2
        self._np = np
        self.roi = roi
        reference_crop = roi.crop(reference_image)
        self._reference_gray = self._gray(reference_crop)
        self.baseline_boxes = tuple(baseline_boxes)
        self.pixel_threshold = int(pixel_threshold)
        self.min_component_ratio = float(min_component_ratio)
        self.max_component_ratio = float(max_component_ratio)
        self.global_change_ratio = float(global_change_ratio)
        self.stable_seconds = float(stable_seconds)
        self.stationary_tolerance_px = float(stationary_tolerance_px)
        self.edge_extra_seconds = float(edge_extra_seconds)
        self.max_gap_seconds = float(max_gap_seconds)
        self.max_camera_shift_px = float(max_camera_shift_px)
        self.masked_overlap = masked_overlap or MaskedOverlapConfig()
        self._regions: list[_PersistentRegion] = []
        self._last_timestamp_sec: float | None = None
        self._scene_anomaly_reason: str | None = None
        self._frame_evidence: _FrameChangeEvidence | None = None

    @staticmethod
    def _gray(image: Any) -> Any:
        import cv2
        import numpy as np

        if getattr(image, "ndim", 0) == 2:
            return np.asarray(image, dtype=np.uint8)
        if getattr(image, "ndim", 0) == 3 and image.shape[2] >= 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError("Reference change detection requires grayscale or BGR images")

    def reset(self) -> None:
        self._regions.clear()
        self._last_timestamp_sec = None
        self._scene_anomaly_reason = None
        self._frame_evidence = None

    @property
    def scene_anomaly_reason(self) -> str | None:
        return self._scene_anomaly_reason

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "reference_change",
            "pixel_threshold": self.pixel_threshold,
            "min_component_ratio": self.min_component_ratio,
            "max_component_ratio": self.max_component_ratio,
            "global_change_ratio": self.global_change_ratio,
            "stable_seconds": self.stable_seconds,
            "stationary_tolerance_px": self.stationary_tolerance_px,
            "edge_extra_seconds": self.edge_extra_seconds,
            "max_gap_seconds": self.max_gap_seconds,
            "max_camera_shift_px": self.max_camera_shift_px,
            "masked_baseline_box_count": len(self.baseline_boxes),
            "masked_overlap": {
                "enabled": self.masked_overlap.enabled,
                "min_overlap_ratio": self.masked_overlap.min_overlap_ratio,
                "min_changed_fraction": self.masked_overlap.min_changed_fraction,
                "max_person_overlap_ratio": self.masked_overlap.max_person_overlap_ratio,
            },
        }

    def _min_component_area(self) -> int:
        return max(64, round(self.roi.bbox.area * self.min_component_ratio))

    def masked_overlap_score(self, detection: Detection) -> float | None:
        """Score a YOLO box using changed pixels hidden by baseline masking.

        The event state machine handles persistence, so this per-frame evidence
        must not add another stability delay.
        """

        evidence = self._frame_evidence
        config = self.masked_overlap
        if not config.enabled or evidence is None:
            return None
        roi_box = self.roi.bbox
        box = detection.bbox
        x1 = max(roi_box.x1, box.x1) - roi_box.x1
        y1 = max(roi_box.y1, box.y1) - roi_box.y1
        x2 = min(roi_box.x2, box.x2) - roi_box.x1
        y2 = min(roi_box.y2, box.y2) - roi_box.y1
        if x2 <= x1 or y2 <= y1:
            return None

        baseline = evidence.baseline_mask[y1:y2, x1:x2] != 0
        person = evidence.person_mask[y1:y2, x1:x2] != 0
        pixel_count = baseline.size
        if (
            self._np.count_nonzero(baseline) / pixel_count < config.min_overlap_ratio
            or self._np.count_nonzero(person) / pixel_count > config.max_person_overlap_ratio
        ):
            return None
        visible_baseline = baseline & ~person
        visible_count = int(self._np.count_nonzero(visible_baseline))
        if visible_count == 0:
            return None
        changed = evidence.changed_before_baseline_mask[y1:y2, x1:x2] != 0
        changed_count = int(self._np.count_nonzero(changed & visible_baseline))
        score = changed_count / visible_count
        if changed_count < self._min_component_area() or score < config.min_changed_fraction:
            return None
        return score

    def _person_mask(self, detections: Iterable[Detection], shape: tuple[int, int]) -> Any:
        height, width = shape
        mask = self._np.zeros((height, width), dtype=self._np.uint8)
        roi_box = self.roi.bbox
        for detection in detections:
            if detection.class_id != 0 and detection.class_name.casefold() != "person":
                continue
            box = detection.bbox
            pad_x = max(1, round(box.width * 0.15))
            pad_y = max(1, round(box.height * 0.15))
            x1 = max(roi_box.x1, box.x1 - pad_x) - roi_box.x1
            y1 = max(roi_box.y1, box.y1 - pad_y) - roi_box.y1
            x2 = min(roi_box.x2, box.x2 + pad_x) - roi_box.x1
            y2 = min(roi_box.y2, box.y2 + pad_y) - roi_box.y1
            if x2 > x1 and y2 > y1:
                self._cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, thickness=-1)
        return mask

    def _baseline_mask(self, shape: tuple[int, int]) -> Any:
        height, width = shape
        mask = self._np.zeros((height, width), dtype=self._np.uint8)
        roi_box = self.roi.bbox
        for box in self.baseline_boxes:
            pad_x = max(2, round(box.width * 0.10))
            pad_y = max(2, round(box.height * 0.10))
            x1 = max(roi_box.x1, box.x1 - pad_x) - roi_box.x1
            y1 = max(roi_box.y1, box.y1 - pad_y) - roi_box.y1
            x2 = min(roi_box.x2, box.x2 + pad_x) - roi_box.x1
            y2 = min(roi_box.y2, box.y2 + pad_y) - roi_box.y1
            if x2 > x1 and y2 > y1:
                self._cv2.rectangle(mask, (x1, y1), (x2 - 1, y2 - 1), 255, thickness=-1)
        return mask

    def _aligned_gray(self, frame: Any) -> Any | None:
        current = self._gray(self.roi.crop(frame))
        if current.shape != self._reference_gray.shape:
            return None
        current_float = self._np.asarray(current, dtype=self._np.float32)
        reference_float = self._np.asarray(self._reference_gray, dtype=self._np.float32)
        (shift_x, shift_y), response = self._cv2.phaseCorrelate(reference_float, current_float)
        if (
            not self._np.isfinite(shift_x)
            or not self._np.isfinite(shift_y)
            or response < 0.02
            or hypot(float(shift_x), float(shift_y)) > self.max_camera_shift_px
        ):
            return None
        transform = self._np.asarray(
            [[1.0, 0.0, -float(shift_x)], [0.0, 1.0, -float(shift_y)]],
            dtype=self._np.float32,
        )
        return self._cv2.warpAffine(
            current,
            transform,
            (current.shape[1], current.shape[0]),
            flags=self._cv2.INTER_LINEAR,
            borderMode=self._cv2.BORDER_REPLICATE,
        )

    def _components(self, frame: Any, detections: list[Detection]) -> list[tuple[BBox, float]] | None:
        cv2, np = self._cv2, self._np
        self._scene_anomaly_reason = None
        self._frame_evidence = None
        current = self._aligned_gray(frame)
        if current is None:
            self._scene_anomaly_reason = "camera_shift_or_unreliable_alignment"
            return None
        blurred_reference = cv2.GaussianBlur(self._reference_gray, (5, 5), 0)
        blurred_current = cv2.GaussianBlur(current, (5, 5), 0)
        difference = cv2.absdiff(blurred_reference, blurred_current)
        _, changed = cv2.threshold(difference, self.pixel_threshold, 255, cv2.THRESH_BINARY)
        changed = cv2.morphologyEx(
            changed,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        changed = cv2.morphologyEx(
            changed,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )

        person_mask = self._person_mask(detections, changed.shape)
        baseline_mask = self._baseline_mask(changed.shape)
        valid = cv2.bitwise_not(cv2.bitwise_or(person_mask, baseline_mask))
        changed_after_mask = cv2.bitwise_and(changed, valid)
        valid_pixels = int(cv2.countNonZero(valid))
        if valid_pixels == 0:
            return []
        if cv2.countNonZero(changed_after_mask) / valid_pixels > self.global_change_ratio:
            self._scene_anomaly_reason = "global_scene_change"
            return None
        self._frame_evidence = _FrameChangeEvidence(changed, baseline_mask, person_mask)

        contour_result = cv2.findContours(changed_after_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[-2]
        roi_box = self.roi.bbox
        roi_area = roi_box.area
        min_area = self._min_component_area()
        max_area = round(roi_area * self.max_component_ratio)
        result: list[tuple[BBox, float]] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = int(cv2.contourArea(contour))
            if area < min_area or area > max_area or width < 8 or height < 8:
                continue
            component = BBox(
                roi_box.x1 + int(x),
                roi_box.y1 + int(y),
                roi_box.x1 + int(x + width),
                roi_box.y1 + int(y + height),
            )
            score = float(cv2.countNonZero(changed_after_mask[y : y + height, x : x + width])) / max(
                1, width * height
            )
            result.append((component, max(0.0, min(1.0, score))))
        return result

    def _is_edge_component(self, box: BBox) -> bool:
        roi = self.roi.bbox
        margin = max(3, round(min(roi.width, roi.height) * 0.01))
        return (
            box.x1 - roi.x1 <= margin
            or box.y1 - roi.y1 <= margin
            or roi.x2 - box.x2 <= margin
            or roi.y2 - box.y2 <= margin
        )

    @staticmethod
    def _matching_distance(left: BBox, right: BBox) -> float:
        left_center = ((left.x1 + left.x2) / 2.0, (left.y1 + left.y2) / 2.0)
        right_center = ((right.x1 + right.x2) / 2.0, (right.y1 + right.y2) / 2.0)
        return hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])

    def detect(
        self,
        frame: Any,
        detections: list[Detection],
        *,
        timestamp_sec: float,
    ) -> list[Detection]:
        if timestamp_sec < 0:
            raise ValueError("timestamp_sec must be non-negative")
        if self._last_timestamp_sec is not None and timestamp_sec < self._last_timestamp_sec:
            self.reset()
        components = self._components(frame, detections)
        if components is None:
            self._regions.clear()
            self._last_timestamp_sec = timestamp_sec
            return []

        self._regions = [
            region
            for region in self._regions
            if timestamp_sec - region.last_seen_sec <= self.max_gap_seconds
        ]
        used_regions: set[int] = set()
        proposals: list[Detection] = []
        roi_diagonal = max(1.0, hypot(self.roi.bbox.width, self.roi.bbox.height))
        for box, score in components:
            candidates: list[tuple[float, float, int]] = []
            for index, region in enumerate(self._regions):
                if index in used_regions:
                    continue
                distance = self._matching_distance(box, region.bbox)
                allowed_distance = max(8.0, min(24.0, 0.20 * roi_diagonal))
                iou = box.iou(region.bbox)
                if iou >= 0.10 or distance <= allowed_distance:
                    candidates.append((iou, -distance, index))
            if candidates:
                _, _, index = max(candidates)
                region = self._regions[index]
                used_regions.add(index)
                if (
                    self._matching_distance(box, region.anchor_bbox)
                    > self.stationary_tolerance_px
                ):
                    # Follow a moving component for identity matching, but
                    # restart persistence whenever it travels materially.
                    region.anchor_bbox = box
                    region.first_seen_sec = timestamp_sec
                smoothed = BBox(
                    round(0.6 * region.bbox.x1 + 0.4 * box.x1),
                    round(0.6 * region.bbox.y1 + 0.4 * box.y1),
                    round(0.6 * region.bbox.x2 + 0.4 * box.x2),
                    round(0.6 * region.bbox.y2 + 0.4 * box.y2),
                )
                region.bbox = smoothed if smoothed.area > 0 else box
                region.last_seen_sec = timestamp_sec
                region.score = max(region.score * 0.6, score)
            else:
                region = _PersistentRegion(box, box, timestamp_sec, timestamp_sec, score)
                self._regions.append(region)
                used_regions.add(len(self._regions) - 1)

            required_seconds = self.stable_seconds + (
                self.edge_extra_seconds if self._is_edge_component(region.bbox) else 0.0
            )
            if timestamp_sec - region.first_seen_sec + 1e-9 < required_seconds:
                continue
            proposals.append(
                Detection(
                    bbox=region.bbox,
                    confidence=max(0.35, min(0.99, region.score)),
                    class_id=SCENE_CHANGE_CLASS_ID,
                    class_name="scene_change",
                    proposal_source="reference_change",
                    reference_change_score=region.score,
                    event_candidate=True,
                )
            )

        self._last_timestamp_sec = timestamp_sec
        return proposals


class HybridDetector:
    """Combine YOLO semantics with persistent reference-change proposals."""

    def __init__(self, detector: Any, change_detector: ReferenceChangeDetector) -> None:
        self.detector = detector
        self.change_detector = change_detector
        self._last_timestamp_sec: float | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "yolo_reference_change",
            "detector": getattr(self.detector, "metadata", {}),
            "reference_change": self.change_detector.metadata,
        }

    @property
    def scene_anomaly_reason(self) -> str | None:
        return self.change_detector.scene_anomaly_reason

    def reset(self) -> None:
        self.change_detector.reset()
        self._last_timestamp_sec = None

    @staticmethod
    def _person(detection: Detection) -> bool:
        return detection.class_id == 0 or detection.class_name.casefold() == "person"

    @staticmethod
    def _overlap_ratio(left: BBox, right: BBox) -> float:
        x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
        x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        return intersection / max(1, min(left.area, right.area))

    def detect_at(self, frame: Any, roi: Any, *, timestamp_sec: float) -> list[Detection]:
        from dataclasses import replace

        yolo_detections = self.detector.detect(frame, roi)
        changes = self.change_detector.detect(
            frame,
            yolo_detections,
            timestamp_sec=timestamp_sec,
        )
        consumed: set[int] = set()
        output: list[Detection] = []
        for change in changes:
            matches = [
                (self._overlap_ratio(change.bbox, detection.bbox), index, detection)
                for index, detection in enumerate(yolo_detections)
                if index not in consumed
                and not self._person(detection)
                and self._overlap_ratio(change.bbox, detection.bbox) >= 0.25
            ]
            if matches:
                _ratio, index, detection = max(matches, key=lambda item: item[0])
                consumed.add(index)
                output.append(
                    Detection(
                        bbox=change.bbox,
                        confidence=max(change.confidence, detection.confidence),
                        class_id=SCENE_CHANGE_CLASS_ID,
                        class_name="scene_change",
                        proposal_source="fused",
                        reference_change_score=change.reference_change_score,
                        event_candidate=True,
                    )
                )
            else:
                output.append(change)

        for index, detection in enumerate(yolo_detections):
            if index in consumed:
                continue
            score = (
                None
                if self._person(detection)
                or any(
                    self._overlap_ratio(change.bbox, detection.bbox) >= 0.8
                    for change in changes
                )
                else self.change_detector.masked_overlap_score(detection)
            )
            if score is None:
                output.append(replace(detection, proposal_source="yolo", event_candidate=False))
            else:
                output.append(
                    replace(
                        detection,
                        proposal_source="yolo_reference_change",
                        reference_change_score=score,
                        event_candidate=True,
                    )
                )
        self._last_timestamp_sec = timestamp_sec
        return output

    def detect(self, frame: Any, roi: Any) -> list[Detection]:
        next_timestamp = 0.0 if self._last_timestamp_sec is None else self._last_timestamp_sec + 0.2
        return self.detect_at(frame, roi, timestamp_sec=next_timestamp)
