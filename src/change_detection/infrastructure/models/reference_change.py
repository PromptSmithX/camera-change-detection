"""Reference-image change proposals for fixed-camera scenes."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Iterable

from change_detection.config import (
    AlignmentConfig,
    BaselineResidualConfig,
    MaskedOverlapConfig,
    SmallComponentConfig,
)
from change_detection.domain import BBox, Detection, SCENE_CHANGE_CLASS_ID


@dataclass(slots=True)
class _PersistentRegion:
    bbox: BBox
    anchor_bbox: BBox
    first_seen_sec: float
    last_seen_sec: float
    score: float
    small_component: bool = False
    proposal_source: str = "reference_change"


@dataclass(frozen=True, slots=True)
class _AlignmentResult:
    shift_x: float
    shift_y: float
    response: float
    valid: bool
    reason: str | None = None

    @property
    def magnitude(self) -> float:
        return hypot(self.shift_x, self.shift_y)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shift_x": self.shift_x,
            "shift_y": self.shift_y,
            "magnitude": self.magnitude,
            "response": self.response,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(slots=True)
class _FrameChangeEvidence:
    changed_before_baseline_mask: Any
    baseline_mask: Any
    person_mask: Any
    alignment: _AlignmentResult


class ReferenceChangeDetector:
    """Propose persistent regions that differ from a calibrated reference.

    The detector is deliberately conservative: people and known baseline
    objects are masked before free-space change components are extracted,
    alignable camera translation is compensated, and a component must persist
    before it is exposed as an event candidate.
    """

    def __init__(
        self,
        reference_image: Any,
        roi: Any,
        *,
        baseline_boxes: Iterable[BBox] = (),
        pixel_threshold: int = 25,
        min_component_ratio: float = 0.0025,
        min_component_area_px: int = 64,
        max_component_ratio: float = 0.25,
        global_change_ratio: float = 0.35,
        stable_seconds: float = 1.0,
        stationary_tolerance_px: float = 6.0,
        edge_extra_seconds: float = 2.0,
        max_gap_seconds: float = 0.6,
        small_component: SmallComponentConfig | None = None,
        masked_overlap: MaskedOverlapConfig | None = None,
        alignment: AlignmentConfig | None = None,
        baseline_residual: BaselineResidualConfig | None = None,
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
        if (
            isinstance(min_component_area_px, bool)
            or int(min_component_area_px) != min_component_area_px
            or int(min_component_area_px) < 1
        ):
            raise ValueError("min_component_area_px must be a positive integer")
        if not 0.0 < float(global_change_ratio) <= 1.0:
            raise ValueError("global_change_ratio must be in (0, 1]")
        if min(
            float(stable_seconds),
            float(stationary_tolerance_px),
            float(edge_extra_seconds),
            float(max_gap_seconds),
        ) < 0:
            raise ValueError("change timing settings must be non-negative")
        self._cv2 = cv2
        self._np = np
        self.roi = roi
        reference_crop = roi.crop(reference_image)
        self._reference_gray = self._gray(reference_crop)
        self.baseline_boxes = tuple(baseline_boxes)
        self.pixel_threshold = int(pixel_threshold)
        self.min_component_ratio = float(min_component_ratio)
        self.min_component_area_px = int(min_component_area_px)
        self.max_component_ratio = float(max_component_ratio)
        self.global_change_ratio = float(global_change_ratio)
        self.stable_seconds = float(stable_seconds)
        self.stationary_tolerance_px = float(stationary_tolerance_px)
        self.edge_extra_seconds = float(edge_extra_seconds)
        self.max_gap_seconds = float(max_gap_seconds)
        self.small_component = small_component or SmallComponentConfig()
        self.masked_overlap = masked_overlap or MaskedOverlapConfig()
        self.alignment = alignment or AlignmentConfig()
        self.baseline_residual = baseline_residual or BaselineResidualConfig()
        self._regions: list[_PersistentRegion] = []
        self._last_timestamp_sec: float | None = None
        self._scene_anomaly_reason: str | None = None
        self._frame_evidence: _FrameChangeEvidence | None = None
        self._alignment_result: _AlignmentResult | None = None

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
        self._alignment_result = None

    @property
    def scene_anomaly_reason(self) -> str | None:
        return self._scene_anomaly_reason

    @property
    def scene_debug(self) -> dict[str, Any] | None:
        return self._alignment_result.to_dict() if self._alignment_result is not None else None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "reference_change",
            "pixel_threshold": self.pixel_threshold,
            "min_component_ratio": self.min_component_ratio,
            "min_component_area_px": self.min_component_area_px,
            "max_component_ratio": self.max_component_ratio,
            "global_change_ratio": self.global_change_ratio,
            "stable_seconds": self.stable_seconds,
            "stationary_tolerance_px": self.stationary_tolerance_px,
            "edge_extra_seconds": self.edge_extra_seconds,
            "max_gap_seconds": self.max_gap_seconds,
            "alignment": {
                "min_response": self.alignment.min_response,
                "max_translation_ratio": self.alignment.max_translation_ratio,
                "max_unfused_translation_ratio": (
                    self.alignment.max_unfused_translation_ratio
                ),
            },
            "masked_baseline_box_count": len(self.baseline_boxes),
            "small_component": {
                "enabled": self.small_component.enabled,
                "min_component_ratio": self.small_component.min_component_ratio,
                "min_component_area_px": self.small_component.min_component_area_px,
                "stable_seconds": self.small_component.stable_seconds,
                "min_fill_ratio": self.small_component.min_fill_ratio,
                "min_aspect_ratio": self.small_component.min_aspect_ratio,
            },
            "masked_overlap": {
                "enabled": self.masked_overlap.enabled,
                "min_overlap_ratio": self.masked_overlap.min_overlap_ratio,
                "min_changed_fraction": self.masked_overlap.min_changed_fraction,
                "max_person_overlap_ratio": self.masked_overlap.max_person_overlap_ratio,
            },
            "baseline_residual": {
                "enabled": self.baseline_residual.enabled,
                "min_component_ratio": self.baseline_residual.min_component_ratio,
                "min_component_area_px": self.baseline_residual.min_component_area_px,
                "stable_seconds": self.baseline_residual.stable_seconds,
                "min_fill_ratio": self.baseline_residual.min_fill_ratio,
                "max_person_overlap_ratio": self.baseline_residual.max_person_overlap_ratio,
                "max_baseline_coverage_ratio": (
                    self.baseline_residual.max_baseline_coverage_ratio
                ),
            },
        }

    def _min_component_area(self) -> int:
        return max(
            self.min_component_area_px,
            round(self.roi.bbox.area * self.min_component_ratio),
        )

    def _small_min_component_area(self) -> int:
        return max(
            self.small_component.min_component_area_px,
            round(self.roi.bbox.area * self.small_component.min_component_ratio),
        )

    def _baseline_residual_min_area(self) -> int:
        return max(
            self.baseline_residual.min_component_area_px,
            round(self.roi.bbox.area * self.baseline_residual.min_component_ratio),
        )

    def _translate_bbox(self, box: BBox, dx: float, dy: float) -> BBox | None:
        roi = self.roi.bbox
        translated = BBox(
            max(roi.x1, round(box.x1 + dx)),
            max(roi.y1, round(box.y1 + dy)),
            min(roi.x2, round(box.x2 + dx)),
            min(roi.y2, round(box.y2 + dy)),
        )
        return translated if translated.area > 0 else None

    def _current_to_reference_bbox(
        self, box: BBox, alignment: _AlignmentResult
    ) -> BBox | None:
        return self._translate_bbox(box, -alignment.shift_x, -alignment.shift_y)

    def _reference_to_current_bbox(
        self, box: BBox, alignment: _AlignmentResult
    ) -> BBox | None:
        return self._translate_bbox(box, alignment.shift_x, alignment.shift_y)

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
        box = self._current_to_reference_bbox(detection.bbox, evidence.alignment)
        if box is None:
            return None
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

    def _person_mask(
        self,
        detections: Iterable[Detection],
        shape: tuple[int, int],
        alignment: _AlignmentResult,
    ) -> Any:
        height, width = shape
        mask = self._np.zeros((height, width), dtype=self._np.uint8)
        roi_box = self.roi.bbox
        for detection in detections:
            if detection.class_id != 0 and detection.class_name.casefold() != "person":
                continue
            box = self._current_to_reference_bbox(detection.bbox, alignment)
            if box is None:
                continue
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

    def _baseline_coverage(self, box: BBox) -> tuple[float, int]:
        coverage = 0.0
        substantial_overlaps = 0
        for baseline in self.baseline_boxes:
            x1 = max(box.x1, baseline.x1)
            y1 = max(box.y1, baseline.y1)
            x2 = min(box.x2, baseline.x2)
            y2 = min(box.y2, baseline.y2)
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            coverage = max(coverage, intersection / max(1, baseline.area))
            if intersection / max(1, box.area) >= 0.10:
                substantial_overlaps += 1
        return coverage, substantial_overlaps

    def _aligned_gray(self, frame: Any) -> tuple[Any, _AlignmentResult] | None:
        current = self._gray(self.roi.crop(frame))
        if current.shape != self._reference_gray.shape:
            self._alignment_result = _AlignmentResult(
                0.0, 0.0, 0.0, False, "shape_mismatch"
            )
            return None
        current_float = self._np.asarray(current, dtype=self._np.float32)
        reference_float = self._np.asarray(self._reference_gray, dtype=self._np.float32)
        (shift_x, shift_y), response = self._cv2.phaseCorrelate(reference_float, current_float)
        if not self._np.isfinite(shift_x) or not self._np.isfinite(shift_y):
            self._alignment_result = _AlignmentResult(
                0.0, 0.0, float(response), False, "non_finite_translation"
            )
            return None
        shift_x = float(shift_x)
        shift_y = float(shift_y)
        response = float(response)
        maximum_shift = max(
            1.0,
            hypot(float(self.roi.bbox.width), float(self.roi.bbox.height))
            * self.alignment.max_translation_ratio,
        )
        reason = None
        if response < self.alignment.min_response:
            reason = "low_alignment_response"
        elif hypot(shift_x, shift_y) > maximum_shift:
            reason = "translation_out_of_range"
        alignment = _AlignmentResult(
            shift_x,
            shift_y,
            response,
            reason is None,
            reason,
        )
        self._alignment_result = alignment
        if not alignment.valid:
            return None
        transform = self._np.asarray(
            [[1.0, 0.0, -shift_x], [0.0, 1.0, -shift_y]],
            dtype=self._np.float32,
        )
        return (
            self._cv2.warpAffine(
                current,
                transform,
                (current.shape[1], current.shape[0]),
                flags=self._cv2.INTER_LINEAR,
                borderMode=self._cv2.BORDER_REPLICATE,
            ),
            alignment,
        )

    def _components(
        self,
        frame: Any,
        detections: list[Detection],
    ) -> list[tuple[BBox, float, bool, str]] | None:
        cv2, np = self._cv2, self._np
        self._scene_anomaly_reason = None
        self._frame_evidence = None
        aligned = self._aligned_gray(frame)
        if aligned is None:
            self._scene_anomaly_reason = "camera_shift_or_unreliable_alignment"
            return None
        current, alignment = aligned
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

        person_mask = self._person_mask(detections, changed.shape, alignment)
        baseline_mask = self._baseline_mask(changed.shape)
        valid = cv2.bitwise_not(cv2.bitwise_or(person_mask, baseline_mask))
        changed_after_mask = cv2.bitwise_and(changed, valid)
        valid_pixels = int(cv2.countNonZero(valid))
        if valid_pixels == 0:
            return []
        if cv2.countNonZero(changed_after_mask) / valid_pixels > self.global_change_ratio:
            self._scene_anomaly_reason = "global_scene_change"
            return None
        self._frame_evidence = _FrameChangeEvidence(
            changed, baseline_mask, person_mask, alignment
        )

        contour_result = cv2.findContours(changed_after_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contour_result[-2]
        roi_box = self.roi.bbox
        roi_area = roi_box.area
        min_area = self._min_component_area()
        small_min_area = (
            self._small_min_component_area()
            if self.small_component.enabled
            else min_area
        )
        max_area = round(roi_area * self.max_component_ratio)
        result: list[tuple[BBox, float, bool, str]] = []
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            area = int(cv2.contourArea(contour))
            if area < small_min_area or area > max_area or width < 8 or height < 8:
                continue
            score = float(cv2.countNonZero(changed_after_mask[y : y + height, x : x + width])) / max(
                1, width * height
            )
            is_small = area < min_area
            aspect_ratio = min(width, height) / max(width, height)
            if is_small and (
                score < self.small_component.min_fill_ratio
                or aspect_ratio < self.small_component.min_aspect_ratio
            ):
                continue
            component = BBox(
                roi_box.x1 + int(x),
                roi_box.y1 + int(y),
                roi_box.x1 + int(x + width),
                roi_box.y1 + int(y + height),
            )
            result.append(
                (
                    component,
                    max(0.0, min(1.0, score)),
                    is_small,
                    "reference_change",
                )
            )

        if self.baseline_residual.enabled:
            visible_baseline = cv2.bitwise_and(baseline_mask, cv2.bitwise_not(person_mask))
            residual_mask = cv2.bitwise_and(changed, visible_baseline)
            residual_contours = cv2.findContours(
                residual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[-2]
            residual_min_area = self._baseline_residual_min_area()
            for contour in residual_contours:
                x, y, width, height = cv2.boundingRect(contour)
                area = int(cv2.contourArea(contour))
                if (
                    area < residual_min_area
                    or area > max_area
                    or width < 8
                    or height < 8
                ):
                    continue
                box_area = max(1, width * height)
                score = float(
                    cv2.countNonZero(residual_mask[y : y + height, x : x + width])
                ) / box_area
                person_overlap = float(
                    cv2.countNonZero(person_mask[y : y + height, x : x + width])
                ) / box_area
                if (
                    score < self.baseline_residual.min_fill_ratio
                    or person_overlap > self.baseline_residual.max_person_overlap_ratio
                ):
                    continue
                component = BBox(
                    roi_box.x1 + int(x),
                    roi_box.y1 + int(y),
                    roi_box.x1 + int(x + width),
                    roi_box.y1 + int(y + height),
                )
                baseline_coverage, overlap_count = self._baseline_coverage(component)
                if (
                    baseline_coverage
                    >= self.baseline_residual.max_baseline_coverage_ratio
                    and overlap_count == 1
                ):
                    continue
                result.append(
                    (
                        component,
                        max(0.0, min(1.0, score)),
                        False,
                        "baseline_residual",
                    )
                )
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
        alignment = self._alignment_result
        if alignment is None or not alignment.valid:
            self._regions.clear()
            self._last_timestamp_sec = timestamp_sec
            return []
        for box, score, is_small, proposal_source in components:
            candidates: list[tuple[float, float, int]] = []
            for index, region in enumerate(self._regions):
                if index in used_regions:
                    continue
                if region.proposal_source != proposal_source:
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
                if region.small_component != is_small:
                    region.small_component = is_small
                    region.anchor_bbox = box
                    region.first_seen_sec = timestamp_sec
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
                region = _PersistentRegion(
                    box,
                    box,
                    timestamp_sec,
                    timestamp_sec,
                    score,
                    small_component=is_small,
                    proposal_source=proposal_source,
                )
                self._regions.append(region)
                used_regions.add(len(self._regions) - 1)

            if region.proposal_source == "baseline_residual":
                required_seconds = self.baseline_residual.stable_seconds
            else:
                required_seconds = (
                    self.small_component.stable_seconds
                    if region.small_component
                    else self.stable_seconds
                ) + (
                    self.edge_extra_seconds
                    if self._is_edge_component(region.bbox)
                    else 0.0
                )
            if timestamp_sec - region.first_seen_sec + 1e-9 < required_seconds:
                continue
            current_bbox = self._reference_to_current_bbox(region.bbox, alignment)
            if current_bbox is None:
                continue
            proposals.append(
                Detection(
                    bbox=current_bbox,
                    confidence=max(0.35, min(0.99, region.score)),
                    class_id=SCENE_CHANGE_CLASS_ID,
                    class_name="scene_change",
                    proposal_source=region.proposal_source,
                    reference_change_score=region.score,
                    event_candidate=True,
                    change_bbox=current_bbox,
                )
            )

        self._last_timestamp_sec = timestamp_sec
        return proposals


class HybridDetector:
    """Combine YOLO semantics with persistent reference-change proposals."""

    def __init__(
        self,
        detector: Any,
        change_detector: ReferenceChangeDetector,
        *,
        fusion_min_overlap_ratio: float = 0.25,
        fusion_min_iou: float = 0.10,
    ) -> None:
        if not 0.0 <= float(fusion_min_overlap_ratio) <= 1.0:
            raise ValueError("fusion_min_overlap_ratio must be between 0 and 1")
        if not 0.0 <= float(fusion_min_iou) <= 1.0:
            raise ValueError("fusion_min_iou must be between 0 and 1")
        self.detector = detector
        self.change_detector = change_detector
        self.fusion_min_overlap_ratio = float(fusion_min_overlap_ratio)
        self.fusion_min_iou = float(fusion_min_iou)
        self._last_timestamp_sec: float | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "yolo_reference_change",
            "detector": getattr(self.detector, "metadata", {}),
            "reference_change": self.change_detector.metadata,
            "fusion_min_overlap_ratio": self.fusion_min_overlap_ratio,
            "fusion_min_iou": self.fusion_min_iou,
        }

    @property
    def scene_anomaly_reason(self) -> str | None:
        return self.change_detector.scene_anomaly_reason

    @property
    def scene_debug(self) -> dict[str, Any] | None:
        return self.change_detector.scene_debug

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

    def _fusion_match(self, change: BBox, detection: BBox) -> bool:
        return (
            self._overlap_ratio(change, detection) >= self.fusion_min_overlap_ratio
            and change.iou(detection) >= self.fusion_min_iou
        )

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
        alignment_debug = getattr(self.change_detector, "scene_debug", None) or {}
        alignment_config = getattr(self.change_detector, "alignment", None)
        maximum_unfused_shift = (
            hypot(float(roi.bbox.width), float(roi.bbox.height))
            * float(alignment_config.max_unfused_translation_ratio)
            if alignment_config is not None
            else float("inf")
        )
        allow_unfused_change = (
            float(alignment_debug.get("magnitude", 0.0)) <= maximum_unfused_shift
        )
        for change in changes:
            if change.proposal_source == "baseline_residual":
                output.append(change)
                continue
            matches = [
                (
                    change.bbox.iou(detection.bbox),
                    self._overlap_ratio(change.bbox, detection.bbox),
                    detection.confidence,
                    index,
                    detection,
                )
                for index, detection in enumerate(yolo_detections)
                if index not in consumed
                and not self._person(detection)
                and self._fusion_match(change.bbox, detection.bbox)
            ]
            if matches:
                _iou, _overlap, _confidence, index, detection = max(
                    matches,
                    key=lambda item: (item[0], item[1], item[2]),
                )
                consumed.add(index)
                output.append(
                    Detection(
                        bbox=detection.bbox,
                        confidence=max(change.confidence, detection.confidence),
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                        mask=detection.mask,
                        proposal_source="fused",
                        reference_change_score=change.reference_change_score,
                        event_candidate=True,
                        change_bbox=change.bbox,
                    )
                )
            else:
                if allow_unfused_change:
                    output.append(change)

        for index, detection in enumerate(yolo_detections):
            if index in consumed:
                continue
            baseline_residual_enabled = bool(
                getattr(
                    getattr(self.change_detector, "baseline_residual", None),
                    "enabled",
                    False,
                )
            )
            score = (
                None
                if self._person(detection)
                or baseline_residual_enabled
                or any(
                    self._fusion_match(change.bbox, detection.bbox)
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
