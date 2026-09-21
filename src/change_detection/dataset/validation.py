"""Validation for the canonical benchmark manifest and annotations."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from change_detection.domain import AnnotationStatus, BBox, EventType, SampleAnnotation, Split

from .io import read_json, resolve_repo_path, sha256_file

ValidationMode = Literal["draft", "official"]


@dataclass(slots=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    code: str
    message: str
    sample_id: str | None = None
    event_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ValidationReport:
    manifest_path: str
    mode: ValidationMode
    issues: list[ValidationIssue]
    sample_count: int = 0
    event_count: int = 0

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "mode": self.mode,
            "ok": self.ok,
            "sample_count": self.sample_count,
            "event_count": self.event_count,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _issue(
    issues: list[ValidationIssue],
    severity: Literal["error", "warning"],
    code: str,
    message: str,
    *,
    sample_id: str | None = None,
    event_id: str | None = None,
) -> None:
    issues.append(ValidationIssue(severity, code, message, sample_id, event_id))


def _validate_bbox(
    issues: list[ValidationIssue],
    bbox: BBox | None,
    *,
    sample_id: str,
    event_id: str | None,
    field_name: str,
    width: int,
    height: int,
    roi: BBox | None = None,
) -> None:
    if bbox is None:
        return
    if not bbox.is_valid_for(width, height):
        _issue(
            issues,
            "error",
            "bbox_out_of_bounds",
            f"{field_name} is outside {width}x{height}: {bbox.to_list()}",
            sample_id=sample_id,
            event_id=event_id,
        )
    if roi is not None and not bbox.is_inside(roi):
        _issue(
            issues,
            "error",
            "bbox_outside_roi",
            f"{field_name} is not contained by ROI: {bbox.to_list()}",
            sample_id=sample_id,
            event_id=event_id,
        )


def _event_error_or_warning(mode: ValidationMode, status: AnnotationStatus) -> Literal["error", "warning"]:
    if mode == "official" or status == AnnotationStatus.VERIFIED:
        return "error"
    return "warning"


def _probe_video(path: Path) -> dict[str, Any] | None:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(result.stderr.strip() or "ffprobe failed")
    value = json.loads(result.stdout)
    stream = (value.get("streams") or [{}])[0]
    format_duration = (value.get("format") or {}).get("duration")
    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        # Effective FPS is the relevant time base for frame/time conversion;
        # nominal r_frame_rate can be a container timebase (e.g. 60000/1001
        # for a 30000/1001 stream).
        "fps": stream.get("avg_frame_rate") or stream.get("r_frame_rate"),
        "frame_count": stream.get("nb_frames"),
        "duration_sec": stream.get("duration") or format_duration,
    }


def _fraction(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    text = str(value)
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            return float(numerator) / float(denominator)
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _validate_source_metadata(
    issues: list[ValidationIssue],
    *,
    source_path: Path,
    sample: SampleAnnotation,
    sample_id: str,
    mode: ValidationMode,
) -> None:
    """Compare canonical metadata with the source without importing OpenCV."""

    video_extensions = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}
    if sample.media_type == "image_sequence":
        if not source_path.is_dir():
            _issue(issues, "error", "invalid_image_sequence_source", "Image sequence path is not a directory", sample_id=sample_id)
            return
        pattern = sample.frame_pattern or "*"
        frame_files = [item for item in source_path.glob(pattern) if item.is_file()]
        if len(frame_files) != sample.frame_count:
            _issue(
                issues,
                "error",
                "frame_count_mismatch",
                f"Image sequence has {len(frame_files)} frames; annotation declares {sample.frame_count}",
                sample_id=sample_id,
            )
        return
    if source_path.suffix.lower() not in video_extensions:
        return
    try:
        metadata = _probe_video(source_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        severity: Literal["error", "warning"] = "error" if mode == "official" else "warning"
        _issue(issues, severity, "media_probe_failed", f"Could not probe source media: {exc}", sample_id=sample_id)
        return
    if metadata is None:
        severity = "error" if mode == "official" else "warning"
        _issue(issues, severity, "media_probe_unavailable", "ffprobe is required to verify video metadata", sample_id=sample_id)
        return
    if metadata.get("width") is not None and int(metadata["width"]) != sample.width:
        _issue(issues, "error", "width_mismatch", "Source width does not match annotation", sample_id=sample_id)
    if metadata.get("height") is not None and int(metadata["height"]) != sample.height:
        _issue(issues, "error", "height_mismatch", "Source height does not match annotation", sample_id=sample_id)
    source_fps = _fraction(metadata.get("fps"))
    if source_fps is not None and abs(source_fps - sample.fps) > 0.01:
        _issue(issues, "error", "fps_mismatch", f"Source FPS {source_fps} != annotation FPS {sample.fps}", sample_id=sample_id)
    source_frames = _fraction(metadata.get("frame_count"))
    if source_frames is not None and int(source_frames) != sample.frame_count:
        _issue(issues, "error", "frame_count_mismatch", "Source frame count does not match annotation", sample_id=sample_id)
    source_duration = _fraction(metadata.get("duration_sec"))
    if source_duration is not None and abs(source_duration - sample.duration_sec) > max(0.1, 1.0 / sample.fps):
        _issue(issues, "error", "duration_mismatch", "Source duration does not match annotation", sample_id=sample_id)


def validate_manifest(
    manifest_path: Path,
    *,
    repo_root: Path | None = None,
    mode: ValidationMode = "draft",
    strict_hashes: bool = False,
) -> ValidationReport:
    """Validate a canonical manifest and every referenced annotation.

    Draft mode permits provisional annotations and reports unresolved event
    fields as warnings. Official mode requires verified samples and complete
    event evidence.
    """

    manifest_path = manifest_path.resolve()
    root = (repo_root or manifest_path.parents[2]).resolve()
    issues: list[ValidationIssue] = []
    try:
        manifest = read_json(manifest_path)
    except Exception as exc:  # pragma: no cover - CLI-facing defensive path
        return ValidationReport(str(manifest_path), mode, [ValidationIssue("error", "manifest_parse", str(exc))])

    required_manifest = {
        "dataset_version",
        "annotation_schema_version",
        "samples",
    }
    for field_name in sorted(required_manifest - set(manifest)):
        _issue(issues, "error", "missing_manifest_field", f"Manifest missing {field_name}")

    samples = manifest.get("samples", [])
    if not isinstance(samples, list):
        _issue(issues, "error", "invalid_samples", "Manifest samples must be an array")
        return ValidationReport(str(manifest_path), mode, issues)

    sample_ids: set[str] = set()
    annotation_paths: set[str] = set()
    source_paths: dict[str, str] = {}
    report = ValidationReport(str(manifest_path), mode, issues, sample_count=len(samples))

    for sample_data in samples:
        if not isinstance(sample_data, Mapping):
            _issue(issues, "error", "invalid_sample", "Sample must be an object")
            continue
        sample_id = str(sample_data.get("video_id", sample_data.get("id", "")))
        if not sample_id:
            _issue(issues, "error", "missing_video_id", "Sample missing video_id")
            continue
        if sample_id in sample_ids:
            _issue(issues, "error", "duplicate_video_id", "Duplicate video_id", sample_id=sample_id)
        sample_ids.add(sample_id)

        required_sample = {
            "video_id",
            "camera_id",
            "path",
            "annotation_path",
            "roi_path",
            "split",
            "scenario_tags",
        }
        for field_name in sorted(required_sample - set(sample_data)):
            _issue(
                issues,
                "error",
                "missing_sample_field",
                f"Sample missing {field_name}",
                sample_id=sample_id,
            )

        split_value = sample_data.get("split")
        if split_value not in {Split.VALIDATION.value, Split.TEST.value}:
            _issue(issues, "error", "invalid_split", f"Invalid split: {split_value!r}", sample_id=sample_id)

        status_value = sample_data.get("quality_status", AnnotationStatus.PROVISIONAL.value)
        try:
            status = AnnotationStatus(status_value)
        except ValueError:
            _issue(issues, "error", "invalid_quality_status", f"Invalid quality_status: {status_value!r}", sample_id=sample_id)
            status = AnnotationStatus.PROVISIONAL
        if mode == "official" and status != AnnotationStatus.VERIFIED:
            _issue(issues, "error", "unverified_sample", "Official benchmark requires verified samples", sample_id=sample_id)

        source_value = str(sample_data.get("path", ""))
        source_paths[sample_id] = source_value
        source_path: Path | None = None
        try:
            source_path = resolve_repo_path(root, source_value)
            if not source_path.exists():
                _issue(issues, "error", "missing_source", f"Source does not exist: {source_value}", sample_id=sample_id)
        except ValueError as exc:
            _issue(issues, "error", "invalid_source_path", str(exc), sample_id=sample_id)

        annotation_value = str(sample_data.get("annotation_path", ""))
        roi_value = str(sample_data.get("roi_path", ""))
        annotation_paths.add(annotation_value)
        try:
            annotation_path = resolve_repo_path(root, annotation_value)
            if not annotation_path.exists():
                _issue(issues, "error", "missing_annotation", f"Annotation does not exist: {annotation_value}", sample_id=sample_id)
                continue
            annotation = read_json(annotation_path)
        except Exception as exc:
            _issue(issues, "error", "annotation_parse", str(exc), sample_id=sample_id)
            continue

        if annotation.get("video_id") != sample_id:
            _issue(issues, "error", "annotation_id_mismatch", "Annotation video_id does not match manifest", sample_id=sample_id)
        if annotation.get("camera_id") != sample_data.get("camera_id"):
            _issue(issues, "error", "annotation_camera_mismatch", "Annotation camera_id does not match manifest", sample_id=sample_id)
        if annotation.get("split") != split_value:
            _issue(issues, "error", "annotation_split_mismatch", "Annotation split does not match manifest", sample_id=sample_id)

        try:
            sample = SampleAnnotation.from_dict(annotation, manifest_sample=sample_data)
        except Exception as exc:
            _issue(issues, "error", "annotation_schema", str(exc), sample_id=sample_id)
            continue

        if sample.quality_status != status:
            _issue(issues, "error", "quality_status_mismatch", "Annotation and manifest quality_status differ", sample_id=sample_id)
        if status == AnnotationStatus.VERIFIED and not sample.reviewer:
            _issue(issues, "error", "missing_reviewer", "Verified sample requires reviewer", sample_id=sample_id)

        if sample.fps <= 0 or sample.width <= 0 or sample.height <= 0 or sample.frame_count <= 0:
            _issue(issues, "error", "invalid_media_metadata", "FPS, dimensions, and frame_count must be positive", sample_id=sample_id)
        if sample.duration_sec <= 0:
            _issue(issues, "error", "invalid_duration", "duration_sec must be positive", sample_id=sample_id)
        if source_path is not None and source_path.exists():
            _validate_source_metadata(
                issues,
                source_path=source_path,
                sample=sample,
                sample_id=sample_id,
                mode=mode,
            )
        if sample.roi_bbox is None:
            _issue(issues, "error", "missing_roi", "ROI bbox is required", sample_id=sample_id)
        else:
            _validate_bbox(issues, sample.roi_bbox, sample_id=sample_id, event_id=None, field_name="roi", width=sample.width, height=sample.height)
        if sample.reference_frame_range is None:
            _issue(issues, "error", "missing_reference", "Reference frame range is required", sample_id=sample_id)
        else:
            start, end = sample.reference_frame_range
            if not (0 <= start <= end < sample.frame_count):
                _issue(issues, "error", "invalid_reference", "Reference frame range is outside media", sample_id=sample_id)

        for event in sample.events:
            report.event_count += 1
            if mode == "official" and event.quality_status != AnnotationStatus.VERIFIED:
                _issue(
                    issues,
                    "error",
                    "unverified_event",
                    "Official benchmark requires verified events",
                    sample_id=sample_id,
                    event_id=event.event_id,
                )
            if not event.event_id:
                _issue(issues, "error", "missing_event_id", "Event missing event_id", sample_id=sample_id)
            if not event.object_id:
                _issue(issues, "error", "missing_object_id", "Event missing object_id", sample_id=sample_id, event_id=event.event_id)
            if event.start_time_sec < 0 or event.start_time_sec > sample.duration_sec:
                _issue(issues, "error", "invalid_start_time", "start_time_sec outside duration", sample_id=sample_id, event_id=event.event_id)
            if event.confirmation_time_sec is not None:
                if not (event.start_time_sec <= event.confirmation_time_sec <= sample.duration_sec):
                    _issue(issues, "error", "invalid_confirmation_time", "confirmation_time_sec must follow start and fit duration", sample_id=sample_id, event_id=event.event_id)
            elif mode == "official":
                _issue(issues, "error", "missing_confirmation_time", "Official event requires confirmation_time_sec", sample_id=sample_id, event_id=event.event_id)
            else:
                _issue(issues, "warning", "missing_confirmation_time", "Provisional event has no confirmation_time_sec", sample_id=sample_id, event_id=event.event_id)
            if event.end_time_sec is not None:
                lower_bound = event.confirmation_time_sec or event.start_time_sec
                if not (lower_bound <= event.end_time_sec <= sample.duration_sec):
                    _issue(issues, "error", "invalid_end_time", "end_time_sec must follow confirmation/start and fit duration", sample_id=sample_id, event_id=event.event_id)
            for field_name, bbox in (
                ("bbox", event.bbox),
                ("baseline_bbox", event.baseline_bbox),
                ("new_bbox", event.new_bbox),
            ):
                _validate_bbox(issues, bbox, sample_id=sample_id, event_id=event.event_id, field_name=field_name, width=sample.width, height=sample.height, roi=sample.roi_bbox)
            for interval_name, intervals in (
                ("occlusion_intervals", event.occlusion_intervals),
                ("lighting_change_intervals", event.lighting_change_intervals),
            ):
                for interval in intervals:
                    if interval[1] > sample.duration_sec:
                        _issue(
                            issues,
                            "error",
                            "interval_outside_duration",
                            f"{interval_name} exceeds sample duration",
                            sample_id=sample_id,
                            event_id=event.event_id,
                        )
            if event.event_type == EventType.FORGOTTEN_OBJECT and event.bbox is None:
                _issue(issues, "error", "missing_forgotten_bbox", "FORGOTTEN_OBJECT requires bbox at confirmation", sample_id=sample_id, event_id=event.event_id)
            if event.event_type == EventType.MOVED_OBJECT:
                severity = _event_error_or_warning(mode, status)
                if event.baseline_bbox is None:
                    _issue(issues, severity, "missing_baseline_bbox", "MOVED_OBJECT requires baseline_bbox", sample_id=sample_id, event_id=event.event_id)
                if event.new_bbox is None:
                    _issue(issues, severity, "missing_new_bbox", "MOVED_OBJECT requires new_bbox", sample_id=sample_id, event_id=event.event_id)
            if event.start_frame is not None and not (0 <= event.start_frame < sample.frame_count):
                _issue(issues, "error", "invalid_start_frame", "start_frame outside media", sample_id=sample_id, event_id=event.event_id)
            if event.confirmation_frame is not None and not (0 <= event.confirmation_frame < sample.frame_count):
                _issue(issues, "error", "invalid_confirmation_frame", "confirmation_frame outside media", sample_id=sample_id, event_id=event.event_id)
            if event.end_frame is not None and not (0 <= event.end_frame < sample.frame_count):
                _issue(issues, "error", "invalid_end_frame", "end_frame outside media", sample_id=sample_id, event_id=event.event_id)
            if event.start_frame is not None and event.confirmation_frame is not None and event.start_frame > event.confirmation_frame:
                _issue(issues, "error", "invalid_frame_order", "start_frame must not follow confirmation_frame", sample_id=sample_id, event_id=event.event_id)
            if event.confirmation_frame is not None and event.end_frame is not None and event.confirmation_frame > event.end_frame:
                _issue(issues, "error", "invalid_frame_order", "confirmation_frame must not follow end_frame", sample_id=sample_id, event_id=event.event_id)
            for field_name, frame_value, time_value in (
                ("start", event.start_frame, event.start_time_sec),
                ("confirmation", event.confirmation_frame, event.confirmation_time_sec),
                ("end", event.end_frame, event.end_time_sec),
            ):
                if frame_value is None:
                    continue
                if time_value is None:
                    severity = "error" if mode == "official" else "warning"
                    _issue(
                        issues,
                        severity,
                        "missing_frame_time_pair",
                        f"{field_name}_frame is present but {field_name}_time_sec is missing",
                        sample_id=sample_id,
                        event_id=event.event_id,
                    )
                    continue
                expected_time = frame_value / sample.fps
                if abs(time_value - expected_time) > max(0.5 / sample.fps, 0.001):
                    _issue(
                        issues,
                        "error",
                        "frame_time_mismatch",
                        f"{field_name} frame/time disagree at FPS {sample.fps}: frame={frame_value}, time={time_value}",
                        sample_id=sample_id,
                        event_id=event.event_id,
                    )

        event_ids = [event.event_id for event in sample.events]
        duplicates = {event_id for event_id in event_ids if event_ids.count(event_id) > 1}
        for event_id in sorted(duplicates):
            _issue(issues, "error", "duplicate_event_id", "Duplicate event_id in sample", sample_id=sample_id, event_id=event_id)

        if mode == "official" and sample.warnings:
            _issue(issues, "warning", "sample_has_warnings", f"Sample retains {len(sample.warnings)} provenance warning(s)", sample_id=sample_id)

        if strict_hashes:
            expected_hash = sample_data.get("annotation_sha256")
            if expected_hash:
                actual_hash = sha256_file(annotation_path)
                if actual_hash != expected_hash:
                    _issue(issues, "error", "annotation_hash_mismatch", "Annotation SHA256 does not match manifest", sample_id=sample_id)
            else:
                _issue(issues, "error", "missing_annotation_hash", "strict_hashes requires annotation_sha256", sample_id=sample_id)

        try:
            roi_path = resolve_repo_path(root, roi_value)
            if not roi_path.exists():
                _issue(issues, "error", "missing_roi_file", f"ROI file does not exist: {roi_value}", sample_id=sample_id)
            else:
                roi_record = read_json(roi_path)
                roi_bbox = BBox.from_value(roi_record.get("bbox", roi_record.get("roi_bbox")))
                if roi_bbox is None:
                    _issue(issues, "error", "roi_file_missing_bbox", "ROI file must contain bbox", sample_id=sample_id)
                elif sample.roi_bbox is not None and roi_bbox != sample.roi_bbox:
                    _issue(issues, "error", "roi_mismatch", "ROI file bbox does not match annotation ROI", sample_id=sample_id)
        except ValueError as exc:
            _issue(issues, "error", "invalid_roi_path", str(exc), sample_id=sample_id)
        except (OSError, json.JSONDecodeError) as exc:
            _issue(issues, "error", "roi_parse", str(exc), sample_id=sample_id)

    for split_name in (Split.VALIDATION.value, Split.TEST.value):
        split_samples = [item for item in samples if item.get("split") == split_name]
        if not split_samples:
            _issue(issues, "warning", "empty_split", f"No samples in {split_name}")

    if len(annotation_paths) != len(samples):
        _issue(issues, "error", "duplicate_annotation_path", "Multiple samples reference the same annotation path")

    return report
