"""Deterministic M2 frame loop and debug artifact writer."""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from change_detection.dataset.io import write_json
from change_detection.domain import FrameContext, Observation, Track
from change_detection.perception import Detector, EmbeddingRefresher, ObservationBuilder, Tracker
from change_detection.association import AssociationEngine
from change_detection.memory import ObjectMemory


class PerceptionRunError(RuntimeError):
    """Raised when M2 cannot create or write its debug artifacts."""


@dataclass(frozen=True, slots=True)
class PerceptionRunResult:
    output_dir: Path
    metadata_path: Path
    observations_path: Path
    video_path: Path
    identity_path: Path | None
    frames_read: int
    frames_processed: int
    detection_count: int
    track_count: int
    observation_count: int
    identity_assignment_count: int = 0


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional runtime extra
        raise PerceptionRunError(
            "M2 debug video requires OpenCV; install the runtime extra"
        ) from exc
    return cv2


def _draw_frame(
    frame: Any,
    roi: Any,
    observations: list[Observation],
    object_ids: dict[int, str] | None = None,
) -> Any:
    cv2 = _cv2()
    canvas = frame.copy()
    box = roi.bbox
    cv2.rectangle(canvas, (box.x1, box.y1), (box.x2 - 1, box.y2 - 1), (255, 180, 0), 2)
    for index, observation in enumerate(observations):
        bbox = observation.bbox
        cv2.rectangle(
            canvas,
            (bbox.x1, bbox.y1),
            (bbox.x2 - 1, bbox.y2 - 1),
            (0, 220, 0),
            2,
        )
        label = (
            f"{observation.detector_class} "
            f"{observation.detector_confidence:.2f} "
            f"tid={observation.tracker_id}"
        )
        if object_ids and index in object_ids:
            label += f" oid={object_ids[index][:8]}"
        text_y = max(16, bbox.y1 - 6)
        cv2.putText(
            canvas,
            label,
            (bbox.x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


class PerceptionRunner:
    """Run detector/tracker/observation stages and write inspectable outputs."""

    def __init__(
        self,
        *,
        source: Any,
        roi: Any,
        detector: Detector,
        tracker: Tracker,
        observation_builder: ObservationBuilder | None = None,
        processing_fps: float = 5.0,
        embedding_refresher: EmbeddingRefresher | None = None,
        association_engine: AssociationEngine | None = None,
        object_memory: ObjectMemory | None = None,
    ) -> None:
        if float(processing_fps) <= 0:
            raise ValueError("processing_fps must be positive")
        self.source = source
        self.roi = roi
        self.detector = detector
        self.tracker = tracker
        self.observation_builder = observation_builder or ObservationBuilder()
        self.processing_fps = float(processing_fps)
        identity_parts = (embedding_refresher, association_engine, object_memory)
        if any(item is not None for item in identity_parts) and not all(item is not None for item in identity_parts):
            raise ValueError("M3 requires embedding_refresher, association_engine, and object_memory together")
        self.embedding_refresher = embedding_refresher
        self.association_engine = association_engine
        self.object_memory = object_memory

    def run(self, output_dir: str | Path, *, run_metadata: dict[str, Any] | None = None) -> PerceptionRunResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        observations_path = output_dir / "observations.jsonl"
        video_path = output_dir / "annotated.mp4"
        identity_path = output_dir / "identity.jsonl" if self.object_memory is not None else None
        metadata_path = output_dir / "metadata.json"
        cv2 = _cv2()
        writer = None
        frames_read = frames_processed = detection_count = track_count = observation_count = identity_assignment_count = 0
        started = time.perf_counter()
        next_process_time: float | None = None
        self.tracker.reset()
        if self.embedding_refresher is not None:
            self.embedding_refresher.reset()
        if self.object_memory is not None:
            self.object_memory.reset()

        try:
            identity_context = (
                identity_path.open("w", encoding="utf-8", newline="\n")
                if identity_path is not None
                else nullcontext(None)
            )
            with observations_path.open("w", encoding="utf-8", newline="\n") as records, identity_context as identity_records:
                while True:
                    frame: FrameContext | None = self.source.read()
                    if frame is None:
                        break
                    frames_read += 1
                    if next_process_time is None:
                        next_process_time = frame.timestamp_sec
                    if frame.timestamp_sec + 1e-9 < next_process_time:
                        continue
                    interval = 1.0 / self.processing_fps
                    while next_process_time <= frame.timestamp_sec + 1e-9:
                        next_process_time += interval

                    detections = self.detector.detect(frame.image, self.roi)
                    tracks = self.tracker.update(detections, frame.image)
                    observations = self.observation_builder.build(
                        frame, detections, tracks, self.roi
                    )
                    object_ids: dict[int, str] = {}
                    if self.object_memory is not None:
                        assert self.embedding_refresher is not None and self.association_engine is not None
                        refresh = self.embedding_refresher.enrich(frame, observations, self.roi.bbox)
                        associations = self.association_engine.match(
                            self.object_memory.objects,
                            observations,
                            roi_bbox=self.roi.bbox,
                            timestamp_sec=frame.timestamp_sec,
                        )
                        memory_update = self.object_memory.update(
                            observations,
                            associations,
                            timestamp_sec=frame.timestamp_sec,
                        )
                        object_ids = {
                            item.observation_index: item.object_id for item in memory_update.assignments
                        }
                        identity_assignment_count += len(memory_update.assignments)
                        assert identity_records is not None
                        identity_records.write(
                            json.dumps(
                                {
                                    "source_id": frame.source_id,
                                    "frame_index": frame.frame_index,
                                    "timestamp_sec": frame.timestamp_sec,
                                    "refresh": {
                                        "encoded_count": refresh.encoded_count,
                                        "reused_count": refresh.reused_count,
                                    },
                                    "assignments": [
                                        {
                                            **item.to_dict(),
                                            "tracker_id": observations[item.observation_index].tracker_id,
                                        }
                                        for item in memory_update.assignments
                                    ],
                                    "candidates": [item.to_dict() for item in associations.candidates],
                                    "transitions": list(memory_update.transitions),
                                    "memory_objects": [item.to_debug_dict() for item in self.object_memory.objects],
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    if writer is None:
                        height, width = frame.image.shape[:2]
                        writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            self.processing_fps,
                            (int(width), int(height)),
                        )
                        if not writer.isOpened():
                            raise PerceptionRunError(f"Could not open video writer: {video_path}")
                    annotated = _draw_frame(frame.image, self.roi, observations, object_ids)
                    writer.write(annotated)
                    records.write(
                        json.dumps(
                            {
                                "source_id": frame.source_id,
                                "frame_index": frame.frame_index,
                                "timestamp_sec": frame.timestamp_sec,
                                "detections": [item.to_dict() for item in detections],
                                "tracks": [item.to_dict() for item in tracks],
                                "observations": [item.to_dict() for item in observations],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    frames_processed += 1
                    detection_count += len(detections)
                    track_count += len(tracks)
                    observation_count += len(observations)
        finally:
            if writer is not None:
                writer.release()

        elapsed = max(0.0, time.perf_counter() - started)
        metadata = {
            "schema_version": 1,
            "source": self.source.metadata.to_dict(),
            "roi": self.roi.to_dict(),
            "processing_fps_configured": self.processing_fps,
            "processing_fps_actual": frames_processed / elapsed if elapsed else 0.0,
            "frames_read": frames_read,
            "frames_processed": frames_processed,
            "detection_count": detection_count,
            "track_count": track_count,
            "observation_count": observation_count,
            "detector": getattr(self.detector, "metadata", {}),
            "tracker": getattr(self.tracker, "metadata", {}),
            "artifacts": {
                "observations": observations_path.name,
                "annotated_video": video_path.name,
            },
        }
        if run_metadata:
            metadata["run"] = run_metadata
        if self.object_memory is not None:
            metadata["identity"] = {
                "enabled": True,
                "identity_assignments": identity_assignment_count,
                "encoder": getattr(self.embedding_refresher.encoder, "metadata", {}),
                "association": asdict(self.association_engine.config),
                "memory": {
                    "missing_grace_seconds": self.object_memory.config.missing_grace_seconds,
                    "observation_history_size": self.object_memory.config.observation_history_size,
                    "embedding_history_size": self.object_memory.config.embedding_history_size,
                },
            }
            metadata["artifacts"]["identity"] = identity_path.name if identity_path is not None else None
        write_json(metadata_path, metadata)
        return PerceptionRunResult(
            output_dir=output_dir,
            metadata_path=metadata_path,
            observations_path=observations_path,
            video_path=video_path,
            identity_path=identity_path,
            frames_read=frames_read,
            frames_processed=frames_processed,
            detection_count=detection_count,
            track_count=track_count,
            observation_count=observation_count,
            identity_assignment_count=identity_assignment_count,
        )
