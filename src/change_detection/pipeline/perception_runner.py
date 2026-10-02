"""Deterministic M2 frame loop and debug artifact writer."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from change_detection.dataset.io import write_json
from change_detection.domain import FrameContext, Observation, Track
from change_detection.events import EventArtifactWriter, EventEngine, EventStore
from change_detection.perception import Detector, EmbeddingRefresher, ObservationBuilder, Tracker
from change_detection.association import AssociationEngine
from change_detection.memory import ObjectMemory
from change_detection.scene import SceneStatusProvider, StableSceneStatusProvider


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
    events_path: Path | None = None
    event_lifecycle_path: Path | None = None
    snapshots_dir: Path | None = None
    event_count: int = 0


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
        warmup_seconds: float = 0.0,
        embedding_refresher: EmbeddingRefresher | None = None,
        association_engine: AssociationEngine | None = None,
        object_memory: ObjectMemory | None = None,
        event_engine: EventEngine | None = None,
        event_store: EventStore | None = None,
        scene_status_provider: SceneStatusProvider | None = None,
        reference_image: Any | None = None,
    ) -> None:
        if float(processing_fps) <= 0:
            raise ValueError("processing_fps must be positive")
        if not math.isfinite(float(warmup_seconds)) or float(warmup_seconds) < 0:
            raise ValueError("warmup_seconds must be finite and non-negative")
        self.source = source
        self.roi = roi
        self.detector = detector
        self.tracker = tracker
        self.observation_builder = observation_builder or ObservationBuilder()
        self.processing_fps = float(processing_fps)
        self.warmup_seconds = float(warmup_seconds)
        identity_parts = (embedding_refresher, association_engine, object_memory)
        if any(item is not None for item in identity_parts) and not all(item is not None for item in identity_parts):
            raise ValueError("M3 requires embedding_refresher, association_engine, and object_memory together")
        self.embedding_refresher = embedding_refresher
        self.association_engine = association_engine
        self.object_memory = object_memory
        event_parts = (event_engine, event_store)
        if any(item is not None for item in event_parts) and not all(item is not None for item in event_parts):
            raise ValueError("Event runtime requires event_engine and event_store together")
        if event_engine is not None and object_memory is None:
            raise ValueError("Event runtime requires ObjectMemory")
        self.event_engine = event_engine
        self.event_store = event_store
        self.scene_status_provider = scene_status_provider or StableSceneStatusProvider()
        self.reference_image = reference_image

    def run(self, output_dir: str | Path, *, run_metadata: dict[str, Any] | None = None) -> PerceptionRunResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        observations_path = output_dir / "observations.jsonl"
        video_path = output_dir / "annotated.mp4"
        identity_path = output_dir / "identity.jsonl" if self.object_memory is not None else None
        events_path = output_dir / "events.jsonl" if self.event_engine is not None else None
        event_lifecycle_path = output_dir / "event_lifecycle.jsonl" if self.event_engine is not None else None
        snapshots_dir = output_dir / "snapshots" if self.event_engine is not None else None
        metadata_path = output_dir / "metadata.json"
        cv2 = _cv2()
        writer = None
        event_writer = (
            EventArtifactWriter(output_dir, reference_image=self.reference_image)
            if self.event_engine is not None
            else None
        )
        event_writer_finalized = False
        frames_read = frames_processed = detection_count = track_count = observation_count = identity_assignment_count = 0
        started = time.perf_counter()
        next_process_time: float | None = None
        first_processed_timestamp: float | None = None
        self.tracker.reset()
        detector_reset = getattr(self.detector, "reset", None)
        if callable(detector_reset):
            detector_reset()
        if self.embedding_refresher is not None:
            self.embedding_refresher.reset()
        if self.object_memory is not None:
            self.object_memory.reset()
        if self.event_engine is not None:
            assert self.event_store is not None
            self.event_engine.reset()
            self.event_store.reset()
            self.scene_status_provider.reset()

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

                    if first_processed_timestamp is None:
                        first_processed_timestamp = frame.timestamp_sec
                        if self.object_memory is not None:
                            # Baseline records are seeded independently of
                            # the runtime media clock.  Align them once to
                            # the first processed frame so a seeked run does
                            # not look like a long re-ID timeout.
                            self.object_memory.reset(timestamp_sec=first_processed_timestamp)

                    scene_status = None
                    if self.event_engine is not None:
                        scene_status = self.scene_status_provider.update(frame)
                        if (
                            self.warmup_seconds > 0
                            and first_processed_timestamp is not None
                            and frame.timestamp_sec - first_processed_timestamp < self.warmup_seconds
                        ):
                            scene_status = replace(
                                scene_status,
                                stable=False,
                                reason=scene_status.reason or "runtime_warmup",
                            )

                    detect_at = getattr(self.detector, "detect_at", None)
                    if callable(detect_at):
                        detections = detect_at(
                            frame.image,
                            self.roi,
                            timestamp_sec=frame.timestamp_sec,
                        )
                    else:
                        detections = self.detector.detect(frame.image, self.roi)
                    anomaly_reason = getattr(self.detector, "scene_anomaly_reason", None)
                    if scene_status is not None and anomaly_reason:
                        scene_status = replace(
                            scene_status,
                            stable=False,
                            reason=scene_status.reason or str(anomaly_reason),
                        )
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
                                    **(
                                        {"scene_status": scene_status.to_dict()}
                                        if scene_status is not None
                                        else {}
                                    ),
                                    **(
                                        {"scene_alignment": scene_debug}
                                        if (
                                            scene_debug := getattr(
                                                self.detector, "scene_debug", None
                                            )
                                        )
                                        is not None
                                        else {}
                                    ),
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
                        if self.event_engine is not None:
                            assert self.event_store is not None and event_writer is not None
                            assert scene_status is not None
                            actions = self.event_engine.update(
                                self.object_memory,
                                memory_update,
                                observations,
                                timestamp_sec=frame.timestamp_sec,
                                scene_status=scene_status,
                                roi_bbox=self.roi.bbox,
                            )
                            for action in actions:
                                result = self.event_store.apply(action)
                                if result is None:
                                    continue
                                event_writer.write_lifecycle(result)
                                if result.public_event is not None:
                                    event_writer.write_confirmation_snapshot(
                                        result.public_event,
                                        frame.image,
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
            if self.event_engine is not None:
                assert self.event_store is not None and event_writer is not None
                event_writer.finalize(self.event_store)
                event_writer_finalized = True
        finally:
            if writer is not None:
                writer.release()
            if event_writer is not None and not event_writer_finalized:
                event_writer.close()

        elapsed = max(0.0, time.perf_counter() - started)
        metadata = {
            "schema_version": 1,
            "source": self.source.metadata.to_dict(),
            "roi": self.roi.to_dict(),
            "processing_fps_configured": self.processing_fps,
            "warmup_seconds": self.warmup_seconds,
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
        if self.event_engine is not None:
            assert self.event_store is not None
            metadata["events"] = {
                "enabled": True,
                "confirmed_count": len(self.event_store.confirmed_events),
                "active_count": len(self.event_store.active_events),
            }
            metadata["artifacts"].update(
                {
                    "events": events_path.name if events_path is not None else None,
                    "event_lifecycle": (
                        event_lifecycle_path.name if event_lifecycle_path is not None else None
                    ),
                    "snapshots": snapshots_dir.name if snapshots_dir is not None else None,
                }
            )
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
            events_path=events_path,
            event_lifecycle_path=event_lifecycle_path,
            snapshots_dir=snapshots_dir,
            event_count=(len(self.event_store.confirmed_events) if self.event_store is not None else 0),
        )
