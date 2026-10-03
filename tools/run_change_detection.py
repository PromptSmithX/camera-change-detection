"""Run M3 identity plus M4/M5 event detection.

The command writes only under ``runs/`` and keeps the M2 observation and M3
identity artifacts alongside public event and snapshot artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.association import AssociationEngine
from change_detection.config import ConfigValidationError, load_config
from change_detection.dataset.io import sha256_file
from change_detection.events import EventEngine, EventOutputError, EventStore
from change_detection.infrastructure.models import (
    DinoV2Encoder,
    HybridDetector,
    ReferenceChangeDetector,
    YoloDetector,
)
from change_detection.infrastructure.tracking import ByteTrackAdapter
from change_detection.memory import ObjectMemory
from change_detection.perception import (
    EmbeddingRefresher,
    PerceptionDependencyError,
    ProposalConsolidator,
)
from change_detection.pipeline import (
    PerceptionRunError,
    PerceptionRunner,
    build_roi,
    build_source,
    resolve_input,
)
from change_detection.scene import BaselineIntegrityError, SceneBaseline, StableSceneStatusProvider
from change_detection.sources import SourceError


def _resolve_output(root: Path, value: str | Path) -> Path:
    output = resolve_input(root, value)
    runs_root = (root / "runs").resolve()
    if output != runs_root and runs_root not in output.parents:
        raise ValueError("Change-detection output must be under the repository runs/ directory")
    return output


def _prepare_runtime_environment(root: Path) -> None:
    config_root = root / "runs" / "ultralytics-config"
    (config_root / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))


def _torch_cache(root: Path) -> Path:
    cache = root / "runs" / "torch-hub"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _resolve_model(root: Path, model: str) -> str:
    candidate = Path(model)
    if candidate.is_absolute():
        return str(candidate)
    local_candidate = root / candidate
    return str(local_candidate.resolve()) if local_candidate.is_file() else model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (defaults to runs/m45/change_detection/<source_id>)",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    source = None
    try:
        config = load_config(config_path.resolve())
        if not config.perception.enabled or not config.perception.encoder.enabled:
            raise ValueError(
                "M4/M5 runtime requires perception.enabled=true and "
                "perception.encoder.enabled=true"
            )
        if not config.baseline.path:
            raise ValueError("M4/M5 runtime requires baseline.path")

        _prepare_runtime_environment(root)
        source = build_source(config, root)
        roi = build_roi(config, root, source)
        baseline_path = resolve_input(root, config.baseline.path)
        baseline = SceneBaseline.load(baseline_path)
        if baseline.schema_version != 2:
            raise BaselineIntegrityError("M4/M5 requires an M3 schema-v2 baseline")
        if baseline.source.source_id != source.metadata.source_id:
            raise ValueError("Baseline source_id does not match runtime source")
        if (baseline.source.width, baseline.source.height) != (
            source.metadata.width,
            source.metadata.height,
        ):
            raise ValueError("Baseline dimensions do not match runtime source")
        if baseline.roi.bbox != roi.bbox:
            raise ValueError("Baseline ROI does not match runtime ROI")

        start_frame = config.runtime.start_frame
        if start_frame is None and source.metadata.media_type != "webcam":
            start_frame = baseline.reference_frame_index
        if start_frame is not None and source.metadata.media_type != "webcam":
            source.seek(start_frame)

        detector_config = config.perception.detector
        tracker_config = config.perception.tracker
        yolo_detector = YoloDetector(
            _resolve_model(root, detector_config.model),
            confidence=detector_config.confidence,
            iou=detector_config.iou,
            device=detector_config.device,
            classes=detector_config.classes,
            max_detections=detector_config.max_detections,
        )
        tracker = ByteTrackAdapter(
            track_activation_threshold=tracker_config.track_activation_threshold,
            lost_track_buffer=tracker_config.lost_track_buffer,
            minimum_matching_threshold=tracker_config.minimum_matching_threshold,
            minimum_consecutive_frames=tracker_config.minimum_consecutive_frames,
        )
        encoder_config = config.perception.encoder
        encoder = DinoV2Encoder(
            encoder_config.model,
            repository=encoder_config.repository,
            device=encoder_config.device or detector_config.device,
            input_size=encoder_config.input_size,
            batch_size=encoder_config.batch_size,
            crop_padding_ratio=encoder_config.crop_padding_ratio,
            cache_dir=_torch_cache(root),
        )
        reference_image = baseline.load_reference_image(baseline_path)
        detector = HybridDetector(
            yolo_detector,
            ReferenceChangeDetector(
                reference_image,
                roi,
                baseline_boxes=(item.bbox for item in baseline.identity_objects()),
                min_component_ratio=(
                    config.perception.reference_change.min_component_ratio
                ),
                min_component_area_px=(
                    config.perception.reference_change.min_component_area_px
                ),
                stable_seconds=config.perception.reference_change.stable_seconds,
                min_current_edge_ratio=(
                    config.perception.reference_change.min_current_edge_ratio
                ),
                small_component=config.perception.reference_change.small_component,
                masked_overlap=config.perception.reference_change.masked_overlap,
                alignment=config.perception.reference_change.alignment,
                baseline_residual=config.perception.reference_change.baseline_residual,
            ),
            fusion_min_overlap_ratio=(
                config.perception.reference_change.fusion_min_overlap_ratio
            ),
            fusion_min_iou=config.perception.reference_change.fusion_min_iou,
            emit_standalone_reference_change=(
                config.perception.reference_change.emit_standalone
            ),
        )
        event_store = EventStore()
        event_engine = EventEngine(forgotten=config.forgotten, moved=config.moved)

        with source:
            result = PerceptionRunner(
                source=source,
                roi=roi,
                detector=detector,
                tracker=tracker,
                proposal_consolidator=(
                    ProposalConsolidator()
                    if config.perception.proposal_fusion.enabled
                    else None
                ),
                processing_fps=config.runtime.processing_fps,
                warmup_seconds=config.runtime.warmup_seconds,
                embedding_refresher=EmbeddingRefresher(
                    encoder,
                    semantic_refresh_fps=encoder_config.semantic_refresh_fps,
                ),
                association_engine=AssociationEngine(
                    config.association,
                    event_reid_iou_threshold=config.forgotten.reid_iou_threshold,
                ),
                object_memory=ObjectMemory(baseline.identity_objects(), config=config.memory),
                event_engine=event_engine,
                event_store=event_store,
                scene_status_provider=StableSceneStatusProvider(),
                reference_image=reference_image,
            ).run(
                _resolve_output(
                    root,
                    args.output
                    if args.output is not None
                    else Path("runs/m45/change_detection") / source.metadata.source_id,
                ),
                run_metadata={
                    "config_path": str(config_path.resolve()),
                    "config": config.to_dict(),
                    "baseline_path": str(baseline_path),
                    "baseline_sha256": sha256_file(baseline_path),
                    "baseline_schema_version": baseline.schema_version,
                    "monitor_start_frame": start_frame,
                },
            )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "metadata": str(result.metadata_path),
                    "observations": str(result.observations_path),
                    "identity": str(result.identity_path) if result.identity_path else None,
                    "events": str(result.events_path) if result.events_path else None,
                    "event_lifecycle": (
                        str(result.event_lifecycle_path)
                        if result.event_lifecycle_path
                        else None
                    ),
                    "snapshots": str(result.snapshots_dir) if result.snapshots_dir else None,
                    "frames_processed": result.frames_processed,
                    "event_count": result.event_count,
                },
                indent=2,
            )
        )
        return 0
    except (
        ConfigValidationError,
        PerceptionDependencyError,
        PerceptionRunError,
        EventOutputError,
        BaselineIntegrityError,
        SourceError,
        OSError,
        ValueError,
    ) as exc:
        print(f"M4/M5 change detection failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if source is not None and not source.closed:
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
