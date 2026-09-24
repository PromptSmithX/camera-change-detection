"""Create an M1 SceneBaseline from a configured frame source.

Example:
    python tools/calibrate.py --config configs/m1.example.json --output runs/run-001/baseline

The command only writes under ``runs/`` by default and never changes the
canonical dataset or locked test artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from uuid import NAMESPACE_URL, uuid5
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.config import ConfigValidationError, load_config
from change_detection.domain import BaselineObject, Observation, Point
from change_detection.infrastructure.models import DinoV2Encoder, YoloDetector
from change_detection.perception import PerceptionDependencyError
from change_detection.pipeline import build_roi, build_source, resolve_input
from change_detection.scene import CalibrationError, CalibrationService, ROIError
from change_detection.sources import SourceError


def _resolve_input(root: Path, value: str) -> Path:
    return resolve_input(root, value)


def _resolve_output(root: Path, value: str) -> Path:
    output = _resolve_input(root, value)
    runs_root = (root / "runs").resolve()
    if output != runs_root and runs_root not in output.parents:
        raise ValueError("Calibration output must be under the repository runs/ directory")
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


def _build_source(config, root: Path):
    return build_source(config, root)


def _build_roi(config, root: Path, source):
    return build_roi(config, root, source)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/m1/baseline"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    source = None
    try:
        config = load_config(config_path.resolve())
        source = _build_source(config, root)
        roi = _build_roi(config, root, source)
        service = CalibrationService(config.calibration, config.stability)
        with source:
            result = service.calibrate(source, roi)
        if config.perception.enabled:
            _prepare_runtime_environment(root)
            detector_config = config.perception.detector
            detector = YoloDetector(
                _resolve_model(root, detector_config.model),
                confidence=detector_config.confidence,
                iou=detector_config.iou,
                device=detector_config.device,
                classes=detector_config.classes,
                max_detections=detector_config.max_detections,
            )
            baseline_detections = detector.detect(result.baseline.reference_image, roi)
            if config.perception.encoder.enabled:
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
                observations = [
                    Observation(
                        frame_index=result.baseline.reference_frame_index,
                        timestamp_sec=result.baseline.reference_timestamp_sec,
                        bbox=detection.bbox,
                        centroid=Point(
                            (detection.bbox.x1 + detection.bbox.x2) / 2.0,
                            (detection.bbox.y1 + detection.bbox.y2) / 2.0,
                        ),
                        detector_class=detection.class_name,
                        detector_class_id=detection.class_id,
                        detector_confidence=detection.confidence,
                        tracker_id=None,
                    )
                    for detection in baseline_detections
                ]
                embeddings = encoder.encode(result.baseline.reference_image, observations, roi.bbox)
                identity_objects = tuple(
                    BaselineObject(
                        object_id=str(
                            uuid5(
                                NAMESPACE_URL,
                                f"{result.baseline.source.source_id}:"
                                f"{result.baseline.reference_frame_index}:{index}:"
                                f"{detection.class_id}:{detection.bbox.to_list()}",
                            )
                        ),
                        bbox=detection.bbox,
                        class_id=detection.class_id,
                        class_name=detection.class_name,
                        confidence=detection.confidence,
                        embedding=embedding,
                    )
                    for index, (detection, embedding) in enumerate(
                        zip(baseline_detections, embeddings, strict=True)
                    )
                )
                result.baseline.set_identity_baseline(identity_objects)
            else:
                result.baseline.baseline_objects = tuple(
                    detection.to_dict() for detection in baseline_detections
                )
        metadata_path = result.baseline.save(_resolve_output(root, str(args.output)))
        print(json.dumps({"baseline": str(metadata_path), **result.baseline.to_dict()}, indent=2))
        return 0
    except (
        ConfigValidationError,
        CalibrationError,
        PerceptionDependencyError,
        ROIError,
        SourceError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        if isinstance(exc, CalibrationError) and exc.diagnostics:
            print(json.dumps(exc.diagnostics, indent=2), file=sys.stderr)
        return 1
    finally:
        if source is not None and not source.closed:
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
