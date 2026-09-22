"""Run the M2 detector/tracker stages and write debug artifacts.

Example:
    python tools/run_perception.py --config configs/m2.example.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.config import ConfigValidationError, load_config
from change_detection.infrastructure.models import YoloDetector
from change_detection.infrastructure.tracking import ByteTrackAdapter
from change_detection.perception import PerceptionDependencyError
from change_detection.pipeline import (
    PerceptionRunError,
    PerceptionRunner,
    build_roi,
    build_source,
    resolve_input,
)
from change_detection.sources import SourceError


def _resolve_output(root: Path, value: str | Path) -> Path:
    output = resolve_input(root, value)
    runs_root = (root / "runs").resolve()
    if output != runs_root and runs_root not in output.parents:
        raise ValueError("M2 output must be under the repository runs/ directory")
    return output


def _prepare_runtime_environment(root: Path) -> None:
    """Keep Ultralytics settings inside ignored project runtime artifacts."""

    config_root = root / "runs" / "ultralytics-config"
    (config_root / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_root))


def _resolve_model(root: Path, model: str) -> str:
    candidate = Path(model)
    if candidate.is_absolute():
        return str(candidate)
    local_candidate = root / candidate
    return str(local_candidate.resolve()) if local_candidate.is_file() else model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/m2/perception"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    source = None
    try:
        config = load_config(config_path.resolve())
        if not config.perception.enabled:
            raise ValueError("M2 perception is disabled; set perception.enabled=true")
        _prepare_runtime_environment(root)
        source = build_source(config, root)
        roi = build_roi(config, root, source)
        detector_config = config.perception.detector
        tracker_config = config.perception.tracker
        detector = YoloDetector(
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
        with source:
            result = PerceptionRunner(
                source=source,
                roi=roi,
                detector=detector,
                tracker=tracker,
                processing_fps=config.runtime.processing_fps,
            ).run(
                _resolve_output(root, args.output),
                run_metadata={
                    "config_path": str(config_path.resolve()),
                    "config": config.to_dict(),
                },
            )
        print(
            json.dumps(
                {
                    "output_dir": str(result.output_dir),
                    "metadata": str(result.metadata_path),
                    "observations": str(result.observations_path),
                    "video": str(result.video_path),
                    "frames_processed": result.frames_processed,
                    "detection_count": result.detection_count,
                    "track_count": result.track_count,
                    "observation_count": result.observation_count,
                },
                indent=2,
            )
        )
        return 0
    except (
        ConfigValidationError,
        PerceptionDependencyError,
        PerceptionRunError,
        SourceError,
        OSError,
        ValueError,
    ) as exc:
        print(f"M2 perception failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if source is not None and not source.closed:
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
