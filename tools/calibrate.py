"""Create an M1 SceneBaseline from a configured frame source.

Example:
    python tools/calibrate.py --config configs/m1.example.json --output runs/run-001/baseline

The command only writes under ``runs/`` by default and never changes the
canonical dataset or locked test artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.config import ConfigValidationError, load_config
from change_detection.scene import BBoxROI, CalibrationError, CalibrationService, ROIError, load_roi_file
from change_detection.sources import ImageSequenceSource, SourceError, VideoSource, WebcamSource


def _resolve_input(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_output(root: Path, value: str) -> Path:
    output = _resolve_input(root, value)
    runs_root = (root / "runs").resolve()
    if output != runs_root and runs_root not in output.parents:
        raise ValueError("Calibration output must be under the repository runs/ directory")
    return output


def _build_source(config, root: Path):
    source_config = config.source
    if source_config.type == "video":
        assert source_config.path is not None
        return VideoSource(
            _resolve_input(root, source_config.path),
            source_id=source_config.source_id,
        )
    if source_config.type == "image_sequence":
        assert source_config.path is not None and source_config.fps is not None
        return ImageSequenceSource(
            _resolve_input(root, source_config.path),
            fps=source_config.fps,
            frame_pattern=source_config.frame_pattern,
            source_id=source_config.source_id,
        )
    return WebcamSource(
        source_config.device_index,
        source_id=source_config.source_id,
        fps=source_config.fps,
    )


def _build_roi(config, root: Path, source):
    roi_config = config.roi
    if roi_config.path:
        return load_roi_file(
            _resolve_input(root, roi_config.path),
            source_width=source.metadata.width,
            source_height=source.metadata.height,
            clip=True,
        )
    if roi_config.coordinates is None:
        raise ROIError("Config must provide roi.coordinates or roi.path")
    return BBoxROI.from_coordinates(
        roi_config.coordinates,
        source_width=source.metadata.width,
        source_height=source.metadata.height,
        clip=True,
    )


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
        metadata_path = result.baseline.save(_resolve_output(root, str(args.output)))
        print(json.dumps({"baseline": str(metadata_path), **result.baseline.to_dict()}, indent=2))
        return 0
    except (ConfigValidationError, CalibrationError, ROIError, SourceError, OSError, ValueError) as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        if isinstance(exc, CalibrationError) and exc.diagnostics:
            print(json.dumps(exc.diagnostics, indent=2), file=sys.stderr)
        return 1
    finally:
        if source is not None and not source.closed:
            source.close()


if __name__ == "__main__":
    raise SystemExit(main())
