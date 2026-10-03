"""Run event-level evaluation on canonical JSON records.

The input files may be arrays of event objects, objects containing an
``events``/``predictions`` array, or runtime JSONL event artifacts. A root
``video_id`` is copied into each event when present.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.dataset.io import write_json
from change_detection.evaluation.evaluator import EvaluationConfig, evaluate_events


def _load_events(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid event JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Event record at {path}:{line_number} must be an object")
            records.append(record)
        return records
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Runtime event artifacts are JSONL while the original evaluator
        # accepted one JSON array/object.  Support both contracts here so a
        # single-run smoke output can be evaluated directly.
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid event JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Event record at {path}:{line_number} must be an object")
            records.append(record)
        return records
    if isinstance(value, list):
        return [dict(item) for item in value]
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object or event array")
    if isinstance(value.get("events"), list):
        return [dict(item) for item in value["events"]]
    if isinstance(value.get("predictions"), list):
        return [dict(item) for item in value["predictions"]]
    raise ValueError(f"{path} must contain an events or predictions array")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--event-boundary-tolerance", type=float, default=1.0)
    parser.add_argument("--forgotten-iou", type=float, default=0.3)
    parser.add_argument("--moved-iou", type=float, default=0.3)
    parser.add_argument(
        "--latency-deadlines-seconds",
        type=float,
        nargs="+",
        default=(3.0, 5.0, 10.0),
    )
    parser.add_argument("--start-tolerance", type=float, default=3.0, help=argparse.SUPPRESS)
    parser.add_argument("--min-overlap", type=float, default=0.1, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ground_truth = _load_events(args.ground_truth)
    predictions = _load_events(args.predictions)
    result = evaluate_events(
        ground_truth,
        predictions,
        config=EvaluationConfig(
            event_boundary_tolerance_seconds=args.event_boundary_tolerance,
            start_tolerance_seconds=args.start_tolerance,
            min_temporal_overlap=args.min_overlap,
            forgotten_iou_threshold=args.forgotten_iou,
            moved_iou_threshold=args.moved_iou,
            latency_deadlines_seconds=tuple(args.latency_deadlines_seconds),
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        write_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
