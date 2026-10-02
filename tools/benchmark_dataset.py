"""Evaluate M4/M5 event artifacts over one canonical dataset split.

Predictions are expected under ``<predictions-dir>/<video_id>/events.jsonl``
or as ``<predictions-dir>/<video_id>.jsonl``.  Missing prediction files are
treated as an empty prediction set so false negatives remain visible in the
report.  Test evaluation is opt-in and never the default.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.dataset.io import read_json, resolve_repo_path, write_json
from change_detection.domain import EventAnnotation, SampleAnnotation
from change_detection.evaluation import (
    EvaluationConfig,
    evaluate_events,
    summarize_timeliness,
)


def _metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _load_prediction_events(path: Path) -> list[EventAnnotation]:
    if not path.is_file():
        return []
    events: list[EventAnnotation] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"Prediction record at {path}:{line_number} must be an object")
            events.append(EventAnnotation.from_dict(value))
    return events


def _prediction_path(
    predictions_dir: Path,
    video_id: str,
    *,
    allow_root_file: bool = False,
) -> Path | None:
    candidates = [
        predictions_dir / video_id / "events.jsonl",
        predictions_dir / f"{video_id}.jsonl",
    ]
    if predictions_dir.name == video_id:
        candidates.append(predictions_dir / "events.jsonl")
    if allow_root_file:
        candidates.append(predictions_dir / "events.jsonl")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _sample_ground_truth(root: Path, sample: Mapping[str, Any]) -> SampleAnnotation:
    annotation_path = resolve_repo_path(root, str(sample["annotation_path"]))
    return SampleAnnotation.from_dict(read_json(annotation_path), manifest_sample=sample)


def _sum_counts(
    destination: dict[str, int],
    source: Mapping[str, Any],
) -> None:
    for field in ("tp", "fp", "fn"):
        destination[field] += int(source.get(field, 0))


def aggregate_results(
    sample_results: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    manifest: Path,
    dataset_version: str | None,
    split: str,
    missing_predictions: list[str],
    test_locked: bool | None = None,
    latency_deadlines_seconds: Iterable[float] = (3.0, 5.0, 10.0),
) -> dict[str, Any]:
    """Aggregate per-sample evaluator output without cross-video matching."""

    overall_counts = {"tp": 0, "fp": 0, "fn": 0}
    type_counts = {
        "FORGOTTEN_OBJECT": {"tp": 0, "fp": 0, "fn": 0},
        "MOVED_OBJECT": {"tp": 0, "fp": 0, "fn": 0},
    }
    outcome_counts = {
        "relocated": {"tp": 0, "fp": 0, "fn": 0},
        "left_scene": {"tp": 0, "fp": 0, "fn": 0},
    }
    errors: Counter[str] = Counter()
    matches: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    confirmation_latencies: list[float] = []
    start_delays: list[float] = []

    for sample_id, result in sample_results:
        _sum_counts(overall_counts, result["overall"])
        for event_type, counts in type_counts.items():
            _sum_counts(counts, result.get(event_type, {}))
        for outcome, counts in outcome_counts.items():
            _sum_counts(counts, result.get("MOVED_OBJECT_OUTCOMES", {}).get(outcome, {}))
        errors.update({str(key): int(value) for key, value in result.get("error_breakdown", {}).items()})
        for match in result.get("matches", []):
            matches.append({"sample_id": sample_id, **dict(match)})
            latency = match.get("confirmation_latency_sec", match.get("latency_sec"))
            if latency is not None:
                confirmation_latencies.append(float(latency))
            start_delay = match.get("start_delay_sec")
            if start_delay is not None:
                start_delays.append(float(start_delay))
        per_sample.append({"sample_id": sample_id, **dict(result)})

    ground_truth_count = overall_counts["tp"] + overall_counts["fn"]
    timeliness = summarize_timeliness(
        confirmation_latencies=confirmation_latencies,
        start_delays=start_delays,
        deadlines_seconds=latency_deadlines_seconds,
        ground_truth_count=ground_truth_count,
    )

    return {
        "dataset": {
            "manifest": str(manifest),
            "dataset_version": dataset_version,
            "split": split,
            "test_locked": test_locked,
            "sample_count": len(per_sample),
            "missing_prediction_count": len(missing_predictions),
            "missing_predictions": list(missing_predictions),
        },
        "overall": _metrics(**overall_counts),
        "FORGOTTEN_OBJECT": _metrics(**type_counts["FORGOTTEN_OBJECT"]),
        "MOVED_OBJECT": _metrics(**type_counts["MOVED_OBJECT"]),
        "MOVED_OBJECT_OUTCOMES": {
            outcome: _metrics(**counts) for outcome, counts in outcome_counts.items()
        },
        "matches": matches,
        "error_breakdown": dict(sorted(errors.items())),
        "timeliness": timeliness,
        "latency": {
            "matched_events": len(confirmation_latencies),
            "mean_confirmation_delta_sec": (
                sum(confirmation_latencies) / len(confirmation_latencies)
                if confirmation_latencies
                else None
            ),
        },
        "per_sample": per_sample,
    }


def aggregate_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the non-diagnostic result allowed for blind test reporting."""

    dataset = result.get("dataset", {})
    return {
        "dataset": {
            "dataset_version": dataset.get("dataset_version"),
            "split": dataset.get("split"),
            "test_locked": dataset.get("test_locked"),
            "sample_count": dataset.get("sample_count"),
            "missing_prediction_count": dataset.get("missing_prediction_count"),
        },
        "overall": dict(result.get("overall", {})),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "data/benchmark/v2/manifest.json",
    )
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="Explicitly permit evaluation of the locked/test split.",
    )
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
    parser.add_argument("--require-object-id", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Emit only aggregate dataset metadata and overall metrics.",
    )
    parser.add_argument(
        "--ignore-predictions-for",
        action="append",
        default=[],
        metavar="VIDEO_ID",
        help="Treat this sample's prediction as missing even if an artifact exists.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest_path = manifest_path.resolve()
    if args.split == "test" and not args.allow_test:
        raise ValueError("Test evaluation is opt-in; pass --allow-test explicitly")

    manifest = read_json(manifest_path)
    if args.split == "test" and not bool(manifest.get("test_locked", False)):
        raise ValueError("Test evaluation requires a manifest marked test_locked=true")
    samples = [
        item
        for item in manifest.get("samples", [])
        if str(item.get("split", "")) == args.split
    ]
    if not samples:
        raise ValueError(f"Manifest has no samples in split={args.split!r}")
    predictions_dir = args.predictions_dir
    if not predictions_dir.is_absolute():
        predictions_dir = root / predictions_dir
    predictions_dir = predictions_dir.resolve()
    evaluation_config = EvaluationConfig(
        event_boundary_tolerance_seconds=args.event_boundary_tolerance,
        start_tolerance_seconds=args.start_tolerance,
        min_temporal_overlap=args.min_overlap,
        forgotten_iou_threshold=args.forgotten_iou,
        moved_iou_threshold=args.moved_iou,
        require_prediction_object_id=args.require_object_id,
        latency_deadlines_seconds=tuple(args.latency_deadlines_seconds),
    )

    sample_results: list[tuple[str, Mapping[str, Any]]] = []
    missing_predictions: list[str] = []
    ignored_prediction_ids = set(args.ignore_predictions_for)
    single_sample = len(samples) == 1
    for sample_data in samples:
        sample_id = str(sample_data["video_id"])
        sample = _sample_ground_truth(root, sample_data)
        prediction_path = (
            None
            if sample_id in ignored_prediction_ids
            else _prediction_path(
                predictions_dir,
                sample_id,
                allow_root_file=single_sample,
            )
        )
        if prediction_path is None:
            missing_predictions.append(sample_id)
        predictions = _load_prediction_events(prediction_path) if prediction_path else []
        sample_results.append(
            (
                sample_id,
                evaluate_events(sample.events, predictions, config=evaluation_config),
            )
        )

    result = aggregate_results(
        sample_results,
        manifest=manifest_path,
        dataset_version=manifest.get("dataset_version"),
        split=args.split,
        missing_predictions=missing_predictions,
        test_locked=bool(manifest.get("test_locked", False)),
        latency_deadlines_seconds=evaluation_config.latency_deadlines_seconds,
    )
    result["evaluation"] = {
        "matching_policy": "confirmation_window",
        "event_boundary_tolerance_seconds": (
            evaluation_config.event_boundary_tolerance_seconds
        ),
        "forgotten_iou_threshold": evaluation_config.forgotten_iou_threshold,
        "moved_iou_threshold": evaluation_config.moved_iou_threshold,
        "require_prediction_object_id": evaluation_config.require_prediction_object_id,
        "latency_deadlines_seconds": list(evaluation_config.latency_deadlines_seconds),
    }
    result["dataset"]["ignored_prediction_ids"] = sorted(ignored_prediction_ids)
    if args.summary_only or args.split == "test":
        result = aggregate_summary(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        write_json(output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
