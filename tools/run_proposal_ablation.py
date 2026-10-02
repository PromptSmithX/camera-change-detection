"""Run validation-only proposal-source ablations and summarize duplication.

This command intentionally has no split option.  It delegates each mode to
``run_validation.py --split validation`` and reads validation artifacts only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.dataset.io import read_json, write_json
from change_detection.domain import BBox


MODES: tuple[tuple[str, bool, bool], ...] = (
    ("full_current", True, True),
    ("without_standalone_reference_change", False, True),
    ("without_baseline_residual", True, False),
    ("yolo_change_corroboration_only", False, False),
)


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path must be inside repository root: {path}") from exc
    return resolved


def _mode_config(
    base: Mapping[str, Any], *, emit_standalone: bool, baseline_residual: bool
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    perception = config.setdefault("perception", {})
    perception["proposal_fusion"] = {"enabled": True}
    reference = perception.setdefault("reference_change", {})
    reference["emit_standalone"] = emit_standalone
    residual = reference.setdefault("baseline_residual", {})
    residual["enabled"] = baseline_residual
    return config


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _bbox(event: Mapping[str, Any]) -> BBox | None:
    for key in ("after_bbox", "new_bbox", "bbox", "before_bbox", "baseline_bbox"):
        try:
            value = BBox.from_value(event.get(key))
        except (TypeError, ValueError):
            value = None
        if value is not None:
            return value
    return None


def _overlap_over_smaller(left: BBox, right: BBox) -> float:
    x1, y1 = max(left.x1, right.x1), max(left.y1, right.y1)
    x2, y2 = min(left.x2, right.x2), min(left.y2, right.y2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection / max(1, min(left.area, right.area))


def _same_region(left: BBox, right: BBox) -> bool:
    return _overlap_over_smaller(left, right) >= 0.25 and left.iou(right) >= 0.10


def _temporal_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_start = float(left.get("started_at_sec", 0.0))
    right_start = float(right.get("started_at_sec", 0.0))
    left_end = float(left.get("ended_at_sec") or left.get("confirmed_at_sec") or left_start)
    right_end = float(right.get("ended_at_sec") or right.get("confirmed_at_sec") or right_start)
    return max(left_start, right_start) <= min(left_end, right_end)


def _event_overlap_count(events_by_sample: Mapping[str, list[dict[str, Any]]]) -> int:
    count = 0
    for events in events_by_sample.values():
        for left, right in combinations(events, 2):
            left_bbox, right_bbox = _bbox(left), _bbox(right)
            if (
                left_bbox is not None
                and right_bbox is not None
                and _temporal_overlap(left, right)
                and _same_region(left_bbox, right_bbox)
            ):
                count += 1
    return count


def _memory_duplicate_pairs(predictions_dir: Path) -> tuple[int, int]:
    unique_pairs: set[tuple[str, str, str]] = set()
    maximum_in_frame = 0
    for identity_path in sorted(predictions_dir.glob("*/identity.jsonl")):
        sample_id = identity_path.parent.name
        for line in identity_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            objects = json.loads(line).get("memory_objects", [])
            frame_pairs = 0
            for left, right in combinations(objects, 2):
                if bool(left.get("is_baseline")) and bool(right.get("is_baseline")):
                    continue
                left_bbox = BBox.from_value(left.get("last_bbox"))
                right_bbox = BBox.from_value(right.get("last_bbox"))
                if left_bbox is None or right_bbox is None or not _same_region(left_bbox, right_bbox):
                    continue
                pair = tuple(sorted((str(left.get("object_id")), str(right.get("object_id")))))
                unique_pairs.add((sample_id, pair[0], pair[1]))
                frame_pairs += 1
            maximum_in_frame = max(maximum_in_frame, frame_pairs)
    return len(unique_pairs), maximum_in_frame


def _source_key(event: Mapping[str, Any]) -> str:
    sources = tuple(str(item) for item in event.get("evidence_sources", []) if str(item))
    return "+".join(sources) if sources else str(event.get("proposal_source") or "unattributed")


def _source_metrics(
    events_by_sample: Mapping[str, list[dict[str, Any]]],
    evaluation: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    matched = {
        (str(item.get("sample_id")), str(item.get("prediction_event_id")))
        for item in evaluation.get("matches", [])
    }
    counts: dict[str, dict[str, int]] = {}
    for sample_id, events in events_by_sample.items():
        for event in events:
            key = _source_key(event)
            bucket = counts.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
            result = "tp" if (sample_id, str(event.get("event_id"))) in matched else "fp"
            bucket[result] += 1
    overall_fn = int((evaluation.get("overall") or {}).get("fn", 0))
    if overall_fn:
        counts["unattributed_missed_ground_truth"] = {"tp": 0, "fp": 0, "fn": overall_fn}
    return dict(sorted(counts.items()))


def _analyze(run_dir: Path) -> dict[str, Any]:
    evaluation = read_json(run_dir / "metrics" / "evaluation.json")
    predictions_dir = run_dir / "predictions"
    events_by_sample = {
        path.parent.name: _events(path)
        for path in sorted(predictions_dir.glob("*/events.jsonl"))
    }
    duplicate_pairs, max_frame_pairs = _memory_duplicate_pairs(predictions_dir)
    return {
        "overall": evaluation.get("overall", {}),
        "per_sample": {
            str(item.get("sample_id")): item.get("overall", {})
            for item in evaluation.get("per_sample", [])
        },
        "by_proposal_evidence": _source_metrics(events_by_sample, evaluation),
        "spatiotemporal_overlapping_event_pairs": _event_overlap_count(events_by_sample),
        "memory_object_same_region_pairs": duplicate_pairs,
        "max_memory_object_same_region_pairs_in_frame": max_frame_pairs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/m45.example.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark/v2/manifest.json"))
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    config_path = _inside(args.config if args.config.is_absolute() else root / args.config, root)
    manifest_path = _inside(args.manifest if args.manifest.is_absolute() else root / args.manifest, root)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("runs/validation") / f"proposal_ablation_{timestamp}"
    output = _inside(output if output.is_absolute() else root / output, root)
    validation_root = (root / "runs" / "validation").resolve()
    if output != validation_root and validation_root not in output.parents:
        raise ValueError("Proposal ablation output must be under runs/validation/")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    (output / "configs").mkdir(parents=True)

    base = read_json(config_path)
    report: dict[str, Any] = {
        "split": "validation",
        "base_config": str(config_path.relative_to(root)),
        "modes": {},
    }
    for mode, emit_standalone, baseline_residual in MODES:
        derived_path = output / "configs" / f"{mode}.json"
        write_json(
            derived_path,
            _mode_config(
                base,
                emit_standalone=emit_standalone,
                baseline_residual=baseline_residual,
            ),
        )
        run_dir = output / mode
        command = [
            sys.executable,
            str(root / "tools" / "run_validation.py"),
            "--root",
            str(root),
            "--config",
            str(derived_path.relative_to(root)),
            "--manifest",
            str(manifest_path.relative_to(root)),
            "--split",
            "validation",
            "--output",
            str(run_dir.relative_to(root)),
        ]
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            report["modes"][mode] = {"status": "failed", "exit_code": completed.returncode}
            write_json(output / "ablation_report.json", report)
            continue
        report["modes"][mode] = {"status": "completed", **_analyze(run_dir)}
        write_json(output / "ablation_report.json", report)

    full = report["modes"].get("full_current", {})
    overall = full.get("overall", {})
    watched = full.get("per_sample", {})

    def watched_passes(suffix: str) -> bool:
        metrics = next(
            (value for sample_id, value in watched.items() if sample_id.endswith(suffix)),
            {},
        )
        return (
            int(metrics.get("tp", 0)) >= 1
            and int(metrics.get("fp", 0)) == 0
            and int(metrics.get("fn", 0)) == 0
        )

    report["acceptance"] = {
        "true_positives_at_least_13": int(overall.get("tp", 0)) >= 13,
        "recall_at_least_0_75": float(overall.get("recall", 0.0)) >= 0.75,
        "false_positives_at_most_10": int(overall.get("fp", 10**9)) <= 10,
        "153001_is_tp_without_fp": watched_passes("153001"),
        "153259_is_tp_without_fp": watched_passes("153259"),
    }
    report["acceptance"]["passed"] = all(report["acceptance"].values())
    write_json(output / "ablation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item.get("status") == "completed" for item in report["modes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
