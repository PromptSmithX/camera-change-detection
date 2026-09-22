"""Create a compact, reproducible QA report for a canonical benchmark manifest.

The report is intentionally useful before review is complete: it records what
is present, what is unresolved, how the split is distributed, and the validator
result.  It does not turn provisional annotations into official benchmark
records.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.dataset.io import read_json, resolve_repo_path, write_json
from change_detection.dataset.validation import validate_manifest
from change_detection.domain import EventType, MovementOutcome, SampleAnnotation


def _sorted_counts(values: Counter[str]) -> dict[str, int]:
    return {key: values[key] for key in sorted(values)}


def build_report(manifest_path: Path, *, repo_root: Path, strict_hashes: bool = True) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    samples = manifest.get("samples", [])
    split_counts: Counter[str] = Counter()
    sample_status_counts: Counter[str] = Counter()
    event_status_counts: Counter[str] = Counter()
    event_type_counts: Counter[str] = Counter()
    scenario_tag_counts: Counter[str] = Counter()
    media_type_counts: Counter[str] = Counter()
    warning_codes: Counter[str] = Counter()
    warning_samples: set[str] = set()
    no_event_samples: list[str] = []
    missing_confirmation: list[dict[str, str]] = []
    incomplete_moved: list[dict[str, str]] = []
    parse_errors: list[dict[str, str]] = []
    sample_ids_by_split: dict[str, list[str]] = {}
    event_count = 0

    for sample_data in samples:
        sample_id = str(sample_data.get("video_id", ""))
        split = str(sample_data.get("split", ""))
        status = str(sample_data.get("quality_status", "provisional"))
        split_counts[split] += 1
        sample_status_counts[status] += 1
        sample_ids_by_split.setdefault(split, []).append(sample_id)
        media_type_counts[str(sample_data.get("media_type", "video"))] += 1
        for tag in sample_data.get("scenario_tags", []):
            scenario_tag_counts[str(tag)] += 1
        try:
            annotation_path = resolve_repo_path(repo_root, str(sample_data["annotation_path"]))
            annotation = read_json(annotation_path)
            sample = SampleAnnotation.from_dict(annotation, manifest_sample=sample_data)
        except Exception as exc:
            parse_errors.append({"sample_id": sample_id, "message": str(exc)})
            continue

        if not sample.events:
            no_event_samples.append(sample_id)
        if sample.warnings:
            warning_samples.add(sample_id)
        for event in sample.events:
            event_count += 1
            event_type_counts[event.event_type.value] += 1
            event_status_counts[event.quality_status.value] += 1
            if event.confirmation_time_sec is None:
                missing_confirmation.append({"sample_id": sample_id, "event_id": event.event_id})
            if event.event_type == EventType.MOVED_OBJECT:
                missing_evidence = (
                    event.baseline_bbox is None
                    or event.movement_outcome is None
                    or (
                        event.movement_outcome == MovementOutcome.RELOCATED
                        and event.new_bbox is None
                    )
                    or (
                        event.movement_outcome == MovementOutcome.LEFT_SCENE
                        and event.new_bbox is not None
                    )
                )
                if missing_evidence:
                    incomplete_moved.append({"sample_id": sample_id, "event_id": event.event_id})

    validation = validate_manifest(
        manifest_path,
        repo_root=repo_root,
        mode="draft",
        strict_hashes=strict_hashes,
    )
    for issue in validation.warnings:
        warning_codes[issue.code] += 1
        if issue.sample_id:
            warning_samples.add(issue.sample_id)

    for values in sample_ids_by_split.values():
        values.sort()
    no_event_samples.sort()
    missing_confirmation.sort(key=lambda item: (item["sample_id"], item["event_id"]))
    incomplete_moved.sort(key=lambda item: (item["sample_id"], item["event_id"]))
    parse_errors.sort(key=lambda item: item["sample_id"])

    return {
        "dataset_version": manifest.get("dataset_version"),
        "annotation_schema_version": manifest.get("annotation_schema_version"),
        "manifest": str(manifest_path),
        "sample_count": len(samples),
        "event_count": event_count,
        "split_counts": _sorted_counts(split_counts),
        "sample_status_counts": _sorted_counts(sample_status_counts),
        "event_status_counts": _sorted_counts(event_status_counts),
        "event_type_counts": _sorted_counts(event_type_counts),
        "scenario_tag_counts": _sorted_counts(scenario_tag_counts),
        "media_type_counts": _sorted_counts(media_type_counts),
        "sample_ids_by_split": sample_ids_by_split,
        "no_event_samples": no_event_samples,
        "excluded_sample_ids": sorted(
            str(item.get("video_id"))
            for item in samples
            if item.get("quality_status") == "excluded"
        ),
        "review_gaps": {
            "missing_confirmation_count": len(missing_confirmation),
            "missing_confirmation": missing_confirmation,
            "incomplete_moved_count": len(incomplete_moved),
            "incomplete_moved": incomplete_moved,
        },
        "warnings": {
            "sample_count_with_warnings": len(warning_samples),
            "by_code": _sorted_counts(warning_codes),
        },
        "parse_errors": parse_errors,
        "validation": validation.to_dict(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/benchmark/v1/manifest.json")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/benchmark/v1/qa_report.json")
    parser.add_argument("--no-strict-hashes", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output = args.output if args.output.is_absolute() else root / args.output
    report = build_report(manifest_path.resolve(), repo_root=root, strict_hashes=not args.no_strict_hashes)
    write_json(output.resolve(), report)
    print(f"Wrote QA report to {output.resolve()}")
    return 0 if report["validation"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
