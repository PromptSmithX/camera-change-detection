"""Create a non-destructive canonical benchmark draft from data/processed.

The migration deliberately does not invent confirmation times or moved-object
destinations.  Those fields remain null and the affected records stay
provisional/excluded until reviewed in the annotation tool.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.dataset.io import read_json, sha256_file, write_json


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _camera_id(sample: dict[str, Any]) -> str:
    group_id = sample.get("group_id")
    if group_id:
        return str(group_id)
    dataset = str(sample.get("dataset", "unknown"))
    source_video_id = str(sample.get("source_video_id", sample.get("id", "unknown")))
    return f"{dataset}:{source_video_id}"


def _scenario_tags(sample: dict[str, Any]) -> list[str]:
    events = sample.get("events", [])
    tags = {str(event.get("type", "")).lower() for event in events if event.get("type")}
    if not tags:
        tags.add("no_event")
    tags.add(f"media_{sample.get('media_type', 'unknown')}")
    return sorted(tags)


def _sample_status(sample: dict[str, Any], excluded_ids: set[str]) -> str:
    return "excluded" if sample.get("id") in excluded_ids else "provisional"


def _canonical_event(event: dict[str, Any], *, sample: dict[str, Any], index: int, status: str) -> dict[str, Any]:
    fps = float(sample["fps"])
    start_frame = int(event["start_frame"])
    end_frame = int(event["end_frame"])
    object_data = event.get("object") or {}
    object_id = f"{sample['id']}:obj_{index + 1:03d}"
    event_type = str(event["type"])
    bbox = object_data.get("bbox")
    baseline_bbox = object_data.get("old_bbox")
    return {
        "event_id": str(event["id"]),
        "type": event_type,
        "object_id": object_id,
        "start_time_sec": start_frame / fps,
        "confirmation_time_sec": None,
        "end_time_sec": end_frame / fps,
        "bbox": bbox,
        "baseline_bbox": baseline_bbox if event_type == "MOVED_OBJECT" else None,
        "new_bbox": None,
        "object_class": object_data.get("class"),
        "start_frame": start_frame,
        "confirmation_frame": None,
        "end_frame": end_frame,
        "occlusion_intervals": [],
        "lighting_change_intervals": [],
        "notes": "Migrated from data/processed; confirmation and identity evidence require review.",
        "difficult": False,
        "ambiguous": bool(status == "excluded"),
        "quality_status": status,
    }


def migrate(
    *,
    manifest_path: Path,
    version_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_manifest = read_json(manifest_path)
    source_version = read_json(version_path)
    disabled_ids = {str(value) for value in source_version.get("disabled_sample_ids", [])}
    excluded_ids = {"meva_meva_bag_forgotten", "meva_meva_bag_moved"}
    output_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir = output_dir / "annotations"
    roi_dir = output_dir / "roi"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    source_annotation_by_id: dict[str, dict[str, Any]] = {}
    for path in (REPO_ROOT / "data" / "processed" / "annotations").rglob("*.json"):
        data = read_json(path)
        source_annotation_by_id[str(data["id"])] = data

    samples: list[dict[str, Any]] = []
    migration_issues: list[dict[str, Any]] = []
    for source_sample in source_manifest.get("samples", []):
        sample_id = str(source_sample["id"])
        if sample_id in disabled_ids:
            continue
        source_annotation = source_annotation_by_id.get(sample_id)
        if source_annotation is None:
            migration_issues.append({"code": "missing_source_annotation", "sample_id": sample_id})
            continue
        status = _sample_status(source_sample, excluded_ids)
        camera_id = _camera_id(source_sample)
        annotation_rel = f"data/benchmark/v1/annotations/{sample_id}.json"
        roi_rel = f"data/benchmark/v1/roi/{sample_id}.json"
        frame_count = int(source_sample["num_frames"])
        fps = float(source_sample["fps"])
        canonical_events = [
            _canonical_event(event, sample=source_sample, index=index, status=status)
            for index, event in enumerate(source_sample.get("events", []))
        ]
        roi_data = source_sample.get("roi", {}).get("bbox")
        reference_range = source_sample.get("reference", {}).get("frame_range")
        canonical_annotation = {
            "video_id": sample_id,
            "camera_id": camera_id,
            "path": source_sample["video_path"],
            "annotation_path": annotation_rel,
            "roi_path": roi_rel,
            "split": source_sample["split"],
            "scenario_tags": _scenario_tags(source_sample),
            "fps": fps,
            "width": int(source_sample["width"]),
            "height": int(source_sample["height"]),
            "frame_count": frame_count,
            "duration_sec": frame_count / fps,
            "media_type": source_sample.get("media_type", "video"),
            "events": canonical_events,
            "roi": {"bbox": roi_data},
            "reference": {"frame_range": reference_range},
            "quality_status": status,
            "reviewer": None,
            "review_notes": "Imported as provisional; human review required before official benchmark.",
            "group_id": source_sample.get("group_id"),
            "warnings": list(source_sample.get("warnings", [])),
            "provenance": {
                "migration": "data/processed -> data/benchmark/v1",
                "source_annotation": source_sample.get("annotation_path"),
                "source_human_verified": source_sample.get("human_verified"),
                "source_provenance": source_sample.get("provenance", {}),
            },
            "source_video_id": source_sample.get("source_video_id"),
            "frame_pattern": source_sample.get("frame_pattern"),
        }
        annotation_path = REPO_ROOT / annotation_rel
        roi_path = REPO_ROOT / roi_rel
        write_json(annotation_path, canonical_annotation)
        write_json(
            roi_path,
            {
                "video_id": sample_id,
                "camera_id": camera_id,
                "bbox": roi_data,
                "width": int(source_sample["width"]),
                "height": int(source_sample["height"]),
                "coordinate_system": "pixel",
                "bbox_semantics": "integer_half_open",
            },
        )
        samples.append(
            {
                "video_id": sample_id,
                "camera_id": camera_id,
                "path": source_sample["video_path"],
                "annotation_path": annotation_rel,
                "roi_path": roi_rel,
                "split": source_sample["split"],
                "scenario_tags": _scenario_tags(source_sample),
                "fps": fps,
                "width": int(source_sample["width"]),
                "height": int(source_sample["height"]),
                "frame_count": frame_count,
                "duration_sec": frame_count / fps,
                "media_type": source_sample.get("media_type", "video"),
                "quality_status": status,
                "reviewer": None,
                "group_id": source_sample.get("group_id"),
                "warnings": list(source_sample.get("warnings", [])),
                "annotation_sha256": sha256_file(annotation_path),
            }
        )

    orphan_ids = sorted(set(source_annotation_by_id) - {sample["video_id"] for sample in samples})
    for orphan_id in orphan_ids:
        migration_issues.append({"code": "annotation_not_in_active_manifest", "sample_id": orphan_id})

    canonical_manifest = {
        "dataset_version": "1.0.0-draft",
        "annotation_schema_version": "1.0",
        "frame_indexing": "zero_based",
        "range_semantics": "inclusive",
        "bbox_semantics": "integer_half_open",
        "test_locked": False,
        "samples": samples,
    }
    canonical_manifest_path = output_dir / "manifest.json"
    write_json(canonical_manifest_path, canonical_manifest)
    manifest_hash = sha256_file(canonical_manifest_path)
    split_counts = {split: sum(1 for sample in samples if sample["split"] == split) for split in ("validation", "test")}
    status_counts = {status: sum(1 for sample in samples if sample["quality_status"] == status) for status in ("provisional", "verified", "excluded")}
    event_counts = {event_type: 0 for event_type in ("FORGOTTEN_OBJECT", "MOVED_OBJECT")}
    for sample in samples:
        annotation = read_json(REPO_ROOT / sample["annotation_path"])
        for event in annotation["events"]:
            event_counts[event["type"]] += 1
    dataset_version = {
        "schema_version": 1,
        "dataset_version": canonical_manifest["dataset_version"],
        "annotation_schema_version": canonical_manifest["annotation_schema_version"],
        "manifest": _rel(canonical_manifest_path),
        "manifest_sha256": manifest_hash,
        "active_samples": len(samples),
        "annotations": [
            {
                "sample_id": sample["video_id"],
                "path": sample["annotation_path"],
                "sha256": sample["annotation_sha256"],
                "quality_status": sample["quality_status"],
            }
            for sample in samples
        ],
        "benchmark": {
            split: {
                "samples": split_counts[split],
                "events": sum(
                    len(read_json(REPO_ROOT / sample["annotation_path"])["events"])
                    for sample in samples
                    if sample["split"] == split
                ),
                "eligible_official_samples": sum(
                    1
                    for sample in samples
                    if sample["split"] == split and sample["quality_status"] == "verified"
                ),
                "excluded_sample_ids": [
                    sample["video_id"]
                    for sample in samples
                    if sample["split"] == split and sample["quality_status"] == "excluded"
                ],
            }
            for split in ("validation", "test")
        },
        "disabled_sample_ids": sorted(disabled_ids),
        "excluded_sample_ids": sorted(excluded_ids),
        "migration_issues": migration_issues,
    }
    write_json(output_dir / "dataset_version.json", dataset_version)
    write_json(
        output_dir / "migration_report.json",
        {
            "source_manifest": _rel(manifest_path),
            "source_version": _rel(version_path),
            "output_manifest": _rel(canonical_manifest_path),
            "sample_count": len(samples),
            "split_counts": split_counts,
            "status_counts": status_counts,
            "event_counts": event_counts,
            "excluded_sample_ids": sorted(excluded_ids),
            "migration_issues": migration_issues,
        },
    )
    return dataset_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/processed/manifest.json")
    parser.add_argument("--version", type=Path, default=REPO_ROOT / "data/processed/dataset_version.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data/benchmark/v1")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    migrate(
        manifest_path=args.manifest.resolve(),
        version_path=args.version.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(f"Created draft benchmark dataset at {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
