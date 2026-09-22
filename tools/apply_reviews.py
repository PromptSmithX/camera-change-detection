"""Materialize local review drafts as a new canonical dataset version.

The command copies annotations/ROI into a new output directory, updates
manifest paths and hashes, and leaves both ``data/processed`` and the source
benchmark draft untouched.  Unreviewed samples remain in their prior quality
state, so the resulting version cannot accidentally become official.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.dataset.io import read_json, resolve_repo_path, sha256_file, write_json
from change_detection.domain import SampleAnnotation


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _safe_output_dir(root: Path, output_dir: Path) -> Path:
    output = output_dir.resolve()
    root = root.resolve()
    if output == root or root not in output.parents:
        raise ValueError("Output dataset must be a child directory of repository root")
    return output


def _canonical_payload(payload: dict[str, Any], sample: dict[str, Any], annotation_path: str, roi_path: str) -> dict[str, Any]:
    value = json.loads(json.dumps(payload))
    value["video_id"] = sample["video_id"]
    value["camera_id"] = sample["camera_id"]
    value["path"] = sample["path"]
    value["annotation_path"] = annotation_path
    value["roi_path"] = roi_path
    value["split"] = sample["split"]
    value["scenario_tags"] = list(payload.get("scenario_tags", sample.get("scenario_tags", [])))
    value["fps"] = sample["fps"]
    value["width"] = sample["width"]
    value["height"] = sample["height"]
    value["frame_count"] = sample["frame_count"]
    value["duration_sec"] = sample["duration_sec"]
    value["media_type"] = sample.get("media_type", value.get("media_type", "video"))
    value["frame_pattern"] = sample.get("frame_pattern", value.get("frame_pattern"))
    return value


def apply_reviews(
    manifest_path: Path,
    *,
    repo_root: Path,
    reviews_dir: Path,
    output_dir: Path,
    dataset_version: str,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    output_dir = _safe_output_dir(repo_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)
    (output_dir / "roi").mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    reviewed_ids: list[str] = []
    missing_review_ids: list[str] = []
    excluded_ids: list[str] = []
    status_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()

    for source_sample in manifest.get("samples", []):
        sample = json.loads(json.dumps(source_sample))
        video_id = str(sample["video_id"])
        source_annotation_path = resolve_repo_path(repo_root, str(source_sample["annotation_path"]))
        review_path = reviews_dir / f"{video_id}.json"
        payload = read_json(review_path) if review_path.exists() else read_json(source_annotation_path)
        if review_path.exists():
            reviewed_ids.append(video_id)
        else:
            # Skip excluded samples that have no review override.
            if source_sample.get("quality_status") == "excluded":
                excluded_ids.append(video_id)
                continue
            missing_review_ids.append(video_id)

        destination_annotation = output_dir / "annotations" / f"{video_id}.json"
        destination_roi = output_dir / "roi" / f"{video_id}.json"
        annotation_rel = _relative(destination_annotation, repo_root)
        roi_rel = _relative(destination_roi, repo_root)
        canonical = _canonical_payload(payload, source_sample, annotation_rel, roi_rel)
        # Parse before writing so malformed review drafts fail without
        # producing a misleading dataset manifest.
        parsed = SampleAnnotation.from_dict(canonical, manifest_sample=sample)
        canonical["quality_status"] = parsed.quality_status.value
        canonical["reviewer"] = parsed.reviewer
        canonical["review_notes"] = parsed.review_notes
        write_json(destination_annotation, canonical)

        if parsed.roi_bbox is None:
            raise ValueError(f"Review has no ROI bbox for {video_id}")
        write_json(
            destination_roi,
            {
                "video_id": video_id,
                "camera_id": parsed.camera_id,
                "bbox": parsed.roi_bbox.to_list(),
                "width": parsed.width,
                "height": parsed.height,
                "coordinate_system": "pixel",
                "bbox_semantics": "integer_half_open",
            },
        )

        sample["annotation_path"] = annotation_rel
        sample["roi_path"] = roi_rel
        sample["annotation_sha256"] = sha256_file(destination_annotation)
        sample["quality_status"] = parsed.quality_status.value
        sample["reviewer"] = parsed.reviewer
        sample["review_notes"] = parsed.review_notes
        sample["warnings"] = list(parsed.warnings)
        sample["scenario_tags"] = list(parsed.scenario_tags)
        samples.append(sample)
        status_counts[parsed.quality_status.value] += 1
        for event in parsed.events:
            event_counts[event.event_type.value] += 1

    output_manifest = output_dir / "manifest.json"
    canonical_manifest = {
        "dataset_version": dataset_version,
        "annotation_schema_version": manifest.get("annotation_schema_version", "1.0"),
        "frame_indexing": manifest.get("frame_indexing", "zero_based"),
        "range_semantics": manifest.get("range_semantics", "inclusive"),
        "bbox_semantics": manifest.get("bbox_semantics", "integer_half_open"),
        "test_locked": False,
        "samples": samples,
    }
    write_json(output_manifest, canonical_manifest)
    manifest_hash = sha256_file(output_manifest)
    result = {
        "source_manifest": _relative(manifest_path, repo_root),
        "output_manifest": _relative(output_manifest, repo_root),
        "dataset_version": dataset_version,
        "sample_count": len(samples),
        "reviewed_sample_count": len(reviewed_ids),
        "reviewed_sample_ids": sorted(reviewed_ids),
        "unreviewed_sample_ids": sorted(missing_review_ids),
        "excluded_sample_ids": sorted(excluded_ids),
        "status_counts": dict(sorted(status_counts.items())),
        "event_counts": dict(sorted(event_counts.items())),
        "manifest_sha256": manifest_hash,
    }
    write_json(output_dir / "apply_reviews_report.json", result)
    write_json(
        output_dir / "dataset_version.json",
        {
            "schema_version": 1,
            "dataset_version": dataset_version,
            "annotation_schema_version": canonical_manifest["annotation_schema_version"],
            "manifest": _relative(output_manifest, repo_root),
            "manifest_sha256": manifest_hash,
            "active_samples": len(samples),
            "reviewed_samples": len(reviewed_ids),
            "unreviewed_samples": len(missing_review_ids),
            "excluded_samples": len(excluded_ids),
            "excluded_sample_ids": sorted(excluded_ids),
            "status_counts": dict(sorted(status_counts.items())),
            "event_counts": dict(sorted(event_counts.items())),
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/benchmark/v1/manifest.json")
    parser.add_argument("--reviews-dir", type=Path, default=REPO_ROOT / "data/benchmark/v1/reviews")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--dataset-version", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    reviews = args.reviews_dir if args.reviews_dir.is_absolute() else root / args.reviews_dir
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    try:
        result = apply_reviews(
            manifest.resolve(),
            repo_root=root,
            reviews_dir=reviews.resolve(),
            output_dir=output,
            dataset_version=args.dataset_version,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Cannot apply reviews: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
