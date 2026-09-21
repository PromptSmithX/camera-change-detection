"""Create a read-only test manifest after the official review gate passes.

The command is deliberately fail-closed.  It will not create a locked
manifest while any sample/event is provisional, excluded, structurally
incomplete, or still carrying validator warnings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from change_detection.dataset.io import read_json, sha256_file, write_json
from change_detection.dataset.validation import validate_manifest
from change_detection.domain import AnnotationStatus, Split, SampleAnnotation


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def lock_test(
    manifest_path: Path,
    *,
    repo_root: Path,
    output_manifest: Path,
    output_lock: Path,
    dataset_version: str,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if manifest.get("test_locked"):
        raise ValueError("Input manifest is already marked test_locked")
    if not dataset_version.strip() or dataset_version == manifest.get("dataset_version"):
        raise ValueError("Locking requires a new dataset_version different from the input manifest")

    report = validate_manifest(
        manifest_path,
        repo_root=repo_root,
        mode="official",
        strict_hashes=True,
    )
    if report.errors:
        raise ValueError(
            f"Official validation failed with {len(report.errors)} error(s); review must finish before locking"
        )
    if report.warnings:
        raise ValueError(
            f"Official validation still has {len(report.warnings)} warning(s); resolve warnings before locking"
        )

    test_samples = [item for item in manifest.get("samples", []) if item.get("split") == Split.TEST.value]
    if not test_samples:
        raise ValueError("Cannot lock an empty test split")
    for sample_data in test_samples:
        if sample_data.get("quality_status") != AnnotationStatus.VERIFIED.value:
            raise ValueError(f"Test sample is not verified: {sample_data.get('video_id')}")
        annotation_path = repo_root / str(sample_data["annotation_path"])
        annotation = read_json(annotation_path)
        parsed = SampleAnnotation.from_dict(annotation, manifest_sample=sample_data)
        for event in parsed.events:
            if event.quality_status != AnnotationStatus.VERIFIED:
                raise ValueError(f"Test event is not verified: {sample_data.get('video_id')}:{event.event_id}")

    test_records = [
        {
            "video_id": item["video_id"],
            "annotation_path": item["annotation_path"],
            "annotation_sha256": item["annotation_sha256"],
        }
        for item in sorted(test_samples, key=lambda item: str(item["video_id"]))
    ]
    locked_manifest = json.loads(json.dumps(manifest))
    locked_manifest["test_locked"] = True
    locked_manifest["dataset_version"] = dataset_version
    locked_manifest["locked_test_sha256"] = _stable_hash(test_records)
    write_json(output_manifest, locked_manifest)
    manifest_hash = sha256_file(output_manifest)
    lock_record = {
        "locked": True,
        "dataset_version": dataset_version,
        "annotation_schema_version": locked_manifest.get("annotation_schema_version"),
        "manifest": str(output_manifest),
        "manifest_sha256": manifest_hash,
        "test_sample_count": len(test_records),
        "test_sample_ids": [item["video_id"] for item in test_records],
        "test_sha256": locked_manifest["locked_test_sha256"],
    }
    write_json(output_lock, lock_record)
    return lock_record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/benchmark/v1/manifest.json")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-manifest", type=Path, default=REPO_ROOT / "data/benchmark/v1/manifest.locked.json")
    parser.add_argument("--output-lock", type=Path, default=REPO_ROOT / "data/benchmark/v1/locked_test.json")
    parser.add_argument("--dataset-version", required=True, help="New version identifier for the immutable locked manifest")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_manifest = args.output_manifest if args.output_manifest.is_absolute() else root / args.output_manifest
    output_lock = args.output_lock if args.output_lock.is_absolute() else root / args.output_lock
    try:
        result = lock_test(
            manifest.resolve(),
            repo_root=root,
            output_manifest=output_manifest.resolve(),
            output_lock=output_lock.resolve(),
            dataset_version=args.dataset_version,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"Cannot lock test split: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
