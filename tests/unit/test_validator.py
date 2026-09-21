import json
from pathlib import Path

from change_detection.dataset.validation import validate_manifest


def write_dataset(root: Path, *, moved_complete: bool = True) -> Path:
    (root / "media.txt").write_text("media", encoding="utf-8")
    annotation_path = root / "annotations" / "vid_1.json"
    roi_path = root / "roi" / "vid_1.json"
    annotation_path.parent.mkdir(parents=True)
    roi_path.parent.mkdir(parents=True)
    event = {
        "event_id": "evt-1",
        "type": "MOVED_OBJECT",
        "object_id": "obj-1",
        "start_time_sec": 1.0,
        "confirmation_time_sec": 2.0,
        "end_time_sec": 3.0,
        "bbox": None,
        "baseline_bbox": [10, 10, 30, 30],
        "new_bbox": [50, 50, 80, 80] if moved_complete else None,
        "start_frame": 10,
        "confirmation_frame": 20,
        "end_frame": 30,
        "quality_status": "provisional",
    }
    annotation = {
        "video_id": "vid_1",
        "camera_id": "cam_1",
        "path": "media.txt",
        "annotation_path": "annotations/vid_1.json",
        "roi_path": "roi/vid_1.json",
        "split": "validation",
        "scenario_tags": ["moved_object"],
        "fps": 10.0,
        "width": 100,
        "height": 100,
        "frame_count": 50,
        "duration_sec": 5.0,
        "events": [event],
        "roi": {"bbox": [0, 0, 100, 100]},
        "reference": {"frame_range": [0, 5]},
        "quality_status": "provisional",
        "warnings": [],
    }
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    roi_path.write_text(json.dumps({"bbox": [0, 0, 100, 100]}), encoding="utf-8")
    manifest = {
        "dataset_version": "1.0.0-draft",
        "annotation_schema_version": "1.0",
        "samples": [
            {
                "video_id": "vid_1",
                "camera_id": "cam_1",
                "path": "media.txt",
                "annotation_path": "annotations/vid_1.json",
                "roi_path": "roi/vid_1.json",
                "split": "validation",
                "scenario_tags": ["moved_object"],
                "quality_status": "provisional",
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_draft_validation_allows_provisional_missing_moved_destination(tmp_path: Path):
    report = validate_manifest(write_dataset(tmp_path, moved_complete=False), repo_root=tmp_path, mode="draft")
    assert report.ok
    assert any(issue.code == "missing_new_bbox" for issue in report.warnings)


def test_official_validation_rejects_missing_moved_destination(tmp_path: Path):
    report = validate_manifest(write_dataset(tmp_path, moved_complete=False), repo_root=tmp_path, mode="official")
    assert not report.ok
    assert any(issue.code == "missing_new_bbox" for issue in report.errors)


def test_complete_provisional_dataset_has_no_structural_errors(tmp_path: Path):
    report = validate_manifest(write_dataset(tmp_path), repo_root=tmp_path, mode="draft")
    assert report.ok


def test_validator_detects_frame_time_mismatch_and_bbox_outside_roi(tmp_path: Path):
    manifest_path = write_dataset(tmp_path)
    annotation_path = tmp_path / "annotations" / "vid_1.json"
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    annotation["events"][0]["start_time_sec"] = 1.5
    annotation["events"][0]["new_bbox"] = [90, 90, 110, 110]
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    report = validate_manifest(manifest_path, repo_root=tmp_path, mode="draft")
    assert any(issue.code == "frame_time_mismatch" for issue in report.errors)
    assert any(issue.code == "bbox_out_of_bounds" for issue in report.errors)
    assert any(issue.code == "bbox_outside_roi" for issue in report.errors)


def test_validator_detects_missing_source(tmp_path: Path):
    manifest_path = write_dataset(tmp_path)
    (tmp_path / "media.txt").unlink()
    report = validate_manifest(manifest_path, repo_root=tmp_path, mode="draft")
    assert any(issue.code == "missing_source" for issue in report.errors)
