import importlib.util
import json
from pathlib import Path

from change_detection.dataset.validation import validate_manifest


def _load_apply_reviews():
    path = Path(__file__).parents[2] / "tools" / "apply_reviews.py"
    spec = importlib.util.spec_from_file_location("apply_reviews_tool", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_reviews


def test_apply_reviews_creates_new_version_without_touching_source(tmp_path: Path):
    (tmp_path / "media.txt").write_text("media", encoding="utf-8")
    (tmp_path / "annotations").mkdir()
    (tmp_path / "roi").mkdir()
    annotation = {
        "video_id": "vid_1",
        "camera_id": "cam_1",
        "path": "media.txt",
        "annotation_path": "annotations/vid_1.json",
        "roi_path": "roi/vid_1.json",
        "split": "validation",
        "scenario_tags": ["no_event"],
        "fps": 10.0,
        "width": 100,
        "height": 100,
        "frame_count": 50,
        "duration_sec": 5.0,
        "events": [],
        "roi": {"bbox": [0, 0, 100, 100]},
        "reference": {"frame_range": [0, 5]},
        "quality_status": "provisional",
        "warnings": [],
    }
    annotation_path = tmp_path / "annotations" / "vid_1.json"
    annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
    (tmp_path / "roi" / "vid_1.json").write_text(json.dumps({"bbox": [0, 0, 100, 100]}), encoding="utf-8")
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
                "scenario_tags": ["no_event"],
                "fps": 10.0,
                "width": 100,
                "height": 100,
                "frame_count": 50,
                "duration_sec": 5.0,
                "media_type": "video",
                "quality_status": "provisional",
                "annotation_sha256": "not-used-by-apply",
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _load_apply_reviews()(
        manifest_path,
        repo_root=tmp_path,
        reviews_dir=tmp_path / "reviews",
        output_dir=tmp_path / "reviewed",
        dataset_version="1.0.1-reviewed",
    )
    assert result["sample_count"] == 1
    assert result["unreviewed_sample_ids"] == ["vid_1"]
    assert manifest_path.read_text(encoding="utf-8") == json.dumps(manifest)
    output_manifest = tmp_path / "reviewed" / "manifest.json"
    report = validate_manifest(output_manifest, repo_root=tmp_path, mode="draft", strict_hashes=True)
    assert report.ok
