from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from tools.benchmark_dataset import aggregate_summary
from tools.run_validation import _evaluation_command, _verify_test_lock


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_aggregate_summary_removes_diagnostic_test_details() -> None:
    result = {
        "dataset": {
            "manifest": "private.json",
            "dataset_version": "synthetic-locked",
            "split": "test",
            "test_locked": True,
            "sample_count": 2,
            "missing_prediction_count": 1,
            "missing_predictions": ["secret-sample"],
            "ignored_prediction_ids": ["secret-sample"],
        },
        "overall": {
            "tp": 1,
            "fp": 2,
            "fn": 3,
            "precision": 1 / 3,
            "recall": 0.25,
            "f1": 2 / 7,
        },
        "per_sample": [{"sample_id": "secret-sample"}],
        "matches": [{"sample_id": "secret-sample", "bbox": [1, 2, 3, 4]}],
        "error_breakdown": {"spatial_mismatch": 1},
    }

    summary = aggregate_summary(result)

    assert summary["overall"] == result["overall"]
    assert summary["dataset"] == {
        "dataset_version": "synthetic-locked",
        "split": "test",
        "test_locked": True,
        "sample_count": 2,
        "missing_prediction_count": 1,
    }
    assert "per_sample" not in summary
    assert "matches" not in summary
    assert "secret-sample" not in json.dumps(summary)


def test_test_lock_verification_accepts_matching_synthetic_receipt(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.locked.json"
    manifest = {
        "dataset_version": "synthetic-locked",
        "test_locked": True,
        "locked_test_sha256": "synthetic-test-digest",
        "samples": [],
    }
    _write_json(manifest_path, manifest)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    lock_path = tmp_path / "locked_test.json"
    _write_json(
        lock_path,
        {
            "locked": True,
            "dataset_version": "synthetic-locked",
            "manifest_sha256": manifest_sha256,
            "test_sha256": "synthetic-test-digest",
        },
    )

    _verify_test_lock(manifest_path, lock_path)

    manifest["dataset_version"] = "tampered"
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="hash"):
        _verify_test_lock(manifest_path, lock_path)


def test_test_evaluation_command_is_opt_in_and_summary_only(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    run_root = root / "runs" / "test" / "synthetic"
    manifest_path = root / "synthetic-manifest.json"
    output_path = run_root / "metrics" / "evaluation.json"
    args = argparse.Namespace(
        split="test",
        event_boundary_tolerance=1.0,
        forgotten_iou=0.3,
        moved_iou=0.3,
        latency_deadlines_seconds=(3.0, 5.0, 10.0),
        require_object_id=False,
    )

    command = _evaluation_command(
        root=root,
        run_root=run_root,
        manifest_path=manifest_path,
        output_path=output_path,
        args=args,
        ignored_sample_ids=[],
    )

    assert command[command.index("--split") + 1] == "test"
    assert "--allow-test" in command
    assert "--summary-only" in command


def test_validation_evaluation_command_remains_detailed(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    args = argparse.Namespace(
        split="validation",
        event_boundary_tolerance=1.0,
        forgotten_iou=0.3,
        moved_iou=0.3,
        latency_deadlines_seconds=(3.0, 5.0, 10.0),
        require_object_id=False,
    )

    command = _evaluation_command(
        root=root,
        run_root=root / "runs" / "validation" / "synthetic",
        manifest_path=root / "synthetic-manifest.json",
        output_path=root / "runs" / "validation" / "synthetic" / "metrics.json",
        args=args,
        ignored_sample_ids=[],
    )

    assert command[command.index("--split") + 1] == "validation"
    assert "--allow-test" not in command
    assert "--summary-only" not in command
