from __future__ import annotations

from pathlib import Path

from tools.benchmark import _load_events
from tools.benchmark_dataset import aggregate_results


def test_benchmark_loads_single_record_jsonl(tmp_path: Path) -> None:
    predictions = tmp_path / "events.jsonl"
    predictions.write_text('{"event_id": "evt-1"}\n', encoding="utf-8")

    assert _load_events(predictions) == [{"event_id": "evt-1"}]


def test_dataset_benchmark_aggregates_per_sample_without_cross_video_matching(tmp_path: Path):
    result = aggregate_results(
        [
            (
                "sample-1",
                {
                    "overall": {"tp": 1, "fp": 1, "fn": 0},
                    "FORGOTTEN_OBJECT": {"tp": 1, "fp": 1, "fn": 0},
                    "MOVED_OBJECT": {"tp": 0, "fp": 0, "fn": 0},
                    "MOVED_OBJECT_OUTCOMES": {
                        "relocated": {"tp": 0, "fp": 0, "fn": 0},
                        "left_scene": {"tp": 0, "fp": 0, "fn": 0},
                    },
                    "matches": [
                        {
                            "prediction_event_id": "pred-1",
                            "ground_truth_event_id": "gt-1",
                            "confirmation_latency_sec": 0.5,
                            "start_delay_sec": 2.0,
                        }
                    ],
                    "error_breakdown": {"duplicate_event": 1},
                    "latency": {"matched_events": 1, "mean_confirmation_delta_sec": 0.5},
                },
            ),
            (
                "sample-2",
                {
                    "overall": {"tp": 0, "fp": 0, "fn": 1},
                    "FORGOTTEN_OBJECT": {"tp": 0, "fp": 0, "fn": 0},
                    "MOVED_OBJECT": {"tp": 0, "fp": 0, "fn": 1},
                    "MOVED_OBJECT_OUTCOMES": {
                        "relocated": {"tp": 0, "fp": 0, "fn": 1},
                        "left_scene": {"tp": 0, "fp": 0, "fn": 0},
                    },
                    "matches": [],
                    "error_breakdown": {"missed_moved_object": 1},
                    "latency": {"matched_events": 0, "mean_confirmation_delta_sec": None},
                },
            ),
        ],
        manifest=tmp_path / "manifest.json",
        dataset_version="test",
        split="validation",
        missing_predictions=["sample-2"],
    )

    assert result["overall"] == {
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    assert result["FORGOTTEN_OBJECT"]["f1"] == 2 / 3
    assert result["MOVED_OBJECT"]["fn"] == 1
    assert result["MOVED_OBJECT_OUTCOMES"]["relocated"]["fn"] == 1
    assert result["latency"]["mean_confirmation_delta_sec"] == 0.5
    assert result["timeliness"]["confirmation_latency_sec"]["median"] == 0.5
    assert result["timeliness"]["deadlines"]["3s"]["on_time_count"] == 1
    assert result["error_breakdown"] == {
        "duplicate_event": 1,
        "missed_moved_object": 1,
    }
