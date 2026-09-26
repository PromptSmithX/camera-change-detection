import json
from pathlib import Path

import pytest

from change_detection.config import ConfigValidationError, load_config, validate_config


def test_config_validation_is_model_independent_and_round_trips(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "dataset_version": "1.0.0-draft",
                "events": {"forgotten": {"candidate_seconds": 1.0, "confirm_seconds": 4.0}},
                "evaluation": {"start_tolerance_seconds": 3.0, "moved_iou_threshold": 0.4},
            }
        ),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.dataset_version == "1.0.0-draft"
    assert config.forgotten.confirm_seconds == 4.0
    assert config.forgotten.min_confidence == 0.35
    assert config.moved.min_identity_score == 0.75
    assert config.evaluation.moved_iou_threshold == 0.4
    assert config.to_dict()["evaluation"]["min_temporal_overlap"] == 0.1
    assert validate_config(config.to_dict()).source.type == "video"


def test_event_runtime_config_round_trips_start_frame_and_warmup():
    config = validate_config(
        {
            "events": {
                "forgotten": {
                    "candidate_seconds": 0.5,
                    "confirm_seconds": 1.5,
                    "max_centroid_jitter_ratio": 0.08,
                },
                "moved": {
                    "confirm_seconds": 2.5,
                    "egress_edge_ratio": 0.2,
                },
            },
            "runtime": {"start_frame": 12, "warmup_seconds": 1.0},
        }
    )
    assert config.forgotten.candidate_seconds == 0.5
    assert config.forgotten.max_centroid_jitter_ratio == 0.08
    assert config.moved.egress_edge_ratio == 0.2
    assert config.runtime.start_frame == 12
    assert config.runtime.warmup_seconds == 1.0
    assert validate_config(config.to_dict()).runtime.start_frame == 12


def test_config_rejects_invalid_threshold_and_unknown_field():
    with pytest.raises(ConfigValidationError, match="<= 1.0"):
        validate_config({"evaluation": {"min_temporal_overlap": 1.1}})
    with pytest.raises(ConfigValidationError, match="Unknown root field"):
        validate_config({"model": {}})


def test_m1_toml_config_loads_source_and_calibration(tmp_path: Path):
    path = tmp_path / "m1.toml"
    path.write_text(
        """
[source]
type = \"image_sequence\"
path = \"frames\"
frame_pattern = \"*.bmp\"
fps = 25.0

[roi]
coordinates = [0, 0, 32, 24]

[calibration]
sample_count = 4
""".strip(),
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.source.type == "image_sequence"
    assert config.source.frame_pattern == "*.bmp"
    assert config.calibration.sample_count == 4


def test_m2_perception_config_round_trips_and_validates():
    config = validate_config(
        {
            "runtime": {"processing_fps": 5},
            "perception": {
                "enabled": True,
                "detector": {
                    "model": "weights/model.pt",
                    "confidence": 0.4,
                    "classes": [0, 2, 2],
                },
                "tracker": {
                    "lost_track_buffer": 15,
                    "minimum_consecutive_frames": 2,
                },
            },
        }
    )
    assert config.perception.enabled is True
    assert config.perception.detector.classes == (0, 2)
    assert config.perception.tracker.lost_track_buffer == 15
    assert validate_config(config.to_dict()).runtime.processing_fps == 5.0

    with pytest.raises(ConfigValidationError, match="processing_fps"):
        validate_config({"runtime": {"processing_fps": 0}})


def test_m3_identity_config_requires_perception_and_validates_weights():
    config = validate_config(
        {
            "perception": {"enabled": True, "encoder": {"enabled": True}},
            "baseline": {"path": "runs/m3/baseline/baseline.json"},
            "association": {"appearance_weight": 0.5, "spatial_weight": 0.2, "size_weight": 0.15, "class_weight": 0.15},
        }
    )
    assert config.perception.encoder.model == "dinov2_vits14"
    assert config.memory.missing_grace_seconds == 1.0
    with pytest.raises(ConfigValidationError, match="requires perception"):
        validate_config({"perception": {"encoder": {"enabled": True}}})
    with pytest.raises(ConfigValidationError, match="weights must sum"):
        validate_config({"association": {"appearance_weight": 0.4}})
