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
    assert config.evaluation.moved_iou_threshold == 0.4
    assert config.to_dict()["evaluation"]["min_temporal_overlap"] == 0.1


def test_config_rejects_invalid_threshold_and_unknown_field():
    with pytest.raises(ConfigValidationError, match="<= 1.0"):
        validate_config({"evaluation": {"min_temporal_overlap": 1.1}})
    with pytest.raises(ConfigValidationError, match="Unknown root field"):
        validate_config({"model": {}})
