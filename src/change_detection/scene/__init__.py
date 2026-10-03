"""Scene ROI, stability, calibration, and baseline contracts."""

from .baseline import BaselineIntegrityError, SceneBaseline
from .calibration import CalibrationError, CalibrationResult, CalibrationService
from .roi import BBoxROI, ROI, ROIError, load_roi_file, roi_from_mapping
from .stability import (
    SceneStabilityMonitor,
    StabilityObservation,
    StabilityState,
    StabilitySummary,
)
from .status import RegionChangeEvidence, SceneStatus, SceneStatusProvider, StableSceneStatusProvider

__all__ = [
    "BBoxROI",
    "BaselineIntegrityError",
    "CalibrationError",
    "CalibrationResult",
    "CalibrationService",
    "ROI",
    "ROIError",
    "RegionChangeEvidence",
    "SceneBaseline",
    "SceneStabilityMonitor",
    "StabilityObservation",
    "StabilityState",
    "StabilitySummary",
    "load_roi_file",
    "roi_from_mapping",
    "SceneStatus",
    "SceneStatusProvider",
    "StableSceneStatusProvider",
]
