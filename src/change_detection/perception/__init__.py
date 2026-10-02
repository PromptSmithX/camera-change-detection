"""Framework-neutral M2 perception contracts and observation building."""

from .contracts import Detector, FeatureEncoder, PerceptionDependencyError, Tracker
from .encoder import EmbeddingRefresher, EncoderRefreshResult
from .observation import ObservationBuildError, ObservationBuilder
from .proposal import ProposalConsolidator

__all__ = [
    "Detector",
    "EmbeddingRefresher",
    "EncoderRefreshResult",
    "FeatureEncoder",
    "ObservationBuildError",
    "ObservationBuilder",
    "PerceptionDependencyError",
    "ProposalConsolidator",
    "Tracker",
]
