"""M3 appearance and geometry association."""

from .engine import AssociationDependencyError, AssociationEngine
from .contracts import AssociationCandidate, AssociationMatch, AssociationResult, AssociationScore

__all__ = [
    "AssociationCandidate",
    "AssociationDependencyError",
    "AssociationEngine",
    "AssociationMatch",
    "AssociationResult",
    "AssociationScore",
]
