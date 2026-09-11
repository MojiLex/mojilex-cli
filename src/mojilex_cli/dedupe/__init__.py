"""Deterministic exact-group and bounded near-candidate APIs."""

from .engine import (
    CollectionCandidate,
    DedupeCandidate,
    DedupeScanReport,
    best_cyclic_alignment,
    explain_pair,
    scan_snapshot,
)
from .store import DedupeIndex, DedupeIndexError

__all__ = [
    "CollectionCandidate",
    "DedupeCandidate",
    "DedupeIndex",
    "DedupeIndexError",
    "DedupeScanReport",
    "best_cyclic_alignment",
    "explain_pair",
    "scan_snapshot",
]
