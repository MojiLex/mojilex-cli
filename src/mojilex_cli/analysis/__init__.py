"""Public deterministic media-analysis API."""

from .backend import DecoderKind, decoder_backend_fingerprint
from .engine import MediaAnalysisAccumulator, analyze_decoded_media, sample_frame_indexes
from .models import (
    AnalysisError,
    DeterministicMediaAnalysis,
    DominantColor,
    FingerprintSignals,
    PerceptualSignals,
    RenderingSignals,
)
from .profiles import AnalysisProfile, known_profile_hashes, load_analysis_profile, profile_sha256

__all__ = [
    "AnalysisError",
    "AnalysisProfile",
    "DecoderKind",
    "DeterministicMediaAnalysis",
    "DominantColor",
    "FingerprintSignals",
    "MediaAnalysisAccumulator",
    "PerceptualSignals",
    "RenderingSignals",
    "analyze_decoded_media",
    "decoder_backend_fingerprint",
    "known_profile_hashes",
    "load_analysis_profile",
    "profile_sha256",
    "sample_frame_indexes",
]
