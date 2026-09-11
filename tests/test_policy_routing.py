from __future__ import annotations

from pathlib import Path

import pytest

from mojilex_cli.ai import DescriptionItem
from mojilex_cli.analysis import DeterministicMediaAnalysis
from mojilex_cli.domain import RoutingReason
from mojilex_cli.media import MediaMetadata, ProcessedMedia
from mojilex_cli.policy import (
    PolicyError,
    RoutingReasonRegistry,
    deterministic_routing_reasons,
    semantic_routing_reasons,
    should_escalate,
)
from test_dataset_helpers import write_fixture


def _description() -> DescriptionItem:
    localized = {
        "text": "A synthetic animated brand mark.",
        "motion_status": "undetermined",
        "usage": ["reaction"],
    }
    return DescriptionItem.model_validate(
        {
            "label": "E001",
            "descriptions": {"ru": localized, "en": localized},
            "facets": {
                "text_content": {
                    "status": "partially-recognized",
                    "dynamics": "changing",
                    "items": [
                        {
                            "value": "AC?",
                            "kind": "word",
                            "script": "Latn",
                            "language": "en",
                            "temporal_scope": "transient",
                            "media_refs": [{"role": "primary"}],
                        }
                    ],
                },
                "content_types": ["logo", "text"],
                "styles": ["flat"],
                "suggested_uses": ["branding"],
                "uncertainties": ["character-or-brand", "motion", "text"],
            },
            "semantic_tags": ["synthetic"],
            "content": {"rating": "sensitive", "warnings": ["flashing"]},
        }
    )


def _processed() -> ProcessedMedia:
    return ProcessedMedia(
        metadata=MediaMetadata(
            kind="animation",
            format="tgs",
            mime_type="application/x-tgsticker",
            sha256="1" * 64,
            byte_size=128,
            width=100,
            height=100,
            animated=True,
            duration_ms=1000,
        ),
        analysis=DeterministicMediaAnalysis.model_validate(
            {
                "color_profile_sha256": "2" * 64,
                "dedupe_profile_sha256": "3" * 64,
                "decoder_backend_fingerprint": "4" * 64,
                "rendering": {
                    "color_behavior": "fixed",
                    "palette_dynamics": "changing",
                    "alpha_mode": "translucent",
                    "visible_area_bp": 100,
                    "dominant_colors": [{"hex": "#ff0000", "family": "red", "coverage_bp": 10000}],
                },
                "fingerprint": {
                    "decoded_payload_sha256": "5" * 64,
                    "canonical_render_sha256": "6" * 64,
                    "shape_sha256": "7" * 64,
                    "perceptual": {
                        "sample_count": 16,
                        "layout_phash64": "A" * 171,
                        "content_phash64": "A" * 171,
                        "alpha_phash64": "A" * 171,
                        "edge_phash64": "A" * 171,
                        "temporal_energy_bp": 100,
                        "low_information": True,
                    },
                },
            }
        ),
        frame_paths=(Path("frame.png"),),
    )


def test_rules_use_only_typed_deterministic_and_validated_semantic_signals(
    tmp_path: Path,
) -> None:
    write_fixture(tmp_path)
    registry = RoutingReasonRegistry.load(tmp_path)

    deterministic = registry.canonicalize(deterministic_routing_reasons(_processed()))
    assert deterministic == (
        RoutingReason.COMPLEX_MOTION,
        RoutingReason.LOW_VISIBILITY,
    )
    semantic = registry.canonicalize(semantic_routing_reasons(_description()))
    assert semantic == (
        RoutingReason.CHARACTER_OR_BRAND,
        RoutingReason.PARTIAL_TEXT,
        RoutingReason.SENSITIVE_CONTENT,
    )


def test_routing_off_never_escalates_and_rules_require_explicit_model() -> None:
    reasons = (RoutingReason.COMPLEX_MOTION,)
    assert not should_escalate("off", reasons, escalation_model="strong-model")
    assert should_escalate("rules", reasons, escalation_model="strong-model")
    assert not should_escalate("rules", (), escalation_model="strong-model")
    with pytest.raises(PolicyError, match="explicit escalation model"):
        should_escalate("rules", reasons, escalation_model=None)
