from __future__ import annotations

import pytest
from pydantic import ValidationError

from mojilex_cli.dataset import validate_snapshot
from mojilex_cli.domain import (
    ConceptMappingStatus,
    Facets,
    PerceptualFingerprint,
    RenderingItem,
    review_payload,
    reviewed_content_sha256,
)
from test_dataset_helpers import make_snapshot, write_fixture


def _codes(snapshot, *, canonical: bool = False) -> set[str]:  # type: ignore[no-untyped-def]
    return {item.code for item in validate_snapshot(snapshot, canonical=canonical).issues}


def test_review_hash_includes_facets_but_excludes_fingerprints(tmp_path) -> None:
    emoji = next(iter(make_snapshot(tmp_path).emojis.values()))
    original = reviewed_content_sha256(emoji)

    migrated = emoji.model_copy(deep=True)
    migrated.fingerprints.items[0].canonical_render_sha256 = "9" * 64
    assert reviewed_content_sha256(migrated) == original
    assert "fingerprints" not in review_payload(migrated)

    changed = emoji.model_copy(deep=True)
    changed.facets.styles = ["cartoon", "flat", "outline"]
    assert reviewed_content_sha256(changed) != original

    concepts_changed = emoji.model_copy(
        deep=True,
        update={
            "concept_ids": ["animal.cat"],
            "concept_mapping_status": ConceptMappingStatus.COMPLETE,
        },
    )
    assert reviewed_content_sha256(concepts_changed) != original
    assert review_payload(concepts_changed)["concept_ids"] == ["animal.cat"]


def test_rendering_palette_and_perceptual_wire_are_fail_closed() -> None:
    with pytest.raises(ValidationError, match="dominant_colors is forbidden"):
        RenderingItem(
            role="primary",
            color_behavior="platform-adaptive",
            palette_dynamics="stable",
            alpha_mode="binary",
            visible_area_bp=100,
            dominant_colors=[{"hex": "#000000", "family": "black", "coverage_bp": 10000}],
        )
    with pytest.raises(ValidationError, match="sorted"):
        RenderingItem(
            role="primary",
            color_behavior="fixed",
            palette_dynamics="stable",
            alpha_mode="opaque",
            visible_area_bp=10000,
            dominant_colors=[
                {"hex": "#ffffff", "family": "white", "coverage_bp": 1000},
                {"hex": "#000000", "family": "black", "coverage_bp": 9000},
            ],
        )
    with pytest.raises(ValidationError, match="must not exceed 10000"):
        RenderingItem(
            role="primary",
            color_behavior="fixed",
            palette_dynamics="stable",
            alpha_mode="opaque",
            visible_area_bp=10000,
            dominant_colors=[
                {"hex": "#ff0000", "family": "red", "coverage_bp": 6000},
                {"hex": "#0000ff", "family": "blue", "coverage_bp": 5000},
            ],
        )
    with pytest.raises(ValidationError, match=r"sample_count \* 8"):
        PerceptualFingerprint(
            encoding="u64be-base64url-nopad",
            sample_count=16,
            layout_phash64="AAAAAAAAAAA",
            content_phash64="AAAAAAAAAAA",
            alpha_phash64="AAAAAAAAAAA",
            edge_phash64="AAAAAAAAAAA",
            temporal_energy_bp=0,
            low_information=False,
        )


def test_text_conditional_rules_and_sorted_sets_are_typed() -> None:
    with pytest.raises(ValidationError, match="content_types to include text"):
        Facets.model_validate(
            {
                "taxonomy_version": "1.0.0",
                "rendering": {
                    "profile": "color-v1",
                    "items": [
                        {
                            "role": "primary",
                            "color_behavior": "fixed",
                            "palette_dynamics": "stable",
                            "alpha_mode": "opaque",
                            "visible_area_bp": 10000,
                            "dominant_colors": [
                                {"hex": "#000000", "family": "black", "coverage_bp": 10000}
                            ],
                        }
                    ],
                },
                "text_content": {
                    "status": "recognized",
                    "dynamics": "stable",
                    "items": [
                        {
                            "value": "404",
                            "kind": "number",
                            "script": "Zyyy",
                            "temporal_scope": "persistent",
                            "media_refs": [{"role": "primary"}],
                        }
                    ],
                },
                "content_types": ["number"],
                "styles": [],
                "suggested_uses": [],
                "uncertainties": [],
            }
        )


def test_canonical_rejects_partial_and_stale_fingerprints(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    from mojilex_cli.domain import Media

    second_media = Media.model_validate(
        {**emoji.media[0].as_dict(), "role": "dark", "variant_id": "dark"}
    )
    second_rendering = RenderingItem.model_validate(
        {
            **emoji.facets.rendering.items[0].as_dict(),
            "role": "dark",
            "variant_id": "dark",
        }
    )
    raw = emoji.as_dict()
    raw["media"].append(second_media.as_dict())
    raw["media"] = sorted(raw["media"], key=lambda item: (item["role"], item.get("variant_id", "")))
    raw["facets"]["rendering"]["items"].append(second_rendering.as_dict())
    raw["facets"]["rendering"]["items"] = sorted(
        raw["facets"]["rendering"]["items"],
        key=lambda item: (item["role"], item.get("variant_id", "")),
    )
    raw["fingerprints"]["status"] = "partial"
    from mojilex_cli.domain import Emoji, media_digest

    raw["fingerprints"]["input_media_digest"] = media_digest(raw["media"])
    partial = Emoji.model_validate(raw)
    snapshot.emojis[partial.id] = partial
    assert "FINGERPRINT_PARTIAL" in _codes(snapshot, canonical=True)

    stale = partial.model_copy(deep=True)
    stale.fingerprints.input_media_digest = "0" * 64
    snapshot.emojis[stale.id] = stale
    assert "FINGERPRINT_STALE" in _codes(snapshot)


def test_telegram_repainting_and_controlled_tag_mapping(tmp_path) -> None:
    snapshot = make_snapshot(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    adaptive = RenderingItem(
        role="primary",
        color_behavior="platform-adaptive",
        palette_dynamics="stable",
        alpha_mode="translucent",
        visible_area_bp=5000,
    )
    facets = emoji.facets.model_copy(deep=True)
    facets.rendering.items = [adaptive]
    emoji.facets = facets
    emoji.semantic_tags = [*emoji.semantic_tags, "reaction"]
    assert {"TELEGRAM_RENDERING", "FACET_TAG_DUPLICATE"}.issubset(_codes(snapshot))


def test_strict_validation_checks_profile_bytes_and_exact_qualification(tmp_path) -> None:
    write_fixture(tmp_path)
    from mojilex_cli.dataset import load_dataset

    snapshot = load_dataset(tmp_path)
    assert validate_snapshot(snapshot, canonical=True).valid
    assert not (tmp_path / "analysis-profiles").exists()

    snapshot.manifest["color_profile_sha256"] = "0" * 64
    assert "PROFILE_HASH" in _codes(snapshot, canonical=True)

    emoji = next(iter(snapshot.emojis.values()))
    emoji.provenance.qualification_id = "mq_unknown-claim"
    assert "QUALIFICATION" in _codes(snapshot, canonical=True)


def test_taxonomy_master_uses_registry_basenames(tmp_path) -> None:
    write_fixture(tmp_path)
    from mojilex_cli.dataset import load_dataset

    snapshot = load_dataset(tmp_path)
    assert validate_snapshot(snapshot, canonical=True).valid
    master = (tmp_path / "taxonomy" / "v1" / "taxonomy.json").read_text("utf-8")
    assert '"path": "content-types.json"' in master
    assert '"path": "taxonomy/v1/content-types.json"' not in master
