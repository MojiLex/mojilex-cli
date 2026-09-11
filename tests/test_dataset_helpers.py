from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mojilex_cli.analysis import known_profile_hashes
from mojilex_cli.dataset.repository import DatasetSnapshot
from mojilex_cli.dataset.staging import AtomicDatasetWriter
from mojilex_cli.domain import (
    Availability,
    Collection,
    ColorFamily,
    Content,
    ContentType,
    Emoji,
    Facets,
    FingerprintItem,
    Fingerprints,
    LocalizedDescription,
    Media,
    Membership,
    PerceptualFingerprint,
    Provenance,
    RenderingFacets,
    RenderingItem,
    Review,
    Style,
    SuggestedUse,
    ToolProvenance,
    Uncertainty,
    collection_id,
    emoji_id,
    media_digest,
    membership_id,
    telegram_set_fingerprint,
)

NOW = "2026-09-10T18:00:00Z"
MEDIA_HASH = "8f3c0e6bb13df8b5e6a4d1c9af4f2c7e184e4e2d4c47e9d63e771a48b40f5312"
NATIVE_EMOJI_ID = "5368324170671202286"
FILE_UNIQUE_ID = "AgADExampleUniqueId"
PROMPT_HASH = "1" * 64
REQUEST_PARAMETERS_HASH = "2" * 64
QUALIFICATION_ID = "mq_standard-v1_test-001"
PROFILE_HASHES = known_profile_hashes()
TAXONOMY_VALUES = {
    "content_types": [item.value for item in ContentType],
    "styles": [item.value for item in Style],
    "suggested_uses": [item.value for item in SuggestedUse],
    "uncertainties": [item.value for item in Uncertainty],
    "color_families": [item.value for item in ColorFamily],
    "platform_contexts": ["telegram"],
}


def make_snapshot(root: Path) -> DatasetSnapshot:
    collection_identifier = collection_id(
        "telegram", "sticker_set.name", "global", "SuspiciousCats"
    )
    emoji_identifier = emoji_id("telegram", "custom_emoji.id", "global", NATIVE_EMOJI_ID)
    membership_identifier = membership_id(collection_identifier, emoji_identifier)
    availability = Availability(
        status="active",
        first_seen_at=NOW,
        last_changed_at=NOW,
        last_verified_at=NOW,
    )
    media = Media(
        role="primary",
        kind="static",
        format="webp",
        mime_type="image/webp",
        sha256=MEDIA_HASH,
        byte_size=128,
        width=100,
        height=100,
        animated=False,
    )
    fingerprints = Fingerprints(
        status="complete",
        profile="dedupe-v1",
        input_media_digest=media_digest([media]),
        items=[
            FingerprintItem(
                role="primary",
                decoded_payload_sha256="3" * 64,
                canonical_render_sha256="4" * 64,
                shape_sha256="5" * 64,
                perceptual=PerceptualFingerprint(
                    encoding="u64be-base64url-nopad",
                    sample_count=1,
                    layout_phash64="AAAAAAAAAAA",
                    content_phash64="AAAAAAAAAAA",
                    alpha_phash64="AAAAAAAAAAA",
                    edge_phash64="AAAAAAAAAAA",
                    temporal_energy_bp=0,
                    low_information=False,
                ),
            )
        ],
    )
    facets = Facets(
        taxonomy_version="1.0.0",
        rendering=RenderingFacets(
            profile="color-v1",
            items=[
                RenderingItem(
                    role="primary",
                    color_behavior="fixed",
                    palette_dynamics="stable",
                    alpha_mode="translucent",
                    visible_area_bp=5000,
                    dominant_colors=[{"hex": "#f7c843", "family": "yellow", "coverage_bp": 6100}],
                )
            ],
        ),
        text_content={"status": "none", "dynamics": "stable", "items": []},
        content_types=["animal", "reaction"],
        styles=["cartoon", "flat"],
        suggested_uses=["message-accent"],
        uncertainties=[],
    )
    emoji = Emoji(
        schema_version="1.0.0",
        entity_type="emoji",
        id=emoji_identifier,
        platform="telegram",
        native_namespace="custom_emoji.id",
        scope_id="global",
        native_id=NATIVE_EMOJI_ID,
        identity_epoch=0,
        availability=availability,
        media=[media],
        fingerprints=fingerprints,
        descriptions={
            "ru": LocalizedDescription(
                text="Жёлтый кот подозрительно поднимает бровь.",
                motion_status="not_applicable",
                usage=["подозрение", "недоверие"],
            ),
            "en": LocalizedDescription(
                text="A yellow cat raises an eyebrow suspiciously.",
                motion_status="not_applicable",
                usage=["suspicion", "doubt"],
            ),
        },
        facets=facets,
        concept_ids=[],
        concept_mapping_status="pending",
        semantic_tags=["cat", "doubt", "suspicious"],
        content=Content(rating="general", warnings=[]),
        provenance=Provenance(
            origin="ai",
            provider="gemini",
            model="test-model",
            model_revision="test-revision",
            prompt_version="1.0.0",
            pipeline_version="1.0.0",
            description_profile="standard-v1",
            prompt_sha256=PROMPT_HASH,
            request_parameters_sha256=REQUEST_PARAMETERS_HASH,
            qualification_id=QUALIFICATION_ID,
            generation_stage="primary",
            routing_policy_version="1.0.0",
            routing_reason_codes=[],
            generated_at=NOW,
            input_media_sha256=[MEDIA_HASH],
            tool=ToolProvenance(name="mojilex-cli", version="0.1.0"),
        ),
        review=Review(status="unreviewed"),
        extensions={
            "telegram": {
                "schema_version": "1.0.0",
                "retrieved_via": "bot_api",
                "custom_emoji_id": NATIVE_EMOJI_ID,
                "file_unique_id": FILE_UNIQUE_ID,
                "fallback_emoji": "🤨",
                "needs_repainting": False,
            }
        },
    )
    membership = Membership(
        schema_version="1.0.0",
        entity_type="membership",
        id=membership_identifier,
        collection_id=collection_identifier,
        emoji_id=emoji_identifier,
        status="active",
        position=0,
        first_seen_at=NOW,
        last_changed_at=NOW,
    )
    collection = Collection(
        schema_version="1.0.0",
        entity_type="collection",
        id=collection_identifier,
        platform="telegram",
        kind="custom_emoji_set",
        native_namespace="sticker_set.name",
        scope_id="global",
        native_id="SuspiciousCats",
        identity_epoch=0,
        title="Suspicious Cats",
        canonical_url="https://t.me/addemoji/SuspiciousCats",
        availability=availability.model_copy(deep=True),
        item_count=1,
        extensions={
            "telegram": {
                "schema_version": "1.0.0",
                "retrieved_via": "bot_api",
                "short_name": "SuspiciousCats",
                "sticker_type": "custom_emoji",
                "set_fingerprint_sha256": telegram_set_fingerprint(
                    [(NATIVE_EMOJI_ID, FILE_UNIQUE_ID)]
                ),
            }
        },
    )
    manifest = {
        "dataset": "mojilex",
        "schema_version": "1.0.0",
        "id_namespace": "47d42c76-38da-5ab5-90fe-7af0ba6c4a27",
        "visual_relation_namespace": "4958ce2d-8120-5c3a-8755-a71d93c0c866",
        "duplicate_group_namespace": "b6a91ddf-1126-5eee-ba5e-b8bd812883df",
        "rights_assignment_namespace": "de845fef-2c83-51e1-a40a-f1bf11139639",
        "taxonomy_version": "1.0.0",
        "color_profile": "color-v1",
        "color_profile_sha256": PROFILE_HASHES["color-v1"],
        "dedupe_profile": "dedupe-v1",
        "dedupe_profile_sha256": PROFILE_HASHES["dedupe-v1"],
        "collection_dedupe_profile": "collection-dedupe-v1",
        "collection_dedupe_profile_sha256": PROFILE_HASHES["collection-dedupe-v1"],
        "default_languages": ["ru", "en"],
        "platforms": ["telegram"],
        "rights_defaults": {"project_profile_id": "mojilex-metadata-only-v1"},
        "canonical_repository": "https://github.com/MojiLex/mojilex",
        "licenses": {"data": "CC0-1.0", "code": "MIT"},
    }
    return DatasetSnapshot(
        root=root,
        manifest=manifest,
        collections={collection.id: collection},
        emojis={emoji.id: emoji},
        memberships={membership.id: membership},
    )


def write_fixture(root: Path) -> DatasetSnapshot:
    root.mkdir(parents=True, exist_ok=True)
    snapshot = make_snapshot(root.resolve())
    writer = AtomicDatasetWriter(root)
    writer.stage_bytes(
        "dataset.json",
        (json.dumps(snapshot.manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    for path, data in snapshot.to_files().items():
        writer.stage_bytes(path, data)
    taxonomy_registries: list[dict[str, str]] = []
    for facet, identifiers in TAXONOMY_VALUES.items():
        filename = f"{facet.replace('_', '-')}.json"
        relative = f"taxonomy/v1/{filename}"
        entries = [
            {
                "id": identifier,
                "name_ru": identifier,
                "name_en": identifier,
                "definition_ru": f"Synthetic definition for {identifier}.",
                "definition_en": f"Synthetic definition for {identifier}.",
                "positive_examples": ["synthetic-positive"],
                "negative_examples": ["synthetic-negative"],
                "status": "active",
            }
            for identifier in sorted(identifiers)
        ]
        registry = {"taxonomy_version": "1.0.0", "facet": facet, "entries": entries}
        registry_bytes = (json.dumps(registry, ensure_ascii=False, indent=2) + "\n").encode()
        taxonomy_registries.append(
            {
                "dictionary_id": facet,
                "path": filename,
                "sha256": hashlib.sha256(registry_bytes).hexdigest(),
            }
        )
        writer.stage_bytes(
            relative,
            registry_bytes,
        )
    taxonomy_master = {
        "taxonomy_version": "1.0.0",
        "status": "active",
        "registries": sorted(taxonomy_registries, key=lambda item: item["path"]),
    }
    writer.stage_bytes(
        "taxonomy/v1/taxonomy.json",
        (json.dumps(taxonomy_master, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    qualification_registry = {
        "schema_version": "1.0.0",
        "registry_id": "model-qualifications-v1",
        "entries": [
            {
                "qualification_id": QUALIFICATION_ID,
                "provider": "gemini",
                "model": "test-model",
                "model_revision": "test-revision",
                "description_profile": "standard-v1",
                "prompt_sha256": PROMPT_HASH,
                "request_parameters_sha256": REQUEST_PARAMETERS_HASH,
                "schema_version": "1.0.0",
                "taxonomy_version": "1.0.0",
                "pipeline_version": "1.0.0",
                "routing_policy_version": "1.0.0",
                "languages": ["en", "ru"],
                "benchmark_id": "synthetic-test-v1",
                "benchmark_sha256": "6" * 64,
                "split_id": "synthetic-split-v1",
                "split_sha256": "7" * 64,
                "report_sha256": "8" * 64,
                "valid_from": "2026-01-01T00:00:00Z",
                "status": "active",
            }
        ],
    }
    writer.stage_bytes(
        "quality/model-qualifications.json",
        (json.dumps(qualification_registry, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    routing_reasons = {
        "schema_version": "1.0.0",
        "registry_id": "routing-reasons-v1",
        "entries": [
            {"id": reason, "definition": f"Synthetic definition for {reason}."}
            for reason in (
                "character-or-brand",
                "complex-motion",
                "facet-conflict",
                "low-visibility",
                "ocr-conflict",
                "partial-text",
                "quality-control-sample",
                "schema-retry-exhausted",
                "sensitive-content",
                "unqualified-model",
            )
        ],
    }
    review_reasons = {
        "schema_version": "1.0.0",
        "registry_id": "review-reasons-v1",
        "entries": [
            {"id": reason, "definition": f"Synthetic definition for {reason}."}
            for reason in (
                "exact-group-description-conflict",
                "moderation-uncertainty",
                "motion-uncertainty",
                "ocr-conflict",
                "unknown-character-or-brand",
                "unqualified-model",
            )
        ],
    }
    review_routing = {
        "schema_version": "1.0.0",
        "policy_id": "review-routing-v1",
        "priority_order": ["blocking", "high", "normal", "low"],
        "rules": [
            {
                "reason_code": "unqualified-model",
                "priority": "blocking",
                "condition": (
                    "ai-result-has-no-exact-active-qualification-and-review-is-not-approved"
                ),
            },
            {
                "reason_code": "moderation-uncertainty",
                "priority": "blocking",
                "condition": (
                    "rating-is-not-general-or-warnings-are-not-empty-and-review-is-not-approved"
                ),
            },
            {
                "reason_code": "motion-uncertainty",
                "priority": "high",
                "condition": "facets-uncertainties-contains-motion",
            },
            {
                "reason_code": "ocr-conflict",
                "priority": "high",
                "condition": "routing-reason-codes-contains-ocr-conflict",
            },
            {
                "reason_code": "unknown-character-or-brand",
                "priority": "high",
                "condition": "facets-uncertainties-contains-character-or-brand",
            },
            {
                "reason_code": "exact-group-description-conflict",
                "priority": "normal",
                "condition": "descriptions-conflict-inside-an-exact-group",
            },
        ],
    }
    for filename, value in (
        ("routing-reasons-v1.json", routing_reasons),
        ("review-reasons-v1.json", review_reasons),
        ("review-routing-v1.json", review_routing),
    ):
        writer.stage_bytes(
            f"quality/{filename}",
            (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode(),
        )
    concepts = {
        "registry_schema_version": "1.0.0",
        "registry_type": "concepts",
        "registry_id": "concepts-v1.synthetic-001",
        "concepts": [
            {
                "id": "animal.cat",
                "status": "active",
                "labels": {"en": "cat", "ru": "кот"},
                "aliases": {"en": ["feline"], "ru": ["кошка"]},
                "definitions": {"en": "A cat.", "ru": "Кот."},
                "parent_ids": [],
                "positive_examples": {"en": ["cat"], "ru": ["кот"]},
                "negative_examples": {"en": ["dog"], "ru": ["собака"]},
            }
        ],
    }
    writer.stage_bytes(
        "taxonomy/v1/concepts.json",
        (json.dumps(concepts, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    operation_decisions = {
        "publish-metadata": {"decision": "allow"},
        "publish-generated-annotations": {"decision": "allow"},
        "store-source-media": {
            "decision": "conditional",
            "conditions": ["transient-run-only"],
        },
        "redistribute-source-media": {"decision": "not-granted"},
        "publish-derived-preview": {"decision": "not-granted"},
        "send-media-to-external-ai": {
            "decision": "conditional",
            "conditions": ["operator-explicit-consent"],
        },
        "use-media-in-public-benchmark": {"decision": "not-granted"},
    }
    rights_profiles = []
    for profile_id, applies_to in (
        ("mojilex-metadata-only-v1", {"project": "mojilex"}),
        ("telegram-index-only-v1", {"platform": "telegram"}),
    ):
        rights_profiles.append(
            {
                "rights_profile_id": profile_id,
                "profile_version": "1.0.0",
                "status": "active",
                "applies_to": applies_to,
                "dataset_grant": {
                    "metadata_and_annotations_license": "CC0-1.0",
                    "exclusions": [
                        "characters",
                        "logos",
                        "source-media",
                        "third-party-copyright",
                        "trademarks",
                    ],
                },
                "operations": operation_decisions,
                "basis": [
                    {
                        "kind": "project-policy",
                        "document": "LICENSING.md",
                        "document_sha256": "9" * 64,
                    }
                ],
                "attribution_required": False,
                "effective_from": "2026-09-11T00:00:00Z",
            }
        )
    rights = {
        "registry_schema_version": "1.0.0",
        "registry_type": "rights-profiles",
        "registry_id": "rights-profiles-v1.synthetic-001",
        "project_default_profile_id": "mojilex-metadata-only-v1",
        "profiles": rights_profiles,
    }
    writer.stage_bytes(
        "rights/profiles.json",
        (json.dumps(rights, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    platform_profile = {
        "profile_schema_version": "1.0.0",
        "platform": "telegram",
        "profile_version": "telegram-capabilities-v1",
        "adapter_contract_version": "1.0.0",
        "default_rights_profile_id": "telegram-index-only-v1",
        "capabilities": [
            {
                "capability_id": "telegram.message-custom-emoji",
                "surface": "message",
                "status": "supported",
                "applies_to": "custom_emoji",
                "authority_class": "official-documentation",
                "evidence_url": "https://core.telegram.org/stickers#custom-emoji",
                "observed_at": "2026-09-11T00:00:00Z",
                "freshness_period_seconds": 7776000,
            }
        ],
    }
    writer.stage_bytes(
        "platforms/telegram.json",
        (json.dumps(platform_profile, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    writer.commit()
    return snapshot
