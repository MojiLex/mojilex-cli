from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import rfc8785
import typer

from mojilex_cli.analysis import load_analysis_profile
from mojilex_cli.commands.read import (
    ReadCommandResult,
    _request_sha256,
    _structured_search,
    _validate_filters,
    execute_read,
)
from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.read.cursor import (
    MAX_CURSOR_BYTES,
    decode_cursor,
    encode_cursor,
    pagination_domain,
)
from mojilex_cli.read.service import SnapshotReader
from mojilex_cli.read.snapshot import (
    LoadedSnapshot,
    _embedded_schema_contracts,
    _schema_contracts,
    _validate_delegated_profile,
    _validate_minimum_reader_version,
    parse_bounded_json,
    validate_embedded_schema_instance,
)
from mojilex_cli.schemas import EMBEDDED_SCHEMA_PATHS, embedded_schemas


def _sha(value: object) -> str:
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


class FakeSnapshot:
    def __init__(self) -> None:
        emoji = {
            "schema_version": "1.0.0",
            "entity_type": "emoji",
            "id": "mxe_11111111-1111-5111-8111-111111111111",
            "platform": "telegram",
            "native_namespace": "custom_emoji.id",
            "scope_id": "global",
            "native_id": "9007199254740993",
            "identity_epoch": 0,
            "availability": {
                "status": "active",
                "first_seen_at": "2026-09-10T18:00:00Z",
            },
            "content": {"rating": "general", "warnings": []},
            "review": {"status": "approved", "review_hash_profile_id": "review-v3"},
        }
        collection = {
            "schema_version": "1.0.0",
            "entity_type": "collection",
            "id": "mxc_22222222-2222-5222-8222-222222222222",
            "platform": "telegram",
            "native_namespace": "sticker_set.name",
            "scope_id": "global",
            "native_id": "SafeCats",
            "identity_epoch": 0,
            "availability": {
                "status": "active",
                "first_seen_at": "2026-09-10T18:00:00Z",
            },
        }
        membership = {
            "schema_version": "1.0.0",
            "entity_type": "membership",
            "id": "mxm_33333333-3333-5333-8333-333333333333",
            "collection_id": collection["id"],
            "emoji_id": emoji["id"],
            "status": "active",
            "position": 0,
        }
        rights = {
            "rights_profile_id": "telegram-index-only-v1",
            "distribution_status": "allowed",
            "attribution_required": False,
        }
        search = {
            "record_schema_version": "1.0.0",
            "entity_type": "search_record",
            "emoji_id": emoji["id"],
            "language": "en",
            "canonical_record_sha256": _sha(emoji),
            "platform": "telegram",
            "native_reference_count": 1,
            "native_references": [
                {
                    "platform": "telegram",
                    "native_namespace": "custom_emoji.id",
                    "scope_id": "global",
                    "native_id": "9007199254740993",
                    "identity_epoch": 0,
                }
            ],
            "native_references_truncated": False,
            "collection_count": 1,
            "collection_ids": [collection["id"]],
            "collections_truncated": False,
            "description": {"text": "A calm cat", "motion": "", "usage": ["calm"]},
            "semantic": {"concept_ids": ["animal.cat"], "semantic_tags": ["cat"]},
            "facets": {
                "animated": False,
                "media_kinds": ["static"],
                "color_behaviors": ["fixed"],
                "color_families": ["yellow"],
                "contains_text": False,
                "content_types": ["animal"],
                "styles": ["flat"],
                "suggested_uses": ["reaction"],
                "uncertainties": [],
            },
            "literal_text": [],
            "availability": {"status": "active", "freshness_status": "unknown"},
            "review": {
                "status": "approved",
                "attested": True,
                "provenance_origin": "human",
                "model_qualification_status": "not-applicable",
                "model_qualification": {"present": False},
                "generation_attestation_status": "not-applicable",
            },
            "content": {"rating": "general", "warnings": []},
            "rights": rights,
            "duplicate_group_count": 0,
            "duplicate_group_ids": [],
            "duplicate_groups_truncated": False,
            "platform_capability_refs": ["telegram.message-custom-emoji"],
            "canonical_locator": {"logical_name": "emojis", "record_key": []},
        }
        self.snapshot_id = "data-2026.09.11.1"
        self.manifest_sha256 = "a" * 64
        self.source_canonical_state_root_sha256 = "b" * 64
        self.manifest = {
            "languages": {
                "available": ["en", "ru"],
                "fallback_profile": "language-fallback-v1",
            },
            "profiles": {"dedupe": {"id": "dedupe-v1", "sha256": "c" * 64}},
            "build": {"source_date_epoch": 1789171199},
        }
        self.descriptors = {"search-en": {}}
        self._rows = {
            "emojis": (emoji,),
            "collections": (collection,),
            "memberships": (membership,),
            "tombstones": (),
            "search-en": (search,),
        }
        self._documents = {
            "platform-profiles": {
                "entries": [
                    {
                        "platform": "telegram",
                        "default_rights_profile_id": "telegram-index-only-v1",
                    }
                ]
            },
            "rights-profiles": {
                "project_default_profile_id": "project-default-v1",
                "profiles": [
                    {
                        "rights_profile_id": "telegram-index-only-v1",
                        "status": "active",
                        "operations": {
                            "publish-metadata": {"decision": "allow"},
                            "publish-generated-annotations": {"decision": "allow"},
                        },
                        "attribution_required": False,
                    }
                ],
            },
        }
        self._profiles = {
            "language_fallback": {
                "profile_id": "language-fallback-v1",
                "lookup_algorithm": "rfc4647-lookup",
                "chains": {"en": ["en"], "ru": ["ru", "en"]},
                "machine_translation": "forbidden",
            },
            "lexical_search": {
                "profile_id": "lexical-search-v1",
                "query_normalization": "NFKC",
                "case_folding": "locale-aware",
                "whitespace_policy": "collapse",
                "token_boundary_policy": "unicode-punctuation-v1",
                "fields": [
                    {"field": "description.text", "weight": 100},
                    {"field": "description.motion", "weight": 60},
                    {"field": "description.usage", "weight": 80},
                    {"field": "semantic.concept_ids", "weight": 120},
                    {"field": "semantic.semantic_tags", "weight": 90},
                    {"field": "literal_text.value", "weight": 110},
                ],
                "match_priority": [
                    "exact-mojilex-id",
                    "exact-native-reference",
                    "exact-concept-id",
                    "exact-phrase",
                    "prefix",
                    "token",
                    "concept-alias",
                ],
                "match_rules": {
                    "field_evaluation": "best-tier-per-field-v1",
                    "exact": "normalized-full-value-v1",
                    "phrase": "normalized-contiguous-substring-v1",
                    "prefix": "normalized-token-prefix-v1",
                    "token": "normalized-token-equality-v1",
                    "concept_alias": "normalized-full-value-v1",
                },
                "match_scores": {
                    "exact_mojilex_id": 1_000_000,
                    "exact_native_reference": 900_000,
                    "exact_concept_id": 800_000,
                    "field_exact_multiplier": 400,
                    "field_phrase_multiplier": 300,
                    "field_prefix_multiplier": 200,
                    "field_token_multiplier": 100,
                    "concept_alias_bonus_multiplier": 1,
                },
                "tie_break": "emoji-id-utf8-bytewise-ascending",
                "stop_word_policy": "none",
                "stemming_policy": "none",
                "ai_translation": "forbidden",
                "embeddings": False,
            },
            "concepts": {
                "concepts": [
                    {
                        "id": "animal.cat",
                        "labels": {"en": "cat", "ru": "кот"},
                        "aliases": {"en": ["feline"], "ru": ["кошка"]},
                    }
                ]
            },
        }

    def rows(self, name: str) -> tuple[dict[str, Any], ...]:
        return self._rows[name]

    def has(self, name: str) -> bool:
        return name in self._rows or name in self._documents or name in self._profiles

    def document(self, name: str) -> dict[str, Any]:
        return self._documents[name]

    def bound_document(self, container: str, field: str) -> dict[str, Any]:
        assert container == "profiles"
        return self._profiles[field]

    def pinned_state(self) -> dict[str, str]:
        return {"snapshot_id": self.snapshot_id, "manifest_sha256": self.manifest_sha256}

    def dataset_context(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "manifest_sha256": self.manifest_sha256,
            "release_verification_status": "integrity-only-unsigned",
            "catalog_status": "unknown",
            "revocation_status": "unknown",
            "control_state_status": "offline-unknown",
            "errata_status": "unknown",
            "catalog_checkpoint": {"present": False},
        }


def test_cursor_round_trip_and_domain_binding() -> None:
    domain = pagination_domain(
        command="search",
        pinned_state={"snapshot_id": "data-2026.09.11.1", "manifest_sha256": "a" * 64},
        semantic_arguments={"query": "cat", "limit": 1},
    )
    cursor = encode_cursor(domain, [-400, "mxe_example"])
    assert decode_cursor(cursor, domain) == [-400, "mxe_example"]
    with pytest.raises(CommandError) as failure:
        decode_cursor(cursor, "b" * 64)
    assert failure.value.error.code == "CURSOR_INVALID"


def test_cursor_rejects_noncanonical_and_nonfinite_payloads() -> None:
    body = b'{"version":"1", "pagination_domain_sha256":"x","last_tuple":[]}'
    encoded = base64.urlsafe_b64encode(body).rstrip(b"=").decode("ascii")
    with pytest.raises(CommandError):
        decode_cursor(f"{encoded}.{'0' * 64}", "x")
    nan_body = b'{"version":"1","pagination_domain_sha256":"x","last_tuple":[NaN]}'
    encoded_nan = base64.urlsafe_b64encode(nan_body).rstrip(b"=").decode("ascii")
    with pytest.raises(CommandError):
        decode_cursor(f"{encoded_nan}.{'0' * 64}", "x")

    with pytest.raises(CommandError) as oversized:
        decode_cursor("x" * (MAX_CURSOR_BYTES + 1), "x")
    assert oversized.value.error.code == "RESOURCE_LIMIT_EXCEEDED"


def test_bounded_json_rejects_bom_and_excessive_depth() -> None:
    with pytest.raises(CommandError) as bom:
        parse_bounded_json(b'\xef\xbb\xbf{"value":1}', source="request.json")
    assert bom.value.error.code == "MANIFEST_INVALID"

    nested = b"[" * 65 + b"0" + b"]" * 65
    with pytest.raises(CommandError) as depth:
        parse_bounded_json(b'{"value":' + nested + b"}", source="request.json")
    assert depth.value.error.code == "RESOURCE_LIMIT_EXCEEDED"

    with pytest.raises(CommandError) as surrogate:
        parse_bounded_json(b'{"value":"\\ud800"}', source="request.json")
    assert surrogate.value.error.code == "MANIFEST_INVALID"


def _embedded_snapshot(tmp_path: Path) -> LoadedSnapshot:
    descriptors: list[dict[str, Any]] = []
    verified_paths: dict[str, Path] = {}
    verified_bytes: dict[str, bytes] = {}
    for index, (uri, embedded) in enumerate(sorted(embedded_schemas().items())):
        path = tmp_path / "schemas" / f"{index:02d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(embedded.payload)
        descriptors.append(
            {
                "uri": uri,
                "resource_kind": "physical",
                "media_type": "application/schema+json",
                "payload_sha256": embedded.sha256,
                "object_sha256": embedded.sha256,
            }
        )
        verified_paths[uri] = path
        verified_bytes[uri] = embedded.payload
    return LoadedSnapshot(
        root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        manifest={"artifacts": [], "resources": []},
        manifest_sha256="a" * 64,
        descriptors={},
        verified_paths={},
        resource_descriptors=tuple(descriptors),
        verified_resource_paths=verified_paths,
        verified_resource_bytes=verified_bytes,
    )


def test_embedded_schema_inventory_is_complete_and_loadable_from_package() -> None:
    schemas = embedded_schemas()
    assert len(schemas) == len(EMBEDDED_SCHEMA_PATHS) == 54
    assert "https://schemas.mojilex.org/v1/extensions/telegram.schema.json" in schemas
    assert "mlx://schemas/distribution/v1/release-manifest.schema.json" in schemas


def test_snapshot_schema_bytes_must_match_embedded_trust_root(tmp_path: Path) -> None:
    snapshot = _embedded_snapshot(tmp_path)
    uri = "mlx://schemas/distribution/v1/release-manifest.schema.json"
    substituted = snapshot.verified_resource_bytes[uri] + b"\n"
    snapshot.verified_resource_bytes[uri] = substituted
    digest = hashlib.sha256(substituted).hexdigest()
    for descriptor in snapshot.resource_descriptors:
        if descriptor["uri"] == uri:
            descriptor["payload_sha256"] = digest
            descriptor["object_sha256"] = digest

    with pytest.raises(CommandError) as failure:
        _schema_contracts(snapshot)
    assert failure.value.error.code == "SCHEMA_UNSUPPORTED"
    assert failure.value.error.details == {
        "schema_uri": uri,
        "expected_sha256": embedded_schemas()[uri].sha256,
        "snapshot_sha256": digest,
    }


def test_snapshot_must_publish_every_required_embedded_schema(tmp_path: Path) -> None:
    snapshot = _embedded_snapshot(tmp_path)
    uri = "mlx://schemas/distribution/v1/distribution-common.schema.json"
    snapshot.resource_descriptors = tuple(
        descriptor for descriptor in snapshot.resource_descriptors if descriptor["uri"] != uri
    )
    snapshot.verified_resource_bytes.pop(uri)

    with pytest.raises(CommandError) as failure:
        _schema_contracts(snapshot)
    assert failure.value.error.code == "SCHEMA_UNSUPPORTED"
    assert failure.value.error.details == {"schema_uri": uri}


def test_unknown_semantic_schema_is_rejected_without_snapshot_trust(tmp_path: Path) -> None:
    snapshot = _embedded_snapshot(tmp_path)
    snapshot.descriptors["future"] = {
        "logical_name": "future",
        "schema_ref": "mlx://schemas/distribution/v2/future.schema.json",
    }

    with pytest.raises(CommandError) as failure:
        _schema_contracts(snapshot)
    assert failure.value.error.code == "SCHEMA_UNSUPPORTED"
    assert failure.value.error.details == {
        "schema_uri": "mlx://schemas/distribution/v2/future.schema.json"
    }


def test_cached_schema_bytes_prevent_post_load_resource_swap(tmp_path: Path) -> None:
    snapshot = _embedded_snapshot(tmp_path)
    uri = "mlx://schemas/distribution/v1/release-manifest.schema.json"
    snapshot.verified_resource_paths[uri].write_bytes(b"{}")
    schemas, _ = _schema_contracts(snapshot)
    assert schemas[uri]["$id"] == uri


def test_delegated_profile_contract_hash_and_body_are_both_enforced() -> None:
    wrapper = json.loads(load_analysis_profile("dedupe-v1").raw_bytes)
    schemas, registry, _ = _embedded_schema_contracts()
    _validate_delegated_profile(
        wrapper,
        schemas=schemas,
        registry=registry,
        location="dedupe-v1",
    )

    wrong_hash = json.loads(json.dumps(wrapper))
    wrong_hash["contract_schema_sha256"] = "0" * 64
    with pytest.raises(CommandError) as mismatch:
        _validate_delegated_profile(
            wrong_hash,
            schemas=schemas,
            registry=registry,
            location="dedupe-v1",
        )
    assert mismatch.value.error.code == "SCHEMA_UNSUPPORTED"

    wrong_body = json.loads(json.dumps(wrapper))
    wrong_body["body"]["candidate_thresholds"]["candidate_limit_default"] = 21
    with pytest.raises(CommandError) as invalid:
        _validate_delegated_profile(
            wrong_body,
            schemas=schemas,
            registry=registry,
            location="dedupe-v1",
        )
    assert invalid.value.error.code == "MANIFEST_INVALID"


def test_delegated_profile_requires_exact_published_contract_schema(tmp_path: Path) -> None:
    snapshot = _embedded_snapshot(tmp_path)
    raw = load_analysis_profile("dedupe-v1").raw_bytes
    wrapper = json.loads(raw)
    profile_uri = "mlx://profiles/dedupe/dedupe-v1/test.json"
    snapshot.resource_descriptors += (
        {
            "uri": profile_uri,
            "resource_kind": "physical",
            "media_type": "application/json",
            "content_schema_ref": ("mlx://schemas/distribution/v1/delegated-profile.schema.json"),
        },
    )
    snapshot.verified_resource_bytes[profile_uri] = raw
    contract_uri = wrapper["contract_schema_ref"]
    snapshot.resource_descriptors = tuple(
        descriptor
        for descriptor in snapshot.resource_descriptors
        if descriptor["uri"] != contract_uri
    )
    snapshot.verified_resource_bytes.pop(contract_uri)

    with pytest.raises(CommandError) as missing:
        _schema_contracts(snapshot)
    assert missing.value.error.code == "SCHEMA_UNSUPPORTED"
    assert missing.value.error.details == {"schema_uri": contract_uri}


def test_bound_document_exposes_only_validated_delegated_profile_body(tmp_path: Path) -> None:
    raw = load_analysis_profile("dedupe-v1").raw_bytes
    digest = hashlib.sha256(raw).hexdigest()
    uri = f"mlx://profiles/dedupe/dedupe-v1/{digest}.json"
    snapshot = LoadedSnapshot(
        root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        manifest={"profiles": {"dedupe": {"id": "dedupe-v1", "sha256": digest}}},
        manifest_sha256="a" * 64,
        descriptors={},
        verified_paths={},
        resource_descriptors=(
            {
                "uri": uri,
                "resource_kind": "physical",
                "payload_sha256": digest,
                "content_schema_ref": (
                    "mlx://schemas/distribution/v1/delegated-profile.schema.json"
                ),
                "bindings": [
                    {
                        "kind": "exact-content",
                        "manifest_pointer": "/profiles/dedupe/sha256",
                    }
                ],
            },
        ),
        verified_resource_bytes={uri: raw},
    )

    effective = snapshot.bound_document("profiles", "dedupe")
    assert effective["profile_id"] == "dedupe-v1"
    assert "candidate_thresholds" in effective
    assert "body" not in effective


@pytest.mark.parametrize("minimum", ["0.1.0", "0.2.0"])
def test_minimum_reader_version_accepts_lower_or_equal(minimum: str) -> None:
    _validate_minimum_reader_version({"minimum_reader_version": minimum})


def test_minimum_reader_version_rejects_higher_and_malformed() -> None:
    with pytest.raises(CommandError) as higher:
        _validate_minimum_reader_version({"minimum_reader_version": "0.2.1"})
    assert higher.value.error.code == "SCHEMA_UNSUPPORTED"

    with pytest.raises(CommandError) as malformed:
        _validate_minimum_reader_version({"minimum_reader_version": "v0.2"})
    assert malformed.value.error.code == "MANIFEST_INVALID"


@pytest.mark.parametrize(
    "content_model,payload,mutated",
    [
        ("recordset-jsonl", b"{}\n", b'{"changed":true}\n'),
        ("singleton-json", b"{}", b'{"changed":true}'),
    ],
)
def test_artifact_is_reverified_on_the_same_bytes_that_are_parsed(
    tmp_path: Path,
    content_model: str,
    payload: bytes,
    mutated: bytes,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(mutated)
    digest = hashlib.sha256(payload).hexdigest()
    descriptor: dict[str, Any] = {
        "logical_name": "artifact",
        "path": "artifact.json",
        "content_model": content_model,
        "object_byte_size": len(payload),
        "uncompressed_byte_size": len(payload),
        "object_sha256": digest,
        "payload_sha256": digest,
        "record_count": 1,
        "logical_record_count": 1,
    }
    snapshot = LoadedSnapshot(
        root=tmp_path,
        manifest_path=tmp_path / "manifest.json",
        manifest={},
        manifest_sha256="a" * 64,
        descriptors={"artifact": descriptor},
        verified_paths={"artifact": path},
    )

    with pytest.raises(CommandError) as failure:
        if content_model == "recordset-jsonl":
            snapshot.rows("artifact")
        else:
            snapshot.document("artifact")
    assert failure.value.error.code == "CHECKSUM_MISMATCH"


def test_structured_search_materializes_safe_defaults(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "query": "cat",
                "language": None,
                "filters": {},
                "pagination": {"limit": 20, "cursor": None},
                "sort": "relevance",
                "view": "agent",
            }
        ),
        encoding="utf-8",
    )
    _, _, filters, limit, cursor, sort, view = _structured_search(request)
    assert filters["availability"] == ["active"]
    assert filters["review"] == ["approved", "qualified-ai"]
    assert filters["rating"] == ["general"]
    assert filters["rights"] == ["allowed"]
    assert filters["require_no_warnings"] is True
    assert (limit, cursor, sort, view) == (20, None, "relevance", "agent")


def test_structured_search_rejects_version_alias(tmp_path: Path) -> None:
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "query": "cat",
                "language": None,
                "filters": {},
                "pagination": {"limit": 20, "cursor": None},
                "sort": "relevance",
                "view": "agent",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CommandError) as failure:
        _structured_search(request)
    assert failure.value.error.code == "REQUEST_SCHEMA_INVALID"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda body: body.update(query="x" * 4097),
        lambda body: body["filters"].update(style=["flat", "flat"]),
        lambda body: body["filters"].update(style=["Not-Kebab"]),
    ],
)
def test_structured_search_enforces_embedded_schema_constraints(
    tmp_path: Path,
    mutation,
) -> None:
    body: dict[str, Any] = {
        "schema_version": "1",
        "query": "cat",
        "language": None,
        "filters": {},
        "pagination": {"limit": 20, "cursor": None},
        "sort": "relevance",
        "view": "agent",
    }
    mutation(body)
    request = tmp_path / "request.json"
    request.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(CommandError) as failure:
        _structured_search(request)
    assert failure.value.error.code == "REQUEST_SCHEMA_INVALID"


def test_unsigned_search_is_diagnostic_but_safe_agent_view_is_empty() -> None:
    gate = LoadedSnapshot(
        root=Path("."),
        manifest_path=Path("manifest.json"),
        manifest={"snapshot_id": "data-2026.09.11.1", "trust_stage": "pre-enforcement"},
        manifest_sha256="a" * 64,
        descriptors={},
        verified_paths={},
    )
    with pytest.raises(CommandError) as unsigned:
        gate.require_diagnostic_opt_in(False)
    assert unsigned.value.error.code == "SIGNATURE_MISSING"

    snapshot = FakeSnapshot()
    reader = SnapshotReader(snapshot)  # type: ignore[arg-type]
    filters = _validate_filters({})
    diagnostic = reader.search(
        query="cat",
        language="en",
        filters=filters,
        sort="relevance",
        view="search",
        limit=20,
        cursor=None,
    )
    assert len(diagnostic["items"]) == 1
    safe = reader.search(
        query="cat",
        language="en",
        filters=filters,
        sort="relevance",
        view="agent",
        limit=20,
        cursor=None,
    )
    assert safe["items"] == []


def test_canonical_rights_are_authoritative_over_search_summary() -> None:
    snapshot = FakeSnapshot()
    search = snapshot._rows["search-en"][0]
    search["rights"] = {
        "rights_profile_id": "forged",
        "distribution_status": "allowed",
        "attribution_required": False,
    }
    reader = SnapshotReader(snapshot)  # type: ignore[arg-type]
    canonical = reader.get(
        str(search["emoji_id"]), view="canonical", language=None, include_sensitive=False
    )
    assert canonical["item"]["record"]["id"] == search["emoji_id"]
    with pytest.raises(CommandError) as failure:
        reader.get(
            str(search["emoji_id"]),
            view="search",
            language="en",
            include_sensitive=False,
        )
    assert failure.value.error.code == "INDEX_CORRUPT"


def test_collection_filter_excludes_inactive_requested_collection() -> None:
    snapshot = FakeSnapshot()
    inactive_id = "mxc_44444444-4444-5444-8444-444444444444"
    snapshot._rows["collections"] += (
        {
            **snapshot._rows["collections"][0],
            "id": inactive_id,
            "native_id": "ArchivedCats",
            "availability": {
                "status": "unavailable",
                "first_seen_at": "2026-09-10T18:00:00Z",
            },
        },
    )
    snapshot._rows["memberships"] += (
        {
            **snapshot._rows["memberships"][0],
            "id": "mxm_55555555-5555-5555-8555-555555555555",
            "collection_id": inactive_id,
        },
    )
    reader = SnapshotReader(snapshot)  # type: ignore[arg-type]
    filters = _validate_filters({"collection": [inactive_id]})
    result = reader.search(
        query="cat",
        language="en",
        filters=filters,
        sort="relevance",
        view="search",
        limit=20,
        cursor=None,
    )
    assert result["items"] == []


def test_resolve_preserves_large_native_identifier_as_string() -> None:
    reader = SnapshotReader(FakeSnapshot())  # type: ignore[arg-type]
    result = reader.resolve(
        platform="telegram",
        namespace="custom_emoji.id",
        scope="global",
        native_id="9007199254740993",
        identity_epoch=None,
        as_of=None,
        include_history=False,
        limit=20,
        cursor=None,
    )
    assert result["resolution_status"] == "current"
    assert result["candidate_count"] == 1
    assert isinstance(result["candidates"][0]["reference_id"], str)


def test_jsonl_item_ordinals_start_at_zero_and_are_contiguous(
    capsys: pytest.CaptureFixture[str],
) -> None:
    execute_read(
        "search",
        lambda: ReadCommandResult(
            result={"next_cursor": {"present": False}},
            dataset=FakeSnapshot().dataset_context(),
            arguments={},
            view="search",
            stream_items=[{"id": "first"}, {"id": "second"}],
        ),
        json_output=False,
        jsonl_output=True,
    )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["record_type"] for row in rows] == [
        "metadata",
        "item",
        "item",
        "summary",
    ]
    assert [rows[1]["ordinal"], rows[2]["ordinal"]] == [0, 1]
    assert rows[-1]["item_count"] == 2


def test_jsonl_failure_after_snapshot_selection_keeps_exact_request_context(
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = FakeSnapshot().dataset_context()
    arguments = {
        "query": "cat",
        "language": "en",
        "filters": _validate_filters({}),
        "sort": "relevance",
        "view": "search",
        "limit": 20,
        "cursor": None,
    }

    def fail() -> ReadCommandResult:
        raise CommandError("INDEX_CORRUPT", "broken index", hint="restore snapshot")

    with pytest.raises(typer.Exit):
        execute_read(
            "search",
            fail,
            json_output=False,
            jsonl_output=True,
            dataset_provider=lambda: dataset,
            request_context_provider=lambda: (arguments, "search"),
        )

    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["record_type"] for row in rows] == ["metadata", "summary"]
    assert rows[0]["view"] == "search"
    assert rows[0]["request_sha256"] == _request_sha256("search", dataset, arguments)
    assert rows[1]["ok"] is False
    validate_embedded_schema_instance(
        rows[0],
        "mlx://schemas/distribution/v1/cli-jsonl-metadata.schema.json",
        location="metadata",
        invalid_code="INTERNAL_ERROR",
    )
    validate_embedded_schema_instance(
        rows[1],
        "mlx://schemas/distribution/v1/cli-jsonl-summary.schema.json",
        location="summary",
        invalid_code="INTERNAL_ERROR",
    )


def test_canonical_output_fails_closed_instead_of_redacting_exact_content(
    capsys: pytest.CaptureFixture[str],
) -> None:
    credential_like_url = "https://api.telegram.org/file/bot123456:telegram-secret-value-value/path"
    dataset = FakeSnapshot().dataset_context()

    def canonical() -> ReadCommandResult:
        return ReadCommandResult(
            result={"record": {"source_url": credential_like_url}},
            dataset=dataset,
            arguments={"emoji_id": "mxe_11111111-1111-5111-8111-111111111111"},
            view="canonical",
        )

    with pytest.raises(typer.Exit):
        execute_read("get", canonical, json_output=True, dataset_provider=lambda: dataset)

    output = capsys.readouterr().out
    envelope = json.loads(output)
    assert envelope["ok"] is False
    assert envelope["status"] == "failed"
    assert "result" not in envelope
    assert envelope["errors"][0]["code"] == "CONTENT_POLICY_BLOCKED"
    assert credential_like_url not in output
    assert "<redacted>" not in output
    validate_embedded_schema_instance(
        envelope,
        "mlx://schemas/distribution/v1/cli-read-envelope.schema.json",
        location="envelope",
        invalid_code="INTERNAL_ERROR",
    )


def test_blocked_jsonl_result_does_not_emit_candidate_items(
    capsys: pytest.CaptureFixture[str],
) -> None:
    credential_like_url = "https://api.telegram.org/file/bot123456:telegram-secret-value-value/path"
    dataset = FakeSnapshot().dataset_context()

    def blocked() -> ReadCommandResult:
        return ReadCommandResult(
            result={"next_cursor": {"present": False}},
            dataset=dataset,
            arguments={"query": "cat"},
            view="search",
            stream_items=[{"description": credential_like_url}],
        )

    with pytest.raises(typer.Exit):
        execute_read(
            "search",
            blocked,
            json_output=False,
            jsonl_output=True,
            dataset_provider=lambda: dataset,
            request_context_provider=lambda: ({"query": "cat"}, "search"),
        )

    output = capsys.readouterr().out
    rows = [json.loads(line) for line in output.splitlines()]
    assert [row["record_type"] for row in rows] == ["metadata", "summary"]
    assert rows[1]["ok"] is False
    assert rows[1]["status"] == "failed"
    assert rows[1]["item_count"] == 0
    assert rows[1]["errors"][0]["code"] == "CONTENT_POLICY_BLOCKED"
    assert credential_like_url not in output
    validate_embedded_schema_instance(
        rows[0],
        "mlx://schemas/distribution/v1/cli-jsonl-metadata.schema.json",
        location="metadata",
        invalid_code="INTERNAL_ERROR",
    )
    validate_embedded_schema_instance(
        rows[1],
        "mlx://schemas/distribution/v1/cli-jsonl-summary.schema.json",
        location="summary",
        invalid_code="INTERNAL_ERROR",
    )
