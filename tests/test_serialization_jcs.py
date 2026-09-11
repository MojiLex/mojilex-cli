import hashlib

import pytest

from mojilex_cli.dataset.serialization import parse_json, parse_jsonl, pretty_json, serialize_emojis
from mojilex_cli.domain.hashes import jcs_bytes, jcs_sha256, media_digest, telegram_set_fingerprint
from test_dataset_helpers import make_snapshot


def test_rfc8785_number_vector_and_hash() -> None:
    value = {"numbers": [333333333.33333329, 1e30, 4.5, 2e-3, 1e-27]}
    expected = b'{"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27]}'
    assert jcs_bytes(value) == expected
    assert jcs_sha256(value) == hashlib.sha256(expected).hexdigest()


def test_normative_telegram_fingerprint_vector() -> None:
    assert (
        telegram_set_fingerprint([("5368324170671202286", "AgADExampleUniqueId")])
        == "35a70b19472747a656a7f5f8f44ec8cd22a31e18e82d122381fa738f78d618d1"
    )


def test_media_digest_ignores_input_order_and_non_identity_metadata(tmp_path) -> None:
    emoji = next(iter(make_snapshot(tmp_path).emojis.values()))
    first = emoji.media[0].model_copy(deep=True)
    second = first.model_copy(update={"role": "dark", "variant_id": "contrast"})
    digest = media_digest([second, first])
    second.byte_size += 1
    assert media_digest([first, second]) == digest


def test_jsonl_is_compact_sorted_and_has_exactly_one_final_lf(tmp_path) -> None:
    emoji = next(iter(make_snapshot(tmp_path).emojis.values()))
    data = serialize_emojis([emoji])
    assert data.endswith(b"\n") and not data.endswith(b"\n\n")
    assert b"\r" not in data
    assert b'"semantic_tags":["cat","doubt","suspicious"]' in data
    assert data.index(b'"fingerprints"') < data.index(b'"descriptions"')
    assert data.index(b'"descriptions"') < data.index(b'"facets"')


def test_parser_rejects_duplicate_keys_bom_crlf_and_missing_final_lf() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_json(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="BOM"):
        parse_json(b"\xef\xbb\xbf{}")
    with pytest.raises(ValueError, match="final LF"):
        parse_jsonl(b"{}")
    with pytest.raises(ValueError, match="CR"):
        parse_jsonl(b"{}\r\n")
    with pytest.raises(ValueError, match="non-I-JSON"):
        parse_json(b'{"value":NaN}')


def test_serializer_rejects_non_string_and_nfc_colliding_object_keys() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        pretty_json({1: "value"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="collide after NFC"):
        pretty_json({"é": 1, "e\u0301": 2})


def test_release_and_provenance_contexts_have_locked_canonical_order() -> None:
    manifest = pretty_json(
        {
            "dataset": "mojilex",
            "schema_version": "1.0.0",
            "git_commit": "0" * 40,
            "counts": {},
            "status_counts": {},
            "profiles": {},
            "quality_registry_sha256": {},
            "platform_registry_sha256": {},
            "payload_sha256": {},
        }
    )
    assert manifest.index('"profiles"') < manifest.index('"quality_registry_sha256"')
    assert manifest.index('"platform_registry_sha256"') < manifest.index('"payload_sha256"')

    provenance = pretty_json(
        {
            "origin": "ai",
            "provider": "google",
            "model": "gemini",
            "prompt_version": "1.0.0",
            "pipeline_version": "1.0.0",
            "description_profile": "standard-v1",
            "model_revision": "revision",
            "tool": {"name": "mojilex-cli", "version": "0.1.0"},
        }
    )
    assert provenance.index('"pipeline_version"') < provenance.index('"description_profile"')
    assert provenance.index('"description_profile"') < provenance.index('"model_revision"')


def test_release_taxonomy_and_duplicate_group_maps_have_locked_order() -> None:
    taxonomy = pretty_json(
        {
            "taxonomy_version": "1.0.0",
            "registries": {
                "uncertainties": [],
                "styles": [],
                "color_families": [],
                "content_types": [],
            },
        }
    )
    assert taxonomy.index('"color_families"') < taxonomy.index('"content_types"')
    assert taxonomy.index('"content_types"') < taxonomy.index('"styles"')
    assert taxonomy.index('"styles"') < taxonomy.index('"uncertainties"')

    group = pretty_json(
        {
            "members": ["mxe_a", "mxe_b"],
            "content_digest": "0" * 64,
            "profile": "dedupe-v1",
            "scope": "entity",
            "group_type": "decoded-exact",
            "id": "mxdg_example",
        }
    )
    assert group.index('"scope"') < group.index('"profile"')
    assert group.index('"profile"') < group.index('"content_digest"')
    assert group.index('"content_digest"') < group.index('"members"')
