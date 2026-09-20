from __future__ import annotations

import asyncio
from copy import deepcopy
from unittest.mock import patch

import pytest

from mojilex_cli.dataset import serialization as ser
from mojilex_cli.domain.models import Collection
from test_dataset_helpers import make_snapshot


def _legacy_jsonl(values, *, membership=False):
    canonical = [ser.canonical_entity(value) for value in values]
    canonical.sort(key=ser._membership_sort_key if membership else lambda value: value["id"])
    return ("".join(ser.compact_json(value) + "\n" for value in canonical)).encode("utf-8")


def test_all_entity_serializers_preserve_legacy_bytes(tmp_path):
    snapshot = make_snapshot(tmp_path)
    collection = next(iter(snapshot.collections.values()))
    emoji = next(iter(snapshot.emojis.values()))
    membership = next(iter(snapshot.memberships.values()))
    varied = emoji.as_dict()
    varied["id"] = "mxe_e\u0301"
    varied["semantic_tags"] = ["z", "a", "z"]
    varied["descriptions"]["ru"]["usage"] = ["e\u0301", "é", "last"]
    varied["extensions"]["extra"] = {"z": [None, True, 0, -0.0, 1.2], "e\u0301": "e\u0301"}
    memberships = [membership, {**membership.as_dict(), "position": 2, "id": "a"}]
    relations = [
        {
            "id": name,
            "entity_type": "visual_relation",
            "evidence": {
                "signals": ["z", "a", "z"],
                "media_pairs": [
                    {"subject_role": "z", "object_role": "primary"},
                    {"subject_role": "a", "object_role": "primary"},
                ],
            },
        }
        for name in ("z", "a")
    ]
    tombstone = {"entity_type": "tombstone", "target_id": "test", "public_note": "e\u0301"}
    for _ in range(2):
        with ser.serialization_scope():
            assert ser.serialize_collection(collection) == ser.pretty_json(
                ser.canonical_entity(collection)
            ).encode("utf-8")
            assert ser.serialize_tombstone(tombstone) == ser.pretty_json(
                ser.canonical_entity(tombstone)
            ).encode("utf-8")
            assert ser.serialize_emojis([varied, emoji]) == _legacy_jsonl([varied, emoji])
            assert ser.serialize_memberships(memberships) == _legacy_jsonl(
                memberships, membership=True
            )
            assert ser.serialize_visual_relations(relations) == _legacy_jsonl(relations)
            assert ser.serialize_emojis([]) == b""
            assert ser.serialize_memberships([]) == b""
            assert ser.serialize_visual_relations([]) == b""


def test_jsonl_arbitrary_mappings_preserve_normalization_and_order():
    values = [
        {"id": "z", "payload": {"e\u0301": ["e\u0301", {"en": "x", "ru": "y"}]}},
        {"id": "a", "payload": []},
    ]
    expected = "".join(ser.compact_json(value) + "\n" for value in reversed(values)).encode()
    assert ser.serialize_jsonl(values) == expected
    assert ser.serialize_jsonl([]) == b""


def test_exact_content_reuse_and_mutation_invalidation(tmp_path):
    emoji = next(iter(make_snapshot(tmp_path).emojis.values()))
    with patch.object(ser, "canonical_entity", wraps=ser.canonical_entity) as canonical:
        with ser.serialization_scope():
            first = ser.serialize_emojis([emoji])
            assert ser.serialize_emojis([emoji.model_copy(deep=True)]) == first
            assert canonical.call_count == 1
            emoji.descriptions["en"].usage.append("new usage")
            changed = ser.serialize_emojis([emoji])
            assert changed != first
            assert canonical.call_count == 2
            emoji.descriptions["en"].usage.pop()
            assert ser.serialize_emojis([emoji]) == first
            assert canonical.call_count == 2
            mutable = ser.canonical_entity(emoji)
            mutable["descriptions"]["en"]["usage"].append("unshared")
            assert ser.serialize_emojis([emoji]) == first


def test_scope_isolation_nesting_and_exception_cleanup(tmp_path):
    collection = next(iter(make_snapshot(tmp_path).collections.values()))
    with patch.object(ser, "canonical_entity", wraps=ser.canonical_entity) as canonical:
        with pytest.raises(RuntimeError, match="test"):
            with ser.serialization_scope():
                memo = ser._serialization_memo.get()
                ser.serialize_collection(collection)
                with ser.serialization_scope():
                    assert ser._serialization_memo.get() is memo
                    ser.serialize_collection(collection)
                assert canonical.call_count == 1
                raise RuntimeError("test")
        assert ser._serialization_memo.get() is None
        assert memo.closed and not memo.entries and memo.size == 0
        with ser.serialization_scope():
            ser.serialize_collection(collection)
        ser.serialize_collection(collection)
        assert canonical.call_count == 3


async def test_scope_shares_safe_immutable_bytes_with_to_thread(tmp_path):
    emoji = next(iter(make_snapshot(tmp_path).emojis.values()))
    with patch.object(ser, "canonical_entity", wraps=ser.canonical_entity) as canonical:
        with ser.serialization_scope():
            expected = ser.serialize_emojis([emoji])
            results = await asyncio.gather(
                *(
                    asyncio.to_thread(ser.serialize_emojis, [emoji.model_copy(deep=True)])
                    for _ in range(32)
                )
            )
            assert results == [expected] * 32
            assert canonical.call_count == 1


def test_mapping_fallback_does_not_cache_or_hide_invalid_keys(tmp_path, monkeypatch):
    collection = next(iter(make_snapshot(tmp_path).collections.values()))
    raw = collection.as_dict()
    raw["extensions"]["extra"] = {"1": "value"}
    monkeypatch.setattr(Collection, "as_dict", lambda self: deepcopy(raw))
    with ser.serialization_scope():
        ser.serialize_collection(collection)
        memo = ser._serialization_memo.get()
        assert len(memo.entries) == 1
        raw["extensions"]["extra"] = {1: "value"}
        with pytest.raises(TypeError, match="keys must be strings"):
            ser.serialize_collection(collection)
        raw["extensions"]["extra"] = {"é": 1, "e\u0301": 2}
        with pytest.raises(ValueError, match="collide after NFC"):
            ser.serialize_collection(collection)
        raw["extensions"]["extra"] = {"x": float("nan")}
        with pytest.raises(ValueError):
            ser.serialize_collection(collection)
        raw["extensions"]["extra"] = {"x": "valid"}
        ser.serialize_collection(raw)
        assert len(memo.entries) == 1


def test_bounded_lru_evicts_old_content_and_oversize_is_not_retained(tmp_path):
    collection = next(iter(make_snapshot(tmp_path).collections.values()))
    with ser.serialization_scope():
        ser.serialize_collection(collection)
        entry_size = ser._serialization_memo.get().size
    with ser.serialization_scope(max_bytes=entry_size):
        first = ser.serialize_collection(collection)
        collection.title = "Different Cats"
        second = ser.serialize_collection(collection)
        memo = ser._serialization_memo.get()
        assert second != first
        assert memo.size <= entry_size
        assert len(memo.entries) == 1
        collection.extensions["extra"] = {"large": "x" * 10000}
        ser.serialize_collection(collection)
        assert memo.size <= entry_size
        assert len(memo.entries) == 1
    with ser.serialization_scope(max_bytes=0):
        ser.serialize_collection(collection)
        assert not ser._serialization_memo.get().entries


def test_strict_native_key_guard_rejects_coercions():
    assert ser._is_json_native({"1": [1, True, None, -0.0]})
    assert not ser._is_json_native({1: "value"})
    assert not ser._is_json_native({"value": (1, 2)})
