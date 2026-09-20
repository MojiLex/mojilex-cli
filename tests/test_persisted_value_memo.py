from __future__ import annotations

import asyncio
import re
from copy import deepcopy
from unittest.mock import patch

import pytest

from mojilex_cli.dataset import validation as val
from mojilex_cli.dataset.repository import DatasetSnapshot


def check(tmp_path, value):
    issues = []
    val._validate_persisted_values(DatasetSnapshot(tmp_path, value), issues)
    return issues


def test_reuses_only_exact_success_and_detects_nested_changes(tmp_path):
    value = {"nested": [{"text": "safe"}]}
    with patch.object(val, "_walk_persisted_value", wraps=val._walk_persisted_value) as walk:
        with val.schema_validation_scope():
            assert not check(tmp_path, value)
            initial = walk.call_count
            assert not check(tmp_path, deepcopy(value))
            assert walk.call_count == initial
            value["nested"][0]["text"] = "e\u0301"
            first = check(tmp_path, value)
            assert any(issue.code == "NFC" for issue in first)
            invalid_count = walk.call_count
            assert check(tmp_path, value) == first
            assert walk.call_count > invalid_count
            value["nested"][0]["text"] = "safe"
            prior = walk.call_count
            assert not check(tmp_path, value)
            assert walk.call_count == prior


def test_native_key_and_value_types_cannot_alias_success(tmp_path):
    with val.schema_validation_scope():
        assert not check(tmp_path, {"1": "safe"})
        assert any(issue.code == "JSON_KEY" for issue in check(tmp_path, {1: "safe"}))
        memo = val._SCHEMA_MEMO.get()
        assert memo is not None
        saved = len(memo.persisted)
        assert not check(tmp_path, {"tuple": ("safe",)})
        assert len(memo.persisted) == saved
        assert not check(tmp_path, {"number": float("nan")})
        assert len(memo.persisted) == saved


def test_rules_changes_invalidate_success(tmp_path, monkeypatch):
    value = {"note": "synthetic-sensitive-marker"}
    with val.schema_validation_scope():
        assert not check(tmp_path, value)
        monkeypatch.setitem(val._SECRET_PATTERNS, "test_pattern", re.compile(b"sensitive-marker"))
        assert any(issue.code == "SECRET" for issue in check(tmp_path, value))
        monkeypatch.delitem(val._SECRET_PATTERNS, "test_pattern")
        monkeypatch.setattr(val, "_FORBIDDEN_PERSISTED_KEYS", {"note"})
        assert any(issue.code == "FORBIDDEN_FIELD" for issue in check(tmp_path, value))


def test_failed_value_preserves_path_at_each_location(tmp_path):
    with val.schema_validation_scope():
        first = check(tmp_path, {"first": {"api_key": "placeholder"}})
        second = check(tmp_path, {"second": {"api_key": "placeholder"}})
    assert first[0].path == "dataset.json/first/api_key"
    assert second[0].path == "dataset.json/second/api_key"


def test_scope_cleanup_nested_scope_and_new_scope(tmp_path):
    with patch.object(val, "_walk_persisted_value", wraps=val._walk_persisted_value) as walk:
        with pytest.raises(RuntimeError, match="test"):
            with val.schema_validation_scope():
                memo = val._SCHEMA_MEMO.get()
                assert not check(tmp_path, {"text": "safe"})
                count = walk.call_count
                with val.schema_validation_scope():
                    assert val._SCHEMA_MEMO.get() is memo
                    assert not check(tmp_path, {"text": "safe"})
                    assert walk.call_count == count
                raise RuntimeError("test")
        assert val._SCHEMA_MEMO.get() is None
        assert memo is not None and not memo.persisted and memo.persisted_bytes == 0
        with val.schema_validation_scope():
            assert not check(tmp_path, {"text": "safe"})
        assert walk.call_count > count


def test_byte_and_entry_limits_do_not_change_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(val, "_PERSISTED_MEMO_BYTES", 45)
    monkeypatch.setattr(val, "_PERSISTED_MEMO_ENTRIES", 2)
    with val.schema_validation_scope():
        memo = val._SCHEMA_MEMO.get()
        assert memo is not None
        for i in range(20):
            assert not check(tmp_path, {"text": str(i)})
            assert len(memo.persisted) <= 2
            assert memo.persisted_bytes <= 45
        size = memo.persisted_bytes
        assert not check(tmp_path, {"text": "x" * 50})
        assert memo.persisted_bytes == size
        assert any(issue.code == "NFC" for issue in check(tmp_path, {"text": "x" * 50 + "e\u0301"}))


async def test_to_thread_shares_bounded_exact_cache(tmp_path):
    with val.schema_validation_scope():
        values = [{"nested": ["safe", index]} for index in range(8)]
        results = await asyncio.gather(
            *(asyncio.to_thread(check, tmp_path, value) for value in values * 4)
        )
        assert not any(results)
        memo = val._SCHEMA_MEMO.get()
        assert memo is not None
        assert len(memo.persisted) == 8
        assert memo.persisted_bytes == sum(len(key[1]) for key in memo.persisted)
