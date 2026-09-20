"""Exact-byte parsing reuse preserves mutations, file checks and operation isolation."""

import os

import pytest

from mojilex_cli.dataset import repository, validation
from mojilex_cli.dataset.serialization import serialize_emojis
from mojilex_cli.domain.models import Emoji
from test_dataset_helpers import make_snapshot, write_fixture
from test_validation_compilation_cache import schema_fixture


def test_parsed_models_are_isolated_and_changed_bytes_revalidated(tmp_path, monkeypatch):
    snapshot = make_snapshot(tmp_path)
    data = serialize_emojis(snapshot.emojis.values())
    calls = []
    parse = repository.parse_jsonl

    def counted(*args, **kwargs):
        calls.append(args[0])
        return parse(*args, **kwargs)

    monkeypatch.setattr(repository, "parse_jsonl", counted)
    with repository.dataset_read_scope():
        first = repository._read_models(data, Emoji, source="one", many=True)
        expected = first[0].native_id
        first[0].native_id = "mutated"
        with repository.dataset_read_scope():
            second = repository._read_models(data, Emoji, source="two", many=True)
        assert second[0].native_id == expected
        assert len(calls) == 1
        repository._read_models(b" " + data, Emoji, source="three", many=True)
        assert len(calls) == 2
        for _ in range(2):
            with pytest.raises(ValueError):
                repository._read_models(b"{}\n", Emoji, source="bad", many=True)
        assert len(calls) == 4
    repository._read_models(data, Emoji, source="outside", many=True)
    assert len(calls) == 5
    assert repository._MODEL_MEMO.get() is None


def test_warm_load_reads_actual_bytes_even_with_same_size_and_mtime(tmp_path):
    write_fixture(tmp_path)
    with validation.schema_validation_scope():
        first = repository.load_dataset(tmp_path)
        path = next((tmp_path / "data" / "telegram" / "emojis").rglob("*.jsonl"))
        saved_stat = path.stat()
        original = path.read_bytes()
        path.write_bytes(b"!" + original[1:])
        os.utime(path, ns=(saved_stat.st_atime_ns, saved_stat.st_mtime_ns))
        with pytest.raises(repository.DatasetLoadError):
            repository.load_dataset(tmp_path)
        assert first.emojis


def test_model_cache_eviction_and_byte_bound(tmp_path, monkeypatch):
    data = serialize_emojis(make_snapshot(tmp_path).emojis.values())
    monkeypatch.setattr(repository, "_MODEL_CACHE_BYTES", len(data) + 1)
    with repository.dataset_read_scope():
        for suffix in (b"", b" ", b"  "):
            repository._read_models(suffix + data, Emoji, source="test", many=True)
            memo = repository._MODEL_MEMO.get()
            assert memo.size <= len(data) + 1
            assert len(memo.entries) <= 1


def test_schema_graph_reuses_parse_only_after_exact_byte_read(tmp_path, monkeypatch):
    snapshot, schema_path = schema_fixture(tmp_path)
    from mojilex_cli.dataset import serialization

    calls = []
    parse = serialization.parse_json

    def counted(*args, **kwargs):
        calls.append(args[0])
        return parse(*args, **kwargs)

    monkeypatch.setattr(serialization, "parse_json", counted)
    with validation.schema_validation_scope():
        root = snapshot.root / "schemas" / "v1"
        first = validation._schema_store(root)
        before = len(calls)
        assert before > 0
        second = validation._schema_store(root)
        assert first[0] is second[0]
        assert len(calls) == before
        schema_path.write_bytes(schema_path.read_bytes() + b"\n")
        third = validation._schema_store(root)
        assert third[2] != first[2]
        assert len(calls) > before


def test_read_rejects_outside_boundary_and_checks_guard_on_every_read(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    root.mkdir()
    path = root / "one.json"
    path.write_bytes(b"{}")
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    with pytest.raises(repository.DatasetLoadError, match="boundary"):
        repository._read(outside, root, {})
    real_guard = repository.assert_no_link_or_reparse
    calls = []

    def guard(path, *, boundary):
        calls.append(path)
        if len(calls) == 2:
            raise ValueError("link or reparse-point traversal is forbidden")
        real_guard(path, boundary=boundary)

    monkeypatch.setattr(repository, "assert_no_link_or_reparse", guard)
    assert repository._read(path, root, {}) == b"{}"
    with pytest.raises(repository.DatasetLoadError, match="reparse"):
        repository._read(path, root, {})
    assert len(calls) == 2
