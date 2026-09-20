"""Local success stamps never authorize changed records, schemas or invalid stamps."""

import sqlite3

from mojilex_cli.dataset import schema_cache, validation
from test_schema_validation_scope import check, count_validation_calls
from test_validation_compilation_cache import schema_fixture


def test_persistent_successes_survive_new_scope_and_invalidate_inputs(tmp_path, monkeypatch):
    snapshot, schema_path = schema_fixture(tmp_path / "repo")
    cache = tmp_path / "cache"
    calls = count_validation_calls(monkeypatch)
    with validation.schema_validation_scope(persistent_cache=cache):
        assert not check(snapshot)
    before = len(calls)
    with validation.schema_validation_scope(persistent_cache=cache):
        assert not check(snapshot)
    assert len(calls) == before
    schema_path.write_bytes(b'{"required":["absent"]}')
    with validation.schema_validation_scope(persistent_cache=cache):
        assert any(issue.code == "SCHEMA" for issue in check(snapshot))
    assert len(calls) > before


def test_invalid_signature_and_changed_rules_force_revalidation(tmp_path, monkeypatch):
    snapshot, _ = schema_fixture(tmp_path / "repo")
    cache = tmp_path / "cache"
    calls = count_validation_calls(monkeypatch)
    with validation.schema_validation_scope(persistent_cache=cache):
        assert not check(snapshot)
    before = len(calls)
    with sqlite3.connect(cache / "successes.sqlite3") as database:
        database.execute("UPDATE successes SET signature=?", (b"x" * 32,))
    with validation.schema_validation_scope(persistent_cache=cache):
        assert not check(snapshot)
    assert len(calls) > before
    before = len(calls)
    monkeypatch.setattr(schema_cache, "_rules_digest", lambda: b"changed implementation")
    with validation.schema_validation_scope(persistent_cache=cache):
        assert not check(snapshot)
    assert len(calls) > before


def test_corrupt_database_or_key_falls_back_to_real_validation(tmp_path, monkeypatch):
    snapshot, _ = schema_fixture(tmp_path / "repo")
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "successes.sqlite3").write_bytes(b"corrupt sqlite database")
    calls = count_validation_calls(monkeypatch)
    with validation.schema_validation_scope(persistent_cache=cache):
        assert not check(snapshot)
    before = len(calls)
    (cache / "authentication.key").write_bytes(b"bad key")
    with validation.schema_validation_scope(persistent_cache=cache):
        assert not check(snapshot)
    assert len(calls) > before


def test_persistent_cache_still_checks_integrity_and_records(tmp_path):
    snapshot, _ = schema_fixture(tmp_path / "repo")
    cache = tmp_path / "cache"
    with validation.schema_validation_scope(persistent_cache=cache):
        assert validation.validate_snapshot(snapshot, schemas=True).valid
    next(iter(snapshot.collections.values())).item_count += 1
    with validation.schema_validation_scope(persistent_cache=cache):
        report = validation.validate_snapshot(snapshot, schemas=True)
        assert any(issue.code == "ITEM_COUNT" for issue in report.issues)


def test_persistent_cache_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(schema_cache, "_LIMIT", 2)
    cache = schema_cache.SchemaSuccessCache(tmp_path / "cache")
    cache.close({bytes([number]) * 32 for number in range(4)})
    reopened = schema_cache.SchemaSuccessCache(tmp_path / "cache")
    assert len(reopened.loaded) == 2
    reopened.close(())


def test_missing_key_and_unwritable_cache_do_not_skip_validation(tmp_path, monkeypatch):
    snapshot, _ = schema_fixture(tmp_path / "repo")
    root = tmp_path / "cache"
    calls = count_validation_calls(monkeypatch)
    with validation.schema_validation_scope(persistent_cache=root):
        assert not check(snapshot)
    before = len(calls)
    (root / "authentication.key").unlink()
    with validation.schema_validation_scope(persistent_cache=root):
        assert not check(snapshot)
    assert len(calls) > before
    before = len(calls)

    def denied(*args, **kwargs):
        raise PermissionError("cache is read-only")

    monkeypatch.setattr(schema_cache.os, "open", denied)
    with validation.schema_validation_scope(persistent_cache=root):
        assert not check(snapshot)
    assert len(calls) > before


def test_changed_record_cannot_reuse_persisted_success(tmp_path):
    import json

    snapshot, schema_path = schema_fixture(tmp_path / "repo")
    root = tmp_path / "cache"
    emoji = next(iter(snapshot.emojis.values()))
    schema_path.write_text(json.dumps({"properties": {"native_id": {"const": emoji.native_id}}}))
    with validation.schema_validation_scope(persistent_cache=root):
        assert not check(snapshot)
    emoji.native_id = "changed"
    with validation.schema_validation_scope(persistent_cache=root):
        assert any(issue.code == "SCHEMA" for issue in check(snapshot))
