from __future__ import annotations

import hashlib
import json

import pytest

from mojilex_cli.cache import AICacheWrite, CacheError, CacheStore
from test_cache_store import _result


def _seed(cache):
    original = cache.put_ai("canonical", _result(), generated_at="2026-01-01T00:00:00Z")
    # A valid row may have noncanonical JSON formatting. Retention must preserve
    # exactly what was stored, not reserialize the model and lose its old form.
    raw = json.dumps(original.model_dump(mode="json"), ensure_ascii=False, indent=2)
    cache._connection.execute(
        "UPDATE ai_cache SET payload_json = ?, created_at = 123, accessed_at = 456 "
        "WHERE cache_key = ?",
        (raw, "canonical"),
    )
    cache._connection.commit()
    archive_key = "rejected-ai-v1:" + hashlib.sha256(("canonical\0" + raw).encode()).hexdigest()
    return original, raw, archive_key


def _replacement(original, revision="new-context-valid"):
    return AICacheWrite(
        key="canonical",
        result=_result().model_copy(update={"model_revision": revision}),
        generated_at="2026-02-01T00:00:00Z",
        aliases=("run-alias",),
        expected_invalid_entry=original,
    )


def _rows(cache, table):
    # Table names are fixed test constants, never external data.
    return [tuple(row) for row in cache._connection.execute(f"SELECT * FROM {table} ORDER BY 1")]


def test_context_repair_archives_original_raw_payload_and_generation_timestamp(tmp_path):
    with CacheStore(tmp_path / "cache.sqlite3") as cache:
        original, raw, archive_key = _seed(cache)
        replacement = _replacement(original)
        result = cache.put_ai_batch(
            (replacement,),
            envelope_key="run-envelope",
            envelope={"cache_key": "canonical", "version": 2},
            repair_request_envelope=True,
        )["canonical"]
        assert result.result == replacement.result
        assert result.generated_at == "2026-02-01T00:00:00Z"
        archive = cache._connection.execute(
            "SELECT payload_json, created_at, accessed_at FROM ai_cache WHERE cache_key = ?",
            (archive_key,),
        ).fetchone()
        assert tuple(archive) == (raw, 123, 456)
        assert json.loads(archive["payload_json"])["generated_at"] == original.generated_at
        assert cache.get_ai_entry(archive_key)[1] == original
        assert cache.get_ai_entry("canonical")[1] == result
        assert cache.get_ai_entry("run-alias", follow_aliases=True) == ("canonical", result)
        assert cache.get_metadata("run-envelope") == {"cache_key": "canonical", "version": 2}


def test_stale_context_repair_cannot_replace_a_newer_valid_entry(tmp_path):
    path = tmp_path / "cache.sqlite3"
    with CacheStore(path) as first, CacheStore(path) as second:
        stale, _, archive_key = _seed(first)
        winner = second.put_ai_batch((_replacement(stale, "winner"),))["canonical"]
        loser = first.put_ai_batch((_replacement(stale, "stale-loser"),))["canonical"]
        assert loser == winner
        assert first.get_ai_entry("canonical")[1] == winner
        assert first.get_ai_entry(archive_key)[1] == stale
        assert first.info()["ai_entries"] == 2


@pytest.mark.parametrize("expected", ["none", "different_timestamp"])
def test_ordinary_or_nonmatching_write_keeps_original_qualification_time(tmp_path, expected):
    with CacheStore(tmp_path / "cache.sqlite3") as cache:
        original, _, archive_key = _seed(cache)
        invalid_entry = (
            None
            if expected == "none"
            else original.model_copy(update={"generated_at": "2026-01-02T00:00:00Z"})
        )
        candidate = _replacement(invalid_entry)
        assert cache.put_ai_batch((candidate,))["canonical"] == original
        assert cache.get_ai_entry(archive_key) is None
        assert cache.info()["ai_entries"] == 1


def test_context_repair_alias_envelope_and_archive_roll_back_together(tmp_path):
    with CacheStore(tmp_path / "cache.sqlite3") as cache:
        original, _, archive_key = _seed(cache)
        cache.put_ai_batch(
            (
                AICacheWrite(
                    key="other",
                    result=_result(),
                    generated_at="2026-01-03T00:00:00Z",
                    aliases=("run-alias",),
                ),
            ),
            envelope_key="run-envelope",
            envelope={"cache_key": "other", "version": 1},
        )
        tables = ("ai_cache", "ai_cache_alias", "metadata_cache")
        before = {table: _rows(cache, table) for table in tables}
        cache._connection.execute(
            """CREATE TRIGGER fail_repair_envelope BEFORE UPDATE ON metadata_cache
               WHEN NEW.cache_key = 'run-envelope'
               BEGIN SELECT RAISE(ABORT, 'synthetic envelope failure'); END"""
        )
        with pytest.raises(CacheError, match="atomically write"):
            cache.put_ai_batch(
                (_replacement(original),),
                envelope_key="run-envelope",
                envelope={"cache_key": "canonical", "version": 2},
                repair_request_envelope=True,
            )
        assert {table: _rows(cache, table) for table in tables} == before
        assert cache.get_ai_entry(archive_key) is None


@pytest.mark.parametrize("collision", ["different_payload", "different_created_at", "matching"])
def test_existing_archive_is_never_overwritten_on_collision(tmp_path, collision):
    with CacheStore(tmp_path / "cache.sqlite3") as cache:
        original, raw, archive_key = _seed(cache)
        archive_payload = "{}" if collision == "different_payload" else raw
        archive_created_at = 124 if collision == "different_created_at" else 123
        cache._connection.execute(
            "INSERT INTO ai_cache VALUES (?, ?, ?, ?)",
            (archive_key, archive_payload, archive_created_at, 789),
        )
        cache._connection.commit()
        before = _rows(cache, "ai_cache")
        if collision == "matching":
            updated = cache.put_ai_batch((_replacement(original),))["canonical"]
            assert updated.result.model_revision == "new-context-valid"
        else:
            with pytest.raises(CacheError, match="archive key collision"):
                cache.put_ai_batch((_replacement(original),))
            assert _rows(cache, "ai_cache") == before
        archived = cache._connection.execute(
            "SELECT payload_json, created_at, accessed_at FROM ai_cache WHERE cache_key = ?",
            (archive_key,),
        ).fetchone()
        assert tuple(archived) == (archive_payload, archive_created_at, 789)
