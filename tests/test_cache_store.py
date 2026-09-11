import sqlite3
from pathlib import Path

import pytest

from mojilex_cli.ai import (
    AIUsage,
    DescriptionBatch,
    DescriptionResult,
    VisionContext,
)
from mojilex_cli.cache import (
    AICacheWrite,
    CacheError,
    CacheStore,
    ai_cache_key,
    canonical_context_hash,
    media_digest,
)


def _result() -> DescriptionResult:
    localized = {
        "text": "A synthetic smiling face.",
        "motion_status": "not_applicable",
        "usage": ["agreement"],
    }
    payload = {
        "items": [
            {
                "label": "E001",
                "descriptions": {"ru": localized, "en": localized},
                "facets": {
                    "text_content": {"status": "none", "dynamics": "stable", "items": []},
                    "content_types": ["reaction"],
                    "styles": ["flat"],
                    "suggested_uses": ["message-accent"],
                    "uncertainties": [],
                },
                "semantic_tags": ["face", "smile"],
                "content": {"rating": "general", "warnings": []},
            }
        ]
    }
    return DescriptionResult(
        batch=DescriptionBatch.model_validate(payload),
        provider="gemini",
        model="gemini-test",
        usage=AIUsage(input_tokens=1, output_tokens=2),
    )


def test_cache_roundtrip_contains_structured_results_but_no_media(tmp_path: Path) -> None:
    path = tmp_path / "cache.sqlite3"
    with CacheStore(path) as cache:
        cache.put_metadata(
            "lookup", {"emoji_id": "mxe_x", "bucket": "aa/bb.jsonl"}, base_sha="a" * 40
        )
        assert cache.get_metadata("lookup", base_sha="a" * 40)["emoji_id"] == "mxe_x"
        cache.put_ai("k", _result())
        assert cache.get_ai("k") == _result()
        with pytest.raises(CacheError, match="unsafe field"):
            cache.put_metadata("bad", {"file_id": "transient"})
        with pytest.raises(CacheError, match="binary"):
            cache.put_metadata("bad2", {"payload": b"media"})


def test_ai_batch_rows_aliases_and_envelope_roll_back_together(tmp_path: Path) -> None:
    with CacheStore(tmp_path / "cache.sqlite3") as cache:
        cache._connection.execute(
            """CREATE TRIGGER fail_second_ai_row
               BEFORE INSERT ON ai_cache
               WHEN NEW.cache_key = 'second'
               BEGIN SELECT RAISE(ABORT, 'forced batch failure'); END"""
        )
        with pytest.raises(CacheError, match="atomically write"):
            cache.put_ai_batch(
                (
                    AICacheWrite(
                        key="first",
                        result=_result(),
                        generated_at="2026-01-01T00:00:00Z",
                        aliases=("alias-first",),
                    ),
                    AICacheWrite(
                        key="second",
                        result=_result(),
                        generated_at="2026-01-01T00:00:00Z",
                        aliases=("alias-second",),
                    ),
                ),
                envelope_key="ai-request-envelope-v1:test",
                envelope={"format_version": 1, "items": []},
            )

        assert cache._connection.execute("SELECT COUNT(*) FROM ai_cache").fetchone()[0] == 0
        assert cache._connection.execute("SELECT COUNT(*) FROM ai_cache_alias").fetchone()[0] == 0
        assert cache.get_metadata("ai-request-envelope-v1:test") is None


def test_ai_envelope_repair_moves_aliases_atomically_or_rolls_back(tmp_path: Path) -> None:
    envelope_key = "ai-request-envelope-v1:repair"
    old_envelope = {"format_version": 1, "items": [{"cache_key": "old"}]}
    new_envelope = {
        "format_version": 1,
        "items": [{"cache_key": "new-first"}, {"cache_key": "new-second"}],
    }
    with CacheStore(tmp_path / "cache.sqlite3") as cache:
        cache.put_ai_batch(
            (
                AICacheWrite(
                    key="old",
                    result=_result(),
                    generated_at="2026-01-01T00:00:00Z",
                    aliases=("run-alias-first",),
                ),
            ),
            envelope_key=envelope_key,
            envelope=old_envelope,
        )
        cache._connection.execute(
            """CREATE TRIGGER fail_repair_second_row
               BEFORE INSERT ON ai_cache
               WHEN NEW.cache_key = 'new-second'
               BEGIN SELECT RAISE(ABORT, 'forced repair failure'); END"""
        )
        writes = (
            AICacheWrite(
                key="new-first",
                result=_result(),
                generated_at="2026-01-02T00:00:00Z",
                aliases=("run-alias-first",),
            ),
            AICacheWrite(
                key="new-second",
                result=_result(),
                generated_at="2026-01-02T00:00:00Z",
                aliases=("run-alias-second",),
            ),
        )
        with pytest.raises(CacheError, match="atomically write"):
            cache.put_ai_batch(
                writes,
                envelope_key=envelope_key,
                envelope=new_envelope,
                repair_request_envelope=True,
            )

        alias_hit = cache.get_ai_entry("run-alias-first", follow_aliases=True)
        assert alias_hit is not None and alias_hit[0] == "old"
        assert cache.get_ai("new-first") is None
        assert cache.get_ai("new-second") is None
        assert cache.get_metadata(envelope_key) == old_envelope

        cache._connection.execute("DROP TRIGGER fail_repair_second_row")
        cache.put_ai_batch(
            writes,
            envelope_key=envelope_key,
            envelope=new_envelope,
            repair_request_envelope=True,
        )
        first_hit = cache.get_ai_entry("run-alias-first", follow_aliases=True)
        second_hit = cache.get_ai_entry("run-alias-second", follow_aliases=True)
        assert first_hit is not None and first_hit[0] == "new-first"
        assert second_hit is not None and second_hit[0] == "new-second"
        assert cache.get_metadata(envelope_key) == new_envelope


def test_cache_keys_are_order_independent_and_context_limited() -> None:
    media_a = [
        {"role": "primary", "variant_id": "dark", "sha256": "b" * 64},
        {"role": "primary", "variant_id": "light", "sha256": "a" * 64},
    ]
    digest = media_digest(media_a)
    assert digest == media_digest(list(reversed(media_a)))
    context_hash = canonical_context_hash(
        VisionContext(
            fallback_emoji="🙂",
            needs_repainting=True,
            frame_count=8,
            background_variants=("light", "dark"),
        )
    )
    key = ai_cache_key(
        media_digest_value=digest,
        provider="gemini",
        model="model",
        model_revision=None,
        prompt_version="1.0.0",
        prompt_sha256="c" * 64,
        schema_version="1.0.0",
        pipeline_version="1.0.0",
        languages=("en", "ru"),
        canonical_context_hash_value=context_hash,
        description_profile="standard-v1",
        taxonomy_version="1.0.0",
        routing_policy_version="1.0.0",
        shown_media_sha256=("d" * 64,),
        request_parameters_sha256="e" * 64,
    )
    assert len(key) == 64
    with pytest.raises(CacheError, match="non-approved"):
        canonical_context_hash({"fallback_emoji": "🙂", "pack_title": "untrusted"})


def test_cache_must_be_outside_repository_and_read_only_blocks_writes(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(CacheError, match="outside"):
        CacheStore(repo / "cache.sqlite3", repository_root=repo)

    path = tmp_path / "cache.sqlite3"
    with CacheStore(path):
        pass
    with CacheStore(path, read_only=True) as cache:
        with pytest.raises(CacheError, match="read-only"):
            cache.put_metadata("x", {"safe": True})


def test_v1_ai_payload_is_ignored_then_safely_upgraded_with_immutable_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-cache.sqlite3"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE ai_cache (
            cache_key TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            accessed_at INTEGER NOT NULL
        ) WITHOUT ROWID;
        PRAGMA user_version=1;
        """
    )
    connection.execute(
        "INSERT INTO ai_cache(cache_key, payload_json, created_at, accessed_at) "
        "VALUES (?, ?, 1, 1)",
        ("actual", _result().model_dump_json()),
    )
    connection.commit()
    connection.close()

    with CacheStore(path) as cache:
        # A v1 entry has no trustworthy generation instant, so reusing it for
        # qualification would be unsafe.
        assert cache.get_ai("actual") is None
        first = cache.put_ai(
            "actual",
            _result(),
            generated_at="2026-01-01T00:00:00Z",
            aliases=("request-without-revision",),
        )
        assert first.generated_at == "2026-01-01T00:00:00Z"
        resolved = cache.get_ai_entry("request-without-revision", follow_aliases=True)
        assert resolved is not None
        assert resolved[0] == "actual"
        assert resolved[1] == first

        # Re-inserting an identical deterministic key must not restamp it.
        second = cache.put_ai(
            "actual",
            _result(),
            generated_at="2026-12-31T23:59:59Z",
        )
        assert second.generated_at == first.generated_at

    connection = sqlite3.connect(path)
    assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
    connection.close()
