from __future__ import annotations

import asyncio
import io
import json
from contextlib import contextmanager
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest
from PIL import Image

from media_backend_helpers import require_native_media_limits
from mojilex_cli.analysis import DeterministicMediaAnalysis
from mojilex_cli.commands import dedupe as command_module
from mojilex_cli.commands import dedupe_backfill as module
from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.config import MojiLexConfig
from mojilex_cli.dataset import DatasetLoadError, load_dataset, validate_snapshot
from mojilex_cli.dataset.transaction import AtomicWriteError
from mojilex_cli.dedupe.store import DedupeIndexError
from mojilex_cli.domain import Emoji, Media, emoji_id, media_digest, telegram_set_fingerprint
from mojilex_cli.media import MediaProcessor, SourceChangedDuringRunError, TemporaryMediaRun
from mojilex_cli.sources import SourceEmoji
from test_dataset_helpers import make_snapshot, write_fixture


def _incomplete(snapshot):
    emoji = next(iter(snapshot.emojis.values()))
    payload = emoji.as_dict()
    payload["fingerprints"].update(status="partial", items=[])
    snapshot.emojis[emoji.id] = Emoji.model_validate(payload)
    return emoji


def _fake(monkeypatch, emoji, *, missing=False, changed=False):
    class Adapter:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def validate_credentials(self):
            pass

        async def fetch_emojis(self, ids):
            assert ids == (emoji.native_id,)
            return (
                {}
                if missing
                else {
                    emoji.native_id: SimpleNamespace(
                        media_format="webp",
                        declared_file_size=128,
                        needs_repainting=False,
                    )
                }
            )

        async def fetch_media(self, item):
            yield b"test fixture only"

    analysis = DeterministicMediaAnalysis(
        color_profile_sha256=module_profile(emoji, "color"),
        dedupe_profile_sha256=module_profile(emoji, "dedupe"),
        decoder_backend_fingerprint="a" * 64,
        rendering=emoji.facets.rendering.items[0].model_dump(
            mode="json", exclude={"role", "variant_id"}
        ),
        fingerprint=emoji.fingerprints.items[0].model_dump(
            mode="json", exclude={"role", "variant_id"}
        ),
    )

    class Processor:
        def __init__(self, temporary):
            pass

        async def process_stream(self, stream, **kwargs):
            assert kwargs["expected_sha256"] == emoji.media[0].sha256
            metadata = emoji.media[0].as_dict()
            if changed:
                metadata["sha256"] = "f" * 64
            return SimpleNamespace(analysis=analysis, dataset_metadata=lambda: metadata)

    monkeypatch.setattr(module, "TelegramBotAPI", Adapter)
    monkeypatch.setattr(module, "MediaProcessor", Processor)
    monkeypatch.setattr(
        module, "load_credentials", lambda: SimpleNamespace(telegram_bot_token="fixture-only")
    )


def module_profile(emoji, kind):
    from mojilex_cli.analysis import profile_sha256

    return profile_sha256(f"{kind}-v1")


def test_legacy_load_is_explicit_and_preserves_original_bytes(tmp_path):
    write_fixture(tmp_path)
    path = next((tmp_path / "data").glob("*/emojis/*/*.jsonl"))
    payload = json.loads(path.read_text())
    payload.pop("fingerprints")
    raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
    path.write_bytes(raw)
    with pytest.raises(DatasetLoadError):
        load_dataset(tmp_path)
    loaded = load_dataset(tmp_path, allow_missing_fingerprints=True)
    assert next(iter(loaded.emojis.values())).fingerprints.status.value == "partial"
    assert path.read_bytes() == raw


def test_backfill_uses_verified_media_without_ai_or_semantic_changes(tmp_path, monkeypatch):
    snapshot = make_snapshot(tmp_path)
    original = _incomplete(snapshot)
    _fake(monkeypatch, original)
    before = snapshot.to_files()
    after, ids = asyncio.run(
        module.backfill_fingerprints(snapshot, snapshot.emojis, MojiLexConfig())
    )
    assert ids == (original.id,)
    assert after.emojis[original.id].as_dict() == original.as_dict()
    assert snapshot.to_files() == before


@pytest.mark.parametrize("active", [True, False])
def test_missing_media_never_fabricates_active_fingerprints(tmp_path, monkeypatch, active):
    snapshot = make_snapshot(tmp_path)
    original = _incomplete(snapshot)
    if not active:
        snapshot.emojis[original.id].availability.status = "unavailable"
    _fake(monkeypatch, original, missing=True)
    if active:
        with pytest.raises(CommandError, match="Active legacy"):
            asyncio.run(module.backfill_fingerprints(snapshot, snapshot.emojis, MojiLexConfig()))
    else:
        after, _ = asyncio.run(
            module.backfill_fingerprints(snapshot, snapshot.emojis, MojiLexConfig())
        )
        assert after.emojis[original.id].fingerprints.status.value == "unavailable"
        assert after.emojis[original.id].fingerprints.items == []
    assert snapshot.emojis[original.id].fingerprints.status.value == "partial"


def test_changed_media_fails_before_any_canonical_write(tmp_path, monkeypatch):
    snapshot = make_snapshot(tmp_path)
    original = _incomplete(snapshot)
    _fake(monkeypatch, original, changed=True)
    before = snapshot.to_files()
    with pytest.raises(CommandError, match="metadata changed"):
        asyncio.run(module.backfill_fingerprints(snapshot, snapshot.emojis, MojiLexConfig()))
    assert snapshot.to_files() == before


def test_complete_fingerprint_scan_does_not_need_credentials(tmp_path, monkeypatch):
    snapshot = make_snapshot(tmp_path)
    monkeypatch.setattr(
        module, "load_credentials", lambda: pytest.fail("must not load credentials")
    )
    after, ids = asyncio.run(
        module.backfill_fingerprints(snapshot, snapshot.emojis, MojiLexConfig())
    )
    assert after is snapshot and ids == ()


@pytest.fixture(scope="module")
def verified_media():
    """Real WebP bytes decoded and analyzed by the sandbox, never a fabricated hash."""
    require_native_media_limits()
    buffer = io.BytesIO()
    with Image.new("RGBA", (32, 32), (240, 30, 60, 255)) as picture:
        picture.save(buffer, format="WEBP", lossless=True)
    blob = buffer.getvalue()

    async def prepare():
        async def chunks():
            yield blob

        with TemporaryMediaRun() as temporary:
            processed = await MediaProcessor(temporary).process_stream(
                chunks(), expected_format="webp", declared_size=len(blob)
            )
            assert processed.analysis is not None
            return processed.dataset_metadata(), processed.analysis

    metadata, analysis = asyncio.run(prepare())
    return blob, metadata, analysis


def _verified_legacy(root, verified_media, *, count=1):
    snapshot = write_fixture(root)
    _, metadata, analysis = verified_media
    original = next(iter(snapshot.emojis.values()))
    payload = original.as_dict()
    payload["media"] = [metadata]
    payload["facets"]["rendering"]["items"] = [
        {"role": "primary", **analysis.rendering.model_dump(mode="json", exclude_none=True)}
    ]
    payload["fingerprints"] = {
        "status": "complete",
        "profile": analysis.dedupe_profile,
        "input_media_digest": media_digest([Media.model_validate(metadata)]),
        "items": [{"role": "primary", **analysis.fingerprint.model_dump(mode="json")}],
    }
    payload["provenance"]["input_media_sha256"] = [metadata["sha256"]]
    snapshot.emojis[original.id] = Emoji.model_validate(payload)
    for offset in range(1, count):
        native = str(int(original.native_id) + offset)
        payload["id"] = emoji_id("telegram", "custom_emoji.id", "global", native)
        payload["native_id"] = native
        payload["extensions"]["telegram"]["custom_emoji_id"] = native
        payload["extensions"]["telegram"]["file_unique_id"] = f"FixtureUnique{offset}"
        snapshot.emojis[payload["id"]] = Emoji.model_validate(payload)
    for path, data in snapshot.to_files().items():
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if "emojis" in path.parts:
            records = [json.loads(line) for line in data.decode().splitlines()]
            for record in records:
                record.pop("fingerprints")
            data = ("\n".join(json.dumps(record) for record in records) + "\n").encode()
        destination.write_bytes(data)
    return snapshot


def _wire_command(monkeypatch, snapshot, blob, *, temporary=False, corrupt_ids=()):
    from mojilex_cli.ai.gemini import GeminiVisionProvider
    from mojilex_cli.ai.registry import ProviderRegistry

    def forbid_ai(*args, **kwargs):
        pytest.fail("dedupe backfill must not create or invoke an AI provider")

    monkeypatch.setattr(GeminiVisionProvider, "__init__", forbid_ai)
    monkeypatch.setattr(ProviderRegistry, "create", forbid_ai)
    root = snapshot.root
    config = MojiLexConfig(
        repository={"target": str(root)},
        cache_dir=root.parent / "cache",
        ai={"max_ai_requests": 0},
    )

    @contextmanager
    def workspace(*args, **kwargs):
        yield SimpleNamespace(root=root, temporary=temporary)

    downloads = []

    class Adapter:
        def __init__(self, token, **kwargs):
            assert token == "fixture-only"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def validate_credentials(self):
            pass

        async def fetch_emojis(self, ids):
            return {
                emoji.native_id: SourceEmoji(
                    native_namespace=emoji.native_namespace,
                    scope_id=emoji.scope_id,
                    native_id=emoji.native_id,
                    file_unique_id=emoji.extensions["telegram"]["file_unique_id"],
                    position=0,
                    width=32,
                    height=32,
                    animated=False,
                    video=False,
                    media_format="webp",
                    declared_file_size=len(blob),
                    file_id="transient-fixture-only",
                )
                for emoji in snapshot.emojis.values()
                if emoji.native_id in ids
            }

        async def fetch_media(self, item):
            downloads.append(item.native_id)
            data = blob if item.native_id not in corrupt_ids else b"x" + blob[1:]
            yield data[:16]
            yield data[16:]

    monkeypatch.setattr(command_module, "load_config", lambda **kwargs: config)
    monkeypatch.setattr(command_module, "repository_workspace", workspace)
    monkeypatch.setattr(module, "TelegramBotAPI", Adapter)
    monkeypatch.setattr(
        module, "load_credentials", lambda: SimpleNamespace(telegram_bot_token="fixture-only")
    )
    return downloads, config.cache_dir / "dedupe-index-v1.sqlite3"


def _scan(selector=None):
    return command_module.dedupe_scan_command(
        selector, all_items=selector is None, repo=None, max_candidates=None, profile=None
    )


def _canonical_bytes(root):
    return load_dataset(root, allow_missing_fingerprints=True).source_bytes


def test_command_verified_source_backfill_is_canonical_ai_free_and_idempotent(
    tmp_path, monkeypatch, verified_media
):
    snapshot = _verified_legacy(tmp_path / "repo", verified_media)
    downloads, index_path = _wire_command(monkeypatch, snapshot, verified_media[0])
    result = _scan().result
    loaded = load_dataset(snapshot.root)
    validate_snapshot(loaded, canonical=True, repository_files=True).raise_for_errors()
    assert loaded.to_files() == snapshot.to_files()
    assert result["canonical_writes"] == 1
    assert result["backfilled_emoji_ids"] == sorted(snapshot.emojis)
    assert result["selected_emoji_ids"] == sorted(snapshot.emojis)
    assert len(downloads) == 1 and index_path.is_file()
    assert not any(path.suffix in {".webp", ".png"} for path in snapshot.root.rglob("*"))
    before = _canonical_bytes(snapshot.root)
    monkeypatch.setattr(
        module, "load_credentials", lambda: pytest.fail("complete scan needs no source or AI")
    )
    repeated = _scan().result
    assert repeated["canonical_writes"] == 0
    assert repeated["backfilled_emoji_ids"] == []
    assert _canonical_bytes(snapshot.root) == before


def test_command_second_source_hash_failure_keeps_all_original_bytes(
    tmp_path, monkeypatch, verified_media
):
    snapshot = _verified_legacy(tmp_path / "repo", verified_media, count=2)
    identifiers = sorted(snapshot.emojis)
    corrupt_native = snapshot.emojis[identifiers[1]].native_id
    downloads, index_path = _wire_command(
        monkeypatch, snapshot, verified_media[0], corrupt_ids={corrupt_native}
    )
    before = _canonical_bytes(snapshot.root)
    with pytest.raises(SourceChangedDuringRunError):
        _scan()
    assert downloads == [snapshot.emojis[identifier].native_id for identifier in identifiers]
    assert _canonical_bytes(snapshot.root) == before
    assert not index_path.exists()


def test_command_refuses_to_publish_remaining_unselected_partial_fingerprints(
    tmp_path, monkeypatch, verified_media
):
    snapshot = _verified_legacy(tmp_path / "repo", verified_media, count=2)
    identifier = sorted(snapshot.emojis)[0]
    downloads, index_path = _wire_command(monkeypatch, snapshot, verified_media[0])
    before = _canonical_bytes(snapshot.root)
    with pytest.raises(CommandError, match="Unselected legacy"):
        _scan(identifier)
    assert downloads == [snapshot.emojis[identifier].native_id]
    assert _canonical_bytes(snapshot.root) == before
    assert not index_path.exists()


def test_command_rejects_discarded_temporary_checkout_before_credentials(
    tmp_path, monkeypatch, verified_media
):
    snapshot = _verified_legacy(tmp_path / "repo", verified_media)
    _, index_path = _wire_command(monkeypatch, snapshot, verified_media[0], temporary=True)
    monkeypatch.setattr(
        module, "load_credentials", lambda: pytest.fail("must reject before loading credentials")
    )
    before = _canonical_bytes(snapshot.root)
    with pytest.raises(CommandError, match="persistent local checkout"):
        _scan()
    assert _canonical_bytes(snapshot.root) == before
    assert not index_path.exists()


def test_command_empty_collection_does_not_select_all_or_backfill_other_records(
    tmp_path, monkeypatch, verified_media
):
    snapshot = _verified_legacy(tmp_path / "repo", verified_media)
    collection = next(iter(snapshot.collections.values()))
    collection.item_count = 0
    collection.extensions["telegram"]["set_fingerprint_sha256"] = telegram_set_fingerprint([])
    snapshot.memberships.clear()
    for path, data in snapshot.to_files().items():
        if "emojis" not in path.parts:
            (snapshot.root / path).write_bytes(data)
    _wire_command(monkeypatch, snapshot, verified_media[0])
    monkeypatch.setattr(
        module, "load_credentials", lambda: pytest.fail("empty selection cannot fetch source")
    )
    before = _canonical_bytes(snapshot.root)
    result = _scan(collection.id).result
    assert result["selected_emoji_ids"] == []
    assert result["backfilled_emoji_ids"] == []
    assert result["canonical_writes"] == 0
    assert _canonical_bytes(snapshot.root) == before


def test_command_failed_index_never_publishes_fingerprints(tmp_path, monkeypatch, verified_media):
    snapshot = _verified_legacy(tmp_path / "repo", verified_media)
    _wire_command(monkeypatch, snapshot, verified_media[0])
    before = _canonical_bytes(snapshot.root)

    def fail_index(*args, **kwargs):
        raise DedupeIndexError("fixture index failure")

    monkeypatch.setattr(command_module.DedupeIndex, "update", fail_index)
    with pytest.raises(DedupeIndexError, match="fixture index failure"):
        _scan()
    assert _canonical_bytes(snapshot.root) == before


def test_command_atomic_publication_rechecks_original_source_bytes(
    tmp_path, monkeypatch, verified_media
):
    snapshot = _verified_legacy(tmp_path / "repo", verified_media)
    _wire_command(monkeypatch, snapshot, verified_media[0])
    original_update = command_module.DedupeIndex.update
    manifest_path = snapshot.root / "dataset.json"
    before = _canonical_bytes(snapshot.root)

    def concurrent_edit(*args, **kwargs):
        report = original_update(*args, **kwargs)
        manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
        return report

    monkeypatch.setattr(command_module.DedupeIndex, "update", concurrent_edit)
    with pytest.raises(AtomicWriteError, match="dataset changed since"):
        _scan()
    expected = dict(before)
    expected[PurePosixPath("dataset.json")] += b"\n"
    assert _canonical_bytes(snapshot.root) == expected


def test_command_does_not_mask_noncanonical_untouched_source_bytes(
    tmp_path, monkeypatch, verified_media
):
    snapshot = _verified_legacy(tmp_path / "repo", verified_media)
    _, index_path = _wire_command(monkeypatch, snapshot, verified_media[0])
    manifest_path = snapshot.root / "dataset.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    before = _canonical_bytes(snapshot.root)
    with pytest.raises(ValueError, match="manifest is not canonical"):
        _scan()
    assert _canonical_bytes(snapshot.root) == before
    assert not index_path.exists()


@pytest.mark.parametrize("field", ["media", "dedupe_profile"])
def test_malformed_legacy_backfill_input_has_typed_load_failure(tmp_path, field):
    write_fixture(tmp_path)
    bucket = next((tmp_path / "data").glob("*/emojis/*/*.jsonl"))
    emoji = json.loads(bucket.read_text())
    emoji.pop("fingerprints")
    if field == "media":
        emoji.pop("media")
    else:
        manifest = tmp_path / "dataset.json"
        payload = json.loads(manifest.read_text())
        payload.pop("dedupe_profile")
        manifest.write_text(json.dumps(payload), encoding="utf-8")
    bucket.write_bytes((json.dumps(emoji) + "\n").encode())
    with pytest.raises(DatasetLoadError, match="requires pinned profile and media"):
        load_dataset(tmp_path, allow_missing_fingerprints=True)


def test_backfill_rejects_unsupported_locator_before_credentials(tmp_path, monkeypatch):
    snapshot = make_snapshot(tmp_path)
    original = _incomplete(snapshot)
    snapshot.emojis[original.id].scope_id = "unsupported-scope"
    monkeypatch.setattr(
        module, "load_credentials", lambda: pytest.fail("unsupported locator cannot be fetched")
    )
    with pytest.raises(CommandError, match="cannot verify every"):
        asyncio.run(module.backfill_fingerprints(snapshot, snapshot.emojis, MojiLexConfig()))
