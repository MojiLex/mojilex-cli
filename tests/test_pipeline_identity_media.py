from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path

import pytest
from PIL import Image

from mojilex_cli.ai import RequestBudget
from mojilex_cli.analysis import DeterministicMediaAnalysis as MediaAnalysis
from mojilex_cli.cache import CacheStore
from mojilex_cli.commands.runtime import CommandError, structured_exception
from mojilex_cli.config import AIConfig, MojiLexConfig
from mojilex_cli.domain import (
    DeterministicEmojiAnalysis,
    Review,
    reviewed_content_sha256,
)
from mojilex_cli.media import (
    MediaMetadata,
    ProcessedMedia,
    SourceChangedDuringRunError,
    TemporaryMediaRun,
)
from mojilex_cli.pipeline.runner import (
    _AIState,
    _descriptions_for_collection,
    _prepare_collection_media,
)
from mojilex_cli.pipeline.transform import IdentityConflictError, plan_collection_merge
from mojilex_cli.sources import SourceCollection, SourceEmoji, telegram_set_fingerprint
from test_dataset_helpers import MEDIA_HASH, NATIVE_EMOJI_ID, write_fixture


class _FakeTelegramAdapter:
    def __init__(self, direct: Mapping[str, SourceEmoji] | None = None) -> None:
        self.direct = dict(direct or {})
        self.direct_calls: list[tuple[str, ...]] = []
        self.media_calls: list[str] = []

    async def fetch_emojis(self, native_ids: tuple[str, ...]) -> dict[str, SourceEmoji]:
        self.direct_calls.append(native_ids)
        return {
            native_id: self.direct[native_id]
            for native_id in native_ids
            if native_id in self.direct
        }

    async def fetch_media(self, emoji_ref: SourceEmoji) -> AsyncIterator[bytes]:
        self.media_calls.append(emoji_ref.file_id)
        yield emoji_ref.file_id.encode("utf-8")


class _FakeMediaProcessor:
    def __init__(self, processed: Mapping[str, ProcessedMedia]) -> None:
        self.processed = dict(processed)
        self.expected_hashes: list[str | None] = []

    async def process_stream(
        self,
        chunks: AsyncIterator[bytes],
        *,
        expected_format: str,
        declared_size: int | None = None,
        expected_sha256: str | None = None,
        needs_repainting: bool = False,
    ) -> ProcessedMedia:
        del expected_format, declared_size, needs_repainting
        payload = b"".join([chunk async for chunk in chunks]).decode("utf-8")
        result = self.processed[payload]
        self.expected_hashes.append(expected_sha256)
        if expected_sha256 is not None and result.metadata.sha256 != expected_sha256:
            raise SourceChangedDuringRunError("synthetic staging hash mismatch")
        return result


def _source(snapshot, *, file_id: str = "reported-file") -> SourceCollection:  # type: ignore[no-untyped-def]
    existing = next(iter(snapshot.emojis.values()))
    telegram = existing.extensions["telegram"]
    item = SourceEmoji(
        native_namespace="custom_emoji.id",
        scope_id="global",
        native_id=existing.native_id,
        file_unique_id=str(telegram["file_unique_id"]),
        position=0,
        width=existing.media[0].width,
        height=existing.media[0].height,
        animated=False,
        video=False,
        needs_repainting=False,
        fallback_emoji=str(telegram["fallback_emoji"]),
        declared_file_size=existing.media[0].byte_size,
        media_format="webp",
        file_id=file_id,
    )
    return SourceCollection(
        platform="telegram",
        kind="custom_emoji_set",
        native_namespace="sticker_set.name",
        scope_id="global",
        native_id="SecondSyntheticPack",
        title="Second synthetic pack",
        canonical_url="https://t.me/addemoji/SecondSyntheticPack",
        item_count=1,
        items=(item,),
        extension={
            "schema_version": "1.0.0",
            "retrieved_via": "bot_api",
            "short_name": "SecondSyntheticPack",
            "sticker_type": "custom_emoji",
            "set_fingerprint_sha256": telegram_set_fingerprint((item,)),
        },
    )


def _processed(snapshot, sha256: str) -> ProcessedMedia:  # type: ignore[no-untyped-def]
    existing = next(iter(snapshot.emojis.values()))
    metadata = MediaMetadata.model_validate(
        existing.media[0].model_dump(exclude={"variant_id"}, exclude_none=True)
    ).model_copy(update={"sha256": sha256})
    return ProcessedMedia(
        metadata=metadata,
        analysis=MediaAnalysis(
            color_profile_sha256=str(snapshot.manifest["color_profile_sha256"]),
            dedupe_profile_sha256=str(snapshot.manifest["dedupe_profile_sha256"]),
            decoder_backend_fingerprint="6" * 64,
            rendering=existing.facets.rendering.items[0].model_dump(
                exclude={"role", "variant_id"}, exclude_none=True
            ),
            fingerprint=existing.fingerprints.items[0].model_dump(
                exclude={"role", "variant_id"}, exclude_none=True
            ),
        ),
        frame_paths=(Path("not-read.png"),),
    )


def _analysis(snapshot) -> DeterministicEmojiAnalysis:  # type: ignore[no-untyped-def]
    existing = next(iter(snapshot.emojis.values()))
    return DeterministicEmojiAnalysis(
        color_profile_sha256=str(snapshot.manifest["color_profile_sha256"]),
        dedupe_profile_sha256=str(snapshot.manifest["dedupe_profile_sha256"]),
        rendering=existing.facets.rendering.model_copy(deep=True),
        fingerprints=existing.fingerprints.model_copy(deep=True),
    )


async def _plan_without_ai(
    snapshot,  # type: ignore[no-untyped-def]
    source: SourceCollection,
    processed: Mapping[str, ProcessedMedia],
    tmp_path: Path,
):
    config = MojiLexConfig(ai=AIConfig(provider="gemini", model="test-model"))
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    budget = RequestBudget(max_requests=1)
    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            descriptions, generation = await _descriptions_for_collection(
                snapshot,
                source,
                processed,
                config=config,
                cache=cache,
                budget=budget,
                ai_state=_AIState(),
                api_key=None,
                redescribe="changed",
                overwrite_reviewed=False,
                temporary=temporary,
            )
    finally:
        cache.close()
    assert budget.requests_used == 0
    native_id = source.items[0].native_id
    return plan_collection_merge(
        snapshot,
        source,
        processed,
        descriptions,
        {native_id: _analysis(snapshot)},
        generation,
        timestamp="2026-09-11T20:00:00Z",
    )


@pytest.mark.asyncio
async def test_known_id_same_hash_adds_only_membership_without_ai(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    source = _source(snapshot)
    reported_item = source.items[0].model_copy(
        update={
            "file_unique_id": "conflicting-reported-file-identity",
            "fallback_emoji": "😈",
            "needs_repainting": True,
        }
    )
    reported_extension = dict(source.extension)
    reported_extension["set_fingerprint_sha256"] = telegram_set_fingerprint((reported_item,))
    source = source.model_copy(update={"items": (reported_item,), "extension": reported_extension})
    same = _processed(snapshot, MEDIA_HASH)
    adapter = _FakeTelegramAdapter()
    processor = _FakeMediaProcessor({"reported-file": same})

    prepared_source, prepared = await _prepare_collection_media(
        snapshot,
        adapter,  # type: ignore[arg-type]
        source,
        processor,  # type: ignore[arg-type]
        concurrency=2,
        expected_hashes={NATIVE_EMOJI_ID: (MEDIA_HASH,)},
    )
    plan = await _plan_without_ai(snapshot, prepared_source, prepared, tmp_path)

    assert adapter.direct_calls == []
    assert adapter.media_calls == ["reported-file"]
    assert processor.expected_hashes == [None]
    assert prepared_source.items[0].file_unique_id == "AgADExampleUniqueId"
    assert prepared_source.items[0].fallback_emoji == "🤨"
    assert prepared_source.items[0].needs_repainting is False
    assert len(plan.snapshot.emojis) == 1
    assert len(plan.snapshot.collections) == 2
    assert len(plan.snapshot.memberships) == 2
    assert plan.created == 0
    assert plan.updated == 0


@pytest.mark.asyncio
async def test_reported_hash_mismatch_direct_refetch_restores_existing_before_merge(
    tmp_path: Path,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    existing = next(iter(snapshot.emojis.values()))
    existing.review = Review(
        status="approved",
        reviewed_at="2026-09-11T19:00:00Z",
        reviewer="synthetic-reviewer",
        reviewed_content_sha256=reviewed_content_sha256(existing),
    )
    before_emoji = existing.as_dict()
    source = _source(snapshot)
    direct_item = source.items[0].model_copy(update={"file_id": "direct-file"})
    adapter = _FakeTelegramAdapter({NATIVE_EMOJI_ID: direct_item})
    processor = _FakeMediaProcessor(
        {
            "reported-file": _processed(snapshot, "a" * 64),
            "direct-file": _processed(snapshot, MEDIA_HASH),
        }
    )

    prepared_source, prepared = await _prepare_collection_media(
        snapshot,
        adapter,  # type: ignore[arg-type]
        source,
        processor,  # type: ignore[arg-type]
        concurrency=2,
        expected_hashes={NATIVE_EMOJI_ID: (MEDIA_HASH,)},
    )
    plan = await _plan_without_ai(snapshot, prepared_source, prepared, tmp_path)

    assert adapter.direct_calls == [(NATIVE_EMOJI_ID,)]
    assert adapter.media_calls == ["reported-file", "direct-file"]
    assert processor.expected_hashes == [None, None]
    assert prepared[NATIVE_EMOJI_ID].metadata.sha256 == MEDIA_HASH
    assert prepared_source.items[0].file_id == "direct-file"
    assert len(plan.snapshot.emojis) == 1
    assert len(plan.snapshot.memberships) == 2
    assert plan.created == 0
    assert plan.updated == 0
    assert plan.snapshot.emojis[existing.id].as_dict() == before_emoji


@pytest.mark.asyncio
async def test_direct_refetch_confirms_changed_hash_and_requires_new_ai(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    before = snapshot.to_files()
    source = _source(snapshot)
    direct_item = source.items[0].model_copy(update={"file_id": "direct-file"})
    changed = _processed(snapshot, "a" * 64)
    adapter = _FakeTelegramAdapter({NATIVE_EMOJI_ID: direct_item})
    processor = _FakeMediaProcessor({"reported-file": changed, "direct-file": changed})

    prepared_source, prepared = await _prepare_collection_media(
        snapshot,
        adapter,  # type: ignore[arg-type]
        source,
        processor,  # type: ignore[arg-type]
        concurrency=2,
        expected_hashes={NATIVE_EMOJI_ID: ("a" * 64,)},
    )

    assert prepared[NATIVE_EMOJI_ID].metadata.sha256 == "a" * 64
    assert prepared_source.items[0].file_id == "direct-file"
    assert adapter.direct_calls == [(NATIVE_EMOJI_ID,)]
    assert adapter.media_calls == ["reported-file", "direct-file"]
    assert snapshot.to_files() == before

    frame = tmp_path / "changed-frame.png"
    Image.new("RGBA", (256, 256), (255, 0, 0, 255)).save(frame, format="PNG")
    prepared = {
        NATIVE_EMOJI_ID: prepared[NATIVE_EMOJI_ID].model_copy(update={"frame_paths": (frame,)})
    }

    config = MojiLexConfig(ai=AIConfig(provider="gemini", model="test-model"))
    cache = CacheStore(tmp_path / "changed-cache.sqlite3", repository_root=snapshot.root)
    budget = RequestBudget(max_requests=1)
    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            with pytest.raises(CommandError) as caught:
                await _descriptions_for_collection(
                    snapshot,
                    prepared_source,
                    prepared,
                    config=config,
                    cache=cache,
                    budget=budget,
                    ai_state=_AIState(),
                    api_key=None,
                    redescribe="changed",
                    overwrite_reviewed=False,
                    temporary=temporary,
                )
    finally:
        cache.close()
    assert caught.value.error.code == "CREDENTIAL_MISSING"
    assert budget.requests_used == 0
    assert snapshot.to_files() == before


@pytest.mark.asyncio
async def test_confirmed_change_still_honors_resumed_expected_hash(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    before = snapshot.to_files()
    source = _source(snapshot)
    direct_item = source.items[0].model_copy(update={"file_id": "direct-file"})
    changed = _processed(snapshot, "a" * 64)
    adapter = _FakeTelegramAdapter({NATIVE_EMOJI_ID: direct_item})
    processor = _FakeMediaProcessor({"reported-file": changed, "direct-file": changed})

    with pytest.raises(SourceChangedDuringRunError):
        await _prepare_collection_media(
            snapshot,
            adapter,  # type: ignore[arg-type]
            source,
            processor,  # type: ignore[arg-type]
            concurrency=2,
            expected_hashes={NATIVE_EMOJI_ID: (MEDIA_HASH,)},
        )

    assert adapter.direct_calls == [(NATIVE_EMOJI_ID,)]
    assert adapter.media_calls == ["reported-file", "direct-file"]
    assert snapshot.to_files() == before


@pytest.mark.asyncio
async def test_persistent_direct_hash_mismatch_fails_before_dataset_or_ai(
    tmp_path: Path,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    before = snapshot.to_files()
    source = _source(snapshot, file_id="reported-sensitive-file-id")
    direct_item = source.items[0].model_copy(update={"file_id": "direct-sensitive-file-id"})
    adapter = _FakeTelegramAdapter({NATIVE_EMOJI_ID: direct_item})
    processor = _FakeMediaProcessor(
        {
            "reported-sensitive-file-id": _processed(snapshot, "a" * 64),
            "direct-sensitive-file-id": _processed(snapshot, "b" * 64),
        }
    )

    with pytest.raises(IdentityConflictError) as caught:
        await _prepare_collection_media(
            snapshot,
            adapter,  # type: ignore[arg-type]
            source,
            processor,  # type: ignore[arg-type]
            concurrency=2,
            expected_hashes={NATIVE_EMOJI_ID: (MEDIA_HASH,)},
        )

    error = structured_exception(caught.value)
    assert error.code == "IDENTITY_CONFLICT"
    assert error.retryable is False
    assert snapshot.to_files() == before
    assert adapter.direct_calls == [(NATIVE_EMOJI_ID,)]
    assert adapter.media_calls == ["reported-sensitive-file-id", "direct-sensitive-file-id"]
    serialized_error = json.dumps(error.as_dict())
    assert "sensitive-file-id" not in serialized_error
    assert "a" * 64 not in serialized_error
    assert "b" * 64 not in serialized_error
