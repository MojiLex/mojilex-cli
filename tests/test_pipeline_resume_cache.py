from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import AsyncIterator, Mapping, Sequence
from decimal import Decimal
from pathlib import Path

import pytest

from mojilex_cli.ai import (
    CostEstimate,
    DescriptionBatch,
    DescriptionItem,
    DescriptionRequest,
    DescriptionResult,
    RequestBudget,
)
from mojilex_cli.analysis import DeterministicMediaAnalysis, decoder_backend_fingerprint
from mojilex_cli.cache import CacheStore
from mojilex_cli.config import AIConfig, MojiLexConfig, ProcessingConfig
from mojilex_cli.dedupe import scan_snapshot
from mojilex_cli.domain import Review, media_digest, reviewed_content_sha256
from mojilex_cli.media import (
    MediaMetadata,
    MediaProcessor,
    ProcessedMedia,
    SourceChangedDuringRunError,
    TemporaryMediaRun,
)
from mojilex_cli.pipeline import runner as pipeline_runner
from mojilex_cli.pipeline.runner import (
    _ai_request_plan_sha256,
    _AICacheTrace,
    _AIRequestIdentity,
    _AIState,
    _cache_deterministic_analyses,
    _cache_deterministic_analysis,
    _cache_key,
    _cached_dedupe_report,
    _checkpoint_ai_keys,
    _checkpoint_dedupe_report,
    _checkpoint_media_item,
    _checkpoint_stage,
    _descriptions_for_collection,
    _deterministic_cache_storage_key,
    _deterministic_key,
    _deterministic_render_context_sha256,
    _prepare_collection_media,
    _request_traces_from_checkpoint,
    _restore_deterministic_cache_entry,
    _resume_dedupe_selected_ids,
    _source_descriptor_sha256,
    _vision_context,
)
from mojilex_cli.pipeline.transform import IdentityConflictError, SemanticGenerationMetadata
from mojilex_cli.runs import (
    AIRequestCheckpoint,
    ElementCheckpoint,
    RunCheckpoint,
    new_checkpoint,
)
from mojilex_cli.sources import SourceCollection, SourceEmoji, telegram_set_fingerprint
from test_dataset_helpers import NATIVE_EMOJI_ID, write_fixture

_PAYLOAD = b"resume-cache-media" * 8
_GENERATED_AT = "2026-01-02T03:04:05Z"


class _CountingAdapter:
    def __init__(
        self,
        payloads: Mapping[str, bytes],
        direct: Mapping[str, SourceEmoji] | None = None,
    ) -> None:
        self.payloads = dict(payloads)
        self.direct = dict(direct or {})
        self.media_calls: list[str] = []
        self.direct_calls: list[tuple[str, ...]] = []

    async def fetch_emojis(self, native_ids: tuple[str, ...]) -> dict[str, SourceEmoji]:
        self.direct_calls.append(native_ids)
        return {
            native_id: self.direct[native_id]
            for native_id in native_ids
            if native_id in self.direct
        }

    async def fetch_media(self, emoji_ref: SourceEmoji) -> AsyncIterator[bytes]:
        self.media_calls.append(emoji_ref.file_id)
        payload = self.payloads[emoji_ref.file_id]
        midpoint = max(1, len(payload) // 2)
        yield payload[:midpoint]
        yield payload[midpoint:]


class _CountingProcessor(MediaProcessor):
    def __init__(self, run: TemporaryMediaRun, analysis: DeterministicMediaAnalysis) -> None:
        super().__init__(run)
        self.analysis = analysis
        self.decode_calls = 0
        self.analysis_calls = 0

    async def process_stream(
        self,
        chunks: AsyncIterator[bytes],
        *,
        expected_format: str,
        declared_size: int | None = None,
        expected_sha256: str | None = None,
        needs_repainting: bool = False,
    ) -> ProcessedMedia:
        del needs_repainting
        self.decode_calls += 1
        self.analysis_calls += 1
        payload = b"".join([chunk async for chunk in chunks])
        digest = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise SourceChangedDuringRunError("synthetic expected hash mismatch")
        assert expected_format == "webp"
        assert declared_size in {None, len(payload)}
        return ProcessedMedia(
            metadata=MediaMetadata(
                kind="static",
                format="webp",
                mime_type="image/webp",
                sha256=digest,
                byte_size=len(payload),
                width=100,
                height=100,
                animated=False,
            ),
            analysis=self.analysis,
            frame_paths=(Path("synthetic-decoded-frame.png"),),
        )

    async def process_stream_reusing_analysis(
        self,
        chunks: AsyncIterator[bytes],
        *,
        expected_format: str,
        declared_size: int | None,
        cached: ProcessedMedia,
        needs_repainting: bool = False,
    ) -> ProcessedMedia:
        payload = b"".join([chunk async for chunk in chunks])
        digest = hashlib.sha256(payload).hexdigest()
        if digest != cached.metadata.sha256:
            raise SourceChangedDuringRunError("synthetic expected hash mismatch")
        self.decode_calls += 1
        assert expected_format == cached.metadata.format
        assert declared_size in {None, len(payload)}
        dark_paths = (
            (Path("synthetic-decoded-dark-frame.png"),) if cached.semantic_has_dark_render else ()
        )
        del needs_repainting
        return cached.model_copy(
            update={
                "frame_paths": (Path("synthetic-decoded-frame.png"),),
                "dark_frame_paths": dark_paths,
                "rendered_frame_count": cached.semantic_frame_count,
                "has_dark_render": cached.semantic_has_dark_render,
            }
        )


class _SelectiveProvider:
    name = "gemini"
    _PREFIX = b"\x89PNG\r\n\x1a\nresume-item:"

    def __init__(
        self,
        *,
        fail_native_id: str | None = None,
        model_revision: str = "revision-paid-once",
    ) -> None:
        self.fail_native_id = fail_native_id
        self.model_revision = model_revision
        self.calls: list[str] = []

    def estimate(self, request: DescriptionRequest) -> CostEstimate:
        del request
        return CostEstimate(upper_bound_usd=Decimal("0"), note="test")

    async def describe(self, request: DescriptionRequest) -> DescriptionResult:
        data = request.images[0].data
        assert data.startswith(self._PREFIX)
        native_id = data.removeprefix(self._PREFIX).decode("utf-8")
        self.calls.append(native_id)
        if native_id == self.fail_native_id:
            raise RuntimeError("synthetic provider interruption")
        return DescriptionResult(
            batch=DescriptionBatch(items=(_description(),)),
            provider=self.name,
            model=request.model,
            model_revision=self.model_revision,
        )


def _analysis(snapshot: object) -> DeterministicMediaAnalysis:
    emoji = next(iter(snapshot.emojis.values()))  # type: ignore[attr-defined]
    return DeterministicMediaAnalysis(
        color_profile_sha256=str(snapshot.manifest["color_profile_sha256"]),  # type: ignore[attr-defined]
        dedupe_profile_sha256=str(snapshot.manifest["dedupe_profile_sha256"]),  # type: ignore[attr-defined]
        decoder_backend_fingerprint=decoder_backend_fingerprint("webp"),
        rendering=emoji.facets.rendering.items[0].model_dump(
            exclude={"role", "variant_id"}, exclude_none=True
        ),
        fingerprint=emoji.fingerprints.items[0].model_dump(
            exclude={"role", "variant_id"}, exclude_none=True
        ),
    )


def _processed(snapshot: object, payload: bytes = _PAYLOAD) -> ProcessedMedia:
    return ProcessedMedia(
        metadata=MediaMetadata(
            kind="static",
            format="webp",
            mime_type="image/webp",
            sha256=hashlib.sha256(payload).hexdigest(),
            byte_size=len(payload),
            width=100,
            height=100,
            animated=False,
        ),
        analysis=_analysis(snapshot),
        frame_paths=(Path("must-not-be-read.png"),),
    )


def _item(
    native_id: str,
    *,
    unique_id: str,
    file_id: str,
    payload: bytes = _PAYLOAD,
    fallback_emoji: str | None = None,
) -> SourceEmoji:
    return SourceEmoji(
        native_namespace="custom_emoji.id",
        scope_id="global",
        native_id=native_id,
        file_unique_id=unique_id,
        position=0,
        width=100,
        height=100,
        animated=False,
        video=False,
        needs_repainting=False,
        fallback_emoji=fallback_emoji,
        declared_file_size=len(payload),
        media_format="webp",
        file_id=file_id,
    )


def _collection(
    items: tuple[SourceEmoji, ...], *, native_id: str = "ResumeCachePack"
) -> SourceCollection:
    positioned = tuple(
        item.model_copy(update={"position": index}) for index, item in enumerate(items)
    )
    return SourceCollection(
        platform="telegram",
        kind="custom_emoji_set",
        native_namespace="sticker_set.name",
        scope_id="global",
        native_id=native_id,
        title="Resume cache test",
        canonical_url=f"https://t.me/addemoji/{native_id}",
        item_count=len(positioned),
        items=positioned,
        extension={
            "schema_version": "1.0.0",
            "retrieved_via": "bot_api",
            "short_name": native_id,
            "sticker_type": "custom_emoji",
            "set_fingerprint_sha256": telegram_set_fingerprint(positioned),
        },
    )


async def _prepare_single_request(
    items: Sequence[SourceEmoji],
    processed: Mapping[str, ProcessedMedia],
    *,
    model: str,
    temporary: TemporaryMediaRun,
) -> pipeline_runner._PreparedAIRequest:
    del temporary
    assert len(items) == 1
    item = items[0]
    label = "E001"
    data = _SelectiveProvider._PREFIX + item.native_id.encode("utf-8")
    context = _vision_context(item, processed[item.native_id])
    request = DescriptionRequest(
        model=model,
        images=(pipeline_runner.VisionImage(data=data, labels=(label,)),),
        expected_labels=(label,),
        context={label: context},
    )
    plan_sha256 = _ai_request_plan_sha256(items, processed, model=model)
    identity = _AIRequestIdentity(
        plan_sha256=plan_sha256,
        request_sha256=hashlib.sha256(
            b"resume-request-v1\0" + model.encode("utf-8") + b"\0" + data
        ).hexdigest(),
        shown_media_sha256=(hashlib.sha256(data).hexdigest(),),
        labels_by_native=((item.native_id, label),),
    )
    return pipeline_runner._PreparedAIRequest(request=request, identity=identity)


def _description() -> DescriptionItem:
    localized = {
        "text": "A cached semantic description.",
        "motion_status": "not_applicable",
        "usage": ["reaction"],
    }
    return DescriptionItem.model_validate(
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
            "semantic_tags": ["cached"],
            "content": {"rating": "general", "warnings": []},
        }
    )


def _seed_resume(
    cache: CacheStore,
    collection: SourceCollection,
    processed: ProcessedMedia,
    config: MojiLexConfig,
    *,
    with_ai: bool = True,
) -> dict[str, ElementCheckpoint]:
    values = {item.native_id: processed for item in collection.items}
    _cache_deterministic_analyses(cache, collection, values)
    plan_sha256 = _ai_request_plan_sha256(
        collection.items,
        values,
        model=config.ai.model,
    )
    request_identity = _AIRequestIdentity(
        plan_sha256=plan_sha256,
        request_sha256=hashlib.sha256(f"request:{plan_sha256}".encode()).hexdigest(),
        shown_media_sha256=(hashlib.sha256(f"shown:{plan_sha256}".encode()).hexdigest(),),
        labels_by_native=tuple(
            (item.native_id, f"E{index:03d}")
            for index, item in enumerate(collection.items, start=1)
        ),
    )
    elements: dict[str, ElementCheckpoint] = {}
    for item in collection.items:
        ai_key: str | None = None
        if with_ai:
            ai_key = _cache_key(
                item,
                processed,
                _vision_context(item, processed),
                config,
                model=config.ai.model,
                model_revision="revision-actual",
                taxonomy_version="1.0.0",
                request_identity=request_identity,
                item_label=request_identity.label_for(item.native_id),
            )
            cache.put_ai(
                ai_key,
                DescriptionResult(
                    batch=DescriptionBatch(items=(_description(),)),
                    provider="gemini",
                    model=config.ai.model,
                    model_revision="revision-actual",
                ),
                generated_at=_GENERATED_AT,
            )
        elements[item.native_id] = ElementCheckpoint(
            stage="ai_cached" if with_ai else "fingerprint_ready",
            source_descriptor_sha256=_source_descriptor_sha256(item),
            media_sha256=(processed.metadata.sha256,),
            deterministic_cache_key=_deterministic_key(processed),
            ai_cache_key=ai_key,
            ai_requests=(
                (
                    AIRequestCheckpoint(
                        stage="primary",
                        model=config.ai.model,
                        model_revision="revision-actual",
                        cache_key=ai_key,
                        plan_sha256=request_identity.plan_sha256,
                        request_sha256=request_identity.request_sha256,
                        shown_media_sha256=request_identity.shown_media_sha256,
                        item_label=request_identity.label_for(item.native_id),
                    )
                    if ai_key is not None
                    else None
                ),
            )
            if ai_key is not None
            else (),
            palette_complete=True,
            fingerprint_complete=True,
            ai_facets_complete=with_ai,
        )
    return elements


@pytest.mark.asyncio
async def test_two_descriptors_rebuild_missing_puzzle_tiles_without_repeating_ai(
    tmp_path: Path,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    first = _item("new-native-1", unique_id="descriptor-one", file_id="first")
    second = _item("new-native-2", unique_id="descriptor-two", file_id="second")
    source = _collection((first, second))
    processed = _processed(snapshot)
    adapter = _CountingAdapter({"first": _PAYLOAD, "second": _PAYLOAD})
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    elements = _seed_resume(cache, source, processed, config)
    payload = cache.get_metadata(
        _deterministic_cache_storage_key(
            _deterministic_key(processed),
            _deterministic_render_context_sha256(first),
        )
    )
    assert payload is not None
    assert "source_descriptor_sha256" not in payload
    verified_outcomes: dict[str, pipeline_runner._SemanticOutcome] = {}

    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            prepared_source, prepared = await _prepare_collection_media(
                snapshot,
                adapter,  # type: ignore[arg-type]
                source,
                processor,
                concurrency=2,
                cache=cache,
                resume_elements=elements,
                config=config,
                taxonomy_version="1.0.0",
                cache_alias_scope="mlxrun_" + "a" * 32,
                verified_semantic_outcomes=verified_outcomes,
            )
            assert set(verified_outcomes) == {first.native_id, second.native_id}
            # A concurrent cache prune after verified preparation cannot force
            # empty-frame media into contact-sheet construction or a paid call.
            cache.prune(older_than_epoch=2**31 - 1)
            budget = RequestBudget(max_requests=0)
            descriptions, _ = await _descriptions_for_collection(
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
                resume_ai_cache_keys={
                    native_id: element.ai_cache_key
                    for native_id, element in elements.items()
                    if element.ai_cache_key is not None
                },
                cache_alias_scope="mlxrun_" + "a" * 32,
                verified_resume_outcomes=verified_outcomes,
            )
    finally:
        cache.close()

    assert set(descriptions) == {first.native_id, second.native_id}
    assert adapter.media_calls == ["first", "second"]
    assert processor.decode_calls == 2
    assert processor.analysis_calls == 0
    assert budget.requests_used == 0
    assert all(value.frame_paths for value in prepared.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["descriptor", "profile", "backend", "config", "legacy"])
async def test_stale_resume_context_falls_back_to_full_processing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    original = _item("new-native", unique_id="descriptor", file_id="reported")
    source = _collection((original,))
    processed = _processed(snapshot)
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    elements = _seed_resume(cache, source, processed, config)
    if mutation == "descriptor":
        changed = original.model_copy(update={"file_unique_id": "descriptor-changed"})
        source = _collection((changed,))
    elif mutation == "profile":
        key = _deterministic_key(processed)
        storage_key = _deterministic_cache_storage_key(
            key,
            _deterministic_render_context_sha256(original),
        )
        payload = cache.get_metadata(storage_key)
        assert payload is not None
        analysis = dict(payload["analysis"])
        analysis["color_profile_sha256"] = "0" * 64
        cache.put_metadata(storage_key, {**payload, "analysis": analysis})
    elif mutation == "backend":
        monkeypatch.setattr(
            pipeline_runner,
            "decoder_backend_fingerprint",
            lambda *args, **kwargs: "0" * 64,
        )
    elif mutation == "config":
        config = config.model_copy(
            update={"ai": config.ai.model_copy(update={"model": "different-primary-model"})}
        )
    else:
        elements = {
            native_id: element.model_copy(update={"source_descriptor_sha256": None})
            for native_id, element in elements.items()
        }
    adapter = _CountingAdapter({"reported": _PAYLOAD})

    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            _, prepared = await _prepare_collection_media(
                snapshot,
                adapter,  # type: ignore[arg-type]
                source,
                processor,
                concurrency=1,
                cache=cache,
                resume_elements=elements,
                config=config,
                taxonomy_version="1.0.0",
                cache_alias_scope="mlxrun_" + "b" * 32,
            )
    finally:
        cache.close()

    assert prepared[original.native_id].metadata.sha256 == processed.metadata.sha256
    assert adapter.media_calls == ["reported"]
    assert processor.decode_calls == 1


@pytest.mark.asyncio
async def test_pruned_deterministic_cache_redecodes_but_reuses_exact_paid_ai(
    tmp_path: Path,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    item = _item("new-native", unique_id="descriptor", file_id="reported")
    source = _collection((item,))
    processed = _processed(snapshot)
    cache_path = tmp_path / "cache.sqlite3"
    cache = CacheStore(cache_path, repository_root=snapshot.root)
    elements = _seed_resume(cache, source, processed, config)
    cache.close()
    connection = sqlite3.connect(cache_path)
    connection.execute("DELETE FROM metadata_cache")
    connection.commit()
    connection.close()
    cache = CacheStore(cache_path, repository_root=snapshot.root)
    adapter = _CountingAdapter({"reported": _PAYLOAD})

    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            prepared_source, prepared = await _prepare_collection_media(
                snapshot,
                adapter,  # type: ignore[arg-type]
                source,
                processor,
                concurrency=1,
                cache=cache,
                resume_elements=elements,
                config=config,
                taxonomy_version="1.0.0",
            )
            budget = RequestBudget(max_requests=0)
            descriptions, _ = await _descriptions_for_collection(
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
                resume_request_traces=_request_traces_from_checkpoint(source.items, elements),
            )
    finally:
        cache.close()

    assert set(descriptions) == {item.native_id}
    assert adapter.media_calls == ["reported"]
    assert processor.decode_calls == 1
    assert budget.requests_used == 0


@pytest.mark.asyncio
async def test_malformed_ai_cache_probe_fails_safe_to_full_decode(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    item = _item("new-native", unique_id="descriptor", file_id="reported")
    source = _collection((item,))
    processed = _processed(snapshot)
    cache_path = tmp_path / "cache.sqlite3"
    cache = CacheStore(cache_path, repository_root=snapshot.root)
    elements = _seed_resume(cache, source, processed, config)
    cache.close()
    connection = sqlite3.connect(cache_path)
    connection.execute("UPDATE ai_cache SET payload_json = ?", ("not-json",))
    connection.commit()
    connection.close()
    cache = CacheStore(cache_path, repository_root=snapshot.root)
    adapter = _CountingAdapter({"reported": _PAYLOAD})

    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            _, prepared = await _prepare_collection_media(
                snapshot,
                adapter,  # type: ignore[arg-type]
                source,
                processor,
                concurrency=1,
                cache=cache,
                resume_elements=elements,
                config=config,
                taxonomy_version="1.0.0",
            )
    finally:
        cache.close()

    assert prepared[item.native_id].metadata.sha256 == processed.metadata.sha256
    assert adapter.media_calls == ["reported"]
    assert processor.decode_calls == 1


@pytest.mark.asyncio
async def test_resume_changed_bytes_fails_closed_before_decode(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    item = _item("new-native", unique_id="descriptor", file_id="reported")
    source = _collection((item,))
    processed = _processed(snapshot)
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    elements = _seed_resume(cache, source, processed, config)
    adapter = _CountingAdapter({"reported": b"X" * len(_PAYLOAD)})

    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            with pytest.raises(SourceChangedDuringRunError):
                await _prepare_collection_media(
                    snapshot,
                    adapter,  # type: ignore[arg-type]
                    source,
                    processor,
                    concurrency=1,
                    cache=cache,
                    resume_elements=elements,
                    config=config,
                    taxonomy_version="1.0.0",
                    cache_alias_scope="mlxrun_" + "c" * 32,
                )
    finally:
        cache.close()

    assert adapter.media_calls == ["reported"]
    assert processor.decode_calls == 0


def _replace_existing_media(snapshot: object, payload: bytes) -> None:
    emoji_id = next(iter(snapshot.emojis))  # type: ignore[attr-defined]
    existing = snapshot.emojis[emoji_id]  # type: ignore[attr-defined]
    digest = hashlib.sha256(payload).hexdigest()
    media = existing.media[0].model_copy(update={"sha256": digest, "byte_size": len(payload)})
    fingerprints = existing.fingerprints.model_copy(
        update={"input_media_digest": media_digest([media])}
    )
    provenance = existing.provenance.model_copy(update={"input_media_sha256": [digest]})
    unreviewed = existing.model_copy(
        update={
            "media": [media],
            "fingerprints": fingerprints,
            "provenance": provenance,
            "review": Review(status="unreviewed"),
        }
    )
    approved = Review(
        status="approved",
        reviewed_at="2026-09-11T19:00:00Z",
        reviewer="resume-cache-test",
        reviewed_content_sha256=reviewed_content_sha256(unreviewed),
        review_hash_profile_id="semantic-review-content-v3",
    )
    snapshot.emojis[emoji_id] = unreviewed.model_copy(  # type: ignore[attr-defined]
        update={"review": approved}
    )


@pytest.mark.asyncio
async def test_approved_unchanged_item_resumes_analysis_without_ai_key(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    _replace_existing_media(snapshot, _PAYLOAD)
    existing = next(iter(snapshot.emojis.values()))
    telegram = existing.extensions["telegram"]
    item = _item(
        NATIVE_EMOJI_ID,
        unique_id=str(telegram["file_unique_id"]),
        file_id="reported",
        fallback_emoji=str(telegram["fallback_emoji"]),
    )
    source = _collection((item,), native_id="SuspiciousCats")
    processed = _processed(snapshot)
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    elements = _seed_resume(cache, source, processed, config, with_ai=False)
    adapter = _CountingAdapter({"reported": _PAYLOAD})

    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            prepared_source, prepared = await _prepare_collection_media(
                snapshot,
                adapter,  # type: ignore[arg-type]
                source,
                processor,
                concurrency=1,
                cache=cache,
                resume_elements=elements,
                config=config,
                taxonomy_version="1.0.0",
                redescribe="changed",
                overwrite_reviewed=False,
            )
            budget = RequestBudget(max_requests=0)
            descriptions, _ = await _descriptions_for_collection(
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

    assert descriptions[item.native_id].descriptions.en.text == (existing.descriptions["en"].text)
    assert adapter.media_calls == ["reported"]
    assert processor.decode_calls == 0
    assert budget.requests_used == 0


@pytest.mark.asyncio
async def test_successful_media_is_checkpointed_before_sibling_failure_and_resumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    _replace_existing_media(snapshot, _PAYLOAD)
    original_files = snapshot.to_files()
    existing = next(iter(snapshot.emojis.values()))
    telegram = existing.extensions["telegram"]
    first = _item(
        NATIVE_EMOJI_ID,
        unique_id=str(telegram["file_unique_id"]),
        file_id="completed",
        fallback_emoji=str(telegram["fallback_emoji"]),
    )
    second_payload = b"media-sibling-payload" * 7
    second = _item(
        "media-sibling-fails",
        unique_id="media-sibling-descriptor",
        file_id="missing",
        payload=second_payload,
    )
    source = _collection((first, second), native_id="SuspiciousCats")
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)

    async def persist_media(item: SourceEmoji, value: ProcessedMedia) -> None:
        nonlocal checkpoint
        _cache_deterministic_analysis(cache, item, value)
        checkpoint = _checkpoint_media_item(checkpoint, item, value)

    try:
        interrupted_adapter = _CountingAdapter({"completed": _PAYLOAD})
        with TemporaryMediaRun(root=tmp_path) as temporary:
            interrupted_processor = _CountingProcessor(temporary, _analysis(snapshot))
            with pytest.raises(KeyError):
                await _prepare_collection_media(
                    snapshot,
                    interrupted_adapter,  # type: ignore[arg-type]
                    source,
                    interrupted_processor,
                    concurrency=2,
                    on_item_completed=persist_media,
                )

        completed = checkpoint.elements[first.native_id]
        assert completed.palette_complete is True
        assert completed.fingerprint_complete is True
        assert second.native_id not in checkpoint.elements
        assert completed.deterministic_cache_key is not None
        assert (
            cache.get_metadata(
                _deterministic_cache_storage_key(
                    completed.deterministic_cache_key,
                    _deterministic_render_context_sha256(first),
                )
            )
            is not None
        )
        assert snapshot.to_files() == original_files

        resumed_adapter = _CountingAdapter({"completed": _PAYLOAD, "missing": second_payload})
        with TemporaryMediaRun(root=tmp_path) as temporary:
            resumed_processor = _CountingProcessor(temporary, _analysis(snapshot))
            prepared_source, prepared = await _prepare_collection_media(
                snapshot,
                resumed_adapter,  # type: ignore[arg-type]
                source,
                resumed_processor,
                concurrency=2,
                cache=cache,
                resume_elements=checkpoint.elements,
                config=config,
                taxonomy_version="1.0.0",
                on_item_completed=persist_media,
            )

            ai_chunks: list[tuple[str, ...]] = []

            async def fake_describe_batch(
                items: Sequence[SourceEmoji],
                *args: object,
                **kwargs: object,
            ) -> dict[str, pipeline_runner._SemanticOutcome]:
                del args, kwargs
                ai_chunks.append(tuple(item.native_id for item in items))
                generation = SemanticGenerationMetadata(
                    provider="gemini",
                    model="primary-model",
                    model_revision="revision-test",
                    prompt_sha256="1" * 64,
                    request_parameters_sha256="2" * 64,
                    generated_at=_GENERATED_AT,
                )
                return {
                    item.native_id: pipeline_runner._SemanticOutcome(
                        description=_description(), generation=generation
                    )
                    for item in items
                }

            monkeypatch.setattr(pipeline_runner, "_describe_batch", fake_describe_batch)
            descriptions, _ = await _descriptions_for_collection(
                snapshot,
                prepared_source,
                prepared,
                config=config,
                cache=cache,
                budget=RequestBudget(max_requests=0),
                ai_state=_AIState(),
                api_key=None,
                redescribe="changed",
                overwrite_reviewed=False,
                temporary=temporary,
            )
    finally:
        cache.close()

    assert set(descriptions) == {first.native_id, second.native_id}
    assert resumed_adapter.media_calls == ["completed", "missing"]
    assert resumed_processor.decode_calls == 1
    assert not prepared[first.native_id].frame_paths
    assert ai_chunks == [(second.native_id,)]
    assert snapshot.to_files() == original_files


@pytest.mark.asyncio
async def test_pending_ai_resume_redecodes_frames_without_recomputing_completed_analysis(
    tmp_path: Path,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    first = _item("pending-ai-completed", unique_id="pending-first", file_id="first")
    second_payload = b"pending-ai-sibling" * 9
    second = _item(
        "pending-ai-sibling",
        unique_id="pending-second",
        file_id="missing",
        payload=second_payload,
    )
    source = _collection((first, second), native_id="PendingAIResume")
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)

    async def persist_media(item: SourceEmoji, value: ProcessedMedia) -> None:
        nonlocal checkpoint
        _cache_deterministic_analysis(cache, item, value)
        checkpoint = _checkpoint_media_item(checkpoint, item, value)

    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            with pytest.raises(KeyError):
                await _prepare_collection_media(
                    snapshot,
                    _CountingAdapter({"first": _PAYLOAD}),  # type: ignore[arg-type]
                    source,
                    processor,
                    concurrency=2,
                    on_item_completed=persist_media,
                )

        adapter = _CountingAdapter({"first": _PAYLOAD, "missing": second_payload})
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            _, prepared = await _prepare_collection_media(
                snapshot,
                adapter,  # type: ignore[arg-type]
                source,
                processor,
                concurrency=2,
                cache=cache,
                resume_elements=checkpoint.elements,
                config=config,
                taxonomy_version="1.0.0",
                on_item_completed=persist_media,
            )
    finally:
        cache.close()

    assert adapter.media_calls == ["first", "missing"]
    assert processor.decode_calls == 2
    assert processor.analysis_calls == 1
    assert prepared[first.native_id].analysis == _analysis(snapshot)
    assert prepared[first.native_id].frame_paths
    assert checkpoint.elements[first.native_id].ai_facets_complete is False


@pytest.mark.asyncio
async def test_successful_ai_chunk_is_checkpointed_before_sibling_failure_and_resumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    original_files = snapshot.to_files()
    first = _item("ai-chunk-completed", unique_id="ai-first", file_id="first")
    second = _item("ai-chunk-fails", unique_id="ai-second", file_id="second")
    source = _collection((first, second), native_id="AIChunkResume")
    value = _processed(snapshot)
    processed = {first.native_id: value, second.native_id: value}
    config = MojiLexConfig(
        ai=AIConfig(
            provider="gemini",
            model="primary-model",
            model_routing="off",
            ai_concurrency=2,
        ),
        processing=ProcessingConfig(static_batch_size=1),
    )
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    for item in source.items:
        checkpoint = _checkpoint_media_item(checkpoint, item, processed[item.native_id])
    run_scope = checkpoint.run_id
    provider = _SelectiveProvider(fail_native_id=second.native_id)
    completed_chunks: list[tuple[str, ...]] = []

    async def fixed_provider(*args: object, **kwargs: object) -> _SelectiveProvider:
        del args, kwargs
        return provider

    async def persist_ai(
        items: Sequence[SourceEmoji],
        outcomes: Mapping[str, pipeline_runner._SemanticOutcome],
    ) -> None:
        nonlocal checkpoint
        completed_chunks.append(tuple(item.native_id for item in items))
        traces = {
            native_id: outcome.request_trace
            for native_id, outcome in outcomes.items()
            if outcome.request_trace
        }
        generation = {native_id: outcome.generation for native_id, outcome in outcomes.items()}
        partial_source = source.model_copy(update={"item_count": len(items), "items": tuple(items)})
        checkpoint = _checkpoint_ai_keys(
            checkpoint,
            partial_source,
            {},
            config,
            generation,
            taxonomy_version="1.0.0",
            request_traces=traces,
        )
        checkpoint = _checkpoint_stage(
            checkpoint,
            tuple(item.native_id for item in items),
            "ai_cached",
        )

    monkeypatch.setattr(pipeline_runner, "_prepare_ai_request", _prepare_single_request)
    monkeypatch.setattr(pipeline_runner, "_provider_for_model", fixed_provider)
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            with pytest.raises(RuntimeError, match="synthetic provider interruption"):
                await _descriptions_for_collection(
                    snapshot,
                    source,
                    processed,
                    config=config,
                    cache=cache,
                    budget=RequestBudget(max_requests=2),
                    ai_state=_AIState(),
                    api_key="unused-test-key",
                    redescribe="changed",
                    overwrite_reviewed=False,
                    temporary=temporary,
                    cache_alias_scope=run_scope,
                    on_chunk_completed=persist_ai,
                )

        assert completed_chunks == [(first.native_id,)]
        assert checkpoint.elements[first.native_id].ai_facets_complete is True
        assert checkpoint.elements[first.native_id].ai_requests
        assert checkpoint.elements[second.native_id].ai_facets_complete is False
        assert cache.info() == {
            "metadata_entries": 1,
            "ai_entries": 1,
            "database_bytes": cache.info()["database_bytes"],
        }
        assert snapshot.to_files() == original_files

        provider.fail_native_id = None
        calls_before_resume = len(provider.calls)
        completed_chunks.clear()
        with TemporaryMediaRun(root=tmp_path) as temporary:
            descriptions, _ = await _descriptions_for_collection(
                snapshot,
                source,
                processed,
                config=config,
                cache=cache,
                budget=RequestBudget(max_requests=1),
                ai_state=_AIState(),
                api_key="unused-test-key",
                redescribe="changed",
                overwrite_reviewed=False,
                temporary=temporary,
                cache_alias_scope=run_scope,
                resume_request_traces=_request_traces_from_checkpoint(
                    source.items, checkpoint.elements
                ),
                on_chunk_completed=persist_ai,
            )
    finally:
        cache.close()

    assert set(descriptions) == {first.native_id, second.native_id}
    assert provider.calls[calls_before_resume:] == [second.native_id]
    assert completed_chunks == [(first.native_id,), (second.native_id,)]
    assert checkpoint.elements[second.native_id].ai_facets_complete is True
    assert snapshot.to_files() == original_files


@pytest.mark.asyncio
async def test_ai_envelope_recovers_paid_result_when_checkpoint_callback_crashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    item = _item("ai-before-checkpoint-crash", unique_id="ai-crash", file_id="first")
    source = _collection((item,), native_id="AICallbackCrash")
    value = _processed(snapshot)
    processed = {item.native_id: value}
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off"),
        processing=ProcessingConfig(static_batch_size=1),
    )
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    checkpoint = _checkpoint_media_item(checkpoint, item, value)
    provider = _SelectiveProvider()

    async def fixed_provider(*args: object, **kwargs: object) -> _SelectiveProvider:
        del args, kwargs
        return provider

    async def crash_before_checkpoint(
        items: Sequence[SourceEmoji],
        outcomes: Mapping[str, pipeline_runner._SemanticOutcome],
    ) -> None:
        del items, outcomes
        raise RuntimeError("synthetic checkpoint write interruption")

    monkeypatch.setattr(pipeline_runner, "_prepare_ai_request", _prepare_single_request)
    monkeypatch.setattr(pipeline_runner, "_provider_for_model", fixed_provider)
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            with pytest.raises(RuntimeError, match="checkpoint write interruption"):
                await _descriptions_for_collection(
                    snapshot,
                    source,
                    processed,
                    config=config,
                    cache=cache,
                    budget=RequestBudget(max_requests=1),
                    ai_state=_AIState(),
                    api_key="unused-test-key",
                    redescribe="changed",
                    overwrite_reviewed=False,
                    temporary=temporary,
                    cache_alias_scope=checkpoint.run_id,
                    on_chunk_completed=crash_before_checkpoint,
                )

        assert checkpoint.elements[item.native_id].ai_facets_complete is False
        assert cache.info()["ai_entries"] == 1
        assert cache.info()["metadata_entries"] == 1
        calls_before_resume = len(provider.calls)
        stale_checkpoint_trace = _AICacheTrace(
            stage="primary",
            model="primary-model",
            model_revision="stale-revision",
            cache_key="f" * 64,
            request_identity=_AIRequestIdentity(
                plan_sha256=_ai_request_plan_sha256(source.items, processed, model="primary-model"),
                request_sha256="e" * 64,
                shown_media_sha256=("d" * 64,),
                labels_by_native=((item.native_id, "E001"),),
            ),
        )
        with TemporaryMediaRun(root=tmp_path) as temporary:
            descriptions, _ = await _descriptions_for_collection(
                snapshot,
                source,
                processed,
                config=config,
                cache=cache,
                budget=RequestBudget(max_requests=0),
                ai_state=_AIState(),
                api_key=None,
                redescribe="changed",
                overwrite_reviewed=False,
                temporary=temporary,
                cache_alias_scope=checkpoint.run_id,
                resume_request_traces={item.native_id: (stale_checkpoint_trace,)},
            )
    finally:
        cache.close()

    assert set(descriptions) == {item.native_id}
    assert len(provider.calls) == calls_before_resume


@pytest.mark.asyncio
async def test_full_provider_retry_atomically_repairs_poisoned_run_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    item = _item("ai-envelope-repair", unique_id="ai-repair", file_id="first")
    source = _collection((item,), native_id="AIEnvelopeRepair")
    processed = {item.native_id: _processed(snapshot)}
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off"),
        processing=ProcessingConfig(static_batch_size=1),
    )
    run_scope = "mlxrun_" + "a" * 32
    provider = _SelectiveProvider()

    async def fixed_provider(*args: object, **kwargs: object) -> _SelectiveProvider:
        del args, kwargs
        return provider

    monkeypatch.setattr(pipeline_runner, "_prepare_ai_request", _prepare_single_request)
    monkeypatch.setattr(pipeline_runner, "_provider_for_model", fixed_provider)
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            await _descriptions_for_collection(
                snapshot,
                source,
                processed,
                config=config,
                cache=cache,
                budget=RequestBudget(max_requests=1),
                ai_state=_AIState(),
                api_key="unused-test-key",
                redescribe="changed",
                overwrite_reviewed=False,
                temporary=temporary,
                cache_alias_scope=run_scope,
            )
        envelope_row = cache._connection.execute(
            "SELECT cache_key FROM metadata_cache WHERE cache_key LIKE 'ai-request-envelope-v1:%'"
        ).fetchone()
        assert envelope_row is not None
        envelope_key = str(envelope_row[0])
        cache.put_metadata(envelope_key, {"malformed": True})
        calls_before_cached_repair = len(provider.calls)
        with TemporaryMediaRun(root=tmp_path) as temporary:
            cached_repair, _ = await _descriptions_for_collection(
                snapshot,
                source,
                processed,
                config=config,
                cache=cache,
                budget=RequestBudget(max_requests=0),
                ai_state=_AIState(),
                api_key=None,
                redescribe="changed",
                overwrite_reviewed=False,
                temporary=temporary,
                cache_alias_scope=run_scope,
            )
        assert set(cached_repair) == {item.native_id}
        assert len(provider.calls) == calls_before_cached_repair
        repaired_envelope = cache.get_metadata(envelope_key)
        assert repaired_envelope is not None and repaired_envelope.get("format_version") == 1

        cache.put_metadata(envelope_key, {"malformed": True})
        cache._connection.execute("DELETE FROM ai_cache")
        cache._connection.commit()

        provider.model_revision = "revision-after-repair"
        with TemporaryMediaRun(root=tmp_path) as temporary:
            repaired, _ = await _descriptions_for_collection(
                snapshot,
                source,
                processed,
                config=config,
                cache=cache,
                budget=RequestBudget(max_requests=1),
                ai_state=_AIState(),
                api_key="unused-test-key",
                redescribe="changed",
                overwrite_reviewed=False,
                temporary=temporary,
                cache_alias_scope=run_scope,
            )
        calls_after_repair = len(provider.calls)
        with TemporaryMediaRun(root=tmp_path) as temporary:
            resumed, _ = await _descriptions_for_collection(
                snapshot,
                source,
                processed,
                config=config,
                cache=cache,
                budget=RequestBudget(max_requests=0),
                ai_state=_AIState(),
                api_key=None,
                redescribe="changed",
                overwrite_reviewed=False,
                temporary=temporary,
                cache_alias_scope=run_scope,
            )
    finally:
        cache.close()

    assert set(repaired) == {item.native_id}
    assert resumed == repaired
    assert len(provider.calls) == calls_after_repair == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direct_payload", "expected_sha", "error"),
    [
        (_PAYLOAD, hashlib.sha256(_PAYLOAD).hexdigest(), None),
        (b"N" * len(_PAYLOAD), hashlib.sha256(b"N" * len(_PAYLOAD)).hexdigest(), None),
        (b"T" * len(_PAYLOAD), None, IdentityConflictError),
    ],
)
async def test_guarded_resume_always_runs_reported_direct_canonical_three_way(
    tmp_path: Path,
    direct_payload: bytes,
    expected_sha: str | None,
    error: type[Exception] | None,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    _replace_existing_media(snapshot, _PAYLOAD)
    existing = next(iter(snapshot.emojis.values()))
    telegram = existing.extensions["telegram"]
    reported_payload = b"N" * len(_PAYLOAD)
    reported = _item(
        NATIVE_EMOJI_ID,
        unique_id=str(telegram["file_unique_id"]),
        file_id="reported",
        fallback_emoji=str(telegram["fallback_emoji"]),
        payload=reported_payload,
    )
    direct = reported.model_copy(update={"file_id": "direct"})
    source = _collection((reported,), native_id="DifferentPack")
    processed = _processed(snapshot)
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    elements = _seed_resume(cache, source, processed, config)
    adapter = _CountingAdapter(
        {"reported": reported_payload, "direct": direct_payload},
        {NATIVE_EMOJI_ID: direct},
    )

    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            if error is not None:
                with pytest.raises(error):
                    await _prepare_collection_media(
                        snapshot,
                        adapter,  # type: ignore[arg-type]
                        source,
                        processor,
                        concurrency=1,
                        cache=cache,
                        resume_elements=elements,
                        config=config,
                        taxonomy_version="1.0.0",
                    )
                prepared = None
            else:
                _, prepared = await _prepare_collection_media(
                    snapshot,
                    adapter,  # type: ignore[arg-type]
                    source,
                    processor,
                    concurrency=1,
                    cache=cache,
                    resume_elements=elements,
                    config=config,
                    taxonomy_version="1.0.0",
                )
                if expected_sha == hashlib.sha256(_PAYLOAD).hexdigest():
                    budget = RequestBudget(max_requests=0)
                    descriptions, _ = await _descriptions_for_collection(
                        snapshot,
                        source,
                        prepared,
                        config=config,
                        cache=cache,
                        budget=budget,
                        ai_state=_AIState(),
                        api_key=None,
                        redescribe="changed",
                        overwrite_reviewed=False,
                        temporary=temporary,
                        resume_request_traces=_request_traces_from_checkpoint(
                            source.items, elements
                        ),
                    )
                    assert set(descriptions) == {NATIVE_EMOJI_ID}
                    assert budget.requests_used == 0
    finally:
        cache.close()

    assert adapter.media_calls == ["reported", "direct"]
    assert adapter.direct_calls == [(NATIVE_EMOJI_ID,)]
    assert processor.decode_calls == 2
    if expected_sha is not None:
        assert prepared is not None
        assert prepared[NATIVE_EMOJI_ID].metadata.sha256 == expected_sha


def test_dedupe_checkpoint_is_reused_only_for_exact_final_snapshot_and_marks_native_item(
    tmp_path: Path,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    emoji_id = next(iter(snapshot.emojis))
    config = MojiLexConfig()
    selected = {emoji_id}
    report = scan_snapshot(
        snapshot,
        selected_emoji_ids=selected,
        max_candidates=config.dedupe.max_candidates,
        mode=config.dedupe.mode,
    ).as_dict()
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    ).model_copy(
        update={
            "elements": {
                NATIVE_EMOJI_ID: ElementCheckpoint(
                    stage="ai_cached",
                    palette_complete=True,
                    fingerprint_complete=True,
                    ai_facets_complete=True,
                )
            }
        }
    )

    persisted = _checkpoint_dedupe_report(
        checkpoint,
        snapshot,
        selected,
        config,
        report,
    )

    native = persisted.elements[NATIVE_EMOJI_ID]
    assert native.stage == "candidate_scanned"
    assert native.candidate_scan_complete is True
    assert native.review_complete is False
    assert _cached_dedupe_report(persisted, snapshot, selected, config) == report
    assert _resume_dedupe_selected_ids(persisted, snapshot, set()) == selected

    rebased = snapshot.clone()
    collection_id = next(iter(rebased.collections))
    rebased.collections[collection_id] = rebased.collections[collection_id].model_copy(
        update={"title": "Title changed on fetched base"}
    )
    assert _cached_dedupe_report(persisted, rebased, selected, config) is None
    assert _resume_dedupe_selected_ids(persisted, rebased, set()) == set()
    changed_config = config.model_copy(
        update={
            "dedupe": config.dedupe.model_copy(
                update={"max_candidates": config.dedupe.max_candidates + 1}
            )
        }
    )
    assert _cached_dedupe_report(persisted, snapshot, selected, changed_config) is None


def test_legacy_checkpoint_without_new_guards_remains_loadable_but_cannot_fast_resume(
    tmp_path: Path,
) -> None:
    del tmp_path
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    ).model_copy(update={"elements": {"native": ElementCheckpoint(stage="fingerprint_ready")}})
    payload = checkpoint.model_dump(mode="json")
    payload.pop("dedupe_scan")
    element = payload["elements"]["native"]
    assert isinstance(element, dict)
    element.pop("source_descriptor_sha256")

    restored = RunCheckpoint.model_validate(payload)

    assert restored.dedupe_scan is None
    assert restored.elements["native"].source_descriptor_sha256 is None


@pytest.mark.parametrize("first_repainting", [False, True])
def test_deterministic_cache_keeps_both_render_contexts_for_identical_media(
    tmp_path: Path,
    first_repainting: bool,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    base_item = _item("same-native", unique_id="same-descriptor", file_id="same-file")
    items = {
        False: base_item,
        True: base_item.model_copy(update={"needs_repainting": True}),
    }
    base_processed = _processed(snapshot)
    values = {
        False: base_processed,
        True: base_processed.model_copy(
            update={
                "dark_frame_paths": (Path("synthetic-dark-frame.png"),),
                "has_dark_render": True,
            }
        ),
    }
    assert _deterministic_key(values[False]) == _deterministic_key(values[True])
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    order = (first_repainting, not first_repainting)
    for repainting in order:
        _cache_deterministic_analyses(
            cache,
            _collection((items[repainting],)),
            {items[repainting].native_id: values[repainting]},
        )

    try:
        assert cache.info()["metadata_entries"] == 2
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            restored = {}
            for repainting in (False, True):
                item = items[repainting]
                value = values[repainting]
                checkpoint = ElementCheckpoint(
                    stage="fingerprint_ready",
                    source_descriptor_sha256=_source_descriptor_sha256(item),
                    media_sha256=(value.metadata.sha256,),
                    deterministic_cache_key=_deterministic_key(value),
                    palette_complete=True,
                    fingerprint_complete=True,
                )
                restored[repainting] = _restore_deterministic_cache_entry(
                    cache,
                    item,
                    processor,
                    checkpoint,
                )
    finally:
        cache.close()

    assert restored[False] is not None
    assert restored[True] is not None
    assert restored[False].semantic_has_dark_render is False
    assert restored[True].semantic_has_dark_render is True


def test_transparent_non_repainting_media_preserves_dark_render_context(
    tmp_path: Path,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    item = _item("transparent", unique_id="transparent-descriptor", file_id="file")
    value = _processed(snapshot).model_copy(
        update={
            "dark_frame_paths": (Path("synthetic-dark-frame.png"),),
            "has_dark_render": True,
        }
    )
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    try:
        _cache_deterministic_analysis(cache, item, value)
        checkpoint = ElementCheckpoint(
            stage="fingerprint_ready",
            source_descriptor_sha256=_source_descriptor_sha256(item),
            media_sha256=(value.metadata.sha256,),
            deterministic_cache_key=_deterministic_key(value),
            palette_complete=True,
            fingerprint_complete=True,
        )
        with TemporaryMediaRun(root=tmp_path) as temporary:
            restored = _restore_deterministic_cache_entry(
                cache,
                item,
                _CountingProcessor(temporary, _analysis(snapshot)),
                checkpoint,
            )
    finally:
        cache.close()

    assert restored is not None
    assert restored.semantic_has_dark_render is True
