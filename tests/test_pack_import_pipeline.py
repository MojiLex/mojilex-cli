"""Real import orchestration with event-gated synthetic Telegram media."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from mojilex_cli.config import MojiLexConfig, ProcessingConfig, TelegramConfig
from mojilex_cli.dataset.validation import validate_dataset
from mojilex_cli.media.models import MediaRenderError
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import RunStore
from mojilex_cli.sources import SourceCollection, SourceEmoji
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import (
    _PAYLOAD,
    _analysis,
    _collection,
    _CountingAdapter,
    _CountingProcessor,
    _item,
)


def _pack(name: str, *items: str) -> SourceCollection:
    return _collection(
        tuple(_item(item, unique_id=item, file_id=item) for item in items), native_id=name
    )


async def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sources: Sequence[SourceCollection],
    before_media: Callable[[SourceEmoji], Awaitable[None]],
    *,
    pack_concurrency: int = 3,
    file_analysis_mode: str = "sequential",
) -> SimpleNamespace:
    snapshot = write_fixture(tmp_path / "d")
    (snapshot.root / ".gitignore").write_text(".mojilex/\n", encoding="utf-8")
    for args in (
        ("init", "-b", "main"),
        ("config", "core.longpaths", "true"),
        ("add", "."),
        ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "base"),
        ("remote", "add", "origin", "https://github.com/MojiLex/mojilex.git"),
    ):
        await asyncio.to_thread(
            subprocess.run,
            ["git", "-C", str(snapshot.root), *args],
            capture_output=True,
            check=True,
        )
    config = MojiLexConfig(
        repository={"target": str(snapshot.root), "publish": "local"},
        processing=ProcessingConfig(
            pack_concurrency=pack_concurrency,
            file_analysis_mode=file_analysis_mode,
        ),
        telegram=TelegramConfig(download_concurrency=1),
        cache_dir=tmp_path / "c",
        runs_dir=tmp_path / "r",
    )
    state = SimpleNamespace(
        config=config,
        run_ids=[],
        fetched=[],
        downloads=[],
        processors=[],
        prepare_entered=[],
        prepare_finished=[],
        first_decode_download_count=None,
        metadata_ready=asyncio.Event(),
        prepare_events={source.native_id: asyncio.Event() for source in sources},
    )
    collections = {source.canonical_url: source for source in sources}

    class Adapter(_CountingAdapter):
        def __init__(self, *_args, **_kwargs):
            super().__init__({item.file_id: _PAYLOAD for src in sources for item in src.items})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            pass

        async def validate_credentials(self):
            pass

        def canonicalize(self, value):
            return value

        async def fetch_collection(self, reference):
            source = collections[reference]
            state.fetched.append(source.native_id)
            if len(state.fetched) == len(sources):
                state.metadata_ready.set()
            return source

        async def fetch_media(self, item):
            state.downloads.append(item.native_id)
            await before_media(item)
            async for chunk in super().fetch_media(item):
                yield chunk

    class Processor(_CountingProcessor):
        async def process_stream(self, *args, **kwargs):
            if state.first_decode_download_count is None:
                state.first_decode_download_count = len(state.downloads)
            value = await super().process_stream(*args, **kwargs)
            assert self.run.path is not None
            frame = self.run.path / f"frame-{self.decode_calls}.png"
            Image.new("RGB", (256, 256), "white").save(frame)
            return value.model_copy(update={"frame_paths": (frame,)})

    def processor(temporary, **_kwargs):
        value = Processor(temporary, _analysis(snapshot))
        state.processors.append(value)
        return value

    prepare = runner._prepare_collection_media

    async def observed_prepare(snapshot, adapter, source, processor, **kwargs):
        state.prepare_entered.append(source.native_id)
        state.prepare_events[source.native_id].set()
        try:
            return await prepare(snapshot, adapter, source, processor, **kwargs)
        finally:
            state.prepare_finished.append(source.native_id)

    monkeypatch.setattr(runner, "load_config", lambda: config)
    monkeypatch.setattr(runner, "_resolved_config", lambda _options: config)
    # The synthetic dataset has real domain records but omits external JSON schemas.
    monkeypatch.setattr(
        runner, "validate_dataset", lambda root, **_: validate_dataset(root, strict=False)
    )
    monkeypatch.setattr(
        runner,
        "load_credentials",
        lambda: SimpleNamespace(telegram_bot_token="fixture", github_token=None),
    )
    monkeypatch.setattr(runner, "TelegramBotAPI", Adapter)
    monkeypatch.setattr(runner, "MediaProcessor", processor)
    monkeypatch.setattr(runner, "report_run_id", state.run_ids.append)
    monkeypatch.setattr(runner, "_prepare_collection_media", observed_prepare)
    state.sources = tuple(source.canonical_url for source in sources)
    return state


async def _cancel(task: asyncio.Task) -> None:
    if not task.done():
        task.cancel()
    results = await asyncio.gather(task, return_exceptions=True)
    for result in results:
        if isinstance(result, BaseException) and not isinstance(result, asyncio.CancelledError):
            raise result


async def _wait_for_gate(event: asyncio.Event, task: asyncio.Task) -> None:
    """Report early import failure rather than waiting for an event it cannot emit."""
    waiter = asyncio.create_task(event.wait())
    try:
        done, _ = await asyncio.wait(
            {waiter, task}, timeout=15, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            result = task.result()
            assert not result.errors, result.errors
            assert event.is_set(), "Import finished before the expected media event"
        if waiter not in done and not event.is_set():
            raise TimeoutError("Import did not reach the expected media event within 15 seconds")
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.parametrize("pack_concurrency", [1, 3])
async def test_import_finishes_current_pack_before_starting_next(
    tmp_path_factory, monkeypatch, pack_concurrency
):
    release = asyncio.Event()
    active_ready = asyncio.Event()
    active: set[str] = set()
    maximum = 0

    async def block(item):
        nonlocal maximum
        active.add(item.native_id)
        maximum = max(maximum, len(active))
        if len(active) == 1:
            active_ready.set()
        try:
            await release.wait()
        finally:
            active.remove(item.native_id)

    sources = tuple(_pack(f"ParallelPack{i}", f"item{i}") for i in range(5))
    state = await _harness(
        tmp_path_factory.mktemp("pi"),
        monkeypatch,
        sources,
        block,
        pack_concurrency=pack_concurrency,
    )
    task = asyncio.create_task(runner._run_import(state.sources, runner.PipelineOptions()))
    try:
        await _wait_for_gate(active_ready, task)
        assert len(state.fetched) == 1
        assert len(state.prepare_entered) == 1
        assert not state.prepare_finished
        assert active == {"item0"}
        release.set()
        result = await asyncio.wait_for(task, timeout=15)
    finally:
        await _cancel(task)
    assert not result.errors
    assert result.result["collections_imported"] == 5
    assert maximum == 1
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    assert len(checkpoint.elements) == 5
    assert len(checkpoint.safe_parameters["source_memberships"]) == 5
    assert checkpoint.ai_requests_used == 0


async def test_prepare_all_starts_independent_packs_together(tmp_path_factory, monkeypatch):
    release = asyncio.Event()

    async def block(_item):
        await release.wait()

    sources = tuple(_pack(f"PreparePack{i}", f"prepare{i}") for i in range(3))
    state = await _harness(
        tmp_path_factory.mktemp("prepare-all"),
        monkeypatch,
        sources,
        block,
        pack_concurrency=3,
        file_analysis_mode="prepare_all",
    )
    task = asyncio.create_task(runner._run_import(state.sources, runner.PipelineOptions()))
    try:
        await _wait_for_gate(state.metadata_ready, task)
        assert set(state.prepare_entered) == {source.native_id for source in sources}
        release.set()
        result = await asyncio.wait_for(task, timeout=15)
    finally:
        await _cancel(task)
    assert not result.errors
    assert result.result["collections_imported"] == 3


async def test_metadata_strategy_defers_all_media_to_describe(tmp_path_factory, monkeypatch):
    async def unexpected(_item):
        pytest.fail("metadata-only import started a media download")

    sources = (_pack("MetadataOnly", "metadata-item"),)
    state = await _harness(
        tmp_path_factory.mktemp("metadata-only"), monkeypatch, sources, unexpected
    )
    result = await runner._run_import(
        state.sources,
        runner.PipelineOptions(import_strategy="metadata"),
    )
    assert not result.errors
    assert result.result["collections_imported"] == 1
    assert state.downloads == []
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    assert checkpoint.elements == {}
    assert checkpoint.safe_parameters["import_strategy"] == "metadata"


async def test_download_all_finishes_network_phase_before_first_decode(
    tmp_path_factory, monkeypatch
):
    async def allow(_item):
        return None

    sources = (
        _pack("DownloadFirstA", "download-a"),
        _pack("DownloadFirstB", "download-b"),
    )
    state = await _harness(
        tmp_path_factory.mktemp("download-all"),
        monkeypatch,
        sources,
        allow,
        pack_concurrency=2,
        file_analysis_mode="download_all",
    )
    result = await runner._run_import(
        state.sources,
        runner.PipelineOptions(import_strategy="download_all"),
    )
    assert not result.errors
    assert state.first_decode_download_count == 2
    assert sorted(state.downloads) == ["download-a", "download-b"]
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    assert all(element.fingerprint_complete for element in checkpoint.elements.values())


async def test_download_all_resume_reuses_each_persisted_original(tmp_path_factory, monkeypatch):
    blocked = asyncio.Event()
    release = asyncio.Event()

    async def block_second(item):
        if item.native_id == "download-second":
            blocked.set()
            await release.wait()

    state = await _harness(
        tmp_path_factory.mktemp("raw-resume"),
        monkeypatch,
        (_pack("RawResume", "download-first", "download-second"),),
        block_second,
        file_analysis_mode="download_all",
    )
    task = asyncio.create_task(
        runner._run_import(
            state.sources,
            runner.PipelineOptions(import_strategy="download_all"),
        )
    )
    try:
        await _wait_for_gate(blocked, task)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await _cancel(task)

    checkpoint = RunStore(state.config.runs_dir).load(state.run_ids[0])
    assert checkpoint.status == "interrupted"
    assert checkpoint.elements["download-first"].stage == "media_verified"
    assert state.downloads == ["download-first", "download-second"]

    release.set()
    resumed = await runner.run_resume(checkpoint.run_id)
    assert not resumed.errors
    assert resumed.status.value == "succeeded"
    assert state.downloads == ["download-first", "download-second", "download-second"]
    assert state.first_decode_download_count == 3


async def test_unresolved_pack_failure_stops_queue_and_resume_continues_in_order(
    tmp_path_factory, monkeypatch
):
    failing = True

    async def maybe_fail(item):
        if item.native_id == "bad" and failing:
            raise MediaRenderError("synthetic per-pack rendering failure")

    state = await _harness(
        tmp_path_factory.mktemp("pe"),
        monkeypatch,
        (_pack("FailurePack", "bad"), _pack("GoodOne", "one"), _pack("GoodTwo", "two")),
        maybe_fail,
    )
    result = await runner._run_import(state.sources, runner.PipelineOptions())
    assert result.status.value == "failed"
    assert result.result["collections_imported"] == 0
    assert len(result.errors) == 1
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    assert set(checkpoint.elements) == set()
    assert all(element.fingerprint_complete for element in checkpoint.elements.values())
    assert len(checkpoint.safe_parameters["source_memberships"]) == 1
    assert state.fetched == ["FailurePack"]
    assert state.downloads == ["bad"] * state.config.telegram.max_attempts
    prior_downloads = len(state.downloads)
    prior_processors = len(state.processors)
    failing = False
    resumed = await runner.run_resume(result.run_id)
    assert not resumed.errors
    assert resumed.status.value == "succeeded"
    assert state.downloads[prior_downloads:] == ["bad", "one", "two"]
    assert sum(p.decode_calls for p in state.processors[prior_processors:]) == 3
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    assert set(checkpoint.elements) == {"bad", "one", "two"}


async def test_cancelled_import_keeps_current_pack_progress_and_resumes_missing_files(
    tmp_path_factory, monkeypatch
):
    release = asyncio.Event()
    blocked_ready = asyncio.Event()
    blocked = set()

    async def block_second(item):
        if item.native_id.endswith("second"):
            blocked.add(item.native_id)
            if len(blocked) == 1:
                blocked_ready.set()
            await release.wait()

    state = await _harness(
        tmp_path_factory.mktemp("pc"),
        monkeypatch,
        (_pack("CancelOne", "onefirst", "onesecond"), _pack("CancelTwo", "twofirst", "twosecond")),
        block_second,
    )
    task = asyncio.create_task(runner._run_import(state.sources, runner.PipelineOptions()))
    try:
        await _wait_for_gate(blocked_ready, task)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await _cancel(task)
    checkpoint = RunStore(state.config.runs_dir).load(state.run_ids[0])
    assert checkpoint.status == "interrupted"
    assert set(checkpoint.elements) == {"onefirst"}
    assert all(element.fingerprint_complete for element in checkpoint.elements.values())
    assert len(checkpoint.safe_parameters["source_memberships"]) == 1
    prior_downloads = len(state.downloads)
    prior_processors = len(state.processors)
    release.set()
    resumed = await runner.run_resume(checkpoint.run_id)
    assert not resumed.errors
    assert state.downloads[prior_downloads:] == ["onesecond", "twofirst", "twosecond"]
    assert sum(p.decode_calls for p in state.processors[prior_processors:]) == 3
    checkpoint = RunStore(state.config.runs_dir).load(checkpoint.run_id)
    assert checkpoint.status == "succeeded"
    assert len(checkpoint.elements) == 4
    assert checkpoint.ai_requests_used == 0


@pytest.mark.parametrize("duplicate_pack", [False, True])
async def test_shared_emoji_or_duplicate_pack_reuses_completed_owner_before_next_pack(
    tmp_path_factory, monkeypatch, duplicate_pack
):
    shared_blocked = asyncio.Event()
    independent_blocked = asyncio.Event()
    release_shared = asyncio.Event()
    release_independent = asyncio.Event()

    async def gate(item):
        if item.native_id == "shared":
            shared_blocked.set()
            await release_shared.wait()
        elif item.native_id == "independent":
            independent_blocked.set()
            await release_independent.wait()

    first = _pack("OwnerPack", "shared", "owneronly")
    second = first if duplicate_pack else _pack("DependentPack", "shared", "dependentonly")
    state = await _harness(
        tmp_path_factory.mktemp("pd"),
        monkeypatch,
        (first, second, _pack("IndependentPack", "independent")),
        gate,
    )
    task = asyncio.create_task(runner._run_import(state.sources, runner.PipelineOptions()))
    try:
        await _wait_for_gate(shared_blocked, task)
        assert state.prepare_entered == ["OwnerPack"]
        assert state.downloads.count("shared") == 1
        release_shared.set()
        await _wait_for_gate(independent_blocked, task)
        assert state.prepare_entered == ["OwnerPack", second.native_id, "IndependentPack"]
        release_independent.set()
        result = await asyncio.wait_for(task, timeout=15)
    finally:
        await _cancel(task)
    assert not result.errors
    assert result.result["collections_imported"] == 3
    assert state.prepare_entered[-1] == "IndependentPack"
    assert state.downloads.count("shared") == 1
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    expected = {item.native_id for source in (first, second) for item in source.items}
    assert set(checkpoint.elements) == expected | {"independent"}
    assert checkpoint.safe_parameters["source_memberships"][first.native_id] == [
        "shared",
        "owneronly",
    ]
    if not duplicate_pack:
        assert checkpoint.safe_parameters["source_memberships"][second.native_id] == [
            "shared",
            "dependentonly",
        ]
