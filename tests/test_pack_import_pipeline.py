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
        processing=ProcessingConfig(pack_concurrency=pack_concurrency),
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
    await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("pack_concurrency", [1, 3])
async def test_import_bounds_active_packs_and_overlaps_disjoint_media(
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
        if len(active) == pack_concurrency:
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
        await asyncio.wait_for(active_ready.wait(), timeout=15)
        assert len(state.fetched) == pack_concurrency
        assert len(state.prepare_entered) == pack_concurrency
        assert not state.prepare_finished
        assert active == {f"item{i}" for i in range(pack_concurrency)}
        release.set()
        result = await asyncio.wait_for(task, timeout=15)
    finally:
        await _cancel(task)
    assert not result.errors
    assert result.result["collections_imported"] == 5
    assert maximum == pack_concurrency
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    assert len(checkpoint.elements) == 5
    assert len(checkpoint.safe_parameters["source_memberships"]) == 5
    assert checkpoint.ai_requests_used == 0


async def test_recoverable_pack_failure_keeps_other_packs_and_resumes_only_missing(
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
    assert result.status.value == "partial"
    assert result.result["collections_imported"] == 2
    assert len(result.errors) == 1
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    assert set(checkpoint.elements) == {"one", "two"}
    assert all(element.fingerprint_complete for element in checkpoint.elements.values())
    assert len(checkpoint.safe_parameters["source_memberships"]) == 3
    prior_downloads = len(state.downloads)
    prior_processors = len(state.processors)
    failing = False
    resumed = await runner.run_resume(result.run_id)
    assert not resumed.errors
    assert resumed.status.value == "succeeded"
    assert state.downloads[prior_downloads:] == ["bad"]
    assert sum(p.decode_calls for p in state.processors[prior_processors:]) == 1
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    assert set(checkpoint.elements) == {"bad", "one", "two"}


async def test_cancelled_parallel_import_keeps_each_pack_progress_and_resumes_missing_files(
    tmp_path_factory, monkeypatch
):
    release = asyncio.Event()
    blocked_ready = asyncio.Event()
    blocked = set()

    async def block_second(item):
        if item.native_id.endswith("second"):
            blocked.add(item.native_id)
            if len(blocked) == 2:
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
        await asyncio.wait_for(blocked_ready.wait(), timeout=15)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await _cancel(task)
    checkpoint = RunStore(state.config.runs_dir).load(state.run_ids[0])
    assert checkpoint.status == "interrupted"
    assert set(checkpoint.elements) == {"onefirst", "twofirst"}
    assert all(element.fingerprint_complete for element in checkpoint.elements.values())
    assert len(checkpoint.safe_parameters["source_memberships"]) == 2
    prior_downloads = len(state.downloads)
    prior_processors = len(state.processors)
    release.set()
    resumed = await runner.run_resume(checkpoint.run_id)
    assert not resumed.errors
    assert set(state.downloads[prior_downloads:]) == {"onesecond", "twosecond"}
    assert len(state.downloads[prior_downloads:]) == 2
    assert sum(p.decode_calls for p in state.processors[prior_processors:]) == 2
    checkpoint = RunStore(state.config.runs_dir).load(checkpoint.run_id)
    assert checkpoint.status == "succeeded"
    assert len(checkpoint.elements) == 4
    assert checkpoint.ai_requests_used == 0


@pytest.mark.parametrize("duplicate_pack", [False, True])
async def test_shared_emoji_or_duplicate_pack_waits_for_owner_while_disjoint_pack_runs(
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
        await asyncio.wait_for(
            asyncio.gather(
                shared_blocked.wait(), independent_blocked.wait(), state.metadata_ready.wait()
            ),
            timeout=15,
        )
        assert state.prepare_entered == ["OwnerPack", "IndependentPack"]
        assert state.downloads.count("shared") == 1
        release_shared.set()
        release_independent.set()
        result = await asyncio.wait_for(task, timeout=15)
    finally:
        await _cancel(task)
    assert not result.errors
    assert result.result["collections_imported"] == 3
    assert state.prepare_entered[-1] == second.native_id
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
