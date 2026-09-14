import asyncio
import gc
from collections import Counter

import pytest

from mojilex_cli.concurrency import bounded_map
from mojilex_cli.media import MediaLimitError, TemporaryMediaRun
from mojilex_cli.media.models import MediaRenderError
from mojilex_cli.media.processor import SourceChangedDuringRunError
from mojilex_cli.pipeline import runner
from mojilex_cli.sources.base import SourceNetworkError, SourceRateLimitError
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import (
    _PAYLOAD,
    _analysis,
    _collection,
    _CountingAdapter,
    _CountingProcessor,
    _item,
)


@pytest.mark.parametrize("error_type", [MediaRenderError, SourceNetworkError])
async def test_retry_only_failed_media_preserves_completed_siblings(
    tmp_path, monkeypatch, error_type
):
    snapshot = write_fixture(tmp_path / "d")
    source = _collection(
        tuple(_item(name, unique_id=name, file_id=name) for name in ("first", "second"))
    )
    counts = Counter()
    saved = []
    real_sleep = asyncio.sleep

    async def no_backoff(_):
        await real_sleep(0)

    monkeypatch.setattr(runner.asyncio, "sleep", no_backoff)

    class Adapter(_CountingAdapter):
        async def fetch_media(self, item):
            counts[item.native_id] += 1
            if item.native_id == "first" and counts[item.native_id] < 3:
                raise error_type("temporary media failure")
            async for chunk in super().fetch_media(item):
                yield chunk

    async def save(item, value):
        saved.append(item.native_id)

    with TemporaryMediaRun(root=tmp_path) as temporary:
        result = await runner._process_media(
            Adapter({"first": _PAYLOAD, "second": _PAYLOAD}),
            source,
            _CountingProcessor(temporary, _analysis(snapshot)),
            concurrency=2,
            max_attempts=3,
            on_item_completed=save,
        )
    assert set(result) == {"first", "second"}
    assert counts == {"first": 3, "second": 1}
    assert Counter(saved) == {"first": 1, "second": 1}


@pytest.mark.parametrize(
    "error_type", [MediaLimitError, SourceChangedDuringRunError, SourceRateLimitError]
)
async def test_safety_and_identity_failures_are_never_retried(tmp_path, error_type):
    snapshot = write_fixture(tmp_path / "d")
    item = _item("first", unique_id="first", file_id="first")
    calls = []

    class Adapter:
        async def fetch_media(self, item):
            calls.append(item.native_id)
            raise error_type("permanent")
            yield b""  # pragma: no cover

    with TemporaryMediaRun(root=tmp_path) as temporary:
        with pytest.raises(error_type):
            await runner._process_media(
                Adapter(),
                _collection((item,)),
                _CountingProcessor(temporary, _analysis(snapshot)),
                concurrency=2,
                max_attempts=3,
            )
    assert calls == ["first"]


async def test_renderer_failure_restarts_processing_but_saves_once(tmp_path, monkeypatch):
    snapshot = write_fixture(tmp_path / "d")
    item = _item("first", unique_id="first", file_id="first")
    saved = []
    real_sleep = asyncio.sleep

    async def no_backoff(_):
        await real_sleep(0)

    monkeypatch.setattr(runner.asyncio, "sleep", no_backoff)

    class Processor(_CountingProcessor):
        attempts = 0

        async def process_stream(self, chunks, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                async for _ in chunks:
                    pass
                raise MediaRenderError("worker transient failure")
            return await super().process_stream(chunks, **kwargs)

    async def save(item, value):
        saved.append(item.native_id)

    adapter = _CountingAdapter({"first": _PAYLOAD})
    with TemporaryMediaRun(root=tmp_path) as temporary:
        processor = Processor(temporary, _analysis(snapshot))
        result = await runner._process_media(
            adapter,
            _collection((item,)),
            processor,
            concurrency=2,
            max_attempts=3,
            on_item_completed=save,
        )
    assert set(result) == {"first"}
    assert processor.attempts == 2
    assert adapter.media_calls == ["first", "first"]
    assert saved == ["first"]


async def test_cancelled_pool_retrieves_delayed_gather_exception():
    loop = asyncio.get_running_loop()
    handler = loop.get_exception_handler()
    unhandled = []
    loop.set_exception_handler(lambda _, context: unhandled.append(context))
    try:
        for _ in range(20):
            started = asyncio.Event()
            cleaned = asyncio.Event()
            release = asyncio.Event()

            async def worker(value, started=started, cleaned=cleaned, release=release):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cleaned.set()
                    await release.wait()

            task = asyncio.create_task(bounded_map(range(2), worker, concurrency=2))
            await started.wait()
            task.cancel()
            await cleaned.wait()
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            del task
        gc.collect()
        await asyncio.sleep(0)
        assert not unhandled
    finally:
        loop.set_exception_handler(handler)


async def test_download_progress_advances_before_decode_finishes(tmp_path, monkeypatch):
    snapshot = write_fixture(tmp_path / "d")
    item = _item("first", unique_id="first", file_id="first")
    batches = []
    original_progress = runner.BatchProgress

    def progress(*args, **kwargs):
        batch = original_progress(*args, **kwargs)
        batches.append(batch)
        return batch

    monkeypatch.setattr(runner, "BatchProgress", progress)

    class Processor(_CountingProcessor):
        async def process_stream(self, chunks, **kwargs):
            payload = [chunk async for chunk in chunks]
            batch = batches[0]
            assert batch.downloaded == {"first"}
            assert batch.completed == 0
            assert batch.active == {"first": "render"}

            async def replay():
                for chunk in payload:
                    yield chunk

            return await super().process_stream(replay(), **kwargs)

    with TemporaryMediaRun(root=tmp_path) as temporary:
        await runner._process_media(
            _CountingAdapter({"first": _PAYLOAD}),
            _collection((item,)),
            Processor(temporary, _analysis(snapshot)),
            concurrency=1,
        )
    assert batches[0].completed == 1
