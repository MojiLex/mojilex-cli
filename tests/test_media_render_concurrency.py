"""Network concurrency must not multiply decoder CPU/process budgets."""

import asyncio
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from mojilex_cli.media import MediaError, MediaProcessor, TemporaryMediaRun


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_render_concurrency_rejects_invalid_limits(tmp_path: Path, value: int) -> None:
    with TemporaryMediaRun(root=tmp_path) as run:
        with pytest.raises(ValueError, match="render concurrency"):
            MediaProcessor(run, render_concurrency=value)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2, 4])
async def test_downloads_overlap_while_decoders_stay_bounded(tmp_path: Path, limit: int) -> None:
    loop = asyncio.get_running_loop()
    downloads_ready = asyncio.Event()
    workers_ready = asyncio.Event()
    release = threading.Event()
    lock = threading.Lock()
    downloads = 0
    active = 0
    peak = 0
    calls = 0

    class BlockingWorker:
        def process(self, *args: object, **kwargs: object) -> object:
            nonlocal active, peak, calls
            with lock:
                active += 1
                peak = max(peak, active)
                calls += 1
                if active == limit:
                    loop.call_soon_threadsafe(workers_ready.set)
            try:
                assert release.wait(10), "test did not release the decoder"
                raise MediaError("synthetic decoder completion")
            finally:
                with lock:
                    active -= 1

    async def chunks() -> AsyncIterator[bytes]:
        nonlocal downloads
        yield b"synthetic-media"
        downloads += 1
        if downloads == 15:
            downloads_ready.set()

    with TemporaryMediaRun(root=tmp_path) as run:
        processor = MediaProcessor(run, worker=BlockingWorker(), render_concurrency=limit)  # type: ignore[arg-type]
        tasks = [
            asyncio.create_task(processor.process_stream(chunks(), expected_format="webp"))
            for _ in range(15)
        ]
        try:
            await asyncio.wait_for(downloads_ready.wait(), timeout=5)
            await asyncio.wait_for(workers_ready.wait(), timeout=5)
            assert calls == limit
            assert downloads == 15
            assert peak == limit
        finally:
            release.set()
            outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(value, MediaError) for value in outcomes)
        assert calls == 15
        assert peak <= limit


@pytest.mark.asyncio
async def test_cancellation_keeps_decoder_slot_until_thread_is_reaped(tmp_path: Path) -> None:
    loop = asyncio.get_running_loop()
    first_started = asyncio.Event()
    queued_downloaded = asyncio.Event()
    release = threading.Event()
    calls = 0

    class BlockingWorker:
        def process(self, *args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            loop.call_soon_threadsafe(first_started.set)
            assert release.wait(10), "test did not release the decoder"
            raise MediaError("synthetic decoder completion")

    async def chunks(*, queued: bool = False) -> AsyncIterator[bytes]:
        yield b"synthetic-media"
        if queued:
            queued_downloaded.set()

    with TemporaryMediaRun(root=tmp_path) as run:
        processor = MediaProcessor(run, worker=BlockingWorker(), render_concurrency=1)  # type: ignore[arg-type]
        first = asyncio.create_task(processor.process_stream(chunks(), expected_format="webp"))
        queued = asyncio.create_task(
            processor.process_stream(chunks(queued=True), expected_format="webp")
        )
        try:
            await asyncio.wait_for(first_started.wait(), timeout=5)
            await asyncio.wait_for(queued_downloaded.wait(), timeout=5)
            first.cancel()
            await asyncio.sleep(0)
            first.cancel()
            await asyncio.sleep(0)
            assert not first.done()
            assert calls == 1
            queued.cancel()
            await asyncio.sleep(0)
            assert queued.cancelled()
            assert run.path is not None and run.path.is_dir()
        finally:
            release.set()
            outcomes = await asyncio.gather(first, queued, return_exceptions=True)
        assert all(isinstance(value, asyncio.CancelledError) for value in outcomes)
        assert calls == 1
