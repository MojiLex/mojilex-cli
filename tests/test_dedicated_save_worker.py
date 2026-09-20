import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar

import pytest

from mojilex_cli.concurrency import run_blocking, run_blocking_on


def test_dedicated_save_worker_progresses_while_media_pool_is_full():
    async def scenario():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()

        def media():
            loop.call_soon_threadsafe(started.set)
            release.wait()

        with (
            ThreadPoolExecutor(max_workers=1) as media_pool,
            ThreadPoolExecutor(max_workers=1) as save_pool,
        ):
            loop.set_default_executor(media_pool)
            media_task = asyncio.create_task(run_blocking(media))
            try:
                await asyncio.wait_for(started.wait(), 2)
                result = await asyncio.wait_for(run_blocking_on(save_pool, lambda: "saved"), 2)
                assert result == "saved"
                assert not media_task.done()
                assert not release.is_set()
            finally:
                release.set()
                await media_task

    asyncio.run(scenario())


@pytest.mark.parametrize("dedicated", [False, True])
async def test_worker_preserves_context_and_keyword_arguments(dedicated):
    context = ContextVar("save_worker_context", default="outside")
    token = context.set("inside")

    def read_context(value, *, suffix, executor):
        return context.get(), value + suffix, executor

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            if dedicated:
                result = await run_blocking_on(
                    executor, read_context, "a", suffix="b", executor="target keyword"
                )
            else:
                result = await run_blocking(
                    read_context, "a", suffix="b", executor="target keyword"
                )
        assert result == ("inside", "ab", "target keyword")
    finally:
        context.reset(token)


async def test_cancelled_save_drains_worker_before_owner_exits():
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()
    worker_finished = threading.Event()
    owner_exited = asyncio.Event()

    def save():
        loop.call_soon_threadsafe(started.set)
        release.wait()
        worker_finished.set()

    with ThreadPoolExecutor(max_workers=1) as executor:

        async def owner():
            try:
                await run_blocking_on(executor, save)
            finally:
                assert worker_finished.is_set()
                owner_exited.set()

        task = asyncio.create_task(owner())
        try:
            await asyncio.wait_for(started.wait(), 2)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert not owner_exited.is_set()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 2)
        assert owner_exited.is_set()
        assert worker_finished.is_set()
