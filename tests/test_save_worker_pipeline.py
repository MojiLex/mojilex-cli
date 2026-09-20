import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

from mojilex_cli.concurrency import run_blocking
from mojilex_cli.pipeline import runner
from test_add_run_staging import saved_add  # noqa: F401
from test_incremental_pack_readiness import _assert_durable_pack, _real_packs
from test_pack_describe_pipeline import pipeline  # noqa: F401


def test_ready_pack_is_saved_while_peer_media_saturates_default_executor(request, monkeypatch):
    state = request.getfixturevalue("pipeline")
    _real_packs(state, monkeypatch, dedupe="exact")
    alpha, beta = state.sources

    async def scenario():
        loop = asyncio.get_running_loop()
        started = [asyncio.Event(), asyncio.Event()]
        release_media = threading.Event()
        alpha_ready = asyncio.Event()
        stages = []
        media_tasks = []

        def blocked_decoder(index):
            loop.call_soon_threadsafe(started[index].set)
            release_media.wait()

        def stage(source, phase):
            stages.append((source, phase))
            if source == alpha.canonical_url and phase == "ready":
                alpha_ready.set()

        async def media(snapshot, adapter, source, processor, **kwargs):
            if source.native_id == beta.native_id:
                # Start only after the runner's shared initialization is done.
                media_tasks.extend(
                    asyncio.create_task(run_blocking(blocked_decoder, index)) for index in range(2)
                )
                await asyncio.gather(*media_tasks)
            else:
                await asyncio.gather(*(event.wait() for event in started))
            return await state.media(snapshot, adapter, source, processor, **kwargs)

        monkeypatch.setattr(runner, "report_pack_stage", stage)
        monkeypatch.setattr(runner, "_prepare_collection_media", media)
        with ThreadPoolExecutor(max_workers=2) as media_pool:
            loop.set_default_executor(media_pool)
            task = asyncio.create_task(state.run())
            try:
                await asyncio.wait_for(alpha_ready.wait(), 30)
                assert not task.done()
                assert len(media_tasks) == 2 and all(not job.done() for job in media_tasks)
                assert not release_media.is_set()
                assert (beta.canonical_url, "ready") not in stages
                checkpoint = _assert_durable_pack(state, alpha)
                assert checkpoint.elements[alpha.items[0].native_id].candidate_scan_complete

                release_media.set()
                result = await asyncio.wait_for(task, 30)
                assert not result.errors
                _assert_durable_pack(state, beta)
            finally:
                # Release threads before cancelling/draining their owners.
                release_media.set()
                if not task.done():
                    task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
                if media_tasks:
                    await asyncio.gather(*media_tasks, return_exceptions=True)

    asyncio.run(scenario())
