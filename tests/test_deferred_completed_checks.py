# ruff: noqa: F811
import asyncio
import threading
from types import SimpleNamespace

import pytest

from mojilex_cli.commands import import_reuse
from mojilex_cli.commands.queue_progress import SHARED, pack_queue_scope
from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.pipeline import runner
from test_import_reuse import A, B, do_import, saved_runs  # noqa: F401
from test_saved_run_pipeline import saved_queue  # noqa: F401


def test_interactive_fast_import_defers_completed_checks(saved_runs, monkeypatch):
    create, _, _ = saved_runs
    ready = create((A,), phase="describe")
    unfinished = create((B,), phase="import")

    def must_not_wait(*args, **kwargs):
        pytest.fail("completed metadata check blocked import")

    monkeypatch.setattr(import_reuse, "refresh_completed_imports", must_not_wait)
    with pack_queue_scope():
        result = do_import([A, B], preparation="metadata")
        assert SHARED.get().completed_checks == {A: (ready.run_id, A)}
    assert result.result["analysis_selectors"] == [
        f"{ready.run_id}:PackAlpha",
        f"{unfinished.run_id}:PackBravo",
    ]


@pytest.mark.parametrize("changed", [False, True])
async def test_completed_metadata_does_not_block_unfinished_work_or_overwrite_writer(
    saved_queue, monkeypatch, changed
):
    _, locks = saved_queue
    active = asyncio.Event()
    fetched = asyncio.Event()
    finished = False
    calls = []

    async def metadata(sources, config, **kwargs):
        assert sources == ("Done",)
        await asyncio.wait_for(active.wait(), 2)
        fetched.set()
        return {"Done": object()}

    async def describe(run_id, options):
        nonlocal finished
        calls.append(options.selected_sources)
        if options.selected_sources == ("Todo",):
            active.set()
            await asyncio.wait_for(fetched.wait(), 2)
            finished = True
        return CommandResult(run_id=run_id, status="succeeded")

    def refresh(existing, config, **kwargs):
        assert finished
        assert not locks
        assert "Done" in kwargs["fresh"]
        return {"Done": (SimpleNamespace(run_id="first"), "Done")}

    monkeypatch.setattr(import_reuse, "_fetch_completed_metadata", metadata)
    monkeypatch.setattr(import_reuse, "refresh_completed_imports", refresh)
    monkeypatch.setattr(import_reuse, "_completed_source", lambda *_: not changed)
    monkeypatch.setattr(runner, "_run_describe", describe)
    with pack_queue_scope():
        queue = SHARED.get()
        queue.completed_checks["Done"] = ("first", "Done")
        result = await asyncio.wait_for(
            runner._run_describe_many([("first", ("Done", "Todo"))], runner.PipelineOptions()), 4
        )
        assert not queue.completed_checks
    assert result.status == "succeeded"
    assert calls == ([("Todo",), ("Done",)] if changed else [("Todo",)])


async def test_failed_metadata_check_does_not_cancel_unfinished_work(saved_queue, monkeypatch):
    calls = []

    async def metadata(*args, **kwargs):
        raise RuntimeError("offline")

    async def describe(run_id, options):
        await asyncio.sleep(0)
        calls.append(options.selected_sources)
        return CommandResult(run_id=run_id, status="succeeded")

    monkeypatch.setattr(import_reuse, "_fetch_completed_metadata", metadata)
    monkeypatch.setattr(runner, "_run_describe", describe)
    with pack_queue_scope():
        SHARED.get().completed_checks["Done"] = ("first", "Done")
        result = await runner._run_describe_many(
            [("first", ("Done", "Todo"))], runner.PipelineOptions()
        )
    assert calls == [("Todo",)]
    assert result.errors
    assert result.status == "failed"


async def test_unlimited_queue_starts_without_waiting_for_slow_saved_run(saved_queue, monkeypatch):
    config, _ = saved_queue
    config = config.model_copy(
        update={"ai": config.ai.model_copy(update={"max_ai_requests": None, "max_cost_usd": None})}
    )
    monkeypatch.setattr(runner, "_resolved_config", lambda _: config)
    started = threading.Event()
    original = runner.RunStore.load_for_resume

    def load(self, run_id, **kwargs):
        if run_id == "second":
            assert started.wait(5), "slow restoration blocked an independently ready run"
        return original(self, run_id, **kwargs)

    async def describe(run_id, options):
        if run_id == "first":
            started.set()
        return CommandResult(run_id=run_id, status="succeeded")

    monkeypatch.setattr(runner.RunStore, "load_for_resume", load)
    monkeypatch.setattr(runner, "_run_describe", describe)
    result = await runner._run_describe_many(
        [("first", ("A",)), ("second", ("B",))], runner.PipelineOptions()
    )
    assert result.status == "succeeded"
    assert started.is_set()
