import asyncio
from contextlib import contextmanager
from decimal import Decimal
from types import SimpleNamespace

import pytest

from mojilex_cli.ai.base import BudgetExceededError, CostEstimate, RequestBudget
from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.concurrency import current_batch_limits
from mojilex_cli.config import MojiLexConfig
from mojilex_cli.pipeline import runner


@pytest.fixture
def saved_queue(monkeypatch, tmp_path):
    config = MojiLexConfig(runs_dir=tmp_path)
    config = config.model_copy(
        update={"processing": config.processing.model_copy(update={"pack_concurrency": 2})}
    )
    locks = set()

    class Store:
        def __init__(self, *_):
            pass

        def load_for_resume(self, run_id, **kwargs):
            return SimpleNamespace(ai_requests_used=0, ai_cost_reserved_usd=0)

        @contextmanager
        def execution_lock(self, run_id):
            assert run_id not in locks
            locks.add(run_id)
            try:
                yield
            finally:
                locks.remove(run_id)

    monkeypatch.setattr(runner, "RunStore", Store)
    monkeypatch.setattr(runner, "_resolved_config", lambda _: config)
    return config, locks


async def test_saved_groups_run_together_and_reunite_alternating_selectors(
    saved_queue, monkeypatch
):
    _, locks = saved_queue
    started = {}
    limits = []
    both = asyncio.Event()

    async def describe(run_id, options):
        started[run_id] = options.selected_sources
        limits.append(current_batch_limits())
        if len(started) == 2:
            both.set()
        await asyncio.wait_for(both.wait(), 2)
        return CommandResult(run_id=run_id, status="succeeded")

    monkeypatch.setattr(runner, "_run_describe", describe)
    result = await runner._run_describe_many(
        [("first", ("A",)), ("second", ("B",)), ("first", ("C",)), ("second", ("A",))],
        runner.PipelineOptions(file_analysis_mode="fast"),
    )
    assert result.status == "succeeded"
    assert started == {"first": ("A", "C"), "second": ("B",)}
    assert limits[0] is limits[1] and limits[0] is not None
    assert not locks
    assert current_batch_limits() is None


async def test_saved_group_failure_cancels_and_drains_other_run(saved_queue, monkeypatch):
    _, locks = saved_queue
    second_started = asyncio.Event()
    second_cleaned = asyncio.Event()

    async def describe(run_id, options):
        if run_id == "first":
            await second_started.wait()
            raise RuntimeError("terminal")
        second_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            second_cleaned.set()

    monkeypatch.setattr(runner, "_run_describe", describe)
    with pytest.raises(RuntimeError, match="terminal"):
        await runner._run_describe_many(
            [("first", ("A",)), ("second", ("B",))], runner.PipelineOptions()
        )
    assert second_cleaned.is_set()
    assert not locks
    assert current_batch_limits() is None


@pytest.mark.parametrize("by_cost", [False, True])
async def test_multiple_saved_runs_cannot_multiply_ai_budget(saved_queue, monkeypatch, by_cost):
    config, _ = saved_queue
    config = config.model_copy(
        update={
            "ai": config.ai.model_copy(
                update={
                    "max_ai_requests": None if by_cost else 3,
                    "max_cost_usd": Decimal("0.06") if by_cost else None,
                }
            )
        }
    )
    monkeypatch.setattr(runner, "_resolved_config", lambda _: config)
    paid = []
    started = set()
    both = asyncio.Event()

    async def describe(run_id, options):
        local = RequestBudget(max_requests=None)
        started.add(run_id)
        if len(started) == 2:
            both.set()
        await both.wait()
        for _ in range(3):
            try:
                await local.reserve(CostEstimate(upper_bound_usd=Decimal("0.02"), note="test"))
            except BudgetExceededError:
                break
            paid.append(run_id)
            await asyncio.sleep(0)
        return CommandResult(
            run_id=run_id,
            status="succeeded",
            result={
                "sources_processed": 1,
                "ai_requests": local.requests_used,
                "ai_cost_reserved_usd": str(local.cost_reserved),
            },
        )

    monkeypatch.setattr(runner, "_run_describe", describe)
    result = await runner._run_describe_many(
        [("first", ("A",)), ("second", ("B",))], runner.PipelineOptions()
    )
    assert len(paid) == result.result["ai_requests"] == 3
    assert result.result["ai_cost_reserved_usd"] == "0.06"
    assert result.result["sources_processed"] == 2
