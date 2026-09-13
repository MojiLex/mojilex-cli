from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from mojilex_cli.ai import AIError, AIOutputError, CostEstimate, RequestBudget
from mojilex_cli.ai.base import UnknownCostError
from mojilex_cli.cache import CacheStore
from mojilex_cli.commands import progress as progress_module
from mojilex_cli.config import AIConfig, MojiLexConfig, ProcessingConfig
from mojilex_cli.media import TemporaryMediaRun
from mojilex_cli.pipeline import runner
from test_dataset_helpers import write_fixture
from test_pipeline_identity_media import _processed, _source


@pytest.mark.parametrize("concurrency", [1, 2])
async def test_first_ai_failure_stops_queued_batches_but_saves_active_success(
    tmp_path, monkeypatch, concurrency
):
    snapshot = write_fixture(tmp_path / "dataset")
    source = _source(snapshot)
    items = tuple(
        source.items[0].model_copy(update={"native_id": str(1000 + index), "position": index})
        for index in range(8)
    )
    source = source.model_copy(update={"items": items, "item_count": len(items)})
    processed = {item.native_id: _processed(snapshot, "a" * 64) for item in items}
    config = MojiLexConfig(
        ai=AIConfig(model="test-model", ai_concurrency=concurrency),
        processing=ProcessingConfig(static_batch_size=2),
    )
    monkeypatch.setattr(runner, "_needs_generated_description", lambda *args, **kwargs: True)
    monkeypatch.setattr(runner, "_recover_ai_request_traces", lambda *args, **kwargs: {})
    calls = []
    saved = []
    reports = []
    counters = []
    monkeypatch.setattr(runner, "report_progress", reports.append)
    monkeypatch.setattr(progress_module, "report_progress", reports.append)

    def make_progress(*args, **kwargs):
        counter = progress_module.BatchProgress(*args, **kwargs)
        counters.append(counter)
        return counter

    monkeypatch.setattr(runner, "BatchProgress", make_progress)
    second_started = asyncio.Event()
    first_failed = asyncio.Event()
    failure = (
        UnknownCostError("approval declined")
        if concurrency == 1
        else AIError("authentication failed")
    )

    async def describe(chunk, *_args, **_kwargs):
        calls.append(chunk[0].native_id)
        callback = runner._AI_PROGRESS_CALLBACK.get()
        assert callback is not None
        callback("approval")
        callback("request")
        if chunk[0].native_id == items[0].native_id:
            if concurrency == 2:
                await second_started.wait()
            callback("retry")
            first_failed.set()
            raise failure
        second_started.set()
        await first_failed.wait()
        await asyncio.sleep(0)
        return {item.native_id: SimpleNamespace(request_trace=("trace",)) for item in chunk}

    async def checkpoint(chunk, generated):
        assert set(generated) == {item.native_id for item in chunk}
        saved.extend(item.native_id for item in chunk)

    monkeypatch.setattr(runner, "_describe_batch", describe)
    with (
        CacheStore(tmp_path / "cache.sqlite3") as cache,
        TemporaryMediaRun(root=tmp_path) as temporary,
    ):
        with pytest.raises(type(failure)) as caught:
            await runner._descriptions_for_collection(
                snapshot,
                source,
                processed,
                config=config,
                cache=cache,
                budget=RequestBudget(max_requests=3),
                ai_state=runner._AIState(),
                api_key=None,
                redescribe="all",
                overwrite_reviewed=False,
                temporary=temporary,
                on_chunk_completed=checkpoint,
            )
    assert caught.value is failure
    assert calls == [items[index * 2].native_id for index in range(concurrency)]
    assert saved == ([items[2].native_id, items[3].native_id] if concurrency == 2 else [])
    assert counters[0].failed == 2
    assert counters[0].completed == 2 * (concurrency - 1)
    assert counters[0].active == {}
    assert counters[0].queue_stopped is True
    assert f"not started: {8 - 2 * concurrency} emojis" in reports[-1]
    assert "stopped" in reports[-1]
    assert any("AI batch 1/4: 2 emojis" in line and "retrying" in line for line in reports)
    assert runner._AI_PROGRESS_CALLBACK.get() is None


async def test_pipeline_single_request_forwards_real_recovery_events():
    events = []
    attempts = []
    request = SimpleNamespace(model="test-model", expected_labels=("A",))
    result = SimpleNamespace(
        provider="gemini",
        model=request.model,
        batch=SimpleNamespace(items=(SimpleNamespace(label="A"),)),
    )

    async def describe(actual_request):
        assert actual_request is request
        attempts.append(1)
        if len(attempts) == 1:
            raise AIOutputError("synthetic invalid response")
        return result

    provider = SimpleNamespace(
        name="gemini",
        describe=describe,
        estimate=lambda _request: CostEstimate(upper_bound_usd=0, note="test"),
    )
    token = runner._AI_PROGRESS_CALLBACK.set(events.append)
    budget = RequestBudget(max_requests=2)
    try:
        actual = await runner._describe_single(
            None,
            None,
            model=request.model,
            budget=budget,
            provider=provider,
            context=None,
            temporary=None,
            prepared_request=SimpleNamespace(request=request),
        )
    finally:
        runner._AI_PROGRESS_CALLBACK.reset(token)
    assert actual is result
    assert budget.requests_used == 2
    assert events == ["approval", "request", "retry", "approval", "request"]


@pytest.mark.parametrize("concurrency", [1, 4])
async def test_ai_scheduler_overlaps_batches_with_shared_budget(tmp_path, monkeypatch, concurrency):
    snapshot = write_fixture(tmp_path / "dataset")
    source = _source(snapshot)
    items = tuple(
        source.items[0].model_copy(update={"native_id": str(2000 + index), "position": index})
        for index in range(8)
    )
    source = source.model_copy(update={"items": items, "item_count": len(items)})
    processed = {item.native_id: _processed(snapshot, "a" * 64) for item in items}
    config = MojiLexConfig(
        ai=AIConfig(model="test-model", ai_concurrency=concurrency),
        processing=ProcessingConfig(static_batch_size=1),
    )
    budget = RequestBudget(max_requests=len(items))
    active = peak = 0
    concurrent_started = asyncio.Event()
    saved = []

    async def describe(chunk, *_args, **kwargs):
        nonlocal active, peak
        assert kwargs["budget"] is budget
        await budget.reserve(CostEstimate(upper_bound_usd=0, note="synthetic"))
        active += 1
        peak = max(peak, active)
        if active == concurrency:
            concurrent_started.set()
        await concurrent_started.wait()
        await asyncio.sleep(0)
        active -= 1
        return {
            item.native_id: SimpleNamespace(
                request_trace=("trace",), description=None, generation=None
            )
            for item in chunk
        }

    async def checkpoint(chunk, generated):
        assert set(generated) == {item.native_id for item in chunk}
        saved.extend(generated)

    monkeypatch.setattr(runner, "_describe_batch", describe)
    monkeypatch.setattr(runner, "_recover_ai_request_traces", lambda *args, **kwargs: {})
    with CacheStore(tmp_path / "cache.sqlite3") as cache:
        await asyncio.wait_for(
            runner._descriptions_for_collection(
                snapshot,
                source,
                processed,
                config=config,
                cache=cache,
                budget=budget,
                ai_state=runner._AIState(),
                api_key=None,
                redescribe="all",
                overwrite_reviewed=False,
                temporary=None,
                on_chunk_completed=checkpoint,
            ),
            timeout=5,
        )
    assert peak == concurrency
    assert active == 0
    assert budget.requests_used == len(items)
    assert len(saved) == len(set(saved)) == len(items)
