from __future__ import annotations

import asyncio
from collections import Counter
from decimal import Decimal
from types import SimpleNamespace

import pytest

from mojilex_cli.ai import AIOutputError, BudgetExceededError, RequestBudget
from mojilex_cli.ai.base import AIError, AITransientError, UnknownCostError
from mojilex_cli.cache import CacheStore
from mojilex_cli.config import AIConfig, MojiLexConfig, ProcessingConfig
from mojilex_cli.media import TemporaryMediaRun
from mojilex_cli.pipeline import runner
from test_ai_recovery_checkpoint import _PREFIX, _prepare_exact_request, _RecoveryProvider
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import _collection, _item, _processed


@pytest.fixture
def queue_run(tmp_path, monkeypatch):
    snapshot = write_fixture(tmp_path / "dataset")
    original_files = snapshot.to_files()
    items = tuple(
        _item(f"deferred-{i}", unique_id=f"unique-{i}", file_id=f"file-{i}") for i in range(6)
    )
    source = _collection(items)
    state = SimpleNamespace(items=items, saved={}, checkpoints=[], counters=[], reports=[])
    real_progress = runner.BatchProgress

    def progress(*args, **kwargs):
        value = real_progress(*args, **kwargs)
        state.counters.append(value)
        return value

    monkeypatch.setattr(runner, "BatchProgress", progress)
    monkeypatch.setattr(runner, "report_progress", state.reports.append)
    monkeypatch.setattr(runner, "_prepare_ai_request", _prepare_exact_request)

    async def persist(chunk, outcomes):
        if getattr(state, "persist_failure", None) is not None:
            raise state.persist_failure
        assert set(outcomes) == {item.native_id for item in chunk}
        assert not (state.saved.keys() & outcomes.keys()), "completed items must be saved once"
        assert all(outcome.request_trace for outcome in outcomes.values())
        state.checkpoints.extend(outcomes)
        state.saved.update(outcomes)

    async def run(provider, budget, *, concurrency=1, batch_size=3):
        async def fixed_provider(*args, **kwargs):
            return provider

        monkeypatch.setattr(runner, "_provider_for_model", fixed_provider)
        config = MojiLexConfig(
            ai=AIConfig(model="primary-model", ai_concurrency=concurrency),
            processing=ProcessingConfig(static_batch_size=batch_size),
        )
        with (
            CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root) as cache,
            TemporaryMediaRun(root=tmp_path) as temporary,
        ):
            try:
                return await runner._descriptions_for_collection(
                    snapshot,
                    source,
                    {item.native_id: _processed(snapshot) for item in items},
                    config=config,
                    cache=cache,
                    budget=budget,
                    ai_state=runner._AIState(),
                    api_key=None,
                    redescribe="changed",
                    overwrite_reviewed=False,
                    temporary=temporary,
                    cache_alias_scope="deferred-queue-test",
                    on_chunk_completed=persist,
                )
            finally:
                assert snapshot.to_files() == original_files
                assert budget.requests_used <= budget.max_requests
                assert budget.requests_used == len(provider.calls)
                assert budget.cost_reserved == Decimal("0.01") * len(provider.calls)

    state.run = run
    return state


class RecoveringProvider(_RecoveryProvider):
    def __init__(self, bad_id, error_type=AIOutputError):
        super().__init__([], None)
        self.bad_id = bad_id
        self.error_type = error_type
        self.bad_attempts = 0

    async def describe(self, request):
        native_ids = tuple(image.data.removeprefix(_PREFIX).decode() for image in request.images)
        if native_ids == (self.bad_id,):
            self.bad_attempts += 1
            if self.bad_attempts <= (2 if self.error_type is AIOutputError else 3):
                self.calls.append(native_ids)
                self.events.append(("request", native_ids))
                raise self.error_type("temporary synthetic failure")
        return await super().describe(request)


@pytest.mark.parametrize("bad_index", [0, 1])
async def test_invalid_first_or_middle_item_does_not_discard_other_work(queue_run, bad_index):
    bad = queue_run.items[bad_index].native_id
    provider = RecoveringProvider(bad)
    budget = RequestBudget(max_requests=30)
    result, _ = await queue_run.run(provider, budget)
    expected = {item.native_id for item in queue_run.items}
    assert set(result) == set(queue_run.saved) == expected
    singleton_calls = Counter(call[0] for call in provider.calls if len(call) == 1)
    assert singleton_calls[bad] == 3
    assert all(singleton_calls[native_id] == 1 for native_id in expected - {bad})
    # The later batch is processed before a new round retries the bad singleton.
    assert provider.calls.index(tuple(item.native_id for item in queue_run.items[3:])) < max(
        index for index, call in enumerate(provider.calls) if call == (bad,)
    )
    progress = queue_run.counters[-1]
    assert progress.completed == 6
    assert progress.failed == 0
    assert not progress.active


async def test_permanent_invalid_item_uses_shared_budget_and_preserves_valid_siblings(queue_run):
    bad = queue_run.items[0].native_id
    provider = _RecoveryProvider([], bad)
    budget = RequestBudget(max_requests=18)
    with pytest.raises(BudgetExceededError):
        await queue_run.run(provider, budget)
    expected_saved = {item.native_id for item in queue_run.items} - {bad}
    assert set(queue_run.saved) == expected_saved
    assert budget.requests_used == 18
    singleton_calls = Counter(call[0] for call in provider.calls if len(call) == 1)
    assert all(singleton_calls[native_id] == 1 for native_id in expected_saved)
    assert queue_run.counters[-1].completed == 5


async def test_transient_provider_failure_is_deferred_and_successes_are_not_repeated(
    queue_run, monkeypatch
):
    original_sleep = asyncio.sleep

    async def no_wait(_):
        await original_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", no_wait)
    bad = queue_run.items[0].native_id
    provider = RecoveringProvider(bad, AITransientError)
    result, _ = await queue_run.run(provider, RequestBudget(max_requests=20), batch_size=1)
    assert set(result) == {item.native_id for item in queue_run.items}
    assert provider.calls == [(bad,)] * 3 + [(item.native_id,) for item in queue_run.items[1:]] + [
        (bad,)
    ]
    assert queue_run.counters[-1].completed == 6
    assert queue_run.counters[-1].failed == 0


async def test_cancellation_does_not_start_another_round(queue_run):
    class CancelledProvider(_RecoveryProvider):
        async def describe(self, request):
            ids = tuple(image.data.removeprefix(_PREFIX).decode() for image in request.images)
            self.calls.append(ids)
            raise asyncio.CancelledError

    provider = CancelledProvider([], None)
    with pytest.raises(asyncio.CancelledError):
        await queue_run.run(provider, RequestBudget(max_requests=20), batch_size=1)
    assert provider.calls == [(queue_run.items[0].native_id,)]
    assert not queue_run.saved


@pytest.mark.parametrize("failure_type", [AIError, UnknownCostError, BudgetExceededError])
async def test_fatal_failure_stops_queued_but_saves_an_inflight_success(queue_run, failure_type):
    second_started = asyncio.Event()
    first_failed = asyncio.Event()
    first_id = queue_run.items[0].native_id
    second_id = queue_run.items[1].native_id

    class FatalProvider(_RecoveryProvider):
        async def describe(self, request):
            ids = tuple(image.data.removeprefix(_PREFIX).decode() for image in request.images)
            if ids == (first_id,):
                self.calls.append(ids)
                await second_started.wait()
                first_failed.set()
                raise failure_type("synthetic fatal failure")
            if ids == (second_id,):
                second_started.set()
                await first_failed.wait()
                await asyncio.sleep(0)
            return await super().describe(request)

    provider = FatalProvider([], None)
    with pytest.raises(failure_type):
        await queue_run.run(provider, RequestBudget(max_requests=20), concurrency=2, batch_size=1)
    assert set(provider.calls) == {(first_id,), (second_id,)}
    assert set(queue_run.saved) == {second_id}
    assert queue_run.counters[-1].queue_stopped


async def test_checkpoint_error_is_fatal_even_if_its_type_is_ai_output_error(queue_run):
    failure = AIOutputError("synthetic checkpoint validation failure")
    queue_run.persist_failure = failure
    provider = _RecoveryProvider([], None)
    with pytest.raises(AIOutputError) as caught:
        await queue_run.run(provider, RequestBudget(max_requests=20), batch_size=1)
    assert caught.value is failure
    assert provider.calls == [(queue_run.items[0].native_id,)]
    assert not queue_run.saved


async def test_local_failure_without_request_or_progress_does_not_repeat_forever(
    queue_run, monkeypatch
):
    preparations = []

    async def invalid_local_request(items, *_args, **_kwargs):
        preparations.append(tuple(item.native_id for item in items))
        raise AIOutputError("synthetic local request construction failure")

    monkeypatch.setattr(runner, "_prepare_ai_request", invalid_local_request)
    provider = _RecoveryProvider([], None)
    budget = RequestBudget(max_requests=20)
    with pytest.raises(AIOutputError, match="synthetic local request construction"):
        await queue_run.run(provider, budget, batch_size=1)
    assert preparations == [(item.native_id,) for item in queue_run.items]
    assert budget.requests_used == 0
    assert not provider.calls
    assert not queue_run.saved
