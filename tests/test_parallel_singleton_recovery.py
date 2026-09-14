"""Singleton recovery overlaps requests and retains paid results on budget exhaustion."""

import asyncio
from decimal import Decimal

import pytest

from mojilex_cli.ai import BudgetExceededError, RequestBudget
from mojilex_cli.cache import CacheStore
from mojilex_cli.concurrency import batch_limits
from mojilex_cli.config import AIConfig, MojiLexConfig, ProcessingConfig
from mojilex_cli.media import TemporaryMediaRun
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import RunStore, new_checkpoint
from test_ai_recovery_checkpoint import _PREFIX, _prepare_exact_request, _RecoveryProvider
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import _collection, _item, _processed


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["success", "budget"])
async def test_parallel_recovery_saves_each_result_and_drains_paid_peers(
    tmp_path, monkeypatch, scenario
):
    snapshot = write_fixture(tmp_path / "dataset")
    items = tuple(
        _item(f"parallel-{i}", unique_id=f"unique-{i}", file_id=f"file-{i}") for i in range(3)
    )
    source = _collection(items)
    processed = {item.native_id: _processed(snapshot) for item in items}
    config = MojiLexConfig(
        ai=AIConfig(model="primary-model", model_routing="off", ai_concurrency=2),
        processing=ProcessingConfig(static_batch_size=8),
    )
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    for item in items:
        checkpoint = runner._checkpoint_media_item(checkpoint, item, processed[item.native_id])
    store = RunStore(tmp_path / "runs", repository_root=snapshot.root)
    store.save(checkpoint)
    both_started = asyncio.Event()
    release_second = asyncio.Event()
    first_saved = asyncio.Event()
    denied = asyncio.Event()
    active = 0
    maximum = 0
    reservations = []

    def reserve_durably(requests, cost):
        nonlocal checkpoint
        reservations.append((requests, cost))
        checkpoint = checkpoint.model_copy(
            update={"ai_requests_used": requests, "ai_cost_reserved_usd": cost}
        )
        store.save(checkpoint)

    class Budget(RequestBudget):
        async def reserve(self, estimate, **kwargs):
            try:
                await super().reserve(estimate, **kwargs)
            except BudgetExceededError:
                denied.set()
                raise

    budget = Budget(
        max_requests=4 if scenario == "budget" else 5, reservation_recorder=reserve_durably
    )

    class Provider(_RecoveryProvider):
        async def describe(self, request):
            nonlocal active, maximum
            assert store.load(checkpoint.run_id).ai_requests_used == budget.requests_used
            if len(request.expected_labels) > 1:
                return await super().describe(request)
            native_id = request.images[0].data.removeprefix(_PREFIX).decode()
            active += 1
            maximum = max(maximum, active)
            if active == 2:
                both_started.set()
            try:
                if native_id == items[0].native_id:
                    await both_started.wait()
                if native_id == items[1].native_id:
                    await release_second.wait()
                return await super().describe(request)
            finally:
                active -= 1

    provider = Provider([], None)

    async def fixed_provider(*args, **kwargs):
        return provider

    async def persist(items_to_save, outcomes):
        nonlocal checkpoint
        partial = source.model_copy(
            update={"item_count": len(items_to_save), "items": tuple(items_to_save)}
        )
        checkpoint = runner._checkpoint_ai_keys(
            checkpoint,
            partial,
            {},
            config,
            {key: value.generation for key, value in outcomes.items()},
            taxonomy_version="1.0.0",
            request_traces={key: value.request_trace for key, value in outcomes.items()},
        )
        checkpoint = runner._checkpoint_stage(checkpoint, tuple(outcomes), "ai_cached")
        store.save(checkpoint)
        if items[0].native_id in outcomes:
            first_saved.set()

    monkeypatch.setattr(runner, "_prepare_ai_request", _prepare_exact_request)
    monkeypatch.setattr(runner, "_provider_for_model", fixed_provider)
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    task = None
    try:
        with (
            TemporaryMediaRun(root=tmp_path) as temporary,
            batch_limits(downloads=1, renders=1, ai=2, max_temp_bytes=16 * 1024 * 1024),
        ):
            task = asyncio.create_task(
                runner._describe_batch(
                    items,
                    processed,
                    config=config,
                    cache=cache,
                    budget=budget,
                    ai_state=runner._AIState(),
                    api_key=None,
                    temporary=temporary,
                    taxonomy_version="1.0.0",
                    qualifications=runner.ModelQualificationRegistry.load(snapshot.root),
                    routing_registry=runner.RoutingReasonRegistry.load(snapshot.root),
                    cache_alias_scope=checkpoint.run_id,
                    on_item_completed=persist,
                )
            )
            try:
                await asyncio.wait_for(both_started.wait(), timeout=3)
                await asyncio.wait_for(first_saved.wait(), timeout=3)
                saved = store.load(checkpoint.run_id)
                assert saved.elements[items[0].native_id].ai_facets_complete
                assert not saved.elements[items[1].native_id].ai_facets_complete
                if scenario == "budget":
                    await asyncio.wait_for(denied.wait(), timeout=3)
                    assert not task.done(), (
                        "A paid sibling must finish and persist after budget exhaustion"
                    )
                release_second.set()
                if scenario == "budget":
                    with pytest.raises(BudgetExceededError):
                        await asyncio.wait_for(task, timeout=3)
                else:
                    result = await asyncio.wait_for(task, timeout=3)
                    assert set(result) == {item.native_id for item in items}
            finally:
                release_second.set()
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        saved = store.load(checkpoint.run_id)
        expected = 2 if scenario == "budget" else 3
        assert sum(item.ai_facets_complete for item in saved.elements.values()) == expected
        assert saved.elements[items[1].native_id].ai_facets_complete
        assert cache.info()["ai_entries"] == expected
        assert maximum == 2
        assert budget.requests_used == len(provider.calls) == len(reservations) == expected + 2
        assert saved.ai_requests_used == budget.requests_used
        assert (
            saved.ai_cost_reserved_usd
            == budget.cost_reserved
            == Decimal("0.01") * budget.requests_used
        )
    finally:
        cache.close()
