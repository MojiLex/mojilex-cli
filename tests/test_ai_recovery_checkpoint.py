from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path

import pytest

from mojilex_cli.ai import (
    AIOutputError,
    BudgetExceededError,
    CostEstimate,
    DescriptionBatch,
    DescriptionRequest,
    DescriptionResult,
    RequestBudget,
    VisionImage,
    describe_with_recovery,
)
from mojilex_cli.cache import CacheStore
from mojilex_cli.config import AIConfig, MojiLexConfig, ProcessingConfig
from mojilex_cli.media import ProcessedMedia, TemporaryMediaRun
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import new_checkpoint
from mojilex_cli.sources import SourceEmoji
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import _collection, _description, _item, _processed

_PREFIX = b"\x89PNG\r\n\x1a\nexact-recovery:"


async def _prepare_exact_request(
    items: Sequence[SourceEmoji],
    processed: Mapping[str, ProcessedMedia],
    *,
    model: str,
    temporary: TemporaryMediaRun,
) -> runner._PreparedAIRequest:
    del temporary
    labels = tuple(f"E{index:03d}" for index in range(1, len(items) + 1))
    images = tuple(
        VisionImage(data=_PREFIX + item.native_id.encode(), labels=(label,))
        for item, label in zip(items, labels, strict=True)
    )
    request = DescriptionRequest(
        model=model,
        images=images,
        expected_labels=labels,
        context={
            label: runner._vision_context(item, processed[item.native_id])
            for item, label in zip(items, labels, strict=True)
        },
    )
    identity = runner._AIRequestIdentity(
        plan_sha256=runner._ai_request_plan_sha256(items, processed, model=model),
        request_sha256=hashlib.sha256(
            model.encode() + b"\0" + b"\0".join(image.data for image in images)
        ).hexdigest(),
        shown_media_sha256=tuple(hashlib.sha256(image.data).hexdigest() for image in images),
        labels_by_native=tuple(
            (item.native_id, label) for item, label in zip(items, labels, strict=True)
        ),
    )
    return runner._PreparedAIRequest(request=request, identity=identity)


class _RecoveryProvider:
    name = "gemini"

    def __init__(self, events: list[tuple[str, tuple[str, ...]]], fail_id: str | None) -> None:
        self.events = events
        self.fail_id = fail_id
        self.calls: list[tuple[str, ...]] = []

    def estimate(self, request: DescriptionRequest) -> CostEstimate:
        return CostEstimate(upper_bound_usd=Decimal("0.01"), note="synthetic test price")

    async def describe(self, request: DescriptionRequest) -> DescriptionResult:
        # The injected renderer tags each exact image. Singleton fallback must
        # never accidentally send the failed multi-item contact sheet again.
        assert all(image.data.startswith(_PREFIX) for image in request.images)
        native_ids = tuple(image.data.removeprefix(_PREFIX).decode() for image in request.images)
        assert len(native_ids) == len(request.expected_labels)
        self.calls.append(native_ids)
        self.events.append(("request", native_ids))
        if len(native_ids) > 1 or native_ids[0] == self.fail_id:
            raise AIOutputError("synthetic schema_validation: $.items (missing)")
        assert request.expected_labels == ("E001",)
        return DescriptionResult(
            batch=DescriptionBatch(items=(_description(),)),
            provider=self.name,
            model=request.model,
            model_revision="synthetic-revision",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["success", "later_invalid", "budget", "single_invalid"])
async def test_exact_batch_recovery_checkpoints_items_before_later_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    original_files = snapshot.to_files()
    first = _item("recovery-first", unique_id="first-unique", file_id="first")
    second = _item("recovery-second", unique_id="second-unique", file_id="second")
    source = _collection((first,) if scenario == "single_invalid" else (first, second))
    processed = {item.native_id: _processed(snapshot) for item in source.items}
    config = MojiLexConfig(
        ai=AIConfig(model="primary-model", model_routing="off", ai_concurrency=1),
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
    for item in source.items:
        checkpoint = runner._checkpoint_media_item(checkpoint, item, processed[item.native_id])
    events: list[tuple[str, tuple[str, ...]]] = []
    fail_id = (
        first.native_id
        if scenario == "single_invalid"
        else second.native_id
        if scenario == "later_invalid"
        else None
    )
    provider = _RecoveryProvider(events, fail_id)
    budget = RequestBudget(max_requests=3 if scenario == "budget" else 10)
    progress_objects = []
    real_progress = runner.BatchProgress

    def capture_progress(*args, **kwargs):
        progress = real_progress(*args, **kwargs)
        progress_objects.append(progress)
        return progress

    async def fixed_provider(*args, **kwargs):
        return provider

    async def persist_items(items, outcomes):
        nonlocal checkpoint
        native_ids = tuple(item.native_id for item in items)
        events.append(("checkpoint", native_ids))
        assert set(outcomes) == set(native_ids)
        assert all(outcome.request_trace for outcome in outcomes.values())
        partial_source = source.model_copy(update={"item_count": len(items), "items": tuple(items)})
        checkpoint = runner._checkpoint_ai_keys(
            checkpoint,
            partial_source,
            {},
            config,
            {native_id: outcome.generation for native_id, outcome in outcomes.items()},
            taxonomy_version="1.0.0",
            request_traces={
                native_id: outcome.request_trace for native_id, outcome in outcomes.items()
            },
        )
        checkpoint = runner._checkpoint_stage(checkpoint, native_ids, "ai_cached")

    monkeypatch.setattr(runner, "_prepare_ai_request", _prepare_exact_request)
    monkeypatch.setattr(runner, "_provider_for_model", fixed_provider)
    monkeypatch.setattr(runner, "BatchProgress", capture_progress)
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    try:
        with TemporaryMediaRun(root=tmp_path) as temporary:

            async def run():
                return await runner._descriptions_for_collection(
                    snapshot,
                    source,
                    processed,
                    config=config,
                    cache=cache,
                    budget=budget,
                    ai_state=runner._AIState(),
                    api_key=None,
                    redescribe="changed",
                    overwrite_reviewed=False,
                    temporary=temporary,
                    cache_alias_scope=checkpoint.run_id,
                    on_chunk_completed=persist_items,
                )

            if scenario == "success":
                descriptions, _ = await run()
                assert set(descriptions) == {first.native_id, second.native_id}
            elif scenario == "budget":
                with pytest.raises(BudgetExceededError):
                    await run()
            else:
                with pytest.raises(AIOutputError, match="synthetic schema_validation"):
                    await run()
        cached_count = cache.info()["ai_entries"]
    finally:
        cache.close()

    pair = (first.native_id, second.native_id)
    if scenario == "single_invalid":
        assert provider.calls == [(first.native_id,)] * 2
        assert events == [("request", (first.native_id,))] * 2
        completed, failed, batches = 0, 1, 0
    else:
        expected_calls = [pair, pair, (first.native_id,)]
        if scenario != "budget":
            expected_calls += [(second.native_id,)] * (2 if scenario == "later_invalid" else 1)
        assert provider.calls == expected_calls
        # Completion is durable before the next item gets a paid attempt.
        assert events[3] == ("checkpoint", (first.native_id,))
        assert checkpoint.elements[first.native_id].ai_facets_complete
        assert checkpoint.elements[first.native_id].ai_requests
        assert checkpoint.elements[second.native_id].ai_facets_complete is (scenario == "success")
        completed, failed, batches = (2, 0, 1) if scenario == "success" else (1, 1, 0)
    assert cached_count == completed
    assert budget.requests_used == len(provider.calls)
    assert budget.requests_used <= budget.max_requests
    assert budget.cost_reserved == Decimal("0.01") * len(provider.calls)
    progress = progress_objects[0]
    assert (progress.completed, progress.failed, progress.completed_batches) == (
        completed,
        failed,
        batches,
    )
    assert not progress.active and not progress.active_counts
    assert progress.queue_stopped is (scenario != "success")
    assert snapshot.to_files() == original_files

    if scenario in {"later_invalid", "budget"}:
        # Recover the exact saved singleton via the durable cache envelope, even
        # when the caller does not supply checkpoint traces explicitly.
        previous_calls = len(provider.calls)
        provider.fail_id = None
        budget = RequestBudget(
            max_requests=10,
            requests_used=budget.requests_used,
            cost_reserved=budget.cost_reserved,
        )
        cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
        try:
            with TemporaryMediaRun(root=tmp_path) as temporary:
                descriptions, _ = await run()
            assert cache.info()["ai_entries"] == 2
        finally:
            cache.close()
        assert set(descriptions) == {first.native_id, second.native_id}
        assert provider.calls[previous_calls:] == [(second.native_id,)]
        assert budget.requests_used == previous_calls + 1
        assert checkpoint.elements[second.native_id].ai_facets_complete
        resumed_progress = progress_objects[-1]
        assert resumed_progress.completed == 2
        assert resumed_progress.failed == 0
        assert resumed_progress.completed_batches == 1
        assert snapshot.to_files() == original_files


@pytest.mark.asyncio
@pytest.mark.parametrize("item_count", [1, 2])
async def test_recovery_preserves_last_validation_error_without_extra_attempts(
    tmp_path: Path, item_count: int
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    items = tuple(
        _item(f"error-{i}", unique_id=f"unique-{i}", file_id=f"file-{i}") for i in range(item_count)
    )
    processed = {item.native_id: _processed(snapshot) for item in items}
    with TemporaryMediaRun(root=tmp_path) as temporary:
        prepared = await _prepare_exact_request(
            items, processed, model="primary-model", temporary=temporary
        )
    errors = [AIOutputError("first invalid response"), AIOutputError("last schema failure")]
    provider = _RecoveryProvider([], None)

    async def fail(request):
        raise errors.pop(0)

    provider.describe = fail
    last_error = errors[-1]
    budget = RequestBudget(max_requests=10)
    with pytest.raises(AIOutputError) as captured:
        await describe_with_recovery(provider, prepared.request, budget)
    assert captured.value is last_error
    assert budget.requests_used == 2
    assert not errors
