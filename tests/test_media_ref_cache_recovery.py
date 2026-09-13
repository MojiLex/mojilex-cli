from __future__ import annotations

from decimal import Decimal

from mojilex_cli.ai import CostEstimate, DescriptionBatch, DescriptionResult, RequestBudget
from mojilex_cli.cache import CacheStore
from mojilex_cli.config import AIConfig, MojiLexConfig
from mojilex_cli.media import TemporaryMediaRun
from mojilex_cli.pipeline import runner
from test_ai_recovery_checkpoint import _prepare_exact_request
from test_dataset_helpers import write_fixture
from test_media_ref_cache_boundary import described_text
from test_pipeline_resume_cache import _item, _processed


async def test_same_revision_recovery_preserves_old_raw_row_and_resumes_fresh_answer(
    tmp_path, monkeypatch
):
    snapshot = write_fixture(tmp_path / "dataset")
    source = _item("binding-recovery", unique_id="binding-unique", file_id="file")
    processed = _processed(snapshot)
    context = runner._vision_context(source, processed)
    config = MojiLexConfig(ai=AIConfig(model="primary-model", model_routing="off"))
    scope = "synthetic-binding-recovery"
    old = DescriptionResult(
        batch=DescriptionBatch(items=(described_text({"role": "alternate"}),)),
        provider="gemini",
        model=config.ai.model,
        model_revision="same-synthetic-revision",
    )
    fresh = old.model_copy(
        update={"batch": DescriptionBatch(items=(described_text({"role": "light"}),))}
    )
    calls = []

    class Provider:
        name = "gemini"

        def estimate(self, request):
            return CostEstimate(upper_bound_usd=Decimal("0.01"), note="synthetic")

        async def describe(self, request):
            calls.append(request)
            assert request == prepared.request
            return fresh

    async def provider(*args, **kwargs):
        return Provider()

    async def no_provider(*args, **kwargs):
        raise AssertionError("the recovered exact answer must resume without AI")

    monkeypatch.setattr(runner, "_prepare_ai_request", _prepare_exact_request)
    monkeypatch.setattr(runner, "_provider_for_model", provider)
    budget = RequestBudget(max_requests=100, requests_used=63, cost_reserved=Decimal("0.63"))
    with (
        CacheStore(tmp_path / "cache.sqlite3") as cache,
        TemporaryMediaRun(root=tmp_path) as temporary,
    ):
        prepared = await _prepare_exact_request(
            (source,), {source.native_id: processed}, model=config.ai.model, temporary=temporary
        )

        def key(revision):
            return runner._cache_key(
                source,
                processed,
                context,
                config,
                model=config.ai.model,
                model_revision=revision,
                taxonomy_version="1.0.0",
                request_identity=prepared.identity,
                item_label="E001",
            )

        old_key = key(old.model_revision)
        cache.put_ai(
            old_key,
            old,
            generated_at="2026-09-13T00:00:00Z",
            aliases=runner._cache_aliases(scope, key(None)),
        )
        original_row = cache.get_ai_entry(old_key)
        assert original_row is not None
        original_payload = cache._connection.execute(
            "SELECT payload_json FROM ai_cache WHERE cache_key = ?", (old_key,)
        ).fetchone()["payload_json"]
        traces = []

        async def load(request_budget, **kwargs):
            return await runner._load_or_describe_single(
                source,
                processed,
                model=config.ai.model,
                config=config,
                cache=cache,
                budget=request_budget,
                ai_state=runner._AIState(),
                api_key=None,
                context=context,
                temporary=temporary,
                taxonomy_version="1.0.0",
                cache_alias_scope=scope,
                **kwargs,
            )

        recovered = await load(budget, resume_cache_key=old_key, trace_out=traces)
        assert recovered.result == fresh
        assert len(calls) == 1
        assert budget.max_requests == 100 and budget.requests_used == 64
        assert budget.cost_reserved == Decimal("0.64")
        assert len(traces) == 1
        trace = traces[0]
        assert trace.request_identity == prepared.identity
        assert trace.model_revision == old.model_revision
        assert trace.cache_key == old_key
        assert cache.get_ai(trace.cache_key) == fresh
        archived = cache._connection.execute(
            "SELECT cache_key, payload_json FROM ai_cache WHERE cache_key LIKE ?",
            ("rejected-ai-v1:%",),
        ).fetchall()
        assert len(archived) == 1
        assert archived[0]["payload_json"] == original_payload
        archive_key = archived[0]["cache_key"]
        archived_entry = cache.get_ai_entry(archive_key)
        assert archived_entry is not None and archived_entry[1] == original_row[1]
        runner._validate_actual_result(
            recovered.result, "gemini", config.ai.model, contexts={"E001": context}
        )

        monkeypatch.setattr(runner, "_provider_for_model", no_provider)
        for resume_args in (
            {"resume_cache_key": trace.cache_key, "request_identity": trace.request_identity},
            {},  # Reconstruct the exact request and follow the durable run alias.
        ):
            resumed_traces = []
            exhausted = RequestBudget(max_requests=0)
            resumed = await load(exhausted, trace_out=resumed_traces, **resume_args)
            assert resumed == recovered
            assert resumed_traces == traces
            assert exhausted.requests_used == 0
            assert cache.get_ai_entry(archive_key) == archived_entry
            assert cache.get_ai(trace.cache_key) == fresh
        assert len(calls) == 1
