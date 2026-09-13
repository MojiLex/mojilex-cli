from __future__ import annotations

import pytest

from mojilex_cli.ai import (
    CostEstimate,
    DescriptionBatch,
    DescriptionItem,
    DescriptionResult,
    RequestBudget,
)
from mojilex_cli.cache import CacheStore
from mojilex_cli.config import AIConfig, MojiLexConfig
from mojilex_cli.media import TemporaryMediaRun
from mojilex_cli.pipeline import runner
from test_ai_recovery_checkpoint import _prepare_exact_request
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import _collection, _description, _item, _processed, _seed_resume


def described_text(reference):
    value = _description().model_dump(mode="json")
    value["facets"]["content_types"] = sorted(set(value["facets"]["content_types"]) | {"text"})
    value["facets"]["text_content"] = {
        "status": "recognized",
        "dynamics": "stable",
        "items": [
            {
                "value": "OK",
                "kind": "word",
                "script": "Latn",
                "language": "en",
                "temporal_scope": "persistent",
                "media_refs": [reference],
            }
        ],
    }
    return DescriptionItem.model_validate(value)


@pytest.mark.parametrize(
    "reference",
    [
        {"role": "alternate"},
        {"role": "dark"},
        {"role": "primary", "variant_id": "invented"},
    ],
)
async def test_unbound_live_reply_never_becomes_a_completed_cache_row(
    tmp_path, monkeypatch, reference
):
    snapshot = write_fixture(tmp_path / "dataset")
    source = _collection((_item("binding-test", unique_id="binding-unique", file_id="file"),))
    processed = {source.items[0].native_id: _processed(snapshot)}
    config = MojiLexConfig(ai=AIConfig(model="primary-model"))
    saved = []
    calls = []

    class Provider:
        name = "gemini"

        def estimate(self, request):
            return CostEstimate(upper_bound_usd=0, note="synthetic")

        async def describe(self, request):
            calls.append(request)
            assert cache.info()["ai_entries"] == 0
            assert not saved
            item = described_text(reference if len(calls) == 1 else {"role": "light"})
            return DescriptionResult(
                batch=DescriptionBatch(items=(item,)), provider=self.name, model=request.model
            )

    async def provider(*args, **kwargs):
        return Provider()

    async def completed(items, outcomes):
        saved.extend(outcomes)

    monkeypatch.setattr(runner, "_provider_for_model", provider)
    monkeypatch.setattr(runner, "_prepare_ai_request", _prepare_exact_request)
    budget = RequestBudget(max_requests=2)
    with (
        CacheStore(tmp_path / "cache.sqlite3") as cache,
        TemporaryMediaRun(root=tmp_path) as temporary,
    ):
        descriptions, _ = await runner._descriptions_for_collection(
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
            temporary=temporary,
            on_chunk_completed=completed,
        )
        assert cache.info()["ai_entries"] == 1
        assert len(calls) == budget.requests_used == 2
        assert saved == [source.items[0].native_id]
        # Cache semantics stay in the original request view; transform alone
        # binds the reference to primary for the public dataset.
        assert descriptions[saved[0]].facets.text_content.items[0].media_refs[0].role == "light"


async def test_unbound_old_cache_is_a_miss_without_deleting_original_row(tmp_path, monkeypatch):
    import test_pipeline_resume_cache as fixtures

    snapshot = write_fixture(tmp_path / "dataset")
    source = _collection((_item("cached-binding", unique_id="cached-unique", file_id="file"),))
    config = MojiLexConfig(ai=AIConfig(model="primary-model"))
    processed = _processed(snapshot)
    invalid = described_text({"role": "alternate"})
    monkeypatch.setattr(fixtures, "_description", lambda: invalid)
    with (
        CacheStore(tmp_path / "cache.sqlite3") as cache,
        TemporaryMediaRun(root=tmp_path) as temporary,
    ):
        elements = _seed_resume(cache, source, processed, config)
        key = elements[source.items[0].native_id].ai_cache_key
        before = cache.get_ai_entry(key)
        _, outcomes, _ = await runner._resume_cached_processed_media(
            snapshot,
            source,
            runner.MediaProcessor(temporary),
            cache=cache,
            resume_elements=elements,
            config=config,
            taxonomy_version="1.0.0",
            cache_alias_scope=None,
            forbidden_native_ids=set(),
            redescribe="all",
            overwrite_reviewed=False,
        )
        assert outcomes == {}
        assert cache.get_ai_entry(key) == before
        assert cache.info()["ai_entries"] == 1
