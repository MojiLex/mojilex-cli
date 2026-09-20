# ruff: noqa: F811
import asyncio

from mojilex_cli.pipeline import runner
from test_pack_describe_pipeline import pipeline, saved_add  # noqa: F401
from test_pipeline_resume_cache import _collection, _item, _processed


async def test_final_pack_pass_reuses_early_results_without_another_request(tmp_path, monkeypatch):
    from mojilex_cli.ai.base import RequestBudget
    from mojilex_cli.cache import CacheStore
    from mojilex_cli.config import MojiLexConfig
    from mojilex_cli.media import TemporaryMediaRun
    from test_dataset_helpers import write_fixture
    from test_pipeline_transform import _description, _generation, _source

    snapshot = write_fixture(tmp_path / "dataset")
    source = _source(snapshot)
    native_id = source.items[0].native_id
    outcome = runner._SemanticOutcome(_description(snapshot), _generation())

    async def forbidden(*args, **kwargs):
        raise AssertionError("paid request repeated for an early completed item")

    monkeypatch.setattr(runner, "_describe_batch", forbidden)
    budget = RequestBudget(max_requests=0)
    state = runner._AIState()
    with CacheStore(tmp_path / "cache.sqlite3") as cache, TemporaryMediaRun() as temporary:
        descriptions, _ = await runner._descriptions_for_collection(
            snapshot,
            source,
            {native_id: _processed(snapshot)},
            config=MojiLexConfig(),
            cache=cache,
            budget=budget,
            ai_state=state,
            api_key=None,
            redescribe="all",
            overwrite_reviewed=True,
            temporary=temporary,
            verified_resume_outcomes={native_id: outcome},
            already_counted_cache_ids=frozenset({native_id}),
        )
    assert descriptions[native_id] == outcome.description
    assert budget.requests_used == 0
    assert state.cache_hits == 0


async def test_ai_receives_ready_media_before_last_media_of_same_pack(pipeline, monkeypatch):
    state = pipeline
    state.sources = (
        _collection(
            (
                _item("first", unique_id="first", file_id="first"),
                _item("last", unique_id="last", file_id="last"),
            ),
            native_id="PackAlpha",
        ),
        state.sources[1],
    )
    state.config = state.config.model_copy(
        update={
            "processing": state.config.processing.model_copy(
                update={
                    "file_analysis_mode": "fast",
                    "static_batch_size": 1,
                }
            )
        }
    )
    early_ai = asyncio.Event()
    last_ready = False

    async def media(snapshot, adapter, source, processor, **kwargs):
        nonlocal last_ready
        if source.native_id != "PackAlpha":
            return await state.media(snapshot, adapter, source, processor, **kwargs)
        values = {}
        for item in source.items:
            if item.native_id == "last":
                await asyncio.wait_for(early_ai.wait(), 5)
                last_ready = True
            values[item.native_id] = _processed(snapshot)
            await kwargs["on_item_completed"](item, values[item.native_id])
        return source, values

    async def describe(snapshot, source, processed, **kwargs):
        if source.native_id == "PackAlpha" and len(source.items) == 1:
            if source.items[0].native_id == "first":
                assert not last_ready
                early_ai.set()
        return await state.describe(snapshot, source, processed, **kwargs)

    monkeypatch.setattr(runner, "_prepare_collection_media", media)
    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)
    monkeypatch.setattr(runner, "_needs_generated_description", lambda *a, **k: True)
    result = await asyncio.wait_for(state.run(), 15)
    assert early_ai.is_set()
    assert last_ready
    assert not result.errors
