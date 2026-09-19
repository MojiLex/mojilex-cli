from __future__ import annotations

from pathlib import Path

import pytest

from mojilex_cli.cache import CacheStore
from mojilex_cli.config import AIConfig, MojiLexConfig
from mojilex_cli.media import ProcessedMedia, TemporaryMediaRun
from mojilex_cli.pipeline import runner
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import (
    _PAYLOAD,
    _analysis,
    _collection,
    _CountingAdapter,
    _CountingProcessor,
    _item,
    _processed,
    _seed_resume,
)


class _Retained:
    def __init__(self, *, tile: bool, frames: bool) -> None:
        self.tile = tile
        self.frames = frames
        self.calls: list[str] = []

    def get_composition_tile(self, key: str, expected: ProcessedMedia):
        self.calls.append("tile")
        if not self.tile:
            return None
        return expected.model_copy(
            update={"composition_tile_path": Path("tile.png"), "composition_tile_sha256": "a" * 64}
        )

    def get(self, key: str, expected: ProcessedMedia):
        self.calls.append("frames")
        if not self.frames:
            return None
        return expected.model_copy(update={"frame_paths": (Path("retained-frame.png"),)})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "with_ai", "tile", "frames", "calls", "downloads", "decodes"),
    [
        ("static", True, True, False, ["tile"], 0, 0),
        ("repainting", True, False, False, [], 0, 0),
        ("animation", True, False, False, [], 0, 0),
        ("video", True, False, False, [], 0, 0),
        ("static", True, False, True, ["tile"], 1, 1),
        ("static", True, False, False, ["tile"], 1, 1),
        ("static", False, True, True, ["frames"], 0, 0),
        ("static", False, True, False, ["frames"], 1, 1),
    ],
)
async def test_exact_semantic_cache_restores_only_pixels_still_needed(
    tmp_path, monkeypatch, kind, with_ai, tile, frames, calls, downloads, decodes
):
    snapshot = write_fixture(tmp_path / "dataset")
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    item = _item("new-native-1", unique_id="saved-unique", file_id="first")
    processed = _processed(snapshot)
    if kind == "repainting":
        item = item.model_copy(update={"needs_repainting": True})
        processed = processed.model_copy(update={"dark_frame_paths": (Path("dark.png"),)})
    elif kind in {"animation", "video"}:
        fmt = "tgs" if kind == "animation" else "webm"
        item = item.model_copy(
            update={"media_format": fmt, "animated": kind == "animation", "video": kind == "video"}
        )
        processed = processed.model_copy(
            update={
                "metadata": processed.metadata.model_copy(
                    update={
                        "kind": kind,
                        "format": fmt,
                        "mime_type": "application/x-tgsticker"
                        if kind == "animation"
                        else "video/webm",
                        "animated": True,
                        "duration_ms": 1000,
                    }
                ),
                "frame_paths": tuple(Path(f"frame-{i}.png") for i in range(8)),
            }
        )
    source = _collection((item,))
    retained = _Retained(tile=tile, frames=frames)
    monkeypatch.setattr(runner, "get_retained_store", lambda *args, **kwargs: retained)
    monkeypatch.setattr(
        runner,
        "_decoder_backend_candidates",
        lambda *args: (processed.analysis.decoder_backend_fingerprint,),
    )
    adapter = _CountingAdapter({"first": _PAYLOAD})
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    try:
        elements = _seed_resume(cache, source, processed, config, with_ai=with_ai)
        outcomes = {}
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            _, result = await runner._prepare_collection_media(
                snapshot,
                adapter,
                source,
                processor,
                concurrency=1,
                cache=cache,
                resume_elements=elements,
                config=config,
                taxonomy_version="1.0.0",
                cache_alias_scope="mlxrun_" + "a" * 32,
                verified_semantic_outcomes=outcomes,
            )
        assert retained.calls == calls
        assert len(adapter.media_calls) == downloads
        assert processor.decode_calls == decodes
        assert bool(outcomes) is with_ai
        restored = result[item.native_id]
        if with_ai and (tile or kind != "static"):
            assert restored.frame_paths == restored.dark_frame_paths == ()
        if not with_ai:
            assert restored.frame_paths, "A new AI request must retain real rendered frames"
        if with_ai and kind == "static" and not tile:
            assert restored.frame_paths, "Missing puzzle pixels must be regenerated"
            assert processor.analysis_calls == 0, "Exact analysis can still be reused"
    finally:
        cache.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("descriptor_changed", [False, True])
async def test_public_semantics_reuse_requires_exact_saved_descriptor(
    tmp_path, monkeypatch, descriptor_changed
):
    from mojilex_cli.ai import RequestBudget
    from test_dataset_helpers import NATIVE_EMOJI_ID
    from test_pipeline_resume_cache import _replace_existing_media

    snapshot = write_fixture(tmp_path / "dataset")
    _replace_existing_media(snapshot, _PAYLOAD)
    existing = next(iter(snapshot.emojis.values()))
    telegram = existing.extensions["telegram"]
    item = _item(
        NATIVE_EMOJI_ID,
        unique_id=str(telegram["file_unique_id"]),
        file_id="reported",
        fallback_emoji=str(telegram["fallback_emoji"]),
    )
    source = _collection((item,), native_id="SuspiciousCats")
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    processed = _processed(snapshot)
    retained = _Retained(tile=True, frames=False)
    monkeypatch.setattr(runner, "get_retained_store", lambda *args, **kwargs: retained)
    adapter = _CountingAdapter({"reported": _PAYLOAD})
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    try:
        elements = _seed_resume(cache, source, processed, config, with_ai=False)
        if descriptor_changed:
            item = item.model_copy(update={"file_unique_id": "changed-descriptor"})
            source = _collection((item,), native_id="SuspiciousCats")
        with TemporaryMediaRun(root=tmp_path) as temporary:
            processor = _CountingProcessor(temporary, _analysis(snapshot))
            prepared_source, prepared = await runner._prepare_collection_media(
                snapshot,
                adapter,
                source,
                processor,
                concurrency=1,
                cache=cache,
                resume_elements=elements,
                config=config,
                taxonomy_version="1.0.0",
                cache_alias_scope="mlxrun_" + "b" * 32,
            )
            budget = RequestBudget(max_requests=0)
            descriptions, _ = await runner._descriptions_for_collection(
                snapshot,
                prepared_source,
                prepared,
                config=config,
                cache=cache,
                budget=budget,
                ai_state=runner._AIState(),
                api_key=None,
                redescribe="changed",
                overwrite_reviewed=False,
                temporary=temporary,
            )
        assert budget.requests_used == 0
        assert descriptions[item.native_id].descriptions.en.text == existing.descriptions["en"].text
        assert retained.calls == ([] if descriptor_changed else ["tile"])
        assert adapter.media_calls == (["reported"] if descriptor_changed else [])
        assert processor.decode_calls == int(descriptor_changed)
        if not descriptor_changed:
            assert prepared[item.native_id].frame_paths == ()
    finally:
        cache.close()
