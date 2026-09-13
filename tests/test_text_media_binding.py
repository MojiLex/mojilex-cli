from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import pytest
from PIL import Image

from mojilex_cli.ai import (
    AIOutputError,
    DescriptionBatch,
    DescriptionItem,
    DescriptionResult,
    VisionContext,
)
from mojilex_cli.ai.media_refs import bind_primary_media_references
from mojilex_cli.analysis import analyze_decoded_media
from mojilex_cli.cache import CacheStore
from mojilex_cli.domain import Emoji
from mojilex_cli.pipeline import runner, transform
from test_ai_gemini import _payload as _no_text_payload
from test_ai_semantic_facets import _payload as _text_payload
from test_dataset_helpers import make_snapshot
from test_pipeline_transform import _generation, _processed, _source


def _description(refs: list[dict[str, str]]) -> DescriptionItem:
    payload = _text_payload()
    payload["items"][0]["facets"]["text_content"]["items"][0]["media_refs"] = refs
    return DescriptionBatch.model_validate(payload).items[0]


@pytest.mark.parametrize(
    "refs,backgrounds",
    [
        ([{"role": "primary"}], ("light",)),
        ([{"role": "light"}], ("light",)),
        ([{"role": "dark"}], ("light", "dark")),
        ([{"role": "light"}, {"role": "dark"}], ("light", "dark")),
        ([{"role": "primary"}, {"role": "light"}, {"role": "dark"}], ("light", "dark")),
    ],
)
def test_render_view_binding_changes_only_refs_and_is_idempotent(refs, backgrounds) -> None:
    original = _description(refs)
    before = original.model_dump(mode="json", exclude_none=True)
    bound = bind_primary_media_references(original, background_variants=backgrounds)
    expected = deepcopy(before)
    expected["facets"]["text_content"]["items"][0]["media_refs"] = [{"role": "primary"}]
    assert bound.model_dump(mode="json", exclude_none=True) == expected
    assert original.model_dump(mode="json", exclude_none=True) == before
    assert bind_primary_media_references(bound, background_variants=backgrounds) == bound
    assert bound.facets.text_content.items[0].value == "404"


def test_absent_text_is_preserved_without_inventing_items_or_refs() -> None:
    original = DescriptionBatch.model_validate(_no_text_payload()).items[0]
    assert bind_primary_media_references(original, background_variants=("light",)) == original
    assert not original.facets.text_content.items


@pytest.mark.parametrize(
    "refs,backgrounds",
    [
        ([{"role": "dark"}], ("light",)),
        ([{"role": "alternate"}], ("light", "dark")),
        ([{"role": "primary", "variant_id": "synthetic-private-id"}], ("light", "dark")),
        ([{"role": "light", "variant_id": "light"}], ("light",)),
        ([{"role": "dark", "variant_id": "dark"}], ("light", "dark")),
        ([{"role": "primary"}, {"role": "alternate"}], ("light", "dark")),
    ],
)
def test_binding_rejects_unknown_variants_and_views_without_losing_text(refs, backgrounds) -> None:
    original = _description(refs)
    before = original.model_dump(mode="json")
    with pytest.raises(AIOutputError) as captured:
        bind_primary_media_references(original, background_variants=backgrounds)
    assert original.model_dump(mode="json") == before
    assert "synthetic-private-id" not in str(captured.value)


def test_binding_does_not_rewrite_the_immutable_raw_ai_cache(tmp_path: Path) -> None:
    raw = DescriptionResult(
        batch=DescriptionBatch(items=(_description([{"role": "light"}, {"role": "dark"}]),)),
        provider="gemini",
        model="synthetic-model",
        model_revision="synthetic-revision",
    )
    before = raw.model_dump(mode="json")
    cache = CacheStore(tmp_path / "cache.sqlite3")
    try:
        cache.put_ai("a" * 64, raw, generated_at="2026-09-13T00:00:00Z")
        cached = cache.get_ai("a" * 64)
        assert cached is not None
        bound = bind_primary_media_references(
            cached.batch.items[0], background_variants=("light", "dark")
        )
        assert [ref.role for ref in bound.facets.text_content.items[0].media_refs] == ["primary"]
        reread = cache.get_ai("a" * 64)
        assert reread is not None and reread.model_dump(mode="json") == before
        assert cached.model_dump(mode="json") == before
    finally:
        cache.close()


@pytest.mark.parametrize("case", ["shown_views", "unshown_dark", "invented_variant"])
def test_runner_validates_context_binding_without_rewriting_accepted_raw_result(case: str) -> None:
    backgrounds = ("light", "dark") if case == "shown_views" else ("light",)
    refs = (
        [{"role": "light"}, {"role": "dark"}]
        if case == "shown_views"
        else [{"role": "dark"}]
        if case == "unshown_dark"
        else [{"role": "primary", "variant_id": "synthetic-private-id"}]
    )
    result = DescriptionResult(
        batch=DescriptionBatch(items=(_description(refs),)), provider="gemini", model="test-model"
    )
    before = result.model_dump(mode="json")
    context = VisionContext(
        needs_repainting=case == "shown_views", frame_count=1, background_variants=backgrounds
    )
    if case == "shown_views":
        runner._validate_actual_result(result, "gemini", "test-model", contexts={"E001": context})
    else:
        with pytest.raises(AIOutputError):
            runner._validate_actual_result(
                result, "gemini", "test-model", contexts={"E001": context}
            )
    assert result.model_dump(mode="json") == before


def test_batch_binding_checks_each_items_own_render_context() -> None:
    first = _description([{"role": "dark"}])
    second = first.model_copy(update={"label": "E002"})
    result = DescriptionResult(
        batch=DescriptionBatch(items=(first, second)), provider="gemini", model="test-model"
    )
    contexts = {
        "E001": VisionContext(
            needs_repainting=True, frame_count=1, background_variants=("light", "dark")
        ),
        "E002": VisionContext(
            needs_repainting=False, frame_count=1, background_variants=("light",)
        ),
    }
    with pytest.raises(AIOutputError):
        runner._validate_actual_result(
            result, "gemini", "test-model", require_single=False, contexts=contexts
        )


@pytest.mark.parametrize("adaptive", [False, True])
def test_transform_binds_rendered_views_to_existing_media_and_keeps_provenance(
    tmp_path: Path, adaptive: bool
) -> None:
    snapshot = make_snapshot(tmp_path / "dataset")
    source = _source(snapshot).items[0].model_copy(update={"needs_repainting": adaptive})
    processed = _processed(snapshot, tmp_path)
    frame_path = processed.frame_paths[0]
    with Image.new(
        "RGBA", (processed.metadata.width, processed.metadata.height), (15, 30, 45, 255)
    ) as frame:
        frame.save(frame_path)
        analysis = analyze_decoded_media((frame,), needs_repainting=adaptive)
    dark_paths = ()
    if adaptive:
        dark_path = tmp_path / "synthetic-dark-frame.png"
        with Image.new(
            "RGBA", (processed.metadata.width, processed.metadata.height), (240, 240, 240, 255)
        ) as frame:
            frame.save(dark_path)
        dark_paths = (dark_path,)
    processed = processed.model_copy(
        update={
            "analysis": analysis,
            "dark_frame_paths": dark_paths,
            "has_dark_render": adaptive,
        }
    )
    bound_analysis = runner._bind_deterministic_analyses({source.native_id: processed})[
        source.native_id
    ]
    refs = [{"role": "light"}, {"role": "dark"}] if adaptive else [{"role": "light"}]
    description = _description(refs)
    raw_before = description.model_dump(mode="json")
    generation = _generation()
    provenance_before = asdict(generation)
    emoji = transform._emoji(
        "telegram",
        source,
        processed,
        description,
        bound_analysis,
        generation,
        manifest=snapshot.manifest,
        epoch=0,
        now="2026-09-13T00:00:00Z",
    )
    # Exercise the complete public domain model, not only adapter validation.
    validated = Emoji.model_validate(emoji.model_dump(mode="json"))
    assert [(media.role.value, media.variant_id) for media in validated.media] == [
        ("primary", None)
    ]
    assert [ref.key for ref in validated.facets.text_content.items[0].media_refs] == [
        ("primary", "")
    ]
    assert validated.facets.text_content.items[0].value == "404"
    assert validated.provenance.prompt_version == generation.prompt_version
    assert validated.provenance.prompt_sha256 == generation.prompt_sha256
    assert validated.provenance.request_parameters_sha256 == generation.request_parameters_sha256
    assert validated.provenance.model_revision == generation.model_revision
    assert description.model_dump(mode="json") == raw_before
    assert asdict(generation) == provenance_before
