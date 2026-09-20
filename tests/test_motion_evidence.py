from __future__ import annotations

import pytest
from PIL import Image

from mojilex_cli.domain.models import LocalizedDescription
from mojilex_cli.media.motion import observed_frame_variation
from mojilex_cli.pipeline.transform import plan_collection_merge
from test_dataset_helpers import make_animated_snapshot
from test_pipeline_transform import _analysis, _description, _generation, _processed, _source


def _frames(root, role, *, changed=False):
    paths = []
    for index in range(8):
        image = Image.new("RGB", (256, 256), "white" if role == "light" else "black")
        if changed and index == 7:
            image.putpixel((100, 100), (1, 2, 3))
        path = root / f"{role}-{index}.png"
        image.save(path)
        image.close()
        paths.append(path)
    return tuple(paths)


@pytest.mark.parametrize("change", [None, "light", "dark"])
def test_exact_samples_compare_each_background_and_notice_one_pixel(tmp_path, change):
    snapshot = make_animated_snapshot(tmp_path)
    processed = _processed(snapshot, tmp_path).model_copy(
        update={
            "frame_paths": _frames(tmp_path, "light", changed=change == "light"),
            "dark_frame_paths": _frames(tmp_path, "dark", changed=change == "dark"),
            "rendered_frame_count": 8,
            "has_dark_render": True,
        }
    )
    assert observed_frame_variation(processed) is (change is not None)
    assert "observed_frame_variation" not in processed.model_dump(mode="json")
    assert "observed_frame_variation" not in processed.dataset_metadata()

    missing = processed.model_copy(update={"dark_frame_paths": ()})
    assert observed_frame_variation(missing) is None
    processed.frame_paths[0].unlink()
    assert observed_frame_variation(processed) is None


@pytest.mark.parametrize("observation", [False, True, None])
def test_frameless_observation_suppresses_only_unsupported_motion_and_keeps_raw_cache(
    tmp_path, observation
):
    snapshot = make_animated_snapshot(tmp_path)
    original = next(iter(snapshot.emojis.values()))
    for language, value in original.descriptions.items():
        original.descriptions[language] = LocalizedDescription.model_validate(
            {**value.as_dict(), "motion_status": "described", "motion": "Moves left."}
        )
    cached = _description(snapshot)
    raw = cached.model_dump(mode="json")
    source = _source(snapshot)
    native_id = source.items[0].native_id
    processed = _processed(snapshot, tmp_path).model_copy(
        update={
            "frame_paths": (),
            "dark_frame_paths": (),
            "rendered_frame_count": 8,
            "observed_frame_variation": observation,
        }
    )
    generation = _generation()
    results = []
    for _ in range(2):
        plan = plan_collection_merge(
            snapshot,
            source,
            {native_id: processed},
            {native_id: cached},
            {native_id: _analysis(snapshot)},
            {native_id: generation},
            timestamp="2026-09-11T18:00:00Z",
        )
        result = next(iter(plan.snapshot.emojis.values()))
        results.append(result.as_dict())
        for language, description in result.descriptions.items():
            assert description.text == original.descriptions[language].text
            assert description.usage == original.descriptions[language].usage
            assert description.motion_status.value == (
                "undetermined" if observation is False else "described"
            )
            assert description.motion == (None if observation is False else "Moves left.")
        assert ("motion" in result.facets.uncertainties) == (observation is False)
        if observation is False:
            assert result.provenance.prompt_sha256 == generation.prompt_sha256
            assert result.provenance.model == generation.model
        assert cached.model_dump(mode="json") == raw
    assert results[0] == results[1]
