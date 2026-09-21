"""Read-only browsing must not revive rejected raw AI motion claims."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mojilex_cli.ai import DescriptionResult
from mojilex_cli.cache import CacheStore
from mojilex_cli.commands import interactive, packs
from mojilex_cli.config import MojiLexConfig
from mojilex_cli.i18n import use_ui_language
from mojilex_cli.runs import ElementCheckpoint
from test_cache_store import _result
from test_dataset_helpers import NATIVE_EMOJI_ID, write_fixture
from test_pack_commands import _save, _state


@pytest.mark.parametrize("staging_state", ["absent", "missing", "corrupt", "partial"])
@pytest.mark.parametrize("motion_status", ["described", "not_applicable", "undetermined"])
def test_cache_only_motion_is_explicitly_uncertain_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    staging_state: str,
    motion_status: str,
) -> None:
    config = MojiLexConfig(cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs")
    monkeypatch.setattr(packs, "load_config", lambda: config)
    payload = _result().model_dump(mode="json")
    value = payload["batch"]["items"][0]
    for description in value["descriptions"].values():
        description["motion_status"] = motion_status
        if motion_status == "described":
            description["motion"] = "Unsupported cached movement."
    if motion_status == "undetermined":
        value["facets"]["uncertainties"] = ["motion"]
    assert config.cache_dir is not None
    with CacheStore(config.cache_dir / "cache-v1.sqlite3") as cache:
        cache.put_ai("a" * 64, DescriptionResult.model_validate(payload))
    extra = {}
    if staging_state != "absent":
        staging = tmp_path / "staging"
        extra["staging_repository"] = str(staging)
        if staging_state == "corrupt":
            staging.mkdir()
            (staging / "manifest.json").write_text("{broken", encoding="utf-8")
        elif staging_state == "partial":
            # A readable dataset for another pack is not an authoritative result.
            write_fixture(staging)
    checkpoint = _save(
        config,
        elements={"123": ElementCheckpoint(stage="ai_cached", ai_cache_key="a" * 64)},
        extra=extra,
    )
    before = _state(tmp_path)
    result = packs.show_pack_command(checkpoint.run_id)
    shown = result.result["items"][0]
    assert shown["source"] == "ai_cache"
    assert any("showing AI answers from the cache" in warning for warning in result.warnings)
    for language, description in shown["descriptions"].items():
        original = value["descriptions"][language]
        assert description["text"] == original["text"]
        assert description["usage"] == original["usage"]
        assert description["motion_status"] == (
            "undetermined" if motion_status == "described" else motion_status
        )
        assert not description.get("motion")
    assert shown["semantic_tags"] == value["semantic_tags"]
    assert shown["content"] == value["content"]
    assert shown["facets"]["uncertainties"] == (
        [] if motion_status == "not_applicable" else ["motion"]
    )
    assert _state(tmp_path) == before


def test_canonical_described_motion_is_preserved_without_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "staging"
    write_fixture(staging)
    path = next((staging / "data" / "telegram" / "emojis").rglob("*.jsonl"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["media"][0].update(
        kind="animation",
        format="tgs",
        mime_type="application/x-tgsticker",
        animated=True,
        duration_ms=1000,
    )
    perceptual = record["fingerprints"]["items"][0]["perceptual"]
    perceptual["sample_count"] = 16
    for key in ("layout_phash64", "content_phash64", "alpha_phash64", "edge_phash64"):
        perceptual[key] = "A" * 171
    for description in record["descriptions"].values():
        description.update(motion_status="described", motion="A visible dot moves left.")
    path.write_bytes((json.dumps(record) + "\n").encode())
    config = MojiLexConfig(cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs")
    monkeypatch.setattr(packs, "load_config", lambda: config)
    checkpoint = _save(
        config,
        names=("SuspiciousCats",),
        elements={NATIVE_EMOJI_ID: ElementCheckpoint(stage="mapped")},
        extra={"staging_repository": str(staging)},
    )
    before = _state(tmp_path)
    result = packs.show_pack_command(checkpoint.run_id)
    assert not result.warnings
    item = result.result["items"][0]
    assert item["source"] == "staging"
    assert item["descriptions"]["en"]["motion_status"] == "described"
    assert item["descriptions"]["en"]["motion"] == "A visible dot moves left."
    assert _state(tmp_path) == before


@pytest.mark.parametrize("structured", [False, True])
def test_interactive_view_displays_fallback_reason(
    monkeypatch: pytest.MonkeyPatch, structured: bool
) -> None:
    from test_interactive_ui import view

    result = view()
    warning = "Cached motion claims are shown as undetermined."
    result.warnings.append(
        {"message": warning, "details": {"private": "must-not-display"}} if structured else warning
    )
    monkeypatch.setattr(interactive, "show_pack_command", lambda selector: result)
    shown = []
    monkeypatch.setattr(
        interactive, "select", lambda *args, **kwargs: shown.append(kwargs["detail"])
    )
    assert interactive.browse_descriptions("NewsEmoji") is result
    assert warning in shown[0]
    assert "must-not-display" not in shown[0]


def test_cache_fallback_warning_is_localized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = MojiLexConfig(cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs")
    monkeypatch.setattr(packs, "load_config", lambda: config)
    assert config.cache_dir is not None
    with CacheStore(config.cache_dir / "cache-v1.sqlite3") as cache:
        cache.put_ai("a" * 64, _result())
    checkpoint = _save(
        config,
        elements={"123": ElementCheckpoint(stage="ai_cached", ai_cache_key="a" * 64)},
        extra={"staging_repository": str(tmp_path / "missing")},
    )
    with use_ui_language("ru"):
        result = packs.show_pack_command(checkpoint.run_id)
    assert "Сохранённые результаты недоступны." in result.warnings
    assert any("Описанное движение не подтверждено" in warning for warning in result.warnings)
