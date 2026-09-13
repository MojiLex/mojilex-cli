from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mojilex_cli.ai import DescriptionResult
from mojilex_cli.cache import CacheStore
from mojilex_cli.commands import packs
from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.config import MojiLexConfig
from mojilex_cli.runs import (
    ElementCheckpoint,
    RunCheckpoint,
    RunStore,
    RunStoreError,
    new_checkpoint,
)
from test_cache_store import _result
from test_dataset_helpers import NATIVE_EMOJI_ID, write_fixture


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MojiLexConfig:
    value = MojiLexConfig(cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs")
    monkeypatch.setattr(packs, "load_config", lambda: value)
    return value


def _save(
    config: MojiLexConfig,
    *,
    run_digit: str = "a",
    minute: int = 1,
    names: tuple[str, ...] = ("NewsEmoji",),
    repository: str = "MojiLex/mojilex",
    elements: dict[str, ElementCheckpoint] | None = None,
    extra: dict[str, object] | None = None,
) -> RunCheckpoint:
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={
            "sources": [f"https://t.me/addemoji/{name}" for name in names],
            **(extra or {}),
        },
        cli_version="0.2.0",
        schema_version="1.0.0",
        target_repository=repository,
        base_revision="f" * 40,
        run_id="mlxrun_" + run_digit * 32,
    ).model_copy(
        update={
            "updated_at": datetime(2026, 1, 1, 12, minute, tzinfo=UTC),
            "status": "interrupted",
            "elements": elements or {},
        }
    )
    assert config.runs_dir is not None
    RunStore(config.runs_dir).save(checkpoint)
    return checkpoint


def _state(root: Path) -> dict[str, tuple[int, bytes]]:
    return {
        str(path.relative_to(root)): (path.stat().st_mtime_ns, path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


def test_latest_exact_case_insensitive_name_and_explicit_run_id(config: MojiLexConfig) -> None:
    old = _save(config)
    latest = _save(config, run_digit="b", minute=2)
    assert packs.resolve_pack_run("newsemoji").run_id == latest.run_id
    assert packs.resolve_pack_run("https://t.me/addemoji/NEWSEMOJI").run_id == latest.run_id
    assert packs.resolve_pack_run(old.run_id).run_id == old.run_id
    result = packs.list_packs_command()
    assert [item["run_id"] for item in result.result["packs"]] == [latest.run_id]
    assert result.result["saved_runs"] == 2
    assert result.result["packs"][0]["names"] == ["NewsEmoji"]
    assert "target_repository" not in json.dumps(result.result)


@pytest.mark.parametrize("other", ["target", "source_group"])
def test_same_name_different_group_or_repository_is_ambiguous(
    config: MojiLexConfig,
    other: str,
) -> None:
    first = _save(config)
    second = _save(
        config,
        run_digit="b",
        minute=2,
        names=("NewsEmoji", "OtherPack") if other == "source_group" else ("NewsEmoji",),
        repository="Other/Repository" if other == "target" else "MojiLex/mojilex",
    )
    with pytest.raises(CommandError) as error:
        packs.resolve_pack_run("NewsEmoji")
    assert error.value.error.code == "CONFIG_INVALID"
    assert packs.resolve_pack_run(first.run_id).run_id == first.run_id
    assert len(packs.list_packs_command().result["packs"]) == 2
    assert second.run_id in error.value.error.details["runs"]


def test_missing_name_has_fallback_code_and_list_does_not_create_storage(
    config: MojiLexConfig,
) -> None:
    assert packs.list_packs_command().result["packs"] == []
    assert config.runs_dir is not None and not config.runs_dir.exists()
    assert config.cache_dir is not None and not config.cache_dir.exists()
    with pytest.raises(CommandError) as error:
        packs.resolve_pack_run("NewsEmoji")
    assert error.value.error.code == "CONFIG_MISSING"
    with pytest.raises(RunStoreError):
        packs.resolve_pack_run("mlxrun_invalid")


def test_missing_cache_shows_partial_saved_progress_without_writes(
    config: MojiLexConfig,
    tmp_path: Path,
) -> None:
    checkpoint = _save(
        config,
        elements={
            "ready": ElementCheckpoint(stage="ai_cached", ai_cache_key="a" * 64),
            "pending": ElementCheckpoint(stage="media_verified"),
        },
    )
    before = _state(tmp_path)
    result = packs.show_pack_command("NewsEmoji")
    assert result.run_id == checkpoint.run_id
    assert result.result["items"] == []
    assert result.result["counts"] == {"ready": 0, "pending": 1, "missing": 1, "invalid": 0}
    assert result.warnings
    assert _state(tmp_path) == before


def test_review_reads_exact_keys_and_preserves_warning_and_full_bilingual_text(
    config: MojiLexConfig,
    tmp_path: Path,
) -> None:
    assert config.cache_dir is not None
    payload = _result().model_dump(mode="json")
    description = payload["batch"]["items"][0]
    description["content"]["warnings"] = ["flashing"]
    description["descriptions"]["ru"].update(
        {
            "text": "Saved complete Russian description.",
            "motion_status": "described",
            "motion": "The light flashes.",
            "usage": ["attention", "warning"],
        }
    )
    with CacheStore(config.cache_dir / "cache-v1.sqlite3") as cache:
        cache.put_ai("a" * 64, DescriptionResult.model_validate(payload), aliases=("b" * 64,))
        cache._connection.execute(
            "INSERT INTO ai_cache VALUES(?,?,?,?)",
            ("c" * 64, "invalid json", 1, 1),
        )
        cache._connection.commit()
    _save(
        config,
        elements={
            "ready": ElementCheckpoint(stage="ai_cached", ai_cache_key="a" * 64),
            "alias_only": ElementCheckpoint(stage="ai_cached", ai_cache_key="b" * 64),
            "invalid": ElementCheckpoint(stage="ai_cached", ai_cache_key="c" * 64),
        },
    )
    before = _state(tmp_path)
    result = packs.show_pack_command("NewsEmoji", review=True)
    assert result.result["review"] is True
    assert result.result["counts"] == {"ready": 1, "pending": 0, "missing": 1, "invalid": 1}
    item = result.result["items"][0]
    assert item["native_id"] == "ready"
    assert item["content"] == {"rating": "general", "warnings": ["flashing"]}
    assert item["review_status"] == "unreviewed"
    assert item["descriptions"]["ru"]["text"] == "Saved complete Russian description."
    assert item["descriptions"]["ru"]["motion"] == "The light flashes."
    assert item["descriptions"]["ru"]["usage"] == ["attention", "warning"]
    assert item["descriptions"]["en"]["text"] == "A synthetic smiling face."
    assert item["semantic_tags"] == ["face", "smile"]
    assert item["source"] == "ai_cache"
    assert _state(tmp_path) == before


def test_staging_pack_is_visible_without_cache(config: MojiLexConfig, tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    snapshot = write_fixture(staging)
    checkpoint = _save(
        config,
        names=("SuspiciousCats",),
        extra={"staging_repository": str(staging)},
        elements={NATIVE_EMOJI_ID: ElementCheckpoint(stage="ai_cached", ai_cache_key="a" * 64)},
    )
    before = _state(tmp_path)
    result = packs.show_pack_command(checkpoint.run_id, review=True)
    assert result.result["counts"] == {"ready": 1, "pending": 0, "missing": 0, "invalid": 0}
    item = result.result["items"][0]
    assert item["source"] == "staging"
    assert item["native_id"] == NATIVE_EMOJI_ID
    emoji = next(iter(snapshot.emojis.values()))
    assert item["descriptions"]["ru"]["text"] == emoji.descriptions["ru"].text
    assert item["review_status"] == emoji.review.status.value
    assert _state(tmp_path) == before


def test_selected_pack_in_multi_pack_run_uses_saved_membership(config: MojiLexConfig) -> None:
    checkpoint = _save(
        config,
        names=("NewsEmoji", "OtherPack"),
        elements={
            "news": ElementCheckpoint(stage="media_verified"),
            "other": ElementCheckpoint(stage="media_verified"),
        },
        extra={"source_memberships": {"NewsEmoji": ["news"], "OtherPack": ["other"]}},
    )
    shown = packs.show_pack_command("NewsEmoji").result
    assert shown["counts"]["pending"] == 1
    assert shown["pack"]["names"] == ["NewsEmoji"]
    assert shown["pack"]["items"] == 1
    assert shown["pack"]["ai_ready"] == 0
    whole = packs.show_pack_command(checkpoint.run_id).result
    assert whole["pack"]["names"] == ["NewsEmoji", "OtherPack"]
    assert whole["pack"]["items"] == 2
    assert packs.list_packs_command().result["packs"][0]["items"] == 2


def test_corrupt_run_is_skipped_without_modifying_it(config: MojiLexConfig, tmp_path: Path) -> None:
    _save(config)
    assert config.runs_dir is not None
    (config.runs_dir / ("mlxrun_" + "b" * 32 + ".json")).write_text("broken", encoding="utf-8")
    before = _state(tmp_path)
    result = packs.list_packs_command()
    assert len(result.result["packs"]) == 1
    assert result.warnings
    assert _state(tmp_path) == before


def test_summary_uses_validated_ai_stage_and_original_budget(config: MojiLexConfig) -> None:
    checkpoint = _save(
        config,
        elements={
            "ready": ElementCheckpoint(
                stage="ai_facets_ready", ai_cache_key="a" * 64, ai_facets_complete=True
            ),
            "unfinished": ElementCheckpoint(stage="ai_cached", ai_cache_key="b" * 64),
        },
        extra={"max_ai_requests": 75},
    )
    checkpoint = checkpoint.model_copy(update={"ai_requests_used": 69})
    assert config.runs_dir is not None
    RunStore(config.runs_dir).save(checkpoint)
    summary = packs.list_packs_command().result["packs"][0]
    assert summary["ai_ready"] == 1
    assert summary["requests_used"] == 69
    assert summary["max_ai_requests"] == 75
    assert summary["max_ai_requests_source"] == "checkpoint"


def test_show_never_recovers_or_cleans_pending_staging_transaction(
    config: MojiLexConfig,
    tmp_path: Path,
) -> None:
    staging = tmp_path / "staging"
    write_fixture(staging)
    transaction = staging / ".mojilex-atomic-write"
    transaction.mkdir()
    _save(
        config,
        names=("SuspiciousCats",),
        extra={"staging_repository": str(staging)},
        elements={NATIVE_EMOJI_ID: ElementCheckpoint(stage="media_verified")},
    )
    before = _state(tmp_path)
    result = packs.show_pack_command("SuspiciousCats")
    assert result.result["counts"]["ready"] == 0
    assert result.warnings
    assert transaction.is_dir()
    assert _state(tmp_path) == before


def test_pack_operation_selects_completed_results_or_unfinished_progress(
    config: MojiLexConfig,
) -> None:
    assert config.runs_dir is not None and config.cache_dir is not None
    store = RunStore(config.runs_dir)
    imported = _save(config, run_digit="c", minute=0).model_copy(
        update={"command": "import", "status": "succeeded"}
    )
    store.save(imported)
    described = _save(
        config,
        elements={
            str(index): ElementCheckpoint(
                stage="ai_facets_ready",
                ai_cache_key="a" * 64,
                ai_facets_complete=True,
            )
            for index in range(100)
        },
    ).model_copy(update={"command": "describe", "status": "succeeded"})
    store.save(described)
    unfinished = _save(
        config,
        run_digit="b",
        minute=2,
        elements={str(index): ElementCheckpoint(stage="media_verified") for index in range(4)},
    )
    with CacheStore(config.cache_dir / "cache-v1.sqlite3") as cache:
        cache.put_ai("a" * 64, _result())
    assert packs.resolve_pack_run("NewsEmoji").run_id == unfinished.run_id
    assert packs.resolve_pack_run("NewsEmoji", purpose="latest").run_id == unfinished.run_id
    assert packs.resolve_pack_run("NewsEmoji", purpose="view").run_id == described.run_id
    assert packs.resolve_pack_run("NewsEmoji", purpose="publish").run_id == described.run_id
    assert packs.resolve_pack_run("NewsEmoji", purpose="resume").run_id == unfinished.run_id
    assert packs.show_pack_command("NewsEmoji").run_id == described.run_id
    assert packs.show_pack_command("NewsEmoji").result["counts"]["ready"] == 100
    row = packs.list_packs_command().result["packs"][0]
    assert row["run_id"] == described.run_id
    assert row["ai_ready"] == 100
    assert row["latest_unfinished"]["run_id"] == unfinished.run_id
    assert row["latest_unfinished"]["items"] == 4
    assert row["latest_unfinished"]["ai_ready"] == 0
    for purpose in ("latest", "view", "publish", "resume"):
        assert (
            packs.resolve_pack_run(unfinished.run_id, purpose=purpose).run_id == unfinished.run_id
        )
        assert packs.resolve_pack_run(imported.run_id, purpose=purpose).run_id == imported.run_id
    assert packs.show_pack_command(unfinished.run_id).run_id == unfinished.run_id


def test_publish_name_needs_completed_describe_without_redirecting_explicit_id(
    config: MojiLexConfig,
) -> None:
    checkpoint = _save(config)
    with pytest.raises(CommandError) as error:
        packs.resolve_pack_run("NewsEmoji", purpose="publish")
    assert error.value.error.code == "CONFIG_MISSING"
    assert packs.resolve_pack_run(checkpoint.run_id, purpose="publish").run_id == checkpoint.run_id


def test_resume_uses_latest_unfinished_even_when_completed_run_is_newer(
    config: MojiLexConfig,
) -> None:
    unfinished = _save(config)
    completed = _save(config, run_digit="b", minute=2).model_copy(update={"status": "succeeded"})
    assert config.runs_dir is not None
    RunStore(config.runs_dir).save(completed)
    assert packs.resolve_pack_run("NewsEmoji", purpose="resume").run_id == unfinished.run_id


@pytest.mark.parametrize("purpose", ["latest", "view", "publish", "resume"])
def test_purpose_selection_does_not_hide_repository_ambiguity(
    config: MojiLexConfig,
    purpose: str,
) -> None:
    _save(config)
    _save(config, run_digit="b", repository="Other/repository")
    with pytest.raises(CommandError) as error:
        packs.resolve_pack_run("NewsEmoji", purpose=purpose)
    assert error.value.error.code == "CONFIG_INVALID"


@pytest.mark.parametrize("purpose", ["describe", "resume", "publish"])
def test_named_mutation_never_expands_to_other_packs_in_same_run(
    config: MojiLexConfig,
    purpose: str,
) -> None:
    checkpoint = _save(config, names=("NewsEmoji", "OtherPack")).model_copy(
        update={"command": "describe", "status": "succeeded"}
    )
    assert config.runs_dir is not None
    RunStore(config.runs_dir).save(checkpoint)
    with pytest.raises(CommandError) as error:
        packs.resolve_pack_run("NewsEmoji", purpose=purpose)
    assert error.value.error.code == "CONFIG_INVALID"
    assert "multiple packs" in error.value.error.message
    assert checkpoint.run_id in error.value.error.hint
    assert "all packs" in error.value.error.hint
    assert packs.resolve_pack_run(checkpoint.run_id, purpose=purpose).run_id == checkpoint.run_id
    assert packs.resolve_pack_run("NewsEmoji", purpose="view").run_id == checkpoint.run_id


def test_describe_purpose_preserves_latest_single_pack_selection(config: MojiLexConfig) -> None:
    _save(config)
    latest = _save(config, run_digit="b", minute=2)
    assert packs.resolve_pack_run("NewsEmoji", purpose="describe").run_id == latest.run_id
