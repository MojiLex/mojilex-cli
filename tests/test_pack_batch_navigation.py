from __future__ import annotations

import pytest

from mojilex_cli.commands import interactive, packs, workflow
from mojilex_cli.commands.runtime import CommandError, CommandResult
from mojilex_cli.config import MojiLexConfig
from mojilex_cli.runs import ElementCheckpoint, RunStore
from test_pack_commands import _save, _state


@pytest.fixture
def batch(tmp_path, monkeypatch):
    config = MojiLexConfig(cache_dir=tmp_path / "cache", runs_dir=tmp_path / "runs")
    for module in (packs, interactive, workflow):
        monkeypatch.setattr(module, "load_config", lambda: config)
    checkpoint = _save(
        config,
        names=("FirstPack", "SecondPack", "NotStarted"),
        elements={
            "one": ElementCheckpoint(stage="fingerprint_ready", fingerprint_complete=True),
            "two": ElementCheckpoint(stage="media_verified"),
        },
        extra={
            "source_memberships": {"FirstPack": ["one"], "SecondPack": ["two", "missing"]},
            "max_ai_requests": 75,
        },
    ).model_copy(update={"command": "import", "ai_requests_used": 12})
    RunStore(config.runs_dir).save(checkpoint)
    return config, checkpoint


def test_old_batch_lists_individual_packs_without_writing_or_losing_missing_items(batch, tmp_path):
    _, checkpoint = batch
    before = _state(tmp_path)
    rows = packs.list_packs_command().result["packs"]
    assert len(rows) == 3
    by_name = {row["names"][0]: row for row in rows}
    assert by_name["FirstPack"]["items"] == 1
    assert by_name["SecondPack"]["items"] == 2
    assert by_name["NotStarted"]["items"] == 0
    for name, row in by_name.items():
        assert row["selector"] == f"{checkpoint.run_id}:{name}"
        assert row["requests_used"] == 12
        assert row["max_ai_requests"] == 75
        assert row["budget_scope"] == "batch"
        assert packs.show_pack_command(row["selector"]).result["pack"]["names"] == [name]
    assert _state(tmp_path) == before


def test_same_pack_in_different_batches_has_one_row_and_scoped_history(batch):
    config, checkpoint = batch
    _save(config, run_digit="b", minute=2, names=("FirstPack", "ThirdPack"))
    rows = packs.list_packs_command().result["packs"]
    assert len(rows) == 4
    page = interactive._pack_state(f"{checkpoint.run_id}:FirstPack")
    assert len(page["history"]) == 2
    assert all(row["names"] == ["FirstPack"] for row in page["history"])
    assert page["sources"] == ["https://t.me/addemoji/FirstPack"]
    assert page["analyzable"] is True


def test_scoped_resume_dispatches_only_selected_pack(batch, monkeypatch):
    _, checkpoint = batch
    calls = []
    monkeypatch.setattr(
        workflow,
        "run_resume_sync",
        lambda run_id, **kw: calls.append((run_id, kw)) or CommandResult(),
    )
    workflow.resume_command(f"{checkpoint.run_id}:SecondPack")
    assert calls[0][0] == checkpoint.run_id
    assert calls[0][1]["selected_sources"] == ("https://t.me/addemoji/SecondPack",)


@pytest.mark.parametrize("form", ["name", "url", "scoped"])
def test_describe_retains_scope_and_parent_run(batch, monkeypatch, form):
    _, checkpoint = batch
    selectors = {
        "name": "FirstPack",
        "url": "https://t.me/addemoji/FirstPack",
        "scoped": f"{checkpoint.run_id}:FirstPack",
    }
    calls = []
    monkeypatch.setattr(
        workflow,
        "run_describe",
        lambda run_id, options: calls.append((run_id, options)) or CommandResult(),
    )
    workflow.describe_command(
        [selectors[form]],
        provider=None,
        model=None,
        max_ai_requests=None,
        max_cost_usd=None,
        allow_unknown_cost=False,
    )
    assert calls[0][0] == checkpoint.run_id
    assert calls[0][1].selected_sources == ("https://t.me/addemoji/FirstPack",)


def test_scoped_selector_rejects_foreign_pack(batch):
    _, checkpoint = batch
    with pytest.raises(CommandError, match="not part"):
        packs.show_pack_command(f"{checkpoint.run_id}:ForeignPack")


def test_page_resume_keeps_scoped_selector(batch, monkeypatch):
    _, checkpoint = batch
    choices = iter([3, None])
    monkeypatch.setattr(interactive, "select", lambda *a, **kw: next(choices))
    monkeypatch.setattr(interactive, "pause", lambda: None)
    calls = []
    selector = f"{checkpoint.run_id}:SecondPack"
    interactive._pack_page(selector, lambda args: calls.append(args) or True)
    assert calls == [["resume", selector]]
