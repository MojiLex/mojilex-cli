from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mojilex_cli import cli
from mojilex_cli.commands import packs, workflow
from mojilex_cli.commands.runtime import CommandError, CommandResult
from mojilex_cli.i18n import use_ui_language

RUN = "mlxrun_" + "a" * 32


@pytest.fixture
def saved(monkeypatch):
    checkpoint = SimpleNamespace(run_id=RUN, command="describe", status="succeeded")
    monkeypatch.setattr(packs, "resolve_pack_run", lambda selector, **kwargs: checkpoint)
    return checkpoint


def _view():
    return CommandResult(
        run_id=RUN,
        result={
            "pack": {"names": ["NewsEmoji"], "items": 1},
            "counts": {"ready": 1, "pending": 0, "missing": 0, "invalid": 0},
            "items": [
                {
                    "native_id": "123",
                    "descriptions": {
                        "ru": {
                            "text": "Взрыв [red]текст[/red]",
                            "motion": "Вспыхивает",
                            "usage": ["Реакция"],
                        }
                    },
                    "content": {"rating": "sensitive", "warnings": ["flashing"]},
                    "semantic_tags": ["взрыв"],
                    "facets": {"text_content": {"items": [{"value": "BOOM"}]}},
                }
            ],
        },
    )


@pytest.mark.parametrize("command", ["show", "review"])
def test_browse_is_readable_retains_warnings_and_does_not_approve(monkeypatch, command):
    called = []
    monkeypatch.setattr(
        packs, "show_pack_command", lambda selector, **kw: called.append((selector, kw)) or _view()
    )
    monkeypatch.setattr(cli, "review_command", lambda *a, **kw: pytest.fail("unexpected approval"))
    with use_ui_language("ru"):
        result = CliRunner().invoke(cli.app, [command, "NewsEmoji"])
    assert result.exit_code == 0, result.output
    assert "Взрыв [red]текст[/red]" in result.stdout
    assert "Вспыхивает" in result.stdout and "BOOM" in result.stdout
    assert "flashing" in result.stdout and "sensitive" in result.stdout
    assert "{'" not in result.stdout
    assert called == [("NewsEmoji", {"review": True} if command == "review" else {})]


@pytest.mark.parametrize("command", ["show", "review"])
def test_browse_json_is_single_envelope(monkeypatch, command):
    monkeypatch.setattr(packs, "show_pack_command", lambda *a, **kw: _view())
    result = CliRunner().invoke(cli.app, [command, "NewsEmoji", "--json"])
    assert result.exit_code == 0, result.output
    assert result.stdout.count("\n") == 1 and result.stderr == ""
    data = json.loads(result.stdout)
    assert data["result"]["items"][0]["content"]["warnings"] == ["flashing"]


def test_list_human_has_names_and_progress_without_internal_paths(monkeypatch):
    monkeypatch.setattr(
        packs,
        "list_packs_command",
        lambda: CommandResult(
            result={
                "packs": [
                    {
                        "names": ["NewsEmoji"],
                        "run_id": RUN,
                        "ai_ready": 80,
                        "items": 100,
                        "status": "interrupted",
                        "updated_at": "2026-09-13T00:00:00+00:00",
                    }
                ]
            }
        ),
    )
    result = CliRunner().invoke(cli.app, ["list"])
    assert result.exit_code == 0, result.output
    assert "NewsEmoji" in result.stdout and "80/100" in result.stdout
    assert RUN not in result.stdout and "{'" not in result.stdout


def test_publish_name_uses_saved_run_without_analysis(monkeypatch, saved):
    calls = []
    monkeypatch.setattr(
        workflow,
        "run_submit",
        lambda target, **kwargs: (
            calls.append((target, kwargs))
            or CommandResult(
                publication={"pull_request_url": "https://github.com/example/data/pull/1"}
            )
        ),
    )
    monkeypatch.setattr(workflow, "run_describe", lambda *a, **kw: pytest.fail("unexpected AI"))
    result = CliRunner().invoke(cli.app, ["publish", "NewsEmoji"])
    assert result.exit_code == 0, result.output
    assert calls[0][0] == RUN and calls[0][1]["publish"] == "pr"
    assert calls[0][1]["direct_push"] is False
    assert "https://github.com/example/data/pull/1" in result.stdout


def test_publish_local_never_selects_upload_mode(monkeypatch, saved):
    calls = []
    monkeypatch.setattr(
        workflow, "run_submit", lambda target, **kw: calls.append(kw) or CommandResult()
    )
    result = CliRunner().invoke(cli.app, ["publish", "NewsEmoji", "--local", "--json"])
    assert result.exit_code == 0, result.output
    assert calls[0]["publish"] == "local" and not calls[0]["direct_push"]


def test_unfinished_pack_cannot_be_published(monkeypatch, saved):
    saved.status = "interrupted"
    monkeypatch.setattr(workflow, "run_submit", lambda *a, **kw: pytest.fail("unexpected publish"))
    result = CliRunner().invoke(cli.app, ["publish", "NewsEmoji", "--json"])
    assert result.exit_code != 0
    assert "not complete" in json.loads(result.stdout)["errors"][0]["message"]


@pytest.mark.parametrize("selector", ["NewsEmoji", RUN])
def test_completed_resume_neither_requests_credentials_nor_runs_pipeline(
    monkeypatch, saved, selector
):
    monkeypatch.setattr(cli, "_with_runtime_secrets", lambda *a, **kw: pytest.fail("credentials"))
    monkeypatch.setattr(workflow, "run_resume_sync", lambda *a, **kw: pytest.fail("pipeline"))
    result = CliRunner().invoke(cli.app, ["resume", selector, "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "noop"


def test_incomplete_resume_name_preserves_id_and_overrides(monkeypatch, saved):
    saved.status = "interrupted"
    calls = []
    monkeypatch.setattr(
        workflow,
        "run_resume_sync",
        lambda run_id, **kw: calls.append((run_id, kw)) or CommandResult(),
    )
    workflow.resume_command("NewsEmoji", ai_concurrency=4, download_concurrency=8)
    assert calls[0][0] == RUN
    assert calls[0][1]["ai_concurrency"] == 4 and calls[0][1]["download_concurrency"] == 8


def test_describe_name_reuses_saved_run(monkeypatch, saved):
    calls = []
    monkeypatch.setattr(
        workflow,
        "run_describe",
        lambda run_id, options: calls.append((run_id, options)) or CommandResult(),
    )
    workflow.describe_command(
        ["NewsEmoji"],
        provider=None,
        model=None,
        max_ai_requests=100,
        max_cost_usd=None,
        allow_unknown_cost=False,
    )
    assert calls[0][0] == RUN and calls[0][1].max_ai_requests == 100


def test_ambiguous_selector_does_not_publish(monkeypatch):
    def ambiguous(selector, **kwargs):
        raise CommandError("CONFIG_INVALID", "ambiguous", hint="Choose a run from mojilex list.")

    monkeypatch.setattr(packs, "resolve_pack_run", ambiguous)
    monkeypatch.setattr(workflow, "run_submit", lambda *a, **kw: pytest.fail("publish"))
    result = CliRunner().invoke(cli.app, ["publish", "NewsEmoji", "--json"])
    assert (
        result.exit_code != 0 and json.loads(result.stdout)["errors"][0]["message"] == "ambiguous"
    )


def test_existing_explicit_review_action_is_supported(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "review_command", lambda *args: calls.append(args) or CommandResult())
    result = CliRunner().invoke(
        cli.app, ["review", "mxe_example", "approve", "--reviewer", "Tester"]
    )
    assert result.exit_code == 0, result.output
    assert calls[0][1:] == ("mxe_example", "approve", "Tester")
