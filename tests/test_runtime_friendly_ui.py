from __future__ import annotations

import json

import pytest
import typer

from mojilex_cli.commands import runtime
from mojilex_cli.i18n import use_ui_language


def test_command_result_capture_is_scoped_and_preserves_exact_identity():
    first = runtime.CommandResult(run_id="first")
    second = runtime.CommandResult(run_id="second")
    with runtime.capture_command_results() as outer:
        runtime.execute("import", lambda: first, json_output=False, quiet=True)
        with runtime.capture_command_results() as inner:
            runtime.execute("import", lambda: second, json_output=False, quiet=True)
        assert inner == [second]
        assert outer == [first]
        runtime.execute("describe", lambda: first, json_output=False, quiet=True)
    assert outer == [first, first]
    assert runtime._RESULT_COLLECTOR.get() is None


def test_command_result_capture_does_not_invent_a_result_on_exception():
    def fail():
        raise runtime.CommandError("CONFIG_INVALID", "invalid", hint="fix it")

    with runtime.capture_command_results() as results, pytest.raises(typer.Exit):
        runtime.execute("import", fail, json_output=False, quiet=True)
    assert results == []
    assert runtime._RESULT_COLLECTOR.get() is None


@pytest.mark.parametrize("opened", [True, False])
def test_gallery_output_avoids_internal_path_when_browser_opened(capsys, opened):
    with use_ui_language("en"):
        runtime.execute(
            "gallery",
            lambda: runtime.CommandResult(
                result={
                    "gallery_path": "private-workspace/gallery.html",
                    "browser_opened": opened,
                    "counts": {"ready": 100},
                }
            ),
            json_output=False,
        )
    output = capsys.readouterr().out
    assert "Descriptions: 100" in output
    assert ("private-workspace" in output) is (not opened)
    assert "browser_opened" not in output


def test_settings_are_readable_with_explanations(capsys):
    with use_ui_language("en"):
        runtime.execute(
            "settings",
            lambda: runtime.CommandResult(
                result={
                    "view": "settings",
                    "settings": [
                        {
                            "key": "ai.max_ai_requests",
                            "label": "Request limit",
                            "value": 100,
                            "description": "Includes retries.",
                            "source": "default",
                            "editable": True,
                        }
                    ],
                    "notes": ["Saved progress keeps its existing limit."],
                }
            ),
            json_output=False,
        )
    output = capsys.readouterr().out
    assert "Request limit" in output and "100" in output
    assert "Includes retries." in output
    assert "Saved progress keeps its existing limit." in output
    assert "'description':" not in output


def test_sync_noop_explains_existing_packs_without_internal_lists(capsys):
    with use_ui_language("ru"):
        runtime.execute(
            "sync",
            lambda: runtime.CommandResult(
                status=runtime.RunStatus.NOOP,
                result={
                    "added_packs": [],
                    "skipped_packs": ["internal-id"],
                    "selected_runs": [],
                    "changed_paths": [],
                    "ready_runs_checked": 1,
                    "already_on_github": 1,
                },
            ),
            json_output=False,
        )
    output = capsys.readouterr().out
    assert "уже есть на GitHub" in output
    assert "Ничего не отправлено" in output
    assert "internal-id" not in output and "[]" not in output


def test_sync_no_ready_runs_explains_nothing_was_sent(capsys):
    with use_ui_language("ru"):
        runtime.execute(
            "sync",
            lambda: runtime.CommandResult(
                status=runtime.RunStatus.NOOP,
                result={"ready_runs_checked": 0, "changed_paths": [], "added_packs": []},
            ),
            json_output=False,
        )
    output = capsys.readouterr().out
    assert "Нет готовых сохранённых паков" in output
    assert "ничего не отправлено" in output


def test_sync_preview_shows_counts_without_exposing_paths_or_run_ids(capsys):
    with use_ui_language("ru"):
        runtime.execute(
            "sync",
            lambda: runtime.CommandResult(
                result={
                    "added_packs": ["internal-pack-id"],
                    "selected_runs": ["internal-run-id"],
                    "changed_paths": ["data/internal/path.json"],
                },
                publication={"mode": "local", "preview": True},
            ),
            json_output=False,
        )
    output = capsys.readouterr().out
    assert "Подготовлено паков: 1; изменено файлов: 1" in output
    assert "Эта команда не отправляла" in output
    assert "internal-" not in output


def test_completion_distinguishes_local_save_from_github_and_hides_paths(capsys):
    with use_ui_language("en"):
        runtime.execute(
            "describe",
            lambda: runtime.CommandResult(
                result={
                    "sources_processed": 1,
                    "items_added": 100,
                    "ai_requests": 63,
                    "staging_repository": "internal-folder",
                    "changed_paths": ["data/emoji.jsonl"],
                    "review_routing": {"internal": "details"},
                },
                publication={"mode": "staging", "path": "internal-folder"},
            ),
            json_output=False,
        )
    output = capsys.readouterr().out
    assert "Emojis added: 100" in output
    assert "AI requests used: 63" in output
    assert "Results saved locally" in output
    assert "mojilex list" in output
    assert "internal-folder" not in output and "review_routing" not in output


@pytest.mark.parametrize("json_output", [False, True])
def test_validation_failure_is_actionable_without_altering_machine_details(capsys, json_output):
    detail = "$.items[].facets.suggested_uses[] (literal_error)"

    def fail():
        raise runtime.CommandError("AI_OUTPUT_INVALID", detail, hint="Original technical hint")

    with use_ui_language("en"), pytest.raises(typer.Exit):
        runtime.execute("describe", fail, json_output=json_output)
    output = capsys.readouterr()
    if json_output:
        assert json.loads(output.out)["errors"][0]["message"] == detail
    else:
        assert "could not produce a valid description" in output.err
        assert "mojilex resume NAME" in output.err
        assert detail not in output.err


def test_verbose_validation_failure_retains_technical_details(capsys):
    def fail():
        raise runtime.CommandError("AI_OUTPUT_INVALID", "specific schema path", hint="fix")

    with pytest.raises(typer.Exit):
        runtime.execute("describe", fail, json_output=False, verbose=True)
    assert "specific schema path" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command,result,next_command",
    [
        ("import", {"collections_imported": 1}, "describe"),
        ("describe", {"sources_processed": 1}, "show"),
    ],
)
def test_named_completion_shows_copyable_next_step(capsys, command, result, next_command):
    runtime.execute(
        command,
        lambda: runtime.CommandResult(result={**result, "pack_name": "NewsEmoji"}),
        json_output=False,
    )
    assert f"mojilex {next_command} NewsEmoji" in capsys.readouterr().out
