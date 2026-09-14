from __future__ import annotations

import time
from io import StringIO

import pytest
import typer
from rich.console import Console
from rich.text import Text

from mojilex_cli import cli
from mojilex_cli.commands import runtime


def test_import_activity_names_the_whole_media_processing_stage(monkeypatch) -> None:
    monkeypatch.setattr(runtime, "current_ui_language", lambda: "ru")
    assert runtime._command_activity("import") == "Импорт и обработка пака"
    monkeypatch.setattr(runtime, "current_ui_language", lambda: "en")
    assert runtime._command_activity("import") == "Importing and processing pack"


@pytest.fixture
def terminal(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=90)
    monkeypatch.setattr(runtime, "Console", lambda **kw: console)
    context = runtime._CommandContext("test")
    token = runtime._COMMAND_CONTEXT.set(context)
    try:
        yield context, output
    finally:
        runtime.finish_live_progress()
        runtime._stop_operation_live(context)
        runtime._COMMAND_CONTEXT.reset(token)


def test_operation_animates_while_synchronous_work_is_blocked(terminal):
    context, output = terminal
    with runtime.operation_progress("Uploading to GitHub"):
        assert context.operation_live is not None
        first = len(output.getvalue())
        time.sleep(0.6)
        assert len(output.getvalue()) > first
        context.operations[-1] = ("Uploading to GitHub", time.monotonic() - 65)
        context.operation_live.refresh()
        assert "01:05" in output.getvalue()
        assert "%" not in output.getvalue()
    assert context.operation_live is None
    assert context.operations == []


def test_nested_stage_restores_parent_and_closes_on_failure(terminal):
    context, _ = terminal
    with runtime.operation_progress("Publication"):
        with pytest.raises(RuntimeError), runtime.operation_progress("Push"):
            assert context.operations[-1][0] == "Push"
            raise RuntimeError("connection lost")
        assert context.operations[-1][0] == "Publication"
    assert not context.operations and context.operation_live is None


def test_prompt_suspends_both_indicators_and_blocks_heartbeat_redraw(terminal):
    context, output = terminal
    with runtime.operation_progress("Preparing"):
        assert context.operation_live is not None
        with runtime.suspend_progress():
            assert context.operation_live is None
            before = output.getvalue()
            runtime.pause_live_progress(False)
            runtime.update_live_progress(Text("AI heartbeat"))
            assert output.getvalue() == before
            assert context.live is None
        runtime.finish_live_progress()
        assert context.operation_live is not None
    assert context.operation_live is None


def test_item_progress_replaces_activity_without_nested_live_displays(terminal):
    context, _ = terminal
    with runtime.operation_progress("Analyzing"):
        runtime.update_live_progress(Text("3 / 10"))
        assert context.operation_live is None and context.live is not None
        runtime.finish_live_progress()
        assert context.live is None and context.operation_live is not None
    assert context.operation_live is None


@pytest.mark.parametrize("quiet,json_output", [(True, False), (False, True)])
def test_operation_never_displays_in_quiet_or_json(terminal, quiet, json_output):
    context, output = terminal
    context.quiet = quiet
    context.json_output = json_output
    with runtime.operation_progress("Hidden"):
        assert context.operation_live is None
    assert output.getvalue() == ""


def test_confirmation_and_secret_prompt_pause_activity(terminal, monkeypatch):
    context, _ = terminal

    def confirm(*a, **kw):
        assert context.operation_live is None and context.prompt_depth == 1
        return True

    def prompt(*a, **kw):
        assert context.operation_live is None and context.prompt_depth == 1
        return "example"

    monkeypatch.setattr(runtime, "ui_confirm", confirm)
    monkeypatch.setattr(typer, "prompt", prompt)
    with runtime.operation_progress("Preparing upload"):
        runtime.require_confirmation("Upload?", yes=False, non_interactive=False, json_output=False)
        assert context.operation_live is not None
        assert cli._progress_prompt("Missing key", hide_input=True) == "example"
        assert context.operation_live is not None


def test_official_pack_confirmation_suspends_spinner_with_negative_default(terminal, monkeypatch):
    context, _ = terminal
    calls = []

    def confirm(message, *, default):
        assert context.operation_live is None and context.prompt_depth == 1
        calls.append(default)
        return False

    monkeypatch.setattr(cli, "ui_confirm", confirm)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    callback = cli._official_confirmation_callback(
        non_interactive=False, json_output=False, quiet=False
    )
    with runtime.operation_progress("Importing"):
        assert callback("Already published packs?") is False
        assert context.operation_live is not None
    assert calls == [False]


def test_generic_command_spinner_closes_after_failure(terminal):
    outer_context, _ = terminal
    captured = []

    def fail():
        context = runtime._COMMAND_CONTEXT.get()
        captured.append(context)
        assert context is not None and context.operation_live is not None
        raise RuntimeError("failed")

    with pytest.raises(typer.Exit):
        runtime.execute("publish", fail, json_output=False)
    assert captured[0].operation_live is None and not captured[0].operations
    assert runtime._COMMAND_CONTEXT.get() is outer_context


def test_doctor_installer_owns_terminal_until_it_returns(terminal, monkeypatch):
    contexts = []
    result = runtime.CommandResult(result={"install_commands": ["installer"], "checks": {}})
    monkeypatch.setattr(cli, "doctor_command", lambda: result)

    def installer(checks):
        context = runtime._COMMAND_CONTEXT.get()
        assert context is not None
        assert context.operation_live is None and context.prompt_depth == 1
        contexts.append(context)
        return result

    monkeypatch.setattr(cli, "install_media_dependencies_command", installer)
    cli.doctor(install=True, non_interactive=False, json_output=False, quiet=False, debug=False)
    assert len(contexts) == 1 and contexts[0].operation_live is None
