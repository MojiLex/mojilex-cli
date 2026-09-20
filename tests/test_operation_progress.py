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
        original = context.operations[-1]
        context.operations[-1] = ("Uploading to GitHub", time.monotonic() - 65)
        context.operation_live.refresh()
        assert "01:05" in output.getvalue()
        context.operations[-1] = original
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


def test_startup_displays_pulsing_bar_immediately_without_invented_percentage(terminal):
    from rich.console import Group
    from rich.progress_bar import ProgressBar

    context, output = terminal
    with runtime.operation_progress("Reading saved packs"):
        assert context.operation_live is not None
        view = runtime._operation_view(context)
        assert isinstance(view, Group)
        bars = [item for item in view.renderables if isinstance(item, ProgressBar)]
        assert len(bars) == 1
        assert bars[0].pulse is True and bars[0].total is None
        assert "Reading saved packs" in output.getvalue()
        assert "%" not in output.getvalue()
        with runtime.operation_progress("Checking official repository"):
            nested = runtime._operation_view(context)
            assert isinstance(nested, Group)
            assert bars[0] in nested.renderables
        runtime.begin_pack_queue(["Alpha"])
        assert context.operation_live is None and context.live is not None


def test_preparation_dashboard_keeps_elapsed_alive_and_clears_on_error(terminal):
    context, output = terminal
    runtime.begin_pack_queue(["Alpha"])
    with pytest.raises(RuntimeError), runtime.preparation_progress("Loading saved dataset"):
        assert "Loading saved dataset" in output.getvalue()
        assert context.operation_live is None
        assert context.live is not None
        before = len(output.getvalue())
        time.sleep(0.7)
        assert len(output.getvalue()) > before
        assert context.pack_queue.stages == {"Alpha": "waiting"}
        raise RuntimeError("failed")
    assert context.preparations == []
    assert context.preparation_thread is None
    assert context.operations == []
    plain = StringIO()
    Console(file=plain).print(context.progress_view)
    assert "Loading saved dataset" not in plain.getvalue()


def test_concurrent_preparations_survive_out_of_order_exit(terminal):
    context, output = terminal
    runtime.begin_pack_queue(["Alpha", "Beta"])
    first = runtime.preparation_progress("Loading Alpha")
    second = runtime.preparation_progress("Loading Beta")
    first.__enter__()
    second.__enter__()
    heartbeat = context.preparation_thread
    assert "concurrent operations: 2" in output.getvalue()
    first.__exit__(None, None, None)
    assert [label for label, _ in context.preparations] == ["Loading Beta"]
    assert [label for label, _ in context.operations] == ["Loading Beta"]
    assert context.preparation_thread is heartbeat
    second.__exit__(None, None, None)
    assert context.preparations == []
    assert context.operations == []
    assert context.preparation_thread is None


def test_preparation_heartbeat_does_not_redraw_prompt(terminal):
    context, output = terminal
    runtime.begin_pack_queue(["Alpha"])
    with runtime.preparation_progress("Loading dataset"):
        with runtime.suspend_progress():
            before = output.getvalue()
            time.sleep(0.7)
            assert output.getvalue() == before
        assert context.preparations


@pytest.mark.parametrize("quiet,json_output", [(True, False), (False, True)])
def test_preparation_stays_silent_for_machine_and_quiet(terminal, quiet, json_output):
    context, output = terminal
    context.quiet = quiet
    context.json_output = json_output
    with runtime.preparation_progress("Hidden"):
        assert context.preparation_thread is None
        assert not context.preparations
    assert output.getvalue() == ""
