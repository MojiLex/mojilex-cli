from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from mojilex_cli import cli
from mojilex_cli.commands import interactive as ui
from mojilex_cli.commands import packs
from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.i18n import use_ui_language

RUN = "mlxrun_" + "a" * 32
UNFINISHED = "mlxrun_" + "b" * 32


def view():
    return CommandResult(
        run_id=RUN,
        result={
            "pack": {
                "names": ["NewsEmoji"],
                "items": 1,
                "requests_used": 5,
                "max_ai_requests": 100,
            },
            "counts": {"ready": 1, "pending": 0, "missing": 0, "invalid": 0},
            "items": [
                {
                    "native_id": "123",
                    "descriptions": {
                        "ru": {"text": "Взрыв", "motion": "Вспыхивает"},
                        "en": {"text": "Explosion"},
                    },
                    "content": {"warnings": ["flashing"], "rating": "sensitive"},
                }
            ],
        },
    )


@pytest.fixture
def navigation(monkeypatch):
    monkeypatch.setattr(ui, "pause", lambda: None)
    monkeypatch.setattr(ui, "is_interactive", lambda: True)

    def choices(*values):
        iterator = iter(values)
        monkeypatch.setattr(ui, "select", lambda *a, **kw: next(iterator))

    return choices


def test_bare_command_opens_menu_only_in_terminal(monkeypatch):
    calls = []
    monkeypatch.setattr(ui, "is_interactive", lambda: True)
    monkeypatch.setattr(ui, "run_menu", lambda dispatch: calls.append(dispatch))
    result = CliRunner().invoke(cli.app, [])
    assert result.exit_code == 0
    assert calls == [ui.dispatch_command]
    monkeypatch.setattr(ui, "is_interactive", lambda: False)
    result = CliRunner().invoke(cli.app, [])
    assert result.exit_code == 2
    assert len(calls) == 1


@pytest.mark.parametrize("command", ["show", "review"])
def test_terminal_view_uses_browser_but_json_stays_machine_readable(monkeypatch, command):
    calls = []
    monkeypatch.setattr(ui, "is_interactive", lambda: True)
    monkeypatch.setattr(ui, "browse_descriptions", lambda name: calls.append(name) or view())
    monkeypatch.setattr(packs, "show_pack_command", lambda *a, **kw: view())
    result = CliRunner().invoke(cli.app, [command, "NewsEmoji"])
    assert result.exit_code == 0
    assert calls == ["NewsEmoji"]
    result = CliRunner().invoke(cli.app, [command, "NewsEmoji", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["command"] == command
    assert calls == ["NewsEmoji"]


def test_all_fields_bypasses_interactive_view(monkeypatch):
    monkeypatch.setattr(ui, "is_interactive", lambda: True)
    monkeypatch.setattr(ui, "browse_descriptions", lambda *a: pytest.fail("interactive viewer"))
    monkeypatch.setattr(packs, "show_pack_command", lambda *a, **kw: view())
    result = CliRunner().invoke(cli.app, ["show", "NewsEmoji", "--all"])
    assert result.exit_code == 0
    assert "flashing" in result.stdout and "Explosion" in result.stdout


def test_gallery_json_does_not_open_browser(monkeypatch):
    from mojilex_cli.commands import gallery

    calls = []
    monkeypatch.setattr(
        gallery, "gallery_command", lambda pack, **kw: calls.append((pack, kw)) or CommandResult()
    )
    result = CliRunner().invoke(cli.app, ["show", "NewsEmoji", "--browser", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["command"] == "gallery"
    assert calls == [("NewsEmoji", {"open_browser": False})]


def test_navigation_handles_windows_and_posix_arrows(monkeypatch):
    monkeypatch.setattr(ui.Console, "clear", lambda self: None)
    keys = iter(["\xe0P", "\x1b[B", "\x1b[A", "\r"])
    monkeypatch.setattr(ui.typer, "getchar", lambda: next(keys))
    assert ui.select("Test", ["one", "two", "three"]) == 1


def test_escape_never_selects_an_action(monkeypatch):
    monkeypatch.setattr(ui.Console, "clear", lambda self: None)
    monkeypatch.setattr(ui.typer, "getchar", lambda: "\x1b")
    assert ui.select("Test", ["Publish"]) is None


def test_view_search_and_language_switch_are_read_only(navigation, monkeypatch):
    navigation(0, 1, 2, None)
    monkeypatch.setattr(ui, "show_pack_command", lambda selector: view())
    monkeypatch.setattr(ui.typer, "prompt", lambda *a, **kw: "")
    seen = []
    monkeypatch.setattr(ui, "_browse_item", lambda item, language: seen.append(language))
    with use_ui_language("ru"):
        result = ui.browse_descriptions("NewsEmoji")
    assert result.result["items"][0]["content"]["warnings"] == ["flashing"]
    assert seen == ["en"]


def test_details_stay_visible_until_explicit_return(navigation, monkeypatch):
    navigation(0, None)
    output = StringIO()
    monkeypatch.setattr(ui, "Console", lambda: Console(file=output, width=100, height=30))
    monkeypatch.setattr(Console, "pager", lambda *a, **kw: pytest.fail("system pager used"))
    reads = []

    def key():
        assert "Explosion" in output.getvalue()
        assert "flashing" in output.getvalue()
        assert "Enter/Esc" in output.getvalue()
        reads.append(True)
        return "\r"

    monkeypatch.setattr(ui.typer, "getchar", key)
    ui._browse_item(view().result["items"][0], "ru")
    assert reads == [True]


@pytest.mark.parametrize(
    "down,up,end,home",
    [("\xe0Q", "\xe0I", "\xe0O", "\xe0G"), ("\x1b[6~", "\x1b[5~", "\x1b[F", "\x1b[H")],
)
def test_details_scroll_pages_and_keep_last_page_open(monkeypatch, down, up, end, home):
    output = StringIO()
    console = Console(file=output, width=100, height=10)
    monkeypatch.setattr(ui, "Console", lambda: console)
    monkeypatch.setattr(console, "clear", lambda: (output.seek(0), output.truncate()))
    steps = iter(
        [
            ("ROW-00", down),
            ("ROW-06", up),
            ("ROW-00", end),
            ("ROW-19", down),
            ("ROW-19", home),
            ("ROW-00", "\x1b"),
        ]
    )

    def key():
        expected, key = next(steps)
        assert expected in output.getvalue()
        return key

    monkeypatch.setattr(ui.typer, "getchar", key)
    ui._read_text("Details", "\n".join(f"ROW-{index:02}" for index in range(20)))
    assert next(steps, None) is None


def state():
    return {
        "saved": view(),
        "history": [],
        "unfinished": {"run_id": UNFINISHED, "ai_ready": 0, "items": 4},
        "github": "Not checked",
        "target": "example/data",
        "publishable": True,
        "run_id": RUN,
        "sources": ["https://t.me/addemoji/NewsEmoji"],
    }


def test_resume_from_page_targets_visible_unfinished_run(navigation, monkeypatch):
    navigation(3, None)
    monkeypatch.setattr(ui, "_pack_state", lambda selector: state())
    calls = []
    ui._pack_page(RUN, lambda args: calls.append(args) or True)
    assert calls == [["resume", UNFINISHED]]


@pytest.mark.parametrize("accept", [True, False])
def test_publication_uses_exact_ready_run_after_explicit_action(navigation, monkeypatch, accept):
    navigation(5, None)
    monkeypatch.setattr(ui, "_pack_state", lambda selector: state())
    monkeypatch.setattr(ui, "confirm", lambda *a, **kw: accept)
    calls = []
    ui._pack_page(RUN, lambda args: calls.append(args) or True)
    assert calls == ([["publish", RUN, "--yes"]] if accept else [])


def test_failed_import_does_not_start_ai(navigation):
    calls = []
    ui._analyze("https://t.me/addemoji/NewsEmoji", lambda args: calls.append(args) or False)
    assert calls == [["import", "https://t.me/addemoji/NewsEmoji"]]


@pytest.mark.parametrize("source", ["https://t.me/addemoji/NewsEmoji", r"C:\packs\links.txt"])
def test_menu_analysis_uses_imported_id_without_automatic_publication(
    navigation, monkeypatch, source
):
    from contextlib import contextmanager

    from mojilex_cli.commands import runtime

    @contextmanager
    def captured():
        yield [SimpleNamespace(run_id=RUN)]

    monkeypatch.setattr(runtime, "capture_command_results", captured)
    monkeypatch.setattr(ui, "_resolve", lambda *a, **kw: pytest.fail("rediscovered by name"))
    calls = []
    ui._analyze(
        source,
        lambda args: calls.append(args) or True,
        repository="example/selected",
    )
    assert calls == [
        ["import", source, "--repo", "example/selected"],
        ["describe", RUN],
    ]


def test_menu_escape_does_not_load_data_or_run_commands(navigation, monkeypatch):
    navigation(None)
    monkeypatch.setattr(ui, "list_packs_command", lambda: pytest.fail("loaded data"))
    ui.run_menu(lambda args: pytest.fail("command ran"))


def test_basic_help_hides_advanced_commands_but_full_help_preserves_them():
    import typer

    from mojilex_cli.i18n import localize_command_tree

    command = typer.main.get_command(cli.app)
    localize_command_tree(command, "ru")
    assert command.commands["review"].hidden
    assert command.commands["submit"].hidden
    assert not command.commands["settings"].hidden
    assert not command.commands["publish"].hidden
    result = CliRunner().invoke(cli.app, ["--help-all"])
    assert result.exit_code == 0
    assert "submit" in result.stdout and "review" in result.stdout


@pytest.mark.parametrize("command", ["show", "review", "settings", "gallery"])
def test_real_entrypoint_json_never_opens_menu_or_browser(monkeypatch, capsys, command):
    from mojilex_cli.commands import gallery, settings

    monkeypatch.setenv("MOJILEX_UI_LANGUAGE", "en")
    monkeypatch.setattr(ui, "is_interactive", lambda: True)
    monkeypatch.setattr(ui, "browse_descriptions", lambda *a: pytest.fail("interactive viewer"))
    monkeypatch.setattr(ui, "_settings", lambda *a: pytest.fail("interactive settings"))
    monkeypatch.setattr(packs, "show_pack_command", lambda *a, **kw: view())
    monkeypatch.setattr(settings, "settings_command", lambda: CommandResult(result={"safe": True}))
    browser_calls = []
    monkeypatch.setattr(
        gallery, "gallery_command", lambda *a, **kw: browser_calls.append(kw) or CommandResult()
    )
    arguments = ["mojilex", command]
    if command != "settings":
        arguments.append("NewsEmoji")
    monkeypatch.setattr(cli.sys, "argv", [*arguments, "--json"])
    cli.main()
    output = capsys.readouterr()
    assert json.loads(output.out)["command"] == command
    assert output.err == ""
    assert browser_calls == ([{"open_browser": False}] if command == "gallery" else [])


def test_real_json_boundary_uses_new_command_name(monkeypatch, capsys):
    monkeypatch.setenv("MOJILEX_UI_LANGUAGE", "en")
    monkeypatch.setattr(cli.sys, "argv", ["mojilex", "show", "--json"])
    with pytest.raises(SystemExit):
        cli.main()
    assert json.loads(capsys.readouterr().out)["command"] == "show"


def test_settings_invalid_value_can_be_corrected_without_leaving(navigation, monkeypatch):
    from mojilex_cli.commands import settings
    from mojilex_cli.commands.runtime import CommandError

    navigation(0, None)
    monkeypatch.setattr(
        settings,
        "settings_command",
        lambda: CommandResult(
            result={
                "settings": [
                    {
                        "key": "ai_concurrency",
                        "label": "Parallel",
                        "value": 1,
                        "description": "Concurrent requests",
                        "source": "user",
                        "editable": True,
                        "minimum": 1,
                        "maximum": 16,
                    }
                ],
                "notes": [],
            }
        ),
    )
    prompts = iter(["invalid", "4"])
    monkeypatch.setattr(ui.typer, "prompt", lambda *a, **kw: next(prompts))
    monkeypatch.setattr(ui, "confirm", lambda *a, **kw: True)
    changes = []

    def update(key, value):
        changes.append(value)
        if value == "invalid":
            raise CommandError("CONFIG_INVALID", "Invalid", hint="Enter a number")
        return CommandResult(result={"label": "Parallel", "value": 4, "note": "Saved"})

    monkeypatch.setattr(settings, "update_setting_command", update)
    ui._settings(lambda args: pytest.fail("ran another command"))
    assert changes == ["invalid", "4"]
