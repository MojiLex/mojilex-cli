from io import StringIO

import pytest
from typer.testing import CliRunner

from mojilex_cli import cli
from mojilex_cli.commands import runtime, workflow
from mojilex_cli.commands.runtime import CommandError, CommandResult


def test_file_path_bom_comments_duplicates(tmp_path):
    path = tmp_path / "my links.txt"
    path.write_text(
        "# packs\nhttps://t.me/addemoji/First\n\nhttps://t.me/addemoji/Second\nhttps://t.me/addemoji/First\n",
        encoding="utf-8-sig",
    )
    assert workflow.collect_sources(
        [f'"{path}"'], from_file=None, use_stdin=False, stream=StringIO()
    ) == (
        "https://t.me/addemoji/First",
        "https://t.me/addemoji/Second",
    )


@pytest.mark.parametrize(
    "content", [b"", b"\xff\xfe", b"x" * (1024 * 1024 + 1)], ids=["empty", "encoding", "oversize"]
)
def test_invalid_file_rejected_before_import(tmp_path, content):
    path = tmp_path / "links.txt"
    path.write_bytes(content)
    with pytest.raises(CommandError):
        workflow.collect_sources([str(path)], from_file=None, use_stdin=False, stream=StringIO())


@pytest.mark.parametrize("explicit", [False, True])
def test_import_passes_whole_file_to_one_run(tmp_path, monkeypatch, explicit):
    path = tmp_path / "links.txt"
    path.write_text("https://t.me/addemoji/First\nhttps://t.me/addemoji/Second")
    seen = []
    monkeypatch.setattr(cli, "_with_runtime_secrets", lambda action, **kw: action())
    monkeypatch.setattr(cli, "_pack_action", lambda action, *a: action())
    monkeypatch.setattr(
        workflow, "import_command", lambda sources, **kw: seen.append(sources) or CommandResult()
    )
    args = ["import", *(["--from-file"] if explicit else []), str(path), "--json"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert seen == [("https://t.me/addemoji/First", "https://t.me/addemoji/Second")]


def test_ai_and_publish_default_yes_but_other_confirmation_default_no(monkeypatch):
    defaults = []
    monkeypatch.setattr(
        runtime, "ui_confirm", lambda text, **kw: defaults.append(kw["default"]) or True
    )
    cli._confirmation_callback(yes=False, non_interactive=False, json_output=False, default=True)(
        "Publish?"
    )
    cli._unknown_cost_callback(yes=False, non_interactive=False, json_output=False)(20)
    runtime.require_confirmation("Delete?", yes=False, non_interactive=False, json_output=False)
    assert defaults == [True, True, False]


def test_default_yes_does_not_authorize_machine_mode():
    with pytest.raises(CommandError):
        cli._confirmation_callback(yes=False, non_interactive=True, json_output=False)("Publish?")


@pytest.mark.parametrize("yes", [False, True])
def test_sync_cli_machine_mode_requires_single_explicit_approval(monkeypatch, yes):
    from mojilex_cli.pipeline import batch

    published = []

    def sync(**kwargs):
        kwargs["confirmation"]("Send two packs to owner/repo?")
        published.append(True)
        return CommandResult(result={"added_packs": ["first", "second"]})

    monkeypatch.setattr(batch, "sync_packs_command", sync)
    result = CliRunner().invoke(cli.app, ["sync", "--json", *(["--yes"] if yes else [])])
    assert (result.exit_code == 0) == yes
    assert published == ([True] if yes else [])
