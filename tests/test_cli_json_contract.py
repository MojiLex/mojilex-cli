import json
import sys

import pytest
import typer

from mojilex_cli import cli
from mojilex_cli.commands.runtime import CommandResult, execute


@pytest.mark.parametrize(
    "arguments",
    [
        ["--json", "takedown", "mxe_test"],
        ["takedown", "--json", "mxe_test"],
        ["takedown", "mxe_test", "--json"],
        ["--json", "not-a-command"],
        ["add", "https://t.me/addemoji/Pack", "--max-items", "nope", "--json"],
        ["--json", "--help"],
    ],
)
def test_usage_errors_with_json_emit_exactly_one_envelope(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    arguments: list[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["mojilex", *arguments])

    with pytest.raises(SystemExit) as captured_exit:
        cli.main()

    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert captured_exit.value.code == 2
    assert captured.err == ""
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["ok"] is False
    assert payload["status"] == "failed"
    assert payload["errors"][0]["code"] == "CONFIG_INVALID"


def test_nested_command_accepts_json_between_group_and_leaf(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["mojilex", "config", "--json", "show"])
    monkeypatch.setattr(
        cli,
        "config_show_command",
        lambda: CommandResult(result={"mode": "machine"}),
    )

    cli.main()

    captured = capsys.readouterr()
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["ok"] is True
    assert payload["command"] == "config show"
    assert payload["result"] == {"mode": "machine"}


def test_abort_is_interrupted_with_exit_130_and_json_envelope(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def abort() -> CommandResult:
        raise typer.Abort()

    with pytest.raises(typer.Exit) as captured_exit:
        execute("takedown", abort, json_output=True)

    captured = capsys.readouterr()
    assert captured_exit.value.exit_code == 130
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "interrupted"
    assert payload["errors"][0]["code"] == "INTERRUPTED"


def test_debug_traceback_and_human_error_are_sanitized(
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "opaque-client-secret"

    def fail() -> CommandResult:
        raise RuntimeError(f"client_secret={secret} Bearer {secret}")

    with pytest.raises(typer.Exit) as captured_exit:
        execute("doctor", fail, json_output=False, debug=True)

    captured = capsys.readouterr()
    rendered = captured.out + captured.err
    assert captured_exit.value.exit_code == 1
    assert secret not in rendered
    assert "[REDACTED]" in rendered
