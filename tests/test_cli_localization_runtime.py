from __future__ import annotations

# ruff: noqa: RUF001
import json
import sys

import pytest
import typer
from typer.testing import CliRunner

from mojilex_cli.ai.base import UnknownCostError
from mojilex_cli.cli import main
from mojilex_cli.commands.runtime import CommandResult, execute
from mojilex_cli.i18n import confirm, text, use_ui_language


@pytest.mark.parametrize(
    "message",
    [
        "Could not inspect the configured dataset repository.",
        "Check Git and access to the configured local repository.",
        "Could not create an isolated dataset checkout.",
        "Check Git, repository access, and the configured base branch.",
        "eligible emoji mxe_test has incomplete concept mapping; "
        "complete concept mapping before building a release snapshot",
    ],
)
def test_workspace_and_pending_snapshot_messages_are_localized(message: str) -> None:
    with use_ui_language("ru"):
        assert text(message) != message
    with use_ui_language("en"):
        assert text(message) == message


@pytest.mark.parametrize(
    ("arguments", "translated", "original"),
    [
        (["resume"], "Не указан обязательный аргумент", "Missing argument"),
        (["resolve"], "Не указан обязательный параметр", "Missing option"),
        (["import", "--download-concurrenc", "4"], "Возможные параметры", "Possible options"),
        (["import", "--garbage"], "Неизвестный параметр", "No such option"),
        (["import", "--download-concurrency", "nope"], "не целое число", "is not a valid"),
        (
            ["import", "--download-concurrency", "0"],
            "вне допустимого диапазона",
            "is not in the range",
        ),
        (["import", "--download-concurrency"], "нужно значение", "requires an argument"),
    ],
)
def test_russian_parser_errors_are_display_only(
    arguments: list[str],
    translated: str,
    original: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["mojilex", "--ui-language", "ru", *arguments])
    with pytest.raises(SystemExit) as failure:
        main()
    output = capsys.readouterr().err
    assert failure.value.code == 2
    if arguments[-1] != "--download-concurrency":
        assert "Использование:" in output
    assert "Usage:" not in output
    assert translated in output
    assert original not in output


def test_russian_parser_json_keeps_english_machine_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["mojilex", "--ui-language", "ru", "resume", "--json"])
    with pytest.raises(SystemExit) as failure:
        main()
    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert failure.value.code == 2
    assert payload["errors"][0]["code"] == "CONFIG_INVALID"
    assert "Missing parameter" in payload["errors"][0]["message"]
    assert not output.err


@pytest.mark.parametrize("json_output", [False, True])
def test_unknown_cost_translates_only_human_output(
    json_output: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    message = "provider cost is unknown; explicit approval is required"

    def fail() -> None:
        raise UnknownCostError(message)

    with use_ui_language("ru"), pytest.raises(typer.Exit):
        execute("describe", fail, json_output=json_output)
    output = capsys.readouterr()
    if json_output:
        payload = json.loads(output.out)
        assert payload["errors"][0]["code"] == "UNKNOWN_COST"
        assert payload["errors"][0]["message"] == message
        assert payload["errors"][0]["hint"] == "Correct the reported condition and retry."
    else:
        assert "Стоимость запросов к провайдеру неизвестна" in output.err
        assert "Устраните указанную причину" in output.err
        assert message not in output.err


@pytest.mark.parametrize(
    "arguments",
    [["snapshots", "--channel", "stable"], ["snapshot", "pull"], ["snapshot", "update"]],
)
def test_snapshot_unavailability_is_russian(
    arguments: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["mojilex", "--ui-language", "ru", *arguments])
    with pytest.raises(SystemExit) as failure:
        main()
    output = capsys.readouterr().err
    assert failure.value.code == 6
    assert "MIRROR_UNAVAILABLE" in output
    assert "снимк" in output or "каталог" in output
    assert "Подсказка:" in output
    assert "not configured" not in output


@pytest.mark.parametrize(
    ("answer", "expected"),
    [("да\n", True), ("Y\n", True), ("нет\n", False), ("n\n", False), ("\n", False)],
)
def test_russian_confirmation_accepts_both_languages_and_defaults_to_no(
    answer: str,
    expected: bool,
) -> None:
    app = typer.Typer()

    @app.command()
    def ask() -> None:
        typer.echo(str(confirm("Подтвердить?")))

    with use_ui_language("ru"):
        result = CliRunner().invoke(app, input=answer)
    assert result.exit_code == 0
    assert "[да/Нет]" in result.output
    assert result.output.rstrip().endswith(str(expected))


def test_russian_confirmation_reprompts_invalid_input() -> None:
    app = typer.Typer()

    @app.command()
    def ask() -> None:
        typer.echo(str(confirm("Подтвердить?")))

    with use_ui_language("ru"):
        result = CliRunner().invoke(app, input="perhaps\nнет\n")
    assert result.exit_code == 0
    assert "Введите да/нет или y/n" in result.output
    assert result.output.rstrip().endswith("False")


def test_dynamic_translations_preserve_counts_identifiers_and_english() -> None:
    source = "Checking source 2/3: Pack_Name"
    approval = (
        "Authorize up to 7 additional AI requests for this run, including retries? "
        "The USD cost is unknown. This is a one-time approval for this invocation."
    )
    with use_ui_language("ru"):
        assert text(source) == "Проверка источника 2/3: Pack_Name"
        assert "до 7 дополнительных AI-запросов" in text(approval)
        assert "30 с." in text("Gemini request timed out after 30 seconds.")
        assert "7.5 с." in text("Gemini model check timed out after 7.5 seconds.")
    with use_ui_language("en"):
        assert text(source) == source
        assert text(approval) == approval


@pytest.mark.parametrize(
    "message",
    [
        "Diagnostic read from an integrity-checked unsigned snapshot; "
        "safe_eligible is always false.",
        "The local snapshot has integrity checks but no enforceable release signature.",
        "Rerun with --allow-unverified only for diagnostic use of this exact local snapshot.",
    ],
)
def test_snapshot_trust_diagnostics_are_translated_without_changing_english(message: str) -> None:
    with use_ui_language("ru"):
        assert "снимк" in text(message)
        assert text(message) != message
    with use_ui_language("en"):
        assert text(message) == message


@pytest.mark.parametrize("json_output", [False, True])
def test_human_warning_mapping_and_boolean_do_not_change_machine_values(
    json_output: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    message = (
        "Diagnostic read from an integrity-checked unsigned snapshot; "
        "safe_eligible is always false."
    )
    with use_ui_language("ru"):
        execute(
            "doctor",
            lambda: CommandResult(
                result={"ready": True},
                warnings=[{"code": "INTEGRITY_ONLY_UNSIGNED", "message": message}],
            ),
            json_output=json_output,
        )
    output = capsys.readouterr().out
    if json_output:
        payload = json.loads(output)
        assert payload["result"]["ready"] is True
        assert payload["warnings"][0]["message"] == message
    else:
        assert "Да" in output
        assert "Диагностическое чтение снимка" in output
        assert "INTEGRITY_ONLY_UNSIGNED" in output
        assert message not in output
