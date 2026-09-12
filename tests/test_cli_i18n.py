from __future__ import annotations

import pytest
import typer

from mojilex_cli.cli import app
from mojilex_cli.i18n import (
    extract_ui_language,
    localize_command_tree,
    use_ui_language,
)


def test_russian_help_describes_every_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    command = typer.main.get_command(app)
    localize_command_tree(command, "ru")

    pending = [command]
    while pending:
        current = pending.pop()
        assert current.help
        pending.extend(getattr(current, "commands", {}).values())

    with use_ui_language("ru"):
        with pytest.raises(SystemExit) as captured_exit:
            command.main(
                args=["--help"],
                prog_name="mojilex",
                windows_expand_args=False,
            )

    output = capsys.readouterr().out
    assert captured_exit.value.code == 0
    assert "Команды" in output
    assert "Проверить черновик" in output
    assert "Язык интерфейса" in output
    panel_titles = (
        "Анализ и публикация паков",
        "Поиск и чтение",
        "Проверка и модерация",
        "Снимки и тесты",
        "Настройка и обслуживание",
    )
    assert all(title in output for title in panel_titles)
    assert [output.index(title) for title in panel_titles] == sorted(
        output.index(title) for title in panel_titles
    )
    assert output.index("add") < output.index("import") < output.index("describe")


def test_ui_language_flag_is_global_and_cli_overrides_environment() -> None:
    language, arguments = extract_ui_language(
        ["submit", "mlxrun_test", "--ui-language", "ru", "--publish", "local"],
        environment={"MOJILEX_UI_LANGUAGE": "en"},
    )

    assert language == "ru"
    assert arguments == ["submit", "mlxrun_test", "--publish", "local"]


def test_ui_language_comes_from_environment() -> None:
    language, arguments = extract_ui_language(
        ["doctor"], environment={"MOJILEX_UI_LANGUAGE": "ru"}
    )

    assert language == "ru"
    assert arguments == ["doctor"]
