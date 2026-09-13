from __future__ import annotations

import re

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
    commands = re.findall(r"(?m)^\s*│\s+(list|show|import|describe|publish)\s", output)
    assert commands == ["list", "show", "import", "describe", "publish"]


def test_ui_language_flag_is_global_and_cli_overrides_environment() -> None:
    language, arguments = extract_ui_language(
        ["submit", "mlxrun_test", "--ui-language", "ru", "--publish", "local"],
        environment={"MOJILEX_UI_LANGUAGE": "en"},
    )

    assert language == "ru"
    assert arguments == ["submit", "mlxrun_test", "--publish", "local"]


def test_ui_language_comes_from_environment() -> None:
    language, arguments = extract_ui_language(["doctor"], environment={"MOJILEX_UI_LANGUAGE": "ru"})

    assert language == "ru"
    assert arguments == ["doctor"]


def test_ui_language_comes_from_saved_user_config(tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('ui_language = "ru"\n', encoding="utf-8")

    language, arguments = extract_ui_language(["doctor"], environment={}, user_path=config)

    assert language == "ru"
    assert arguments == ["doctor"]


def test_ui_language_environment_overrides_saved_user_config(tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('ui_language = "ru"\n', encoding="utf-8")

    language, _arguments = extract_ui_language(
        ["doctor"], environment={"MOJILEX_UI_LANGUAGE": "en"}, user_path=config
    )

    assert language == "en"


def test_explicit_empty_ui_language_is_rejected_instead_of_falling_back(tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('ui_language = "ru"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must be en or ru"):
        extract_ui_language(
            ["doctor", "--ui-language="],
            environment={"MOJILEX_UI_LANGUAGE": "ru"},
            user_path=config,
        )
