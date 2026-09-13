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
                args=["--help-all"],
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

    language, arguments = extract_ui_language(
        ["doctor"], environment={}, user_path=config, project_path=tmp_path / "project.toml"
    )

    assert language == "ru"
    assert arguments == ["doctor"]


def test_ui_language_environment_overrides_saved_user_config(tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('ui_language = "ru"\n', encoding="utf-8")

    language, _arguments = extract_ui_language(
        ["doctor"],
        environment={"MOJILEX_UI_LANGUAGE": "en"},
        user_path=config,
        project_path=tmp_path / "project.toml",
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
            project_path=tmp_path / "project.toml",
        )


@pytest.mark.parametrize(
    ("arguments", "environment", "expected"),
    [
        (["list"], {}, "ru"),
        (["list"], {"MOJILEX_UI_LANGUAGE": "en"}, "en"),
        (["list", "--ui-language", "ru"], {"MOJILEX_UI_LANGUAGE": "en"}, "ru"),
    ],
)
def test_project_language_obeys_complete_precedence(
    tmp_path,
    arguments,
    environment,
    expected,
) -> None:
    user = tmp_path / "user.toml"
    project = tmp_path / "project.toml"
    user.write_text('ui_language="en"\n')
    project.write_text('ui_language="ru"\n')
    language, cleaned = extract_ui_language(
        arguments,
        environment=environment,
        user_path=user,
        project_path=project,
    )
    assert language == expected
    assert cleaned == ["list"]


def test_language_discovers_project_in_current_directory(tmp_path, monkeypatch) -> None:
    user = tmp_path / "user.toml"
    user.write_text('ui_language="en"\n')
    (tmp_path / ".mojilex.toml").write_text('ui_language="ru"\n')
    monkeypatch.chdir(tmp_path)
    language, _ = extract_ui_language(["list"], environment={}, user_path=user)
    assert language == "ru"


@pytest.mark.parametrize("content", ["", '[ai]\nmodel="test"\n', "[invalid"])
def test_project_without_readable_language_falls_back_to_user(tmp_path, content) -> None:
    user = tmp_path / "user.toml"
    project = tmp_path / "project.toml"
    user.write_text('ui_language="ru"\n')
    project.write_text(content)
    language, _ = extract_ui_language(
        ["list"],
        environment={},
        user_path=user,
        project_path=project,
    )
    assert language == "ru"


def test_project_language_edit_takes_effect_on_next_command(tmp_path) -> None:
    from mojilex_cli.commands.settings import update_setting_command

    user = tmp_path / "user.toml"
    project = tmp_path / "project.toml"
    user.write_text('ui_language="en"\n')
    project.write_text('ui_language="en"\n')
    update_setting_command(
        "ui_language",
        "ru",
        environment={},
        user_path=user,
        project_path=project,
    )
    language, _ = extract_ui_language(
        ["list"],
        environment={},
        user_path=user,
        project_path=project,
    )
    assert language == "ru"
    assert user.read_text() == 'ui_language="en"\n'


def test_explicit_project_path_does_not_read_real_project(tmp_path, monkeypatch) -> None:
    (tmp_path / ".mojilex.toml").write_text('ui_language="ru"\n')
    monkeypatch.chdir(tmp_path)
    language, _ = extract_ui_language(
        ["list"],
        environment={},
        user_path=tmp_path / "missing-user.toml",
        project_path=tmp_path / "missing-project.toml",
    )
    assert language == "en"


def test_invalid_project_language_is_rejected_without_echoing_input(tmp_path) -> None:
    project = tmp_path / "project.toml"
    project.write_text('ui_language="private-invalid-language"\n')
    with pytest.raises(ValueError, match="must be en or ru") as error:
        extract_ui_language(
            ["list"],
            environment={},
            user_path=tmp_path / "missing-user.toml",
            project_path=project,
        )
    assert "private-invalid-language" not in str(error.value)
