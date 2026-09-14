from pathlib import Path

import pytest

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.commands.settings import settings_command, update_setting_command
from mojilex_cli.config import load_config


def paths(tmp_path: Path) -> dict[str, Path]:
    return {"project_path": tmp_path / "project.toml", "user_path": tmp_path / "user.toml"}


def test_settings_show_effective_sources_without_credentials(tmp_path: Path) -> None:
    files = paths(tmp_path)
    files["user_path"].write_text('[ai]\nmodel="saved-model"\nai_concurrency=2\n')
    files["project_path"].write_text("[ai]\nai_concurrency=4\n")
    result = settings_command(
        **files, environment={"MOJILEX_MAX_AI_REQUESTS": "12", "GEMINI_API_KEY": "private-key"}
    ).result
    rows = {row["key"]: row for row in result["settings"]}
    assert rows["model"]["value"] == "saved-model"
    assert rows["model"]["source"] == "user"
    assert rows["ai_concurrency"]["value"] == 4
    assert rows["ai_concurrency"]["source"] == "project"
    assert rows["max_ai_requests"]["value"] == 12
    assert rows["max_ai_requests"]["source"] == "environment"
    assert rows["max_ai_requests"]["editable"] is False
    assert rows["download_attempts"]["source"] == "default"
    assert "private-key" not in str(result)


def test_edit_project_value_preserves_other_settings_and_comments(tmp_path: Path) -> None:
    files = paths(tmp_path)
    user = b"# User file\n[ai]\nai_concurrency = 2\nmax_ai_requests = 100\n"
    project = (
        b'# Project\r\n[ai]\r\nai_concurrency = 4 # speed\r\nmodel = "my#model"\r\n'
        b"\r\n[telegram]\r\nmax_attempts = 7 # keep\r\n"
    )
    files["user_path"].write_bytes(user)
    files["project_path"].write_bytes(project)
    result = update_setting_command("ai_concurrency", "8", **files, environment={}).result
    assert result["value"] == 8 and result["scope"] == "project"
    assert files["user_path"].read_bytes() == user
    assert files["project_path"].read_bytes() == project.replace(b"= 4 # speed", b"= 8 # speed")
    assert load_config(**files, environment={}).ai.max_ai_requests == 100


def test_edit_user_does_not_copy_environment_or_project_values(tmp_path: Path) -> None:
    files = paths(tmp_path)
    files["project_path"].write_text('[ai]\nmodel="project-model"\n')
    update_setting_command(
        "download_attempts", "8", **files, environment={"MOJILEX_MAX_AI_REQUESTS": "12"}
    )
    assert files["user_path"].read_text() == "\n[telegram]\nmax_attempts = 8\n"
    assert "12" not in files["user_path"].read_text()
    assert load_config(**files, environment={}).ai.max_ai_requests == 100


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("ai_concurrency", "0"),
        ("ai_concurrency", "17"),
        ("ai_concurrency", "oops"),
        ("download_attempts", "9"),
        ("max_ai_requests", "-1"),
        ("max_cost_usd", "NaN"),
        ("max_cost_usd", "-1"),
        ("timeout_seconds", "0"),
        ("provider", "unsupported"),
        ("ui_language", "xx"),
        ("model", ""),
        ("model", "github_pat_abcdefghijklmnopqrstuvwxyz123456"),
        ("GEMINI_API_KEY", "private-key"),
    ],
)
def test_invalid_setting_never_changes_or_creates_files(
    tmp_path: Path, key: str, value: str
) -> None:
    files = paths(tmp_path)
    files["user_path"].write_bytes(b"# Keep this\n")
    with pytest.raises(CommandError):
        update_setting_command(key, value, **files, environment={})
    assert files["user_path"].read_bytes() == b"# Keep this\n"
    assert not files["project_path"].exists()
    assert not list(tmp_path.glob(".mojilex-settings-*"))


def test_environment_override_rejects_before_writing(tmp_path: Path) -> None:
    files = paths(tmp_path)
    with pytest.raises(CommandError) as failure:
        update_setting_command(
            "model", "new-model", **files, environment={"MOJILEX_MODEL": "override-model"}
        )
    assert "MOJILEX_MODEL" in failure.value.error.hint
    assert "override-model" not in str(failure.value)
    assert not files["user_path"].exists()


@pytest.mark.parametrize(
    "content",
    [
        'ai.model = "old-model"\n',
        'ai = { model = "old-model" }\n',
        '[ai]\nmodel = """old-model"""\n',
    ],
)
def test_complex_existing_toml_is_preserved_when_edit_is_not_supported(
    tmp_path: Path,
    content: str,
) -> None:
    files = paths(tmp_path)
    files["user_path"].write_text(content)
    with pytest.raises(CommandError):
        update_setting_command("model", "new-model", **files, environment={})
    assert files["user_path"].read_text() == content


def test_root_and_missing_section_edits_preserve_structure(tmp_path: Path) -> None:
    files = paths(tmp_path)
    files["user_path"].write_text('# top\n[ai]\nmodel="keep"\n')
    update_setting_command("ui_language", "ru", **files, environment={})
    update_setting_command("download_concurrency", "12", **files, environment={})
    update_setting_command("max_cost_usd", "1.25", **files, environment={})
    loaded = load_config(**files, environment={})
    assert loaded.ui_language == "ru"
    assert loaded.telegram.download_concurrency == 12
    assert str(loaded.ai.max_cost_usd) == "1.25"
    assert loaded.ai.model == "keep"
    assert "# top" in files["user_path"].read_text()


def test_config_race_is_not_overwritten(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import mojilex_cli.commands.settings as module

    files = paths(tmp_path)
    files["user_path"].write_text("[ai]\nai_concurrency=2\n")
    original = module.tempfile.mkstemp

    def race(**kwargs: object) -> tuple[int, str]:
        files["user_path"].write_text("[ai]\nai_concurrency=3\n")
        return original(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(module.tempfile, "mkstemp", race)
    with pytest.raises(CommandError, match="changed while editing"):
        update_setting_command("ai_concurrency", "4", **files, environment={})
    assert files["user_path"].read_text() == "[ai]\nai_concurrency=3\n"
    assert not list(tmp_path.glob(".mojilex-settings-*"))


@pytest.mark.parametrize("value", ["none", "", "None"])
def test_clear_cost_limit_preserves_comments_and_other_limits(tmp_path: Path, value: str) -> None:
    files = paths(tmp_path)
    files["user_path"].write_text("[ai]\nmax_cost_usd=2.5 # my limit\nmax_ai_requests=40\n")
    result = update_setting_command("max_cost_usd", value, **files, environment={}).result
    assert result["value"] is None
    assert files["user_path"].read_text() == "[ai]\n# my limit\nmax_ai_requests=40\n"
    assert load_config(**files, environment={}).ai.max_ai_requests == 40


def test_clear_absent_cost_limit_does_not_create_file(tmp_path: Path) -> None:
    files = paths(tmp_path)
    result = update_setting_command("max_cost_usd", "none", **files, environment={}).result
    assert result["value"] is None
    assert not files["user_path"].exists()


def test_clear_project_cost_reports_inherited_user_limit(tmp_path: Path) -> None:
    files = paths(tmp_path)
    files["project_path"].write_text("[ai]\nmax_cost_usd=2.5\n")
    user = "[ai]\nmax_cost_usd=10\n"
    files["user_path"].write_text(user)
    result = update_setting_command("max_cost_usd", "none", **files, environment={}).result
    assert result["value"] == "10"
    assert files["user_path"].read_text() == user


def test_russian_settings_are_localized(tmp_path: Path) -> None:
    from mojilex_cli.i18n import use_ui_language

    with use_ui_language("ru"):
        result = settings_command(**paths(tmp_path), environment={}).result
        saved = update_setting_command(
            "ai_concurrency", "4", **paths(tmp_path), environment={}
        ).result
    provider = next(row for row in result["settings"] if row["key"] == "provider")
    assert provider["label"] == "Сервис ИИ"
    assert "Приоритет" in result["notes"][0]
    assert "Сохранено" in saved["note"]


def test_analysis_confirmation_setting_round_trip_preserves_budget(tmp_path):
    from mojilex_cli.commands.system import _config_toml
    from mojilex_cli.config import load_config

    files = paths(tmp_path)
    files["user_path"].write_text('[ai]\nmax_ai_requests="unlimited"\nmax_cost_usd=2.5\n')
    for raw, expected in (("true", True), ("false", False)):
        update_setting_command("confirm_before_analysis", raw, **files, environment={})
        configured = load_config(**files, environment={})
        assert configured.ai.confirm_before_analysis is expected
        assert configured.ai.max_ai_requests is None
        assert str(configured.ai.max_cost_usd) == "2.5"
        assert f"confirm_before_analysis = {raw}" in _config_toml(configured)
    with pytest.raises(CommandError):
        update_setting_command("confirm_before_analysis", "maybe", **files, environment={})
