"""Legacy pack concurrency stays compatible while pack execution remains sequential."""

from pathlib import Path

import pytest

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.commands.settings import settings_command, update_setting_command
from mojilex_cli.commands.system import _config_toml
from mojilex_cli.config import ConfigError, MojiLexConfig, load_config
from mojilex_cli.config.models import AIConfig, ProcessingConfig, TelegramConfig
from mojilex_cli.i18n import normalize_ui_language, use_ui_language


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {"project_path": tmp_path / "project.toml", "user_path": tmp_path / "user.toml"}


def test_legacy_config_gets_pack_default_without_changing_resource_limits(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths["user_path"].write_text(
        "[telegram]\ndownload_concurrency=15\n"
        "[processing]\nrender_concurrency=4\nrender_timeout_seconds=15\n"
        "max_temp_bytes=123456\n[ai]\nai_concurrency=2\nmax_ai_requests=43\n"
        "max_cost_usd=1.25\n",
        encoding="utf-8",
    )
    before = paths["user_path"].read_bytes()
    config = load_config(**paths, environment={})
    assert config.processing.pack_concurrency == 3
    assert config.telegram.download_concurrency == 15
    assert config.processing.render_concurrency == 4
    assert config.processing.render_timeout_seconds == 15
    assert config.processing.max_temp_bytes == 123456
    assert config.ai.ai_concurrency == 2
    assert config.ai.max_ai_requests == 43
    assert str(config.ai.max_cost_usd) == "1.25"
    assert paths["user_path"].read_bytes() == before


@pytest.mark.parametrize("value", [1, 8])
def test_pack_concurrency_boundary_values_roundtrip_init_config(tmp_path: Path, value: int) -> None:
    config = MojiLexConfig(
        processing=ProcessingConfig(pack_concurrency=value, render_concurrency=4),
        telegram=TelegramConfig(download_concurrency=15),
        ai=AIConfig(ai_concurrency=2, max_ai_requests=43),
    )
    paths = _paths(tmp_path)
    paths["user_path"].write_text(_config_toml(config), encoding="utf-8")
    loaded = load_config(**paths, environment={})
    assert loaded.processing == config.processing
    assert loaded.telegram == config.telegram
    assert loaded.ai == config.ai


@pytest.mark.parametrize("value", ["0", "9", "-1", "1.5", "invalid"])
def test_invalid_environment_pack_limit_fails_closed(tmp_path: Path, value: str) -> None:
    with pytest.raises(ConfigError):
        load_config(**_paths(tmp_path), environment={"MOJILEX_PACK_CONCURRENCY": value})


def test_pack_concurrency_layer_precedence_and_environment_edit_protection(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths["user_path"].write_text("[processing]\npack_concurrency=2\n", encoding="utf-8")
    paths["project_path"].write_text("[processing]\npack_concurrency=4\n", encoding="utf-8")
    env = {"MOJILEX_PACK_CONCURRENCY": "5"}
    assert load_config(**paths, environment={}).processing.pack_concurrency == 4
    assert load_config(**paths, environment=env).processing.pack_concurrency == 5
    assert (
        load_config(
            **paths, environment=env, cli={"processing": {"pack_concurrency": 1}}
        ).processing.pack_concurrency
        == 1
    )
    before = {key: path.read_bytes() for key, path in paths.items()}
    with pytest.raises(CommandError):
        update_setting_command("pack_concurrency", "6", **paths, environment=env)
    assert {key: path.read_bytes() for key, path in paths.items()} == before


@pytest.mark.parametrize("value", ["0", "9", "invalid"])
def test_invalid_setting_preserves_existing_file(tmp_path: Path, value: str) -> None:
    paths = _paths(tmp_path)
    before = b"# Keep limits\n[processing]\nrender_concurrency=4\n"
    paths["user_path"].write_bytes(before)
    with pytest.raises(CommandError):
        update_setting_command("pack_concurrency", value, **paths, environment={})
    assert paths["user_path"].read_bytes() == before
    assert not paths["project_path"].exists()


def test_pack_setting_edit_changes_only_its_project_value(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    user = b"[ai]\nmax_ai_requests=123\nai_concurrency=2\n"
    project = (
        b"# My machine\r\n[processing]\r\npack_concurrency=3 # overlap\r\n"
        b"render_concurrency=4\r\nrender_timeout_seconds=15\r\n"
        b"[telegram]\r\ndownload_concurrency=15\r\n"
    )
    paths["user_path"].write_bytes(user)
    paths["project_path"].write_bytes(project)
    update_setting_command("pack_concurrency", "1", **paths, environment={})
    assert paths["user_path"].read_bytes() == user
    assert paths["project_path"].read_bytes() == project.replace(
        b"pack_concurrency=3", b"pack_concurrency=1"
    )


@pytest.mark.parametrize(
    ("language", "expected_label", "expected_fragments"),
    [
        (
            "en",
            "Legacy pack concurrency",
            ("compatibility", "sequentially", "within the current pack"),
        ),
        (
            "ru",
            "Прежняя параллельность паков",
            ("совместимости", "последовательно", "внутри текущего пака"),
        ),
    ],
)
def test_pack_setting_is_visible_with_bounds_and_localized_description(
    tmp_path: Path,
    language: str,
    expected_label: str,
    expected_fragments: tuple[str, ...],
) -> None:
    with use_ui_language(normalize_ui_language(language)):
        result = settings_command(**_paths(tmp_path), environment={}).result
    rows = {row["key"]: row for row in result["settings"]}
    row = rows["pack_concurrency"]
    assert row["value"] == 3
    assert row["minimum"] == 1 and row["maximum"] == 8
    assert row["editable"] and row["source"] == "default"
    assert row["label"] == expected_label
    assert all(fragment in row["description"] for fragment in expected_fragments)
