from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console
from typer.testing import CliRunner

from mojilex_cli import cli
from mojilex_cli.commands import interactive, packs, settings, workflow
from mojilex_cli.commands.runtime import CommandError, CommandResult, _render_pack_result
from mojilex_cli.commands.system import _config_toml
from mojilex_cli.config import AIConfig, ConfigError, MojiLexConfig, load_config
from mojilex_cli.i18n import text, use_ui_language


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {"user_path": tmp_path / "user.toml", "project_path": tmp_path / "project.toml"}


@pytest.mark.parametrize("raw", ["unlimited", "UNLIMITED", " unlimited "])
def test_unlimited_config_and_environment_roundtrip_preserve_explicit_disable(tmp_path, raw):
    paths = _paths(tmp_path)
    paths["user_path"].write_text(f"[ai]\nmax_ai_requests={json.dumps(raw)}\n")
    config = load_config(**paths, environment={})
    assert config.ai.max_ai_requests is None
    paths["user_path"].write_text(_config_toml(config))
    assert 'max_ai_requests = "unlimited"' in paths["user_path"].read_text()
    assert load_config(**paths, environment={}).ai.max_ai_requests is None
    assert (
        load_config(**paths, environment={"MOJILEX_MAX_AI_REQUESTS": "0"}).ai.max_ai_requests == 0
    )
    assert (
        load_config(**paths, environment={"MOJILEX_MAX_AI_REQUESTS": raw}).ai.max_ai_requests
        is None
    )
    assert AIConfig().max_ai_requests == 100


@pytest.mark.parametrize("raw", ["unlimited", "без лимита"])
def test_edit_unlimited_preserves_other_config_and_zero_remains_no_requests(tmp_path, raw):
    paths = _paths(tmp_path)
    before = (
        b"# Keep\r\n[ai]\r\nmax_ai_requests=23 # budget\r\n"
        b"max_cost_usd=1.25\r\nai_concurrency=4\r\n"
    )
    paths["user_path"].write_bytes(before)
    result = settings.update_setting_command("max_ai_requests", raw, **paths, environment={})
    assert result.result["value"] is None
    assert paths["user_path"].read_bytes() == before.replace(
        b"max_ai_requests=23", b'max_ai_requests="unlimited"'
    )
    settings.update_setting_command("max_ai_requests", "0", **paths, environment={})
    config = load_config(**paths, environment={})
    assert config.ai.max_ai_requests == 0
    assert str(config.ai.max_cost_usd) == "1.25"
    assert config.ai.ai_concurrency == 4


@pytest.mark.parametrize("raw", ["-1", "none", "1.5", "", "infinite"])
def test_invalid_request_setting_fails_without_modifying_config(tmp_path, raw):
    paths = _paths(tmp_path)
    before = b'[ai]\nmax_ai_requests="unlimited"\n'
    paths["user_path"].write_bytes(before)
    with pytest.raises(CommandError):
        settings.update_setting_command("max_ai_requests", raw, **paths, environment={})
    assert paths["user_path"].read_bytes() == before
    with pytest.raises(ConfigError):
        load_config(**paths, environment={"MOJILEX_MAX_AI_REQUESTS": raw})


def test_unlimited_setting_uses_project_layer_and_honors_environment_override(tmp_path):
    paths = _paths(tmp_path)
    paths["user_path"].write_text("[ai]\nmax_ai_requests=17\n")
    paths["project_path"].write_text('[ai]\nmax_ai_requests="unlimited"\n')
    result = settings.settings_command(**paths, environment={}).result
    row = next(row for row in result["settings"] if row["key"] == "max_ai_requests")
    assert row["source"] == "project" and row["value"] is None
    settings.update_setting_command("max_ai_requests", "35", **paths, environment={})
    assert load_config(**paths, environment={}).ai.max_ai_requests == 35
    assert paths["user_path"].read_text() == "[ai]\nmax_ai_requests=17\n"
    with pytest.raises(CommandError):
        settings.update_setting_command(
            "max_ai_requests", "unlimited", **paths, environment={"MOJILEX_MAX_AI_REQUESTS": "5"}
        )


@pytest.mark.parametrize("command", ["add", "describe", "resume"])
@pytest.mark.parametrize("raw, expected", [("unlimited", "unlimited"), ("0", 0), ("27", 27)])
def test_cli_request_limit_reaches_workflow_without_losing_zero_or_unlimited(
    monkeypatch, command, raw, expected
):
    captured = []

    def fake(*_args, **kwargs):
        captured.append(kwargs)
        return CommandResult()

    monkeypatch.setattr(workflow, f"{command}_command", fake)
    monkeypatch.setattr(cli, "_with_runtime_secrets", lambda action, **_kwargs: action())
    monkeypatch.setattr(cli, "_pack_action", lambda action, *_args: action())
    monkeypatch.setattr(
        packs,
        "resolve_pack_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            run_id="saved",
            status="interrupted",
            command="add",
            safe_parameters={"sources": ["https://t.me/addemoji/SamplePack"]},
        ),
    )
    selector = "https://t.me/addemoji/SamplePack" if command == "add" else "saved"
    result = CliRunner().invoke(cli.app, [command, selector, "--max-ai-requests", raw, "--json"])
    assert result.exit_code == 0, result.output
    assert captured[0]["max_ai_requests"] == expected


@pytest.mark.parametrize("command", ["add", "describe", "resume"])
def test_invalid_cli_budget_rejected_before_credentials_or_work(monkeypatch, command):
    monkeypatch.setattr(cli, "_with_runtime_secrets", lambda *_a, **_k: pytest.fail("credentials"))
    result = CliRunner().invoke(cli.app, [command, "saved", "--max-ai-requests", "-1"])
    assert result.exit_code == 2
    assert "unlimited" in result.output


def test_settings_order_keeps_parallel_limits_together_and_defines_official_database(tmp_path):
    with use_ui_language("ru"):
        rows = settings.settings_command(**_paths(tmp_path), environment={}).result["settings"]
    keys = [row["key"] for row in rows]
    assert keys[:8] == [
        "ui_language",
        "official_pack_policy",
        "file_analysis_mode",
        "provider",
        "model",
        "confirm_before_analysis",
        "max_ai_requests",
        "max_cost_usd",
    ]
    assert keys[8:12] == [
        "pack_concurrency",
        "download_concurrency",
        "render_concurrency",
        "ai_concurrency",
    ]
    official = rows[1]["description"]
    assert (
        "GitHub" in official and "локального кэша" in official and "Enter означает Нет" in official
    )
    budget = rows[6]["description"]
    assert "всех паков файла" in budget and "пазлов" in budget and "unlimited" in budget


@pytest.mark.parametrize("language, expected", [("ru", "Без лимита"), ("en", "Unlimited")])
def test_unlimited_is_readable_in_menu_and_cli_table(tmp_path, language, expected):
    paths = _paths(tmp_path)
    paths["user_path"].write_text(_config_toml(MojiLexConfig(ai=AIConfig(max_ai_requests=None))))
    output = StringIO()
    with use_ui_language(language):
        assert interactive._setting_value(None, key="max_ai_requests") == expected
        result = settings.settings_command(**paths, environment={}).result
        assert _render_pack_result(Console(file=output, width=200), "settings", result)
        updated = settings.update_setting_command(
            "max_ai_requests", "unlimited", **paths, environment={}
        )
        assert _render_pack_result(Console(file=output, width=200), "settings", updated.result)
    assert output.getvalue().count(expected) == 2
    assert "None" not in output.getvalue()


def test_unlimited_prompt_default_does_not_accidentally_restore_finite_limit(monkeypatch):
    choices = iter([0, None])
    saved = []
    monkeypatch.setattr(interactive, "select", lambda *_a, **_k: next(choices))
    monkeypatch.setattr(interactive, "_notice", lambda *_a: None)
    monkeypatch.setattr(
        settings,
        "settings_command",
        lambda: CommandResult(
            result={
                "settings": [
                    {
                        "key": "max_ai_requests",
                        "label": "Budget",
                        "value": None,
                        "description": "unlimited",
                        "source": "user",
                        "editable": True,
                    }
                ],
                "notes": [],
            }
        ),
    )

    def prompt(*_args, **kwargs):
        assert kwargs["default"] == "unlimited"
        return kwargs["default"]

    def update(key, value):
        saved.append((key, value))
        return CommandResult(result={"label": "Budget", "value": None})

    monkeypatch.setattr(interactive.typer, "prompt", prompt)
    monkeypatch.setattr(settings, "update_setting_command", update)
    interactive._settings(lambda _args: pytest.fail("unexpected command"))
    assert saved == [("max_ai_requests", "unlimited")]


def test_unknown_cost_unlimited_prompt_is_explicit_and_localized(monkeypatch):
    messages = []
    monkeypatch.setattr(
        cli, "require_confirmation", lambda message, **_kwargs: messages.append(message)
    )
    assert cli._unknown_cost_callback(yes=False, non_interactive=False, json_output=False)(None)
    assert "without a request-count limit" in messages[0] and "None" not in messages[0]
    translated = text(messages[0], language="ru")
    assert "без ограничения количества" in translated and "неизвестна" in translated


@pytest.mark.parametrize("suffix", ["Request count is unlimited.", "Request limit: 100."])
def test_ai_plan_budget_message_is_localized(suffix):
    message = (
        "AI plan: 30 item(s), 2 candidate batch(es), provider=gemini, model=test. "
        "Exact cache hits can reduce requests; retries, escalation and puzzle checks "
        "share the run budget. " + suffix
    )
    translated = text(message, language="ru")
    assert "проверки пазлов" in translated
    assert (
        "без лимита" in translated
        if "unlimited" in suffix
        else "Лимит запросов: 100." in translated
    )
