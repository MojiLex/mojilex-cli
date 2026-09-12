from __future__ import annotations

import json
import os

import pytest
from typer.testing import CliRunner

from mojilex_cli.cli import _unknown_cost_callback, _with_runtime_secrets, app
from mojilex_cli.commands import workflow
from mojilex_cli.commands.runtime import CommandError, CommandResult
from test_dataset_helpers import write_fixture


def test_validate_json_is_one_machine_readable_object(tmp_path) -> None:
    write_fixture(tmp_path)
    result = CliRunner().invoke(app, ["validate", str(tmp_path), "--no-strict", "--json"])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert result.stdout.count("\n") == 1
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["command"] == "validate"
    assert payload["status"] == "succeeded"
    assert payload["result"]["issues"] == 0
    assert payload["errors"] == []


def test_validate_json_preserves_stable_validation_exit_code(tmp_path) -> None:
    result = CliRunner().invoke(app, ["validate", str(tmp_path), "--strict", "--json"])

    assert result.exit_code == 10
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert payload["errors"][0]["code"] == "VALIDATION_FAILED"


def test_unknown_ai_cost_fails_closed_in_machine_mode_with_opt_in_hint() -> None:
    authorize = _unknown_cost_callback(yes=False, non_interactive=True, json_output=True)

    with pytest.raises(CommandError) as captured:
        authorize(3)

    assert captured.value.error.code == "UNKNOWN_COST"
    assert captured.value.error.details == {
        "new_ai_requests": 3,
        "estimated_cost_usd": None,
    }
    assert "--allow-unknown-cost" in captured.value.error.hint


def test_yes_explicitly_authorizes_unknown_ai_cost() -> None:
    authorize = _unknown_cost_callback(yes=True, non_interactive=True, json_output=True)

    assert authorize(1)


def test_runtime_secret_prompt_is_ephemeral(monkeypatch) -> None:
    monkeypatch.setenv("MOJILEX_DISABLE_STORED_CREDENTIALS", "1")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    prompted: list[str] = []
    supplied = iter(("telegram-test-value", "gemini-test-value"))

    def action() -> CommandResult:
        assert os.environ["TELEGRAM_BOT_TOKEN"] == "telegram-test-value"
        assert os.environ["GEMINI_API_KEY"] == "gemini-test-value"
        return CommandResult(result={"ready": True})

    result = _with_runtime_secrets(
        action,
        names=("TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY"),
        prompt=lambda label: prompted.append(label) or next(supplied),
    )

    assert result.result == {"ready": True}
    assert prompted == ["Telegram Bot API token", "Gemini API key"]
    assert os.environ.get("TELEGRAM_BOT_TOKEN") is None
    assert os.environ.get("GEMINI_API_KEY") is None


def test_runtime_secret_prompt_preserves_existing_environment(monkeypatch) -> None:
    monkeypatch.setenv("MOJILEX_DISABLE_STORED_CREDENTIALS", "1")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "existing-test-value")

    result = _with_runtime_secrets(
        lambda: CommandResult(
            result={
                "token": os.environ["TELEGRAM_BOT_TOKEN"],
            }
        ),
        names=("TELEGRAM_BOT_TOKEN",),
        prompt=lambda _label: pytest.fail("existing credential must not be prompted"),
    )

    assert result.result == {"token": "existing-test-value"}
    assert os.environ["TELEGRAM_BOT_TOKEN"] == "existing-test-value"


def test_runtime_secret_uses_stored_value_without_prompt(monkeypatch) -> None:
    from mojilex_cli.config import credential_store

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        credential_store,
        "read_stored_credentials",
        lambda _names: {"GEMINI_API_KEY": "stored-gemini-value"},
    )

    result = _with_runtime_secrets(
        lambda: CommandResult(
            result={"available": os.environ["GEMINI_API_KEY"] == "stored-gemini-value"}
        ),
        names=("GEMINI_API_KEY",),
        prompt=lambda _label: pytest.fail("stored credential must not be prompted"),
    )

    assert result.result == {"available": True}
    assert "GEMINI_API_KEY" not in os.environ


def test_set_credentials_uses_hidden_prompts_and_never_renders_values(monkeypatch) -> None:
    from mojilex_cli import cli

    captured: dict[str, str] = {}
    supplied = iter(("telegram-hidden-value", "gemini-hidden-value"))

    def save(values: dict[str, str]) -> CommandResult:
        captured.update(values)
        return CommandResult(result={"saved": ["telegram", "gemini"]})

    monkeypatch.setattr(cli, "config_set_credentials_command", save)
    monkeypatch.setattr(
        cli,
        "_secret_prompt",
        lambda **_kwargs: lambda _label: next(supplied),
    )
    result = CliRunner().invoke(app, ["config", "set-credentials"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "TELEGRAM_BOT_TOKEN": "telegram-hidden-value",
        "GEMINI_API_KEY": "gemini-hidden-value",
    }
    assert "hidden-value" not in result.output


def test_doctor_offers_and_runs_windows_installer(monkeypatch) -> None:
    from types import SimpleNamespace

    from mojilex_cli import cli
    from mojilex_cli.output import RunStatus

    checks = {"media": []}
    initial = CommandResult(
        result={"ready": False, "checks": checks, "install_commands": ["installer"]},
        status=RunStatus.PARTIAL,
    )
    installed = CommandResult(result={"ready": True})
    monkeypatch.setattr(cli, "doctor_command", lambda: initial)
    monkeypatch.setattr(cli, "sys", SimpleNamespace(stdin=SimpleNamespace(isatty=lambda: True)))
    monkeypatch.setattr(cli.typer, "confirm", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        cli,
        "install_media_dependencies_command",
        lambda received: installed if received is checks else pytest.fail("wrong checks"),
    )

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "ready" in result.output
    assert "True" in result.output


def test_uninstall_yes_executes_exact_preview_without_second_prompt(monkeypatch) -> None:
    from mojilex_cli import cli

    preview = {
        "package": "mojilex-cli",
        "data_root": "synthetic-data-root",
    }
    calls: list[bool] = []
    monkeypatch.setattr(cli, "uninstall_preview_command", lambda **_kwargs: preview)
    monkeypatch.setattr(
        cli,
        "uninstall_command",
        lambda *, keep_data: (
            calls.append(keep_data) or CommandResult(result={**preview, "scheduled": True})
        ),
    )

    result = CliRunner().invoke(
        app,
        ["uninstall", "--yes", "--non-interactive", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["command"] == "uninstall"
    assert payload["result"]["scheduled"] is True
    assert calls == [False]


def test_clear_credentials_requires_confirmation_and_never_returns_values(monkeypatch) -> None:
    from mojilex_cli import cli

    monkeypatch.setattr(
        cli,
        "config_clear_credentials_command",
        lambda: CommandResult(result={"deleted": ["telegram", "gemini"]}),
    )
    result = CliRunner().invoke(
        app,
        ["config", "clear-credentials", "--yes", "--non-interactive", "--json"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["command"] == "config clear-credentials"
    assert payload["result"]["deleted"] == ["telegram", "gemini"]


def test_sensitive_add_prompt_is_deferred_until_exact_plan_exists(monkeypatch) -> None:
    def fake_add_command(sources, **kwargs):  # type: ignore[no-untyped-def]
        assert sources == ("https://t.me/addemoji/TestPack",)
        confirmed = kwargs["confirmation"](
            "Exact affected IDs: mxe_exact. Exact changed paths: data/emojis/exact.json."
        )
        assert confirmed is True
        return CommandResult(result={"planned": True})

    monkeypatch.setattr(workflow, "add_command", fake_add_command)

    result = CliRunner().invoke(
        app,
        ["add", "https://t.me/addemoji/TestPack", "--overwrite-reviewed"],
        input="y\n",
    )

    assert result.exit_code == 0, result.output
    assert "Exact affected IDs: mxe_exact" in result.output
    assert "Proceed with the explicitly sensitive" not in result.output


def test_resume_yes_forwards_non_persisted_runtime_authorization(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_resume_command(run_id, **kwargs):  # type: ignore[no-untyped-def]
        captured["run_id"] = run_id
        captured["confirmed"] = kwargs["confirmation"](
            "Reconfirm exact new-identity plan for mxe_exact."
        )
        captured["unknown_cost"] = kwargs["unknown_cost_confirmation"](2)
        return CommandResult(result={"resumed": True})

    monkeypatch.setattr(workflow, "resume_command", fake_resume_command)

    result = CliRunner().invoke(
        app,
        [
            "resume",
            "mlxrun_0123456789abcdef0123456789abcdef",
            "--yes",
            "--non-interactive",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "run_id": "mlxrun_0123456789abcdef0123456789abcdef",
        "confirmed": True,
        "unknown_cost": True,
    }
    assert "Reconfirm exact new-identity plan" not in result.output
