from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from mojilex_cli.commands import system
from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.config import Credentials, MojiLexConfig


def _checks(*, can_write: bool = True, can_publish_pr: bool = True) -> dict[str, object]:
    return {
        "python": {"available": True, "path": "python"},
        "git": {"available": True, "path": "git"},
        "github_cli": {"available": True, "path": "gh"},
        "git_identity": {
            "available": True,
            "name_configured": True,
            "email_configured": True,
            "source": "effective repository/config identity",
        },
        "media": [
            {"name": name, "available": True, "fixture_decoded": True, "detail": "ok"}
            for name in ("webp", "tgs/rlottie-rgba", "webm/ffmpeg")
        ],
        "github_access": {
            "authenticated": True,
            "can_publish_pr": can_publish_pr,
            "can_write": can_write,
            "detail": "GitHub write access verified" if can_write else "missing",
        },
    }


def test_init_requires_explicit_supported_model_before_writing(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"

    with pytest.raises(CommandError, match="explicit AI model ID"):
        system.init_command(
            repo="MojiLex/mojilex",
            provider="gemini",
            model="",
            publish="pr",
            config_path=target,
            force=False,
        )

    assert not target.exists()


def test_init_checks_runtime_and_writes_only_non_secret_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.toml"
    secrets = Credentials(
        telegram_bot_token="123456:telegram-secret-value-value",
        gemini_api_key="gemini-secret-value",
        github_token="github-secret-value",
    )
    monkeypatch.setattr(system, "load_credentials", lambda: secrets)
    monkeypatch.setattr(
        system,
        "_effective_git_identity",
        lambda _repository: ("Misha20062006", "perogovskij@gmail.com"),
    )
    monkeypatch.setattr(system, "_system_checks", lambda **_kwargs: _checks())

    result = system.init_command(
        repo="MojiLex/mojilex",
        provider="gemini",
        model="gemini-explicit-model",
        publish="pr",
        config_path=target,
        force=False,
    )

    assert result.result["ready"] is True
    assert result.result["credential_presence"] == {
        "telegram": True,
        "gemini": True,
        "github": True,
    }
    assert result.result["checks"] == _checks()
    raw = target.read_text(encoding="utf-8")
    assert "secret-value" not in raw
    parsed = tomllib.loads(raw)
    assert parsed["ai"]["model"] == "gemini-explicit-model"
    assert parsed["git_identity"] == {
        "name": "Misha20062006",
        "email": "perogovskij@gmail.com",
    }


def test_init_reports_missing_credentials_and_github_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.toml"
    monkeypatch.setattr(system, "load_credentials", Credentials)
    monkeypatch.setattr(
        system,
        "_effective_git_identity",
        lambda _repository: ("Misha20062006", "perogovskij@gmail.com"),
    )
    monkeypatch.setattr(
        system,
        "_system_checks",
        lambda **_kwargs: _checks(can_write=False, can_publish_pr=False),
    )

    result = system.init_command(
        repo="MojiLex/mojilex",
        provider="gemini",
        model="gemini-explicit-model",
        publish="pr",
        config_path=target,
        force=False,
    )

    assert result.result["ready"] is False
    warnings = "\n".join(str(item) for item in result.warnings)
    assert "TELEGRAM_BOT_TOKEN" in warnings
    assert "GEMINI_API_KEY" in warnings
    assert "missing" in warnings


def test_init_accepts_read_only_target_for_supported_fork_pr_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "config.toml"
    monkeypatch.setattr(
        system,
        "load_credentials",
        lambda: Credentials(
            telegram_bot_token="123456:telegram-secret-value-value",
            gemini_api_key="gemini-secret-value",
            github_token="github-secret-value",
        ),
    )
    monkeypatch.setattr(
        system,
        "_effective_git_identity",
        lambda _repository: ("Contributor", "contributor@example.test"),
    )
    monkeypatch.setattr(
        system,
        "_system_checks",
        lambda **_kwargs: _checks(can_write=False, can_publish_pr=True),
    )

    result = system.init_command(
        repo="MojiLex/mojilex",
        provider="gemini",
        model="gemini-explicit-model",
        publish="pr",
        config_path=target,
        force=False,
    )

    assert result.result["ready"] is True
    assert not any("GitHub" in str(item) for item in result.warnings)


def test_doctor_includes_active_python_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    checks = _checks()
    checks.pop("github_access")
    monkeypatch.setattr(system, "_system_checks", lambda **_kwargs: checks)
    monkeypatch.setattr(system, "load_credentials", Credentials)
    monkeypatch.setattr(system, "load_config", MojiLexConfig)

    result = system.doctor_command()

    assert result.result["checks"]["python"]["available"] is True
    assert result.result["ready"] is True


def test_doctor_recognizes_identity_saved_by_the_setup_wizard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks = _checks()
    checks["git_identity"] = {"available": False}
    config = MojiLexConfig.model_validate(
        {"git_identity": {"name": "Author", "email": "author@example.test"}}
    )
    monkeypatch.setattr(system, "_system_checks", lambda **_kwargs: checks)
    monkeypatch.setattr(system, "load_credentials", Credentials)
    monkeypatch.setattr(system, "load_config", lambda: config)
    result = system.doctor_command()
    assert result.result["checks"]["git_identity"]["available"] is True
    assert not any("Git user.name" in str(warning) for warning in result.warnings)


@pytest.mark.parametrize(
    "value",
    ['quoted "name"', "line\nbreak", "tab\tvalue", "control\x00end", r"C:\local\path", "Кириллица"],
)
def test_init_toml_string_roundtrips_without_escaping_into_configuration(value: str) -> None:
    parsed = tomllib.loads("value = " + system._quoted(value) + "\n")
    assert parsed == {"value": value}


def test_init_wizard_selects_config_and_saves_missing_identity_only_in_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "wizard.toml"
    answers = iter(
        (
            "Example/dataset",
            "gemini",
            "chosen-model",
            "en,ru",
            "local",
            "Author",
            "author@example.test",
        )
    )
    prompts: list[str] = []

    def prompt(label: str, _default: str) -> str:
        prompts.append(label)
        return next(answers)

    checks = _checks()
    checks["git_identity"] = {"available": False}
    monkeypatch.setattr(system, "_effective_git_identity", lambda _repo: None)
    monkeypatch.setattr(system, "_system_checks", lambda **_kwargs: checks)
    monkeypatch.setattr(system, "load_credentials", Credentials)
    monkeypatch.setattr(
        system,
        "_run_git",
        lambda *_args, **_kwargs: pytest.fail("wizard must not change Git config"),
    )
    result = system.init_command(
        repo="MojiLex/mojilex",
        provider="gemini",
        model="",
        publish="pr",
        config_path=target,
        force=False,
        prompt=prompt,
    )
    parsed = tomllib.loads(target.read_text(encoding="utf-8"))
    assert parsed["repository"]["target"] == "Example/dataset"
    assert parsed["repository"]["publish"] == "local"
    assert parsed["ai"]["model"] == "chosen-model"
    assert parsed["ai"]["languages"] == ["en", "ru"]
    assert parsed["git_identity"] == {"name": "Author", "email": "author@example.test"}
    assert result.result["checks"]["git_identity"]["available"] is True
    assert len(prompts) == 7


def test_init_existing_file_refuses_before_any_wizard_prompt(tmp_path: Path) -> None:
    target = tmp_path / "config.toml"
    target.write_text("preserved", encoding="utf-8")
    with pytest.raises(CommandError, match="already exists"):
        system.init_command(
            repo="MojiLex/mojilex",
            provider="gemini",
            model="",
            publish="pr",
            config_path=target,
            force=False,
            prompt=lambda *_args: pytest.fail("must not prompt before overwrite refusal"),
        )
    assert target.read_text(encoding="utf-8") == "preserved"


@pytest.mark.parametrize("mode", ("tty", "noninteractive", "json", "quiet", "pipe"))
def test_init_cli_never_prompts_in_machine_or_noninteractive_mode(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from mojilex_cli import cli
    from mojilex_cli.commands.runtime import CommandResult

    selected: list[object] = []
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: mode != "pipe"))
    monkeypatch.setattr(cli, "execute", lambda _command, operation, **_kwargs: operation())

    def initialize(**kwargs):
        selected.append(kwargs["prompt"])
        return CommandResult()

    monkeypatch.setattr(cli, "init_command", initialize)
    cli.initialize(
        model="configured-model",
        non_interactive=mode == "noninteractive",
        json_output=mode == "json",
        quiet=mode == "quiet",
    )
    assert callable(selected[0]) is (mode == "tty")
