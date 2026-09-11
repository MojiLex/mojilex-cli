from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from mojilex_cli.commands import system
from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.config import Credentials


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

    result = system.doctor_command()

    assert result.result["checks"]["python"]["available"] is True
    assert result.result["ready"] is True
