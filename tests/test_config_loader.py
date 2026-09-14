from pathlib import Path

import pytest

from mojilex_cli.config import (
    AIConfig,
    ConfigError,
    MojiLexConfig,
    load_config,
    load_credentials,
    safe_config_dict,
)


def test_config_precedence_and_safe_defaults(tmp_path: Path) -> None:
    user = tmp_path / "user.toml"
    project = tmp_path / "project.toml"
    user.write_text(
        '[repository]\ntarget="user/repo"\n[ai]\nmodel="user-model"\n', encoding="utf-8"
    )
    project.write_text(
        '[repository]\ntarget="project/repo"\n[ai]\nmodel="project-model"\n', encoding="utf-8"
    )

    config = load_config(
        cli={"ai": {"model": "cli-model"}, "repository": {"target": None}},
        environment={"MOJILEX_REPO": "env/repo", "MOJILEX_KEYFRAMES": "12"},
        project_path=project,
        user_path=user,
    )

    assert config.repository.target == "env/repo"
    assert config.ai.model == "cli-model"
    assert config.processing.keyframes == 12
    assert config.ai.languages == ("ru", "en")
    assert "token" not in repr(safe_config_dict(config)).lower()


def test_config_rejects_secrets_and_credential_urls(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.toml"
    unsafe.write_text('[telegram]\nbot_token="secret"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="secrets are forbidden"):
        load_config(environment={}, project_path=unsafe, user_path=tmp_path / "missing")

    with pytest.raises(ConfigError, match="credentials"):
        load_config(
            cli={"repository": {"target": "https://token@github.com/MojiLex/mojilex"}},
            environment={},
            project_path=tmp_path / "none",
            user_path=tmp_path / "none2",
        )


def test_credentials_are_runtime_only_and_hidden() -> None:
    credentials = load_credentials(
        {
            "TELEGRAM_BOT_TOKEN": "12345:abcdefghijklmnopqrstuvwxyz",
            "GH_TOKEN": "github-secret",
        }
    )
    assert "github-secret" not in repr(credentials)
    assert set(credentials.redaction_values()) == {
        "12345:abcdefghijklmnopqrstuvwxyz",
        "github-secret",
    }


def test_secret_values_are_rejected_and_safe_dump_is_defensive(tmp_path: Path) -> None:
    token = "github_pat_abcdefghijklmnopqrstuvwxyz123456"
    with pytest.raises(ConfigError, match="credentials") as failure:
        load_config(
            cli={"ai": {"model": token}},
            environment={},
            project_path=tmp_path / "none",
            user_path=tmp_path / "none2",
        )
    assert token not in str(failure.value)

    manually_constructed = MojiLexConfig(ai=AIConfig(model=token))
    assert safe_config_dict(manually_constructed)["ai"]["model"] == "<redacted>"


def test_validation_error_does_not_echo_invalid_input(tmp_path: Path) -> None:
    invalid = "private-invalid-publish-mode"
    with pytest.raises(ConfigError) as failure:
        load_config(
            cli={"repository": {"publish": invalid}},
            environment={},
            project_path=tmp_path / "none",
            user_path=tmp_path / "none2",
        )
    assert invalid not in str(failure.value)


def test_confirmation_defaults_and_environment_override(tmp_path: Path) -> None:
    paths = {"project_path": tmp_path / "project.toml", "user_path": tmp_path / "user.toml"}
    defaults = load_config(environment={}, **paths)
    assert defaults.processing.official_pack_policy == "skip"
    assert defaults.ai.confirm_before_analysis is False
    overridden = load_config(environment={"MOJILEX_CONFIRM_BEFORE_ANALYSIS": "true"}, **paths)
    assert overridden.ai.confirm_before_analysis is True
    assert overridden.ai.max_ai_requests == defaults.ai.max_ai_requests
    with pytest.raises(ConfigError):
        load_config(environment={"MOJILEX_CONFIRM_BEFORE_ANALYSIS": "maybe"}, **paths)
