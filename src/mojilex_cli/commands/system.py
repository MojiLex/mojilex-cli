"""Configuration and installation diagnostics."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from mojilex_cli.config import (
    MojiLexConfig,
    default_user_config_path,
    load_config,
    load_credentials,
    safe_config_dict,
)
from mojilex_cli.config.secrets import assert_no_secret_keys
from mojilex_cli.github import GitHubCLI, GitHubError, RepositoryRef
from mojilex_cli.media import probe_media_backends
from mojilex_cli.output import RunStatus

from .runtime import CommandError, CommandResult


def config_show_command() -> CommandResult:
    config = load_config()
    credentials = load_credentials()
    return CommandResult(
        result={
            "configuration": safe_config_dict(config),
            "credentials": {
                "telegram": bool(credentials.telegram_bot_token),
                "gemini": bool(credentials.gemini_api_key),
                "openai": bool(credentials.openai_api_key),
                "github": bool(credentials.github_token),
            },
        }
    )


def init_command(
    *,
    repo: str,
    provider: str,
    model: str,
    publish: str,
    config_path: Path | None,
    force: bool,
    languages: Sequence[str] = ("ru", "en"),
    prompt: Callable[[str, str], str] | None = None,
) -> CommandResult:
    target = (config_path or default_user_config_path()).expanduser().resolve()
    if target.exists() and not force:
        raise CommandError(
            "CONFIG_INVALID",
            f"Configuration already exists: {target}",
            hint=(
                "Initialization is already complete. init never requests or stores API keys. "
                "Run `mojilex add <PUBLIC_PACK_URL>` without --non-interactive; it will request "
                "missing Telegram and Gemini credentials with hidden input. Use --force only "
                "to replace the reviewed non-secret configuration."
            ),
        )
    if prompt is not None:
        repo = prompt("Target dataset path or OWNER/REPO", repo).strip()
        provider = prompt("AI provider (MVP: gemini)", provider).strip()
        model = prompt("Exact AI model ID (no model is selected automatically)", model).strip()
        languages = tuple(
            part.strip()
            for part in prompt(
                "Languages, comma-separated (MVP requires ru,en)", ",".join(languages)
            ).split(",")
            if part.strip()
        )
        publish = prompt("Publication mode (local or pr)", publish).strip()
    selected_provider = provider.strip().lower()
    selected_model = model.strip()
    if selected_provider != "gemini":
        raise CommandError(
            "CONFIG_INVALID",
            f"Unsupported AI provider: {selected_provider or '<empty>'}",
            hint="The MVP currently supports --provider gemini.",
        )
    if not selected_model:
        raise CommandError(
            "CONFIG_INVALID",
            "An explicit AI model ID is required.",
            hint="Pass --model with the exact Gemini model ID you intend to use.",
        )

    identity = _effective_git_identity(repo)
    configured_identity = False
    if identity is None and prompt is not None:
        name = prompt("Git author name to save only in MojiLex config (blank to skip)", "").strip()
        if name:
            email = prompt("Git author email to save only in MojiLex config", "").strip()
            if not email or any(ord(char) < 32 or ord(char) == 127 for char in name + email):
                raise CommandError(
                    "CONFIG_INVALID",
                    "Git identity requires a valid name and email.",
                    hint="Enter non-empty values without control characters, or skip the name.",
                )
            identity = (name, email)
            configured_identity = True
    candidate_payload: dict[str, Any] = {
        "repository": {"target": repo, "base_branch": "main", "publish": publish},
        "ai": {
            "provider": selected_provider,
            "model": selected_model,
            "languages": list(languages),
        },
    }
    if identity is not None:
        candidate_payload["git_identity"] = {"name": identity[0], "email": identity[1]}
    assert_no_secret_keys(candidate_payload)
    candidate = MojiLexConfig.model_validate(candidate_payload)
    credentials = load_credentials()
    checks = _system_checks(
        repository=repo,
        github_token=credentials.github_token,
        include_github=True,
    )
    if configured_identity:
        checks["git_identity"] = {
            "available": True,
            "name_configured": True,
            "email_configured": True,
            "source": "non-secret MojiLex configuration (Git global config unchanged)",
        }
    content = _config_toml(candidate)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=".mojilex-config-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)
    selected_credential = credentials.gemini_api_key
    warnings = _check_warnings(checks, required_publication=publish)
    if not credentials.telegram_bot_token:
        warnings.append(
            "TELEGRAM_BOT_TOKEN is not set; an interactive `mojilex add ...` run will request "
            "it with hidden input and use it only for that run. Set it in the environment for "
            "--non-interactive, --json, or --quiet."
        )
    if not selected_credential:
        warnings.append(
            "GEMINI_API_KEY is not set; an interactive `mojilex add ...` run will request it "
            "with hidden input and use it only for that run. Set it in the environment for "
            "--non-interactive, --json, or --quiet."
        )
    github_ready = bool(checks["github_access"].get("can_publish_pr"))
    ready = bool(
        checks["python"]["available"]
        and checks["git"]["available"]
        and checks["git_identity"]["available"]
        and all(item["available"] and item["fixture_decoded"] for item in checks["media"])
        and credentials.telegram_bot_token
        and selected_credential
        and (publish == "local" or github_ready)
    )
    return CommandResult(
        result={
            "config_path": str(target),
            "configuration": safe_config_dict(candidate),
            "credential_presence": {
                "telegram": bool(credentials.telegram_bot_token),
                selected_provider: bool(selected_credential),
                "github": bool(credentials.github_token),
            },
            "checks": checks,
            "ready": ready,
        },
        warnings=warnings,
    )


def doctor_command() -> CommandResult:
    config = load_config()
    checks = _system_checks(repository=config.repository.target, include_github=False)
    if config.git_identity.name and config.git_identity.email:
        checks["git_identity"] = {
            "available": True,
            "name_configured": True,
            "email_configured": True,
            "source": "non-secret MojiLex configuration",
        }
    credentials = load_credentials()
    checks["credentials"] = {
        "telegram": bool(credentials.telegram_bot_token),
        "gemini": bool(credentials.gemini_api_key),
        "openai": bool(credentials.openai_api_key),
        "github": bool(credentials.github_token),
    }
    warnings = _check_warnings(checks)
    install_commands = _media_install_commands(checks)
    if install_commands:
        warnings.append("Install missing media backends with: " + install_commands[0])
    ready = bool(
        checks["python"]["available"]
        and checks["git"]["available"]
        and all(item["available"] and item["fixture_decoded"] for item in checks["media"])
    )
    return CommandResult(
        result={
            "checks": checks,
            "ready": ready,
            "install_commands": install_commands,
        },
        warnings=warnings,
        status=RunStatus.SUCCEEDED if ready else RunStatus.PARTIAL,
    )


def _media_install_commands(
    checks: dict[str, Any],
    *,
    platform_name: str | None = None,
    script_path: Path | None = None,
) -> list[str]:
    missing = {
        item["name"]
        for item in checks["media"]
        if not item["available"] or not item["fixture_decoded"]
    }
    if (platform_name or os.name) != "nt" or not missing.intersection(
        {"tgs/rlottie-rgba", "webm/ffmpeg"}
    ):
        return []
    script = script_path or (
        Path(__file__).resolve().parents[1] / "installers" / "install_media_windows.ps1"
    )
    if not script.is_file():
        return []
    escaped = str(script).replace('"', '""')
    return [f'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "{escaped}"']


def _system_checks(
    *,
    repository: str | None = None,
    github_token: str | None = None,
    include_github: bool,
) -> dict[str, Any]:
    identity = _effective_git_identity(repository)
    checks: dict[str, Any] = {
        "python": _program_check(sys.executable, ["--version"]),
        "git": _program_check("git", ["--version"]),
        "github_cli": _program_check("gh", ["--version"]),
        "git_identity": {
            "available": identity is not None,
            "name_configured": bool(identity and identity[0]),
            "email_configured": bool(identity and identity[1]),
            "source": "effective repository/config identity" if identity else None,
        },
    }
    backends = probe_media_backends()
    checks["media"] = [item.model_dump(mode="json") for item in backends]
    if include_github:
        checks["github_access"] = _github_access_check(repository, github_token)
    return checks


def _check_warnings(
    checks: dict[str, Any], *, required_publication: str | None = None
) -> list[dict[str, Any] | str]:
    warnings: list[dict[str, Any] | str] = []
    missing = [
        item["name"]
        for item in checks["media"]
        if not item["available"] or not item["fixture_decoded"]
    ]
    if missing:
        warnings.append(
            "Unavailable media backends: "
            + ", ".join(missing)
            + ". Install FFmpeg for WebM and the MojiLex rlottie RGBA adapter for TGS."
        )
    if not checks["python"]["available"]:
        warnings.append("The active Python interpreter could not execute --version.")
    if not checks["git"]["available"]:
        warnings.append("Git is not available on PATH.")
    if not checks["git_identity"]["available"]:
        warnings.append(
            "Git user.name and user.email are not both configured; set them in the dataset "
            "repository or add both values to the non-secret MojiLex configuration."
        )
    github = checks.get("github_access")
    if (
        required_publication == "pr"
        and isinstance(github, dict)
        and not github.get("can_publish_pr")
    ):
        warnings.append(str(github.get("detail") or "GitHub PR access was not verified."))
    return warnings


def _effective_git_identity(repository: str | None) -> tuple[str, str] | None:
    working_directory: Path | None = None
    if repository:
        candidate = Path(repository).expanduser()
        if candidate.is_dir():
            working_directory = candidate.resolve()
    name = _git_value("user.name", cwd=working_directory)
    email = _git_value("user.email", cwd=working_directory)
    if name and email:
        return name, email
    return None


def _github_access_check(repository: str | None, token: str | None) -> dict[str, Any]:
    if shutil.which("gh") is None:
        return {
            "authenticated": False,
            "can_publish_pr": False,
            "can_write": False,
            "detail": "GitHub CLI is not installed; PR publication is unavailable.",
        }
    try:
        github = GitHubCLI(token=token, timeout_seconds=20)
        github.auth_status()
        login, _ = github.current_user()
        repository_ref = _repository_ref(repository)
        if repository_ref is None:
            return {
                "authenticated": True,
                "user": login,
                "can_publish_pr": False,
                "can_write": False,
                "detail": (
                    "GitHub authentication works, but the target repository is not identifiable."
                ),
            }
        info = github.repository_info(repository_ref)
        return {
            "authenticated": True,
            "user": login,
            "repository": info.name_with_owner,
            "permission": info.permission,
            "default_branch": info.default_branch,
            "can_publish_pr": True,
            "can_write": info.can_write,
            "detail": "GitHub target write access verified"
            if info.can_write
            else (
                "GitHub target read access verified; PR publication will create or verify "
                "the authenticated user's writable fork"
            ),
        }
    except GitHubError as exc:
        return {
            "authenticated": False,
            "can_publish_pr": False,
            "can_write": False,
            "detail": f"GitHub access could not be verified: {exc}",
        }


def _repository_ref(repository: str | None) -> RepositoryRef | None:
    if not repository:
        return None
    value = repository
    candidate = Path(repository).expanduser()
    if candidate.is_dir():
        completed = _run_git(
            ["remote", "get-url", "origin"],
            cwd=candidate.resolve(),
        )
        if completed is None:
            return None
        value = completed
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    try:
        return RepositoryRef.parse(value)
    except GitHubError:
        return None


def _program_check(command: str, args: list[str]) -> dict[str, Any]:
    path = shutil.which(command)
    if path is None:
        return {"available": False}
    try:
        completed = subprocess.run(
            [path, *args], capture_output=True, text=True, check=False, timeout=10, shell=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"available": False}
    return {"available": completed.returncode == 0, "path": path}


def _git_value(key: str, *, cwd: Path | None = None) -> str | None:
    return _run_git(["config", "--get", key], cwd=cwd)


def _run_git(arguments: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
            shell=False,
            cwd=cwd,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _quoted(value: str) -> str:
    # JSON's basic-string escapes are also valid TOML, including controls.
    return json.dumps(value, ensure_ascii=False)


def _config_toml(config: MojiLexConfig) -> str:
    lines = [
        "# Non-secret MojiLex configuration. Keep API tokens in environment variables.",
        "[repository]",
        f"target = {_quoted(config.repository.target)}",
        f"base_branch = {_quoted(config.repository.base_branch)}",
        f"publish = {_quoted(config.repository.publish)}",
        "",
        "[telegram]",
        f"timeout_seconds = {config.telegram.timeout_seconds:g}",
        f"download_concurrency = {config.telegram.download_concurrency}",
        "",
        "[ai]",
        f"provider = {_quoted(config.ai.provider)}",
        f"model = {_quoted(config.ai.model)}",
        "languages = [" + ", ".join(_quoted(language) for language in config.ai.languages) + "]",
        f"max_ai_requests = {config.ai.max_ai_requests}",
        f"ai_concurrency = {config.ai.ai_concurrency}",
        "",
        "[processing]",
        f"static_batch_size = {config.processing.static_batch_size}",
        f"animated_batch_size = {config.processing.animated_batch_size}",
        f"keyframes = {config.processing.keyframes}",
        f"render_timeout_seconds = {config.processing.render_timeout_seconds:g}",
        "",
    ]
    if config.git_identity.name is not None and config.git_identity.email is not None:
        lines.extend(
            [
                "[git_identity]",
                f"name = {_quoted(config.git_identity.name)}",
                f"email = {_quoted(config.git_identity.email)}",
                "",
            ]
        )
    return "\n".join(lines)
