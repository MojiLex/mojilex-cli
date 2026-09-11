"""Deterministic five-level configuration resolution."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path, user_config_path, user_state_path
from pydantic import ValidationError

from .models import ConfigError, Credentials, MojiLexConfig
from .secrets import SECRET_ENV_NAMES, assert_no_secret_keys, redact_mapping, redact_text

_ENV_PATHS: dict[str, tuple[str, ...]] = {
    "MOJILEX_REPO": ("repository", "target"),
    "MOJILEX_BASE_BRANCH": ("repository", "base_branch"),
    "MOJILEX_PUBLISH": ("repository", "publish"),
    "MOJILEX_TELEGRAM_TIMEOUT_SECONDS": ("telegram", "timeout_seconds"),
    "MOJILEX_DOWNLOAD_CONCURRENCY": ("telegram", "download_concurrency"),
    "MOJILEX_PROVIDER": ("ai", "provider"),
    "MOJILEX_MODEL": ("ai", "model"),
    "MOJILEX_LANGUAGES": ("ai", "languages"),
    "MOJILEX_MAX_AI_REQUESTS": ("ai", "max_ai_requests"),
    "MOJILEX_MAX_COST_USD": ("ai", "max_cost_usd"),
    "MOJILEX_AI_CONCURRENCY": ("ai", "ai_concurrency"),
    "MOJILEX_MODEL_ROUTING": ("ai", "model_routing"),
    "MOJILEX_ESCALATION_MODEL": ("ai", "escalation_model"),
    "MOJILEX_DEDUPE": ("dedupe", "mode"),
    "MOJILEX_MAX_DEDUPE_CANDIDATES": ("dedupe", "max_candidates"),
    "MOJILEX_DEDUPE_PROFILE": ("dedupe", "profile"),
    "MOJILEX_KEYFRAMES": ("processing", "keyframes"),
    "MOJILEX_RENDER_TIMEOUT_SECONDS": ("processing", "render_timeout_seconds"),
    "MOJILEX_CACHE_DIR": ("cache_dir",),
    "MOJILEX_RUNS_DIR": ("runs_dir",),
}


def default_user_config_path() -> Path:
    return user_config_path("mojilex", "MojiLex") / "config.toml"


def default_cache_dir() -> Path:
    return user_cache_path("mojilex", "MojiLex")


def default_runs_dir() -> Path:
    return user_state_path("mojilex", "MojiLex") / "runs"


def _read_toml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot read configuration {path}: {redact_text(exc)}") from exc
    try:
        assert_no_secret_keys(data, path=str(path))
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return data


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def _coerce_env(name: str, value: str) -> Any:
    if name == "MOJILEX_LANGUAGES":
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if name in {
        "MOJILEX_TELEGRAM_TIMEOUT_SECONDS",
        "MOJILEX_MAX_COST_USD",
        "MOJILEX_RENDER_TIMEOUT_SECONDS",
    }:
        return value
    if name in {
        "MOJILEX_DOWNLOAD_CONCURRENCY",
        "MOJILEX_MAX_AI_REQUESTS",
        "MOJILEX_AI_CONCURRENCY",
        "MOJILEX_MAX_DEDUPE_CANDIDATES",
        "MOJILEX_KEYFRAMES",
    }:
        try:
            return int(value)
        except ValueError as exc:
            raise ConfigError(f"{name} must be an integer") from exc
    return value


def _environment_layer(environment: Mapping[str, str]) -> dict[str, Any]:
    layer: dict[str, Any] = {}
    for name, path in _ENV_PATHS.items():
        if name not in environment:
            continue
        cursor = layer
        for component in path[:-1]:
            cursor = cursor.setdefault(component, {})
        cursor[path[-1]] = _coerce_env(name, environment[name])
    return layer


def load_config(
    *,
    cli: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
    project_path: Path | None = None,
    user_path: Path | None = None,
) -> MojiLexConfig:
    """Resolve CLI > env > project > user > safe defaults.

    ``cli`` uses the same nested shape as :class:`MojiLexConfig`; keys whose
    value is ``None`` are ignored so parser defaults do not accidentally win.
    """

    environment = os.environ if environment is None else environment
    project_path = Path.cwd() / ".mojilex.toml" if project_path is None else project_path
    user_path = default_user_config_path() if user_path is None else user_path
    defaults = MojiLexConfig(
        cache_dir=default_cache_dir(),
        runs_dir=default_runs_dir(),
    ).model_dump(mode="python")
    merged = _deep_merge(defaults, _read_toml(user_path))
    merged = _deep_merge(merged, _read_toml(project_path))
    merged = _deep_merge(merged, _environment_layer(environment))
    cli_layer = _drop_none(dict(cli or {}))
    try:
        assert_no_secret_keys(cli_layer, path="command arguments")
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    merged = _deep_merge(merged, cli_layer)
    try:
        assert_no_secret_keys(merged, path="resolved configuration")
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    try:
        return MojiLexConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(_safe_validation_error(exc)) from exc


def _drop_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    return value


def load_credentials(environment: Mapping[str, str] | None = None) -> Credentials:
    environment = os.environ if environment is None else environment
    github_token = environment.get("GH_TOKEN") or environment.get("GITHUB_TOKEN")
    return Credentials(
        telegram_bot_token=environment.get("TELEGRAM_BOT_TOKEN"),
        gemini_api_key=environment.get("GEMINI_API_KEY"),
        openai_api_key=environment.get("OPENAI_API_KEY"),
        github_token=github_token,
    )


def safe_config_dict(config: MojiLexConfig) -> dict[str, Any]:
    value = redact_mapping(config.model_dump(mode="json"))
    if not isinstance(value, dict):
        raise ConfigError("configuration serialization returned an invalid shape")
    return value


def _safe_validation_error(error: ValidationError) -> str:
    details: list[str] = []
    for item in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(component) for component in item.get("loc", ())) or "config"
        message = redact_text(item.get("msg", "invalid value"))
        details.append(f"{location}: {message}")
    return "configuration validation failed: " + "; ".join(details[:20])


def secret_environment(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return only known credential variables for explicit child filtering/tests."""

    environment = os.environ if environment is None else environment
    return {key: environment[key] for key in SECRET_ENV_NAMES if key in environment}
