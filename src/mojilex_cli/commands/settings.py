"""Small, non-secret settings API for the ordinary CLI and interactive menu.

``settings_command`` returns display rows, including the source of each effective
value. ``update_setting_command(key, value)`` changes one whitelisted setting in
its project layer when defined there, otherwise in the user layer. Environment
overrides must be removed explicitly before editing. Neither function reads keys,
contacts a provider, nor changes existing run budgets.
"""

# ruff: noqa: RUF001 -- intentional Russian interface strings.

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from mojilex_cli.config import (
    ConfigError,
    MojiLexConfig,
    default_user_config_path,
    load_config,
    safe_config_dict,
)
from mojilex_cli.config.loader import _read_toml
from mojilex_cli.i18n import current_ui_language

from .runtime import CommandError, CommandResult


@dataclass(frozen=True)
class Setting:
    path: tuple[str, ...]
    label: str
    description: str
    environment: str | None = None
    kind: str = "text"
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[str, ...] = ()


SETTINGS: dict[str, Setting] = {
    "ui_language": Setting(
        ("ui_language",),
        "Interface language",
        "Language used by menus and messages.",
        "MOJILEX_UI_LANGUAGE",
        choices=("ru", "en"),
    ),
    "official_pack_policy": Setting(
        ("processing", "official_pack_policy"),
        "Packs already published in MojiLex",
        "Checks the shared MojiLex database on GitHub, not your local cache. "
        "ask: one confirmation, Enter means No; skip: omit silently; allow: do not check.",
        "MOJILEX_OFFICIAL_PACK_POLICY",
        choices=("ask", "skip", "allow"),
    ),
    "provider": Setting(
        ("ai", "provider"),
        "AI provider",
        "The service that creates descriptions.",
        "MOJILEX_PROVIDER",
        choices=("gemini",),
    ),
    "model": Setting(
        ("ai", "model"),
        "AI model",
        "Exact model ID; no model is chosen automatically.",
        "MOJILEX_MODEL",
    ),
    "max_ai_requests": Setting(
        ("ai", "max_ai_requests"),
        "AI request limit for the whole operation",
        "Shared by all packs, including every pack in a file, retries and puzzle checks. "
        "Enter unlimited to disable; 0 prevents requests. Saved runs keep their budget.",
        "MOJILEX_MAX_AI_REQUESTS",
        "integer",
        0,
    ),
    "max_cost_usd": Setting(
        ("ai", "max_cost_usd"),
        "Cost limit (USD)",
        "Applies when cost is known. Blank or none removes this file's cost override.",
        "MOJILEX_MAX_COST_USD",
        "decimal",
        0,
    ),
    "pack_concurrency": Setting(
        ("processing", "pack_concurrency"),
        "Legacy pack concurrency",
        "Kept for configuration compatibility. Packs run sequentially; download, decoder and AI "
        "concurrency apply within the current pack.",
        "MOJILEX_PACK_CONCURRENCY",
        "integer",
        1,
        8,
    ),
    "download_concurrency": Setting(
        ("telegram", "download_concurrency"),
        "Parallel media downloads",
        "Maximum simultaneous media downloads within the current pack.",
        "MOJILEX_DOWNLOAD_CONCURRENCY",
        "integer",
        1,
        32,
    ),
    "render_concurrency": Setting(
        ("processing", "render_concurrency"),
        "Parallel media decoders",
        "Maximum simultaneous media decoders within the current pack. "
        "Too many CPU-heavy decoders can cause timeouts.",
        "MOJILEX_RENDER_CONCURRENCY",
        "integer",
        1,
        8,
    ),
    "ai_concurrency": Setting(
        ("ai", "ai_concurrency"),
        "Parallel AI requests",
        "Maximum simultaneous AI requests within the current pack, subject to provider limits "
        "and the saved run budget.",
        "MOJILEX_AI_CONCURRENCY",
        "integer",
        1,
        16,
    ),
    "download_attempts": Setting(
        ("telegram", "max_attempts"),
        "Download connection attempts",
        "Maximum attempts for a transient download error, including the first attempt.",
        kind="integer",
        minimum=1,
        maximum=8,
    ),
    "timeout_seconds": Setting(
        ("telegram", "timeout_seconds"),
        "Download timeout (seconds)",
        "Time allowed for a Telegram request before retrying a temporary failure.",
        "MOJILEX_TELEGRAM_TIMEOUT_SECONDS",
        "decimal",
        0,
        120,
    ),
}

_RUSSIAN: dict[str, tuple[str, str]] = {
    "official_pack_policy": (
        "Паки, уже опубликованные в MojiLex",
        "Проверка общей базы MojiLex на GitHub, а не локального кэша. "
        "ask — спросить один раз, Enter означает Нет; skip — пропускать; allow — не проверять.",
    ),
    "pack_concurrency": (
        "Прежняя параллельность паков",
        "Сохраняется для совместимости конфигурации. Паки выполняются последовательно; "
        "параллельность скачиваний, декодеров и ИИ действует внутри текущего пака.",
    ),
    "render_concurrency": (
        "Параллельные декодеры медиа",
        "Максимум одновременных декодеров медиа внутри текущего пака. "
        "Слишком много декодеров вызывает таймауты.",
    ),
    "provider": ("Сервис ИИ", "Сервис, который создаёт описания эмодзи."),
    "model": ("Модель ИИ", "Точное название модели; программа не выбирает модель автоматически."),
    "ai_concurrency": (
        "Параллельные запросы к ИИ",
        "Максимум одновременных запросов к ИИ внутри текущего пака, с учётом квот сервиса "
        "и бюджета запуска.",
    ),
    "max_ai_requests": (
        "Лимит запросов ИИ на всю операцию",
        "Общий для всех паков, в том числе всех паков файла, повторов и проверок пазлов. "
        "unlimited — без лимита; 0 — запрет запросов. У сохранённых запусков свой бюджет.",
    ),
    "max_cost_usd": (
        "Лимит стоимости (USD)",
        "Работает при известной цене. Пустое значение или none убирает лимит этого файла настроек.",
    ),
    "download_concurrency": (
        "Параллельные скачивания",
        "Максимум одновременных скачиваний файлов внутри текущего пака.",
    ),
    "download_attempts": (
        "Попытки подключения при скачивании",
        "Максимум попыток при временном сбое, включая первую попытку.",
    ),
    "timeout_seconds": (
        "Ожидание Telegram (секунды)",
        "Через сколько секунд ожидания повторять временно неудачный запрос.",
    ),
    "ui_language": ("Язык интерфейса", "Язык меню и сообщений программы."),
}


def _localized(english: str, russian: str) -> str:
    return russian if current_ui_language() == "ru" else english


def _label(key: str) -> str:
    return _localized(SETTINGS[key].label, _RUSSIAN[key][0])


def _paths(
    project_path: Path | None,
    user_path: Path | None,
) -> tuple[Path, Path]:
    return (
        project_path if project_path is not None else Path.cwd() / ".mojilex.toml",
        user_path if user_path is not None else default_user_config_path(),
    )


def _get(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _has(data: Mapping[str, Any], path: tuple[str, ...]) -> bool:
    return _get(data, path) is not None


def settings_command(
    *,
    environment: Mapping[str, str] | None = None,
    project_path: Path | None = None,
    user_path: Path | None = None,
) -> CommandResult:
    """Read effective ordinary settings without querying credentials or services."""
    environment = os.environ if environment is None else environment
    project_path, user_path = _paths(project_path, user_path)
    config = load_config(environment=environment, project_path=project_path, user_path=user_path)
    safe = safe_config_dict(config)
    project, user = _read_toml(project_path), _read_toml(user_path)
    rows: list[dict[str, Any]] = []
    for key, setting in SETTINGS.items():
        overridden = bool(setting.environment and setting.environment in environment)
        source = (
            "environment"
            if overridden
            else "project"
            if _has(project, setting.path)
            else "user"
            if _has(user, setting.path)
            else "default"
        )
        rows.append(
            {
                "key": key,
                "label": _label(key),
                "description": _localized(setting.description, _RUSSIAN[key][1]),
                "value": _get(safe, setting.path),
                "source": source,
                "editable": not overridden,
                "environment_variable": setting.environment if overridden else None,
                "choices": list(setting.choices),
                "minimum": setting.minimum,
                "maximum": setting.maximum,
            }
        )
    return CommandResult(
        result={
            "view": "settings",
            "settings": rows,
            "notes": [
                _localized(
                    "Priority: command options, environment, project, then user settings.",
                    "Приоритет настроек: параметры команды → окружение → проект → общие настройки.",
                ),
                _localized(
                    "Changes apply to new operations. Saved runs keep their settings and limits.",
                    "Изменения действуют для новых операций. "
                    "Сохранённые запуски сохраняют свои настройки.",
                ),
                _localized(
                    "Temporary failures are retried automatically. "
                    "Retries use the AI request limit.",
                    "Временные сбои повторяются автоматически. "
                    "Повторы расходуют лимит запросов ИИ.",
                ),
            ],
        }
    )


def _invalid(message: str, hint: str) -> CommandError:
    return CommandError("CONFIG_INVALID", message, hint=hint)


def _parse_value(setting: Setting, raw: str) -> Any:
    value = raw.strip()
    if setting.path == ("ai", "max_cost_usd") and value.lower() in {"", "none"}:
        return None
    if setting.path == ("ai", "max_ai_requests") and value.lower() in {
        "unlimited",
        "без лимита",
    }:
        return "unlimited"
    try:
        if setting.kind == "integer":
            parsed: Any = int(value)
        elif setting.kind == "decimal":
            parsed = Decimal(value)
            if not parsed.is_finite():
                raise ValueError("non-finite")
            if not math.isfinite(float(parsed)):
                raise ValueError("out of range")
        else:
            parsed = value
            if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
                raise ValueError("empty or control characters")
            if setting.choices and value not in setting.choices:
                raise ValueError("unsupported choice")
        if setting.minimum is not None and parsed < setting.minimum:
            raise ValueError("below minimum")
        if setting.maximum is not None and parsed > setting.maximum:
            raise ValueError("above maximum")
    except (ValueError, InvalidOperation) as exc:
        raise _invalid(
            "The setting value is invalid. Nothing was changed.",
            "Use one of the listed choices or a number within the displayed range.",
        ) from exc
    return parsed


def _literal(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, str) else str(value)


def _line_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
        elif char == "\\" and quote == '"':
            escaped = True
        elif quote is not None:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "#":
            return " " + value[index:].rstrip("\r\n")
    return ""


def _replace_scalar(content: str, setting: Setting, value: Any, *, exists: bool) -> str:
    """Preserve all unrelated text; unusual TOML syntax fails closed below."""
    newline = "\r\n" if "\r\n" in content else "\n"
    lines = content.splitlines(keepends=True)
    section = ".".join(setting.path[:-1])
    key = setting.path[-1]
    active = ""
    insert_at = len(lines) if section else 0
    found_section = not section
    for index, line in enumerate(lines):
        header = re.match(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$", line.rstrip("\r\n"))
        if header:
            if active == section and found_section:
                insert_at = index
            active = header.group(1).strip()
            if active == section:
                found_section = True
                insert_at = index + 1
            continue
        if active != section:
            continue
        if not exists:
            insert_at = index + 1
        match = re.match(
            rf"^(\s*(?:{re.escape(key)}|\"{re.escape(key)}\"|'"
            rf"{re.escape(key)}')\s*=\s*)(.*)$",
            line.rstrip("\r\n"),
        )
        if match:
            if '"""' in match.group(2) or "'''" in match.group(2):
                break
            comment = _line_comment(match.group(2))
            lines[index] = (
                (comment.lstrip() + newline if comment else "")
                if value is None
                else match.group(1) + _literal(value) + comment + newline
            )
            return "".join(lines)
    if exists:
        raise _invalid(
            "This setting uses a complex TOML form. Nothing was changed.",
            "Use a single setting line in its named TOML section before editing it here.",
        )
    if value is None:
        return content
    addition = f"{key} = {_literal(value)}{newline}"
    if not found_section:
        return content.rstrip("\r\n") + newline + f"[{section}]" + newline + addition
    if insert_at and not lines[insert_at - 1].endswith(("\n", "\r")):
        lines[insert_at - 1] += newline
    lines.insert(insert_at, addition)
    return "".join(lines)


def update_setting_command(
    key: str,
    value: str,
    *,
    environment: Mapping[str, str] | None = None,
    project_path: Path | None = None,
    user_path: Path | None = None,
) -> CommandResult:
    """Validate then atomically save one explicit non-secret setting edit.

    Project fields are edited in place; otherwise edits go to user config. No
    environment value is copied into either document. Unknown/credential keys,
    overridden values and invalid candidates fail without writing a file.
    """
    if key not in SETTINGS:
        raise _invalid("Unknown setting. Nothing was changed.", "Open `mojilex settings`.")
    environment = os.environ if environment is None else environment
    setting = SETTINGS[key]
    if setting.environment and setting.environment in environment:
        raise _invalid(
            "An environment variable overrides this setting. Nothing was changed.",
            f"Unset {setting.environment} in this terminal, then edit the setting again.",
        )
    project_path, user_path = _paths(project_path, user_path)
    # Validate the complete existing layers before making any change.
    current_config = load_config(
        environment=environment, project_path=project_path, user_path=user_path
    )
    project, user = _read_toml(project_path), _read_toml(user_path)
    scope = "project" if _has(project, setting.path) else "user"
    target = project_path if scope == "project" else user_path
    original = project if scope == "project" else user
    if target.is_symlink():
        raise _invalid(
            "Linked configuration was not changed.", "Edit the actual configuration file."
        )
    before = target.read_bytes() if target.exists() else None
    candidate = _parse_value(setting, value)
    patch: dict[str, Any] = {}
    cursor = patch
    for part in setting.path[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[setting.path[-1]] = candidate
    # Reuse the real resolver and model validators, including secret rejection.
    try:
        if candidate is None:
            resolved = current_config.model_dump(mode="python")
            resolved["ai"]["max_cost_usd"] = (
                _get(user, setting.path) if scope == "project" else None
            )
            validated = MojiLexConfig.model_validate(resolved)
        else:
            validated = load_config(
                cli=patch, environment=environment, project_path=project_path, user_path=user_path
            )
    except ConfigError as exc:
        raise _invalid("The setting value is invalid. Nothing was changed.", str(exc)) from exc
    if candidate is None and not _has(original, setting.path):
        return _saved_result(key, validated, scope)
    rendered = _replace_scalar(
        before.decode("utf-8") if before is not None else "",
        setting,
        candidate,
        exists=_has(original, setting.path),
    )
    expected = deepcopy(original)
    cursor = expected
    for part in setting.path[:-1]:
        cursor = cursor.setdefault(part, {})
    # TOML numbers become int/float; compare to the exact candidate literal parsed the same way.
    if candidate is None:
        cursor.pop(setting.path[-1], None)
    else:
        cursor[setting.path[-1]] = tomllib.loads(f"value = {_literal(candidate)}")["value"]
    try:
        if tomllib.loads(rendered) != expected:
            raise ValueError("unrelated configuration changed")
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        raise _invalid(
            "The configuration could not be safely edited. Nothing was changed.",
            "Use a single setting line in its named TOML section.",
        ) from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=".mojilex-settings-", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(rendered.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        current = target.read_bytes() if target.exists() else None
        if current != before:
            raise _invalid(
                "Configuration changed while editing. Nothing was overwritten.",
                "Open settings again and repeat the edit.",
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return _saved_result(key, validated, scope)


def _saved_result(key: str, validated: MojiLexConfig, scope: str) -> CommandResult:
    return CommandResult(
        result={
            "view": "setting_updated",
            "key": key,
            "label": _label(key),
            "value": _get(safe_config_dict(validated), SETTINGS[key].path),
            "scope": scope,
            "note": _localized(
                "Saved for new operations. Existing run settings and limits are unchanged.",
                "Сохранено для новых операций. Настройки и лимиты прежних запусков не изменены.",
            ),
        }
    )
