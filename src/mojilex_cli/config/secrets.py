"""Secret detection and output redaction shared by service adapters."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

SECRET_ENV_NAMES = frozenset(
    {"TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY", "OPENAI_API_KEY", "GH_TOKEN", "GITHUB_TOKEN"}
)
_SECRET_KEY = re.compile(r"(?:^|_)(?:token|secret|password|api_?key|credential)(?:$|_)", re.I)
_TELEGRAM_BOT_PATH = re.compile(r"(?i)(/bot)[^/\s?#]+")
_AUTHORITY_CREDENTIALS = re.compile(r"(?i)(https?://)[^/@\s]+@")
_KNOWN_CREDENTIALS = (
    re.compile(r"(?<![A-Za-z0-9])[0-9]{5,20}:[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"(?<![A-Za-z0-9])sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
)


def is_secret_key(key: str) -> bool:
    return key.upper() in SECRET_ENV_NAMES or bool(_SECRET_KEY.search(key))


def url_has_credentials(value: str) -> bool:
    """Return true for URL userinfo, including token-only userinfo."""

    try:
        parsed = urlsplit(value)
    except ValueError:
        return True
    return parsed.scheme.lower() in {"http", "https", "ssh", "git"} and (
        parsed.username is not None or parsed.password is not None
    )


def redact_text(value: object, secrets: tuple[str, ...] = ()) -> str:
    text = str(value)
    for secret in sorted((item for item in secrets if item), key=len, reverse=True):
        text = text.replace(secret, "<redacted>")
    for pattern in _KNOWN_CREDENTIALS:
        text = pattern.sub("<redacted>", text)
    text = _TELEGRAM_BOT_PATH.sub(r"\1<redacted>", text)
    text = _AUTHORITY_CREDENTIALS.sub(r"\1<redacted>@", text)
    return text


def contains_secret_text(value: str) -> bool:
    return redact_text(value) != value


def redact_mapping(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>" if is_secret_key(str(key)) else redact_mapping(item, secrets)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_mapping(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_mapping(item, secrets) for item in value)
    return redact_text(value, secrets) if isinstance(value, str) else value


def assert_no_secret_keys(value: Any, *, path: str = "config") -> None:
    if isinstance(value, str):
        if contains_secret_text(value) or url_has_credentials(value):
            raise ValueError(f"credentials are forbidden in {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if is_secret_key(key_text):
                raise ValueError(f"secrets are forbidden in {path}: {key_text}")
            assert_no_secret_keys(item, path=f"{path}.{key_text}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_no_secret_keys(item, path=f"{path}[{index}]")
