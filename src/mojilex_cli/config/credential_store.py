"""Opt-in storage for API credentials in the operating-system keyring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import keyring
from keyring.errors import KeyringError

from .models import ConfigError

SERVICE_NAME = "mojilex-cli"
STORED_CREDENTIAL_NAMES: tuple[str, ...] = (
    "TELEGRAM_BOT_TOKEN",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
)


def read_stored_credentials(
    names: Sequence[str] = STORED_CREDENTIAL_NAMES,
    *,
    strict: bool = False,
) -> dict[str, str]:
    """Return available keyring values without ever serializing them to disk ourselves."""

    _validate_names(names)
    values: dict[str, str] = {}
    try:
        for name in names:
            value = keyring.get_password(SERVICE_NAME, name)
            if value:
                values[name] = value
    except (KeyringError, OSError) as exc:
        if strict:
            raise _unavailable_error() from exc
        return {}
    return values


def store_credentials(values: Mapping[str, str]) -> tuple[str, ...]:
    """Replace selected keyring entries and attempt rollback if a write fails."""

    names = tuple(values)
    _validate_names(names)
    normalized = {name: value.strip() for name, value in values.items()}
    if any(not value for value in normalized.values()):
        raise ConfigError("credentials cannot be empty")

    previous = read_stored_credentials(names, strict=True)
    saved: list[str] = []
    try:
        for name, value in normalized.items():
            keyring.set_password(SERVICE_NAME, name, value)
            saved.append(name)
    except (KeyringError, OSError) as exc:
        _restore_credentials(names, previous)
        raise _unavailable_error() from exc
    return tuple(saved)


def delete_stored_credentials(
    names: Sequence[str] = STORED_CREDENTIAL_NAMES,
) -> tuple[str, ...]:
    """Delete selected entries, treating an already absent entry as a no-op."""

    _validate_names(names)
    existing = read_stored_credentials(names, strict=True)
    deleted: list[str] = []
    try:
        for name in names:
            if name not in existing:
                continue
            keyring.delete_password(SERVICE_NAME, name)
            deleted.append(name)
    except (KeyringError, OSError) as exc:
        raise _unavailable_error() from exc
    return tuple(deleted)


def _restore_credentials(names: Sequence[str], previous: Mapping[str, str]) -> None:
    for name in names:
        try:
            if name in previous:
                keyring.set_password(SERVICE_NAME, name, previous[name])
            elif keyring.get_password(SERVICE_NAME, name) is not None:
                keyring.delete_password(SERVICE_NAME, name)
        except (KeyringError, OSError):
            # The original error is more useful; rollback is best-effort across backends.
            continue


def _validate_names(names: Sequence[str]) -> None:
    unknown = sorted(set(names).difference(STORED_CREDENTIAL_NAMES))
    if unknown:
        raise ConfigError("unsupported stored credential name")


def _unavailable_error() -> ConfigError:
    return ConfigError(
        "the operating-system credential store is unavailable; "
        "use environment variables for this session"
    )
