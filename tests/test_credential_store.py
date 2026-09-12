from __future__ import annotations

from pathlib import Path

import pytest
from keyring.errors import PasswordSetError

from mojilex_cli.config import credential_store
from mojilex_cli.config.loader import load_credentials
from mojilex_cli.config.models import ConfigError


def test_stored_credentials_fill_only_missing_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        credential_store,
        "read_stored_credentials",
        lambda _names: {
            "TELEGRAM_BOT_TOKEN": "stored-telegram",
            "GEMINI_API_KEY": "stored-gemini",
        },
    )

    credentials = load_credentials()
    overridden = load_credentials(
        {
            "TELEGRAM_BOT_TOKEN": "environment-telegram",
            "GEMINI_API_KEY": "environment-gemini",
        }
    )

    assert credentials.telegram_bot_token == "stored-telegram"
    assert credentials.gemini_api_key == "stored-gemini"
    assert overridden.telegram_bot_token == "environment-telegram"
    assert overridden.gemini_api_key == "environment-gemini"


def test_store_and_delete_credentials_never_return_secret_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(
        credential_store.keyring,
        "get_password",
        lambda service, name: backend.get((service, name)),
    )
    monkeypatch.setattr(
        credential_store.keyring,
        "set_password",
        lambda service, name, value: backend.__setitem__((service, name), value),
    )
    monkeypatch.setattr(
        credential_store.keyring,
        "delete_password",
        lambda service, name: backend.pop((service, name)),
    )

    saved = credential_store.store_credentials(
        {"GEMINI_API_KEY": "gemini-test-value", "TELEGRAM_BOT_TOKEN": "telegram-test-value"}
    )
    deleted = credential_store.delete_stored_credentials()

    assert saved == ("GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN")
    assert deleted == ("TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY")
    assert backend == {}
    assert "test-value" not in repr((saved, deleted))


def test_store_rolls_back_when_keyring_rejects_a_later_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = {
        (credential_store.SERVICE_NAME, "TELEGRAM_BOT_TOKEN"): "old-telegram",
    }

    def set_password(service: str, name: str, value: str) -> None:
        if name == "GEMINI_API_KEY" and value == "new-gemini":
            raise PasswordSetError("synthetic failure")
        backend[(service, name)] = value

    monkeypatch.setattr(
        credential_store.keyring,
        "get_password",
        lambda service, name: backend.get((service, name)),
    )
    monkeypatch.setattr(credential_store.keyring, "set_password", set_password)
    monkeypatch.setattr(
        credential_store.keyring,
        "delete_password",
        lambda service, name: backend.pop((service, name)),
    )

    with pytest.raises(ConfigError, match="credential store is unavailable"):
        credential_store.store_credentials(
            {"TELEGRAM_BOT_TOKEN": "new-telegram", "GEMINI_API_KEY": "new-gemini"}
        )

    assert backend == {(credential_store.SERVICE_NAME, "TELEGRAM_BOT_TOKEN"): "old-telegram"}


def test_windows_credential_api_os_error_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        credential_store.keyring,
        "get_password",
        lambda *_args: (_ for _ in ()).throw(OSError(1312, "synthetic backend detail")),
    )

    with pytest.raises(ConfigError) as captured:
        credential_store.read_stored_credentials(strict=True)

    assert "operating-system credential store is unavailable" in str(captured.value)
    assert "1312" not in str(captured.value)
    assert "synthetic backend detail" not in str(captured.value)


def test_windows_uninstaller_script_is_packaged_source() -> None:
    script = (
        Path(__file__).parents[1] / "src" / "mojilex_cli" / "installers" / "uninstall_windows.ps1"
    )
    assert script.is_file()
    content = script.read_text(encoding="utf-8")
    assert "& $UvPath tool uninstall mojilex-cli" in content
    assert "Refusing to remove unexpected data directory" in content
