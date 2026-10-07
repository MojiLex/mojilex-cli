"""Select a provider's secret without exposing it in configuration or diagnostics."""

from .models import Credentials


def provider_credential_name(provider: str) -> str:
    names = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
    try:
        return names[provider]
    except KeyError:
        raise ValueError(f"unsupported AI provider: {provider}") from None


def provider_api_key(credentials: Credentials, provider: str) -> str | None:
    provider_credential_name(provider)
    return credentials.gemini_api_key if provider == "gemini" else credentials.openai_api_key
