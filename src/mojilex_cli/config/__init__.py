"""Configuration API."""

from .loader import (
    default_cache_dir,
    default_runs_dir,
    default_user_config_path,
    load_config,
    load_config_file,
    load_credentials,
    safe_config_dict,
)
from .models import (
    AIConfig,
    ConfigError,
    Credentials,
    DedupeConfig,
    GitIdentityConfig,
    MojiLexConfig,
    ProcessingConfig,
    RepositoryConfig,
    TelegramConfig,
)
from .secrets import redact_mapping, redact_text, url_has_credentials

__all__ = [
    "AIConfig",
    "ConfigError",
    "Credentials",
    "DedupeConfig",
    "GitIdentityConfig",
    "MojiLexConfig",
    "ProcessingConfig",
    "RepositoryConfig",
    "TelegramConfig",
    "default_cache_dir",
    "default_runs_dir",
    "default_user_config_path",
    "load_config",
    "load_config_file",
    "load_credentials",
    "redact_mapping",
    "redact_text",
    "safe_config_dict",
    "url_has_credentials",
]
