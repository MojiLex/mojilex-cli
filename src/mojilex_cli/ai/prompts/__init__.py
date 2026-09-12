"""Task-local selection of immutable prompt versions for exact cache provenance."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from types import ModuleType
from typing import Any

from . import v1_1_0, v1_2_0

PROMPT_VERSION = v1_2_0.PROMPT_VERSION
SYSTEM_PROMPT = v1_2_0.SYSTEM_PROMPT
_VERSIONS = {"1.1.0": v1_1_0, "1.2.0": v1_2_0}
_PROMPT_VERSION: ContextVar[str] = ContextVar("mojilex_prompt_version", default=PROMPT_VERSION)


def current_prompt_version() -> str:
    return _PROMPT_VERSION.get()


@contextmanager
def use_prompt_version(version: str) -> Iterator[None]:
    if version not in _VERSIONS:
        raise ValueError("unsupported prompt version")
    token = _PROMPT_VERSION.set(version)
    try:
        yield
    finally:
        _PROMPT_VERSION.reset(token)


def _current() -> ModuleType:
    return _VERSIONS[current_prompt_version()]


def prompt_templates() -> dict[str, Any]:
    return _current().prompt_templates()  # type: ignore[no-any-return]


def prompt_manifest() -> dict[str, Any]:
    return _current().prompt_manifest()  # type: ignore[no-any-return]


def prompt_template_bytes() -> bytes:
    return _current().prompt_template_bytes()  # type: ignore[no-any-return]


def prompt_sha256() -> str:
    return _current().prompt_sha256()  # type: ignore[no-any-return]


def prompt_manifest_sha256() -> str:
    return _current().prompt_manifest_sha256()  # type: ignore[no-any-return]


def gemini_request_parameters() -> dict[str, Any]:
    return _current().gemini_request_parameters()  # type: ignore[no-any-return]


def gemini_request_parameters_sha256() -> str:
    return _current().gemini_request_parameters_sha256()  # type: ignore[no-any-return]


def build_prompt(expected_labels: tuple[str, ...], *, animated: bool) -> str:
    return _current().build_prompt(expected_labels, animated=animated)  # type: ignore[no-any-return]


def build_context_prompt(context_json: str) -> str:
    return _current().build_context_prompt(context_json)  # type: ignore[no-any-return]


__all__ = [
    "PROMPT_VERSION",
    "SYSTEM_PROMPT",
    "build_context_prompt",
    "build_prompt",
    "current_prompt_version",
    "gemini_request_parameters",
    "gemini_request_parameters_sha256",
    "prompt_manifest",
    "prompt_manifest_sha256",
    "prompt_sha256",
    "prompt_template_bytes",
    "prompt_templates",
    "use_prompt_version",
]
