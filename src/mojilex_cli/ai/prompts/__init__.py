"""Task-local selection of immutable prompt versions for exact cache provenance."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from threading import RLock
from types import ModuleType
from typing import Any

from mojilex_cli.domain.hashes import jcs_sha256

from . import v1_1_0, v1_2_0, v1_2_1

PROMPT_VERSION = v1_2_1.PROMPT_VERSION
SYSTEM_PROMPT = v1_2_1.SYSTEM_PROMPT
_VERSIONS = {"1.1.0": v1_1_0, "1.2.0": v1_2_0, "1.2.1": v1_2_1}
_PROMPT_VERSION: ContextVar[str] = ContextVar("mojilex_prompt_version", default=PROMPT_VERSION)


class _PromptContracts:
    def __init__(self) -> None:
        self.values: dict[tuple[object, ...], Any] = {}
        self.lock = RLock()


_CONTRACTS: ContextVar[_PromptContracts | None] = ContextVar(
    "mojilex_prompt_contracts", default=None
)


@contextmanager
def prompt_contract_scope() -> Iterator[None]:
    """Reuse immutable contracts within one operation, including its child tasks."""
    if _CONTRACTS.get() is not None:
        yield
        return
    token = _CONTRACTS.set(_PromptContracts())
    try:
        yield
    finally:
        _CONTRACTS.reset(token)


def _contract(name: str) -> Any:
    module = _current()
    function = getattr(module, name)
    cache = _CONTRACTS.get()
    if cache is None:
        return function()
    key = (current_prompt_version(), function)
    with cache.lock:
        if key not in cache.values:
            cache.values[key] = deepcopy(function())
        # Callers may modify transport dictionaries; never expose cached objects.
        return deepcopy(cache.values[key])


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
    return _contract("prompt_templates")  # type: ignore[no-any-return]


def prompt_manifest() -> dict[str, Any]:
    return _contract("prompt_manifest")  # type: ignore[no-any-return]


def prompt_template_bytes() -> bytes:
    return _contract("prompt_template_bytes")  # type: ignore[no-any-return]


def prompt_sha256() -> str:
    return _contract("prompt_sha256")  # type: ignore[no-any-return]


def prompt_manifest_sha256() -> str:
    return _contract("prompt_manifest_sha256")  # type: ignore[no-any-return]


def gemini_request_parameters() -> dict[str, Any]:
    return _contract("gemini_request_parameters")  # type: ignore[no-any-return]


def gemini_request_parameters_sha256() -> str:
    module = _current()
    cache = _CONTRACTS.get()
    if cache is None:
        return module.gemini_request_parameters_sha256()  # type: ignore[no-any-return]
    key = (
        current_prompt_version(),
        module.gemini_request_parameters,
        module.gemini_request_parameters_sha256,
    )
    with cache.lock:
        if key not in cache.values:
            # Every supported immutable version defines this digest as JCS of
            # the complete parameters. Reuse exactly the cached transport body.
            cache.values[key] = jcs_sha256(gemini_request_parameters())
        return str(cache.values[key])


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
    "prompt_contract_scope",
    "prompt_manifest",
    "prompt_manifest_sha256",
    "prompt_sha256",
    "prompt_template_bytes",
    "prompt_templates",
    "use_prompt_version",
]
