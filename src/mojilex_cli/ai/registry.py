"""Explicit provider registry; providers never silently fall back."""

from __future__ import annotations

from collections.abc import Callable

from .base import VisionProvider

ProviderFactory = Callable[..., VisionProvider]


class ProviderRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        if not name or name in self._factories:
            raise ValueError(f"provider is invalid or already registered: {name}")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: object) -> VisionProvider:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise ValueError(f"unknown AI provider: {name}") from exc
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_registry() -> ProviderRegistry:
    from .gemini import GeminiVisionProvider

    registry = ProviderRegistry()
    registry.register("gemini", GeminiVisionProvider)
    return registry
