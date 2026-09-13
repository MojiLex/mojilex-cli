"""Real isolated-decoder prerequisites, checked without reading media."""

from functools import cache

import pytest

from mojilex_cli.media.sandbox import hard_resource_limits_available


@cache
def native_media_limits_available() -> bool:
    return hard_resource_limits_available()


def require_native_media_limits() -> None:
    if not native_media_limits_available():
        pytest.skip("native media unavailable: OS rejected the exact hard worker memory limits")
