"""Apply Unix limits in a fresh interpreter before importing any media decoder.

This file is executed directly: importing the media package first would load the
decoder stack before its limits, and preexec_fn inherits the parent's memory and
thread state until exec (which can already exceed the macOS memory limit).
"""

from __future__ import annotations

import sys
from typing import Any

HARD_MAX_WORKER_MEMORY = 512 * 1024 * 1024


def apply_limits(memory_bytes: int) -> None:
    import resource

    resource_api: Any = resource
    if not 0 < memory_bytes <= HARD_MAX_WORKER_MEMORY:
        raise ValueError("invalid worker memory limit")
    limits = {
        "RLIMIT_AS": memory_bytes,
        "RLIMIT_DATA": memory_bytes,
        "RLIMIT_CPU": 30,
        "RLIMIT_FSIZE": 64 * 1024 * 1024,
        "RLIMIT_NOFILE": 64,
    }
    for name, requested in limits.items():
        kind = getattr(resource_api, name)
        soft, hard = resource_api.getrlimit(kind)
        # Respect stricter limits inherited from the launcher or host.
        bound = min(
            [
                requested,
                *(value for value in (soft, hard) if value != resource_api.RLIM_INFINITY),
            ]
        )
        try:
            resource_api.setrlimit(kind, (bound, bound))
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"cannot apply worker limit {name}") from exc


def main() -> None:
    try:
        apply_limits(int(sys.argv[1]))
    except (ImportError, IndexError, OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"isolated media worker limits unavailable: {exc}")
        raise SystemExit(1) from None
    if sys.argv[2:] == ["--probe"]:
        sys.stdout.write("MOJILEX_RESOURCE_LIMITS_OK\n")
        return
    # Import only after all limits succeeded. No untrusted media is read earlier.
    import runpy

    sys.argv = ["mojilex_cli.media.worker", *sys.argv[2:]]
    runpy.run_module("mojilex_cli.media.worker", run_name="__main__")


if __name__ == "__main__":
    main()
