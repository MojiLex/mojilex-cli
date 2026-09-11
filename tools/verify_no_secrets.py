"""Fail CI when a tracked file contains a credential-shaped literal.

Only the exact synthetic values used by redaction tests are ignored. Matches are
reported by path and rule name, never by value, so CI logs cannot amplify a leak.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("telegram-bot-token", re.compile(rb"(?<![0-9])[0-9]{5,12}:[A-Za-z0-9_-]{20,}")),
    (
        "github-token",
        re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ),
    ("google-api-key", re.compile(rb"\bAIza[0-9A-Za-z_-]{30,}\b")),
    ("openai-api-key", re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    (
        "bearer-token",
        re.compile(rb"\bBearer[ \t]+[A-Za-z0-9._~+/-]{20,}={0,2}\b", re.IGNORECASE),
    ),
    ("aws-access-key", re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("slack-token", re.compile(rb"\bxox[baprs]-[0-9A-Za-z-]{20,}\b")),
    (
        "credential-url",
        re.compile(rb"https?://[^\s/@:]+:[^\s/@]+@[^\s/]+", re.IGNORECASE),
    ),
    (
        "private-key",
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
)

_ALLOWED_SYNTHETIC_VALUES = (
    b"12345:abcdefghijklmnopqrstuvwxyz",
    b"123456:telegram-secret-value-value",
    b"123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",
    b"123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij",
    b"github_pat_abcdefghijklmnopqrstuvwxyz123456",
)


def tracked_secret_findings(root: Path) -> tuple[tuple[str, str], ...]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        shell=False,
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise RuntimeError("git ls-files failed; refusing to skip the repository secret scan")
    findings: list[tuple[str, str]] = []
    for raw_relative in completed.stdout.split(b"\0"):
        if not raw_relative:
            continue
        relative = raw_relative.decode("utf-8", errors="strict")
        path = (root / relative).resolve()
        if not path.is_file() or root.resolve() not in path.parents:
            raise RuntimeError(f"tracked path is missing or escaped the repository: {relative}")
        payload = path.read_bytes()
        for fixture in _ALLOWED_SYNTHETIC_VALUES:
            payload = payload.replace(fixture, b"[synthetic-redaction-fixture]")
        # Credential URLs targeting the reserved test domain are intentionally synthetic.
        payload = re.sub(
            rb"https?://[^\s/@:]+:[^\s/@]+@example\.test(?:/[^\s]*)?",
            b"https://example.test/fixture",
            payload,
            flags=re.IGNORECASE,
        )
        for name, pattern in _PATTERNS:
            if pattern.search(payload):
                findings.append((relative.replace("\\", "/"), name))
    return tuple(sorted(set(findings)))


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    try:
        findings = tracked_secret_findings(root)
    except (OSError, RuntimeError, subprocess.TimeoutExpired, UnicodeDecodeError) as exc:
        print(f"secret scan failed closed: {exc}", file=sys.stderr)
        return 2
    if findings:
        for path, rule in findings:
            print(f"potential secret: {path} ({rule})", file=sys.stderr)
        return 1
    print("tracked-file secret scan passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
