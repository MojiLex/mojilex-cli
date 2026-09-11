from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from urllib.parse import quote

from tools.verify_no_secrets import tracked_secret_findings


def test_tracked_secret_scan_detects_values_without_echoing_them(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, shell=False)
    (tmp_path / "safe.txt").write_text(
        "synthetic fixture 12345:abcdefghijklmnopqrstuvwxyz\n", encoding="utf-8"
    )
    (tmp_path / "unsafe.txt").write_text("github_pat_" + "A" * 40 + "\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "safe.txt", "unsafe.txt"],
        check=True,
        shell=False,
    )

    findings = tracked_secret_findings(tmp_path)

    assert findings == (("unsafe.txt", "github-token"),)


def test_tracked_secret_scan_detects_openai_and_bearer_credentials(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, shell=False)
    (tmp_path / "openai.txt").write_text(
        "OPENAI_API_KEY=sk-proj-" + "A" * 32 + "\n", encoding="utf-8"
    )
    (tmp_path / "bearer.txt").write_text(
        "Authorization: Bearer " + "B" * 32 + "\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "openai.txt", "bearer.txt"],
        check=True,
        shell=False,
    )

    findings = tracked_secret_findings(tmp_path)

    assert findings == (
        ("bearer.txt", "bearer-token"),
        ("openai.txt", "openai-api-key"),
    )


def test_tracked_secret_scan_allows_safe_credential_placeholders(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, shell=False)
    (tmp_path / "placeholders.txt").write_text(
        "OPENAI_API_KEY=<set-in-environment>\n"
        "example prefix: sk-placeholder\n"
        "Authorization: Bearer <token>\n"
        "Authorization: Bearer redacted\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "placeholders.txt"],
        check=True,
        shell=False,
    )

    assert tracked_secret_findings(tmp_path) == ()


def test_tracked_secret_scan_detects_base64_encoded_credentials(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, shell=False)
    secret = ("github_pat_" + "A" * 40).encode("ascii")
    (tmp_path / "encoded.txt").write_bytes(base64.b64encode(secret) + b"\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "encoded.txt"],
        check=True,
        shell=False,
    )

    assert tracked_secret_findings(tmp_path) == (("encoded.txt", "github-token"),)


def test_tracked_secret_scan_detects_url_encoded_credentials(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, shell=False)
    secret = "123456:" + "A" * 32
    (tmp_path / "encoded.txt").write_text(quote(secret, safe="") + "\n", encoding="ascii")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "encoded.txt"],
        check=True,
        shell=False,
    )

    assert tracked_secret_findings(tmp_path) == (("encoded.txt", "telegram-bot-token"),)
