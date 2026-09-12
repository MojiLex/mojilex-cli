"""Decoder provenance must identify the same runtime inside the safe worker."""

import os
import subprocess
import sys

from mojilex_cli.analysis import decoder_backend_fingerprint
from mojilex_cli.media.sandbox import _worker_environment


def test_worker_fingerprint_matches_parent_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "synthetic-do-not-forward")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-do-not-forward")
    environment = _worker_environment(tmp_path)
    assert "TELEGRAM_BOT_TOKEN" not in environment
    assert "GEMINI_API_KEY" not in environment
    for name in ("PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432"):
        if name in os.environ:
            assert environment.get(name) == os.environ[name]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from mojilex_cli.analysis import decoder_backend_fingerprint; "
            "print(decoder_backend_fingerprint('webp'))",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert completed.stdout.strip() == decoder_backend_fingerprint("webp")
