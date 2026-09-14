from __future__ import annotations

import os
from pathlib import Path

import pytest

from mojilex_cli.analysis import (
    AnalysisError,
    backend,
    decoder_backend_fingerprint,
    webm_backend_fingerprints,
)


@pytest.fixture
def executables(tmp_path, monkeypatch):
    selected = {"ffmpeg": tmp_path / "ffmpeg", "ffprobe": tmp_path / "ffprobe"}
    selected["ffmpeg"].write_bytes(b"ffmpeg executable version one")
    selected["ffprobe"].write_bytes(b"ffprobe executable version one")
    monkeypatch.setattr(backend.shutil, "which", lambda command: str(selected[command]))
    return selected


def test_webm_batch_matches_six_original_variants_and_reads_each_binary_once(
    executables, monkeypatch
):
    reads = []
    original_open = Path.open

    def opened(path, *args, **kwargs):
        if path in executables.values() and args and args[0] == "rb":
            reads.append(path.name)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    expected = tuple(
        decoder_backend_fingerprint("webm", webm_codec=codec, webm_preserve_alpha=alpha)
        for codec in ("av1", "vp8", "vp9")
        for alpha in (False, True)
    )
    assert reads == ["ffmpeg", "ffprobe"] * 6
    reads.clear()
    assert webm_backend_fingerprints() == expected
    assert reads == ["ffmpeg", "ffprobe"]


@pytest.mark.parametrize("name", ["ffmpeg", "ffprobe"])
def test_webm_batch_rechecks_exact_bytes_between_probes_even_with_same_stat(executables, name):
    before = webm_backend_fingerprints()
    path = executables[name]
    info = path.stat()
    contents = path.read_bytes()
    path.write_bytes(contents[:-3] + b"two")
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert path.stat().st_size == info.st_size
    assert path.stat().st_mtime_ns == info.st_mtime_ns
    after = webm_backend_fingerprints()
    assert all(previous != current for previous, current in zip(before, after, strict=True))


def test_webm_batch_does_not_reuse_probe_after_decoder_disappears(executables):
    assert len(webm_backend_fingerprints()) == 6
    executables["ffprobe"].unlink()
    with pytest.raises(AnalysisError, match="cannot be fingerprinted"):
        webm_backend_fingerprints()
