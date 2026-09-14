import shutil
import struct
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from mojilex_cli.media import MediaLimits, worker
from mojilex_cli.media.webm_alpha import _alpha_packets, decode_separate_alpha


def element(tag, payload):
    size = len(payload)
    assert size < 127
    return tag.to_bytes((tag.bit_length() + 7) // 8, "big") + bytes([128 + size]) + payload


def fixture(*, flags=0, track=1, alpha_id=1):
    tracks = element(0x1654AE6B, element(0xAE, element(0xD7, b"\x01") + element(0x86, b"V_VP9")))
    block = element(0xA1, bytes([128 + track]) + struct.pack(">hB", 33, flags) + b"color")
    alpha = element(
        0x75A1, element(0xA6, element(0xEE, bytes([alpha_id])) + element(0xA5, b"alpha"))
    )
    cluster = element(0x1F43B675, element(0xE7, b"\x00") + element(0xA0, block + alpha))
    return element(0x18538067, tracks + cluster)


def test_alpha_packets_retain_original_bytes_and_timestamp():
    assert _alpha_packets(fixture()) == [(33000, b"alpha")]


@pytest.mark.parametrize("kwargs", [dict(flags=2), dict(track=2), dict(alpha_id=2)])
def test_ambiguous_alpha_mapping_is_rejected(kwargs):
    with pytest.raises(ValueError):
        _alpha_packets(fixture(**kwargs))


def test_truncated_alpha_block_is_rejected():
    with pytest.raises(ValueError):
        _alpha_packets(fixture()[:-1])


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)
def test_separate_alpha_preserves_pixels_and_all_frames(tmp_path: Path, monkeypatch):
    path = tmp_path / "source.webm"
    pixels = bytes((255, 0, 0, 0)) * 128 + bytes((0, 255, 0, 128)) * 128
    encoded = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            "16x16",
            "-r",
            "10",
            "-i",
            "pipe:0",
            "-c:v",
            "libvpx-vp9",
            "-lossless",
            "1",
            "-an",
            str(path),
        ],
        input=pixels * 3,
        capture_output=True,
        timeout=20,
    )
    assert encoded.returncode == 0
    reference = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-c:v",
            "libvpx-vp9",
            "-i",
            str(path),
            "-fps_mode",
            "passthrough",
            "-vf",
            "alphaextract",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ],
        capture_output=True,
        timeout=20,
    )
    assert reference.returncode == 0
    assert len(reference.stdout) == 3 * 256
    limits = MediaLimits(frames=4)
    info = worker.inspect_webm(path, limits)
    output = tmp_path / "separate"
    output.mkdir()
    result = decode_separate_alpha(
        path,
        output,
        info,
        limits,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        needs_repainting=False,
        analyze=True,
    )
    assert result is not None
    frames, analysis = result
    try:
        assert len(frames) == 4
        assert analysis.analysis_scope == "full-decoded-stream"
        assert analysis.rendering.alpha_mode == "translucent"
        for frame in frames:
            assert frame.getpixel((0, 0))[3] == 0
            assert 128 <= frame.getpixel((0, 15))[3] <= 129
            assert frame.getchannel("A").tobytes() == reference.stdout[:256]
    finally:
        for frame in frames:
            frame.close()
    assert not (output / "original-alpha.ivf").exists()

    def broken(*a, **kw):
        raise RuntimeError("synthetic libvpx failure")

    monkeypatch.setattr(worker, "_render_webm", broken)
    processed = worker.process(
        path,
        tmp_path / "pipeline",
        "webm",
        limits,
        needs_repainting=False,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        rlottie_renderer="unused",
    )
    assert (
        processed["analysis"]["decoder_backend_fingerprint"] == analysis.decoder_backend_fingerprint
    )
    with Image.open(processed["frame_paths"][0]) as preview:
        assert preview.size == (256, 256)


def test_fallback_fingerprint_is_accepted_by_resume_cache(monkeypatch):
    from types import SimpleNamespace

    from mojilex_cli.media.webm_alpha import separate_alpha_fingerprint
    from mojilex_cli.pipeline import runner

    monkeypatch.setattr(runner, "decoder_backend_fingerprint", lambda *a, **kw: "a" * 64)
    processor = SimpleNamespace(worker=SimpleNamespace(ffmpeg="ffmpeg", ffprobe="ffprobe"))
    assert separate_alpha_fingerprint("a" * 64) in runner._decoder_backend_candidates(
        "webm", processor
    )
