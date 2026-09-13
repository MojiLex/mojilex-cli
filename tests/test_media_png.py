import hashlib
from pathlib import Path

import pytest
from PIL import Image

from media_backend_helpers import require_native_media_limits
from mojilex_cli.analysis import decoder_backend_fingerprint
from mojilex_cli.domain import Media
from mojilex_cli.media import MediaError, MediaLimitError, MediaLimits, SafeMediaWorker, worker
from mojilex_cli.media.inspect import inspect_png, sniff_format


def _png(path: Path) -> bytes:
    with Image.new("RGBA", (16, 12), (80, 120, 200, 128)) as image:
        image.save(path, format="PNG")
    return path.read_bytes()


@pytest.mark.parametrize("isolated", [False, True])
def test_static_telegram_png_preserves_original_format_hash_and_alpha(
    tmp_path: Path, isolated: bool
) -> None:
    source = tmp_path / "telegram-static.webp"
    original = _png(source)
    if isolated:
        require_native_media_limits()
        media = SafeMediaWorker().process(source, tmp_path / "out", expected_format="webp")
        metadata = media.dataset_metadata()
        analysis = media.analysis
        paths = media.frame_paths
        assert media.composition_tile_path is not None
    else:
        result = worker.process(
            source,
            tmp_path / "out",
            "webp",
            MediaLimits(),
            needs_repainting=False,
            ffmpeg="unused",
            ffprobe="unused",
            rlottie_renderer="unused",
        )
        metadata = result["metadata"]
        from mojilex_cli.analysis import DeterministicMediaAnalysis

        analysis = DeterministicMediaAnalysis.model_validate(result["analysis"])
        paths = result["frame_paths"]
        assert result["composition_tile"] is not None
    assert metadata["format"] == "png"
    assert metadata["mime_type"] == "image/png"
    assert metadata["sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["byte_size"] == len(original)
    assert source.read_bytes() == original
    assert Media.model_validate(metadata).format.value == "png"
    assert analysis.rendering.alpha_mode == "translucent"
    assert analysis.decoder_backend_fingerprint == decoder_backend_fingerprint("png")
    assert len(paths) == 1


def test_png_pixel_limit_precedes_decompression(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "image.png"
    _png(source)
    from PIL import PngImagePlugin

    monkeypatch.setattr(
        PngImagePlugin.PngImageFile, "load", lambda *a, **kw: pytest.fail("decoded oversized PNG")
    )
    with pytest.raises(MediaLimitError, match="pixel"):
        inspect_png(source, MediaLimits(max_pixels=100))


def test_png_metadata_uses_the_display_orientation(tmp_path: Path) -> None:
    source = tmp_path / "oriented.png"
    with Image.new("RGBA", (16, 12), "red") as image:
        exif = Image.Exif()
        exif[274] = 6
        image.save(source, format="PNG", exif=exif)
    assert inspect_png(source, MediaLimits()) == {"width": 12, "height": 16, "duration_ms": None}


def test_animated_png_is_not_reinterpreted_as_static(tmp_path: Path) -> None:
    source = tmp_path / "animation.png"
    with Image.new("RGBA", (8, 8), "red") as first, Image.new("RGBA", (8, 8), "blue") as second:
        first.save(
            source, format="PNG", save_all=True, append_images=[second], duration=100, loop=0
        )
    assert sniff_format(source) == "png"
    with pytest.raises(MediaError, match="animated PNG"):
        inspect_png(source, MediaLimits())


def test_truncated_png_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bad.png"
    source.write_bytes(_png(tmp_path / "valid.png")[:40])
    with pytest.raises(MediaError, match="PNG cannot be decoded"):
        inspect_png(source, MediaLimits())


@pytest.mark.parametrize("expected", ["tgs", "webm"])
def test_png_does_not_override_an_animated_source_type(tmp_path: Path, expected: str) -> None:
    source = tmp_path / "wrong.png"
    _png(source)
    with pytest.raises(MediaError, match="expected format"):
        SafeMediaWorker().process(source, tmp_path / "out", expected_format=expected)
