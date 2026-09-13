import hashlib
from pathlib import Path

import pytest
from PIL import Image

from mojilex_cli.media.models import MediaLimits, ProcessedMedia
from mojilex_cli.media.resume import RetainedMediaStore
from mojilex_cli.media.sandbox import SafeMediaWorker


def _process(tmp_path: Path, *, size: int = 100, repaint: bool = False) -> ProcessedMedia:
    source = tmp_path / "tile.webp"
    with Image.new("RGBA", (size, size), (10, 30, 90, 128)) as image:
        image.putpixel((0, 0), (200, 100, 40, 255))
        image.save(source, format="WEBP", lossless=True)
    return SafeMediaWorker(MediaLimits()).process(
        source, tmp_path / "output", expected_format="webp", needs_repainting=repaint
    )


def test_isolated_worker_preserves_native_tile_and_alpha(tmp_path: Path) -> None:
    value = _process(tmp_path)
    assert value.composition_tile_path is not None
    tile = value.composition_tile_path
    assert value.composition_tile_sha256 == hashlib.sha256(tile.read_bytes()).hexdigest()
    with Image.open(tile) as image:
        assert image.size == (100, 100)
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0)) == (200, 100, 40, 255)
        assert image.getpixel((50, 50)) == (10, 30, 90, 128)
    with Image.open(value.frame_paths[0]) as preview:
        assert preview.size == (256, 256)
        assert preview.mode == "RGB"
    assert "composition_tile_path" not in value.model_dump()
    assert "composition_tile_sha256" not in value.model_dump()


@pytest.mark.parametrize("size,repaint", [(257, False), (100, True)])
def test_unsupported_tile_is_skipped(tmp_path: Path, size: int, repaint: bool) -> None:
    value = _process(tmp_path, size=size, repaint=repaint)
    assert value.composition_tile_path is None
    assert value.composition_tile_sha256 is None


def test_retained_tile_roundtrip_and_corruption(tmp_path: Path) -> None:
    value = _process(tmp_path)
    store = RetainedMediaStore(tmp_path / "retained", 2_000_000)
    key = "a" * 64
    assert store.put(key, value)
    expected = ProcessedMedia(
        metadata=value.metadata,
        analysis=value.analysis,
        frame_paths=(),
        rendered_frame_count=1,
        has_dark_render=value.semantic_has_dark_render,
    )
    restored = store.get(key, expected)
    assert restored is not None
    assert restored.composition_tile_path is not None
    assert value.composition_tile_path is not None
    assert restored.composition_tile_path.read_bytes() == value.composition_tile_path.read_bytes()
    assert restored.composition_tile_sha256 == value.composition_tile_sha256
    restored.composition_tile_path.write_bytes(b"corrupt")
    assert store.get(key, expected) is None
