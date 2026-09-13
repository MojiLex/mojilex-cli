import hashlib
import io
import math
import random
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from pydantic import ValidationError

from mojilex_cli.composition.detector import (
    MAX_TILES,
    Composition,
    Member,
    Tile,
    assemble,
    candidates,
    load_tile,
    seam_score,
)


def _tile(image: Image.Image, index: int) -> Tile:
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    return Tile(Member(native_id=str(index), media_sha256=digest, tile_sha256=digest), image)


def _texture(seed: int, size: tuple[int, int]) -> Image.Image:
    rng = random.Random(seed)
    coarse = Image.new("RGB", (13, 9))
    coarse.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(117)])
    return coarse.resize(size, Image.Resampling.BICUBIC).convert("RGBA")


def _split(seed: int = 19) -> tuple[Image.Image, list[Tile]]:
    full = _texture(seed, (192, 128))
    return full, [
        _tile(full.crop((x * 64, y * 64, (x + 1) * 64, (y + 1) * 64)), y * 3 + x + 1)
        for y in range(2)
        for x in range(3)
    ]


def test_continuous_texture_reassembles_without_pixel_changes() -> None:
    full, tiles = _split()
    found = candidates(tiles)
    assert len(found) == 1
    group = found[0]
    assert (group.columns, group.rows) == (3, 2)
    assert [member.native_id for member in group.members] == [str(i) for i in range(1, 7)]
    assert not group.verified and group.verifier_model is None
    data = assemble(group, {tile.member.native_id: tile for tile in tiles})
    with Image.open(io.BytesIO(data)) as assembled:
        assert assembled.size == full.size
        assert assembled.tobytes() == full.tobytes()


@pytest.mark.parametrize("seed", range(10))
def test_input_order_does_not_invent_another_picture(seed: int) -> None:
    _, tiles = _split()
    random.Random(seed).shuffle(tiles)
    found = candidates(tiles)
    assert len(found) == 1
    assert [member.native_id for member in found[0].members] == [str(i) for i in range(1, 7)]


@pytest.mark.parametrize("seed", range(20))
def test_unrelated_textured_holdouts_have_zero_candidates(seed: int) -> None:
    tiles = [_tile(_texture(seed * 100 + i, (64, 64)), i + 1) for i in range(12)]
    assert candidates(tiles) == []


@pytest.mark.parametrize("background", [(0, 0, 0, 0), (242, 242, 242, 255)])
@pytest.mark.parametrize("kind", ["letters", "numbers", "icons", "flat"])
def test_normal_symbols_and_blank_borders_are_not_fragment_evidence(background, kind) -> None:
    tiles = []
    for index in range(12):
        image = Image.new("RGBA", (64, 64), background)
        draw = ImageDraw.Draw(image)
        if kind == "letters":
            draw.text((25, 25), chr(ord("A") + index), fill="black")
        elif kind == "numbers":
            draw.text((25, 25), str(index), fill="black")
        elif kind == "icons":
            draw.ellipse((10, 12, 53, 50), fill=(index * 20, 50, 180, 255))
        tiles.append(_tile(image, index + 1))
    assert candidates(tiles) == []


def test_repeated_textured_tiles_are_ambiguous_and_not_assembled() -> None:
    _, original = _split()
    repeated = [_tile(tile.image.copy(), index + 1) for index, tile in enumerate(original * 2)]
    assert candidates(repeated) == []


def test_input_count_and_duplicate_identity_are_rejected() -> None:
    _, tiles = _split()
    assert candidates(tiles[:1]) == []
    assert candidates([tiles[0]] * (MAX_TILES + 1)) == []
    assert candidates([tiles[0], *tiles]) == []


def test_transparency_and_featureless_seams_cannot_establish_relationship() -> None:
    varied = [(i * 5, i * 5, i * 5, 255) for i in range(48)]
    transparent = [(r, g, b, 0) for r, g, b, _ in varied]
    assert math.isinf(seam_score(transparent, transparent))
    assert math.isinf(seam_score(varied, transparent))
    flat = [(100, 100, 100, 255)] * 48
    assert math.isinf(seam_score(flat, flat))
    assert math.isinf(seam_score(varied, flat))
    assert seam_score(varied, varied) == 0


@pytest.mark.parametrize("change", ["shape", "duplicate", "verified", "coerced", "extra"])
def test_malformed_composition_is_rejected(change: str) -> None:
    _, tiles = _split()
    raw = candidates(tiles)[0].model_dump()
    if change == "shape":
        raw["rows"] = 3
    elif change == "duplicate":
        raw["members"][0] = raw["members"][1]
    elif change == "verified":
        raw["verified"] = True
    elif change == "coerced":
        raw["verified"] = "true"
    else:
        raw["other"] = "unknown"
    with pytest.raises(ValidationError):
        Composition.model_validate(raw)


def _png(path: Path, *, mode="RGBA", size=(64, 64), animated=False) -> str:
    image = Image.new(mode, size, "red")
    if animated:
        image.save(
            path,
            format="PNG",
            save_all=True,
            append_images=[Image.new(mode, size, "blue")],
            duration=100,
        )
    else:
        image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "case",
    [
        "rgb",
        "nonsquare",
        "small",
        "large",
        "animated",
        "wronghash",
        "missing",
        "oversized",
        "invalid",
    ],
)
def test_load_tile_rejects_unsafe_input(tmp_path: Path, case: str) -> None:
    path = tmp_path / "tile.png"
    kwargs = {}
    if case == "rgb":
        kwargs["mode"] = "RGB"
    elif case == "nonsquare":
        kwargs["size"] = (64, 63)
    elif case == "small":
        kwargs["size"] = (31, 31)
    elif case == "large":
        kwargs["size"] = (257, 257)
    elif case == "animated":
        kwargs["animated"] = True
    digest = _png(path, **kwargs)
    if case == "wronghash":
        digest = "0" * 64
    elif case == "missing":
        path.unlink()
    elif case in {"oversized", "invalid"}:
        path.write_bytes(b"x" * (512 * 1024 + 1 if case == "oversized" else 10))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_tile("123", "a" * 64, path, digest) is None


def test_load_tile_validates_identity_and_preserves_rgba(tmp_path: Path) -> None:
    path = tmp_path / "tile.png"
    digest = _png(path)
    tile = load_tile("123", "a" * 64, path, digest)
    assert tile is not None and tile.image.size == (64, 64) and tile.image.mode == "RGBA"
    assert tile.member.tile_sha256 == digest
    assert load_tile("not-an-id", "a" * 64, path, digest) is None
    assert load_tile("123", "invalid", path, digest) is None


def test_load_tile_rejects_symlink_and_linked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    source = real / "tile.png"
    digest = _png(source)
    link = tmp_path / "linked.png"
    parent = tmp_path / "linked-parent"
    try:
        link.symlink_to(source)
        parent.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink privilege unavailable")
    assert load_tile("123", "a" * 64, link, digest) is None
    assert load_tile("123", "a" * 64, parent / "tile.png", digest) is None
