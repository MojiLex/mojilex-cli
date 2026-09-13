import hashlib
import math
import random

import pytest
from PIL import Image, ImageDraw, ImageFont

from mojilex_cli.composition.detector import Composition, Member, Tile, candidates


def _tile(image: Image.Image, index: int) -> Tile:
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    return Tile(Member(native_id=str(index), media_sha256=digest, tile_sha256=digest), image)


def _split(columns: int, rows: int, *, noise: int = 0) -> list[Tile]:
    rng = random.Random(7719)
    coarse = Image.new("RGB", (columns * 4 + 1, rows * 4 + 1))
    coarse.putdata(
        [tuple(rng.randrange(256) for _ in range(3)) for _ in range(coarse.width * coarse.height)]
    )
    full = coarse.resize((columns * 64, rows * 64), Image.Resampling.BICUBIC).convert("RGBA")
    if noise:
        full.putdata(
            [
                (
                    *[max(0, min(255, value + rng.randint(-noise, noise))) for value in pixel[:3]],
                    255,
                )
                for y in range(full.height)
                for x in range(full.width)
                for pixel in [full.getpixel((x, y))]
            ]
        )
    return [
        _tile(full.crop((x * 64, y * 64, (x + 1) * 64, (y + 1) * 64)), y * columns + x + 1)
        for y in range(rows)
        for x in range(columns)
    ]


def _layout(group: Composition):
    return group.columns, group.rows, tuple(member.native_id for member in group.members)


@pytest.mark.parametrize("shape", [(2, 1), (1, 2), (1, 4), (4, 1)])
def test_complete_horizontal_and_vertical_strips(shape) -> None:
    tiles = _split(*shape)
    found = candidates(tiles)
    assert (shape[0], shape[1], tuple(str(i + 1) for i in range(len(tiles)))) in {
        _layout(group) for group in found
    }
    assert all(not group.verified for group in found)


@pytest.mark.parametrize("noise", [8, 16, 24])
def test_fine_pixel_noise_does_not_hide_a_continuous_scene(noise: int) -> None:
    found = candidates(_split(3, 2, noise=noise))
    assert (3, 2, ("1", "2", "3", "4", "5", "6")) in {_layout(group) for group in found}


def test_dark_thin_cutout_continuation_with_most_of_edge_transparent() -> None:
    full = Image.new("RGBA", (128, 64))
    for y in range(24, 32):
        for x in range(128):
            value = round(8 + x * 0.30 + 18 * (math.sin(y * 0.7) + 1))
            full.putpixel((x, y), (value, value, value, 255))
    tiles = [_tile(full.crop((x * 64, 0, (x + 1) * 64, 64)), x + 1) for x in range(2)]
    assert (2, 1, ("1", "2")) in {_layout(group) for group in candidates(tiles)}


@pytest.mark.parametrize("seed", range(12))
def test_layout_search_is_independent_of_input_order(seed: int) -> None:
    tiles = _split(3, 2, noise=16)
    expected = {_layout(group) for group in candidates(tiles)}
    random.Random(seed).shuffle(tiles)
    assert {_layout(group) for group in candidates(tiles)} == expected


@pytest.mark.parametrize("kind", ["letters", "numbers", "icons"])
def test_unrelated_full_canvas_symbols_are_not_merged(kind: str) -> None:
    images = []
    for index in range(12):
        image = Image.new("RGBA", (64, 64), "white")
        draw = ImageDraw.Draw(image)
        if kind == "icons":
            # Full-height, independent geometric signs with plain shared borders.
            draw.polygon([(32, 0), (60, 32), (32, 63), (4, 32)], fill=(index * 19, 60, 190, 255))
            draw.ellipse((22, 22, 42, 42), fill="white")
        else:
            value = chr(65 + index) if kind == "letters" else str(index)
            draw.text((0, -8), value, fill="black", font=ImageFont.load_default(size=70))
        images.append(_tile(image, index + 1))
    assert candidates(images) == []


@pytest.mark.parametrize("seed", range(12))
def test_two_unrelated_textured_images_do_not_form_a_strip(seed: int) -> None:
    rng = random.Random(seed)
    images = []
    for index in range(2):
        coarse = Image.new("RGB", (9, 9))
        coarse.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(81)])
        images.append(
            _tile(coarse.resize((64, 64), Image.Resampling.BICUBIC).convert("RGBA"), index + 1)
        )
    assert candidates(images) == []


@pytest.mark.parametrize("removed", [{2}, {2, 5}, {1, 3, 4, 6}])
def test_missing_pieces_cannot_be_compacted_into_a_false_rectangle(removed) -> None:
    tiles = [tile for tile in _split(3, 2) if int(tile.member.native_id) not in removed]
    tiles.append(_tile(Image.new("RGBA", (64, 64)), 99))
    for group in candidates(tiles):
        assert "99" not in {member.native_id for member in group.members}
        first = int(group.members[0].native_id) - 1
        x0, y0 = first % 3, first // 3
        for index, member in enumerate(group.members):
            original = int(member.native_id) - 1
            assert (original % 3, original // 3) == (
                x0 + index % group.columns,
                y0 + index // group.columns,
            )


def test_blank_padding_never_expands_a_supported_strip() -> None:
    tiles = _split(2, 1)
    tiles.extend(_tile(Image.new("RGBA", (64, 64)), index) for index in range(10, 15))
    found = candidates(tiles)
    assert {_layout(group) for group in found} == {(2, 1, ("1", "2"))}
