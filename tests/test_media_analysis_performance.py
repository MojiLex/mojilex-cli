"""Optimized analysis must preserve the profile's exact integer operations."""

import random

import pytest
from PIL import Image

from mojilex_cli.analysis.engine import (
    _COSINE_Q14,
    _fit_rgba,
    _phash64,
    _resize_l_nearest,
    _rgba_color_counts,
)
from mojilex_cli.media.worker import _transparent_rgb_is_zero


def test_native_hidden_rgb_check_covers_every_alpha_and_channel() -> None:
    visible = bytes(channel for alpha in range(1, 256) for channel in (19, 137, 251, alpha))
    assert _transparent_rgb_is_zero(visible + bytes(4))
    for channel in range(3):
        hidden = bytearray(4)
        hidden[channel] = 1
        assert not _transparent_rgb_is_zero(visible + hidden)


def test_high_entropy_palette_falls_back_with_exact_pixels() -> None:
    raw = b"".join(index.to_bytes(3, "big") + b"\xff" for index in range(257 * 257))
    image = Image.frombytes("RGBA", (257, 257), raw)
    try:
        recovered = bytearray()
        for count, color in _rgba_color_counts(image):
            assert count == 1
            recovered.extend(color)
        assert recovered == raw
    finally:
        image.close()


@pytest.mark.parametrize(
    ("width", "height", "box_size"),
    [
        (512, 512, 224),
        (512, 257, 224),
        (63, 256, 224),
        (100, 100, 224),
        (57, 83, 224),
        (2, 3, 7),
        (3, 2, 9),
        (1, 7, 224),
    ],
)
def test_native_resize_matches_integer_profile(width: int, height: int, box_size: int) -> None:
    source = Image.frombytes(
        "RGBA", (width, height), random.Random(12).randbytes(width * height * 4)
    )
    result = _fit_rgba(source, box_size)
    try:
        expected = bytearray()
        for y in range(result.height):
            source_y = min(height - 1, ((2 * y + 1) * height) // (2 * result.height))
            for x in range(result.width):
                source_x = min(width - 1, ((2 * x + 1) * width) // (2 * result.width))
                expected.extend(source.getpixel((source_x, source_y)))
        assert result.tobytes() == bytes(expected)
    finally:
        source.close()
        result.close()


@pytest.mark.parametrize("seed", [0, 13, 97])
def test_separable_dct_matches_original_integer_coefficients(seed: int) -> None:
    values = random.Random(seed).randbytes(47 * 65)
    resized = _resize_l_nearest(values, (47, 65), (32, 32))
    coefficients = []
    for vertical in range(8):
        for horizontal in range(8):
            coefficients.append(
                sum(
                    resized[y * 32 + x] * _COSINE_Q14[horizontal][x] * _COSINE_Q14[vertical][y]
                    for y in range(32)
                    for x in range(32)
                )
            )
    median = sorted(coefficients[1:])[31]
    expected = 0
    for coefficient in coefficients:
        expected = (expected << 1) | int(coefficient > median)
    assert _phash64(values, (47, 65)) == expected
