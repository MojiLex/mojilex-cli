"""Ephemeral side-by-side preview generation for human dedupe review."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw

from mojilex_cli.media import ProcessedMedia


def build_side_by_side_preview(
    left: ProcessedMedia,
    right: ProcessedMedia,
    output: Path,
    *,
    left_label: str,
    right_label: str,
    right_shift: int = 0,
) -> Path:
    """Build a PNG whose rows are synchronized temporary frame comparisons."""

    left_paths = left.frame_paths
    right_paths = _shifted(right.frame_paths, right_shift)
    count = max(len(left_paths), len(right_paths))
    if count == 0:
        raise ValueError("dedupe preview requires rendered frames")
    cell = 256
    label_height = 28
    rows: list[Image.Image] = []
    try:
        for index in range(count):
            left_image = _frame(left_paths[min(index, len(left_paths) - 1)], cell)
            right_image = _frame(right_paths[min(index, len(right_paths) - 1)], cell)
            row = Image.new("RGB", (cell * 2, cell + label_height), "#202020")
            row.paste(left_image, (0, label_height))
            row.paste(right_image, (cell, label_height))
            draw = ImageDraw.Draw(row)
            draw.text((8, 7), left_label[:48], fill="white")
            draw.text((cell + 8, 7), right_label[:48], fill="white")
            draw.text((cell - 26, 7), f"{index + 1}", fill="#bbbbbb")
            left_image.close()
            right_image.close()
            rows.append(row)
        output.parent.mkdir(parents=True, exist_ok=False)
        canvas = Image.new("RGB", (cell * 2, len(rows) * (cell + label_height)), "#202020")
        try:
            for index, row in enumerate(rows):
                canvas.paste(row, (0, index * (cell + label_height)))
            canvas.save(output, format="PNG", optimize=False)
        finally:
            canvas.close()
        return output
    finally:
        for row in rows:
            row.close()


def _shifted(paths: Sequence[Path], shift: int) -> tuple[Path, ...]:
    values = tuple(paths)
    if not values:
        return values
    normalized = shift % len(values)
    return values[normalized:] + values[:normalized]


def _frame(path: Path, size: int) -> Image.Image:
    with Image.open(path) as source:
        source.load()
        return source.convert("RGB").resize((size, size), Image.Resampling.NEAREST)
