"""Deterministic temporary contact-sheet construction."""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field

from .models import ProcessedMedia

MAX_SHEET_SIDE = 4096


class ContactSheetInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    identifier: str
    media: ProcessedMedia


class ContactSheet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: Path = Field(exclude=True, repr=False)
    variant: str
    mapping: dict[str, str]
    animated: bool


def build_contact_sheets(
    items: Sequence[ContactSheetInput], output_dir: Path
) -> tuple[ContactSheet, ...]:
    if not items:
        return ()
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("contact-sheet output must be an existing private directory")
    static: list[tuple[int, ContactSheetInput]] = []
    animated: list[tuple[int, ContactSheetInput]] = []
    for number, item in enumerate(items, start=1):
        label = f"E{number:03d}"
        target = animated if item.media.metadata.animated else static
        target.append((number, item))
        if len(label) != 4:
            raise ValueError("contact sheet supports at most 999 items per invocation")
    sheets: list[ContactSheet] = []
    for offset in range(0, len(static), 16):
        sheets.extend(_build_static(static[offset : offset + 16], output_dir))
    if animated:
        frame_count = len(animated[0][1].media.frame_paths)
        if frame_count < 4 or frame_count > 16:
            raise ValueError("animated contact sheet requires 4-16 frames")
        if any(len(item.media.frame_paths) != frame_count for _, item in animated):
            raise ValueError("animation frame counts must match within a sheet run")
        tile = min(256, (MAX_SHEET_SIDE - 64) // frame_count)
        row_height = tile + 24
        rows_per_sheet = min(16, MAX_SHEET_SIDE // row_height)
        for offset in range(0, len(animated), rows_per_sheet):
            sheets.extend(
                _build_animated(animated[offset : offset + rows_per_sheet], output_dir, tile)
            )
    return tuple(sheets)


def expected_labels(sheets: Sequence[ContactSheet]) -> tuple[str, ...]:
    labels: dict[str, None] = {}
    for sheet in sheets:
        for label in sheet.mapping:
            labels[label] = None
    return tuple(labels)


def validate_response_labels(expected: Sequence[str], actual: Sequence[str]) -> None:
    if len(actual) != len(set(actual)):
        raise ValueError("AI response contains duplicate contact-sheet labels")
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected) - set(actual))
        unknown = sorted(set(actual) - set(expected))
        raise ValueError(f"AI response label mismatch; missing={missing}, unknown={unknown}")


def _build_static(
    batch: Sequence[tuple[int, ContactSheetInput]], output_dir: Path
) -> list[ContactSheet]:
    columns = min(4, len(batch))
    rows = math.ceil(len(batch) / columns)
    cell_width, cell_height = 272, 292
    variants = ["light"]
    if any(item.media.dark_frame_paths for _, item in batch):
        variants.append("dark")
    result: list[ContactSheet] = []
    for variant in variants:
        canvas = Image.new("RGB", (columns * cell_width, rows * cell_height), "#F2F2F2")
        draw = ImageDraw.Draw(canvas)
        mapping: dict[str, str] = {}
        for slot, (number, item) in enumerate(batch):
            label = f"E{number:03d}"
            mapping[label] = item.identifier
            frames = item.media.dark_frame_paths if variant == "dark" else item.media.frame_paths
            if not frames:
                frames = item.media.frame_paths
            with Image.open(frames[0]) as frame:
                image = frame.convert("RGB")
                x = (slot % columns) * cell_width + 8
                y = (slot // columns) * cell_height + 8
                canvas.paste(image, (x, y))
            draw.rectangle((x, y + 256, x + 255, y + 283), fill="#111111")
            draw.text((x + 8, y + 263), label, fill="#FFFFFF", font=ImageFont.load_default())
        path = output_dir / f"contact-{uuid.uuid4().hex}-{variant}.png"
        canvas.save(path, format="PNG", optimize=False)
        result.append(ContactSheet(path=path, variant=variant, mapping=mapping, animated=False))
    return result


def _build_animated(
    batch: Sequence[tuple[int, ContactSheetInput]], output_dir: Path, tile: int
) -> list[ContactSheet]:
    count = len(batch[0][1].media.frame_paths)
    row_height = tile + 24
    width = 64 + count * tile
    variants = ["light"]
    if any(item.media.dark_frame_paths for _, item in batch):
        variants.append("dark")
    result: list[ContactSheet] = []
    for variant in variants:
        canvas = Image.new("RGB", (width, row_height * len(batch)), "#F2F2F2")
        draw = ImageDraw.Draw(canvas)
        mapping: dict[str, str] = {}
        for row, (number, item) in enumerate(batch):
            label = f"E{number:03d}"
            mapping[label] = item.identifier
            y = row * row_height
            draw.rectangle((0, y, 63, y + row_height - 1), fill="#111111")
            draw.text((8, y + 8), label, fill="#FFFFFF", font=ImageFont.load_default())
            frames = item.media.dark_frame_paths if variant == "dark" else item.media.frame_paths
            if not frames:
                frames = item.media.frame_paths
            for index, path in enumerate(frames):
                with Image.open(path) as frame:
                    image = frame.convert("RGB")
                    if tile != 256:
                        image = image.resize((tile, tile), Image.Resampling.LANCZOS)
                    canvas.paste(image, (64 + index * tile, y))
            draw.text(
                (68, y + tile + 5),
                "chronological frames - one looping animation",
                fill="#111111",
                font=ImageFont.load_default(),
            )
        path = output_dir / f"contact-{uuid.uuid4().hex}-{variant}.png"
        canvas.save(path, format="PNG", optimize=False)
        if canvas.width > MAX_SHEET_SIDE or canvas.height > MAX_SHEET_SIDE:
            path.unlink(missing_ok=True)
            raise ValueError("contact sheet exceeds 4096x4096")
        result.append(ContactSheet(path=path, variant=variant, mapping=mapping, animated=True))
    return result
