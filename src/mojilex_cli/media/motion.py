"""Exact evidence from the chronological images actually supplied to vision."""

from __future__ import annotations

import io
import struct
import zlib

from PIL import Image

from .models import HARD_MAX_FILE_BYTES, ProcessedMedia


def observed_frame_variation(value: ProcessedMedia) -> bool | None:
    """False means every complete background sequence is pixel-identical.

    This describes sampled evidence, not the original animation's entire timeline.
    Missing or invalid frames provide no conclusion. Background differences alone
    are not movement; even one changed pixel within either sequence is variation.
    """
    if not value.metadata.animated:
        return None
    if not value.frame_paths and not value.dark_frame_paths:
        return value.observed_frame_variation
    count = value.semantic_frame_count
    sequences = [value.frame_paths]
    if value.semantic_has_dark_render:
        sequences.append(value.dark_frame_paths)
    if count < 2 or any(len(paths) != count for paths in sequences):
        return None
    # These reads use the same bounded, private-file checks as retained frames.
    from .resume import _png, _read

    varied = False
    try:
        for paths in sequences:
            first = None
            for path in paths:
                data = _read(path, HARD_MAX_FILE_BYTES)
                _png(data)
                with Image.open(io.BytesIO(data)) as image:
                    with image.convert("RGBA") as rgba:
                        pixels = rgba.tobytes()
                if first is None:
                    first = pixels
                elif pixels != first:
                    varied = True
    except (
        OSError,
        ValueError,
        SyntaxError,
        struct.error,
        zlib.error,
        Image.DecompressionBombError,
    ):
        return None
    return varied
