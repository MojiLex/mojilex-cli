"""Content-based validation for WebP, TGS, and WebM inputs."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from .models import (
    HARD_MAX_DURATION_MS,
    HARD_MAX_FILE_BYTES,
    HARD_MAX_PIXELS,
    HARD_MAX_TGS_JSON_BYTES,
    MediaDependencyError,
    MediaError,
    MediaLimitError,
    MediaLimits,
)

_WEBM_VIDEO_CODECS = frozenset({"vp8", "vp9", "av1"})
_EBML_DOCTYPE_ID = 0x4282


def validate_input_file(path: Path, limits: MediaLimits) -> int:
    if path.is_symlink() or not path.is_file():
        raise MediaError("media input must be a regular non-symlink file")
    size = path.stat().st_size
    if size <= 0:
        raise MediaError("media file is empty")
    if size > min(limits.max_file_bytes, HARD_MAX_FILE_BYTES):
        raise MediaLimitError("media file exceeds the 20 MiB hard limit")
    return size


def sniff_format(path: Path) -> str:
    with path.open("rb") as stream:
        header = stream.read(16)
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header[:2] == b"\x1f\x8b":
        return "tgs"
    if header[:4] == b"\x1aE\xdf\xa3":
        return "webm"
    raise MediaError("unsupported or malformed media signature")


def inspect_png(path: Path, limits: MediaLimits) -> dict[str, object]:
    """Validate a bounded static PNG while retaining its original encoded bytes."""
    validate_input_file(path, limits)
    try:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise MediaError("content is not PNG")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > limits.max_pixels:
                raise MediaLimitError("decoded PNG exceeds the pixel limit")
            if getattr(image, "is_animated", False) or getattr(image, "n_frames", 1) != 1:
                raise MediaError("animated PNG is not a supported static Telegram media type")
            image.verify()
        with Image.open(path) as decoded:
            decoded.load()
            oriented = ImageOps.exif_transpose(decoded)
            try:
                width, height = oriented.size
            finally:
                if oriented is not decoded:
                    oriented.close()
    except Image.DecompressionBombError as exc:
        raise MediaLimitError("decoded PNG exceeds the pixel limit") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise MediaError("PNG cannot be decoded") from exc
    return {"width": width, "height": height, "duration_ms": None}


def inspect_webp(path: Path, limits: MediaLimits) -> dict[str, object]:
    validate_input_file(path, limits)
    try:
        with Image.open(path) as image:
            if image.format != "WEBP":
                raise MediaError("content is not WebP")
            image.load()
            oriented = ImageOps.exif_transpose(image)
            try:
                width, height = oriented.size
            finally:
                if oriented is not image:
                    oriented.close()
            if width <= 0 or height <= 0 or width * height > limits.max_pixels:
                raise MediaLimitError("decoded WebP exceeds the pixel limit")
            frames = getattr(image, "n_frames", 1)
            duration_ms = 0
            for index in range(frames):
                image.seek(index)
                image.load()
                duration_ms += max(1, int(image.info.get("duration", 0) or 0))
    except Image.DecompressionBombError as exc:
        raise MediaLimitError("decoded WebP exceeds the pixel limit") from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise MediaError("WebP cannot be decoded") from exc
    if frames > 1:
        if duration_ms > limits.max_duration_ms:
            raise MediaLimitError("WebP animation exceeds the duration limit")
        # Telegram custom emoji WebP is static in MVP; reject ambiguity.
        raise MediaError("animated WebP is not an MVP Telegram media type")
    return {"width": width, "height": height, "duration_ms": None}


def inspect_tgs(path: Path, limits: MediaLimits) -> tuple[dict[str, object], dict[str, Any]]:
    validate_input_file(path, limits)
    decompressed = bytearray()
    try:
        with gzip.open(path, "rb") as stream:
            while chunk := stream.read(64 * 1024):
                decompressed.extend(chunk)
                if len(decompressed) > min(limits.max_tgs_json_bytes, HARD_MAX_TGS_JSON_BYTES):
                    raise MediaLimitError("unpacked TGS JSON exceeds 8 MiB")
    except (gzip.BadGzipFile, EOFError, OSError) as exc:
        raise MediaError("TGS gzip stream is malformed") from exc
    try:
        document = json.loads(decompressed)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MediaError("TGS does not contain valid UTF-8 JSON") from exc
    if not isinstance(document, dict):
        raise MediaError("TGS root must be a JSON object")
    _validate_json_depth(document, max_depth=64)
    _reject_lottie_external_content(document)
    try:
        width = _positive_int(document["w"])
        height = _positive_int(document["h"])
        frame_rate = float(document["fr"])
        first_frame = float(document.get("ip", 0))
        last_frame = float(document["op"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MediaError("TGS dimensions or timeline are invalid") from exc
    if document.get("v") is None or frame_rate <= 0 or last_frame <= first_frame:
        raise MediaError("TGS header or timeline is invalid")
    if width * height > min(limits.max_pixels, HARD_MAX_PIXELS):
        raise MediaLimitError("decoded TGS exceeds the pixel limit")
    duration_ms = round((last_frame - first_frame) * 1000 / frame_rate)
    if duration_ms <= 0 or duration_ms > min(limits.max_duration_ms, HARD_MAX_DURATION_MS):
        raise MediaLimitError("TGS animation exceeds the duration limit")
    return {"width": width, "height": height, "duration_ms": duration_ms}, document


def inspect_webm(path: Path, limits: MediaLimits, *, ffprobe: str = "ffprobe") -> dict[str, object]:
    validate_input_file(path, limits)
    if sniff_format(path) != "webm":
        raise MediaError("content is not WebM")
    if _read_ebml_doctype(path) != "webm":
        raise MediaError("EBML document type is not WebM")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,duration,pix_fmt:stream_tags=alpha_mode:"
        "format=duration,format_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        process = subprocess.run(
            command,
            shell=False,
            check=False,
            capture_output=True,
            timeout=min(limits.worker_timeout_seconds, 10.0),
            env=_decoder_environment(path.parent),
        )
    except FileNotFoundError as exc:
        raise MediaDependencyError("ffprobe is required for WebM") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaLimitError("ffprobe exceeded its time limit") from exc
    if process.returncode != 0 or len(process.stdout) > 1024 * 1024:
        raise MediaError("WebM container or video stream is invalid")
    try:
        data = json.loads(process.stdout)
        streams = data["streams"]
        stream = streams[0]
        format_info = data.get("format", {})
        width = _positive_int(stream["width"])
        height = _positive_int(stream["height"])
        codec = stream["codec_name"]
        if not isinstance(codec, str) or codec not in _WEBM_VIDEO_CODECS:
            raise MediaError("WebM video codec is not allowed")
        duration_value = stream.get("duration") or format_info.get("duration")
        duration_ms = round(float(duration_value) * 1000)
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MediaError("WebM probe result is incomplete") from exc
    if "webm" not in str(format_info.get("format_name", "")):
        raise MediaError("container is not WebM")
    if width * height > min(limits.max_pixels, HARD_MAX_PIXELS):
        raise MediaLimitError("decoded WebM exceeds the pixel limit")
    if duration_ms <= 0 or duration_ms > min(limits.max_duration_ms, HARD_MAX_DURATION_MS):
        raise MediaLimitError("WebM video exceeds the duration limit")
    pixel_format = stream.get("pix_fmt")
    tags = stream.get("tags")
    alpha_mode = tags.get("alpha_mode") if isinstance(tags, Mapping) else None
    has_alpha = alpha_mode in {1, "1"} or (
        isinstance(pixel_format, str)
        and (pixel_format.startswith(("rgba", "bgra", "argb", "abgr", "yuva")))
    )
    return {
        "width": width,
        "height": height,
        "duration_ms": duration_ms,
        "codec": codec,
        "has_alpha": has_alpha,
    }


def _read_ebml_doctype(path: Path) -> str:
    """Read the bounded EBML header and return its declared document type."""

    with path.open("rb") as stream:
        data = stream.read(64 * 1024)
    if not data.startswith(b"\x1aE\xdf\xa3"):
        raise MediaError("WebM EBML header is missing")
    try:
        header_size, position = _read_ebml_vint(data, 4)
        header_end = position + header_size
        if header_end > len(data):
            raise ValueError
        while position < header_end:
            element_id, position = _read_ebml_id(data, position)
            element_size, position = _read_ebml_vint(data, position)
            element_end = position + element_size
            if element_end > header_end:
                raise ValueError
            if element_id == _EBML_DOCTYPE_ID:
                return data[position:element_end].decode("ascii").lower()
            position = element_end
    except (UnicodeDecodeError, ValueError) as exc:
        raise MediaError("WebM EBML header is malformed") from exc
    raise MediaError("WebM EBML document type is missing")


def _read_ebml_id(data: bytes, position: int) -> tuple[int, int]:
    width = _ebml_vint_width(data, position, max_width=4)
    end = position + width
    return int.from_bytes(data[position:end], "big"), end


def _read_ebml_vint(data: bytes, position: int) -> tuple[int, int]:
    width = _ebml_vint_width(data, position, max_width=8)
    marker = 1 << (8 - width)
    value = data[position] & (marker - 1)
    end = position + width
    for byte in data[position + 1 : end]:
        value = (value << 8) | byte
    if value == (1 << (7 * width)) - 1:
        raise ValueError("unknown-size EBML element is forbidden in the header")
    return value, end


def _ebml_vint_width(data: bytes, position: int, *, max_width: int) -> int:
    if position >= len(data) or data[position] == 0:
        raise ValueError("invalid EBML variable integer")
    first = data[position]
    for width in range(1, max_width + 1):
        if first & (1 << (8 - width)):
            if position + width > len(data):
                break
            return width
    raise ValueError("invalid EBML variable integer")


def _validate_json_depth(value: object, *, max_depth: int) -> None:
    pending: list[tuple[object, int]] = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if depth > max_depth:
            raise MediaLimitError("TGS JSON exceeds the nesting limit")
        if isinstance(item, dict):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)


def _reject_lottie_external_content(document: Mapping[str, Any]) -> None:
    assets = document.get("assets", [])
    if not isinstance(assets, list):
        raise MediaError("TGS assets must be a list")
    for asset in assets:
        if isinstance(asset, Mapping) and any(key in asset for key in ("p", "u", "e")):
            raise MediaError("external or embedded raster resources are forbidden in TGS")
    pending: list[object] = [document]
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {"x", "expression"} and isinstance(child, str) and child.strip():
                    raise MediaError("executable expressions are forbidden in TGS")
                pending.append(child)
        elif isinstance(value, list):
            pending.extend(value)


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError
    result = int(value)
    if result <= 0 or result != float(value):
        raise ValueError
    return result


def _decoder_environment(temp_dir: Path) -> dict[str, str]:
    """Minimal environment: notably no API/GitHub/Telegram credentials."""

    result: dict[str, str] = {"PATH": os.environ.get("PATH", ""), "LC_ALL": "C"}
    for name in ("SYSTEMROOT", "WINDIR"):
        if name in os.environ:
            result[name] = os.environ[name]
    result["TMP"] = str(temp_dir)
    result["TEMP"] = str(temp_dir)
    return result
