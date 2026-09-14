"""Stable, path-free identity for the decoder stack used by analysis."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
from pathlib import Path
from typing import Literal

import PIL
from PIL import features

from .models import AnalysisError

DecoderKind = Literal["in-memory-rgba", "webp", "png", "tgs", "webm"]
_WEBM_CODECS = frozenset({"av1", "vp8", "vp9"})


def decoder_backend_fingerprint(
    kind: DecoderKind,
    *,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    rlottie_renderer: str = "mojilex-rlottie-rgba",
    webm_codec: str | None = None,
    webm_preserve_alpha: bool | None = None,
) -> str:
    """Return SHA-256 of a canonical decoder descriptor with no local paths.

    External tools are identified by their exact executable bytes. This avoids
    persisting configured paths or executing a version command outside the media
    sandbox. Runtime/architecture fields cover dynamically linked behavior that an
    executable digest alone cannot fully identify.
    """

    if kind not in {"in-memory-rgba", "webp", "png", "tgs", "webm"}:
        raise AnalysisError("unknown decoder backend kind")

    descriptor = _base_descriptor(kind)
    if kind == "tgs":
        descriptor["rlottie_rgba_executable_sha256"] = _executable_sha256(rlottie_renderer)
    elif kind == "webm":
        variant = _webm_variant(webm_codec, webm_preserve_alpha)
        descriptor.update(
            {
                "ffmpeg_executable_sha256": _executable_sha256(ffmpeg),
                "ffprobe_executable_sha256": _executable_sha256(ffprobe),
                **variant,
            }
        )
    return _descriptor_fingerprint(descriptor)


def webm_backend_fingerprints(
    *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe"
) -> tuple[str, ...]:
    """Probe all codec/alpha variants, reading each executable once per call.

    The descriptor bytes and variant order match six independent fingerprints.
    No identity is cached across calls: every later probe rereads the binaries.
    """
    descriptor = _base_descriptor("webm")
    descriptor.update(
        {
            "ffmpeg_executable_sha256": _executable_sha256(ffmpeg),
            "ffprobe_executable_sha256": _executable_sha256(ffprobe),
        }
    )
    return tuple(
        _descriptor_fingerprint({**descriptor, **_webm_variant(codec, alpha)})
        for codec in ("av1", "vp8", "vp9")
        for alpha in (False, True)
    )


def _base_descriptor(kind: DecoderKind) -> dict[str, object]:
    descriptor: dict[str, object] = {
        "descriptor": "decoder-backend-v1",
        "kind": kind,
        "machine": platform.machine().lower(),
        "pillow": PIL.__version__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation().lower(),
        "system": platform.system().lower(),
    }
    if kind in {"webp", "tgs"}:
        descriptor["pillow_webp"] = features.version("webp")
    if kind == "png":
        descriptor["pillow_zlib"] = features.version("zlib")
    return descriptor


def _webm_variant(codec: str | None, preserve_alpha: bool | None) -> dict[str, object]:
    if codec not in _WEBM_CODECS or preserve_alpha is None:
        raise AnalysisError("WebM backend fingerprint requires codec and alpha mode")
    return {
        "webm_codec": codec,
        "webm_decode_mode": (
            f"libvpx-{codec}" if preserve_alpha and codec in {"vp8", "vp9"} else "ffmpeg-auto"
        ),
        "webm_preserve_alpha": preserve_alpha,
    }


def _descriptor_fingerprint(descriptor: dict[str, object]) -> str:
    payload = json.dumps(
        descriptor, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _executable_sha256(command: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise AnalysisError("configured media decoder executable is unavailable")
    try:
        path = Path(resolved).resolve(strict=True)
        if not path.is_file():
            raise OSError("not a regular file")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise AnalysisError("configured media decoder cannot be fingerprinted") from exc
    return digest.hexdigest()
