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

DecoderKind = Literal["in-memory-rgba", "webp", "tgs", "webm"]
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

    if kind not in {"in-memory-rgba", "webp", "tgs", "webm"}:
        raise AnalysisError("unknown decoder backend kind")

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
    if kind == "tgs":
        descriptor["rlottie_rgba_executable_sha256"] = _executable_sha256(rlottie_renderer)
    elif kind == "webm":
        if webm_codec not in _WEBM_CODECS or webm_preserve_alpha is None:
            raise AnalysisError("WebM backend fingerprint requires codec and alpha mode")
        descriptor.update(
            {
                "ffmpeg_executable_sha256": _executable_sha256(ffmpeg),
                "ffprobe_executable_sha256": _executable_sha256(ffprobe),
                "webm_codec": webm_codec,
                "webm_decode_mode": (
                    f"libvpx-{webm_codec}"
                    if webm_preserve_alpha and webm_codec in {"vp8", "vp9"}
                    else "ffmpeg-auto"
                ),
                "webm_preserve_alpha": webm_preserve_alpha,
            }
        )
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
