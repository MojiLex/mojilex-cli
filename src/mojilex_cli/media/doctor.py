"""Actual synthetic-fixture probes used by ``mojilex doctor``."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from .models import BackendStatus, MediaLimits
from .sandbox import SafeMediaWorker, hard_resource_limits_available


def probe_media_backends(
    *,
    ffmpeg: str | None = None,
    ffprobe: str | None = None,
    rlottie_renderer: str | None = None,
) -> tuple[BackendStatus, ...]:
    """Decode generated, non-copyrighted fixtures instead of trusting ``--version``."""

    if not hard_resource_limits_available():
        detail = "OS hard CPU/RAM worker limits are unavailable"
        return tuple(
            BackendStatus(name=name, available=False, fixture_decoded=False, detail=detail)
            for name in ("webp", "tgs/rlottie-rgba", "webm/ffmpeg")
        )
    limits = MediaLimits(frames=4, worker_timeout_seconds=30)
    worker = SafeMediaWorker(
        limits, ffmpeg=ffmpeg, ffprobe=ffprobe, rlottie_renderer=rlottie_renderer
    )
    with tempfile.TemporaryDirectory(prefix="mojilex-doctor-") as raw_root:
        root = Path(raw_root)
        statuses = [_probe_webp(root, worker)]
        statuses.append(_probe_tgs(root, worker, worker.rlottie_renderer))
        statuses.append(_probe_webm(root, worker, worker.ffmpeg, worker.ffprobe))
    return tuple(statuses)


def _probe_webp(root: Path, worker: SafeMediaWorker) -> BackendStatus:
    source = root / "fixture.webp"
    Image.new("RGBA", (8, 8), (255, 0, 0, 128)).save(source, "WEBP", lossless=True)
    try:
        worker.process(source, root / "webp-out", expected_format="webp")
    except Exception as exc:
        return BackendStatus(
            name="webp", available=True, fixture_decoded=False, detail=_safe_detail(exc, root)
        )
    return BackendStatus(name="webp", available=True, fixture_decoded=True, detail="ok")


def _probe_tgs(root: Path, worker: SafeMediaWorker, executable: str) -> BackendStatus:
    available = shutil.which(executable) is not None
    if not available:
        return BackendStatus(
            name="tgs/rlottie-rgba",
            available=False,
            fixture_decoded=False,
            detail=f"{executable} was not found on PATH",
        )
    document = {
        "v": "5.7.4",
        "fr": 30,
        "ip": 0,
        "op": 4,
        "w": 32,
        "h": 32,
        "nm": "synthetic MojiLex doctor fixture",
        "ddd": 0,
        "assets": [],
        "layers": [
            {
                "ddd": 0,
                "ind": 1,
                "ty": 4,
                "nm": "semistransparent red square",
                "sr": 1,
                "ks": {
                    "o": {"a": 0, "k": 100},
                    "r": {"a": 0, "k": 0},
                    "p": {"a": 0, "k": [16, 16, 0]},
                    "a": {"a": 0, "k": [0, 0, 0]},
                    "s": {"a": 0, "k": [100, 100, 100]},
                },
                "ao": 0,
                "shapes": [
                    {
                        "ty": "rc",
                        "d": 1,
                        "s": {"a": 0, "k": [16, 16]},
                        "p": {"a": 0, "k": [0, 0]},
                        "r": {"a": 0, "k": 0},
                        "nm": "square",
                    },
                    {
                        "ty": "fl",
                        "c": {"a": 0, "k": [1, 0, 0, 1]},
                        "o": {"a": 0, "k": 50},
                        "r": 1,
                        "bm": 0,
                        "nm": "half-alpha red",
                    },
                ],
                "ip": 0,
                "op": 4,
                "st": 0,
                "bm": 0,
            }
        ],
    }
    source = root / "fixture.tgs"
    with source.open("wb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as stream:
            stream.write(json.dumps(document, separators=(",", ":")).encode())
    try:
        processed = worker.process(source, root / "tgs-out", expected_format="tgs")
        if processed.analysis is None or processed.analysis.rendering.alpha_mode != "translucent":
            raise RuntimeError("rlottie RGBA renderer did not preserve semitransparent alpha")
        colors = processed.analysis.rendering.dominant_colors or ()
        if not colors or colors[0].hex != "#ff0000":
            raise RuntimeError("rlottie RGBA renderer did not unpremultiply color losslessly")
        if len(processed.frame_paths) != 4:
            raise RuntimeError("rlottie RGBA renderer did not preserve the validated timeline")
    except Exception as exc:
        return BackendStatus(
            name="tgs/rlottie-rgba",
            available=True,
            fixture_decoded=False,
            detail=_safe_detail(exc, root),
        )
    return BackendStatus(name="tgs/rlottie-rgba", available=True, fixture_decoded=True, detail="ok")


def _probe_webm(root: Path, worker: SafeMediaWorker, ffmpeg: str, ffprobe: str) -> BackendStatus:
    available = shutil.which(ffmpeg) is not None and shutil.which(ffprobe) is not None
    if not available:
        return BackendStatus(
            name="webm/ffmpeg",
            available=False,
            fixture_decoded=False,
            detail="ffmpeg and ffprobe are required on PATH",
        )
    source = root / "fixture.webm"
    environment = {"PATH": os.environ.get("PATH", ""), "LC_ALL": "C"}
    for name in ("SYSTEMROOT", "WINDIR"):
        if name in os.environ:
            environment[name] = os.environ[name]
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=16x16:d=0.2",
                "-c:v",
                "libvpx-vp9",
                "-an",
                "-y",
                str(source),
            ],
            shell=False,
            capture_output=True,
            timeout=20,
            check=False,
            env=environment,
        )
        if completed.returncode != 0:
            raise RuntimeError("ffmpeg could not generate its synthetic fixture")
        worker.process(source, root / "webm-out", expected_format="webm")
    except Exception as exc:
        return BackendStatus(
            name="webm/ffmpeg",
            available=True,
            fixture_decoded=False,
            detail=_safe_detail(exc, root),
        )
    return BackendStatus(name="webm/ffmpeg", available=True, fixture_decoded=True, detail="ok")


def _safe_detail(exc: Exception, root: Path) -> str:
    return f"{type(exc).__name__}: {exc}".replace(str(root), "<temp>")[:500]
