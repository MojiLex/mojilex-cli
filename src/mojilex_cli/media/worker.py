"""Media decoder worker entry point.

This module is launched by :mod:`sandbox` in a resource-limited child process.
It receives no credentials and only operates on generated local paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import subprocess
import sys
from collections.abc import Mapping
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageChops, ImageOps, ImageStat

from mojilex_cli.analysis import (
    AnalysisError,
    DeterministicMediaAnalysis,
    MediaAnalysisAccumulator,
    analyze_decoded_media,
    decoder_backend_fingerprint,
)
from mojilex_cli.analysis.engine import sample_frame_indexes
from mojilex_cli.analysis.profiles import load_analysis_profile
from mojilex_cli.media.inspect import inspect_tgs, inspect_webm, inspect_webp, sniff_format
from mojilex_cli.media.models import MediaLimits

_LIGHT = (242, 242, 242, 255)
_DARK = (30, 30, 30, 255)
_CANVAS = 256
_CONTENT = 224
_TGS_RGBA_MAGIC = b"MLXRGBA1"
_TGS_RGBA_VERSION = 1
_TGS_RGBA_HEADER = struct.Struct("<8sIIIIQ")


def process(
    source: Path,
    output_dir: Path,
    expected_format: str,
    limits: MediaLimits,
    *,
    needs_repainting: bool,
    ffmpeg: str,
    ffprobe: str,
    rlottie_renderer: str,
    render_only: bool = False,
    expected_dark_render: bool | None = None,
) -> dict[str, Any]:
    if render_only != (expected_dark_render is not None):
        raise ValueError("render-only mode requires an exact background render marker")
    if sniff_format(source) != expected_format:
        raise ValueError("media signature does not match the expected Telegram format")
    output_dir.mkdir(parents=False, exist_ok=False)
    if expected_format == "webp":
        info = inspect_webp(source, limits)
        rgba_frames = [_load_rgba(source)]
        analysis = (
            None
            if render_only
            else analyze_decoded_media(
                rgba_frames,
                loop_mode="once",
                needs_repainting=needs_repainting,
                backend_fingerprint=decoder_backend_fingerprint("webp"),
            )
        )
        kind, mime, animated = "static", "image/webp", False
    elif expected_format == "tgs":
        info, document = inspect_tgs(source, limits)
        rgba_frames, analysis = _render_tgs(
            source,
            output_dir,
            document,
            limits,
            rlottie_renderer,
            needs_repainting=needs_repainting,
            analyze=not render_only,
        )
        kind, mime, animated = "animation", "application/x-tgsticker", True
    elif expected_format == "webm":
        info = inspect_webm(source, limits, ffprobe=ffprobe)
        duration_value = info["duration_ms"]
        codec_value = info["codec"]
        has_alpha_value = info["has_alpha"]
        if (
            not isinstance(duration_value, int)
            or not isinstance(codec_value, str)
            or not isinstance(has_alpha_value, bool)
        ):
            raise ValueError("WebM probe did not return complete typed metadata")
        rgba_frames = _render_webm(
            source,
            output_dir,
            duration_value,
            limits,
            ffmpeg,
            codec=codec_value,
            preserve_alpha=has_alpha_value,
        )
        analysis = (
            None
            if render_only
            else _analyze_webm_full_stream(
                source,
                info,
                limits,
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                needs_repainting=needs_repainting,
            )
        )
        kind, mime, animated = "video", "video/webm", True
    else:
        raise ValueError("unsupported media format")

    light_paths: list[str] = []
    dark_paths: list[str] = []
    decoded_alpha = any(_has_transparency(frame) for frame in rgba_frames)
    declared_alpha = bool(info.get("has_alpha", False))
    analyzed_alpha = analysis is not None and analysis.rendering.alpha_mode in {
        "binary",
        "translucent",
    }
    # Render-only resumes reuse a full-stream analysis verified by the parent
    # against the downloaded bytes. Its absence here is intentional; sampled
    # frames can all be opaque even when other frames contain transparency.
    if not render_only and declared_alpha and not analyzed_alpha:
        raise RuntimeError("ffmpeg did not preserve the WebM alpha channel")
    observed_dark_requirement = (
        needs_repainting
        or declared_alpha
        or analyzed_alpha
        or decoded_alpha
        or any(_low_contrast_on_light(frame) for frame in rgba_frames)
    )
    if render_only:
        assert expected_dark_render is not None
        if observed_dark_requirement and not expected_dark_render:
            raise RuntimeError("cached render context contradicts the decoded media")
        require_dark = expected_dark_render
    else:
        require_dark = observed_dark_requirement
    for index, frame in enumerate(rgba_frames):
        light = output_dir / f"frame-{index:02d}-light.png"
        _on_canvas(frame, _LIGHT).save(light, format="PNG", optimize=False)
        light_paths.append(str(light))
        if require_dark:
            dark = output_dir / f"frame-{index:02d}-dark.png"
            _on_canvas(frame, _DARK).save(dark, format="PNG", optimize=False)
            dark_paths.append(str(dark))
        frame.close()
    metadata = {
        "role": "primary",
        "kind": kind,
        "format": expected_format,
        "mime_type": mime,
        "sha256": _sha256(source),
        "byte_size": source.stat().st_size,
        "width": info["width"],
        "height": info["height"],
        "animated": animated,
    }
    if animated:
        metadata["duration_ms"] = info["duration_ms"]
    return {
        "metadata": metadata,
        "analysis": analysis.model_dump(mode="json") if analysis is not None else None,
        "frame_paths": light_paths,
        "dark_frame_paths": dark_paths,
    }


def _load_rgba(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image.load()
        oriented = ImageOps.exif_transpose(image)
        try:
            return oriented.convert("RGBA")
        finally:
            if oriented is not image:
                oriented.close()


def _render_tgs(
    source: Path,
    output_dir: Path,
    document: dict[str, Any],
    limits: MediaLimits,
    executable: str,
    *,
    needs_repainting: bool,
    analyze: bool = True,
) -> tuple[list[Image.Image], DeterministicMediaAnalysis | None]:
    executable_name = Path(executable).name.lower().removesuffix(".exe")
    if executable_name == "lottie2gif":
        raise RuntimeError("lottie2gif is unsupported because it discards the TGS alpha channel")
    if not executable or shutil.which(executable) is None:
        raise RuntimeError("the MojiLex rlottie RGBA renderer is required for TGS")
    try:
        expected_width = int(document["w"])
        expected_height = int(document["h"])
        timeline_frames = Decimal(str(document["op"])) - Decimal(str(document.get("ip", 0)))
    except (InvalidOperation, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("TGS native canvas or timeline is invalid") from exc
    if timeline_frames != timeline_frames.to_integral_value():
        raise RuntimeError("TGS native canvas or timeline is invalid")
    total = int(timeline_frames)
    maximum_frames = int(
        cast(dict[str, Any], load_analysis_profile("dedupe-v1").data["resource_limits"])[
            "max_full_frames"
        ]
    )
    maximum_bytes = int(
        cast(dict[str, Any], load_analysis_profile("dedupe-v1").data["resource_limits"])[
            "max_decoded_rgba_bytes"
        ]
    )
    frame_bytes = expected_width * expected_height * 4
    if total < 2 or total > maximum_frames or frame_bytes * total > maximum_bytes:
        raise RuntimeError("TGS renderer timeline exceeds the full-stream analysis limits")
    json_path = output_dir / "validated-lottie.json"
    # Re-serialize the already validated object; never give gzip quirks to the renderer.
    json_path.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    command = [
        executable,
        str(json_path),
        str(expected_width),
        str(expected_height),
        str(total),
    ]
    try:
        renderer = subprocess.Popen(
            command,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=output_dir,
        )
    except OSError as exc:
        raise RuntimeError("MojiLex rlottie RGBA renderer could not start") from exc
    try:
        try:
            frame_rate = Decimal(str(document["fr"]))
            duration_us = int(
                (Decimal(1_000_000) / frame_rate).quantize(Decimal(1), rounding=ROUND_HALF_UP)
            )
        except (InvalidOperation, KeyError, ValueError, ZeroDivisionError) as exc:
            raise RuntimeError("TGS frame duration cannot be represented") from exc
        accumulator = (
            MediaAnalysisAccumulator(
                width=expected_width,
                height=expected_height,
                frame_count=total,
                animated=True,
                loop_mode="loop",
                needs_repainting=needs_repainting,
                backend_fingerprint=decoder_backend_fingerprint("tgs", rlottie_renderer=executable),
            )
            if analyze
            else None
        )
        output_indexes = _sample_indexes(total, limits.frames)
        perceptual_indexes = sample_frame_indexes((duration_us,) * total, 16) if analyze else []
        required_indexes = set(output_indexes) | set(perceptual_indexes)
        sampled: dict[int, Image.Image] = {}
        try:
            if renderer.stdout is None:
                raise RuntimeError("TGS RGBA renderer has no output stream")
            _validate_tgs_rgba_header(
                renderer.stdout,
                width=expected_width,
                height=expected_height,
                frame_count=total,
                payload_bytes=frame_bytes * total,
            )
            for index in range(total):
                payload = _read_exact(renderer.stdout, frame_bytes)
                if payload is None:
                    raise RuntimeError("TGS RGBA stream is truncated")
                if not _transparent_rgb_is_zero(payload):
                    raise RuntimeError("TGS RGBA stream contains color in transparent pixels")
                native = Image.frombytes("RGBA", (expected_width, expected_height), payload)
                try:
                    if accumulator is not None:
                        accumulator.add_frame(native, duration_us=duration_us)
                    if index in required_indexes:
                        sampled[index] = native.copy()
                finally:
                    native.close()
            if renderer.stdout.read(1):
                raise RuntimeError("TGS RGBA stream contains trailing data")
            try:
                return_code = renderer.wait(timeout=limits.worker_timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("TGS RGBA renderer exceeded its time limit") from exc
            if return_code != 0:
                raise RuntimeError("MojiLex rlottie RGBA renderer failed")
            if set(sampled) != required_indexes:
                raise RuntimeError("TGS RGBA stream did not expose the full source timeline")
            analysis = (
                accumulator.finalize([sampled[index] for index in perceptual_indexes])
                if accumulator is not None
                else None
            )
            frames = [sampled[index].copy() for index in output_indexes]
            return frames, analysis
        finally:
            for frame in sampled.values():
                frame.close()
    finally:
        json_path.unlink(missing_ok=True)
        if renderer.poll() is None:
            renderer.kill()
        try:
            renderer.communicate(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            renderer.kill()


def _validate_tgs_rgba_header(
    stream: Any,
    *,
    width: int,
    height: int,
    frame_count: int,
    payload_bytes: int,
) -> None:
    header = _read_exact(stream, _TGS_RGBA_HEADER.size)
    if header is None:
        raise RuntimeError("TGS RGBA stream header is truncated")
    magic, version, actual_width, actual_height, actual_frames, actual_payload = (
        _TGS_RGBA_HEADER.unpack(header)
    )
    if magic != _TGS_RGBA_MAGIC or version != _TGS_RGBA_VERSION:
        raise RuntimeError("TGS RGBA stream header is unsupported")
    if (actual_width, actual_height) != (width, height):
        raise RuntimeError("TGS RGBA renderer changed the native canvas dimensions")
    if actual_frames != frame_count:
        raise RuntimeError("TGS RGBA renderer did not expose the full source timeline")
    if actual_payload != payload_bytes:
        raise RuntimeError("TGS RGBA stream payload length is invalid")


def _transparent_rgb_is_zero(payload: bytes) -> bool:
    # Mask away visible pixels in native Pillow code. A nonzero remaining RGB
    # channel is precisely a violation of the lossless stream's hidden-RGB rule.
    image = Image.frombytes("RGBA", (len(payload) // 4, 1), payload)
    alpha = image.getchannel("A")
    visible = alpha.point((0,) + (255,) * 255)
    try:
        image.paste((0, 0, 0, 0), mask=visible)
        return image.getbbox(alpha_only=False) is None
    finally:
        visible.close()
        alpha.close()
        image.close()


def _render_webm(
    source: Path,
    output_dir: Path,
    duration_ms: int,
    limits: MediaLimits,
    executable: str,
    *,
    codec: str,
    preserve_alpha: bool,
    filename_prefix: str = "decoded",
) -> list[Image.Image]:
    if not executable or shutil.which(executable) is None:
        raise RuntimeError("ffmpeg is required for WebM")
    frames: list[Image.Image] = []
    frame_step = duration_ms / limits.frames / 1000
    for index in range(limits.frames):
        # The first attempt is exactly t_i=(i+0.5)*D/N as required by the dataset
        # contract. A guarded earlier seek is used only when FFmpeg produced no frame.
        timestamp = (index + 0.5) * frame_step
        rendered = output_dir / f"{filename_prefix}-{index:02d}.png"
        completed = _decode_webm_frame(
            source,
            rendered,
            timestamp,
            limits,
            executable,
            codec=codec,
            preserve_alpha=preserve_alpha,
        )
        if completed.returncode != 0 and rendered.is_file():
            rendered.unlink(missing_ok=True)
            raise RuntimeError("ffmpeg returned an error after producing a WebM frame")
        if not rendered.is_file():
            fallback = max(0.0, timestamp - min(frame_step * 0.45, 0.05))
            completed = _decode_webm_frame(
                source,
                rendered,
                fallback,
                limits,
                executable,
                codec=codec,
                preserve_alpha=preserve_alpha,
            )
        if completed.returncode != 0 or not rendered.is_file():
            raise RuntimeError("ffmpeg failed to decode a deterministic WebM frame")
        try:
            frames.append(_load_rgba(rendered))
        finally:
            rendered.unlink(missing_ok=True)
    return frames


def _decode_webm_frame(
    source: Path,
    rendered: Path,
    timestamp: float,
    limits: MediaLimits,
    executable: str,
    *,
    codec: str,
    preserve_alpha: bool,
) -> subprocess.CompletedProcess[bytes]:
    command = [
        executable,
        "-v",
        "error",
        "-nostdin",
        "-ss",
        f"{timestamp:.6f}",
    ]
    if preserve_alpha and codec in {"vp8", "vp9"}:
        command.extend(["-c:v", "libvpx" if codec == "vp8" else "libvpx-vp9"])
    command.extend(
        [
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-vf",
            "format=rgba",
            "-y",
            str(rendered),
        ]
    )
    return subprocess.run(
        command,
        shell=False,
        capture_output=True,
        timeout=limits.worker_timeout_seconds,
        check=False,
    )


def _analyze_webm_full_stream(
    source: Path,
    info: Mapping[str, object],
    limits: MediaLimits,
    *,
    ffmpeg: str,
    ffprobe: str,
    needs_repainting: bool,
) -> DeterministicMediaAnalysis:
    width = info.get("width")
    height = info.get("height")
    codec = info.get("codec")
    preserve_alpha = info.get("has_alpha")
    duration_ms = info.get("duration_ms")
    if (
        not isinstance(width, int)
        or not isinstance(height, int)
        or not isinstance(codec, str)
        or not isinstance(preserve_alpha, bool)
        or not isinstance(duration_ms, int)
    ):
        raise AnalysisError("WebM analysis metadata is incomplete")
    durations = _webm_frame_durations(
        source, ffprobe=ffprobe, duration_ms=duration_ms, timeout=limits.worker_timeout_seconds
    )
    resource_limits = load_analysis_profile("dedupe-v1").data["resource_limits"]
    if not isinstance(resource_limits, Mapping):
        raise AnalysisError("dedupe profile resource limits are invalid")
    maximum_frames = resource_limits.get("max_full_frames")
    maximum_bytes = resource_limits.get("max_decoded_rgba_bytes")
    if not isinstance(maximum_frames, int) or not isinstance(maximum_bytes, int):
        raise AnalysisError("dedupe profile resource limits are incomplete")
    frame_bytes = width * height * 4
    if len(durations) > maximum_frames or frame_bytes * len(durations) > maximum_bytes:
        raise AnalysisError("WebM full decoded stream exceeds the analysis profile limits")
    accumulator = MediaAnalysisAccumulator(
        width=width,
        height=height,
        frame_count=len(durations),
        animated=True,
        loop_mode="loop",
        needs_repainting=needs_repainting,
        backend_fingerprint=decoder_backend_fingerprint(
            "webm",
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
            webm_codec=codec,
            webm_preserve_alpha=preserve_alpha,
        ),
    )
    sample_indexes = sample_frame_indexes(durations, 16)
    required_indexes = set(sample_indexes)
    sampled: dict[int, Image.Image] = {}
    command = [ffmpeg, "-v", "error", "-nostdin"]
    if preserve_alpha and codec in {"vp8", "vp9"}:
        command.extend(["-c:v", "libvpx" if codec == "vp8" else "libvpx-vp9"])
    command.extend(
        [
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "pipe:1",
        ]
    )
    try:
        process = subprocess.Popen(
            command,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        raise AnalysisError("cannot start ffmpeg full-stream decoder") from exc
    try:
        if process.stdout is None:
            raise AnalysisError("ffmpeg full-stream decoder has no output pipe")
        for frame_index, frame_duration in enumerate(durations):
            payload = _read_exact(process.stdout, frame_bytes)
            if payload is None:
                raise AnalysisError("ffmpeg decoded fewer WebM frames than ffprobe reported")
            frame = Image.frombytes("RGBA", (width, height), payload)
            try:
                accumulator.add_frame(frame, duration_us=frame_duration)
                if frame_index in required_indexes:
                    sampled[frame_index] = frame.copy()
            finally:
                frame.close()
        if process.stdout.read(1):
            raise AnalysisError("ffmpeg decoded more WebM frames than ffprobe reported")
        try:
            return_code = process.wait(timeout=limits.worker_timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            raise AnalysisError("ffmpeg full-stream decoder exceeded its time limit") from exc
        if return_code != 0:
            raise AnalysisError("ffmpeg full-stream decoder failed")
        if set(sampled) != required_indexes:
            raise AnalysisError("WebM perceptual timeline samples are incomplete")
        try:
            return accumulator.finalize(tuple(sampled[index] for index in sample_indexes))
        finally:
            for frame in sampled.values():
                frame.close()
    finally:
        if process.poll() is None:
            process.kill()
        try:
            process.communicate(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()


def _webm_frame_durations(
    source: Path, *, ffprobe: str, duration_ms: int, timeout: float
) -> tuple[int, ...]:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time,pkt_duration_time,duration_time",
        "-of",
        "json",
        str(source),
    ]
    try:
        completed = subprocess.run(
            command,
            shell=False,
            capture_output=True,
            timeout=min(timeout, 10.0),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AnalysisError("ffprobe could not enumerate the full WebM timeline") from exc
    profile_limits = load_analysis_profile("dedupe-v1").data["resource_limits"]
    if not isinstance(profile_limits, Mapping):
        raise AnalysisError("dedupe profile resource limits are invalid")
    metadata_limit = profile_limits.get("max_frame_metadata_bytes")
    if not isinstance(metadata_limit, int):
        raise AnalysisError("dedupe profile frame metadata limit is invalid")
    if completed.returncode != 0 or len(completed.stdout) > metadata_limit:
        raise AnalysisError("WebM full-frame metadata is invalid or oversized")
    try:
        payload = json.loads(completed.stdout)
        raw_frames = payload["frames"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AnalysisError("WebM full-frame metadata is malformed") from exc
    if not isinstance(raw_frames, list) or not raw_frames:
        raise AnalysisError("WebM full-frame timeline is empty")
    timestamps: list[int | None] = []
    durations: list[int | None] = []
    for raw_frame in raw_frames:
        if not isinstance(raw_frame, Mapping):
            raise AnalysisError("WebM frame metadata entry is invalid")
        timestamps.append(_seconds_to_microseconds(raw_frame.get("best_effort_timestamp_time")))
        duration = raw_frame.get("pkt_duration_time") or raw_frame.get("duration_time")
        durations.append(_seconds_to_microseconds(duration))
    first_timestamp = timestamps[0]
    for index, duration in enumerate(durations):
        if duration is not None and duration > 0:
            continue
        current = timestamps[index]
        following = timestamps[index + 1] if index + 1 < len(timestamps) else None
        if current is not None and following is not None and following > current:
            durations[index] = following - current
        elif current is not None and first_timestamp is not None and index == len(durations) - 1:
            durations[index] = duration_ms * 1000 - (current - first_timestamp)
    if any(value is None or value <= 0 for value in durations):
        raise AnalysisError("WebM presentation durations cannot be proven exactly")
    return tuple(cast(int, value) for value in durations)


def _seconds_to_microseconds(value: object) -> int | None:
    if value is None:
        return None
    try:
        result = (Decimal(str(value)) * Decimal(1_000_000)).quantize(
            Decimal(1), rounding=ROUND_HALF_UP
        )
    except (InvalidOperation, ValueError):
        return None
    converted = int(result)
    return converted if converted >= 0 else None


def _read_exact(stream: Any, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def _sample_indexes(total: int, count: int) -> list[int]:
    return [min(total - 1, int((index + 0.5) * total / count)) for index in range(count)]


def _on_canvas(image: Image.Image, background: tuple[int, int, int, int]) -> Image.Image:
    work = image.convert("RGBA")
    work.thumbnail((_CONTENT, _CONTENT), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (_CANVAS, _CANVAS), background)
    offset = ((_CANVAS - work.width) // 2, (_CANVAS - work.height) // 2)
    canvas.alpha_composite(work, offset)
    return canvas.convert("RGB")


def _low_contrast_on_light(image: Image.Image) -> bool:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    if ImageStat.Stat(alpha).mean[0] < 4:
        return True
    composited = Image.new("RGBA", rgba.size, _LIGHT)
    composited.alpha_composite(rgba)
    gray = composited.convert("L")
    difference = ImageChops.difference(gray, Image.new("L", gray.size, 242))
    return ImageStat.Stat(difference, mask=alpha).mean[0] < 24


def _has_transparency(image: Image.Image) -> bool:
    minimum, _ = cast(tuple[int, int], image.convert("RGBA").getchannel("A").getextrema())
    return minimum < 255


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=("webp", "tgs", "webm"), required=True)
    parser.add_argument("--limits", required=True)
    parser.add_argument("--needs-repainting", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--dark-render", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--rlottie-renderer", default="mojilex-rlottie-rgba")
    args = parser.parse_args()
    try:
        limits = MediaLimits.model_validate_json(args.limits)
        if args.dark_render and not args.render_only:
            raise ValueError("--dark-render requires --render-only")
        result = process(
            Path(args.source),
            Path(args.output),
            args.format,
            limits,
            needs_repainting=args.needs_repainting,
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            rlottie_renderer=args.rlottie_renderer,
            render_only=args.render_only,
            expected_dark_render=args.dark_render if args.render_only else None,
        )
        sys.stdout.write(json.dumps(result, ensure_ascii=True, separators=(",", ":")))
    except Exception as exc:  # worker boundary: parent maps this to a safe typed error
        sys.stderr.write(f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
