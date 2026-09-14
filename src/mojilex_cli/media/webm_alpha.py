"""Bounded VP9 fallback that decodes the original color and alpha independently."""

from __future__ import annotations

import hashlib
import struct
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

from mojilex_cli.analysis import (
    DeterministicMediaAnalysis,
    MediaAnalysisAccumulator,
    decoder_backend_fingerprint,
)
from mojilex_cli.analysis.engine import sample_frame_indexes

from .inspect import _read_ebml_id, _read_ebml_vint
from .models import MediaLimits


def separate_alpha_fingerprint(base: str) -> str:
    return hashlib.sha256((base + ":separate-native-alpha-v1").encode()).hexdigest()


def _elements(data: bytes, start: int, end: int) -> Iterator[tuple[int, int, int]]:
    while start < end:
        identifier, cursor = _read_ebml_id(data, start)
        size, payload = _read_ebml_vint(data, cursor)
        stop = payload + size
        if stop > end:
            raise ValueError("truncated WebM element")
        yield identifier, payload, stop
        start = stop


def _alpha_packets(data: bytes) -> list[tuple[int, bytes]]:
    """Accept only a single VP9 track with one unlaced alpha block per frame."""
    segments = [(a, b) for tag, a, b in _elements(data, 0, len(data)) if tag == 0x18538067]
    if len(segments) != 1:
        raise ValueError("expected one finite WebM segment")
    children = list(_elements(data, *segments[0]))
    scale = 1_000_000
    tracks = []
    for tag, a, b in children:
        if tag == 0x1549A966:
            for field, c, d in _elements(data, a, b):
                if field == 0x2AD7B1:
                    scale = int.from_bytes(data[c:d], "big")
        if tag == 0x1654AE6B:
            for field, c, d in _elements(data, a, b):
                if field == 0xAE:
                    tracks.append({k: data[e:f] for k, e, f in _elements(data, c, d)})
    if len(tracks) != 1 or tracks[0].get(0x86) != b"V_VP9":
        raise ValueError("expected one VP9 track")
    track = int.from_bytes(tracks[0].get(0xD7, b""), "big")
    packets = []
    for tag, a, b in children:
        if tag != 0x1F43B675:
            continue
        cluster = list(_elements(data, a, b))
        clocks = [int.from_bytes(data[c:d], "big") for k, c, d in cluster if k == 0xE7]
        if len(clocks) != 1:
            raise ValueError("cluster timestamp missing")
        for field, c, d in cluster:
            if field == 0xA3:
                raise ValueError("alpha block missing")
            if field != 0xA0:
                continue
            group = list(_elements(data, c, d))
            blocks = [(e, f) for k, e, f in group if k == 0xA1]
            additions = [(e, f) for k, e, f in group if k == 0x75A1]
            if len(blocks) != 1 or len(additions) != 1:
                raise ValueError("ambiguous alpha mapping")
            e, f = blocks[0]
            number, pos = _read_ebml_vint(data, e)
            if number != track or pos + 3 >= f or data[pos + 2] & 6:
                raise ValueError("unsupported track or lacing")
            pts_ns = (clocks[0] + struct.unpack_from(">h", data, pos)[0]) * scale
            if pts_ns < 0 or pts_ns % 1000:
                raise ValueError("unrepresentable timestamp")
            more = list(_elements(data, *additions[0]))
            if len(more) != 1 or more[0][0] != 0xA6:
                raise ValueError("ambiguous block additions")
            fields = {k: data[e:f] for k, e, f in _elements(data, more[0][1], more[0][2])}
            if int.from_bytes(fields.get(0xEE, b"\x01"), "big") != 1 or not fields.get(0xA5):
                raise ValueError("alpha payload missing")
            packets.append((pts_ns // 1000, fields[0xA5]))
    return packets


def decode_separate_alpha(
    source: Path,
    output: Path,
    info: dict[str, Any],
    limits: MediaLimits,
    *,
    ffmpeg: str,
    ffprobe: str,
    needs_repainting: bool,
    analyze: bool,
) -> tuple[list[Image.Image], DeterministicMediaAnalysis | None] | None:
    from .worker import (
        _read_exact,
        _webm_frame_metadata,
        _webm_presentation_intervals,
    )

    try:
        if source.stat().st_size > limits.max_file_bytes:
            return None
        packets = _alpha_packets(source.read_bytes())
        metadata = _webm_frame_metadata(
            source, ffprobe=ffprobe, timeout=limits.worker_timeout_seconds
        )
        intervals = _webm_presentation_intervals(metadata, duration_ms=info["duration_ms"])
        if [t for t, _ in packets] != [a for a, _ in intervals]:
            return None
    except (OSError, ValueError, RuntimeError):
        return None
    width, height = info["width"], info["height"]
    count = len(packets)
    if (
        not 1 <= count <= 600
        or not 1 <= width <= 65535
        or not 1 <= height <= 65535
        or width * height * count * 4 > 512 * 1024 * 1024
    ):
        return None
    alpha_path = output / "original-alpha.ivf"
    # IVF only wraps the original VP9 alpha bytes; it does not re-encode them.
    header = struct.pack(
        "<4sHH4sHHIIII", b"DKIF", 0, 32, b"VP90", width, height, 1_000_000, 1, count, 0
    )
    alpha_path.write_bytes(header + b"".join(struct.pack("<IQ", len(b), t) + b for t, b in packets))
    processes = []
    sampled = {}
    try:
        for path, alpha in ((source, False), (alpha_path, True)):
            command = [
                ffmpeg,
                "-v",
                "error",
                "-xerror",
                "-nostdin",
                "-threads",
                "1",
                "-filter_threads",
                "1",
                "-c:v",
                "vp9",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-vf",
                "extractplanes=y" if alpha else "format=rgba",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray" if alpha else "rgba",
                "-threads",
                "1",
                "pipe:1",
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                )
            )
        durations = tuple(b - a for a, b in intervals)
        perceptual = sample_frame_indexes(durations, 16) if analyze else []
        midpoints = [
            (i + 0.5) * info["duration_ms"] / limits.frames / 1000 for i in range(limits.frames)
        ]
        selected = [
            next(i for i, (a, b) in enumerate(intervals) if a <= int(t * 1_000_000 + 0.5) < b)
            for t in midpoints
        ]
        required = set(selected) | set(perceptual)
        fingerprint = separate_alpha_fingerprint(
            decoder_backend_fingerprint(
                "webm",
                ffmpeg=ffmpeg,
                ffprobe=ffprobe,
                webm_codec="vp9",
                webm_preserve_alpha=True,
            )
        )
        accumulator = (
            MediaAnalysisAccumulator(
                width=width,
                height=height,
                frame_count=count,
                animated=True,
                loop_mode="loop",
                needs_repainting=needs_repainting,
                backend_fingerprint=fingerprint,
            )
            if analyze
            else None
        )
        for index, duration in enumerate(durations):
            color = _read_exact(processes[0].stdout, width * height * 4)
            alpha_bytes = _read_exact(processes[1].stdout, width * height)
            if color is None or alpha_bytes is None:
                raise RuntimeError("separate WebM alpha timeline is truncated")
            with Image.frombytes("RGBA", (width, height), color) as frame:
                with Image.frombytes("L", (width, height), alpha_bytes) as plane:
                    frame.putalpha(plane)
                if accumulator is not None:
                    accumulator.add_frame(frame, duration_us=duration)
                if index in required:
                    sampled[index] = frame.copy()
        for process in processes:
            assert process.stdout is not None
            if process.stdout.read(1) or process.wait(timeout=limits.worker_timeout_seconds) != 0:
                raise RuntimeError("separate WebM alpha decode is incomplete")
        analysis = accumulator.finalize([sampled[i] for i in perceptual]) if accumulator else None
        return [sampled[i].copy() for i in selected], analysis
    finally:
        for frame in sampled.values():
            frame.close()
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
        alpha_path.unlink(missing_ok=True)
