import asyncio
import gzip
import io
import json
import os
import shutil
import struct
import subprocess
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from PIL import Image

from media_backend_helpers import require_native_media_limits
from mojilex_cli.media import (
    ContactSheetInput,
    MediaError,
    MediaLimitError,
    MediaLimits,
    MediaProcessor,
    SafeMediaWorker,
    SourceChangedDuringRunError,
    TemporaryMediaRun,
    build_contact_sheets,
    validate_response_labels,
)
from mojilex_cli.media import sandbox as sandbox_module
from mojilex_cli.media import worker as worker_module
from mojilex_cli.media.inspect import inspect_tgs, inspect_webm

_RGBA_HEADER = struct.Struct("<8sIIIIQ")


def _rgba_stream(
    frames: list[bytes],
    *,
    width: int,
    height: int,
    header_width: int | None = None,
    header_height: int | None = None,
    header_frames: int | None = None,
    payload_length: int | None = None,
    magic: bytes = b"MLXRGBA1",
    trailer: bytes = b"",
) -> bytes:
    payload = b"".join(frames)
    return (
        _RGBA_HEADER.pack(
            magic,
            1,
            width if header_width is None else header_width,
            height if header_height is None else header_height,
            len(frames) if header_frames is None else header_frames,
            len(payload) if payload_length is None else payload_length,
        )
        + payload
        + trailer
    )


class _FakeRendererProcess:
    def __init__(self, payload: bytes, *, returncode: int = 0) -> None:
        self.stdout = io.BytesIO(payload)
        self.returncode: int | None = None
        self._final_returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, *, timeout: float | None = None) -> int:
        del timeout
        self.returncode = self._final_returncode
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9

    def communicate(self, *, timeout: float | None = None) -> tuple[bytes, bytes]:
        del timeout
        if self.returncode is None:
            self.returncode = self._final_returncode
        return self.stdout.read(), b""


def test_tgs_inspection_is_bounded_and_rejects_external_assets(tmp_path: Path) -> None:
    valid = {"v": "5.7", "fr": 30, "ip": 0, "op": 30, "w": 100, "h": 100, "assets": []}
    source = tmp_path / "valid.tgs"
    source.write_bytes(gzip.compress(json.dumps(valid).encode(), mtime=0))
    info, _ = inspect_tgs(source, MediaLimits())
    assert info["duration_ms"] == 1000

    valid["assets"] = [{"p": "https://evil.example/a.png"}]
    source.write_bytes(gzip.compress(json.dumps(valid).encode(), mtime=0))
    with pytest.raises(MediaError, match="resources"):
        inspect_tgs(source, MediaLimits())


def test_tgs_analysis_consumes_the_full_renderer_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "synthetic-renderer.exe"
    executable.write_bytes(b"synthetic deterministic renderer")
    source = tmp_path / "source.tgs"
    source.write_bytes(b"source bytes are not read by the mocked renderer")
    output = tmp_path / "output"
    output.mkdir()
    document = {
        "v": "5.7",
        "fr": 2,
        "ip": 7,
        "op": 9,
        "w": 2,
        "h": 2,
        "assets": [],
    }

    def fake_popen(command: list[str], **kwargs: object) -> _FakeRendererProcess:
        del kwargs
        width, height, count = (int(value) for value in command[2:5])
        colors = ((255, 0, 0, 255), (0, 0, 255, 255), (0, 255, 0, 255))
        frames = [bytes(colors[index % len(colors)]) * (width * height) for index in range(count)]
        return _FakeRendererProcess(_rgba_stream(frames, width=width, height=height))

    monkeypatch.setattr(worker_module.shutil, "which", lambda _: str(executable))
    monkeypatch.setattr(worker_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        worker_module, "decoder_backend_fingerprint", lambda *_args, **_kw: "0" * 64
    )
    frames, analysis = worker_module._render_tgs(
        source,
        output,
        document,
        MediaLimits(frames=4),
        str(executable),
        needs_repainting=False,
    )
    try:
        assert len(frames) == 4
        assert analysis.analysis_scope == "full-decoded-stream"
        assert analysis.fingerprint.perceptual.sample_count == 16
        assert analysis.rendering.palette_dynamics == "changing"
    finally:
        for frame in frames:
            frame.close()
    assert not tuple(output.iterdir())


@pytest.mark.parametrize("frame_count", [1, 2])
def test_tgs_rgba_contract_passes_native_dimensions_and_frame_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame_count: int
) -> None:
    executable = tmp_path / "mojilex-rlottie-rgba.exe"
    executable.write_bytes(b"synthetic lossless renderer")
    output = tmp_path / "output"
    output.mkdir()
    document = {
        "v": "5.7",
        "fr": 2,
        "ip": 0,
        "op": frame_count,
        "w": 2,
        "h": 2,
        "assets": [],
    }
    observed: list[str] = []

    def fake_popen(command: list[str], **kwargs: object) -> _FakeRendererProcess:
        del kwargs
        observed.extend(command)
        return _FakeRendererProcess(
            _rgba_stream(
                [bytes((255, 0, 0, 128)) * 4, bytes((0, 0, 255, 255)) * 4][:frame_count],
                width=2,
                height=2,
            )
        )

    monkeypatch.setattr(worker_module.shutil, "which", lambda _: str(executable))
    monkeypatch.setattr(worker_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        worker_module, "decoder_backend_fingerprint", lambda *_args, **_kw: "0" * 64
    )
    frames, analysis = worker_module._render_tgs(
        tmp_path / "source.tgs",
        output,
        document,
        MediaLimits(frames=4),
        str(executable),
        needs_repainting=False,
    )
    try:
        assert observed[2:5] == ["2", "2", str(frame_count)]
        assert analysis.rendering.alpha_mode == "translucent"
        assert analysis.analysis_scope == "full-decoded-stream"
        assert len(frames) == 4
    finally:
        for frame in frames:
            frame.close()


@pytest.mark.parametrize(
    ("width", "height", "frame_count"),
    [(2, 2, 0), (2, 2, 601), (512, 512, 1500), (4000, 4000, 24)],
)
def test_tgs_single_frame_support_retains_stream_safety_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    width: int,
    height: int,
    frame_count: int,
) -> None:
    monkeypatch.setattr(worker_module.shutil, "which", lambda value: value)
    monkeypatch.setattr(
        worker_module.subprocess, "Popen", lambda *a, **kw: pytest.fail("unsafe renderer launch")
    )
    with pytest.raises(RuntimeError, match="full-stream analysis limits"):
        worker_module._render_tgs(
            tmp_path / "source.tgs",
            tmp_path,
            {"w": width, "h": height, "ip": 0, "op": frame_count, "fr": 60},
            MediaLimits(),
            "renderer",
            needs_repainting=False,
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("truncated-header", "header is truncated"),
        ("wrong-magic", "header is unsupported"),
        ("wrong-dimensions", "changed the native canvas dimensions"),
        ("wrong-frame-count", "full source timeline"),
        ("truncated-payload", "stream is truncated"),
        ("trailing-data", "trailing data"),
        ("hidden-rgb", "color in transparent pixels"),
    ],
)
def test_tgs_rgba_contract_rejects_invalid_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
) -> None:
    executable = tmp_path / "mojilex-rlottie-rgba.exe"
    executable.write_bytes(b"synthetic lossless renderer")
    output = tmp_path / "output"
    output.mkdir()

    def fake_popen(command: list[str], **kwargs: object) -> _FakeRendererProcess:
        del kwargs
        valid_frames = [bytes((255, 0, 0, 128)) * 4] * 2
        if case == "truncated-header":
            payload = b"MLXRGBA"
        elif case == "wrong-magic":
            payload = _rgba_stream(valid_frames, width=2, height=2, magic=b"NOTRGBA1")
        elif case == "wrong-dimensions":
            payload = _rgba_stream(valid_frames, width=2, height=2, header_width=3)
        elif case == "wrong-frame-count":
            payload = _rgba_stream(valid_frames, width=2, height=2, header_frames=3)
        elif case == "truncated-payload":
            payload = _rgba_stream(valid_frames, width=2, height=2)[:-1]
        elif case == "trailing-data":
            payload = _rgba_stream(valid_frames, width=2, height=2, trailer=b"x")
        else:
            invalid = bytes((1, 2, 3, 0)) * 4
            payload = _rgba_stream([invalid, invalid], width=2, height=2)
        return _FakeRendererProcess(payload)

    monkeypatch.setattr(worker_module.shutil, "which", lambda _: str(executable))
    monkeypatch.setattr(worker_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        worker_module, "decoder_backend_fingerprint", lambda *_args, **_kw: "0" * 64
    )
    with pytest.raises(RuntimeError, match=message):
        worker_module._render_tgs(
            tmp_path / "source.tgs",
            output,
            {"v": "5.7", "fr": 2, "ip": 0, "op": 2, "w": 2, "h": 2, "assets": []},
            MediaLimits(frames=4),
            str(executable),
            needs_repainting=False,
        )


def test_lottie2gif_is_rejected_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "lottie2gif.exe"
    executable.write_bytes(b"lossy renderer")
    monkeypatch.setattr(
        worker_module.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("lottie2gif must never execute"),
    )
    with pytest.raises(RuntimeError, match="discards the TGS alpha channel"):
        worker_module._render_tgs(
            tmp_path / "source.tgs",
            tmp_path,
            {"v": "5.7", "fr": 2, "ip": 0, "op": 2, "w": 2, "h": 2, "assets": []},
            MediaLimits(frames=4),
            str(executable),
            needs_repainting=False,
        )


@pytest.mark.asyncio
async def test_temporary_run_stream_limits_and_cleans_up(tmp_path: Path) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield b"12"
        yield b"34"

    run_path: Path
    with TemporaryMediaRun(root=tmp_path, limits=MediaLimits(max_file_bytes=3)) as run:
        run_path = run.path
        with pytest.raises(MediaLimitError):
            await run.write_stream(chunks())
    assert not run_path.exists()


@pytest.mark.asyncio
async def test_media_hash_mismatch_is_a_stable_source_changed_error(tmp_path: Path) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield b"changed-media"

    with TemporaryMediaRun(root=tmp_path) as run:
        processor = MediaProcessor(run)
        with pytest.raises(SourceChangedDuringRunError) as captured:
            await processor.process_stream(
                chunks(),
                expected_format="webp",
                expected_sha256="0" * 64,
            )

    assert captured.value.code == "SOURCE_CHANGED_DURING_RUN"


@pytest.mark.asyncio
async def test_media_processor_runs_blocking_worker_off_event_loop(tmp_path: Path) -> None:
    class ThreadRecordingWorker:
        def __init__(self) -> None:
            self.thread_id: int | None = None

        def process(self, *args: object, **kwargs: object) -> object:
            self.thread_id = threading.get_ident()
            raise MediaError("synthetic worker stop")

    async def chunks() -> AsyncIterator[bytes]:
        yield b"not-a-real-media"

    main_thread = threading.get_ident()
    worker = ThreadRecordingWorker()
    with TemporaryMediaRun(root=tmp_path) as run:
        processor = MediaProcessor(run, worker=worker)  # type: ignore[arg-type]
        with pytest.raises(MediaError, match="synthetic worker"):
            await processor.process_stream(chunks(), expected_format="webp")
    assert worker.thread_id is not None and worker.thread_id != main_thread


def test_webp_runs_in_isolated_worker_and_contact_sheet_is_bounded(tmp_path: Path) -> None:
    require_native_media_limits()
    source = tmp_path / "fixture.bin"
    Image.new("RGBA", (32, 16), (255, 255, 255, 128)).save(source, "WEBP", lossless=True)
    processed = SafeMediaWorker(MediaLimits(frames=4)).process(
        source, tmp_path / "render", expected_format="webp", needs_repainting=False
    )
    assert processed.metadata.sha256
    assert processed.metadata.width == 32
    assert processed.analysis is not None
    assert processed.analysis.analysis_scope == "full-decoded-stream"
    assert processed.analysis.fingerprint.perceptual.sample_count == 1
    assert len(processed.frame_paths) == 1
    assert len(processed.dark_frame_paths) == 1
    sheets = build_contact_sheets(
        [ContactSheetInput(identifier="mxe_synthetic", media=processed)], tmp_path
    )
    with Image.open(sheets[0].path) as sheet:
        assert sheet.width <= 4096 and sheet.height <= 4096
    assert sheets[0].mapping == {"E001": "mxe_synthetic"}
    with pytest.raises(ValueError, match="duplicate"):
        validate_response_labels(("E001",), ("E001", "E001"))


def test_render_only_webp_recreates_frames_without_running_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fixture.bin"
    Image.new("RGBA", (32, 16), (255, 255, 255, 128)).save(source, "WEBP", lossless=True)
    initial = worker_module.process(
        source,
        tmp_path / "initial",
        "webp",
        MediaLimits(frames=4),
        needs_repainting=False,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        rlottie_renderer="unused",
    )
    assert initial["analysis"] is not None

    def fail_analysis(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail("render-only resume must not recompute deterministic analysis")

    monkeypatch.setattr(worker_module, "analyze_decoded_media", fail_analysis)
    resumed = worker_module.process(
        source,
        tmp_path / "resumed",
        "webp",
        MediaLimits(frames=4),
        needs_repainting=False,
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        rlottie_renderer="unused",
        render_only=True,
        expected_dark_render=True,
    )

    assert resumed["analysis"] is None
    assert len(resumed["frame_paths"]) == 1
    assert len(resumed["dark_frame_paths"]) == 1


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg system prerequisite is not installed",
)
def test_short_webm_yields_every_deterministic_sample(tmp_path: Path) -> None:
    require_native_media_limits()
    source = tmp_path / "short.webm"
    environment = {"PATH": os.environ.get("PATH", ""), "LC_ALL": "C"}
    for name in ("SYSTEMROOT", "WINDIR"):
        if name in os.environ:
            environment[name] = os.environ[name]
    completed = subprocess.run(
        [
            "ffmpeg",
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
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")

    processed = SafeMediaWorker(MediaLimits(frames=4)).process(
        source, tmp_path / "webm-render", expected_format="webm"
    )
    assert processed.metadata.duration_ms == 200
    assert len(processed.frame_paths) == 4
    assert processed.analysis is not None
    assert processed.analysis.fingerprint.perceptual.sample_count == 16


@pytest.mark.asyncio
async def test_temporary_run_accounts_generated_outputs_without_double_counting(
    tmp_path: Path,
) -> None:
    async def chunks() -> AsyncIterator[bytes]:
        yield b"1234"

    with TemporaryMediaRun(root=tmp_path, limits=MediaLimits(max_run_temp_bytes=8)) as run:
        await run.write_stream(chunks())
        assert run.path is not None
        rendered = run.path / "rendered.png"
        rendered.write_bytes(b"5678")
        assert run.account_outputs((rendered,)) == 8
        assert run.account_outputs((rendered,)) == 8
        rendered.write_bytes(b"56789")
        with pytest.raises(MediaLimitError, match="temporary disk"):
            run.account_outputs((rendered,))
        outside = tmp_path / "outside.png"
        outside.write_bytes(b"x")
        with pytest.raises(MediaLimitError, match="escaped"):
            run.account_outputs((outside,))


@pytest.mark.asyncio
async def test_temporary_run_reservations_are_concurrency_safe(tmp_path: Path) -> None:
    release = asyncio.Event()
    both_reserved = asyncio.Event()
    reservations = 0
    reservation_lock = asyncio.Lock()

    async def chunks() -> AsyncIterator[bytes]:
        nonlocal reservations
        yield b"123"
        async with reservation_lock:
            reservations += 1
            if reservations == 2:
                both_reserved.set()
        await release.wait()
        yield b"45"

    with TemporaryMediaRun(root=tmp_path, limits=MediaLimits(max_run_temp_bytes=6)) as run:
        first = asyncio.create_task(run.write_stream(chunks()))
        second = asyncio.create_task(run.write_stream(chunks()))
        await asyncio.wait_for(both_reserved.wait(), timeout=2)
        release.set()
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
        assert sum(isinstance(value, tuple) for value in outcomes) == 1
        assert sum(isinstance(value, MediaLimitError) for value in outcomes) == 1
        assert run.bytes_written == 5


def test_webm_inspection_rejects_non_allowlisted_video_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Minimal valid EBML header containing DocType="webm".
    source = tmp_path / "codec.webm"
    source.write_bytes(b"\x1aE\xdf\xa3\x87\x42\x82\x84webm")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        payload = {
            "streams": [
                {
                    "codec_name": "h264",
                    "width": 16,
                    "height": 16,
                    "duration": "1.0",
                    "pix_fmt": "yuv420p",
                }
            ],
            "format": {"duration": "1.0", "format_name": "matroska,webm"},
        }
        return subprocess.CompletedProcess([], 0, json.dumps(payload).encode(), b"")

    monkeypatch.setattr("mojilex_cli.media.inspect.subprocess.run", fake_run)
    with pytest.raises(MediaError, match="codec"):
        inspect_webm(source, MediaLimits())


def test_webm_sampling_tries_exact_midpoints_before_any_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.webm"
    source.write_bytes(b"unused by mocked ffmpeg")
    timestamps: list[float] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        timestamp = float(command[command.index("-ss") + 1])
        timestamps.append(timestamp)
        Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(command[-1], "PNG")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(worker_module.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(worker_module.subprocess, "run", fake_run)
    frames = worker_module._render_webm(
        source,
        tmp_path,
        1000,
        MediaLimits(frames=4),
        "ffmpeg",
        codec="vp9",
        preserve_alpha=False,
    )
    try:
        assert timestamps == pytest.approx([0.125, 0.375, 0.625, 0.875])
    finally:
        for frame in frames:
            frame.close()


def test_webm_sampling_falls_back_only_when_primary_produces_no_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input.webm"
    source.write_bytes(b"unused by mocked ffmpeg")
    timestamps: list[float] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        timestamp = float(command[command.index("-ss") + 1])
        timestamps.append(timestamp)
        if len(timestamps) != 1:
            Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(command[-1], "PNG")
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(worker_module.shutil, "which", lambda _: "ffmpeg")
    monkeypatch.setattr(worker_module.subprocess, "run", fake_run)
    frames = worker_module._render_webm(
        source,
        tmp_path,
        1000,
        MediaLimits(frames=4),
        "ffmpeg",
        codec="vp9",
        preserve_alpha=False,
    )
    try:
        assert timestamps[:3] == pytest.approx([0.125, 0.075, 0.375])
    finally:
        for frame in frames:
            frame.close()


def test_worker_base_exception_invokes_tree_termination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fixture.webp"
    Image.new("RGB", (2, 2), "red").save(source, "WEBP")

    class FakeProcess:
        returncode = None

        def communicate(self, **kwargs: object) -> tuple[bytes, bytes]:
            raise KeyboardInterrupt

    process = FakeProcess()
    terminated: list[object] = []
    monkeypatch.setattr(sandbox_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(sandbox_module, "_attach_windows_job", lambda *args: object())
    monkeypatch.setattr(
        sandbox_module,
        "_terminate_worker",
        lambda child, job: terminated.append((child, job)),
    )
    with pytest.raises(KeyboardInterrupt):
        SafeMediaWorker().process(source, tmp_path / "output", expected_format="webp")
    assert terminated and terminated[0][0] is process


def test_worker_timeout_reports_configured_limit_and_terminates_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fixture.webp"
    Image.new("RGB", (2, 2), "red").save(source, "WEBP")

    class FakeProcess:
        returncode = None

        def communicate(self, *, timeout: float) -> tuple[bytes, bytes]:
            assert timeout == 2.5
            raise subprocess.TimeoutExpired("worker", timeout)

    process = FakeProcess()
    job = object()
    terminated: list[object] = []
    monkeypatch.setattr(sandbox_module.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(sandbox_module, "_attach_windows_job", lambda *args: job)
    monkeypatch.setattr(
        sandbox_module,
        "_terminate_worker",
        lambda child, attached_job: terminated.append((child, attached_job)),
    )
    with pytest.raises(MediaError, match=r"exceeded the 2\.5 second wall-time limit"):
        SafeMediaWorker(MediaLimits(worker_timeout_seconds=2.5)).process(
            source, tmp_path / "output", expected_format="webp"
        )
    assert terminated == [(process, job)]


def test_unix_termination_targets_worker_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    class FakeProcess:
        pid = 123

        def poll(self) -> None:
            return None

        def communicate(self, **kwargs: object) -> tuple[bytes, bytes]:
            return b"", b""

        def kill(self) -> None:
            raise AssertionError("direct-child kill must not be used on Unix")

    monkeypatch.setattr(
        sandbox_module.os,
        "killpg",
        lambda pid, sig: calls.append((pid, sig)),
        raising=False,
    )
    sandbox_module._terminate_worker(FakeProcess(), None, platform="posix")
    assert calls == [(123, getattr(sandbox_module.signal, "SIGKILL", 9))]
