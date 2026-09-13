"""Cross-platform process isolation for untrusted media decoders."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from mojilex_cli.analysis import DeterministicMediaAnalysis

from .inspect import sniff_format, validate_input_file
from .models import (
    HARD_MAX_WORKER_MEMORY,
    MediaDependencyError,
    MediaError,
    MediaLimits,
    MediaMetadata,
    MediaRenderError,
    ProcessedMedia,
)
from .resume import _composition_png, _read

_MAX_WORKER_OUTPUT = 1024 * 1024


class SafeMediaWorker:
    def __init__(
        self,
        limits: MediaLimits | None = None,
        *,
        ffmpeg: str | None = None,
        ffprobe: str | None = None,
        rlottie_renderer: str | None = None,
    ) -> None:
        self.limits = limits or MediaLimits()
        self.ffmpeg = ffmpeg or os.environ.get("MOJILEX_FFMPEG", "ffmpeg")
        self.ffprobe = ffprobe or os.environ.get("MOJILEX_FFPROBE", "ffprobe")
        self.rlottie_renderer = rlottie_renderer or _default_rlottie_renderer()

    def process(
        self,
        source: Path,
        output_dir: Path,
        *,
        expected_format: str,
        needs_repainting: bool = False,
        cached_analysis: DeterministicMediaAnalysis | None = None,
        expected_dark_render: bool | None = None,
    ) -> ProcessedMedia:
        if (cached_analysis is None) != (expected_dark_render is None):
            raise MediaError(
                "cached analysis and expected render context must be provided together"
            )
        source = source.resolve(strict=True)
        validate_input_file(source, self.limits)
        actual_format = sniff_format(source)
        # Telegram's static flag does not distinguish PNG uploads from WebP.
        # Preserve the actual format and original hash, never relabel its bytes.
        if expected_format == "webp" and actual_format == "png":
            expected_format = "png"
        if actual_format != expected_format:
            raise MediaError("media content does not match its expected format")
        if output_dir.exists():
            raise MediaError("worker output directory must not already exist")
        if output_dir.parent.resolve(strict=True) != source.parent.resolve(strict=True):
            raise MediaError("worker input and output must share the private run directory")
        command = [
            sys.executable,
            "-m",
            "mojilex_cli.media.worker",
            "--source",
            str(source),
            "--output",
            str(output_dir),
            "--format",
            expected_format,
            "--limits",
            self.limits.model_dump_json(),
            "--ffmpeg",
            self.ffmpeg,
            "--ffprobe",
            self.ffprobe,
            "--rlottie-renderer",
            self.rlottie_renderer,
        ]
        if needs_repainting:
            command.append("--needs-repainting")
        if cached_analysis is not None:
            command.append("--render-only")
            if expected_dark_render:
                command.append("--dark-render")
        environment = _worker_environment(source.parent)
        kwargs: dict[str, Any] = {}
        if os.name != "nt":
            command = [
                sys.executable,
                # Do not prepend media/: its inspect.py shadows the stdlib module.
                "-P",
                str(Path(__file__).with_name("unix_worker.py")),
                str(self.limits.worker_memory_bytes),
                *command[3:],
            ]
            kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen(
                command,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                **kwargs,
            )
        except OSError as exc:
            raise MediaDependencyError("cannot start isolated Python media worker") from exc
        job = _attach_windows_job(process, self.limits.worker_memory_bytes)
        if os.name == "nt" and job is None:
            _terminate_worker(process, None)
            raise MediaDependencyError(
                "Windows Job Object limits could not be applied; media backend is unsafe"
            )
        try:
            stdout, stderr = process.communicate(timeout=self.limits.worker_timeout_seconds)
        except subprocess.TimeoutExpired:
            _terminate_worker(process, job)
            job = None
            raise MediaRenderError(
                f"media worker exceeded the {self.limits.worker_timeout_seconds:g} "
                "second wall-time limit"
            ) from None
        except BaseException:
            _terminate_worker(process, job)
            job = None
            raise
        finally:
            if job is not None:
                job.close()
        if len(stdout) > _MAX_WORKER_OUTPUT or len(stderr) > _MAX_WORKER_OUTPUT:
            raise MediaRenderError("media worker output exceeded the diagnostic limit")
        if process.returncode != 0:
            detail = stderr.decode("utf-8", errors="replace")[:500]
            # Worker has no secrets or URLs; still avoid leaking local paths.
            detail = detail.replace(str(source), "<media>").replace(str(source.parent), "<temp>")
            if "isolated media worker limits unavailable" in detail or (
                "required" in detail and ("ffmpeg" in detail or "rlottie" in detail)
            ):
                raise MediaDependencyError(detail)
            raise MediaRenderError(detail or "media worker failed")
        try:
            payload = json.loads(stdout)
            metadata = MediaMetadata.model_validate(payload["metadata"])
            raw_analysis = payload["analysis"]
            if cached_analysis is None:
                analysis = DeterministicMediaAnalysis.model_validate(raw_analysis)
            elif raw_analysis is not None:
                raise ValueError("render-only worker unexpectedly returned analysis")
            else:
                analysis = cached_analysis
            frames = _validated_paths(payload["frame_paths"], output_dir)
            dark_frames = _validated_paths(payload["dark_frame_paths"], output_dir)
            tile_path = None
            tile_sha256 = None
            tile = payload.get("composition_tile")
            if tile is not None:
                if (
                    expected_format not in {"webp", "png"}
                    or needs_repainting
                    or not isinstance(tile, dict)
                    or set(tile) != {"path", "sha256"}
                ):
                    raise ValueError("unexpected composition tile")
                tile_path = _validated_paths([tile["path"]], output_dir)[0]
                if tile_path.name != "composition-tile.png":
                    raise ValueError("unexpected composition tile name")
                data = _read(Path(tile["path"]), 512 * 1024)
                _composition_png(data, (metadata.width, metadata.height))
                tile_sha256 = hashlib.sha256(data).hexdigest()
                if tile_sha256 != tile["sha256"]:
                    raise ValueError("composition tile checksum mismatch")
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MediaRenderError("media worker returned an invalid manifest") from exc
        expected_count = 1 if expected_format in {"webp", "png"} else self.limits.frames
        if len(frames) != expected_count or (dark_frames and len(dark_frames) != expected_count):
            raise MediaRenderError("media worker returned an unexpected frame count")
        if expected_dark_render is not None and bool(dark_frames) is not expected_dark_render:
            raise MediaRenderError("media worker returned a different background render context")
        return ProcessedMedia(
            metadata=metadata,
            analysis=analysis,
            frame_paths=frames,
            dark_frame_paths=dark_frames,
            rendered_frame_count=len(frames),
            has_dark_render=bool(dark_frames),
            composition_tile_path=tile_path,
            composition_tile_sha256=tile_sha256,
        )


def _default_rlottie_renderer() -> str:
    configured = os.environ.get("MOJILEX_RLOTTIE_RGBA")
    if configured:
        return configured
    owned = (
        Path.home()
        / ".local"
        / "bin"
        / ("mojilex-rlottie-rgba.exe" if os.name == "nt" else "mojilex-rlottie-rgba")
    )
    return str(owned) if owned.is_file() else "mojilex-rlottie-rgba"


def _validated_paths(values: object, root: Path) -> tuple[Path, ...]:
    if not isinstance(values, list):
        raise ValueError("paths are not a list")
    resolved_root = root.resolve(strict=True)
    result: list[Path] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("path is not a string")
        path = Path(value).resolve(strict=True)
        if path.is_symlink() or path.parent != resolved_root or path.suffix.lower() != ".png":
            raise ValueError("worker path escaped its output directory")
        result.append(path)
    return tuple(result)


def _worker_environment(temp_dir: Path) -> dict[str, str]:
    package_root = Path(__file__).resolve().parents[2]
    result = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONPATH": str(package_root),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
        "LC_ALL": "C",
        "TMP": str(temp_dir),
        "TEMP": str(temp_dir),
    }
    # Python reads these non-secret architecture fields for platform.machine()
    # on Windows; dropping them changes the worker's decoder fingerprint.
    for name in ("SYSTEMROOT", "WINDIR", "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432"):
        if name in os.environ:
            result[name] = os.environ[name]
    return result


def _windows_kernel32() -> Any:
    """Resolve Windows-only APIs at runtime, with pointer-sized handle signatures."""
    import ctypes
    from ctypes import wintypes

    ctypes_api = cast(Any, ctypes)
    kernel32 = ctypes_api.windll.kernel32
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    ]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


class _WindowsJob:
    def __init__(self, handle: int) -> None:
        self.handle = handle

    def close(self) -> None:
        if self.handle:
            _windows_kernel32().CloseHandle(self.handle)
            self.handle = 0


def _attach_windows_job(process: subprocess.Popen[bytes], memory_bytes: int) -> _WindowsJob | None:
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount",
                    "WriteOperationCount",
                    "OtherOperationCount",
                    "ReadTransferCount",
                    "WriteTransferCount",
                    "OtherTransferCount",
                )
            ]

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BASIC_LIMIT),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = _windows_kernel32()
        handle = kernel32.CreateJobObjectW(None, None)
        if not handle:
            return None
        info = EXTENDED_LIMIT()
        # JOB_OBJECT_LIMIT_JOB_MEMORY is cumulative across the entire process tree.
        # A per-process cap would let decoder children each consume the full budget.
        info.BasicLimitInformation.LimitFlags = 0x200 | 0x2000  # job memory, kill on close
        info.JobMemoryLimit = min(memory_bytes, HARD_MAX_WORKER_MEMORY)
        process_handle = cast(Any, process)._handle
        ok = kernel32.SetInformationJobObject(
            handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ) and kernel32.AssignProcessToJobObject(handle, wintypes.HANDLE(process_handle))
        if not ok:
            kernel32.CloseHandle(handle)
            return None
        return _WindowsJob(handle)
    except (AttributeError, OSError):
        return None


def _terminate_worker(
    process: subprocess.Popen[bytes], job: _WindowsJob | None, *, platform: str | None = None
) -> None:
    """Terminate the whole isolated worker tree and reap the direct child."""

    platform = os.name if platform is None else platform
    if platform == "nt" and job is not None:
        # Closing a kill-on-close Job Object terminates all attached descendants.
        job.close()
    try:
        if process.poll() is None:
            if platform == "nt":
                process.kill()
            else:
                kill_group = cast(Callable[[int, int], None], vars(os)["killpg"])
                kill_group(process.pid, int(vars(signal).get("SIGKILL", 9)))
    except (OSError, ProcessLookupError):
        pass
    try:
        process.communicate(timeout=5)
    except BaseException:
        # Cleanup must never replace the timeout/cancellation which brought us here.
        try:
            process.kill()
        except OSError:
            pass


def hard_resource_limits_available() -> bool:
    if os.name != "nt":
        # Constants alone do not prove enforcement is available: Darwin can
        # reject a 512 MiB address-space cap even in a fresh interpreter. Probe
        # the exact bootstrap and all limits without importing any decoder.
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-P",
                    str(Path(__file__).with_name("unix_worker.py")),
                    str(HARD_MAX_WORKER_MEMORY),
                    "--probe",
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=5,
                check=False,
                env=_worker_environment(Path.cwd()),
            )
            return completed.returncode == 0 and completed.stdout == b"MOJILEX_RESOURCE_LIMITS_OK\n"
        except (OSError, subprocess.TimeoutExpired):
            return False
    # A real attach is verified during the fixture probe; availability of APIs is a first check.
    try:
        return bool(_windows_kernel32().CreateJobObjectW)
    except (AttributeError, OSError):
        return False
