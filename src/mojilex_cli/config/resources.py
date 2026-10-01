"""Choose operation concurrency from a startup hardware snapshot."""

from __future__ import annotations

import ctypes
import os
import shutil
import struct
import tempfile

from .models import MojiLexConfig

_WORKER_MEMORY = 512 * 1024**2


def _physical_cores_from_windows_topology(data: bytes) -> int | None:
    """Count processor-core records from SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX."""
    offset = 0
    cores = 0
    while offset < len(data):
        if len(data) - offset < 8:
            return None
        relationship, size = struct.unpack_from("<II", data, offset)
        if size < 8 or size > len(data) - offset:
            return None
        if relationship == 0:  # RelationProcessorCore
            if size < 32:
                return None
            cores += 1
        offset += size
    return cores or None


def physical_cpu_count() -> int | None:
    """Probe Windows physical cores; callers retain logical-CPU fallback elsewhere."""
    if os.name != "nt":
        return None
    try:
        windll = getattr(ctypes, "windll", None)
        if windll is None:
            return None
        query = windll.kernel32.GetLogicalProcessorInformationEx
        query.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        query.restype = ctypes.c_int
        length = ctypes.c_uint32()
        query(0, None, ctypes.byref(length))
        if not 8 <= length.value <= 1024**2:
            return None
        buffer = ctypes.create_string_buffer(length.value)
        if not query(0, buffer, ctypes.byref(length)) or length.value > len(buffer):
            return None
        return _physical_cores_from_windows_topology(buffer.raw[: length.value])
    except (AttributeError, OSError, ValueError):
        return None


def available_temp_disk_bytes() -> int | None:
    """Return free space on the actual temporary volume."""
    try:
        return shutil.disk_usage(tempfile.gettempdir()).free
    except OSError:
        return None


def available_memory_bytes() -> int | None:
    """Return currently available RAM, without external commands or dependencies."""
    try:
        if os.name == "nt":

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_uint32),
                    ("load", ctypes.c_uint32),
                    *[
                        (name, ctypes.c_uint64)
                        for name in (
                            "total",
                            "available",
                            "total_page",
                            "available_page",
                            "total_virtual",
                            "available_virtual",
                            "extended",
                        )
                    ],
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            windll = getattr(ctypes, "windll", None)
            if windll is None:
                return None
            if windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.available)
            return None
        sysconf = getattr(os, "sysconf", None)
        if sysconf is None:
            return None
        return int(sysconf("SC_AVPHYS_PAGES")) * int(sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None


def resolved_resource_config(config: MojiLexConfig) -> MojiLexConfig:
    """Raise automatic capacity without editing persisted settings or request budgets.

    Explicit larger values remain usable; manual mode uses exact configured values.
    This is a startup sizing policy, not a provider quota or an adaptive scheduler.
    """
    if config.processing.performance_mode == "manual":
        return config
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    available = available_memory_bytes()
    # Native rasterizers use internal worker threads too. SMT siblings are not
    # separate decoder capacity; on Windows prefer physical cores when known.
    render_target = min(cpu_count, physical_cpu_count() or cpu_count)
    if available is not None:
        render_target = min(render_target, max(1, available * 3 // 4 // _WORKER_MEMORY))
    # Keep preparation focused on enough packs to feed the decoders. Spreading
    # them across many more packs delays completion of each whole pack. Provider
    # requests still overlap with preparation and use their separate I/O target.
    overlap_target = render_target * 4
    temp_bytes = config.processing.max_temp_bytes
    free_disk = available_temp_disk_bytes()
    if free_disk is not None:
        if available is not None:
            temp_bytes = max(temp_bytes, available * 2)
        temp_bytes = max(1, min(temp_bytes, free_disk // 4))
    return config.model_copy(
        update={
            "processing": config.processing.model_copy(
                update={
                    "render_concurrency": max(config.processing.render_concurrency, render_target),
                    "pack_concurrency": max(config.processing.pack_concurrency, render_target),
                    "max_temp_bytes": temp_bytes,
                }
            ),
            "telegram": config.telegram.model_copy(
                update={
                    "download_concurrency": max(
                        config.telegram.download_concurrency, cpu_count * 4
                    ),
                }
            ),
            "ai": config.ai.model_copy(
                update={
                    "ai_concurrency": max(config.ai.ai_concurrency, overlap_target),
                }
            ),
        }
    )
