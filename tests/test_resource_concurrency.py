"""Automatic startup sizing and uncapped manual settings."""

import struct

import pytest

from mojilex_cli.commands.settings import settings_command, update_setting_command
from mojilex_cli.config import MojiLexConfig, load_config, resources
from mojilex_cli.media.models import MediaLimits


@pytest.fixture(autouse=True)
def fixed_temp_disk(monkeypatch):
    monkeypatch.setattr(resources, "available_temp_disk_bytes", lambda: 256 * 1024**3)
    monkeypatch.setattr(resources, "physical_cpu_count", lambda: None)


@pytest.mark.parametrize(
    "key", ["download_concurrency", "render_concurrency", "pack_concurrency", "ai_concurrency"]
)
def test_high_concurrency_survives_settings_roundtrip(tmp_path, key):
    paths = {"project_path": tmp_path / "project.toml", "user_path": tmp_path / "user.toml"}
    update_setting_command(key, "128", **paths, environment={})
    row = next(
        row
        for row in settings_command(**paths, environment={}).result["settings"]
        if row["key"] == key
    )
    assert row["value"] == 128
    assert row["maximum"] is None
    config = load_config(**paths, environment={})
    assert config.ai.max_ai_requests == 100


@pytest.mark.parametrize(("memory", "render"), [(32 * 1024**3, 12), (2 * 1024**3, 3), (None, 12)])
def test_auto_uses_hardware_snapshot_without_changing_budget(monkeypatch, memory, render):
    monkeypatch.setattr(resources.os, "process_cpu_count", lambda: 12, raising=False)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: memory)
    original = MojiLexConfig()
    resolved = resources.resolved_resource_config(original)
    assert resolved.processing.render_concurrency == render
    assert resolved.processing.pack_concurrency == render * 4
    assert resolved.telegram.download_concurrency == 48
    assert resolved.ai.ai_concurrency == render * 4
    assert resolved.ai.max_ai_requests == original.ai.max_ai_requests
    assert original.processing.render_concurrency == 2


def test_auto_io_overlap_does_not_multiply_cpu_decoder_processes(monkeypatch):
    monkeypatch.setattr(resources.os, "process_cpu_count", lambda: 12, raising=False)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 32 * 1024**3)
    config = MojiLexConfig(ai={"max_ai_requests": None})

    resolved = resources.resolved_resource_config(config)

    assert resolved.processing.pack_concurrency == 48
    assert resolved.ai.ai_concurrency == 48
    assert resolved.processing.render_concurrency == 12
    assert resolved.telegram.download_concurrency == 48
    assert resolved.ai.max_ai_requests is None


def test_manual_does_not_probe_hardware(monkeypatch):
    config = MojiLexConfig(processing={"performance_mode": "manual", "render_concurrency": 64})
    monkeypatch.setattr(
        resources, "available_memory_bytes", lambda: pytest.fail("manual hardware probe")
    )
    monkeypatch.setattr(
        resources, "available_temp_disk_bytes", lambda: pytest.fail("manual disk probe")
    )
    monkeypatch.setattr(
        resources, "physical_cpu_count", lambda: pytest.fail("manual topology probe")
    )
    assert resources.resolved_resource_config(config) is config


@pytest.mark.parametrize(
    ("logical", "physical", "memory", "render"),
    [
        (12, 6, 32 * 1024**3, 6),
        (4, 6, 32 * 1024**3, 4),
        (12, 6, 2 * 1024**3, 3),
    ],
)
def test_auto_bounds_decoders_by_physical_cores_and_available_resources(
    monkeypatch, logical, physical, memory, render
):
    monkeypatch.setattr(resources.os, "process_cpu_count", lambda: logical, raising=False)
    monkeypatch.setattr(resources, "physical_cpu_count", lambda: physical)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: memory)
    original = MojiLexConfig(ai={"max_ai_requests": None, "max_cost_usd": 7})
    resolved = resources.resolved_resource_config(original)
    assert resolved.processing.render_concurrency == render
    assert resolved.processing.pack_concurrency == render * 4
    assert resolved.ai.ai_concurrency == render * 4
    assert resolved.telegram.download_concurrency == logical * 4
    assert resolved.ai.max_ai_requests is None
    assert resolved.ai.max_cost_usd == 7
    assert original.processing.render_concurrency == 2


def test_windows_topology_counts_cores_instead_of_smt_threads():
    # Each core record includes a GROUP_AFFINITY; an SMT core still counts once.
    core = struct.pack("<II", 0, 48) + bytes(40)
    other = struct.pack("<II", 3, 16) + bytes(8)
    assert resources._physical_cores_from_windows_topology(core + other + core) == 2


@pytest.mark.parametrize(
    "data",
    [
        b"",
        bytes(7),
        struct.pack("<II", 0, 0),
        struct.pack("<II", 0, 48),
        struct.pack("<II", 0, 8),
        struct.pack("<II", 0, 48) + bytes(40) + b"x",
    ],
)
def test_invalid_windows_topology_falls_back(data):
    assert resources._physical_cores_from_windows_topology(data) is None


def test_auto_keeps_explicit_larger_values(monkeypatch):
    monkeypatch.setattr(resources.os, "process_cpu_count", lambda: 2, raising=False)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 8 * 1024**3)
    config = MojiLexConfig(
        processing={"render_concurrency": 32, "pack_concurrency": 64},
        telegram={"download_concurrency": 128},
        ai={"ai_concurrency": 64},
    )
    resolved = resources.resolved_resource_config(config)
    assert resolved.processing.render_concurrency == 32
    assert resolved.processing.pack_concurrency == 64
    assert resolved.telegram == config.telegram
    assert resolved.ai == config.ai


@pytest.mark.parametrize(
    ("memory", "disk", "configured", "expected"),
    [(16, 256, 2, 32), (16, 16, 2, 4), (16, 16, 12, 4), (None, 256, 2, 2), (16, None, 2, 2)],
)
def test_auto_workspace_scales_with_resources_and_reserves_disk(
    monkeypatch, memory, disk, configured, expected
):
    gib = 1024**3
    monkeypatch.setattr(
        resources, "available_memory_bytes", lambda: memory * gib if memory else None
    )
    monkeypatch.setattr(
        resources, "available_temp_disk_bytes", lambda: disk * gib if disk else None
    )
    config = MojiLexConfig(processing={"max_temp_bytes": configured * gib})
    resolved = resources.resolved_resource_config(config)
    assert resolved.processing.max_temp_bytes == expected * gib
    assert MediaLimits(max_run_temp_bytes=expected * gib).max_run_temp_bytes == expected * gib
    assert config.processing.max_temp_bytes == configured * gib
