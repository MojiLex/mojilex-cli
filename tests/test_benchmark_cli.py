from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mojilex_cli.cli import app
from mojilex_cli.commands import benchmark as benchmark_commands
from mojilex_cli.commands.runtime import CommandResult


def test_benchmark_dedupe_cli_has_exact_manifest_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "dedupe.json"
    manifest.write_text("{}", encoding="utf-8")
    seen: list[Path] = []
    monkeypatch.setattr(
        benchmark_commands,
        "benchmark_dedupe_command",
        lambda path: (seen.append(path), CommandResult(result={"passed": True}))[1],
    )

    result = CliRunner().invoke(app, ["benchmark-dedupe", "--manifest", str(manifest), "--json"])

    assert result.exit_code == 0, result.output
    assert seen == [manifest]
    assert json.loads(result.stdout)["command"] == "benchmark-dedupe"


def test_benchmark_model_cli_requires_explicit_target_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "model.json"
    manifest.write_text("{}", encoding="utf-8")
    seen: list[tuple[str, str, Path]] = []

    def command(*, provider_name: str, model_id: str, benchmark_manifest: Path) -> CommandResult:
        seen.append((provider_name, model_id, benchmark_manifest))
        return CommandResult(result={"passed": True})

    monkeypatch.setattr(benchmark_commands, "benchmark_model_command", command)
    result = CliRunner().invoke(
        app,
        [
            "benchmark-model",
            "--provider",
            "gemini",
            "--model",
            "gemini-test",
            "--benchmark-manifest",
            str(manifest),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    assert seen == [("gemini", "gemini-test", manifest)]
    assert json.loads(result.stdout)["command"] == "benchmark-model"
