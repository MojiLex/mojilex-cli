from contextlib import nullcontext
from decimal import Decimal
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mojilex_cli import cli
from mojilex_cli.commands import packs, workflow
from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.pipeline import runner


@pytest.fixture(autouse=True)
def saved_interrupted_run(monkeypatch):
    monkeypatch.setattr(
        packs,
        "resolve_pack_run",
        lambda selector, **kwargs: SimpleNamespace(
            run_id=selector,
            status="interrupted",
            command="add",
        ),
    )


@pytest.mark.parametrize("concurrency", [None, 1, 8, 32, 128])
def test_resume_cli_passes_optional_download_concurrency(monkeypatch, concurrency):
    captured = {}

    def resume(run_id, **kwargs):
        captured.update(kwargs)
        return CommandResult()

    monkeypatch.setattr(workflow, "resume_command", resume)
    monkeypatch.setattr(cli, "_with_runtime_secrets", lambda action, **kwargs: action())
    args = ["resume", "mlxrun_test", "--ai-concurrency", "3", "--non-interactive"]
    if concurrency is not None:
        args.extend(["--download-concurrency", str(concurrency)])
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert captured["download_concurrency"] == concurrency
    assert captured["ai_concurrency"] == 3


@pytest.mark.parametrize("concurrency", [0, -1])
def test_resume_cli_rejects_invalid_download_concurrency(monkeypatch, concurrency):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid concurrency must not start a run")

    monkeypatch.setattr(workflow, "resume_command", unexpected)
    result = CliRunner().invoke(
        cli.app, ["resume", "mlxrun_test", "--download-concurrency", str(concurrency)]
    )
    assert result.exit_code == 2, result.output


@pytest.mark.parametrize("concurrency", [None, 8])
def test_resume_command_forwards_only_explicit_download_override(monkeypatch, concurrency):
    captured = {}

    def resume(run_id, **kwargs):
        captured.update(kwargs)
        return CommandResult()

    monkeypatch.setattr(workflow, "run_resume_sync", resume)
    workflow.resume_command("mlxrun_test", download_concurrency=concurrency)
    assert captured == {
        "confirmation": None,
        "unknown_cost_confirmation": None,
        **({"download_concurrency": concurrency} if concurrency is not None else {}),
    }


def test_sync_resume_forwards_download_override_under_run_lock(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(runner, "load_config", lambda: SimpleNamespace(runs_dir=tmp_path))
    monkeypatch.setattr(
        runner, "RunStore", lambda path: SimpleNamespace(execution_lock=lambda _: nullcontext())
    )

    async def resume(run_id, **kwargs):
        captured.update(kwargs)
        return CommandResult()

    monkeypatch.setattr(runner, "run_resume", resume)
    runner.run_resume_sync("mlxrun_test", download_concurrency=8)
    assert captured["download_concurrency"] == 8
    assert captured["ai_concurrency"] is None


@pytest.mark.parametrize("command", ["import", "describe", "add"])
@pytest.mark.parametrize("concurrency", [None, 8])
async def test_resume_download_override_preserves_saved_ai_budget(
    tmp_path, monkeypatch, command, concurrency
):
    checkpoint = SimpleNamespace(
        run_id="mlxrun_test",
        command=command,
        target_repository=str(tmp_path),
        elements={},
        safe_parameters={
            "sources": [],
            "languages": ["ru", "en"],
            "staging_repository": str(tmp_path),
            "download_concurrency": 2,
            "ai_concurrency": 3,
            "max_ai_requests": 37,
            "max_cost_usd": "1.25",
        },
        ai_requests_used=14,
        ai_cost_reserved_usd=Decimal("0.25"),
    )
    captured = {}
    monkeypatch.setattr(runner, "load_config", lambda: SimpleNamespace(runs_dir=tmp_path))
    monkeypatch.setattr(
        runner,
        "RunStore",
        lambda path: SimpleNamespace(load_for_resume=lambda *args, **kwargs: checkpoint),
    )

    async def execute(sources, options, **kwargs):
        captured["options"] = options
        captured["checkpoint"] = kwargs["resume_checkpoint"]
        return CommandResult()

    monkeypatch.setattr(runner, "_run_add", execute)
    monkeypatch.setattr(runner, "_run_import", execute)
    await runner.run_resume(checkpoint.run_id, download_concurrency=concurrency)
    options = captured["options"]
    assert options.download_concurrency == (2 if concurrency is None else concurrency)
    assert options.ai_concurrency == 3
    assert options.max_ai_requests == 37
    assert options.max_cost_usd == Decimal("1.25")
    assert captured["checkpoint"] is checkpoint
    assert checkpoint.ai_requests_used == 14
    assert checkpoint.ai_cost_reserved_usd == Decimal("0.25")
