from decimal import Decimal
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mojilex_cli import cli
from mojilex_cli.cli import app
from mojilex_cli.commands import packs, workflow
from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.config import MojiLexConfig, ProcessingConfig
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import RunStore, new_checkpoint


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


@pytest.mark.parametrize("command", ["describe", "resume"])
@pytest.mark.parametrize("concurrency", [1, 4, 16])
def test_cli_forwards_ai_concurrency(monkeypatch, command, concurrency):
    captured = {}

    def execute(target, **kwargs):
        captured.update(kwargs)
        return CommandResult()

    monkeypatch.setattr(workflow, f"{command}_command", execute)
    monkeypatch.setattr(cli, "_with_runtime_secrets", lambda action, **kwargs: action())
    result = CliRunner().invoke(
        app,
        [command, "mlxrun_test", "--ai-concurrency", str(concurrency), "--non-interactive"],
    )
    assert result.exit_code == 0, result.output
    assert captured["ai_concurrency"] == concurrency


@pytest.mark.parametrize("command", ["describe", "resume"])
@pytest.mark.parametrize("concurrency", [0, 17])
def test_cli_rejects_out_of_range_ai_concurrency(monkeypatch, command, concurrency):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid concurrency must be rejected before starting work")

    monkeypatch.setattr(workflow, f"{command}_command", unexpected)
    result = CliRunner().invoke(app, [command, "mlxrun_test", "--ai-concurrency", str(concurrency)])
    assert result.exit_code == 2, result.output


@pytest.mark.parametrize("concurrency", [None, 4])
def test_describe_command_preserves_optional_concurrency(monkeypatch, concurrency):
    captured = {}

    def describe(run_id, options):
        captured["options"] = options
        return CommandResult()

    monkeypatch.setattr(workflow, "run_describe", describe)
    workflow.describe_command(
        ["mlxrun_test"],
        provider=None,
        model=None,
        max_ai_requests=None,
        max_cost_usd=None,
        allow_unknown_cost=False,
        ai_concurrency=concurrency,
    )
    assert captured["options"].ai_concurrency == concurrency
    assert captured["options"].max_ai_requests is None
    assert captured["options"].max_cost_usd is None


def test_resume_command_forwards_explicit_concurrency(monkeypatch):
    captured = {}

    def resume(run_id, **kwargs):
        captured.update(kwargs)
        return CommandResult()

    monkeypatch.setattr(workflow, "run_resume_sync", resume)
    workflow.resume_command("mlxrun_test", ai_concurrency=4)
    assert captured["ai_concurrency"] == 4
    assert "max_ai_requests" not in captured
    assert "max_cost_usd" not in captured


@pytest.mark.parametrize("command", ["describe", "resume"])
@pytest.mark.parametrize("concurrency", [None, 4])
async def test_pipeline_concurrency_override_preserves_saved_budget(
    tmp_path, monkeypatch, command, concurrency
):
    checkpoint = new_checkpoint(
        command="describe",
        target_repository=str(tmp_path),
        cli_version="0.2.0",
        schema_version=runner.SCHEMA_VERSION,
        base_revision="a" * 40,
        safe_parameters={
            "sources": ["https://t.me/addemoji/TestPack"],
            "languages": ["ru", "en"],
            "staging_repository": str(tmp_path),
            "ai_concurrency": 2,
            "download_concurrency": 15,
            "file_analysis_mode": "prepare_all",
            "max_ai_requests": 37,
            "max_cost_usd": "1.25",
        },
    ).model_copy(update={"ai_requests_used": 14, "ai_cost_reserved_usd": Decimal("0.25")})
    captured = {}
    config = MojiLexConfig(
        runs_dir=tmp_path / "runs", processing=ProcessingConfig(official_pack_policy="allow")
    )
    RunStore(config.runs_dir).save(checkpoint)
    monkeypatch.setattr(runner, "load_config", lambda: config)

    async def run_add(sources, options, **kwargs):
        captured["options"] = options
        captured["checkpoint"] = kwargs["resume_checkpoint"]
        return CommandResult()

    monkeypatch.setattr(runner, "_run_add", run_add)
    if command == "describe":
        await runner._run_describe(
            checkpoint.run_id,
            runner.PipelineOptions(
                ai_concurrency=concurrency, download_concurrency=24, file_analysis_mode="fast"
            ),
        )
    else:
        await runner.run_resume(checkpoint.run_id, ai_concurrency=concurrency)

    assert captured["options"].ai_concurrency == (2 if concurrency is None else concurrency)
    assert captured["options"].download_concurrency == (24 if command == "describe" else 15)
    assert captured["options"].file_analysis_mode == (
        "fast" if command == "describe" else "prepare_all"
    )
    assert captured["options"].max_ai_requests == 37
    assert captured["options"].max_cost_usd == Decimal("1.25")
    assert captured["checkpoint"].run_id == checkpoint.run_id
    assert captured["checkpoint"].ai_requests_used == checkpoint.ai_requests_used
    assert captured["checkpoint"].ai_cost_reserved_usd == checkpoint.ai_cost_reserved_usd
    assert checkpoint.ai_requests_used == 14
    assert checkpoint.ai_cost_reserved_usd == Decimal("0.25")
