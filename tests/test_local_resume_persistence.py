from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.github import RepositoryRef
from mojilex_cli.pipeline import runner
from test_pipeline_workspaces import _source_repository


def test_remote_local_result_survives_temporary_source_cleanup(tmp_path):
    with TemporaryDirectory(dir=tmp_path) as raw:
        source, _ = _source_repository(Path(raw))
        workspace = runner.RepositoryWorkspace(
            root=source, target=RepositoryRef.parse("MojiLex/mojilex"), temporary=True
        )
        retained = runner._persistent_local_workspace(
            workspace, runs_dir=tmp_path / "runs", run_id="mlxrun_" + "a" * 32, base_branch="main"
        )
        (retained.root / "dataset.json").write_text('{"saved":true}\n', encoding="utf-8")
    assert not source.exists()
    assert not retained.temporary
    assert (retained.root / "dataset.json").read_text(encoding="utf-8") == '{"saved":true}\n'


def test_explicit_local_checkout_is_not_relocated(tmp_path):
    workspace = runner.RepositoryWorkspace(
        root=tmp_path, target=RepositoryRef.parse("MojiLex/mojilex"), temporary=False
    )
    assert (
        runner._persistent_local_workspace(
            workspace, runs_dir=tmp_path / "runs", run_id="mlxrun_" + "a" * 32, base_branch="main"
        )
        is workspace
    )


@pytest.mark.parametrize("publish,staged", [("local", False), ("local", True), ("pr", False)])
async def test_resume_preserves_local_destination_and_paid_budget(
    tmp_path, monkeypatch, publish, staged
):
    original = tmp_path / "original"
    staging = tmp_path / "saved-staging"
    parameters = {
        "sources": [],
        "languages": ["ru", "en"],
        "repository": str(original),
        "publish": publish,
        "max_ai_requests": 2,
    }
    if staged:
        parameters["staging_repository"] = str(staging)
    checkpoint = SimpleNamespace(
        run_id="mlxrun_test",
        command="add",
        target_repository="MojiLex/mojilex",
        elements={},
        safe_parameters=parameters,
        ai_requests_used=2,
        ai_cost_reserved_usd=0,
    )
    monkeypatch.setattr(runner, "load_config", lambda: SimpleNamespace(runs_dir=tmp_path / "runs"))
    monkeypatch.setattr(
        runner, "RunStore", lambda _: SimpleNamespace(load_for_resume=lambda *a, **k: checkpoint)
    )
    captured = {}

    async def execute(sources, options, **kwargs):
        captured["options"] = options
        captured["checkpoint"] = kwargs["resume_checkpoint"]
        return CommandResult()

    monkeypatch.setattr(runner, "_run_add", execute)
    await runner.run_resume(checkpoint.run_id)
    expected = str(staging if staged else original) if publish == "local" else "MojiLex/mojilex"
    assert captured["options"].repository == expected
    assert captured["options"].max_ai_requests == 2
    assert captured["checkpoint"].ai_requests_used == 2
