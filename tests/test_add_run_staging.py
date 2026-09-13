from __future__ import annotations

import shutil
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from mojilex_cli.ai import RequestBudget
from mojilex_cli.cache import CacheStore
from mojilex_cli.commands.runtime import CommandError, CommandResult
from mojilex_cli.config import AIConfig, MojiLexConfig, ProcessingConfig
from mojilex_cli.dataset import DatasetValidationError
from mojilex_cli.github import RepositoryRef
from mojilex_cli.media import TemporaryMediaRun
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import PublicationCheckpoint, RunStore, new_checkpoint
from test_ai_recovery_checkpoint import _prepare_exact_request, _RecoveryProvider
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import _collection, _item, _processed
from test_pipeline_workspaces import _git


@pytest.fixture
def saved_add(tmp_path, monkeypatch):
    # Pytest's nested Windows paths can exceed Git's legacy 260-character limit.
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.longpaths")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "true")
    snapshot = write_fixture(tmp_path / "source")
    root = snapshot.root
    (root / ".gitignore").write_text(".mojilex/\n", encoding="utf-8")
    # Exercise the persistent POSIX lock artifact on Windows too.
    (root / ".mojilex" / "locks" / "dataset-transaction-v1.lock").touch()
    shutil.copytree(Path(runner.__file__).parents[1] / "schemas" / "v1", root / "schemas" / "v1")
    _git(root, "init", "--initial-branch=main")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "base",
    )
    _git(root, "remote", "add", "origin", "https://github.com/MojiLex/mojilex.git")
    config = MojiLexConfig(
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        ai=AIConfig(model="primary-model", max_ai_requests=100),
        processing=ProcessingConfig(static_batch_size=1),
    )
    source = _collection((_item("stage-cached", unique_id="unique", file_id="file"),))
    options = runner.PipelineOptions(
        repository="MojiLex/mojilex",
        model="primary-model",
        publish="pr",
        base="main",
        max_ai_requests=100,
        max_cost_usd=Decimal("5"),
        ai_concurrency=4,
        download_concurrency=8,
    )
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters=runner._safe_parameters((source.canonical_url,), options),
        cli_version="0.1.0",
        schema_version=runner.SCHEMA_VERSION,
        target_repository="MojiLex/mojilex",
        base_revision=_git(root, "rev-parse", "HEAD"),
    ).model_copy(
        update={"status": "failed", "ai_requests_used": 63, "ai_cost_reserved_usd": Decimal("0.63")}
    )
    store = RunStore(config.runs_dir)
    store.save(checkpoint)
    state = SimpleNamespace(
        snapshot=snapshot,
        root=root,
        config=config,
        checkpoint=checkpoint,
        store=store,
        source=source,
        target=RepositoryRef.parse("MojiLex/mojilex"),
        workspace_calls=[],
    )

    @contextmanager
    def workspace(target, branch, **kwargs):
        state.workspace_calls.append((target, branch))
        yield SimpleNamespace(root=root, target=state.target)

    monkeypatch.setattr(runner, "load_config", lambda: config)
    monkeypatch.setattr(runner, "repository_workspace", workspace)
    return state


async def test_add_to_staging_preserves_budget_and_reuses_exact_cached_ai(
    saved_add, monkeypatch, tmp_path
):
    state = saved_add
    processed = {item.native_id: _processed(state.snapshot) for item in state.source.items}
    provider = _RecoveryProvider([], None)

    async def fixed_provider(*args, **kwargs):
        return provider

    monkeypatch.setattr(runner, "_prepare_ai_request", _prepare_exact_request)
    monkeypatch.setattr(runner, "_provider_for_model", fixed_provider)
    with (
        CacheStore(
            state.config.cache_dir / "cache-v1.sqlite3", repository_root=state.root
        ) as cache,
        TemporaryMediaRun(root=tmp_path) as temporary,
    ):
        traces = {}
        _, generations = await runner._descriptions_for_collection(
            state.snapshot,
            state.source,
            processed,
            config=state.config,
            cache=cache,
            budget=RequestBudget(max_requests=1),
            ai_state=runner._AIState(),
            api_key=None,
            redescribe="changed",
            overwrite_reviewed=False,
            temporary=temporary,
            cache_alias_scope=state.checkpoint.run_id,
            request_traces_out=traces,
        )
        item = state.source.items[0]
        checkpoint = runner._checkpoint_media_item(
            state.checkpoint, item, processed[item.native_id]
        )
        checkpoint = runner._checkpoint_ai_keys(
            checkpoint,
            state.source,
            processed,
            state.config,
            generations,
            taxonomy_version=str(state.snapshot.manifest["taxonomy_version"]),
            request_traces=traces,
        )
        state.store.save(checkpoint)
        previous_calls = list(provider.calls)

        async def run_staged(sources, options, **kwargs):
            resumed = kwargs["resume_checkpoint"]
            # Transition intent is durable before processing can be interrupted.
            assert state.store.load(checkpoint.run_id) == resumed
            assert resumed.command == "describe"
            assert resumed.run_id == checkpoint.run_id
            assert resumed.target_repository == checkpoint.target_repository
            assert resumed.base_revision == checkpoint.base_revision
            assert resumed.elements == checkpoint.elements
            assert resumed.ai_requests_used == 63
            assert resumed.ai_cost_reserved_usd == Decimal("0.63")
            assert resumed.safe_parameters["max_ai_requests"] == options.max_ai_requests == 100
            assert options.max_cost_usd == Decimal("5")
            assert options.ai_concurrency == 4 and options.download_concurrency == 8
            assert options.publish == resumed.safe_parameters["publish"] == "local"
            assert options.direct_push is resumed.safe_parameters["direct_push"] is False
            staging = Path(options.repository)
            assert options.repository == resumed.safe_parameters["staging_repository"]
            assert options.repository == resumed.safe_parameters["repository"]
            assert staging != state.root
            assert _git(staging, "rev-parse", "HEAD") == checkpoint.base_revision
            assert _git(staging, "remote", "get-url", "origin") == (
                "https://github.com/MojiLex/mojilex.git"
            )
            assert kwargs["stage_only"] is True and kwargs["resume_id"] == checkpoint.run_id
            budget = RequestBudget(
                max_requests=100, requests_used=63, cost_reserved=Decimal("0.63")
            )
            descriptions, _ = await runner._descriptions_for_collection(
                state.snapshot,
                state.source,
                processed,
                config=state.config,
                cache=cache,
                budget=budget,
                ai_state=runner._AIState(),
                api_key=None,
                redescribe="changed",
                overwrite_reviewed=False,
                temporary=temporary,
                cache_alias_scope=kwargs["resume_id"],
                resume_request_traces=traces,
            )
            assert set(descriptions) == {item.native_id}
            assert budget.requests_used == 63 and budget.cost_reserved == Decimal("0.63")
            return CommandResult(run_id=resumed.run_id, publication={"mode": "staging"})

        monkeypatch.setattr(runner, "_run_add", run_staged)
        result = await runner._run_describe(checkpoint.run_id)
        assert result.publication == {"mode": "staging"}
        assert provider.calls == previous_calls
    assert state.workspace_calls == [("MojiLex/mojilex", "main")]
    assert _git(state.root, "status", "--porcelain") == ""


@pytest.mark.parametrize("phase", ["prepared", "candidate_pushed", "completed"])
async def test_add_with_publication_checkpoint_cannot_transition_to_staging(
    saved_add, monkeypatch, phase
):
    state = saved_add
    checkpoint = state.checkpoint.model_copy(
        update={
            "publication": PublicationCheckpoint(
                mode="pr",
                remote="origin",
                base_branch="main",
                expected_old_base=state.checkpoint.base_revision,
                candidate_sha="b" * 40,
                candidate_branch="mojilex/add/test",
                phase=phase,
            )
        }
    )
    state.store.save(checkpoint)
    with pytest.raises(CommandError, match="publication checkpoint"):
        await runner._run_describe(checkpoint.run_id)
    assert state.store.load(checkpoint.run_id) == checkpoint
    assert not state.workspace_calls
    assert not (state.config.runs_dir / "workspaces").exists()


@pytest.mark.parametrize("mismatch", ["base", "target", "invalid_dataset"])
async def test_add_staging_rejects_foreign_base_target_or_invalid_dataset(saved_add, mismatch):
    state = saved_add
    if mismatch == "base":
        state.checkpoint = state.checkpoint.model_copy(update={"base_revision": "c" * 40})
        state.store.save(state.checkpoint)
    elif mismatch == "target":
        state.target = RepositoryRef.parse("OtherOwner/other")
    else:
        (state.root / "dataset.json").write_text("{}\n", encoding="utf-8")
    expected_error = DatasetValidationError if mismatch == "invalid_dataset" else CommandError
    with pytest.raises(expected_error):
        await runner._run_describe(state.checkpoint.run_id)
    assert state.store.load(state.checkpoint.run_id) == state.checkpoint
    assert not (state.config.runs_dir / "workspaces").exists()


async def test_interruption_after_add_transition_resumes_only_local_staging(saved_add, monkeypatch):
    state = saved_add

    async def interrupted(*args, **kwargs):
        raise RuntimeError("synthetic interruption after staging transition")

    monkeypatch.setattr(runner, "_run_add", interrupted)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        await runner._run_describe(state.checkpoint.run_id)
    transitioned = state.store.load(state.checkpoint.run_id)
    assert transitioned.command == "describe"

    async def resumed(sources, options, **kwargs):
        assert kwargs["stage_only"] is True
        assert options.publish == "local" and options.direct_push is False
        assert options.repository == transitioned.safe_parameters["staging_repository"]
        assert kwargs["resume_checkpoint"].ai_requests_used == 63
        return CommandResult()

    monkeypatch.setattr(runner, "_run_add", resumed)
    await runner.run_resume(state.checkpoint.run_id)
    assert state.workspace_calls == [("MojiLex/mojilex", "main")]


@pytest.mark.parametrize("command", ["import", "describe"])
async def test_existing_staged_run_keeps_its_original_path_and_checkpoint(
    saved_add, monkeypatch, command
):
    state = saved_add
    checkpoint = state.checkpoint.model_copy(
        update={
            "command": command,
            "safe_parameters": {
                **state.checkpoint.safe_parameters,
                "staging_repository": str(state.root),
            },
        }
    )
    state.store.save(checkpoint)

    def no_new_staging(*args, **kwargs):
        pytest.fail("existing staged runs must not create another workspace")

    async def describe(sources, options, **kwargs):
        assert options.repository == str(state.root)
        assert options.publish == "local" and not options.direct_push
        assert kwargs["resume_checkpoint"] == checkpoint
        assert kwargs["stage_only"] is True
        return CommandResult()

    monkeypatch.setattr(runner, "prepare_staging_workspace", no_new_staging)
    monkeypatch.setattr(runner, "_run_add", describe)
    await runner._run_describe(checkpoint.run_id)
    assert state.store.load(checkpoint.run_id) == checkpoint
    assert not state.workspace_calls
