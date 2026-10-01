import asyncio
import hashlib
import threading
from contextlib import contextmanager, suppress
from decimal import Decimal

import pytest

from mojilex_cli.ai import CostEstimate
from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.dataset import load_dataset, validate_dataset
from mojilex_cli.git import DirtyWorktreeError, GitRunner
from mojilex_cli.pipeline import runner
from mojilex_cli.pipeline.staging_guards import staging_guard_exemptions
from mojilex_cli.pipeline.workspaces import staging_workspace_path
from mojilex_cli.runs import RunStore
from staging_fixture_helpers import clone_fixture_repository
from test_add_run_staging import saved_add  # noqa: F401
from test_incremental_pack_readiness import _real_packs
from test_pack_describe_pipeline import pipeline  # noqa: F401


def _private_pipeline(request, monkeypatch):
    state = request.getfixturevalue("pipeline")
    _real_packs(state, monkeypatch, dedupe="exact")
    state.sources = state.sources[:1]
    root = staging_workspace_path(state.config.runs_dir, state.checkpoint.run_id)
    clone_fixture_repository(
        state.root,
        root,
        target=state.target,
        base_revision=state.checkpoint.base_revision,
    )
    state.root = root
    state.config = state.config.model_copy(
        update={"repository": state.config.repository.model_copy(update={"target": str(root)})}
    )

    @contextmanager
    def workspace(*args, **kwargs):
        yield runner.RepositoryWorkspace(root=root, target=state.target, temporary=False)

    monkeypatch.setattr(runner, "repository_workspace", workspace)
    state.latest_checkpoint = lambda: RunStore(state.config.runs_dir).load(state.checkpoint.run_id)

    async def run(checkpoint):
        return await runner._run_add(
            tuple(source.canonical_url for source in state.sources),
            runner.PipelineOptions(),
            resume_id=checkpoint.run_id,
            resume_checkpoint=checkpoint,
            stage_only=True,
        )

    state.run_checkpoint = run
    return state


@pytest.mark.parametrize("phase", ["pack_apply", "pack_receipts", "pack_identity", "final_apply"])
async def test_cancelled_staging_save_preserves_receipts_latest_budget_and_resumes(
    request, monkeypatch, phase
):
    state = _private_pipeline(request, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    target_name = {
        "pack_apply": "_apply_with_rollback",
        "pack_receipts": "staging_output_receipts",
        "pack_identity": "_dedupe_scan_identity",
        "final_apply": "_apply_with_rollback",
    }[phase]
    target_call = 2 if phase == "final_apply" else 1
    original = getattr(runner, target_name)

    def block_completed_operation(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == target_call:
            entered.set()
            assert release.wait(30)
        return result

    original_budget = runner.RequestBudget

    def capture_budget(**kwargs):
        state.budget = original_budget(**kwargs)
        return state.budget

    monkeypatch.setattr(runner, target_name, block_completed_operation)
    monkeypatch.setattr(runner, "RequestBudget", capture_budget)
    task = asyncio.create_task(state.run_checkpoint(state.checkpoint))
    try:
        assert await asyncio.to_thread(entered.wait, 30)
        await state.budget.reserve(
            CostEstimate(upper_bound_usd=Decimal("0.01"), note="synthetic peer reservation")
        )
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 30)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await task

    saved = state.latest_checkpoint()
    assert saved.ai_requests_used == state.checkpoint.ai_requests_used + 1
    assert saved.ai_cost_reserved_usd == state.checkpoint.ai_cost_reserved_usd + Decimal("0.01")
    receipts = saved.safe_parameters["staging_owned_paths"]
    assert receipts
    assert all(
        hashlib.sha256((state.root / relative).read_bytes()).hexdigest() == digest
        for relative, digest in receipts.items()
    )
    snapshot = load_dataset(state.root)
    assert validate_dataset(state.root, strict=True).valid
    recovered = runner._resume_dedupe_selected_ids(saved, snapshot, set())
    new_ids = {
        emoji.id
        for emoji in snapshot.emojis.values()
        if emoji.native_id == state.sources[0].items[0].native_id
    }
    assert new_ids and new_ids <= recovered
    assert saved.elements[state.sources[0].items[0].native_id].candidate_scan_complete

    # A real changed public record requires ownership exemptions on resume.
    state.sources = tuple(
        source.model_copy(update={"title": source.title + " renamed"}) for source in state.sources
    )
    monkeypatch.setattr(runner, target_name, original)
    resumed = await asyncio.wait_for(state.run_checkpoint(saved), 30)
    assert not resumed.errors
    final = state.latest_checkpoint()
    assert final.ai_requests_used == saved.ai_requests_used
    assert final.ai_cost_reserved_usd == saved.ai_cost_reserved_usd
    assert state.sources[0].title in {
        collection.title for collection in load_dataset(state.root).collections.values()
    }


async def test_failed_staging_transaction_does_not_authorize_foreign_write(request, monkeypatch):
    state = _private_pipeline(request, monkeypatch)
    before = load_dataset(state.root)
    relative = "data/telegram/emojis/zz/foreign.json"

    def fail_apply(*args, **kwargs):
        foreign = state.root / relative
        foreign.parent.mkdir(parents=True, exist_ok=True)
        foreign.write_text("synthetic foreign write\n", encoding="utf-8")
        raise CommandError("DIRTY_WORKTREE", "synthetic transaction failure", hint="inspect")

    monkeypatch.setattr(runner, "_apply_with_rollback", fail_apply)
    result = await asyncio.wait_for(state.run_checkpoint(state.checkpoint), 30)
    assert any(error.code == "DIRTY_WORKTREE" for error in result.errors)
    saved = state.latest_checkpoint()
    assert relative not in saved.safe_parameters.get("staging_owned_paths", {})
    assert not staging_guard_exemptions(
        before, before, saved, runs_dir=state.config.runs_dir, changed_paths={relative}
    )
    with pytest.raises(DirtyWorktreeError):
        runner.GitPublisher(GitRunner(state.root)).guard_targets((relative,))
