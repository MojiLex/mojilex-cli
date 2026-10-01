from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import PurePosixPath

import pytest

from mojilex_cli.dataset import apply_snapshot, load_dataset
from mojilex_cli.dataset.layout import (
    collection_path,
    collection_shard,
    emoji_bucket_path,
    memberships_path,
    previous_emoji_bucket_path,
)
from mojilex_cli.git import DirtyWorktreeError, GitRunner
from mojilex_cli.pipeline import runner
from mojilex_cli.pipeline.runner import _changed_paths
from mojilex_cli.pipeline.staging_guards import (
    checkpoint_staging_outputs,
    staging_guard_exemptions,
    staging_output_receipts,
)
from mojilex_cli.pipeline.workspaces import staging_workspace_path
from mojilex_cli.runs import RunStore
from test_add_run_staging import saved_add  # noqa: F401
from test_incremental_pack_readiness import _real_packs
from test_pack_describe_pipeline import pipeline  # noqa: F401
from test_pipeline_workspaces import _git


@pytest.fixture
def private_stage(request):
    state = request.getfixturevalue("saved_add")
    root = staging_workspace_path(state.config.runs_dir, state.checkpoint.run_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(state.root, root)
    state.root = root
    state.snapshot = load_dataset(root)
    state.checkpoint = state.checkpoint.model_copy(
        update={
            "safe_parameters": {
                **state.checkpoint.safe_parameters,
                "staging_repository": str(root),
            }
        }
    )
    return state


def _legacy_paths(state):
    snapshot = state.snapshot
    emoji = next(iter(snapshot.emojis.values()))
    canonical = emoji_bucket_path(emoji.platform, emoji.id)
    legacy = previous_emoji_bucket_path(emoji.platform, emoji.id)
    root = snapshot.root
    root.joinpath(*legacy.parts).parent.mkdir(parents=True, exist_ok=True)
    root.joinpath(*canonical.parts).rename(root.joinpath(*legacy.parts))
    collection = next(iter(snapshot.collections.values()))
    legacy_dir = PurePosixPath(
        "data", collection.platform, "collections", collection_shard(collection.id), collection.id
    )
    for current in (
        collection_path(collection.platform, collection.id),
        memberships_path(collection.platform, collection.id),
        collection_path(collection.platform, collection.id).with_name("README.md"),
    ):
        old = legacy_dir / current.name
        root.joinpath(*old.parts).parent.mkdir(parents=True, exist_ok=True)
        root.joinpath(*current.parts).rename(root.joinpath(*old.parts))
    _git(root, "add", "-A")
    _git(
        root,
        "rm",
        "--cached",
        "--",
        str(legacy),
        str(legacy_dir / "collection.json"),
        str(legacy_dir / "memberships.jsonl"),
    )
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "legacy imported baseline",
    )
    state.checkpoint = state.checkpoint.model_copy(
        update={"base_revision": _git(root, "rev-parse", "HEAD")}
    )
    state.snapshot = load_dataset(root)
    return legacy, canonical, legacy_dir


def _exemptions(state, before, after, paths):
    return staging_guard_exemptions(
        before,
        after,
        state.checkpoint,
        runs_dir=state.config.runs_dir,
        changed_paths=map(str, paths),
    )


def test_legacy_layout_move_exempts_only_byte_identical_sources(private_stage):
    state = private_stage
    old, new, legacy_dir = _legacy_paths(state)
    before = state.snapshot
    after = before.clone()
    paths = _changed_paths(before, after)
    exempt = _exemptions(state, before, after, paths)
    assert exempt == {
        str(old),
        str(legacy_dir / "collection.json"),
        str(legacy_dir / "memberships.jsonl"),
    }
    GitRunner(state.root).ensure_no_overlapping_changes(
        [str(path) for path in paths if str(path) not in exempt]
    )
    apply_snapshot(before, after)
    assert state.root.joinpath(*new.parts).read_bytes() == before.source_bytes[old]
    assert not state.root.joinpath(*old.parts).exists()


@pytest.mark.parametrize("changed_kind", ["emoji", "collection", "membership"])
def test_legacy_unreceipted_semantic_changes_remain_blocked(private_stage, changed_kind):
    state = private_stage
    old, _, legacy_dir = _legacy_paths(state)
    before = state.snapshot
    after = before.clone()
    if changed_kind == "emoji":
        next(iter(after.emojis.values())).descriptions["en"].text += " Additional meaning."
        target = old
    elif changed_kind == "collection":
        next(iter(after.collections.values())).title += " changed"
        target = legacy_dir / "collection.json"
    else:
        next(iter(after.memberships.values())).position += 1
        target = legacy_dir / "memberships.jsonl"
    paths = _changed_paths(before, after)
    exempt = _exemptions(state, before, after, paths)
    assert str(target) not in exempt
    with pytest.raises(DirtyWorktreeError):
        GitRunner(state.root).ensure_no_overlapping_changes(
            [str(path) for path in paths if str(path) not in exempt]
        )


@pytest.mark.parametrize("change", ["source", "destination"])
def test_legacy_move_rejects_changed_source_or_occupied_destination(private_stage, change):
    state = private_stage
    old, new, _ = _legacy_paths(state)
    before = state.snapshot
    after = before.clone()
    target = old if change == "source" else new
    state.root.joinpath(*target.parts).write_bytes(b"foreign file\n")
    assert str(old) not in _exemptions(state, before, after, _changed_paths(before, after))


@pytest.mark.parametrize("link_target", ["source", "destination"])
def test_legacy_move_never_follows_a_new_link(private_stage, link_target, tmp_path):
    state = private_stage
    old, new, _ = _legacy_paths(state)
    before = state.snapshot
    after = before.clone()
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(before.source_bytes[old])
    target = state.root.joinpath(*(old if link_target == "source" else new).parts)
    if link_target == "source":
        target.unlink()
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("This platform does not allow the synthetic symlink fixture")
    assert str(old) not in _exemptions(state, before, after, _changed_paths(before, after))
    assert outside.read_bytes() == before.source_bytes[old]


def test_owned_receipts_survive_reload_but_reject_foreign_edits(private_stage):
    state = private_stage
    before = state.snapshot
    after = before.clone()
    collection = next(iter(after.collections.values()))
    collection.title += " saved by this run"
    paths = _changed_paths(before, after)
    apply_snapshot(before, after)
    delta = staging_output_receipts(
        state.checkpoint,
        after,
        map(str, paths),
        runs_dir=state.config.runs_dir,
    )
    state.checkpoint = checkpoint_staging_outputs(state.checkpoint, delta)
    store = RunStore(state.config.runs_dir)
    store.save(state.checkpoint)
    state.checkpoint = store.load(state.checkpoint.run_id)
    resumed = load_dataset(state.root)
    candidate = resumed.clone()
    candidate.collections[collection.id].title += " next save"
    path = collection_path(collection.platform, collection.id)
    changed = _changed_paths(resumed, candidate)
    assert str(path) in _exemptions(state, resumed, candidate, changed)
    # A change after the baseline read is detected too, before AtomicWriter checks it.
    state.root.joinpath(*path.parts).write_bytes(b"foreign file\n")
    assert str(path) not in _exemptions(state, resumed, candidate, changed)
    # Even a parsed foreign edit after a restart cannot reuse the earlier receipt.
    state.root.joinpath(*path.parts).write_bytes(resumed.source_bytes[path])
    foreign = load_dataset(state.root)
    foreign.collections[collection.id].title += " unrelated edit"
    apply_snapshot(resumed, foreign)
    reloaded = load_dataset(state.root)
    assert str(path) not in _exemptions(state, reloaded, candidate, changed)


@pytest.mark.parametrize("invalid_scope", ["other_root", "other_base"])
def test_normal_checkout_or_wrong_base_never_gets_exemptions(private_stage, invalid_scope):
    state = private_stage
    _legacy_paths(state)
    before = state.snapshot
    after = before.clone()
    if invalid_scope == "other_root":
        state.config = state.config.model_copy(update={"runs_dir": state.config.runs_dir / "other"})
    else:
        state.checkpoint = state.checkpoint.model_copy(update={"base_revision": "f" * 40})
    assert _exemptions(state, before, after, _changed_paths(before, after)) == set()
    assert (
        staging_output_receipts(
            state.checkpoint,
            after,
            map(str, _changed_paths(before, after)),
            runs_dir=state.config.runs_dir,
        )
        == {}
    )


def test_receipt_delta_preserves_the_latest_peer_budget_and_results(private_stage):
    state = private_stage
    snapshot = state.snapshot
    path = next(iter(snapshot.source_bytes))
    delta = staging_output_receipts(
        state.checkpoint, snapshot, [str(path)], runs_dir=state.config.runs_dir
    )
    latest = state.checkpoint.model_copy(
        update={
            "ai_requests_used": state.checkpoint.ai_requests_used + 7,
            "safe_parameters": {**state.checkpoint.safe_parameters, "peer_result": "preserved"},
        }
    )
    merged = checkpoint_staging_outputs(latest, delta)
    assert merged.ai_requests_used == latest.ai_requests_used
    assert merged.safe_parameters["peer_result"] == "preserved"
    assert str(path) in merged.safe_parameters["staging_owned_paths"]


def test_receipt_merge_rejects_unsafe_or_non_dataset_paths(private_stage):
    checkpoint = private_stage.checkpoint.model_copy(
        update={"safe_parameters": {"staging_owned_paths": {"../outside": "1" * 64}}}
    )
    merged = checkpoint_staging_outputs(
        checkpoint,
        {
            "../outside": "1" * 64,
            "/outside": "1" * 64,
            "docs/README.md": "1" * 64,
            "data/../outside": "1" * 64,
            "data//telegram/file": "1" * 64,
            "data/telegram/file": "1" * 64,
        },
    )
    assert merged.safe_parameters["staging_owned_paths"] == {"data/telegram/file": "1" * 64}


async def test_saved_private_pack_can_resume_with_receipts_and_preserved_budget(
    request, monkeypatch
):
    state = request.getfixturevalue("pipeline")
    _real_packs(state, monkeypatch)
    root = staging_workspace_path(state.config.runs_dir, state.checkpoint.run_id)
    root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(state.root, root)
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

    first = await run(state.checkpoint)
    assert not first.errors
    saved = state.latest_checkpoint()
    assert saved.safe_parameters["staging_owned_paths"]
    assert saved.ai_requests_used == state.checkpoint.ai_requests_used
    state.sources = tuple(
        source.model_copy(update={"title": source.title + " renamed"}) for source in state.sources
    )
    resumed = await run(saved)
    assert not resumed.errors
    final = load_dataset(root)
    assert {source.title for source in state.sources} <= {
        collection.title for collection in final.collections.values()
    }
    assert state.latest_checkpoint().ai_requests_used == saved.ai_requests_used
