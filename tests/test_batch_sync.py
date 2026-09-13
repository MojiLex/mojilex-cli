import pytest

from mojilex_cli.pipeline.batch import _add_missing_packs
from test_dataset_helpers import make_snapshot


@pytest.mark.parametrize("mode", ["publish", "local", "existing", "decline", "unavailable"])
def test_sync_publishes_many_saved_runs_once(tmp_path, monkeypatch, mode):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from mojilex_cli.commands.runtime import CommandResult
    from mojilex_cli.pipeline import batch

    snapshot = make_snapshot(tmp_path.resolve())
    collection = next(iter(snapshot.collections.values()))
    snapshot.collections.update({"first": collection, "second": collection})
    config = SimpleNamespace(
        repository=SimpleNamespace(target="owner/repo", base_branch="main"),
        runs_dir=tmp_path / "runs",
    )
    checkpoints = [
        SimpleNamespace(
            command="describe",
            status="succeeded",
            target_repository="owner/repo",
            run_id="mlxrun_" + digit * 32,
            base_revision="a" * 40,
        )
        for digit in ("1", "2")
    ]
    checkpoints.extend(
        [
            SimpleNamespace(command="describe", status="failed", target_repository="owner/repo"),
            SimpleNamespace(command="describe", status="succeeded", target_repository="other/repo"),
            SimpleNamespace(command="import", status="succeeded", target_repository="owner/repo"),
        ]
    )
    monkeypatch.setattr(batch, "load_config", lambda: config)
    monkeypatch.setattr(batch, "_runs", lambda _: (checkpoints, 0))
    monkeypatch.setattr(
        batch.RunStore,
        "load",
        lambda _, run_id: next(
            item for item in checkpoints if getattr(item, "run_id", None) == run_id
        ),
    )
    monkeypatch.setattr(
        batch,
        "repository_workspace",
        lambda *a, **kw: nullcontext(SimpleNamespace(root=tmp_path, target="owner/repo")),
    )
    monkeypatch.setattr(
        batch, "validate_dataset", lambda *a, **kw: SimpleNamespace(raise_for_errors=lambda: None)
    )
    monkeypatch.setattr(batch, "load_dataset", lambda _: snapshot)

    def staging(run):
        if mode == "unavailable" and run is checkpoints[0]:
            raise batch.CommandError("DIRTY_WORKTREE", "Missing workspace", hint="Restore it.")
        return tmp_path

    monkeypatch.setattr(batch, "_staging_path_from_checkpoint", staging)
    monkeypatch.setattr(batch, "snapshot_at_revision", lambda *a: nullcontext(tmp_path))
    additions = iter((["first"], ["second"]))
    monkeypatch.setattr(batch, "_add_missing_packs", lambda *a: (snapshot, next(additions), []))
    monkeypatch.setattr(
        batch, "_changed_paths", lambda *a: () if mode == "existing" else ("data/file.json",)
    )
    monkeypatch.setattr(batch, "_apply_with_rollback", lambda *a: None)
    calls = []

    async def submit(*args, **kwargs):
        calls.append(kwargs)
        return CommandResult(publication={"mode": "pr"})

    monkeypatch.setattr(batch, "_run_submit", submit)
    confirmations = []

    def confirm(message):
        confirmations.append(message)
        return mode != "decline"

    if mode == "decline":
        with pytest.raises(batch.CommandError, match="not confirmed"):
            batch.sync_packs_command(confirmation=confirm)
        assert len(confirmations) == 1 and not calls
        return
    result = batch.sync_packs_command(local=mode == "local", confirmation=confirm)
    if mode in {"local", "existing"}:
        assert not calls and not confirmations
        assert result.publication.get("preview") if mode == "local" else result.status == "noop"
        return
    assert len(calls) == 1
    assert len(confirmations) == 1 and "owner/repo" in confirmations[0]
    assert calls[0]["publish"] == "pr" and not calls[0]["direct_push"]
    assert result.result["added_packs"] == (
        ["first"] if mode == "unavailable" else ["first", "second"]
    )
    if mode == "unavailable":
        assert result.warnings


def test_sync_skips_existing_pack_without_changing_descriptions(tmp_path):
    current = make_snapshot(tmp_path.resolve())
    base = current.clone()
    candidate = current.clone()
    collection_id = next(iter(candidate.collections))
    candidate.collections[collection_id].title = "new title"
    next(iter(candidate.emojis.values())).semantic_tags.append("staged")
    merged, added, skipped = _add_missing_packs(current, base, candidate)
    assert merged.to_files() == current.to_files()
    assert added == [] and skipped == [collection_id]


def test_sync_adds_missing_pack_preserving_shared_emoji(tmp_path):
    candidate = make_snapshot(tmp_path.resolve())
    base = candidate.clone()
    base.collections.clear()
    base.memberships.clear()
    latest = base.clone()
    next(iter(latest.emojis.values())).semantic_tags.append("repository-value")
    merged, added, skipped = _add_missing_packs(latest, base, candidate)
    assert added == list(candidate.collections) and skipped == []
    assert merged.memberships == candidate.memberships
    assert merged.emojis == latest.emojis
    again, additions, _ = _add_missing_packs(merged, base, candidate)
    assert not additions
    assert again.to_files() == merged.to_files()


def test_sync_only_adds_staged_collection_changes(tmp_path):
    candidate = make_snapshot(tmp_path.resolve())
    base = candidate.clone()
    latest = candidate.clone()
    latest.collections.clear()
    latest.memberships.clear()
    merged, added, _ = _add_missing_packs(latest, base, candidate)
    assert not added
    assert merged.to_files() == latest.to_files()


def test_sync_git(tmp_path, monkeypatch):
    """Only replace GitHub cloning; read, merge, write and validate real datasets."""
    import os
    import shutil
    from contextlib import contextmanager
    from pathlib import Path
    from types import SimpleNamespace

    from mojilex_cli import schemas
    from mojilex_cli.config import MojiLexConfig
    from mojilex_cli.dataset import load_dataset, validate_dataset
    from mojilex_cli.domain import collection_id, membership_id
    from mojilex_cli.github import RepositoryRef
    from mojilex_cli.pipeline import batch
    from mojilex_cli.pipeline.workspaces import prepare_staging_workspace
    from mojilex_cli.runs import RunStore, new_checkpoint
    from test_dataset_helpers import write_fixture
    from test_pipeline_workspaces import _git

    count = int(os.environ.get("GIT_CONFIG_COUNT", "0"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", str(count + 1))
    monkeypatch.setenv(f"GIT_CONFIG_KEY_{count}", "core.longpaths")
    monkeypatch.setenv(f"GIT_CONFIG_VALUE_{count}", "true")

    root = tmp_path / "upstream"
    write_fixture(root)
    shutil.copytree(Path(schemas.__file__).parent / "v1", root / "schemas" / "v1")
    (root / ".gitattributes").write_text("* text eol=lf\n", encoding="utf-8")
    (root / ".gitignore").write_text(".mojilex/\n", encoding="utf-8")
    # POSIX file locks persist after release; exercise that artifact on Windows too.
    (root / ".mojilex" / "locks" / "dataset-transaction-v1.lock").touch()
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
        "fixture",
    )
    _git(root, "remote", "add", "origin", "https://github.com/MojiLex/mojilex.git")
    revision = _git(root, "rev-parse", "HEAD")
    config = MojiLexConfig(
        repository={"target": str(root), "base_branch": "main"},
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
    )
    expected = set(load_dataset(root).collections)
    for digit, name in (("a", "FirstNewPack"), ("b", "SecondNewPack")):
        run_id = "mlxrun_" + digit * 32
        staging = prepare_staging_workspace(
            root,
            target=RepositoryRef.parse("MojiLex/mojilex"),
            runs_dir=config.runs_dir,
            run_id=run_id,
            base_branch="main",
            base_revision=revision,
        )
        original = load_dataset(staging)
        candidate = original.clone()
        collection = next(iter(original.collections.values())).model_copy(deep=True)
        collection.id = collection_id("telegram", "sticker_set.name", "global", name)
        collection.native_id = name
        collection.canonical_url = f"https://t.me/addemoji/{name}"
        collection.extensions["telegram"]["short_name"] = name
        member = next(iter(original.memberships.values())).model_copy(deep=True)
        member.collection_id = collection.id
        member.id = membership_id(collection.id, member.emoji_id)
        candidate.collections[collection.id] = collection
        candidate.memberships[member.id] = member
        batch._apply_with_rollback(original, candidate)
        expected.add(collection.id)
        checkpoint = new_checkpoint(
            command="describe",
            safe_parameters={
                "sources": [f"https://t.me/addemoji/{name}"],
                "staging_repository": str(staging),
            },
            cli_version="0.2.0",
            schema_version="1.0.0",
            target_repository="MojiLex/mojilex",
            base_revision=revision,
            run_id=run_id,
        ).model_copy(update={"status": "succeeded"})
        RunStore(config.runs_dir).save(checkpoint)

    merged_root = tmp_path / "latest"

    @contextmanager
    def local_clone(*args, **kwargs):
        assert kwargs["isolated"] is True
        _git(root, "clone", str(root), str(merged_root))
        _git(merged_root, "remote", "set-url", "origin", "https://github.com/MojiLex/mojilex.git")
        yield SimpleNamespace(root=merged_root, target=RepositoryRef.parse("MojiLex/mojilex"))

    monkeypatch.setattr(batch, "load_config", lambda: config)
    monkeypatch.setattr(batch, "repository_workspace", local_clone)
    result = batch.sync_packs_command(local=True)
    actual = load_dataset(merged_root)
    assert set(actual.collections) == expected
    assert len(actual.memberships) == 3 and len(actual.emojis) == 1
    assert actual.emojis == load_dataset(root).emojis
    assert len(result.result["added_packs"]) == 2
    assert result.publication == {"mode": "local", "preview": True}
    assert validate_dataset(merged_root, strict=True).valid
    assert not _git(root, "status", "--porcelain")
