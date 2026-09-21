import pytest

from mojilex_cli.pipeline.batch import _add_missing_packs
from test_dataset_helpers import make_snapshot


@pytest.mark.parametrize(
    "mode",
    [
        "publish",
        "local",
        "existing",
        "already_there",
        "decline",
        "unavailable",
        "new_base",
        "auth_failed",
    ],
)
def test_sync_publishes_many_saved_runs_once(tmp_path, monkeypatch, mode):
    from contextlib import contextmanager, nullcontext
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
    if mode == "new_base":
        checkpoints[1].base_revision = "b" * 40
    checkpoints.extend(
        [
            SimpleNamespace(command="describe", status="failed", target_repository="owner/repo"),
            SimpleNamespace(command="describe", status="succeeded", target_repository="other/repo"),
            SimpleNamespace(command="import", status="succeeded", target_repository="owner/repo"),
        ]
    )
    monkeypatch.setattr(batch, "load_config", lambda: config)
    monkeypatch.setattr(batch, "load_credentials", lambda: SimpleNamespace(github_token=None))
    auth_calls = []

    def auth_status(self):
        from mojilex_cli.github import GitHubError

        auth_calls.append(True)
        assert not checked and not base_reads
        if mode == "auth_failed":
            raise GitHubError("Timeout trying to log in to github.com")

    monkeypatch.setattr(batch.GitHubCLI, "auth_status", auth_status)
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
    from mojilex_cli.dataset.validation import _SCHEMA_MEMO

    checked = []
    base_reads = []
    verified_bases = []
    progress = []

    def validate(root, **kwargs):
        assert _SCHEMA_MEMO.get() is not None
        assert kwargs == {"strict": True}
        checked.append(root)
        return snapshot, SimpleNamespace(raise_for_errors=lambda: None)

    def load_base(root):
        base_reads.append(root)
        return snapshot

    @contextmanager
    def stage(russian, english):
        progress.append(english)
        yield

    monkeypatch.setattr(batch, "load_validated_dataset", validate)
    monkeypatch.setattr(batch, "load_dataset", load_base)
    monkeypatch.setattr(batch, "_verify_saved_base", lambda *args: verified_bases.append(args))
    monkeypatch.setattr(batch, "_publication_progress", stage)

    def staging(run):
        if mode == "unavailable" and run is checkpoints[0]:
            raise batch.CommandError("DIRTY_WORKTREE", "Missing workspace", hint="Restore it.")
        return tmp_path

    monkeypatch.setattr(batch, "_staging_path_from_checkpoint", staging)
    active_bases = []

    @contextmanager
    def base_checkout(root, revision):
        assert not active_bases, "Do not retain checkouts for every historical revision"
        active_bases.append(revision)
        try:
            yield tmp_path
        finally:
            active_bases.remove(revision)

    monkeypatch.setattr(batch, "snapshot_at_revision", base_checkout)
    additions = iter(
        [([], ["first"]), ([], ["second"])]
        if mode == "already_there"
        else [(["first"], []), (["second"], [])]
    )
    monkeypatch.setattr(
        batch,
        "_add_missing_packs",
        lambda *a: (snapshot, *next(additions)),
    )
    monkeypatch.setattr(
        batch,
        "_changed_paths",
        lambda *a: () if mode == "existing" else ("data/file.json",),
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

    if mode == "auth_failed":
        from mojilex_cli.github import GitHubError

        with pytest.raises(GitHubError, match="Timeout"):
            batch.sync_packs_command(confirmation=confirm)
        assert not checked and not base_reads and not confirmations and not calls
        return
    if mode == "decline":
        with pytest.raises(batch.CommandError, match="not confirmed"):
            batch.sync_packs_command(confirmation=confirm)
        assert len(confirmations) == 1 and not calls
        return
    result = batch.sync_packs_command(local=mode == "local", confirmation=confirm)
    assert len(auth_calls) == (0 if mode == "local" else 1)
    assert len(checked) == (2 if mode == "unavailable" else 3)
    assert len(base_reads) == (2 if mode == "new_base" else 1)
    assert len(verified_bases) == (0 if mode in {"unavailable", "new_base"} else 1)
    assert not active_bases
    assert _SCHEMA_MEMO.get() is None
    assert "Validating the current GitHub dataset" in progress
    assert "Finding changes for the pull request" in progress
    if mode not in {"existing", "already_there"}:
        assert "Validating and saving the pull request candidate" in progress
    if mode in {"local", "existing", "already_there"}:
        assert not calls and not confirmations
        assert result.publication.get("preview") if mode == "local" else result.status == "noop"
        if mode == "already_there":
            assert result.result["already_on_github"] == 2
            assert result.result["changed_paths"] == []
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


def test_sync_counts_unchanged_pack_already_in_repository(tmp_path):
    current = make_snapshot(tmp_path.resolve())
    merged, added, skipped = _add_missing_packs(current, current.clone(), current.clone())
    assert merged.to_files() == current.to_files()
    assert added == []
    assert skipped == list(current.collections)


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


def test_sync_completes_existing_pack_preserving_remote_members_and_descriptions(tmp_path):
    from mojilex_cli.dataset import validate_snapshot
    from mojilex_cli.domain import emoji_id, membership_id, telegram_set_fingerprint

    current = make_snapshot(tmp_path.resolve())
    original_collection = next(iter(current.collections.values()))
    original_emoji = next(iter(current.emojis.values()))
    original_member = next(iter(current.memberships.values()))
    base = current.clone()
    candidate = current.clone()
    collection = candidate.collections[original_collection.id]
    collection.title = "staged title must not overwrite repository title"
    candidate.emojis[original_emoji.id].semantic_tags.append("staged")
    new_emoji = original_emoji.model_copy(deep=True)
    new_emoji.native_id = "5368324170671202287"
    new_emoji.id = emoji_id("telegram", "custom_emoji.id", "global", new_emoji.native_id)
    new_emoji.extensions["telegram"]["file_unique_id"] = "AnotherUniqueId"
    new_emoji.extensions["telegram"]["custom_emoji_id"] = new_emoji.native_id
    member = original_member.model_copy(deep=True)
    member.emoji_id = new_emoji.id
    member.id = membership_id(collection.id, new_emoji.id)
    # Candidate removed the original member; synchronization must preserve it,
    # resolving the position collision without rewriting repository records.
    candidate.memberships = {member.id: member}
    candidate.emojis[new_emoji.id] = new_emoji
    collection.extensions["telegram"]["set_fingerprint_sha256"] = telegram_set_fingerprint(
        [(new_emoji.native_id, "AnotherUniqueId")]
    )
    merged, added, skipped = _add_missing_packs(current, base, candidate)
    assert added == [collection.id] and not skipped
    assert merged.collections[collection.id].title == original_collection.title
    assert merged.collections[collection.id].item_count == 2
    assert merged.emojis[original_emoji.id] == original_emoji
    assert merged.memberships[original_member.id] == original_member
    assert merged.memberships[member.id].position == 1
    validate_snapshot(merged, schemas=False).raise_for_errors()
    again, added, skipped = _add_missing_packs(merged, base, candidate)
    assert not added and skipped == [collection.id]
    assert again.to_files() == merged.to_files()


def test_sync_detects_new_member_even_when_collection_metadata_is_unchanged(tmp_path):
    current = make_snapshot(tmp_path.resolve())
    candidate = current.clone()
    base = candidate.clone()
    base.memberships.clear()
    current.memberships.clear()
    current.emojis.clear()
    merged, added, skipped = _add_missing_packs(current, base, candidate)
    assert added == list(candidate.collections) and not skipped
    assert merged.memberships == candidate.memberships
    assert merged.emojis == candidate.emojis


@pytest.mark.parametrize("entity_type", ["collection", "emoji", "membership"])
def test_sync_does_not_restore_tombstoned_records(tmp_path, entity_type):
    from mojilex_cli.domain import Tombstone
    from test_dataset_helpers import NOW

    candidate = make_snapshot(tmp_path.resolve())
    base = candidate.clone()
    base.collections.clear()
    base.memberships.clear()
    current = base.clone()
    records = {
        "collection": candidate.collections,
        "emoji": candidate.emojis,
        "membership": candidate.memberships,
    }
    target_id = next(iter(records[entity_type]))
    current.emojis.pop(target_id, None)
    current.tombstones[target_id] = Tombstone(
        schema_version="1.0.0",
        entity_type="tombstone",
        target_entity_type=entity_type,
        target_id=target_id,
        reason_code="other",
        withheld_at=NOW,
        public_note="Removed.",
    )
    merged, added, skipped = _add_missing_packs(current, base, candidate)
    assert not added and skipped == list(candidate.collections)
    assert merged.to_files() == current.to_files()


@pytest.mark.parametrize("same_media", [True, False])
def test_sync_fills_missing_description_language_only_for_same_media(tmp_path, same_media):
    current = make_snapshot(tmp_path.resolve())
    original = next(iter(current.emojis.values()))
    del original.descriptions["en"]
    base = current.clone()
    candidate = make_snapshot(tmp_path.resolve())
    staged = next(iter(candidate.emojis.values()))
    staged.descriptions["ru"].text = "Staged description must not replace the published value."
    if not same_media:
        staged.media[0].sha256 = "f" * 64
    merged, added, skipped = _add_missing_packs(current, base, candidate)
    result = merged.emojis[original.id]
    assert result.descriptions["ru"] == original.descriptions["ru"]
    assert result.facets == original.facets
    if same_media:
        assert added == list(candidate.collections) and not skipped
        assert result.descriptions["en"] == staged.descriptions["en"]
    else:
        assert not added and skipped == list(candidate.collections)
        assert result.descriptions == original.descriptions


def test_later_pack_supplement_does_not_reuse_completed_pr_identity(tmp_path):
    from mojilex_cli.pipeline.batch import _sync_identifier

    snapshot = make_snapshot(tmp_path.resolve())
    packs = list(snapshot.collections)
    full = _sync_identifier("owner/repo", "main", packs, snapshot)
    assert full == _sync_identifier("owner/repo", "main", packs, snapshot.clone())
    next(iter(snapshot.emojis.values())).descriptions.pop("en")
    assert full != _sync_identifier("owner/repo", "main", packs, snapshot)
    partial = _sync_identifier("owner/repo", "main", packs, snapshot)
    snapshot.memberships.clear()
    assert partial != _sync_identifier("owner/repo", "main", packs, snapshot)


def test_sync_validation_failure_cannot_supply_a_candidate(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mojilex_cli.pipeline import batch

    def fail():
        raise ValueError("invalid saved dataset")

    monkeypatch.setattr(
        batch,
        "load_validated_dataset",
        lambda *args, **kwargs: (None, SimpleNamespace(raise_for_errors=fail)),
    )
    with pytest.raises(ValueError, match="invalid saved dataset"):
        batch._validated_snapshot(tmp_path)


def test_cached_base_does_not_authorize_non_git_saved_directory(tmp_path):
    from mojilex_cli.pipeline import batch

    with pytest.raises(batch.CommandError):
        batch._verify_saved_base(tmp_path, "a" * 40)


def test_cached_base_still_checks_commit_in_each_saved_repository(tmp_path, monkeypatch):
    from mojilex_cli.pipeline import batch

    monkeypatch.setattr(batch, "_local_git_directory", lambda root: root / ".git")

    class MissingBase:
        def __init__(self, root):
            pass

        def current_sha(self, revision):
            raise batch.GitError("missing commit")

    monkeypatch.setattr(batch, "GitRunner", MissingBase)
    with pytest.raises(batch.CommandError, match="saved dataset base"):
        batch._verify_saved_base(tmp_path, "a" * 40)
