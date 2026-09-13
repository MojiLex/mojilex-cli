from __future__ import annotations

from types import SimpleNamespace

import pytest

from mojilex_cli.domain import collection_id, emoji_id, membership_id
from mojilex_cli.pipeline.pack_publication import ready_source_names, scoped_candidate
from test_dataset_helpers import make_snapshot


def add_pack(snapshot, name, *, shared=False):
    collection = next(iter(snapshot.collections.values())).model_copy(deep=True)
    collection.id = collection_id("telegram", "sticker_set.name", "global", name)
    collection.native_id = name
    collection.canonical_url = f"https://t.me/addemoji/{name}"
    collection.extensions["telegram"]["short_name"] = name
    member = next(iter(snapshot.memberships.values())).model_copy(deep=True)
    emoji = next(iter(snapshot.emojis.values())).model_copy(deep=True)
    if not shared:
        emoji.native_id = str(int(emoji.native_id) + len(snapshot.emojis))
        emoji.id = emoji_id("telegram", "custom_emoji.id", "global", emoji.native_id)
        snapshot.emojis[emoji.id] = emoji
    member.collection_id = collection.id
    member.emoji_id = emoji.id
    member.id = membership_id(collection.id, emoji.id)
    snapshot.collections[collection.id] = collection
    snapshot.memberships[member.id] = member
    return collection.id, emoji.id


def test_selected_pack_does_not_publish_other_batch_members(tmp_path):
    base = make_snapshot(tmp_path.resolve())
    candidate = base.clone()
    selected, selected_emoji = add_pack(candidate, "FirstPack")
    unrelated, unrelated_emoji = add_pack(candidate, "SecondPack")
    before_base, before_candidate = base.to_files(), candidate.to_files()

    result = scoped_candidate(base, candidate, {"firstpack"})

    assert set(result.collections) == {*base.collections, selected}
    assert selected_emoji in result.emojis and unrelated_emoji not in result.emojis
    assert all(member.collection_id != unrelated for member in result.memberships.values())
    assert base.to_files() == before_base and candidate.to_files() == before_candidate


def test_selected_pack_preserves_existing_shared_description(tmp_path):
    base = make_snapshot(tmp_path.resolve())
    candidate = base.clone()
    selected, shared_emoji = add_pack(candidate, "SharedPack", shared=True)
    candidate.emojis[shared_emoji].semantic_tags.append("unrelated-change")
    result = scoped_candidate(base, candidate, {"sharedpack"})
    assert selected in result.collections
    assert result.emojis[shared_emoji] == base.emojis[shared_emoji]


def test_selected_pack_keeps_its_own_existing_description_change(tmp_path):
    base = make_snapshot(tmp_path.resolve())
    candidate = base.clone()
    emoji = next(iter(candidate.emojis.values()))
    emoji.semantic_tags.append("selected-change")
    add_pack(candidate, "UnrelatedPack")
    result = scoped_candidate(base, candidate, {"suspiciouscats"})
    assert result.emojis[emoji.id] == emoji
    assert result.collections == base.collections


def test_selected_pack_can_include_new_emoji_shared_with_other_new_pack(tmp_path):
    base = make_snapshot(tmp_path.resolve())
    candidate = base.clone()
    selected, new_id = add_pack(candidate, "FirstPack")
    other, _ = add_pack(candidate, "SecondPack")
    member = next(
        member for member in candidate.memberships.values() if member.collection_id == other
    )
    del candidate.memberships[member.id]
    member.emoji_id = new_id
    member.id = membership_id(other, new_id)
    candidate.memberships[member.id] = member
    result = scoped_candidate(base, candidate, {"firstpack"})
    assert (
        new_id in result.emojis
        and selected in result.collections
        and other not in result.collections
    )


def test_selected_relations_never_pull_in_unrelated_batch_emoji(tmp_path):
    from test_visual_relations import _relation

    base = make_snapshot(tmp_path.resolve())
    candidate = base.clone()
    _, selected = add_pack(candidate, "FirstPack")
    _, other = add_pack(candidate, "SecondPack")
    original = next(iter(base.emojis))

    def relation(left, right):
        return _relation(
            *sorted((candidate.emojis[left], candidate.emojis[right]), key=lambda e: e.id)
        )

    required = relation(selected, original)
    cross_pack = relation(selected, other)
    unrelated = relation(other, original)
    candidate.relations.update({item.id: item for item in (required, cross_pack, unrelated)})
    result = scoped_candidate(base, candidate, {"firstpack"})
    assert set(result.relations) == {required.id}
    assert other not in result.emojis


def test_ready_sources_include_completed_member_of_partial_batch():
    first, second = "https://t.me/addemoji/FirstPack", "https://t.me/addemoji/SecondPack"
    checkpoint = SimpleNamespace(
        command="describe",
        status="partial",
        elements={},
        safe_parameters={
            "sources": [first, second],
            "source_states": {
                first: {"phase": "describe", "status": "succeeded"},
                second: {"phase": "describe", "status": "failed"},
            },
        },
    )
    assert ready_source_names(checkpoint) == {"firstpack"}


@pytest.mark.parametrize("status,expected", [("succeeded", {"firstpack"}), ("partial", set())])
def test_legacy_readiness_requires_successful_complete_description(status, expected):
    checkpoint = SimpleNamespace(
        command="describe",
        status=status,
        elements={},
        safe_parameters={"sources": ["https://t.me/addemoji/FirstPack"]},
    )
    assert ready_source_names(checkpoint) == expected


def test_persisted_pack_publication_preserves_partial_parent_and_other_pack(tmp_path, monkeypatch):
    import os
    import shutil
    from contextlib import contextmanager
    from pathlib import Path

    from mojilex_cli import schemas
    from mojilex_cli.config import MojiLexConfig
    from mojilex_cli.dataset import load_dataset, validate_dataset
    from mojilex_cli.github import RepositoryRef
    from mojilex_cli.pipeline.pack_publication import prepare_pack_publication
    from mojilex_cli.pipeline.runner import _apply_with_rollback
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
    config = MojiLexConfig(repository={"target": str(root)}, runs_dir=tmp_path / "runs")
    identifier = "mlxrun_" + "a" * 32
    staging = prepare_staging_workspace(
        root,
        target=RepositoryRef.parse("MojiLex/mojilex"),
        runs_dir=config.runs_dir,
        run_id=identifier,
        base_branch="main",
        base_revision=revision,
    )
    initial = load_dataset(staging)
    candidate = initial.clone()
    selected, _ = add_pack(candidate, "FirstPack", shared=True)
    other, _ = add_pack(candidate, "SecondPack", shared=True)
    _apply_with_rollback(initial, candidate)
    first, second = "https://t.me/addemoji/FirstPack", "https://t.me/addemoji/SecondPack"
    checkpoint = new_checkpoint(
        command="describe",
        cli_version="0.2.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision=revision,
        run_id=identifier,
        safe_parameters={
            "sources": [first, second],
            "staging_repository": str(staging),
            "max_ai_requests": "unlimited",
            "base": "main",
            "source_states": {
                first: {"phase": "describe", "status": "succeeded"},
                second: {"phase": "describe", "status": "failed"},
            },
        },
    ).model_copy(update={"status": "partial", "ai_requests_used": 17})
    store = RunStore(config.runs_dir)
    parent_file = store.save(checkpoint)
    original_parent = parent_file.read_bytes()
    original_files = load_dataset(staging).to_files()

    derived = prepare_pack_publication(checkpoint, "FirstPack", config)
    derived_path = Path(derived.safe_parameters["staging_repository"])
    validate_dataset(derived_path, strict=True).raise_for_errors()
    result = load_dataset(derived_path)
    assert selected in result.collections and other not in result.collections
    assert result.emojis == initial.emojis
    assert derived.safe_parameters["sources"] == [first]
    assert derived.safe_parameters["publication_source_run"] == checkpoint.run_id
    assert derived.ai_requests_used == 17
    assert store.load(derived.run_id) == derived
    assert parent_file.read_bytes() == original_parent
    assert load_dataset(staging).to_files() == original_files
    from mojilex_cli.runs.store import PublicationCheckpoint

    receipt = PublicationCheckpoint(
        mode="pr",
        remote="origin",
        base_branch="main",
        expected_old_base=revision,
        candidate_sha="b" * 40,
        candidate_branch="mojilex/pack-scope-test",
        phase="completed",
    )
    store.save(derived.model_copy(update={"publication": receipt}))
    again = prepare_pack_publication(checkpoint, "FirstPack", config)
    assert again.run_id == derived.run_id
    assert again.publication == receipt
    assert again.safe_parameters["repository"] == str(derived_path)
    assert load_dataset(derived_path).to_files() == result.to_files()
    from mojilex_cli.commands.runtime import CommandError
    from mojilex_cli.runs.store import RunLockedError

    with pytest.raises(CommandError, match="completed description"):
        prepare_pack_publication(checkpoint, "SecondPack", config)
    with store.execution_lock(checkpoint.run_id), pytest.raises(RunLockedError):
        prepare_pack_publication(checkpoint, "FirstPack", config)

    # Batch synchronization also extracts the one ready pack from this partial
    # run. Its failed neighbor must not be added just because staging contains it.
    from mojilex_cli.pipeline import batch

    merged_root = tmp_path / "latest"

    @contextmanager
    def local_clone(*args, **kwargs):
        _git(root, "clone", str(root), str(merged_root))
        _git(merged_root, "remote", "set-url", "origin", "https://github.com/MojiLex/mojilex.git")
        yield SimpleNamespace(root=merged_root, target=RepositoryRef.parse("MojiLex/mojilex"))

    monkeypatch.setattr(batch, "load_config", lambda: config)
    monkeypatch.setattr(batch, "_runs", lambda _: ([checkpoint], 0))
    monkeypatch.setattr(batch, "repository_workspace", local_clone)
    synchronized = batch.sync_packs_command(local=True)
    assert synchronized.result["added_packs"] == [selected]
    assert other not in load_dataset(merged_root).collections
    assert parent_file.read_bytes() == original_parent
    assert load_dataset(staging).to_files() == original_files

    old_publication = (store.root / f"{derived.run_id}.json").read_bytes()
    changed = load_dataset(staging)
    updated = changed.clone()
    updated.collections[selected].title = "An explicitly updated first pack"
    _apply_with_rollback(changed, updated)
    new_publication = prepare_pack_publication(checkpoint, "FirstPack", config)
    assert new_publication.run_id != derived.run_id
    assert new_publication.publication is None
    assert (store.root / f"{derived.run_id}.json").read_bytes() == old_publication
    assert parent_file.read_bytes() == original_parent
