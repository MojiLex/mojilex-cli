from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from contextlib import closing
from pathlib import PurePosixPath

import pytest

from mojilex_cli.cache import lookup as module
from mojilex_cli.dataset.layout import collection_path, emoji_bucket_path, legacy_bucket_path
from test_dataset_helpers import write_fixture


def _git(root, *arguments):
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


def _commit(root):
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "Synthetic lookup fixture",
    )


@pytest.fixture
def indexed_repo(tmp_path):
    root = (tmp_path / "repo").resolve()
    snapshot = write_fixture(root)
    (root / ".gitignore").write_bytes(b".mojilex/\n.idea/\ndist/\n")
    _git(root, "init", "-b", "main")
    _git(root, "config", "core.autocrlf", "false")
    _commit(root)
    return snapshot, tmp_path.resolve() / "cache" / "lookup-v1.sqlite3"


def _selector(snapshot):
    return next(iter(snapshot.emojis))


def _lookup(snapshot, index, *, read_only=False):
    return module.lookup_authoring_snapshot(
        snapshot.root, index, [_selector(snapshot)], read_only=read_only
    )


def test_cold_build_and_warm_lookup_return_only_selector_view_without_full_scan(
    indexed_repo, monkeypatch
):
    snapshot, index = indexed_repo
    original = {path: (snapshot.root / path).read_bytes() for path in snapshot.to_files()}
    first = _lookup(snapshot, index)
    assert first is not None and index.is_file()
    assert set(first.emojis) == set(snapshot.emojis)
    assert set(first.collections) == set(snapshot.collections)
    assert set(first.memberships) == set(snapshot.memberships)
    assert first.source_bytes == {}  # Not an editable canonical snapshot.
    monkeypatch.setattr(module, "load_dataset", lambda *_: pytest.fail("warm hit must not scan"))
    second = _lookup(snapshot, index)
    assert second is not None and second.to_files() == first.to_files()
    assert original == {path: (snapshot.root / path).read_bytes() for path in original}
    assert _git(snapshot.root, "status", "--porcelain") == ""
    with closing(sqlite3.connect(index)) as connection:
        payload = connection.execute("SELECT payload FROM entries").fetchall()
    persisted = str(payload)
    assert "https://" not in persisted and "description" not in persisted
    assert "file_id" not in persisted and "media" not in persisted


def test_read_only_cold_miss_creates_nothing_and_never_loads(indexed_repo, monkeypatch):
    snapshot, index = indexed_repo
    monkeypatch.setattr(module, "load_dataset", lambda *_: pytest.fail("read-only cold miss"))
    assert _lookup(snapshot, index, read_only=True) is None
    assert not index.parent.exists()


def test_legacy_bucket_cold_and_warm_lookup_preserve_saved_repository(indexed_repo, monkeypatch):
    snapshot, index = indexed_repo
    emoji = snapshot.emojis[_selector(snapshot)]
    current = emoji_bucket_path(emoji.platform, emoji.id)
    legacy = legacy_bucket_path(current)
    assert legacy != current
    (snapshot.root / current).rename(snapshot.root / legacy)
    _commit(snapshot.root)
    before = (snapshot.root / legacy).read_bytes()

    first = _lookup(snapshot, index)
    assert first is not None
    assert first.emojis[emoji.id] == emoji
    with closing(sqlite3.connect(index)) as connection:
        payload = connection.execute(
            "SELECT payload FROM entries WHERE id=?", (emoji.id,)
        ).fetchone()[0]
    assert json.loads(payload)["path"] == str(legacy)
    monkeypatch.setattr(module, "load_dataset", lambda *_: pytest.fail("warm hit must not scan"))
    second = _lookup(snapshot, index, read_only=True)
    assert second is not None and second.emojis == first.emojis
    assert (snapshot.root / legacy).read_bytes() == before
    assert not (snapshot.root / current).exists()
    assert _git(snapshot.root, "status", "--porcelain") == ""


@pytest.mark.parametrize("legacy", [False, True])
def test_cached_bucket_with_wrong_hash_prefix_is_rejected(indexed_repo, monkeypatch, legacy):
    snapshot, index = indexed_repo
    emoji = snapshot.emojis[_selector(snapshot)]
    path = emoji_bucket_path(emoji.platform, emoji.id)
    if legacy:
        old = legacy_bucket_path(path)
        (snapshot.root / path).rename(snapshot.root / old)
        path = old
        _commit(snapshot.root)
    assert _lookup(snapshot, index) is not None
    with closing(sqlite3.connect(index)) as connection, connection:
        payload = connection.execute(
            "SELECT payload FROM entries WHERE id=?", (emoji.id,)
        ).fetchone()[0]
        value = json.loads(payload)
        wrong_digit = "0" if path.stem[0] != "0" else "1"
        value["path"] = str(path.with_name(wrong_digit + path.name[1:]))
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        connection.execute(
            "UPDATE entries SET payload=?,sha256=? WHERE id=?",
            (payload, hashlib.sha256(payload.encode()).hexdigest(), emoji.id),
        )
    monkeypatch.setattr(module, "load_dataset", lambda *_: pytest.fail("invalid cache must miss"))
    assert _lookup(snapshot, index, read_only=True) is None


def test_read_only_warm_hit_has_no_sidecars_or_cache_writes(indexed_repo, monkeypatch):
    snapshot, index = indexed_repo
    assert _lookup(snapshot, index) is not None
    before = index.read_bytes()
    names = sorted(path.name for path in index.parent.iterdir())
    monkeypatch.setattr(module, "load_dataset", lambda *_: pytest.fail("read-only cannot rebuild"))
    assert _lookup(snapshot, index, read_only=True) is not None
    assert index.read_bytes() == before
    assert sorted(path.name for path in index.parent.iterdir()) == names


@pytest.mark.parametrize("dirty", ["tracked", "untracked", "staged"])
def test_dirty_worktree_does_not_use_or_rebuild_index(indexed_repo, monkeypatch, dirty):
    snapshot, index = indexed_repo
    assert _lookup(snapshot, index) is not None
    before = index.read_bytes()
    if dirty in {"tracked", "staged"}:
        path = snapshot.root / "dataset.json"
        path.write_bytes(path.read_bytes() + b"\n")
        if dirty == "staged":
            _git(snapshot.root, "add", "dataset.json")
    else:
        (snapshot.root / f"{dirty}.txt").write_bytes(b"not in HEAD")
    monkeypatch.setattr(
        module, "load_dataset", lambda *_: pytest.fail("dirty fallback must be fast")
    )
    assert _lookup(snapshot, index) is None
    assert index.read_bytes() == before


def test_changed_head_read_only_misses_and_writable_rebuilds(indexed_repo):
    snapshot, index = indexed_repo
    assert _lookup(snapshot, index) is not None
    before = index.read_bytes()
    collection = next(iter(snapshot.collections.values()))
    collection.title = "Changed fixture title"
    path = collection_path(collection.platform, collection.id)
    generated = snapshot.to_files()
    (snapshot.root / path).write_bytes(generated[path])
    catalog = snapshot.root / "data" / "telegram" / "collections" / "README.md"
    catalog.write_bytes(generated[PurePosixPath("data/telegram/collections/README.md")])
    _commit(snapshot.root)
    assert _lookup(snapshot, index, read_only=True) is None
    assert index.read_bytes() == before
    updated = _lookup(snapshot, index)
    assert updated is not None
    assert updated.collections[collection.id].title == "Changed fixture title"
    assert index.read_bytes() != before


def test_same_head_in_different_repository_is_not_a_cache_hit(indexed_repo, tmp_path):
    snapshot, index = indexed_repo
    assert _lookup(snapshot, index) is not None
    destination = tmp_path / "clone"
    subprocess.run(
        ["git", "clone", "--no-hardlinks", str(snapshot.root), str(destination)],
        check=True,
        capture_output=True,
    )
    before = index.read_bytes()
    assert (
        module.lookup_authoring_snapshot(destination, index, [_selector(snapshot)], read_only=True)
        is None
    )
    assert index.read_bytes() == before


@pytest.mark.parametrize("corruption", ["checksum", "path", "missing-row", "database"])
def test_corrupt_index_fails_closed_without_automatic_replacement(
    indexed_repo, monkeypatch, corruption
):
    snapshot, index = indexed_repo
    assert _lookup(snapshot, index) is not None
    if corruption == "database":
        index.write_bytes(b"not a SQLite database")
    else:
        with closing(sqlite3.connect(index)) as connection, connection:
            if corruption == "missing-row":
                connection.execute("DELETE FROM entries WHERE id=?", (_selector(snapshot),))
            else:
                payload = connection.execute(
                    "SELECT payload FROM entries WHERE id=?", (_selector(snapshot),)
                ).fetchone()[0]
                value = json.loads(payload)
                value["path"] = "../../outside.json"
                payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
                digest = (
                    hashlib.sha256(payload.encode()).hexdigest() if corruption == "path" else "x"
                )
                connection.execute(
                    "UPDATE entries SET payload=?,sha256=? WHERE id=?",
                    (payload, digest, _selector(snapshot)),
                )
    before = index.read_bytes()
    monkeypatch.setattr(module, "load_dataset", lambda *_: pytest.fail("corrupt cache must miss"))
    assert _lookup(snapshot, index) is None
    assert index.read_bytes() == before


def test_hidden_modified_content_is_not_trusted_even_with_clean_git_status(indexed_repo):
    snapshot, index = indexed_repo
    assert _lookup(snapshot, index) is not None
    collection = next(iter(snapshot.collections.values()))
    path = collection_path(collection.platform, collection.id)
    _git(snapshot.root, "update-index", "--assume-unchanged", str(path))
    collection.title = "Hidden uncommitted change"
    (snapshot.root / path).write_bytes(snapshot.to_files()[path])
    assert _git(snapshot.root, "status", "--porcelain") == ""
    assert _lookup(snapshot, index) is None


def test_mid_read_head_change_fails_closed(indexed_repo, monkeypatch):
    snapshot, index = indexed_repo
    assert _lookup(snapshot, index) is not None
    original = module._read_subset

    def change_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        (snapshot.root / "new.txt").write_bytes(b"new HEAD")
        _commit(snapshot.root)
        return result

    monkeypatch.setattr(module, "_read_subset", change_after_read)
    assert _lookup(snapshot, index) is None


def test_no_git_native_selectors_and_in_repository_cache_fall_back(tmp_path, indexed_repo):
    snapshot, index = indexed_repo
    orphan = write_fixture(tmp_path / "no-git")
    assert module.lookup_authoring_snapshot(orphan.root, index, [_selector(orphan)]) is None
    assert module.lookup_authoring_snapshot(snapshot.root, index, ["SuspiciousCats"]) is None
    assert (
        module.lookup_authoring_snapshot(
            snapshot.root, snapshot.root / "lookup.sqlite3", [_selector(snapshot)]
        )
        is None
    )
    assert not index.exists()


def test_index_path_alias_cannot_write_inside_repository(indexed_repo):
    snapshot, index = indexed_repo
    aliased = snapshot.root.parent / "unused" / ".." / snapshot.root.name / "lookup.sqlite3"
    assert module.lookup_authoring_snapshot(snapshot.root, aliased, [_selector(snapshot)]) is None
    assert not (snapshot.root / "lookup.sqlite3").exists()
    assert not index.exists()


def test_atomic_rebuild_failure_preserves_previous_index(indexed_repo, monkeypatch):
    snapshot, index = indexed_repo
    assert _lookup(snapshot, index) is not None
    before = index.read_bytes()
    (snapshot.root / "new.txt").write_bytes(b"new HEAD")
    _commit(snapshot.root)

    def fail_replace(*args):
        raise OSError("synthetic atomic replacement failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    assert _lookup(snapshot, index) is None
    assert index.read_bytes() == before
    assert sorted(path.name for path in index.parent.iterdir()) == [index.name]


def test_index_symlink_is_never_followed(indexed_repo):
    snapshot, index = indexed_repo
    assert _lookup(snapshot, index) is not None
    alias = index.with_name("alias.sqlite3")
    try:
        alias.symlink_to(index)
    except OSError:
        pytest.skip("symbolic links are unavailable on this host")
    assert module.lookup_authoring_snapshot(snapshot.root, alias, [_selector(snapshot)]) is None


def test_ignored_ide_artifacts_do_not_disable_cold_or_warm_lookup(indexed_repo, monkeypatch):
    snapshot, index = indexed_repo
    idea = snapshot.root / ".idea"
    idea.mkdir()
    (idea / "workspace.xml").write_bytes(b"local IDE state")
    assert _lookup(snapshot, index) is not None
    monkeypatch.setattr(module, "load_dataset", lambda *_: pytest.fail("warm must use index"))
    assert _lookup(snapshot, index) is not None


@pytest.mark.parametrize("warm", [False, True])
def test_ignored_canonical_shadow_always_falls_back(indexed_repo, monkeypatch, warm):
    snapshot, index = indexed_repo
    if warm:
        assert _lookup(snapshot, index) is not None
    with (snapshot.root / ".git" / "info" / "exclude").open("ab") as stream:
        stream.write(b"\ndata/telegram/emojis/ff/aa.jsonl\n")
    shadow = snapshot.root / "data/telegram/emojis/ff/aa.jsonl"
    shadow.parent.mkdir(parents=True, exist_ok=True)
    shadow.write_bytes(b"")
    assert _git(snapshot.root, "status", "--porcelain") == ""
    before = index.read_bytes() if index.exists() else None
    monkeypatch.setattr(module, "load_dataset", lambda *_: pytest.fail("shadow must fall back"))
    assert _lookup(snapshot, index) is None
    assert (index.read_bytes() if index.exists() else None) == before
