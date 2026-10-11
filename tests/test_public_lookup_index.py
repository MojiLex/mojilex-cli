"""Public-only offline lookup: exact Git provenance, coverage and cache safety."""

from __future__ import annotations

import json
import sqlite3
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from mojilex_cli.dataset.layout import emoji_bucket_path, tombstone_path
from mojilex_cli.domain import emoji_id, membership_id
from mojilex_cli.read import lookup
from mojilex_cli.read.lookup import PUBLIC_REPOSITORY_URL, LookupError, LookupIndex
from test_dataset_helpers import NOW, make_snapshot


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Lookup tests",
            "-c",
            "user.email=lookup-tests@example.test",
            *arguments,
        ],
        capture_output=True,
        check=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _write(root: Path, files: dict[Any, bytes]) -> None:
    for relative, payload in files.items():
        target = root / str(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def _publish(root: Path) -> str:
    _git(root, "add", ".")
    _git(root, "commit", "--allow-empty", "-m", "synthetic public fixture")
    commit = _git(root, "rev-parse", "HEAD")
    _git(root, "update-ref", "refs/remotes/origin/main", commit)
    return commit


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, Any, LookupIndex]:
    root = tmp_path / "public-source"
    root.mkdir()
    _git(root, "init", "--initial-branch=main")
    _git(root, "remote", "add", "origin", PUBLIC_REPOSITORY_URL + ".git")
    snapshot = make_snapshot(root)
    _write(root, snapshot.to_files())
    _publish(root)
    return root, snapshot, LookupIndex(root, tmp_path / "cache")


def _second(snapshot: Any, native_id: str = "1000000000000000002") -> str:
    original = next(iter(snapshot.emojis.values()))
    identifier = emoji_id("telegram", "custom_emoji.id", "global", native_id)
    emoji = original.model_copy(deep=True, update={"id": identifier, "native_id": native_id})
    emoji.descriptions["ru"].text = "Белая собака держит чашку кофе."
    emoji.descriptions["en"].text = "A white dog holds a coffee cup."
    emoji.semantic_tags = ["dog", "coffee"]
    snapshot.emojis[identifier] = emoji
    membership = next(iter(snapshot.memberships.values()))
    new_id = membership_id(membership.collection_id, identifier)
    snapshot.memberships[new_id] = membership.model_copy(
        update={"id": new_id, "emoji_id": identifier, "position": 1}
    )
    return identifier


def test_pending_descriptions_and_provenance_are_searchable_without_network(source, monkeypatch):
    root, snapshot, index = source
    # Any attempted fetch would violate the command set even with no installed credentials.
    original = lookup.subprocess.run

    def offline(command, **kwargs):
        assert not {"fetch", "clone", "ls-remote"}.intersection(command)
        return original(command, **kwargs)

    monkeypatch.setattr(lookup.subprocess, "run", offline)
    result = index.search("желт кот")
    assert result["total"] == 1
    item = result["items"][0]
    assert item["emoji_id"] == next(iter(snapshot.emojis))
    assert set(item["descriptions"]) == {"en", "ru"}
    assert item["concept_mapping_status"] == "pending"
    assert item["review"]["status"] == "unreviewed"
    assert item["provenance"]["origin"] == "ai"
    assert item["packs"][0]["name"] == "SuspiciousCats"
    assert item["origin"] == "public"
    assert "record" not in item
    assert "semantic accuracy" in result["index"]["quality_note"]
    assert result["index"]["pending_concept_mapping"] == 1
    assert result["index"]["source"]["commit"] == _git(root, "rev-parse", "origin/main")
    assert index.search("YELLOW CAT")["total"] == 1
    assert index.search("SuspiciousCats")["total"] == 1
    assert index.get(item["native_id"])["item"]["record"]["descriptions"] == item["descriptions"]


def test_cache_reuses_exact_public_commit_and_ignores_unpublished_worktree_and_head(
    source, monkeypatch
):
    root, snapshot, index = source
    first = index.info()
    assert first["rebuilt"] is True
    public_commit = first["source"]["commit"]
    emoji = next(iter(snapshot.emojis.values()))
    emoji.descriptions["en"].text = "A private unpublished rainbow airplane."
    _second(snapshot)
    _write(root, snapshot.to_files())
    _git(root, "add", ".")
    _git(root, "commit", "-m", "private unpublished commit")
    # Warm queries must not reread either tree/blob data or the dirty filesystem.
    monkeypatch.setattr(
        lookup, "_read_blobs", lambda *args: pytest.fail("rebuilt unchanged public source")
    )
    assert index.search("airplane")["total"] == 0
    assert index.search("dog")["total"] == 0
    assert index.search("cat")["total"] == 1
    assert index.info()["source"]["commit"] == public_commit
    assert index.info()["rebuilt"] is False


def test_public_ref_change_detects_same_size_edits_additions_and_deletes(source):
    root, snapshot, index = source
    assert index.search("cat")["total"] == 1
    emoji = next(iter(snapshot.emojis.values()))
    # Equal-byte-length edits invalidate by new immutable commit, not mtime/size.
    emoji.descriptions["en"].text = emoji.descriptions["en"].text.replace("cat", "owl")
    emoji.semantic_tags = ["owl", "doubt", "suspicious"]
    _second(snapshot)
    _write(root, snapshot.to_files())
    new_commit = _publish(root)
    assert index.search("owl")["total"] == 1
    assert index.search("coffee")["total"] == 1
    assert index.info()["source"]["commit"] == new_commit
    (root / str(emoji_bucket_path(emoji.platform, emoji.id))).unlink()
    _publish(root)
    assert index.search("owl")["total"] == 0
    assert index.info()["indexed_emojis"] == 1


@pytest.mark.parametrize("target", ["emoji", "collection", "membership", "tombstone"])
def test_inactive_and_withheld_records_never_appear(source, target):
    root, snapshot, index = source
    assert index.search("cat")["total"] == 1
    if target == "tombstone":
        identifier = next(iter(snapshot.emojis))
        payload = {
            "schema_version": "1.0.0",
            "entity_type": "tombstone",
            "target_entity_type": "emoji",
            "target_id": identifier,
            "reason_code": "copyright",
            "withheld_at": NOW,
            "public_note": "Synthetic withholding.",
        }
        _write(root, {tombstone_path(identifier): (json.dumps(payload) + "\n").encode()})
    elif target == "membership":
        next(iter(snapshot.memberships.values())).status = "removed_from_collection"
        _write(root, snapshot.to_files())
    else:
        entity = next(
            iter(snapshot.emojis.values() if target == "emoji" else snapshot.collections.values())
        )
        entity.availability.status = "unavailable"
        _write(root, snapshot.to_files())
    _publish(root)
    assert index.search("cat")["total"] == 0


def test_malformed_rows_are_reported_without_dropping_other_valid_rows(source):
    root, snapshot, index = source
    emoji = next(iter(snapshot.emojis.values()))
    path = root / str(emoji_bucket_path(emoji.platform, emoji.id))
    path.write_bytes(path.read_bytes() + b'{"not":"an emoji"}\n')
    _publish(root)
    result = index.search("cat")
    assert result["total"] == 1
    assert result["index"]["invalid_rows"] == 1
    assert result["index"]["complete"] is False
    assert result["index"]["warnings"]


def test_failed_rebuild_keeps_previous_cache_intact(source):
    root, _, index = source
    old = index.info()
    before = index.path.read_bytes()
    (root / "dataset.json").write_text("broken", encoding="utf-8")
    _publish(root)
    with pytest.raises(LookupError, match="manifest"):
        index.refresh()
    assert index.path.read_bytes() == before
    _git(root, "update-ref", "refs/remotes/origin/main", old["source"]["commit"])
    assert index.search("cat")["total"] == 1


def test_invalid_public_tombstone_fails_closed(source):
    root, _, index = source
    index.info()
    _write(root, {"tombstones/00/invalid.json": b"{}\n"})
    _publish(root)
    with pytest.raises(LookupError, match="tombstone"):
        index.search("cat")


def test_oversized_public_tombstone_fails_closed(source, monkeypatch):
    root, _, index = source
    index.info()
    _write(root, {"tombstones/00/oversized.json": b" " * 4096 + b"{}\n"})
    _publish(root)
    monkeypatch.setattr(lookup, "_MAX_ROW_BYTES", 4000)
    with pytest.raises(LookupError, match="tombstone"):
        index.search("cat")


def test_query_provenance_is_bound_to_database_used_for_rows(source, monkeypatch):
    root, snapshot, index = source
    old = index.info()
    next(iter(snapshot.emojis.values())).descriptions["en"].text = "A newly published blue fox."
    _write(root, snapshot.to_files())
    new_commit = _publish(root)
    # Simulate another reader rebuilding after this reader's freshness phase.
    refreshed = index.refresh()
    assert refreshed["source"]["commit"] == new_commit
    monkeypatch.setattr(index, "refresh", lambda: old)
    result = index.search("fox")
    assert result["total"] == 1
    assert result["index"]["source"]["commit"] == new_commit
    assert result["index"]["rebuilt"] is False
    identifier = next(iter(snapshot.emojis))
    assert index.get(identifier)["index"]["source"]["commit"] == new_commit


def test_sql_and_fts_operators_are_literal_bound_queries(source):
    _, _, index = source
    assert index.search("cat OR dog")["total"] == 0
    assert index.search("' OR 1=1 --")["total"] == 0
    assert index.search('" OR "')["total"] == 0
    assert index.search("cat")["total"] == 1
    with pytest.raises(LookupError):
        index.get("' OR 1=1 --")


def test_ranking_pagination_and_parameter_bounds(source):
    root, snapshot, index = source
    _second(snapshot)
    _write(root, snapshot.to_files())
    _publish(root)
    one = index.search("Suspicious", limit=1)
    two = index.search("Suspicious", limit=1, offset=1)
    both = index.search("Suspicious")
    assert one["total"] == two["total"] == 2
    assert one["items"] + two["items"] == both["items"]
    assert one["items"][0]["emoji_id"] != two["items"][0]["emoji_id"]
    for kwargs in ({"limit": 0}, {"limit": 101}, {"offset": -1}):
        with pytest.raises(LookupError):
            index.search("cat", **kwargs)
    for query in ("", " " * 10, "x" * 513):
        with pytest.raises(LookupError):
            index.search(query)


@pytest.mark.parametrize(
    "value",
    ["https://github.com/other/mojilex.git", "https://token@github.com/MojiLex/mojilex.git"],
)
def test_wrong_origin_and_credential_bearing_remote_are_rejected(source, value):
    root, _, index = source
    _git(root, "remote", "set-url", "origin", value)
    with pytest.raises(LookupError, match="origin"):
        index.info()


def test_missing_public_ref_never_falls_back_to_head(source):
    root, _, index = source
    _git(root, "update-ref", "-d", "refs/remotes/origin/main")
    with pytest.raises(LookupError, match="revision"):
        index.info()


def test_partial_clone_is_rejected_without_lazy_network(source):
    root, _, index = source
    _git(root, "config", "remote.origin.promisor", "true")
    with pytest.raises(LookupError, match="complete Git clone"):
        index.info()


def test_cache_inside_source_is_rejected(source):
    root, _, _ = source
    with pytest.raises(LookupError, match="outside"):
        LookupIndex(root, root / "cache")


def test_public_git_symlink_is_rejected_even_with_safe_worktree(source):
    root, _, index = source
    # Update Git index directly: works on Windows without symlink privileges.
    oid = _git(root, "rev-parse", "HEAD:dataset.json")
    _git(
        root,
        "update-index",
        "--add",
        "--cacheinfo",
        "120000",
        oid,
        "data/telegram/emojis/unsafe.jsonl",
    )
    _git(root, "commit", "-m", "synthetic link")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    with pytest.raises(LookupError, match="links"):
        index.info()


def test_cached_payload_corruption_is_detected_and_explicit_rebuild_recovers(source):
    _, _, index = source
    index.info()
    with closing(sqlite3.connect(index.path)) as connection:
        connection.execute("UPDATE documents SET payload='{}'")
        connection.commit()
    with pytest.raises(LookupError, match="corrupt"):
        index.search("cat")
    index.refresh(force=True)
    assert index.search("cat")["total"] == 1


def test_fallback_prefix_search_without_fts5(source, monkeypatch):
    _, _, index = source
    real_connect = lookup.sqlite3.connect

    class WithoutFTS(sqlite3.Connection):
        def execute(self, statement, parameters=()):
            if statement.startswith("CREATE VIRTUAL TABLE"):
                raise sqlite3.OperationalError("no such module: fts5")
            return super().execute(statement, parameters)

    def connect(*args, **kwargs):
        return real_connect(*args, factory=WithoutFTS, **kwargs)

    monkeypatch.setattr(lookup.sqlite3, "connect", connect)
    result = index.search("желт кот")
    assert result["total"] == 1
    assert result["index"]["engine"] == "python-prefix"
    assert index.search("yellow cat")["total"] == 1
