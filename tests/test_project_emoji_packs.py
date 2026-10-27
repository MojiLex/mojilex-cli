"""Project choices constrain actual lookup results across agent environments."""

from __future__ import annotations

import json
import sqlite3

import pytest
from typer.testing import CliRunner

from mojilex_cli import cli
from mojilex_cli.commands import lookup
from mojilex_cli.config import credential_store
from mojilex_cli.domain import collection_id, membership_id
from mojilex_cli.read import project
from mojilex_cli.read.lookup import PUBLIC_REPOSITORY_URL, LookupError, LookupIndex
from test_dataset_helpers import make_snapshot
from test_public_lookup_index import _git, _publish, _second, _write


@pytest.fixture
def catalogue(tmp_path, monkeypatch):
    root = tmp_path / "public"
    root.mkdir()
    _git(root, "init", "--initial-branch=main")
    _git(root, "remote", "add", "origin", PUBLIC_REPOSITORY_URL + ".git")
    snapshot = make_snapshot(root)
    original = next(iter(snapshot.collections.values()))
    tools_id = collection_id("telegram", "sticker_set.name", "global", "BotTools")
    snapshot.collections[tools_id] = original.model_copy(
        update={
            "id": tools_id,
            "native_id": "BotTools",
            "title": "Bot interface symbols",
            "canonical_url": "https://t.me/addemoji/BotTools",
        }
    )
    for number, role in ((2, "help"), (3, "settings")):
        identifier = _second(snapshot, f"100000000000000000{number}")
        emoji = snapshot.emojis[identifier]
        emoji.descriptions["en"].text = f"A cat showing a {role} symbol."
        emoji.descriptions["ru"].text = f"Кот и символ {role}."
        emoji.semantic_tags = ["cat", role]
        old_membership = next(m for m in snapshot.memberships.values() if m.emoji_id == identifier)
        del snapshot.memberships[old_membership.id]
        new_id = membership_id(tools_id, identifier)
        snapshot.memberships[new_id] = old_membership.model_copy(
            update={
                "id": new_id,
                "collection_id": tools_id,
            }
        )
    _write(root, snapshot.to_files())
    _publish(root)
    monkeypatch.setattr(lookup, "default_lookup_repository", lambda: root)
    monkeypatch.setattr(lookup, "default_workspace_dir", lambda: tmp_path / "application")
    monkeypatch.setattr(
        credential_store,
        "read_stored_credentials",
        lambda *args: pytest.fail("lookup read authoring credentials"),
    )
    monkeypatch.setattr(
        lookup, "_public_network_git", lambda *args: pytest.fail("lookup used the network")
    )
    return root, snapshot, LookupIndex(root, tmp_path / "index")


def invoke(*arguments):
    output = CliRunner().invoke(cli.app, ["lookup", *arguments, "--json"])
    body = json.loads(output.stdout)
    assert output.exit_code == 0, body
    return body["result"]


@pytest.mark.parametrize("engine", ["fts5", "python-prefix"])
def test_pack_filter_runs_before_count_ranking_and_pagination(catalogue, engine):
    _, _, index = catalogue
    metadata = index.info()
    if engine == "python-prefix":
        with sqlite3.connect(index.path) as connection:
            metadata["engine"] = engine
            connection.execute(
                "UPDATE metadata SET value=? WHERE key='info'", (json.dumps(metadata),)
            )
    assert index.search("cat")["total"] == 3
    result = index.search("cat", packs=["https://t.me/addemoji/BotTools/"], limit=1, offset=1)
    assert result["total"] == 2
    assert len(result["items"]) == 1
    assert all(any(p["name"] == "BotTools" for p in item["packs"]) for item in result["items"])
    assert result["selected_packs"][0]["url"] == "https://t.me/addemoji/BotTools"
    assert index.search("settings", packs=["SuspiciousCats"])["total"] == 0
    assert index.search("cat", packs=["bottools", "BotTools"])["total"] == 2
    assert index.search("cat", packs=["BotTools", "SuspiciousCats"])["total"] == 3
    with pytest.raises(LookupError, match="selected pack"):
        index.search("cat", packs=["MissingPack"])


def test_pack_discovery_supplies_links_examples_and_context(catalogue):
    _, _, index = catalogue
    packs = index.packs()
    assert packs["total"] == 2
    assert all(item["url"].startswith("https://t.me/addemoji/") for item in packs["items"])
    result = index.packs("settings")["items"]
    assert [item["name"] for item in result] == ["BotTools"]
    assert result[0]["emoji_count"] == 2
    assert "settings" in result[0]["examples"][0]["descriptions"]["en"]["text"]
    assert index.packs("no-such-context")["total"] == 0
    assert index.packs(limit=1, offset=1)["total"] == 2


def test_choice_is_shared_by_cli_agents_nested_folders_and_isolated_projects(
    catalogue, tmp_path, monkeypatch
):
    first = tmp_path / "first-project"
    second = tmp_path / "other-project"
    nested = first / "src"
    nested.mkdir(parents=True)
    second.mkdir()
    _git(first, "init")
    assert invoke("project", "show", "--project", str(first))["configured"] is False
    selected = invoke(
        "project", "select", "--project", str(first), "--pack", "https://t.me/addemoji/BotTools"
    )
    assert selected["packs"] == ["BotTools"]
    path = first / ".mojilex" / "emoji-packs.json"
    before = path.read_bytes()
    monkeypatch.chdir(nested)
    result = invoke("search", "cat")
    assert result["total"] == 2
    assert result["selection"]["source"] == "project"
    assert invoke("project", "show")["packs"] == ["BotTools"]
    assert invoke("search", "cat", "--project", str(first))["total"] == 2
    assert invoke("search", "cat", "--project", str(second))["total"] == 3
    assert invoke("search", "cat", "--pack", "SuspiciousCats")["total"] == 1
    assert path.read_bytes() == before
    # A nested independent Git project must not inherit the enclosing project's choice.
    _git(nested, "init")
    assert invoke("search", "cat")["selection"]["source"] == "all"
    assert not (second / ".mojilex").exists()


def test_missing_or_malformed_selection_never_broadens_search(catalogue, tmp_path):
    path = tmp_path / ".mojilex" / "emoji-packs.json"
    path.parent.mkdir()
    for payload in ('{"version":1,"packs":["MissingPack"]}', '{"bad": true}', "{broken"):
        path.write_text(payload, encoding="utf-8")
        output = CliRunner().invoke(
            cli.app, ["lookup", "search", "cat", "--project", str(tmp_path), "--json"]
        )
        assert output.exit_code != 0
        assert json.loads(output.stdout)["errors"]
    before = path.read_bytes()
    output = CliRunner().invoke(
        cli.app,
        [
            "lookup",
            "project",
            "select",
            "--project",
            str(tmp_path),
            "--pack",
            "MissingPack",
            "--json",
        ],
    )
    assert output.exit_code != 0
    assert path.read_bytes() == before


def test_atomic_save_failure_preserves_previous_choice(tmp_path, monkeypatch):
    project.save_preferences(["FirstPack"], tmp_path)
    path = project.preferences_path(tmp_path)
    before = path.read_bytes()

    def fail(*args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(project.os, "replace", fail)
    with pytest.raises(LookupError, match="Could not save"):
        project.save_preferences(["SecondPack"], tmp_path)
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".emoji-packs-*"))


def test_preferences_reject_links(tmp_path):
    outside = tmp_path / "other.json"
    outside.write_text('{"version":1,"packs":["Outside"]}', encoding="utf-8")
    path = project.preferences_path(tmp_path)
    path.parent.mkdir()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("OS does not permit creating test symlinks")
    with pytest.raises(LookupError, match="links"):
        project.read_preferences(tmp_path)
    with pytest.raises(LookupError, match="links"):
        project.save_preferences(["NewPack"], tmp_path)
    assert "Outside" in outside.read_text(encoding="utf-8")


def test_human_commands_render_links_and_saved_choice(catalogue, tmp_path):
    output = CliRunner().invoke(cli.app, ["--ui-language", "en", "lookup", "packs", "settings"])
    assert output.exit_code == 0, output.output
    assert "https://t.me/addemoji/BotTools" in output.stdout
    output = CliRunner().invoke(
        cli.app,
        [
            "--ui-language",
            "en",
            "lookup",
            "project",
            "select",
            "--pack",
            "BotTools",
            "--project",
            str(tmp_path),
        ],
    )
    assert output.exit_code == 0, output.output
    assert "BotTools" in output.stdout
