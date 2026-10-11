"""Shipped public lookup flow, credentials isolation, and source preservation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mojilex_cli import cli
from mojilex_cli.commands import interactive, lookup
from mojilex_cli.config import credential_store
from mojilex_cli.read.lookup import PUBLIC_REPOSITORY_URL
from test_dataset_helpers import make_snapshot
from test_public_lookup_index import _git, _publish, _write


@pytest.fixture
def public_source(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    _git(root, "init", "--initial-branch=main")
    _git(root, "remote", "add", "origin", PUBLIC_REPOSITORY_URL + ".git")
    snapshot = make_snapshot(root)
    _write(root, snapshot.to_files())
    _publish(root)
    monkeypatch.setattr(lookup, "default_workspace_dir", lambda: tmp_path / "application")
    monkeypatch.setattr(lookup, "default_lookup_repository", lambda: root)
    monkeypatch.setattr(
        credential_store,
        "read_stored_credentials",
        lambda *args: pytest.fail("lookup must not read credentials"),
    )
    monkeypatch.setattr(
        lookup, "_public_network_git", lambda *args: pytest.fail("offline lookup used network")
    )
    return root, snapshot


def test_public_search_get_info_offline_json_and_readable_output(public_source):
    root, snapshot = public_source
    runner = CliRunner()
    for query in ("желт кот", "yellow cat", "SuspiciousCats"):
        output = runner.invoke(cli.app, ["lookup", "search", query, "--json"])
        assert output.exit_code == 0, output.output
        assert output.stdout.count("\n") == 1
        assert output.stderr == ""
        body = json.loads(output.stdout)
        assert body["command"] == "lookup search"
        assert body["result"]["total"] == 1
        assert body["result"]["index"]["source"]["commit"] == _git(root, "rev-parse", "origin/main")
        identifier = body["result"]["items"][0]["native_id"]
    output = runner.invoke(cli.app, ["lookup", "get", identifier, "--json"])
    assert output.exit_code == 0, output.output
    assert json.loads(output.stdout)["result"]["item"]["emoji_id"] == next(iter(snapshot.emojis))
    output = runner.invoke(cli.app, ["lookup", "info", "--json"])
    assert json.loads(output.stdout)["result"]["complete"] is True
    assert json.loads(output.stdout)["result"]["pending_concept_mapping"] == 1
    output = runner.invoke(cli.app, ["lookup", "search", "кот"])
    assert output.exit_code == 0
    assert "SuspiciousCats" in output.stdout
    assert "unreviewed" in output.stdout
    assert "'descriptions':" not in output.stdout


def test_real_entrypoint_machine_mode_and_basic_help(public_source, monkeypatch, capsys):
    monkeypatch.setattr(cli.sys, "argv", ["mojilex", "lookup", "search", "cat", "--json"])
    cli.main()
    output = capsys.readouterr()
    assert output.err == ""
    assert json.loads(output.out)["command"] == "lookup search"
    command = __import__("typer").main.get_command(cli.app)
    from mojilex_cli.i18n import localize_command_tree

    localize_command_tree(command, "ru")
    assert not command.commands["lookup"].hidden


def test_missing_source_is_actionable_without_paid_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(lookup, "default_lookup_repository", lambda: tmp_path / "missing")
    output = CliRunner().invoke(cli.app, ["lookup", "search", "cat", "--json"])
    assert output.exit_code == 2
    error = json.loads(output.stdout)["errors"][0]
    assert error["code"] == "CONFIG_MISSING"
    assert "lookup refresh" in error["hint"]


@pytest.mark.parametrize(
    "arguments,code",
    [
        (["search", "   "], "QUERY_INVALID"),
        (["search", "cat", "--limit", "0"], "CONFIG_INVALID"),
        (["get", "1000000000000000009"], "ENTITY_NOT_FOUND"),
    ],
)
def test_invalid_query_and_missing_identity_are_machine_errors(public_source, arguments, code):
    # CliRunner catches Typer parsing errors itself; main() emits the boundary envelope.
    if code == "CONFIG_INVALID":
        assert CliRunner().invoke(cli.app, ["lookup", *arguments, "--json"]).exit_code == 2
        return
    output = CliRunner().invoke(cli.app, ["lookup", *arguments, "--json"])
    assert output.exit_code != 0
    assert json.loads(output.stdout)["errors"][0]["code"] == code


def test_refresh_keeps_worktree_local_head_and_private_files(public_source, monkeypatch):
    root, _ = public_source
    before = _git(root, "rev-parse", "HEAD")
    private = root / "private-draft.txt"
    private.write_text("private draft", encoding="utf-8")
    calls = []
    monkeypatch.setattr(lookup, "_public_network_git", lambda *args: calls.append(args))
    output = CliRunner().invoke(cli.app, ["lookup", "refresh", "--json"])
    assert output.exit_code == 0, output.output
    assert calls and "fetch" in calls[0]
    assert PUBLIC_REPOSITORY_URL + ".git" in calls[0]
    assert _git(root, "rev-parse", "HEAD") == before
    assert private.read_text(encoding="utf-8") == "private draft"
    calls.clear()
    output = CliRunner().invoke(cli.app, ["lookup", "refresh", "--offline", "--json"])
    assert output.exit_code == 0
    assert calls == []
    assert json.loads(output.stdout)["result"]["public_source_fetched"] is False


def test_wrong_origin_is_rejected_before_fetch(public_source):
    root, _ = public_source
    _git(root, "remote", "set-url", "origin", "https://github.com/example/private.git")
    output = CliRunner().invoke(cli.app, ["lookup", "refresh", "--json"])
    assert output.exit_code == 2
    assert json.loads(output.stdout)["errors"][0]["code"] == "CONFIG_INVALID"


def test_initial_public_clone_is_managed_and_atomic(public_source, tmp_path, monkeypatch):
    fixture, _ = public_source
    target = tmp_path / "application" / "public-lookup-repository"
    monkeypatch.setattr(lookup, "default_lookup_repository", lambda: target)

    def clone(*args):
        assert args[0] == "clone"
        assert "--no-checkout" in args
        assert "--filter=blob:none" not in args
        assert PUBLIC_REPOSITORY_URL + ".git" in args
        shutil.copytree(fixture, Path(args[-1]))

    monkeypatch.setattr(lookup, "_public_network_git", clone)
    output = CliRunner().invoke(cli.app, ["lookup", "refresh", "--json"])
    assert output.exit_code == 0, output.output
    assert (target / ".git").is_dir()
    assert json.loads(output.stdout)["result"]["indexed_emojis"] == 1
    assert not list(target.parent.glob("mojilex-public-lookup-*"))


def test_anonymous_fetch_rejects_local_url_rewrites_before_network(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init", "--initial-branch=main")
    _git(root, "config", "url.https://example.invalid/private/.insteadOf", "https://github.com/")
    with pytest.raises(lookup.CommandError, match="redirected"):
        lookup._public_network_git(
            "-C",
            str(root),
            "fetch",
            "--no-tags",
            PUBLIC_REPOSITORY_URL + ".git",
            "refs/heads/main:refs/remotes/origin/main",
        )


def test_public_git_strips_ai_keys_auth_and_global_git_configuration(monkeypatch):
    for name in ("OPENAI_API_KEY", "GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.setenv(name, "synthetic-secret-not-for-git")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(lookup.subprocess, "run", run)
    lookup._public_network_git(
        "clone", "--no-checkout", PUBLIC_REPOSITORY_URL + ".git", "new-source"
    )
    environment = calls[0][1]["env"]
    assert not {
        "OPENAI_API_KEY",
        "GEMINI_API_KEY",
        "TELEGRAM_BOT_TOKEN",
        "GITHUB_TOKEN",
        "GIT_CONFIG_COUNT",
    }.intersection(environment)
    assert environment["GIT_CONFIG_GLOBAL"] == os.devnull
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "credential.helper=" in calls[0][0]
    assert "core.askPass=" in calls[0][0]


def test_menu_public_search_uses_only_lookup_dispatch(monkeypatch):
    choices = iter((5, 0, 6))
    monkeypatch.setattr(interactive, "select", lambda *args, **kwargs: next(choices))
    monkeypatch.setattr(interactive.typer, "prompt", lambda *args, **kwargs: "calendar")
    monkeypatch.setattr(interactive, "pause", lambda: None)
    calls = []
    interactive.run_menu(lambda args: calls.append(args) or True)
    assert calls == [["lookup", "search", "calendar"]]
