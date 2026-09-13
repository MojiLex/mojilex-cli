from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from mojilex_cli.commands import workflow
from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.config import MojiLexConfig
from test_dataset_helpers import make_snapshot, write_fixture


@pytest.mark.parametrize("hit", (True, False))
def test_authoring_selector_uses_lookup_and_safe_loader_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hit: bool
) -> None:
    from mojilex_cli.cache import lookup

    snapshot = make_snapshot(tmp_path / "dataset")
    identifier = next(iter(snapshot.emojis))
    calls: list[object] = []

    @contextmanager
    def workspace(*_args):
        yield SimpleNamespace(root=snapshot.root)

    def indexed(root, path, selectors, *, read_only):
        calls.append((root, path, selectors, read_only))
        return snapshot if hit else None

    def fallback(root):
        assert not hit, "a verified cache hit must not scan the full tree"
        calls.append(root)
        return snapshot

    monkeypatch.setattr(workflow, "repository_workspace", workspace)
    monkeypatch.setattr(workflow, "load_dataset", fallback)
    monkeypatch.setattr(lookup, "lookup_authoring_snapshot", indexed)
    result = workflow._sources_for_selectors(
        (identifier,),
        str(snapshot.root),
        base_branch="main",
        cache_dir=tmp_path / "cache",
        read_only=True,
    )
    assert result == tuple(item.canonical_url for item in snapshot.collections.values())
    assert calls[0] == (
        snapshot.root,
        tmp_path / "cache" / "lookup-v1.sqlite3",
        (identifier,),
        True,
    )
    assert len(calls) == (1 if hit else 2)


@pytest.mark.parametrize("dry_run", (False, True))
def test_update_selector_propagates_cache_location_and_readonly_preview(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    config = MojiLexConfig(cache_dir=tmp_path / "cache")
    calls: list[dict[str, object]] = []

    def sources(_selectors, _repository, **kwargs):
        calls.append(kwargs)
        return ("https://t.me/addemoji/Example",)

    monkeypatch.setattr(workflow, "load_config", lambda **_kwargs: config)
    monkeypatch.setattr(workflow, "_sources_for_selectors", sources)
    monkeypatch.setattr(workflow, "run_add", lambda *_args: CommandResult())
    workflow.update_command(
        "some-id", all_collections=False, repo=None, dry_run=dry_run, official_pack_policy="allow"
    )
    assert calls == [
        {
            "base_branch": "main",
            "cache_dir": config.cache_dir,
            "read_only": dry_run,
        }
    ]


def test_real_git_authoring_selector_cold_then_readonly_warm_avoids_full_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mojilex_cli.cache import lookup

    snapshot = write_fixture(tmp_path / "dataset")
    (snapshot.root / ".gitignore").write_text(".mojilex/\n", encoding="utf-8")
    for args in (
        ["init", "--quiet"],
        ["add", "--all"],
        [
            "-c",
            "user.name=Test Author",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", *args], cwd=snapshot.root, check=True, capture_output=True)

    @contextmanager
    def workspace(*_args):
        yield SimpleNamespace(root=snapshot.root)

    monkeypatch.setattr(workflow, "repository_workspace", workspace)
    identifier = next(iter(snapshot.emojis))
    arguments = dict(base_branch="main", cache_dir=tmp_path / "cache")
    expected = workflow._sources_for_selectors((identifier,), str(snapshot.root), **arguments)
    index_path = tmp_path / "cache" / "lookup-v1.sqlite3"
    assert index_path.is_file()
    before = index_path.read_bytes()
    monkeypatch.setattr(
        lookup, "load_dataset", lambda *_args: pytest.fail("warm lookup scanned the dataset")
    )
    monkeypatch.setattr(
        workflow, "load_dataset", lambda *_args: pytest.fail("verified warm lookup fell back")
    )
    actual = workflow._sources_for_selectors(
        (identifier,), str(snapshot.root), read_only=True, **arguments
    )
    assert actual == expected
    assert index_path.read_bytes() == before
    assert tuple(index_path.parent.iterdir()) == (index_path,)
