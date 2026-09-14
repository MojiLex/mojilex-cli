from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.config import Credentials, paths
from mojilex_cli.pipeline import runner, storage
from mojilex_cli.pipeline.runner import repository_workspace


@pytest.fixture
def managed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "user_state_path", lambda *_args: tmp_path / "AppData")
    monkeypatch.setattr(runner, "load_credentials", lambda: Credentials())
    return paths.default_repository_path()


@pytest.fixture
def clones(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Exercise real Git/rename/locking, replacing only the network clone source."""
    run = subprocess.run
    source = tmp_path / "source"
    run(["git", "init", "-b", "main", str(source)], check=True, capture_output=True)
    (source / "dataset.json").write_text("{}", encoding="utf-8")
    run(["git", "-C", str(source), "add", "."], check=True, capture_output=True)
    run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "base",
        ],
        check=True,
        capture_output=True,
    )
    calls: list[Path] = []

    def local_clone(args, **kwargs):
        if "-C" in args:
            command_index = args.index("-C") + 2
            if len(args) > command_index + 2 and args[command_index] == "fetch":
                remote_index = command_index + 2
                assert args[remote_index] == "origin"
                return run(
                    [*args[:remote_index], str(source), *args[remote_index + 1 :]],
                    **kwargs,
                )
        if args[:2] != ["git", "clone"]:
            return run(args, **kwargs)
        assert args[-2] == "https://github.com/MojiLex/mojilex.git"
        destination = Path(args[-1])
        calls.append(destination)
        result = run([*args[:-2], str(source), str(destination)], **kwargs)
        if result.returncode == 0:
            run(
                ["git", "-C", str(destination), "remote", "set-url", "origin", args[-2]],
                check=True,
                capture_output=True,
            )
        return result

    monkeypatch.setattr(storage.subprocess, "run", local_clone)
    return calls


def test_default_workspace_clones_once_and_preserves_edits_from_any_cwd(
    managed: Path, clones: list[Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with repository_workspace(str(managed), "main") as workspace:
        assert workspace.root == managed
        assert workspace.temporary is False
        assert str(workspace.target) == "MojiLex/mojilex"
    (managed / "dataset.json").write_text("user changes", encoding="utf-8")
    monkeypatch.chdir(tmp_path.parent)
    with repository_workspace(str(managed), "main"):
        assert (managed / "dataset.json").read_text(encoding="utf-8") == "user changes"
    assert len(clones) == 1


def test_concurrent_initialization_clones_once(managed: Path, clones: list[Path]) -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(storage.ensure_default_repository, managed, "main", None)
            for _ in range(2)
        ]
        for future in futures:
            future.result()
    assert len(clones) == 1


def test_next_open_updates_clean_default_from_remote(
    managed: Path, clones: list[Path], tmp_path: Path
) -> None:
    storage.ensure_default_repository(managed, "main", None)
    source = tmp_path / "source"
    (source / "dataset.json").write_text("new upstream version", encoding="utf-8")
    git = storage.GitRunner(source)
    git.run("add", "dataset.json")
    git.run("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "next")
    with repository_workspace(str(managed), "main"):
        assert storage.GitRunner(managed).current_sha() == git.current_sha()
        assert (managed / "dataset.json").read_text(encoding="utf-8") == "new upstream version"
    assert len(clones) == 1


@pytest.mark.parametrize("local_work", ["commit", "branch", "dirty"])
def test_refresh_preserves_local_work(
    managed: Path,
    clones: list[Path],
    monkeypatch: pytest.MonkeyPatch,
    local_work: str,
) -> None:
    storage.ensure_default_repository(managed, "main", None)
    git = storage.GitRunner(managed)
    if local_work == "branch":
        git.run("checkout", "-b", "personal")
    else:
        (managed / "dataset.json").write_text("local work", encoding="utf-8")
        if local_work == "commit":
            git.run("add", "dataset.json")
            git.run(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-m",
                "local",
            )
    before_sha = git.current_sha()
    before_bytes = (managed / "dataset.json").read_bytes()
    if local_work != "commit":

        def unexpected_fetch(*_args, **_kwargs):
            pytest.fail("dirty or alternate branch checkouts must not fetch")

        monkeypatch.setattr(storage.GitRunner, "fetch", unexpected_fetch)
    storage.ensure_default_repository(managed, "main", None)
    assert git.current_sha() == before_sha
    assert (managed / "dataset.json").read_bytes() == before_bytes
    assert len(clones) == 1


@pytest.mark.parametrize("occupied", ["file", "directory", "unrelated-git"])
def test_occupied_default_is_preserved(managed: Path, clones: list[Path], occupied: str) -> None:
    managed.parent.mkdir(parents=True)
    if occupied == "file":
        sentinel = managed
    else:
        managed.mkdir()
        sentinel = managed / "keep.txt"
    sentinel.write_bytes(b"preserve")
    if occupied == "unrelated-git":
        subprocess.run(["git", "init", str(managed)], check=True, capture_output=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(managed),
                "remote",
                "add",
                "origin",
                "https://github.com/Other/project.git",
            ],
            check=True,
            capture_output=True,
        )
    with pytest.raises(CommandError) as captured:
        storage.ensure_default_repository(managed, "main", None)
    assert captured.value.error.code == "CONFIG_INVALID"
    assert sentinel.read_bytes() == b"preserve"
    assert clones == []


def test_refresh_failure_preserves_existing_checkout(
    managed: Path, clones: list[Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    storage.ensure_default_repository(managed, "main", None)
    before_sha = storage.GitRunner(managed).current_sha()
    before_bytes = (managed / "dataset.json").read_bytes()

    def unavailable(*_args, **_kwargs):
        raise storage.GitError("network unavailable")

    monkeypatch.setattr(storage.GitRunner, "fetch", unavailable)
    with pytest.raises(CommandError) as captured:
        storage.ensure_default_repository(managed, "main", None)
    assert captured.value.error.code == "GIT_CONFLICT"
    assert storage.GitRunner(managed).current_sha() == before_sha
    assert (managed / "dataset.json").read_bytes() == before_bytes


def test_failed_clone_can_be_retried_without_partial_repository(
    managed: Path, clones: list[Path]
) -> None:
    with pytest.raises(CommandError):
        storage.ensure_default_repository(managed, "missing-branch", None)
    assert not managed.exists()
    assert not list(managed.parent.glob(".repository-*"))
    storage.ensure_default_repository(managed, "main", None)
    assert (managed / "dataset.json").is_file()
    assert len(clones) == 2


def test_custom_missing_path_is_not_created(managed: Path, tmp_path: Path) -> None:
    custom = tmp_path / "custom"
    with pytest.raises(ValueError):
        storage.ensure_default_repository(custom, "main", None)
    assert not custom.exists()
    assert not managed.parent.exists()
