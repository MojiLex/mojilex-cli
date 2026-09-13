from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.github import RepositoryRef
from mojilex_cli.pipeline import workspaces
from mojilex_cli.pipeline.workspaces import prepare_staging_workspace, snapshot_at_revision


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "--initial-branch=main", str(root)], check=True)
    (root / "dataset.json").write_text("{}\n", encoding="utf-8")
    _git(root, "add", "dataset.json")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "base",
    )
    _git(root, "remote", "add", "origin", "https://github.com/MojiLex/mojilex.git")
    return root, _git(root, "rev-parse", "HEAD")


def test_staging_workspace_is_persistent_and_bound_to_base(tmp_path: Path) -> None:
    source, revision = _source_repository(tmp_path)
    runs = tmp_path / "runs"
    run_id = "mlxrun_" + "a" * 32
    target = RepositoryRef.parse("MojiLex/mojilex")

    first = prepare_staging_workspace(
        source,
        target=target,
        runs_dir=runs,
        run_id=run_id,
        base_branch="main",
        base_revision=revision,
    )
    second = prepare_staging_workspace(
        source,
        target=target,
        runs_dir=runs,
        run_id=run_id,
        base_branch="main",
        base_revision=revision,
    )

    assert first == second
    assert first.is_dir()
    assert _git(first, "remote", "get-url", "origin") == ("https://github.com/MojiLex/mojilex.git")

    (first / "dataset.json").write_text('{"staged":true}\n', encoding="utf-8")
    with snapshot_at_revision(first, revision) as base:
        assert (base / "dataset.json").read_text(encoding="utf-8") == "{}\n"

    _git(first, "add", "dataset.json")
    _git(
        first,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "forbidden staging commit",
    )
    with pytest.raises(CommandError, match="base revision"):
        prepare_staging_workspace(
            source,
            target=target,
            runs_dir=runs,
            run_id=run_id,
            base_branch="main",
            base_revision=revision,
        )


def test_local_clones_scope_safe_directory_to_the_exact_git_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, revision = _source_repository(tmp_path)
    real_git = workspaces._git
    clone_commands: list[tuple[str, ...]] = []

    def checked_git(*command: str) -> None:
        values = tuple(command)
        if "clone" in values:
            clone_commands.append(values)
        real_git(*command)

    monkeypatch.setattr(workspaces, "_git", checked_git)
    staging = prepare_staging_workspace(
        source,
        target=RepositoryRef.parse("MojiLex/mojilex"),
        runs_dir=tmp_path / "r",
        run_id="mlxrun_" + "b" * 32,
        base_branch="main",
        base_revision=revision,
    )
    with snapshot_at_revision(staging, revision):
        pass

    expected_source = f"safe.directory={(source / '.git').resolve().as_posix()}"
    expected_staging = f"safe.directory={(staging / '.git').resolve().as_posix()}"
    assert clone_commands[0][:2] == ("-c", expected_source)
    assert clone_commands[1][:2] == ("-c", expected_staging)


def test_revision_snapshot_canonicalizes_its_own_temporary_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mojilex_cli.dataset.layout import assert_no_link_or_reparse

    tmp_path = tmp_path.resolve()
    source, revision = _source_repository(tmp_path)
    parent = tmp_path / "real-temp"
    parent.mkdir()
    alias = tmp_path / "temp-alias"
    try:
        alias.symlink_to(parent, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable on this host")
    real_temporary_directory = workspaces.tempfile.TemporaryDirectory

    def temporary_directory(**kwargs):
        return real_temporary_directory(dir=alias, **kwargs)

    monkeypatch.setattr(workspaces.tempfile, "TemporaryDirectory", temporary_directory)
    with snapshot_at_revision(source, revision) as base:
        assert base == base.resolve()
        assert base.is_relative_to(parent)
        assert_no_link_or_reparse(base)
        assert (base / "dataset.json").read_text(encoding="utf-8") == "{}\n"
        # Public dataset path validation must still reject the original alias.
        with pytest.raises(ValueError, match="link or reparse"):
            assert_no_link_or_reparse(alias / base.relative_to(parent))
    assert not base.exists()
