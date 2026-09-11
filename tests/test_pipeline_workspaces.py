from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.github import RepositoryRef
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
