"""Committed private test checkouts, independent of source Git maintenance."""

from pathlib import Path

from mojilex_cli.git import GitRunner
from mojilex_cli.github import RepositoryRef
from test_pipeline_workspaces import _git


def clone_fixture_repository(
    source: Path, destination: Path, *, target: RepositoryRef, base_revision: str
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Git's transport reads the committed tree while source auto-maintenance can
    # repack loose objects; copying a live .git directory races with that work.
    # Synthetic fixtures lack .gitattributes, so preserve their hashed JSON bytes.
    _git(
        source,
        "-c",
        "core.autocrlf=false",
        "clone",
        "--no-local",
        "--no-hardlinks",
        str(source),
        str(destination),
    )
    git = GitRunner(destination)
    assert git.current_sha() == base_revision
    git.run("remote", "set-url", "origin", f"https://github.com/{target}.git")
    # Keep the ignored persistent lock artifact exercised by saved_add.
    lock = destination / ".mojilex" / "locks" / "dataset-transaction-v1.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.touch()
