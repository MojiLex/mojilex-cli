"""Persistent private staging and isolated publication workspaces."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.git import GitRunner
from mojilex_cli.github import RepositoryRef

_RUN_ID = re.compile(r"mlxrun_[0-9a-f]{32}\Z")


def staging_workspace_path(runs_dir: Path, run_id: str) -> Path:
    if not _RUN_ID.fullmatch(run_id):
        raise CommandError(
            "CONFIG_INVALID",
            "Invalid run ID.",
            hint="Pass the exact mlxrun_ identifier returned by import.",
        )
    root = runs_dir.expanduser().resolve() / "workspaces"
    return root / run_id


def prepare_staging_workspace(
    source_repository: Path,
    *,
    target: RepositoryRef,
    runs_dir: Path,
    run_id: str,
    base_branch: str,
    base_revision: str,
) -> Path:
    """Create or verify the private text-only checkout retained for one run."""

    destination = staging_workspace_path(runs_dir, run_id)
    if destination.exists():
        _verify_workspace(destination, target, base_revision)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{run_id}-{uuid.uuid4().hex}.tmp"
    try:
        _git(
            "clone",
            "--no-hardlinks",
            "--no-tags",
            "--single-branch",
            "--branch",
            base_branch,
            str(source_repository),
            str(temporary),
        )
        runner = GitRunner(temporary)
        if runner.current_sha() != base_revision:
            raise CommandError(
                "SOURCE_CHANGED_DURING_RUN",
                "The staging checkout does not match the recorded base revision.",
                hint="Start a new import from the current base.",
            )
        runner.run("remote", "set-url", "origin", f"https://github.com/{target}.git")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    _verify_workspace(destination, target, base_revision)
    return destination


@contextmanager
def snapshot_at_revision(repository: Path, revision: str) -> Iterator[Path]:
    """Yield a detached read-only-by-convention checkout for one recorded commit."""

    with tempfile.TemporaryDirectory(prefix="mojilex-base-") as raw:
        root = Path(raw) / "repository"
        _git("clone", "--no-hardlinks", "--no-checkout", str(repository), str(root))
        _git("-C", str(root), "checkout", "--detach", revision)
        yield root


def _verify_workspace(path: Path, target: RepositoryRef, base_revision: str) -> None:
    if path.is_symlink() or not path.is_dir() or not (path / ".git").exists():
        raise CommandError(
            "DIRTY_WORKTREE",
            "The saved staging workspace is missing or unsafe.",
            hint="Keep the run workspace unchanged, or start a new import.",
        )
    runner = GitRunner(path)
    if runner.current_sha() != base_revision:
        raise CommandError(
            "SOURCE_CHANGED_DURING_RUN",
            "The saved staging workspace no longer matches its recorded base revision.",
            hint="Do not commit inside staging; start a new import if the base changed.",
        )
    remote = runner.remote_url()
    expected = f"https://github.com/{target}.git"
    if remote.removesuffix(".git") != expected.removesuffix(".git"):
        raise CommandError(
            "GIT_CONFLICT",
            "The saved staging workspace points at a different repository.",
            hint="Do not reuse staging directories between repositories.",
        )


def _git(*arguments: str) -> None:
    environment = dict(os.environ)
    for name in (
        "TELEGRAM_BOT_TOKEN",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    ):
        environment.pop(name, None)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    try:
        completed = subprocess.run(
            ["git", *arguments],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            timeout=120,
            shell=False,
            env=environment,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CommandError(
            "GIT_CONFLICT",
            "Could not create an isolated dataset checkout.",
            hint="Check Git, repository access, and the configured base branch.",
        ) from exc
    if completed.returncode != 0:
        raise CommandError(
            "GIT_CONFLICT",
            "Could not create an isolated dataset checkout.",
            hint="Check Git, repository access, and the configured base branch.",
        )
