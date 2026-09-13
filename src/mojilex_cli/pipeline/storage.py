"""Initialize the default persistent dataset without replacing user files."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from filelock import FileLock, Timeout

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.config.paths import default_repository_path
from mojilex_cli.dataset.layout import is_link_or_reparse_point
from mojilex_cli.git.runner import GitError, GitRunner, git_subprocess_environment
from mojilex_cli.github import GitHubError, RepositoryRef


def _validate_repository(path: Path) -> None:
    try:
        git = GitRunner(path)
        root = Path(git.run("rev-parse", "--show-toplevel").stdout.strip()).resolve()
        remote = git.remote_url()
        if remote.startswith("git@github.com:"):
            remote = "https://github.com/" + remote.removeprefix("git@github.com:")
        reference = RepositoryRef.parse(remote)
        if root == path.resolve() and str(reference).casefold() == "mojilex/mojilex":
            return
    except (GitError, GitHubError, ValueError, OSError):
        pass
    raise CommandError(
        "CONFIG_INVALID",
        "The default repository folder is occupied by unrelated or incomplete data.",
        hint="Choose an existing dataset with --repo. Files in this folder were preserved.",
    )


def _refresh_repository(path: Path, base_branch: str, github_token: str | None) -> None:
    """Update clean managed checkouts without changing local work or branch choices."""

    git = GitRunner(path, github_token=github_token)
    if git.run("status", "--porcelain").stdout.strip():
        return
    branch = git.run("symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    if branch.returncode != 0 or branch.stdout.strip() != base_branch:
        return
    git.fetch("origin", base_branch)
    remote_sha = git.current_sha(f"refs/remotes/origin/{base_branch}")
    current_sha = git.current_sha()
    if current_sha == remote_sha or not git.is_ancestor(current_sha, remote_sha):
        return
    # Recheck after network I/O; Git also refuses an update that would overwrite edits.
    if git.run("status", "--porcelain").stdout.strip():
        return
    if (
        git.run("symbolic-ref", "--quiet", "--short", "HEAD", check=False).stdout.strip()
        != base_branch
    ):
        return
    git.run("merge", "--ff-only", remote_sha)


def ensure_default_repository(target: Path, base_branch: str, github_token: str | None) -> None:
    """Clone only the standard application directory, atomically and once."""

    if target.absolute() != default_repository_path():
        raise ValueError("only the default application repository can be initialized")
    # Do not adopt a redirected repository or a workspace controlled through a link.
    if is_link_or_reparse_point(target) or is_link_or_reparse_point(target.parent):
        raise CommandError(
            "CONFIG_INVALID",
            "The default repository must not be a symlink.",
            hint="Choose an existing dataset with --repo.",
        )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(target.parent / "repository.lock"), timeout=120):
            if target.exists():
                _validate_repository(target)
                _refresh_repository(target, base_branch, github_token)
                return
            with tempfile.TemporaryDirectory(prefix=".repository-", dir=target.parent) as raw:
                staged = Path(raw) / "repository"
                with git_subprocess_environment(github_token) as environment:
                    completed = subprocess.run(
                        [
                            "git",
                            "clone",
                            "--filter=blob:none",
                            "--no-tags",
                            "--single-branch",
                            "--branch",
                            base_branch,
                            "--",
                            "https://github.com/MojiLex/mojilex.git",
                            str(staged),
                        ],
                        capture_output=True,
                        check=False,
                        timeout=120,
                        shell=False,
                        env=environment,
                    )
                if completed.returncode != 0:
                    raise CommandError(
                        "GIT_CONFLICT",
                        "Could not initialize the default dataset repository.",
                        hint="Check GitHub access and the configured base branch, then retry.",
                    )
                _validate_repository(staged)
                # Never replace a directory created by another process while cloning.
                if target.exists():
                    raise CommandError(
                        "CONFIG_INVALID",
                        "The repository folder became occupied.",
                        hint=(
                            "Check other running MojiLex processes. Existing files were preserved."
                        ),
                    )
                staged.rename(target)
    except (OSError, subprocess.TimeoutExpired, Timeout, GitError) as exc:
        raise CommandError(
            "GIT_CONFLICT",
            "Could not prepare the application's dataset folder.",
            hint="Check folder permissions, GitHub access, and other running MojiLex processes.",
        ) from exc
