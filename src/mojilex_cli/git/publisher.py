"""Higher-level local branch and explicit-path commit preparation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Literal

from .runner import GitError, GitIdentity, GitRunner, branch_slug


@dataclass(frozen=True)
class PreparedCommit:
    branch: str
    base_sha: str
    commit_sha: str
    paths: tuple[str, ...]
    message: str | None = None
    identity: GitIdentity | None = None


@dataclass(frozen=True)
class TargetGuard:
    """Proof that target paths were clean immediately before the pipeline wrote them."""

    repository: Path
    base_sha: str
    paths: tuple[str, ...]


def make_import_branch(*, pack_name: str | None, run_id: str, batch: bool = False) -> str:
    suffix = run_id.removeprefix("mlxrun_")[:12]
    if not suffix or any(character not in "0123456789abcdef" for character in suffix):
        raise GitError("run ID cannot form a safe branch")
    if batch:
        return f"mojilex/batch/{suffix}"
    return f"mojilex/telegram/{branch_slug(pack_name or 'collection')}-{suffix}"


class GitPublisher:
    def __init__(self, runner: GitRunner) -> None:
        self.runner = runner

    def guard_targets(self, paths: tuple[str | PurePath, ...]) -> TargetGuard:
        """Call before dataset writes, then pass the result to ``prepare_commit``."""

        self.runner.ensure_no_overlapping_changes(paths)
        self.runner.ensure_index_clean()
        return TargetGuard(
            repository=self.runner.repository,
            base_sha=self.runner.current_sha(),
            paths=self.runner.normalize_paths(paths),
        )

    def prepare_commit(
        self,
        *,
        paths: tuple[str | PurePath, ...],
        message: str,
        branch: str,
        base_revision: str = "HEAD",
        identity: GitIdentity | None = None,
        guard: TargetGuard | None = None,
    ) -> PreparedCommit | None:
        if guard is None:
            # This legacy/pre-write form is intentionally strict. A pipeline which has already
            # written target files must use guard_targets() before those writes.
            self.runner.ensure_no_overlapping_changes(paths)
            base_sha = self.runner.current_sha(base_revision)
        else:
            normalized = self.runner.normalize_paths(paths)
            if (
                guard.repository != self.runner.repository
                or guard.paths != normalized
                or self.runner.current_sha(base_revision) != guard.base_sha
            ):
                raise GitError("target-path guard no longer matches repository, paths, or base SHA")
            base_sha = guard.base_sha
        self.runner.create_branch(branch, base_sha)
        staged = self.runner.stage_paths(paths)
        self.runner.ensure_only_targets_staged(paths)
        if not self.runner.has_staged_changes():
            return None
        resolved_identity = self.runner.resolve_identity(identity)
        commit_sha = self.runner.commit(message, identity=resolved_identity)
        return PreparedCommit(
            branch=branch,
            base_sha=base_sha,
            commit_sha=commit_sha,
            paths=staged,
            message=message,
            identity=resolved_identity,
        )

    def reconcile_remote_branch(
        self,
        prepared: PreparedCommit,
        *,
        remote: str,
        path_is_allowed: Callable[[str], bool],
    ) -> PreparedCommit:
        """Reuse a run branch by creating an ordinary fast-forward descendant.

        The resulting commit has the exact already-validated candidate tree.  If the
        base advanced since the earlier submission, it is included as a second parent;
        no history rewrite or force push is needed.
        """

        existing = self.runner.optional_remote_sha(remote, prepared.branch)
        if existing is None:
            return prepared
        self.runner.fetch(remote, prepared.branch)
        fetched = self.runner.current_sha(f"refs/remotes/{remote}/{prepared.branch}")
        if fetched != existing:
            raise GitError("remote run branch changed while it was being fetched")
        merge_base = self.runner.merge_base(prepared.base_sha, existing)
        existing_paths = self.runner.changed_paths_between(merge_base, existing)
        if any(not path_is_allowed(path) for path in existing_paths):
            raise GitError("existing run branch contains changes outside allowed target paths")

        base_is_ancestor = self.runner.is_ancestor(prepared.base_sha, existing)
        same_tree = self.runner.tree_sha(prepared.commit_sha) == self.runner.tree_sha(existing)
        if base_is_ancestor and same_tree:
            commit_sha = existing
        else:
            if prepared.message is None or prepared.identity is None:
                raise GitError("existing run branch cannot be updated without commit provenance")
            parents = (existing,) if base_is_ancestor else (existing, prepared.base_sha)
            commit_sha = self.runner.commit_tree_descendant(
                prepared.commit_sha,
                parents=parents,
                message=prepared.message,
                identity=prepared.identity,
            )
        if not self.runner.is_ancestor(existing, commit_sha):
            raise GitError("updated run branch is not a descendant of its remote commit")
        if not self.runner.is_ancestor(prepared.base_sha, commit_sha):
            raise GitError("updated run branch is not a descendant of the current base")
        if self.runner.tree_sha(commit_sha) != self.runner.tree_sha(prepared.commit_sha):
            raise GitError("updated run branch does not preserve the validated candidate tree")
        return PreparedCommit(
            branch=prepared.branch,
            base_sha=prepared.base_sha,
            commit_sha=commit_sha,
            paths=prepared.paths,
            message=prepared.message,
            identity=prepared.identity,
        )


def publication_mode(*, direct_push: bool, requested: Literal["local", "pr"]) -> str:
    if direct_push and requested == "local":
        raise GitError("direct push cannot be combined with local-only publication")
    return "direct" if direct_push else requested
