"""PR and fail-closed direct-push publication workflows."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

from mojilex_cli.git import GitRunner, PreparedCommit, validate_branch

from .client import (
    GitHubCLI,
    GitHubError,
    PullRequestResult,
    RepositoryInfo,
    RepositoryRef,
    RequiredCheck,
)


@dataclass(frozen=True)
class DirectPushAuthorization:
    explicit_flag: bool
    user_confirmed: bool
    validation_succeeded: bool
    bypass_verified: bool


PublicationPhase = Literal["candidate_pushed", "checks_passed", "completed"]
PublicationProgress: TypeAlias = Callable[[PublicationPhase], None]


class GitHubPublisher:
    def __init__(self, git: GitRunner, github: GitHubCLI) -> None:
        self.git = git
        self.github = github

    def publish_pr(
        self,
        prepared: PreparedCommit,
        *,
        target: RepositoryRef,
        fork: RepositoryRef,
        fork_remote: str,
        base_branch: str,
        title: str,
        body: str,
        progress: PublicationProgress | None = None,
    ) -> PullRequestResult:
        existing = self.github.reconcile_pull_request(
            repository=target,
            head_owner=fork.owner,
            head_branch=prepared.branch,
            base_branch=base_branch,
        )
        if existing is not None and existing.completed:
            _report_progress(progress, "completed")
            return existing
        self.git.push_commit(fork_remote, prepared.commit_sha, prepared.branch)
        _report_progress(progress, "candidate_pushed")
        if existing is not None:
            _report_progress(progress, "completed")
            return existing
        result = self.github.create_or_reuse_pull_request(
            repository=target,
            head_owner=fork.owner,
            head_branch=prepared.branch,
            base_branch=base_branch,
            title=title,
            body=body,
        )
        _report_progress(progress, "completed")
        return result

    async def publish_direct(
        self,
        prepared: PreparedCommit,
        *,
        target: RepositoryRef,
        repository_info: RepositoryInfo,
        remote: str,
        base_branch: str,
        candidate_branch: str,
        required_checks: Sequence[str | RequiredCheck],
        authorization: DirectPushAuthorization,
        progress: PublicationProgress | None = None,
    ) -> str:
        validate_branch(base_branch)
        validate_branch(candidate_branch)
        if not candidate_branch.startswith("mojilex/"):
            raise GitHubError("candidate branch must use the mojilex/** namespace")
        if not authorization.explicit_flag:
            raise GitHubError("direct push requires the explicit --direct-push flag")
        if not authorization.user_confirmed:
            raise GitHubError("direct push requires explicit confirmation")
        if not authorization.validation_succeeded:
            raise GitHubError("direct push requires a fresh successful validation")
        if not repository_info.can_write:
            raise GitHubError("direct push requires verified write permission")
        if not authorization.bypass_verified:
            raise GitHubError("direct push requires verified point bypass permission")
        if repository_info.name_with_owner.lower() != str(target).lower():
            raise GitHubError("permission metadata belongs to a different repository")
        if not required_checks:
            raise GitHubError("direct push requires an explicit non-empty required-check set")
        before = self.git.remote_sha(remote, base_branch)
        if before == prepared.commit_sha:
            _report_progress(progress, "completed")
            return prepared.commit_sha
        if before != prepared.base_sha:
            raise GitHubError("remote base changed before candidate publication")
        if not self.github.has_direct_push_bypass(target, base_branch):
            raise GitHubError("current GitHub actor has no point bypass on every active ruleset")
        candidate = self.git.optional_remote_sha(remote, candidate_branch)
        if candidate is None:
            self.git.push_commit(remote, prepared.commit_sha, candidate_branch)
            candidate = self.git.optional_remote_sha(remote, candidate_branch)
        if candidate != prepared.commit_sha:
            raise GitHubError("remote candidate branch differs from the prepared commit")
        _report_progress(progress, "candidate_pushed")
        await self.github.wait_for_checks(target, prepared.commit_sha, required_checks)
        refreshed_checks = self.github.required_checks(target, base_branch)
        if _required_check_signature(refreshed_checks) != _required_check_signature(
            required_checks
        ):
            raise GitHubError("required-check policy changed while candidate checks were running")
        if not self.github.has_direct_push_bypass(target, base_branch):
            raise GitHubError("point bypass changed while candidate checks were running")
        _report_progress(progress, "checks_passed")
        after_checks = self.git.remote_sha(remote, base_branch)
        if after_checks == prepared.commit_sha:
            _report_progress(progress, "completed")
            return prepared.commit_sha
        if after_checks != prepared.base_sha:
            raise GitHubError("remote base changed while candidate checks were running")
        # GitRunner emits an ordinary non-force refspec. GitHub rules still make the final decision.
        self.git.push_commit(remote, prepared.commit_sha, base_branch)
        published = self.git.remote_sha(remote, base_branch)
        if published != prepared.commit_sha:
            raise GitHubError("remote base does not match the published candidate")
        _report_progress(progress, "completed")
        return prepared.commit_sha


def _report_progress(callback: PublicationProgress | None, phase: PublicationPhase) -> None:
    if callback is not None:
        callback(phase)


def _required_check_signature(
    checks: Sequence[str | RequiredCheck],
) -> tuple[tuple[str, int | None], ...]:
    values = {
        (check.context, check.app_id) if isinstance(check, RequiredCheck) else (check, None)
        for check in checks
    }
    return tuple(sorted(values, key=lambda item: (item[0], item[1] or -1)))
