from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from mojilex_cli.commands import runtime
from mojilex_cli.git import PreparedCommit
from mojilex_cli.github import (
    DirectPushAuthorization,
    GitHubPublisher,
    PullRequestResult,
    RepositoryInfo,
    RepositoryRef,
    RequiredCheckError,
)
from mojilex_cli.i18n import use_ui_language


@pytest.fixture
def active_stages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    active: list[str] = []

    @contextmanager
    def track(label: str) -> Iterator[None]:
        active.append(label)
        try:
            yield
        finally:
            active.pop()

    monkeypatch.setattr(runtime, "operation_progress", track)
    return active


@pytest.mark.parametrize("existing_state", ["new", "open", "merged"])
def test_pr_network_calls_have_live_stage_and_preserve_reuse(
    active_stages: list[str], existing_state: str
) -> None:
    git, github = Mock(), Mock()
    result = PullRequestResult(
        "https://github.com/MojiLex/mojilex/pull/1",
        1,
        reused=existing_state != "new",
        completed=existing_state == "merged",
    )

    def reconcile(**_: object) -> PullRequestResult | None:
        assert active_stages == ["Checking for an existing pull request"]
        return None if existing_state == "new" else result

    def push(*_: object) -> None:
        assert active_stages == ["Uploading data to GitHub"]

    def create(**_: object) -> PullRequestResult:
        assert active_stages == ["Creating the pull request"]
        return result

    github.reconcile_pull_request.side_effect = reconcile
    github.create_or_reuse_pull_request.side_effect = create
    git.push_commit.side_effect = push
    phases: list[str] = []
    with use_ui_language("en"):
        actual = GitHubPublisher(git, github).publish_pr(
            PreparedCommit("mojilex/batch/example", "a" * 40, "b" * 40, ("data/x.json",)),
            target=RepositoryRef.parse("MojiLex/mojilex"),
            fork=RepositoryRef.parse("MojiLex/mojilex"),
            fork_remote="origin",
            base_branch="main",
            title="data: test",
            body="Test metadata",
            progress=phases.append,
        )
    assert actual == result
    assert active_stages == []
    assert git.push_commit.call_count == (0 if existing_state == "merged" else 1)
    assert github.create_or_reuse_pull_request.call_count == (1 if existing_state == "new" else 0)
    assert phases == (
        ["completed"] if existing_state == "merged" else ["candidate_pushed", "completed"]
    )


def test_failed_push_closes_stage_and_does_not_claim_publication(
    active_stages: list[str],
) -> None:
    git, github = Mock(), Mock()
    github.reconcile_pull_request.return_value = None

    def push(*_: object) -> None:
        assert active_stages == ["Uploading data to GitHub"]
        raise ConnectionError("disconnected")

    git.push_commit.side_effect = push
    phases: list[str] = []
    with use_ui_language("en"), pytest.raises(ConnectionError, match="disconnected"):
        GitHubPublisher(git, github).publish_pr(
            PreparedCommit("mojilex/batch/example", "a" * 40, "b" * 40, ("data/x.json",)),
            target=RepositoryRef.parse("MojiLex/mojilex"),
            fork=RepositoryRef.parse("MojiLex/mojilex"),
            fork_remote="origin",
            base_branch="main",
            title="data: test",
            body="Test metadata",
            progress=phases.append,
        )
    assert not active_stages
    assert phases == []
    github.create_or_reuse_pull_request.assert_not_called()


@pytest.mark.parametrize("checks_fail", [False, True])
async def test_direct_push_wait_has_live_stage_and_failure_never_updates_base(
    active_stages: list[str], checks_fail: bool
) -> None:
    git, github = Mock(), Mock()
    git.remote_sha.side_effect = ["a" * 40, "a" * 40, "b" * 40]
    git.optional_remote_sha.return_value = "b" * 40
    github.has_direct_push_bypass.return_value = True
    github.required_checks.return_value = ("validate",)

    async def wait(*_: object) -> None:
        assert active_stages == ["Waiting for required GitHub checks"]
        if checks_fail:
            raise RequiredCheckError("candidate checks failed")

    def push(*_: object) -> None:
        assert active_stages == ["Publishing verified data"]

    github.wait_for_checks = wait
    git.push_commit.side_effect = push
    phases: list[str] = []
    with use_ui_language("en"):
        pending = GitHubPublisher(git, github).publish_direct(
            PreparedCommit("mojilex/batch/example", "a" * 40, "b" * 40, ("data/x.json",)),
            target=RepositoryRef.parse("MojiLex/mojilex"),
            repository_info=RepositoryInfo(
                name_with_owner="MojiLex/mojilex", permission="ADMIN", default_branch="main"
            ),
            remote="origin",
            base_branch="main",
            candidate_branch="mojilex/batch/example",
            required_checks=("validate",),
            authorization=DirectPushAuthorization(True, True, True, True),
            progress=phases.append,
        )
        if checks_fail:
            with pytest.raises(RequiredCheckError, match="candidate checks failed"):
                await pending
        else:
            assert await pending == "b" * 40
    assert not active_stages
    if checks_fail:
        git.push_commit.assert_not_called()
        assert phases == ["candidate_pushed"]
    else:
        git.push_commit.assert_called_once_with("origin", "b" * 40, "main")
        assert phases == ["candidate_pushed", "checks_passed", "completed"]


@pytest.mark.parametrize("returncode", [0, 1])
def test_repository_clone_has_live_stage_and_cleans_up_on_failure(
    monkeypatch: pytest.MonkeyPatch, returncode: int
) -> None:
    from types import SimpleNamespace

    from mojilex_cli.pipeline import runner

    active: list[str] = []

    @contextmanager
    def track(label: str) -> Iterator[None]:
        active.append(label)
        try:
            yield
        finally:
            active.pop()

    def clone(arguments: list[str], **_: object) -> SimpleNamespace:
        assert active == ["Downloading the repository from GitHub"]
        assert arguments[:2] == ["git", "clone"]
        return SimpleNamespace(returncode=returncode)

    @contextmanager
    def environment(_: str) -> Iterator[dict[str, str]]:
        yield {}

    monkeypatch.setattr(runner, "operation_progress", track)
    monkeypatch.setattr(runner.subprocess, "run", clone)
    monkeypatch.setattr(runner, "git_subprocess_environment", environment)
    with use_ui_language("en"):
        if returncode:
            with pytest.raises(runtime.CommandError, match="isolated checkout"):
                with runner.repository_workspace("MojiLex/mojilex", "main", github_token="test"):
                    pytest.fail("a failed clone must not yield a workspace")
        else:
            with runner.repository_workspace("MojiLex/mojilex", "main", github_token="test") as ws:
                assert ws.temporary
                assert not active  # Clone stage must not cover the caller's whole workflow.
    assert not active
