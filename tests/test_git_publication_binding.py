from unittest.mock import Mock

import pytest

from mojilex_cli.git import GitError, GitRunner, PreparedCommit
from mojilex_cli.github import (
    DirectPushAuthorization,
    GitHubPublisher,
    RepositoryInfo,
    RepositoryRef,
)
from test_git_runner import _repository


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/Example/Expected.git",
        "https://github.com/example/expected",
        "git@github.com:EXAMPLE/EXPECTED.git",
        "ssh://git@github.com/example/expected.git",
    ],
)
def test_publication_repository_binding_accepts_equivalent_transports(tmp_path, url):
    git = GitRunner(_repository(tmp_path))
    git.run("remote", "add", "origin", url)
    git.validate_remote_repository("origin", "example/expected")


@pytest.mark.parametrize("mismatch", ["push", "extra_push", "fetch", "rewrite", "stale_fork"])
def test_publication_repository_binding_rejects_every_wrong_destination(tmp_path, mismatch):
    git = GitRunner(_repository(tmp_path))
    remote = "mojilex-fork" if mismatch == "stale_fork" else "origin"
    expected = "https://github.com/example/expected.git"
    other = "https://github.com/example/different.git"
    git.run("remote", "add", remote, expected)
    if mismatch == "extra_push":
        git.run("remote", "set-url", "--add", "--push", remote, expected)
        git.run("remote", "set-url", "--add", "--push", remote, other)
    elif mismatch == "rewrite":
        git.run("config", f"url.{other}.pushInsteadOf", expected)
    elif mismatch in {"fetch", "stale_fork"}:
        git.run("remote", "set-url", remote, other)
        git.run("remote", "set-url", "--push", remote, expected)
    else:
        git.run("remote", "set-url", "--push", remote, other)
    with pytest.raises(GitError, match="authorized publication repository"):
        git.validate_remote_repository(remote, "example/expected")


@pytest.mark.parametrize("change_during_reconciliation", [False, True])
def test_pr_rejects_wrong_push_repository(tmp_path, monkeypatch, change_during_reconciliation):
    git = GitRunner(_repository(tmp_path))
    git.run("remote", "add", "origin", "https://github.com/example/expected.git")
    github = Mock()
    github.reconcile_pull_request.return_value = None
    pushes = []
    monkeypatch.setattr(git, "push_commit", lambda *args: pushes.append(args))

    def change_remote(**kwargs):
        git.run("remote", "set-url", "--push", "origin", "https://github.com/example/different.git")
        return None

    if change_during_reconciliation:
        github.reconcile_pull_request.side_effect = change_remote
    else:
        change_remote()
    with pytest.raises(GitError, match="authorized publication repository"):
        GitHubPublisher(git, github).publish_pr(
            PreparedCommit("mojilex/test", "a" * 40, "b" * 40, ("data/x.json",)),
            target=RepositoryRef.parse("example/expected"),
            fork=RepositoryRef.parse("example/expected"),
            fork_remote="origin",
            base_branch="main",
            title="Synthetic",
            body="Synthetic",
        )
    github.create_or_reuse_pull_request.assert_not_called()
    assert pushes == []
    assert github.reconcile_pull_request.call_count == int(change_during_reconciliation)


@pytest.mark.parametrize("change_during_checks", [False, True])
async def test_direct_push_rejects_wrong_repository(tmp_path, monkeypatch, change_during_checks):
    git = GitRunner(_repository(tmp_path))
    git.run("remote", "add", "origin", "https://github.com/example/expected.git")
    pushes = []
    candidate = "b" * 40
    monkeypatch.setattr(git, "remote_sha", lambda *_: "a" * 40)
    monkeypatch.setattr(git, "optional_remote_sha", lambda *_: candidate if pushes else None)
    monkeypatch.setattr(git, "push_commit", lambda *args: pushes.append(args))
    github = Mock()
    github.has_direct_push_bypass.return_value = True
    github.required_checks.return_value = ("validate",)

    async def checks(*args):
        git.run("remote", "set-url", "--push", "origin", "https://github.com/example/different.git")

    github.wait_for_checks = checks
    if not change_during_checks:
        await checks()
    with pytest.raises(GitError, match="authorized publication repository"):
        await GitHubPublisher(git, github).publish_direct(
            PreparedCommit("mojilex/test", "a" * 40, candidate, ("data/x.json",)),
            target=RepositoryRef.parse("example/expected"),
            repository_info=RepositoryInfo(
                name_with_owner="example/expected", permission="ADMIN", default_branch="main"
            ),
            remote="origin",
            base_branch="main",
            candidate_branch="mojilex/test",
            required_checks=("validate",),
            authorization=DirectPushAuthorization(True, True, True, True),
        )
    assert pushes == ([("origin", candidate, "mojilex/test")] if change_during_checks else [])
