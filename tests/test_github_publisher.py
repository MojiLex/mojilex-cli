import json
from types import MethodType

import pytest

from mojilex_cli.git import PreparedCommit
from mojilex_cli.github import (
    DirectPushAuthorization,
    GitHubCLI,
    GitHubError,
    GitHubPublisher,
    PullRequestResult,
    RepositoryInfo,
    RepositoryRef,
    RequiredCheck,
    RequiredCheckError,
)


def _pr_record(
    *,
    state: str,
    base: str = "main",
    merged_at: str | None = None,
) -> dict[str, object]:
    return {
        "number": 17,
        "url": "https://github.com/MojiLex/mojilex/pull/17",
        "state": state,
        "baseRefName": base,
        "headRefName": "mojilex/batch/123456789abc",
        "headRepositoryOwner": {"login": "MojiLex"},
        "mergedAt": merged_at,
    }


def test_retargeted_open_run_pr_fails_closed() -> None:
    github = GitHubCLI()

    def fake_run(self: GitHubCLI, *arguments: str, **_: object) -> str:
        assert arguments[0:2] == ("pr", "list")
        assert arguments[arguments.index("--state") + 1] == "all"
        return json.dumps([_pr_record(state="OPEN", base="release")])

    github.run = MethodType(fake_run, github)
    with pytest.raises(GitHubError, match="different base"):
        github.reconcile_pull_request(
            repository=RepositoryRef.parse("MojiLex/mojilex"),
            head_owner="MojiLex",
            head_branch="mojilex/batch/123456789abc",
            base_branch="main",
        )


def test_closed_unmerged_run_pr_is_reopened_and_verified() -> None:
    github = GitHubCLI()
    calls: list[tuple[str, ...]] = []

    def fake_run(self: GitHubCLI, *arguments: str, **_: object) -> str:
        calls.append(arguments)
        if arguments[0:2] == ("pr", "list"):
            return json.dumps([_pr_record(state="CLOSED")])
        if arguments[0:2] == ("pr", "reopen"):
            return "Reopened"
        if arguments[0:2] == ("pr", "view"):
            return json.dumps(_pr_record(state="OPEN"))
        raise AssertionError(arguments)

    github.run = MethodType(fake_run, github)
    result = github.reconcile_pull_request(
        repository=RepositoryRef.parse("MojiLex/mojilex"),
        head_owner="MojiLex",
        head_branch="mojilex/batch/123456789abc",
        base_branch="main",
    )

    assert result == PullRequestResult(
        url="https://github.com/MojiLex/mojilex/pull/17",
        number=17,
        reused=True,
        reopened=True,
    )
    assert any(call[0:2] == ("pr", "reopen") for call in calls)
    assert any(call[0:2] == ("pr", "view") for call in calls)


def test_merged_run_pr_is_completed_without_duplicate() -> None:
    github = GitHubCLI()

    def fake_run(self: GitHubCLI, *arguments: str, **_: object) -> str:
        assert arguments[0:2] == ("pr", "list")
        return json.dumps([_pr_record(state="MERGED", merged_at="2026-09-11T00:00:00Z")])

    github.run = MethodType(fake_run, github)
    result = github.reconcile_pull_request(
        repository=RepositoryRef.parse("MojiLex/mojilex"),
        head_owner="MojiLex",
        head_branch="mojilex/batch/123456789abc",
        base_branch="main",
    )

    assert result is not None
    assert result.completed
    assert result.reused


def test_merged_run_pr_does_not_push_branch_again() -> None:
    class FakeGit:
        def __init__(self) -> None:
            self.pushes: list[tuple[str, str, str]] = []

        def validate_remote_repository(self, remote: str, expected_repository: str) -> None:
            assert (remote, expected_repository) == ("origin", "MojiLex/mojilex")

        def push_commit(self, remote: str, sha: str, branch: str) -> None:
            self.pushes.append((remote, sha, branch))

    class FakeGitHub:
        def reconcile_pull_request(self, **_: object) -> PullRequestResult:
            return PullRequestResult(
                url="https://github.com/MojiLex/mojilex/pull/17",
                number=17,
                reused=True,
                completed=True,
            )

        def create_or_reuse_pull_request(self, **_: object) -> PullRequestResult:
            raise AssertionError("a merged run must not create another Pull Request")

    git = FakeGit()
    publisher = GitHubPublisher(git, FakeGitHub())  # type: ignore[arg-type]
    result = publisher.publish_pr(
        PreparedCommit(
            branch="mojilex/batch/123456789abc",
            base_sha="a" * 40,
            commit_sha="b" * 40,
            paths=("data/example.json",),
        ),
        target=RepositoryRef.parse("MojiLex/mojilex"),
        fork=RepositoryRef.parse("MojiLex/mojilex"),
        fork_remote="origin",
        base_branch="main",
        title="data: exact",
        body="Validated.",
    )

    assert result.completed
    assert git.pushes == []


def test_repository_reference_rejects_credentials_and_extra_paths() -> None:
    assert str(RepositoryRef.parse("MojiLex/mojilex")) == "MojiLex/mojilex"
    assert str(RepositoryRef.parse("https://github.com/MojiLex/mojilex.git")) == "MojiLex/mojilex"
    with pytest.raises(GitHubError):
        RepositoryRef.parse("https://token@github.com/MojiLex/mojilex")
    with pytest.raises(GitHubError):
        RepositoryRef.parse("https://github.com/MojiLex/mojilex/extra")


def test_point_bypass_requires_exact_user_on_every_active_ruleset() -> None:
    github = GitHubCLI(token="secret")
    responses = {
        "user": {"login": "owner", "id": 42},
        "repos/MojiLex/mojilex/rulesets?includes_parents=true&per_page=100&page=1": [
            {"id": 1},
            {"id": 2},
        ],
        "repos/MojiLex/mojilex/rulesets/1": {
            "enforcement": "active",
            "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
            "bypass_actors": [{"actor_type": "User", "actor_id": 42, "bypass_mode": "always"}],
        },
        "repos/MojiLex/mojilex/rulesets/2": {
            "enforcement": "active",
            "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
            "bypass_actors": [
                {"actor_type": "RepositoryRole", "actor_id": 5, "bypass_mode": "always"}
            ],
        },
    }

    def fake_api(self: GitHubCLI, endpoint: str, **_: object) -> object:
        return responses[endpoint]

    github.api = MethodType(fake_api, github)
    assert not github.has_direct_push_bypass(RepositoryRef.parse("MojiLex/mojilex"), "main")
    responses["repos/MojiLex/mojilex/rulesets/2"]["bypass_actors"] = [
        {"actor_type": "User", "actor_id": 42, "bypass_mode": "exempt"}
    ]
    assert github.has_direct_push_bypass(RepositoryRef.parse("MojiLex/mojilex"), "main")


def test_rulesets_and_commit_checks_are_paginated_and_preserve_app_identity() -> None:
    github = GitHubCLI()
    repository = RepositoryRef.parse("MojiLex/mojilex")
    first_page = [{"id": number} for number in range(1, 101)]

    def fake_api(self: GitHubCLI, endpoint: str, **_: object) -> object:
        if endpoint.endswith("protection/required_status_checks"):
            return {
                "contexts": ["build"],
                "checks": [{"context": "build", "app_id": 7}],
            }
        if endpoint.endswith("rulesets?includes_parents=true&per_page=100&page=1"):
            return first_page
        if endpoint.endswith("rulesets?includes_parents=true&per_page=100&page=2"):
            return [{"id": 101}]
        if "/rulesets/" in endpoint:
            identifier = int(endpoint.rsplit("/", 1)[1])
            if identifier == 101:
                return {
                    "enforcement": "active",
                    "rules": [
                        {
                            "type": "required_status_checks",
                            "parameters": {
                                "required_status_checks": [{"context": "lint", "integration_id": 9}]
                            },
                        }
                    ],
                }
            return {"enforcement": "disabled"}
        if "check-runs" in endpoint and endpoint.endswith("page=1"):
            return {
                "check_runs": [
                    {
                        "name": f"irrelevant-{number}",
                        "status": "completed",
                        "conclusion": "success",
                        "app": {"id": 1},
                    }
                    for number in range(100)
                ]
            }
        if "check-runs" in endpoint and endpoint.endswith("page=2"):
            return {
                "check_runs": [
                    {
                        "name": "build",
                        "status": "completed",
                        "conclusion": "success",
                        "app": {"id": 7},
                    },
                    {
                        "name": "lint",
                        "status": "completed",
                        "conclusion": "success",
                        "app": {"id": 9},
                    },
                ]
            }
        if "/status?" in endpoint:
            return {
                "statuses": [
                    {"context": "build", "state": "failure"},
                    {"context": "lint", "state": "failure"},
                ]
            }
        raise AssertionError(endpoint)

    github.api = MethodType(fake_api, github)
    required = github.required_checks(repository, "main")
    assert required == (
        RequiredCheck(context="build", app_id=7),
        RequiredCheck(context="lint", app_id=9),
    )
    result = github.commit_checks(repository, "a" * 40, required)
    assert result.successful == ("build (app 7)", "lint (app 9)")
    assert result.complete_and_successful


def test_ruleset_pathname_glob_does_not_cross_slashes() -> None:
    github = GitHubCLI()

    def fake_api(self: GitHubCLI, endpoint: str, **_: object) -> object:
        if endpoint.endswith("protection/required_status_checks"):
            return {"contexts": ["base"], "checks": []}
        if endpoint.endswith("rulesets?includes_parents=true&per_page=100&page=1"):
            return [{"id": 1}]
        if endpoint.endswith("rulesets/1"):
            return {
                "enforcement": "active",
                "conditions": {"ref_name": {"include": ["refs/heads/*"], "exclude": []}},
                "rules": [{"type": "workflows"}],
            }
        raise AssertionError(endpoint)

    github.api = MethodType(fake_api, github)
    assert github.required_checks(RepositoryRef.parse("MojiLex/mojilex"), "release/hotfix") == (
        RequiredCheck(context="base"),
    )


@pytest.mark.parametrize("conclusion", ["neutral", "skipped", "failure", "cancelled"])
def test_required_check_accepts_only_explicit_success(conclusion: str) -> None:
    github = GitHubCLI()

    def fake_api(self: GitHubCLI, endpoint: str, **_: object) -> object:
        if "check-runs" in endpoint:
            return {
                "check_runs": [
                    {
                        "name": "validate",
                        "status": "completed",
                        "conclusion": conclusion,
                        "app": {"id": 7},
                    }
                ]
            }
        if "/status?" in endpoint:
            return {"statuses": []}
        raise AssertionError(endpoint)

    github.api = MethodType(fake_api, github)
    result = github.commit_checks(
        RepositoryRef.parse("MojiLex/mojilex"),
        "a" * 40,
        (RequiredCheck(context="validate", app_id=7),),
    )

    assert not result.complete_and_successful
    assert result.failed == ("validate (app 7)",)


def test_direct_push_rejects_rules_not_provable_from_commit_statuses() -> None:
    github = GitHubCLI()

    def fake_api(self: GitHubCLI, endpoint: str, **_: object) -> object:
        if endpoint.endswith("protection/required_status_checks"):
            return {"contexts": ["base"], "checks": []}
        if endpoint.endswith("rulesets?includes_parents=true&per_page=100&page=1"):
            return [{"id": 1}]
        if endpoint.endswith("rulesets/1"):
            return {
                "enforcement": "active",
                "rules": [{"type": "workflows"}],
            }
        raise AssertionError(endpoint)

    github.api = MethodType(fake_api, github)
    with pytest.raises(RequiredCheckError, match="cannot be proven"):
        github.required_checks(RepositoryRef.parse("MojiLex/mojilex"), "main")


class _FakeGit:
    def __init__(self) -> None:
        self.base = "a" * 40
        self.branches: dict[str, str] = {}
        self.pushes: list[tuple[str, str, str]] = []

    def validate_remote_repository(self, remote: str, expected_repository: str) -> None:
        assert (remote, expected_repository) == ("origin", "MojiLex/mojilex")

    def remote_sha(self, remote: str, branch: str) -> str:
        if branch != "main":
            return self.branches[branch]
        return self.base

    def optional_remote_sha(self, remote: str, branch: str) -> str | None:
        if branch == "main":
            return self.base
        return self.branches.get(branch)

    def push_commit(self, remote: str, sha: str, branch: str) -> None:
        self.pushes.append((remote, sha, branch))
        if branch == "main":
            self.base = sha
        else:
            self.branches[branch] = sha


class _FakeGitHub:
    def __init__(
        self,
        *,
        bypass: bool,
        refreshed: tuple[str | RequiredCheck, ...] = ("validate",),
    ) -> None:
        self.bypass_values = [bypass, bypass]
        self.refreshed = refreshed
        self.waited = False

    def has_direct_push_bypass(self, repository: RepositoryRef, branch: str) -> bool:
        return self.bypass_values.pop(0)

    def required_checks(
        self, repository: RepositoryRef, branch: str
    ) -> tuple[str | RequiredCheck, ...]:
        return self.refreshed

    async def wait_for_checks(self, *args: object, **kwargs: object) -> None:
        self.waited = True


@pytest.mark.asyncio
async def test_direct_push_uses_candidate_checks_then_fast_forward() -> None:
    git = _FakeGit()
    github = _FakeGitHub(bypass=True)
    publisher = GitHubPublisher(git, github)
    prepared = PreparedCommit(
        branch="mojilex/telegram/pack-123456789abc",
        base_sha="a" * 40,
        commit_sha="b" * 40,
        paths=("data/x.json",),
    )
    await publisher.publish_direct(
        prepared,
        target=RepositoryRef.parse("MojiLex/mojilex"),
        repository_info=RepositoryInfo(
            name_with_owner="MojiLex/mojilex", permission="ADMIN", default_branch="main"
        ),
        remote="origin",
        base_branch="main",
        candidate_branch="mojilex/candidate/123",
        required_checks=("validate",),
        authorization=DirectPushAuthorization(True, True, True, True),
    )
    assert github.waited
    assert git.pushes == [
        ("origin", "b" * 40, "mojilex/candidate/123"),
        ("origin", "b" * 40, "main"),
    ]


@pytest.mark.asyncio
async def test_direct_push_rereads_policy_and_stops_if_required_set_changes() -> None:
    git = _FakeGit()
    publisher = GitHubPublisher(
        git, _FakeGitHub(bypass=True, refreshed=(RequiredCheck(context="new-check"),))
    )
    with pytest.raises(GitHubError, match="policy changed"):
        await publisher.publish_direct(
            PreparedCommit("branch", "a" * 40, "b" * 40, ("x",)),
            target=RepositoryRef.parse("MojiLex/mojilex"),
            repository_info=RepositoryInfo(
                name_with_owner="MojiLex/mojilex", permission="ADMIN", default_branch="main"
            ),
            remote="origin",
            base_branch="main",
            candidate_branch="mojilex/candidate/123",
            required_checks=("validate",),
            authorization=DirectPushAuthorization(True, True, True, True),
        )
    assert git.pushes == [("origin", "b" * 40, "mojilex/candidate/123")]


@pytest.mark.asyncio
async def test_direct_push_rereads_bypass_after_candidate_ci() -> None:
    git = _FakeGit()
    github = _FakeGitHub(bypass=True)
    github.bypass_values = [True, False]
    publisher = GitHubPublisher(git, github)
    with pytest.raises(GitHubError, match="bypass changed"):
        await publisher.publish_direct(
            PreparedCommit("branch", "a" * 40, "b" * 40, ("x",)),
            target=RepositoryRef.parse("MojiLex/mojilex"),
            repository_info=RepositoryInfo(
                name_with_owner="MojiLex/mojilex", permission="ADMIN", default_branch="main"
            ),
            remote="origin",
            base_branch="main",
            candidate_branch="mojilex/candidate/123",
            required_checks=("validate",),
            authorization=DirectPushAuthorization(True, True, True, True),
        )
    assert git.pushes == [("origin", "b" * 40, "mojilex/candidate/123")]


@pytest.mark.asyncio
async def test_direct_push_refuses_admin_without_point_bypass() -> None:
    git = _FakeGit()
    publisher = GitHubPublisher(git, _FakeGitHub(bypass=False))
    with pytest.raises(GitHubError, match="point bypass"):
        await publisher.publish_direct(
            PreparedCommit("branch", "a" * 40, "b" * 40, ("x",)),
            target=RepositoryRef.parse("MojiLex/mojilex"),
            repository_info=RepositoryInfo(
                name_with_owner="MojiLex/mojilex", permission="ADMIN", default_branch="main"
            ),
            remote="origin",
            base_branch="main",
            candidate_branch="mojilex/candidate/123",
            required_checks=("validate",),
            authorization=DirectPushAuthorization(True, True, True, True),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_branch", ["mojilex/candidate/123", "main"])
async def test_direct_push_recovers_when_remote_accepts_push_then_client_raises(
    failed_branch: str,
) -> None:
    class AcceptedThenRaisedGit(_FakeGit):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def push_commit(self, remote: str, sha: str, branch: str) -> None:
            super().push_commit(remote, sha, branch)
            if branch == failed_branch and not self.failed:
                self.failed = True
                raise RuntimeError("connection dropped after server accepted the push")

    class AlwaysAuthorizedGitHub(_FakeGitHub):
        def has_direct_push_bypass(self, repository: RepositoryRef, branch: str) -> bool:
            return True

    git = AcceptedThenRaisedGit()
    github = AlwaysAuthorizedGitHub(bypass=True)
    publisher = GitHubPublisher(git, github)
    prepared = PreparedCommit("branch", "a" * 40, "b" * 40, ("x",))
    kwargs = {
        "target": RepositoryRef.parse("MojiLex/mojilex"),
        "repository_info": RepositoryInfo(
            name_with_owner="MojiLex/mojilex", permission="ADMIN", default_branch="main"
        ),
        "remote": "origin",
        "base_branch": "main",
        "candidate_branch": "mojilex/candidate/123",
        "required_checks": ("validate",),
        "authorization": DirectPushAuthorization(True, True, True, True),
    }

    with pytest.raises(RuntimeError, match="connection dropped"):
        await publisher.publish_direct(prepared, **kwargs)  # type: ignore[arg-type]

    phases: list[str] = []
    result = await publisher.publish_direct(
        prepared,
        progress=phases.append,  # type: ignore[arg-type]
        **kwargs,  # type: ignore[arg-type]
    )

    assert result == "b" * 40
    assert git.base == "b" * 40
    assert git.branches["mojilex/candidate/123"] == "b" * 40
    assert [branch for _, _, branch in git.pushes].count(failed_branch) == 1
    assert phases[-1] == "completed"


@pytest.mark.asyncio
async def test_direct_push_fails_closed_for_unexpected_candidate_ref() -> None:
    git = _FakeGit()
    git.branches["mojilex/candidate/123"] = "c" * 40
    publisher = GitHubPublisher(git, _FakeGitHub(bypass=True))

    with pytest.raises(GitHubError, match="candidate branch differs"):
        await publisher.publish_direct(
            PreparedCommit("branch", "a" * 40, "b" * 40, ("x",)),
            target=RepositoryRef.parse("MojiLex/mojilex"),
            repository_info=RepositoryInfo(
                name_with_owner="MojiLex/mojilex", permission="ADMIN", default_branch="main"
            ),
            remote="origin",
            base_branch="main",
            candidate_branch="mojilex/candidate/123",
            required_checks=("validate",),
            authorization=DirectPushAuthorization(True, True, True, True),
        )

    assert git.pushes == []
