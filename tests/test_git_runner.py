import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MethodType

import pytest

from mojilex_cli.git import DirtyWorktreeError, GitError, GitIdentity, GitPublisher, GitRunner
from mojilex_cli.github import RepositoryRef
from mojilex_cli.pipeline.runner import _confirm_direct_push


def _repository(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test User"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"], check=True
    )
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "initial"], check=True, capture_output=True
    )
    return repo


def test_two_phase_guard_distinguishes_pipeline_changes_from_user_dirty_files(
    tmp_path: Path,
) -> None:
    repo = _repository(tmp_path)
    runner = GitRunner(repo)
    publisher = GitPublisher(runner)
    guard = publisher.guard_targets(("data/result.json",))
    (repo / "data").mkdir()
    (repo / "data" / "result.json").write_text("{}\n", encoding="utf-8")
    prepared = publisher.prepare_commit(
        paths=("data/result.json",),
        message="data(telegram): add synthetic pack",
        branch="mojilex/telegram/synthetic-123456789abc",
        guard=guard,
    )
    assert prepared is not None
    assert prepared.paths == ("data/result.json",)


def test_dirty_overlap_is_rejected_before_pipeline_write(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    (repo / "tracked.txt").write_text("user edit\n", encoding="utf-8")
    with pytest.raises(DirtyWorktreeError):
        GitPublisher(GitRunner(repo)).guard_targets(("tracked.txt",))


def test_git_runner_rejects_force_and_credential_remote(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    runner = GitRunner(repo)
    with pytest.raises(GitError, match="force"):
        runner.run("push", "--force", "origin", "main")
    subprocess.run(
        ["git", "-C", str(repo), "remote", "add", "origin", "https://token@github.com/x/y.git"],
        check=True,
    )
    with pytest.raises(GitError, match="credentials"):
        runner.remote_url()


def test_fetch_and_push_reject_non_github_and_separate_push_urls(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    runner = GitRunner(repo)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", repo.as_uri()], check=True)
    with pytest.raises(GitError, match=r"canonical github\.com"):
        runner.fetch("origin", "main")

    subprocess.run(
        ["git", "-C", str(repo), "remote", "set-url", "origin", "https://github.com/x/y.git"],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "remote",
            "set-url",
            "--push",
            "--add",
            "origin",
            repo.as_uri(),
        ],
        check=True,
    )
    with pytest.raises(GitError, match=r"canonical github\.com"):
        runner.push_commit("origin", runner.current_sha(), "main")


def test_git_identity_precedence_is_local_then_explicit_then_global(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    runner = GitRunner(repo)
    assert runner.resolve_identity(GitIdentity("Configured", "configured@example.invalid")) == (
        GitIdentity("Test User", "test@example.invalid")
    )

    subprocess.run(["git", "-C", str(repo), "config", "--unset", "user.name"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "--unset", "user.email"], check=True)
    assert runner.resolve_identity(GitIdentity("Configured", "configured@example.invalid")) == (
        GitIdentity("Configured", "configured@example.invalid")
    )


def test_second_submit_reuses_run_branch_with_fast_forward_descendant(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    bare = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", str(bare)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "push", "origin", "main"], check=True, capture_output=True
    )
    runner = GitRunner(repo)

    def allow_test_remote(self: GitRunner, remote: str, *, push: bool) -> tuple[str, ...]:
        del self, remote, push
        return (str(bare),)

    runner._validated_remote_urls = MethodType(allow_test_remote, runner)  # type: ignore[method-assign]
    publisher = GitPublisher(runner)
    target = "data/result.json"
    guard = publisher.guard_targets((target,))
    (repo / "data").mkdir()
    (repo / target).write_text('{"version":1}\n', encoding="utf-8")
    first = publisher.prepare_commit(
        paths=(target,),
        message="data: first submit",
        branch="mojilex/batch/123456789abc",
        guard=guard,
    )
    assert first is not None
    runner.push_commit("origin", first.commit_sha, first.branch)

    runner.run("switch", "main")
    guard = publisher.guard_targets((target,))
    (repo / "data").mkdir(exist_ok=True)
    (repo / target).write_text('{"version":2}\n', encoding="utf-8")
    second_local = publisher.prepare_commit(
        paths=(target,),
        message="data: second submit",
        branch="local-second-submit",
        guard=guard,
    )
    assert second_local is not None
    second = publisher.reconcile_remote_branch(
        replace(second_local, branch=first.branch),
        remote="origin",
        path_is_allowed=lambda path: path.startswith("data/"),
    )

    assert runner.is_ancestor(first.commit_sha, second.commit_sha)
    assert runner.is_ancestor(second.base_sha, second.commit_sha)
    assert runner.tree_sha(second.commit_sha) == runner.tree_sha(second_local.commit_sha)
    runner.push_commit("origin", second.commit_sha, second.branch)
    assert runner.remote_sha("origin", second.branch) == second.commit_sha


def test_direct_push_confirmation_contains_exact_commit_and_diff(tmp_path: Path) -> None:
    repo = _repository(tmp_path)
    runner = GitRunner(repo)
    publisher = GitPublisher(runner)
    guard = publisher.guard_targets(("data/result.json",))
    (repo / "data").mkdir()
    (repo / "data" / "result.json").write_text("{}\n", encoding="utf-8")
    prepared = publisher.prepare_commit(
        paths=("data/result.json",),
        message="data: exact candidate",
        branch="mojilex/batch/abcdef012345",
        guard=guard,
    )
    assert prepared is not None
    prompts: list[str] = []

    confirmed = _confirm_direct_push(
        lambda message: prompts.append(message) is None,
        runner,
        prepared,
        target=RepositoryRef.parse("MojiLex/mojilex"),
        base_branch="main",
    )

    assert confirmed
    assert prompts == [
        f"Publish exact validated commit {prepared.commit_sha} to MojiLex/mojilex:main? "
        "Candidate branch: mojilex/batch/abcdef012345. "
        "Exact changed paths: data/result.json."
    ]


def test_ephemeral_token_bridge_delivers_and_redacts_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repository(tmp_path)
    token = "github_pat_" + "SensitiveTokenValue1234567890"
    real_run = subprocess.run
    commands: list[tuple[str, ...]] = []
    credential_directories: list[Path] = []
    invocation = 0

    def fake_git(command, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal invocation
        command_tuple = tuple(str(value) for value in command)
        commands.append(command_tuple)
        assert token not in repr(command_tuple)
        environment = kwargs["env"]
        assert token not in environment.values()
        assert "GH_TOKEN" not in environment
        assert "GITHUB_TOKEN" not in environment
        if "remote" in command_tuple and "get-url" in command_tuple:
            assert "MOJILEX_GIT_ASKPASS_SECRET" not in environment
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"https://github.com/MojiLex/mojilex.git\n",
                stderr=b"",
            )
        invocation += 1
        launcher = Path(environment["GIT_ASKPASS"])
        helper = launcher.with_name("askpass.py")
        secret = Path(environment["MOJILEX_GIT_ASKPASS_SECRET"])
        credential_directories.append(launcher.parent)
        assert launcher.is_file()
        assert secret.is_file()
        password = real_run(
            [sys.executable, str(helper), "Password for 'https://github.com':"],
            env=environment,
            capture_output=True,
            check=True,
        ).stdout.decode()
        username = real_run(
            [sys.executable, str(helper), "Username for 'https://github.com':"],
            env=environment,
            capture_output=True,
            check=True,
        ).stdout.decode()
        assert password == token
        assert username == "x-access-token"
        return subprocess.CompletedProcess(
            command,
            0 if invocation == 1 else 1,
            stdout=f"stdout {token}".encode(),
            stderr=f"stderr {token}".encode(),
        )

    monkeypatch.setattr("mojilex_cli.git.runner.subprocess.run", fake_git)
    runner = GitRunner(repo, github_token=token)

    result = runner.run("fetch", "origin", "main")
    assert token not in result.stdout
    assert token not in result.stderr
    assert "<redacted>" in result.stdout
    assert "<redacted>" in result.stderr
    with pytest.raises(GitError) as captured:
        runner.run("push", "origin", "main")
    assert token not in str(captured.value)
    assert "<redacted>" in str(captured.value)
    assert all(token not in repr(command) for command in commands)
    assert credential_directories
    assert all(not directory.exists() for directory in credential_directories)


def test_github_token_does_not_override_ssh_askpass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repository(tmp_path)
    token = "github_pat_" + "SensitiveTokenValue1234567890"
    seen_fetch = False
    monkeypatch.setenv("GIT_ASKPASS", "user-configured-ssh-askpass")

    def fake_git(command, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal seen_fetch
        command_tuple = tuple(str(value) for value in command)
        environment = kwargs["env"]
        if "remote" in command_tuple and "get-url" in command_tuple:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b"git@github.com:MojiLex/mojilex.git\n",
                stderr=b"",
            )
        seen_fetch = True
        assert environment["GIT_ASKPASS"] == "user-configured-ssh-askpass"
        assert "MOJILEX_GIT_ASKPASS_SECRET" not in environment
        assert token not in environment.values()
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("mojilex_cli.git.runner.subprocess.run", fake_git)
    GitRunner(repo, github_token=token).run("fetch", "origin", "main")

    assert seen_fetch
