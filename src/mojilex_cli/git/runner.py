"""Shell-free, bounded Git primitives that never stage unrelated paths."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePath, PurePosixPath
from urllib.parse import urlsplit

from mojilex_cli.config.secrets import redact_text, url_has_credentials

_OBJECT_ID = re.compile(r"[0-9a-f]{40,64}\Z")
_SAFE_REMOTE = re.compile(r"[A-Za-z0-9._-]{1,100}\Z")
_GITHUB_OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,98}[A-Za-z0-9])?\Z")
_GITHUB_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")
_GITHUB_SCP_REMOTE = re.compile(
    r"git@github\.com:(?P<owner>[A-Za-z0-9][A-Za-z0-9-]{0,99})/"
    r"(?P<repository>[A-Za-z0-9_.-]{1,100})(?:\.git)?\Z"
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_BANNED_ARGUMENTS = frozenset({"--force", "-f", "--force-with-lease", "--mirror"})
_REMOTE_AUTH_COMMANDS = frozenset({"fetch", "ls-remote", "push"})
_REMOTE_OPTIONS_WITHOUT_VALUES = frozenset(
    {
        "--atomic",
        "--dry-run",
        "--exit-code",
        "--no-tags",
        "--porcelain",
        "--prune",
        "--quiet",
        "--tags",
        "--verbose",
        "-q",
        "-v",
    }
)
_ASKPASS_SECRET_ENV = "MOJILEX_GIT_ASKPASS_SECRET"
_ASKPASS_SOURCE = """from __future__ import annotations

import os
import sys
from pathlib import Path

prompt = " ".join(sys.argv[1:]).casefold()
if "username" in prompt:
    sys.stdout.write("x-access-token")
elif "password" in prompt:
    secret_path = os.environ.get("MOJILEX_GIT_ASKPASS_SECRET")
    if not secret_path:
        raise SystemExit(1)
    secret = Path(secret_path).read_text(encoding="utf-8")
    if not secret or any(ord(character) < 32 or ord(character) == 127 for character in secret):
        raise SystemExit(1)
    sys.stdout.write(secret)
else:
    raise SystemExit(1)
"""


class GitError(RuntimeError):
    code = "GIT_CONFLICT"


class DirtyWorktreeError(GitError):
    code = "DIRTY_WORKTREE"


class GitIdentityError(GitError):
    code = "CONFIG_MISSING"


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    returncode: int


@dataclass(frozen=True)
class GitIdentity:
    name: str
    email: str


class GitRunner:
    def __init__(
        self,
        repository: Path,
        *,
        executable: str = "git",
        timeout_seconds: float = 60,
        github_token: str | None = None,
    ) -> None:
        self.repository = repository.resolve(strict=True)
        if not (self.repository / ".git").exists():
            # Worktrees use a .git file, so existence is intentional rather than is_dir.
            raise GitError("target path is not a Git working tree")
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self._github_token = _validate_github_token(github_token)
        self._secrets = (self._github_token,) if self._github_token is not None else ()

    def run(
        self,
        *arguments: str,
        check: bool = True,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        self._validate_arguments(arguments)
        command = [self.executable, "-C", str(self.repository), *arguments]
        github_token = self._github_token_for(arguments)
        try:
            with git_subprocess_environment(github_token) as environment:
                completed = subprocess.run(
                    command,
                    shell=False,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    check=False,
                    timeout=timeout_seconds or self.timeout_seconds,
                    env=environment,
                )
        except FileNotFoundError as exc:
            raise GitError("system Git executable was not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitError("Git command exceeded its time limit") from exc
        stdout = redact_text(completed.stdout.decode("utf-8", errors="replace"), self._secrets)
        stderr = redact_text(completed.stderr.decode("utf-8", errors="replace"), self._secrets)
        if len(stdout) + len(stderr) > 8 * 1024 * 1024:
            raise GitError("Git output exceeded the safe diagnostic limit")
        result = CommandResult(stdout=stdout, stderr=stderr, returncode=completed.returncode)
        if check and completed.returncode != 0:
            safe = redact_text(stderr or stdout, self._secrets).strip()[:2000]
            raise GitError(f"Git command failed ({arguments[0] if arguments else 'git'}): {safe}")
        return result

    def current_sha(self, revision: str = "HEAD") -> str:
        result = self.run("rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}")
        value = result.stdout.strip()
        if not _OBJECT_ID.fullmatch(value):
            raise GitError("Git returned an invalid object ID")
        return value

    def remote_sha(self, remote: str, branch: str) -> str:
        value = self.optional_remote_sha(remote, branch)
        if value is None:
            raise GitError("remote branch did not resolve to one commit")
        return value

    def optional_remote_sha(self, remote: str, branch: str) -> str | None:
        """Resolve a remote branch without treating an absent branch as an error."""

        self._validate_remote(remote)
        self._validated_remote_urls(remote, push=False)
        validate_branch(branch)
        result = self.run("ls-remote", "--exit-code", remote, f"refs/heads/{branch}", check=False)
        if result.returncode == 2 and not result.stdout.strip():
            return None
        if result.returncode != 0:
            safe = redact_text(result.stderr or result.stdout).strip()[:2000]
            raise GitError(f"Git command failed (ls-remote): {safe}")
        value = result.stdout.split(maxsplit=1)[0] if result.stdout.strip() else ""
        if not _OBJECT_ID.fullmatch(value):
            raise GitError("remote branch did not resolve to one commit")
        return value

    def remote_url(self, remote: str = "origin") -> str:
        self._validate_remote(remote)
        values = self._validated_remote_urls(remote, push=False)
        return values[0]

    def fetch(self, remote: str, branch: str) -> None:
        self._validate_remote(remote)
        self._validated_remote_urls(remote, push=False)
        validate_branch(branch)
        self.run(
            "fetch", "--no-tags", remote, f"refs/heads/{branch}:refs/remotes/{remote}/{branch}"
        )

    def validate_ref(self, branch: str) -> str:
        validate_branch(branch)
        result = self.run("check-ref-format", "--branch", branch, check=False)
        if result.returncode != 0:
            raise GitError("unsafe Git branch name")
        return branch

    def create_branch(self, branch: str, start_point: str = "HEAD") -> None:
        self.validate_ref(branch)
        start_sha = self.current_sha(start_point)
        self.run("switch", "--create", branch, start_sha)

    def merge_base(self, left: str, right: str) -> str:
        left_sha = self.current_sha(left)
        right_sha = self.current_sha(right)
        result = self.run("merge-base", left_sha, right_sha)
        value = result.stdout.strip()
        if not _OBJECT_ID.fullmatch(value):
            raise GitError("Git returned an invalid merge base")
        return value

    def is_ancestor(self, ancestor: str, descendant: str) -> bool:
        ancestor_sha = self.current_sha(ancestor)
        descendant_sha = self.current_sha(descendant)
        result = self.run("merge-base", "--is-ancestor", ancestor_sha, descendant_sha, check=False)
        if result.returncode not in {0, 1}:
            raise GitError("cannot determine Git commit ancestry")
        return result.returncode == 0

    def changed_paths_between(self, before: str, after: str) -> tuple[str, ...]:
        before_sha = self.current_sha(before)
        after_sha = self.current_sha(after)
        output = self.run("diff", "--name-only", "-z", before_sha, after_sha, "--").stdout
        return tuple(path for path in output.split("\0") if path)

    def tree_sha(self, commit: str) -> str:
        commit_sha = self.current_sha(commit)
        value = self.run("rev-parse", "--verify", f"{commit_sha}^{{tree}}").stdout.strip()
        if not _OBJECT_ID.fullmatch(value):
            raise GitError("Git returned an invalid tree object ID")
        return value

    def commit_tree_descendant(
        self,
        tree_from: str,
        *,
        parents: Sequence[str],
        message: str,
        identity: GitIdentity | None = None,
    ) -> str:
        """Create a non-force descendant with the exact validated candidate tree."""

        if not parents:
            raise GitError("a descendant commit requires at least one parent")
        if not message or "\n" in message or _CONTROL.search(message):
            raise GitError("commit subject must be one safe line")
        tree = self.tree_sha(tree_from)
        parent_shas = tuple(dict.fromkeys(self.current_sha(parent) for parent in parents))
        resolved = self.resolve_identity(identity)
        arguments = [
            "-c",
            f"user.name={resolved.name}",
            "-c",
            f"user.email={resolved.email}",
            "commit-tree",
            tree,
        ]
        for parent in parent_shas:
            arguments.extend(("-p", parent))
        arguments.extend(("-m", message))
        value = self.run(*arguments).stdout.strip()
        if not _OBJECT_ID.fullmatch(value):
            raise GitError("Git returned an invalid descendant commit ID")
        commit_sha = self.current_sha(value)
        if self.tree_sha(commit_sha) != tree:
            raise GitError("descendant commit tree differs from the validated candidate")
        return commit_sha

    def status_paths(self) -> tuple[str, ...]:
        raw = self.run("status", "--porcelain=v1", "-z", "--untracked-files=all").stdout
        fields = raw.split("\0")
        result: list[str] = []
        index = 0
        while index < len(fields) and fields[index]:
            entry = fields[index]
            if len(entry) < 4:
                raise GitError("cannot parse Git worktree status")
            status, path = entry[:2], entry[3:]
            result.append(path)
            if "R" in status or "C" in status:
                index += 1
                if index < len(fields) and fields[index]:
                    result.append(fields[index])
            index += 1
        return tuple(result)

    def ensure_no_overlapping_changes(self, target_paths: Sequence[str | PurePath]) -> None:
        targets = tuple(PurePosixPath(path) for path in self.normalize_paths(target_paths))
        conflicts: list[str] = []
        for dirty in self.status_paths():
            dirty_path = PurePosixPath(dirty.replace("\\", "/"))
            if any(_paths_overlap(dirty_path, target) for target in targets):
                conflicts.append(dirty_path.as_posix())
        if conflicts:
            raise DirtyWorktreeError(
                "calculated target paths overlap uncommitted changes: " + ", ".join(conflicts)
            )

    def normalize_paths(self, paths: Sequence[str | PurePath]) -> tuple[str, ...]:
        return tuple(_safe_relative_path(path, self.repository).as_posix() for path in paths)

    def staged_paths(self) -> tuple[str, ...]:
        output = self.run("diff", "--cached", "--name-only", "-z").stdout
        return tuple(path for path in output.split("\0") if path)

    def ensure_index_clean(self) -> None:
        staged = self.staged_paths()
        if staged:
            raise DirtyWorktreeError(
                "Git index already contains unrelated staged paths: " + ", ".join(staged)
            )

    def ensure_only_targets_staged(self, target_paths: Sequence[str | PurePath]) -> None:
        targets = tuple(PurePosixPath(path) for path in self.normalize_paths(target_paths))
        unrelated = [
            path
            for path in self.staged_paths()
            if not any(_paths_overlap(PurePosixPath(path), target) for target in targets)
        ]
        if unrelated:
            raise DirtyWorktreeError(
                "refusing to commit staged paths outside the calculated target set: "
                + ", ".join(unrelated)
            )

    def stage_paths(self, paths: Sequence[str | PurePath]) -> tuple[str, ...]:
        safe = self.normalize_paths(paths)
        if not safe:
            raise GitError("refusing to stage an empty path set")
        self.run("add", "--", *safe)
        return safe

    def has_staged_changes(self) -> bool:
        result = self.run("diff", "--cached", "--quiet", "--exit-code", check=False)
        if result.returncode not in {0, 1}:
            raise GitError("cannot determine whether the index changed")
        return result.returncode == 1

    def resolve_identity(self, explicit: GitIdentity | None = None) -> GitIdentity:
        local_name = self._config("user.name", "--local")
        local_email = self._config("user.email", "--local")
        if local_name and local_email:
            return _validate_identity(GitIdentity(local_name, local_email))
        if explicit is not None:
            return _validate_identity(explicit)
        global_name = self._config("user.name", "--global")
        global_email = self._config("user.email", "--global")
        if global_name and global_email:
            return _validate_identity(GitIdentity(global_name, global_email))
        raise GitIdentityError("Git user.name and user.email are required")

    def commit(self, message: str, *, identity: GitIdentity | None = None) -> str:
        if not message or "\n" in message or _CONTROL.search(message):
            raise GitError("commit subject must be one safe line")
        if not self.has_staged_changes():
            raise GitError("refusing to create an empty commit")
        resolved = self.resolve_identity(identity)
        self.run(
            "-c",
            f"user.name={resolved.name}",
            "-c",
            f"user.email={resolved.email}",
            "commit",
            "--no-gpg-sign",
            "--message",
            message,
        )
        return self.current_sha()

    def push_commit(self, remote: str, commit_sha: str, destination_branch: str) -> None:
        self._validate_remote(remote)
        self._validated_remote_urls(remote, push=True)
        self.current_sha(commit_sha)
        validate_branch(destination_branch)
        refspec = f"{commit_sha}:refs/heads/{destination_branch}"
        self.run("push", remote, refspec)

    def _validated_remote_urls(self, remote: str, *, push: bool) -> tuple[str, ...]:
        arguments = ["remote", "get-url"]
        if push:
            arguments.append("--push")
        arguments.extend(["--all", remote])
        output = self.run(*arguments).stdout
        values = tuple(line.strip() for line in output.splitlines() if line.strip())
        if not values:
            raise GitError("Git remote has no configured URL")
        for value in values:
            _validate_github_remote_url(value)
        return values

    def _github_token_for(self, arguments: Sequence[str]) -> str | None:
        """Use token askpass only for a validated HTTPS GitHub transport."""

        if self._github_token is None or not _uses_remote_auth(arguments):
            return None
        remote = _remote_auth_target(arguments)
        urls: tuple[str, ...]
        if "://" in remote or remote.startswith("git@"):
            urls = (_validate_github_remote_url(remote),)
        else:
            self._validate_remote(remote)
            urls = self._validated_remote_urls(remote, push=arguments[0] == "push")
        transports = {_github_remote_transport(value) for value in urls}
        if len(transports) != 1:
            raise GitError("Git remote mixes HTTPS and SSH transport URLs")
        return self._github_token if transports == {"https"} else None

    def _config(self, key: str, scope: str) -> str | None:
        result = self.run("config", scope, "--get", key, check=False)
        if result.returncode == 1:
            return None
        if result.returncode != 0:
            raise GitError(f"cannot read Git configuration key {key}")
        return result.stdout.strip() or None

    @staticmethod
    def _validate_arguments(arguments: Iterable[str]) -> None:
        for argument in arguments:
            if not isinstance(argument, str) or _CONTROL.search(argument):
                raise GitError("Git argument contains forbidden control characters")
            if argument in _BANNED_ARGUMENTS or argument.startswith("--force="):
                raise GitError("force push/reset arguments are forbidden")

    @staticmethod
    def _validate_remote(remote: str) -> None:
        if not _SAFE_REMOTE.fullmatch(remote):
            raise GitError("unsafe Git remote name")


def validate_branch(branch: str) -> str:
    if (
        not branch
        or len(branch) > 200
        or _CONTROL.search(branch)
        or branch.startswith("-")
        or branch.startswith("/")
        or branch.endswith(("/", ".", ".lock"))
        or ".." in branch
        or "//" in branch
        or "@{" in branch
        or any(character in branch for character in " ~^:?*[\\")
    ):
        raise GitError("unsafe Git branch name")
    return branch


def branch_slug(value: str, *, max_length: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")[:max_length].rstrip("-")
    return slug or "collection"


def _safe_relative_path(path: str | PurePath, root: Path) -> PurePosixPath:
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve().relative_to(root)
        except ValueError as exc:
            raise GitError("target path escapes the repository") from exc
    pure = PurePosixPath(candidate.as_posix())
    if not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise GitError("unsafe target path")
    return pure


def _paths_overlap(left: PurePosixPath, right: PurePosixPath) -> bool:
    return left == right or left in right.parents or right in left.parents


def _validate_identity(identity: GitIdentity) -> GitIdentity:
    if (
        not identity.name.strip()
        or not identity.email.strip()
        or _CONTROL.search(identity.name)
        or _CONTROL.search(identity.email)
        or "@" not in identity.email
    ):
        raise GitIdentityError("Git identity is incomplete or unsafe")
    return GitIdentity(identity.name.strip(), identity.email.strip())


def _validate_github_remote_url(value: str) -> str:
    if not value or _CONTROL.search(value) or url_has_credentials(value):
        raise GitError("repository remote URL contains forbidden credentials or controls")
    match = _GITHUB_SCP_REMOTE.fullmatch(value)
    if match is not None:
        _validate_github_path(match.group("owner"), match.group("repository"))
        return value
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise GitError("repository remote URL is malformed") from exc
    scheme = parsed.scheme.lower()
    if parsed.hostname != "github.com" or port is not None or parsed.query or parsed.fragment:
        raise GitError("only canonical github.com Git remotes are accepted")
    if scheme == "https":
        if parsed.username is not None or parsed.password is not None:
            raise GitError("repository remote URL contains forbidden credentials")
    elif scheme == "ssh":
        if parsed.username != "git" or parsed.password is not None:
            raise GitError("GitHub SSH remotes must use the git user")
    else:
        raise GitError("only canonical HTTPS or SSH GitHub remotes are accepted")
    components = parsed.path.strip("/").split("/")
    if len(components) != 2 or parsed.path.endswith("/"):
        raise GitError("GitHub remote must identify exactly one repository")
    owner, repository = components
    _validate_github_path(owner, repository.removesuffix(".git"))
    return value


def _validate_github_path(owner: str, repository: str) -> None:
    repository = repository.removesuffix(".git")
    if not _GITHUB_OWNER.fullmatch(owner) or not _GITHUB_REPOSITORY.fullmatch(repository):
        raise GitError("GitHub remote contains an unsafe owner or repository name")


def _git_environment() -> dict[str, str]:
    result = dict(os.environ)
    for name in (
        "TELEGRAM_BOT_TOKEN",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "GH_TOKEN",
        "GITHUB_TOKEN",
    ):
        result.pop(name, None)
    result["GIT_TERMINAL_PROMPT"] = "0"
    result["GIT_CONFIG_NOSYSTEM"] = result.get("GIT_CONFIG_NOSYSTEM", "0")
    return result


@contextmanager
def git_subprocess_environment(github_token: str | None) -> Iterator[dict[str, str]]:
    """Yield a sanitized Git environment with an ephemeral HTTPS askpass bridge."""

    token = _validate_github_token(github_token)
    environment = _git_environment()
    if token is None:
        yield environment
        return

    for name in (
        "GIT_ASKPASS",
        "GIT_ASKPASS_REQUIRE",
        "SSH_ASKPASS",
        _ASKPASS_SECRET_ENV,
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_GLOBAL",
    ):
        environment.pop(name, None)
    for name in tuple(environment):
        if name.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_", "GIT_TRACE")):
            environment.pop(name, None)
    environment.pop("GIT_CURL_VERBOSE", None)

    directory = Path(tempfile.mkdtemp(prefix="mojilex-git-auth-"))
    secret_path = directory / "secret"
    helper_path = directory / "askpass.py"
    launcher_path = directory / ("askpass.cmd" if os.name == "nt" else "askpass")
    cleanup_error: OSError | None = None
    try:
        _write_private_file(secret_path, token)
        _write_private_file(helper_path, _ASKPASS_SOURCE)
        if os.name == "nt":
            launcher = f'@echo off\r\n"{sys.executable}" "%~dp0askpass.py" %*\r\n'
        else:
            launcher = (
                "#!/bin/sh\nexec "
                f'{shlex.quote(sys.executable)} {shlex.quote(str(helper_path))} "$@"\n'
            )
        _write_private_file(launcher_path, launcher)
        launcher_path.chmod(0o700)
        hooks_path = directory / "hooks-disabled"
        hooks_path.mkdir(mode=0o700)
        environment.update(
            {
                "GIT_ASKPASS": str(launcher_path),
                "GIT_ASKPASS_REQUIRE": "force",
                _ASKPASS_SECRET_ENV: str(secret_path),
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_COUNT": "2",
                "GIT_CONFIG_KEY_0": "credential.helper",
                "GIT_CONFIG_VALUE_0": "",
                "GIT_CONFIG_KEY_1": "core.hooksPath",
                "GIT_CONFIG_VALUE_1": str(hooks_path),
            }
        )
        yield environment
    finally:
        try:
            if secret_path.exists():
                size = secret_path.stat().st_size
                with secret_path.open("r+b", buffering=0) as stream:
                    stream.write(b"\0" * size)
                    stream.flush()
                    os.fsync(stream.fileno())
            shutil.rmtree(directory)
        except OSError as exc:
            cleanup_error = exc
        if cleanup_error is not None:
            raise GitError("ephemeral Git credential files could not be removed") from cleanup_error


def _write_private_file(path: Path, content: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, content.encode("utf-8"))
    finally:
        os.close(descriptor)
    path.chmod(0o600)


def _validate_github_token(value: str | None) -> str | None:
    if value is None:
        return None
    if not value or _CONTROL.search(value):
        raise GitError("GitHub token is empty or contains forbidden controls")
    return value


def _uses_remote_auth(arguments: Sequence[str]) -> bool:
    return bool(arguments) and arguments[0] in _REMOTE_AUTH_COMMANDS


def _remote_auth_target(arguments: Sequence[str]) -> str:
    for argument in arguments[1:]:
        if argument == "--":
            continue
        if argument in _REMOTE_OPTIONS_WITHOUT_VALUES:
            continue
        if argument.startswith("-"):
            raise GitError("cannot safely determine Git remote behind an unsupported option")
        return argument
    raise GitError("Git remote command is missing its remote target")


def _github_remote_transport(value: str) -> str:
    if _GITHUB_SCP_REMOTE.fullmatch(value) is not None:
        return "ssh"
    return "https" if urlsplit(value).scheme.lower() == "https" else "ssh"
