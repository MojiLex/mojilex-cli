"""Safe GitHub CLI primitives with JSON-only machine parsing."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from mojilex_cli.config.secrets import redact_text, url_has_credentials
from mojilex_cli.git import validate_branch

_OWNER = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]{1,100}\Z")
_OBJECT_ID = re.compile(r"[0-9a-f]{40,64}\Z")
_TELEGRAM_TOKEN = re.compile(r"\b[0-9]{5,12}:[A-Za-z0-9_-]{20,}\b")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class GitHubError(RuntimeError):
    code = "GITHUB_PUBLISH_FAILED"


class RequiredCheckError(GitHubError):
    code = "REQUIRED_CHECK_FAILED"


class RepositoryRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    owner: str
    name: str

    @classmethod
    def parse(cls, value: str) -> RepositoryRef:
        if "://" not in value:
            parts = value.split("/")
            if len(parts) != 2:
                raise GitHubError("repository must be OWNER/REPO or a canonical GitHub URL")
            owner, name = parts
        else:
            if url_has_credentials(value):
                raise GitHubError("GitHub URL must not contain credentials")
            parsed = urlsplit(value)
            if (
                parsed.scheme.lower() != "https"
                or parsed.hostname != "github.com"
                or parsed.port is not None
                or parsed.query
                or parsed.fragment
            ):
                raise GitHubError("only canonical https://github.com URLs are accepted")
            parts = parsed.path.strip("/").split("/")
            if len(parts) != 2:
                raise GitHubError("GitHub URL must identify exactly one repository")
            owner, name = parts
            name = name.removesuffix(".git")
        if not _OWNER.fullmatch(owner) or not _REPOSITORY.fullmatch(name):
            raise GitHubError("unsafe GitHub owner or repository name")
        return cls(owner=owner, name=name)

    def __str__(self) -> str:
        return f"{self.owner}/{self.name}"


class RepositoryInfo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name_with_owner: str
    permission: Literal["READ", "TRIAGE", "WRITE", "MAINTAIN", "ADMIN"]
    default_branch: str

    @property
    def can_write(self) -> bool:
        return self.permission in {"WRITE", "MAINTAIN", "ADMIN"}


class RequiredCheck(BaseModel):
    """A required status context, optionally pinned to one GitHub App."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    context: str = Field(min_length=1, max_length=255, pattern=r"^[^\x00-\x1f\x7f]+$")
    app_id: int | None = Field(default=None, ge=1)

    @property
    def label(self) -> str:
        return self.context if self.app_id is None else f"{self.context} (app {self.app_id})"


class CommitChecks(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    commit_sha: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    successful: tuple[str, ...]
    pending: tuple[str, ...]
    failed: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def complete_and_successful(self) -> bool:
        return not self.pending and not self.failed and not self.missing


@dataclass(frozen=True)
class PullRequestResult:
    url: str
    number: int | None
    reused: bool
    completed: bool = False
    reopened: bool = False


@dataclass(frozen=True)
class _PullRequestMatch:
    url: str
    number: int
    state: Literal["OPEN", "CLOSED", "MERGED"]
    base_branch: str
    merged: bool


class GitHubCLI:
    def __init__(
        self,
        *,
        executable: str = "gh",
        token: str | None = None,
        timeout_seconds: float = 60,
    ) -> None:
        self.executable = executable
        self._token = token
        self._secrets = (token,) if token else ()
        self.timeout_seconds = timeout_seconds

    def run(self, *arguments: str, check: bool = True) -> str:
        stdout, stderr, returncode = self._execute(*arguments)
        if check and returncode != 0:
            message = redact_text(stderr or stdout, self._secrets).strip()[:2000]
            raise GitHubError(f"GitHub CLI request failed: {message}")
        return stdout

    def _execute(self, *arguments: str) -> tuple[str, str, int]:
        if any(not isinstance(arg, str) or "\x00" in arg or "\r" in arg for arg in arguments):
            raise GitHubError("GitHub CLI argument contains control characters")
        try:
            completed = subprocess.run(
                [self.executable, *arguments],
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                timeout=self.timeout_seconds,
                env=self._environment(),
            )
        except FileNotFoundError as exc:
            raise GitHubError("GitHub CLI executable was not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise GitHubError("GitHub CLI request exceeded its time limit") from exc
        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        if len(stdout) + len(stderr) > 8 * 1024 * 1024:
            raise GitHubError("GitHub CLI output exceeded the safe limit")
        return stdout, stderr, completed.returncode

    def auth_status(self) -> None:
        # This read-only probe can time out even while stored credentials are valid.
        # Retry only transport failures; never repeat publication mutations here.
        for attempt in range(3):
            try:
                self.run("auth", "status", "--hostname", "github.com")
                return
            except GitHubError as exc:
                message = str(exc).casefold()
                authentication_failed = any(
                    marker in message
                    for marker in ("http 401", "http 403", "invalid token", "bad credentials")
                )
                transient = any(
                    marker in message
                    for marker in (
                        "timeout trying to log in",
                        "exceeded its time limit",
                        "i/o timeout",
                        "tls handshake timeout",
                        "connection reset",
                        "temporary failure in name resolution",
                    )
                )
                if authentication_failed or not transient or attempt == 2:
                    raise
                time.sleep(attempt + 1)

    def current_user(self) -> tuple[str, int]:
        value = self.api("user")
        login, actor_id = (
            value.get("login"),
            value.get("id") if isinstance(value, Mapping) else None,
        )
        if (
            not isinstance(login, str)
            or not _OWNER.fullmatch(login)
            or not isinstance(actor_id, int)
        ):
            raise GitHubError("GitHub returned an invalid authenticated user")
        return login, actor_id

    def repository_info(self, repository: RepositoryRef) -> RepositoryInfo:
        raw = self.run(
            "repo",
            "view",
            str(repository),
            "--json",
            "nameWithOwner,viewerPermission,defaultBranchRef",
        )
        try:
            value = json.loads(raw)
            default = value["defaultBranchRef"]["name"]
            return RepositoryInfo(
                name_with_owner=value["nameWithOwner"],
                permission=value["viewerPermission"],
                default_branch=default,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GitHubError("GitHub returned invalid repository metadata") from exc

    def ensure_fork(self, repository: RepositoryRef) -> RepositoryRef:
        self.run("repo", "fork", str(repository), "--clone=false", "--remote=false")
        login, _ = self.current_user()
        fork = RepositoryRef(owner=login, name=repository.name)
        info = self.repository_info(fork)
        if not info.can_write:
            raise GitHubError("authenticated user cannot write to the selected fork")
        return fork

    def create_or_reuse_pull_request(
        self,
        *,
        repository: RepositoryRef,
        head_owner: str,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> PullRequestResult:
        existing = self.reconcile_pull_request(
            repository=repository,
            head_owner=head_owner,
            head_branch=head_branch,
            base_branch=base_branch,
        )
        if existing is not None:
            return existing
        return self._create_pull_request(
            repository=repository,
            head_owner=head_owner,
            head_branch=head_branch,
            base_branch=base_branch,
            title=title,
            body=body,
        )

    def reconcile_pull_request(
        self,
        *,
        repository: RepositoryRef,
        head_owner: str,
        head_branch: str,
        base_branch: str,
    ) -> PullRequestResult | None:
        validate_branch(head_branch)
        validate_branch(base_branch)
        existing = self.run(
            "pr",
            "list",
            "--repo",
            str(repository),
            "--head",
            f"{head_owner}:{head_branch}",
            "--state",
            "all",
            "--limit",
            "1000",
            "--json",
            "number,url,state,baseRefName,headRefName,headRepositoryOwner,mergedAt",
        )
        try:
            matches = json.loads(existing)
        except json.JSONDecodeError as exc:
            raise GitHubError("GitHub returned an invalid Pull Request list") from exc
        if not isinstance(matches, list):
            raise GitHubError("GitHub returned an invalid Pull Request list")
        if len(matches) >= 1000:
            raise GitHubError("Pull Request reconciliation exceeded the safe result limit")
        parsed = [
            _parse_pull_request_match(
                value,
                expected_head_owner=head_owner,
                expected_head_branch=head_branch,
            )
            for value in matches
        ]
        wrong_base = [value for value in parsed if value.base_branch != base_branch]
        if wrong_base:
            raise GitHubError(
                "matching run Pull Request targets a different base branch; "
                "manual resolution required"
            )
        if len(parsed) > 1:
            raise GitHubError("multiple matching Pull Requests require manual resolution")
        if not parsed:
            return None
        match = parsed[0]
        if match.merged:
            return PullRequestResult(
                url=match.url,
                number=match.number,
                reused=True,
                completed=True,
            )
        if match.state == "OPEN":
            return PullRequestResult(url=match.url, number=match.number, reused=True)
        if match.state != "CLOSED":
            raise GitHubError("GitHub returned an unsupported Pull Request state")
        self.run("pr", "reopen", str(match.number), "--repo", str(repository))
        verified_raw = self.run(
            "pr",
            "view",
            str(match.number),
            "--repo",
            str(repository),
            "--json",
            "number,url,state,baseRefName,headRefName,headRepositoryOwner,mergedAt",
        )
        try:
            verified_value = json.loads(verified_raw)
        except json.JSONDecodeError as exc:
            raise GitHubError("GitHub returned invalid reopened Pull Request metadata") from exc
        verified = _parse_pull_request_match(
            verified_value,
            expected_head_owner=head_owner,
            expected_head_branch=head_branch,
        )
        if verified.base_branch != base_branch or verified.state != "OPEN" or verified.merged:
            raise GitHubError(
                "closed Pull Request could not be safely reopened on the expected base"
            )
        return PullRequestResult(
            url=verified.url,
            number=verified.number,
            reused=True,
            reopened=True,
        )

    def _create_pull_request(
        self,
        *,
        repository: RepositoryRef,
        head_owner: str,
        head_branch: str,
        base_branch: str,
        title: str,
        body: str,
    ) -> PullRequestResult:
        validate_branch(head_branch)
        validate_branch(base_branch)
        _validate_public_text(title, self._secrets, single_line=True)
        _validate_public_text(body, self._secrets, single_line=False)
        descriptor, raw_path = tempfile.mkstemp(prefix="mojilex-pr-", suffix=".md")
        body_path = Path(raw_path)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(body)
                if not body.endswith("\n"):
                    stream.write("\n")
            output = self.run(
                "pr",
                "create",
                "--repo",
                str(repository),
                "--head",
                f"{head_owner}:{head_branch}",
                "--base",
                base_branch,
                "--title",
                title,
                "--body-file",
                str(body_path),
            )
        finally:
            body_path.unlink(missing_ok=True)
        url = next(
            (line.strip() for line in output.splitlines() if line.startswith("https://")), ""
        )
        return PullRequestResult(url=_validate_github_url(url), number=None, reused=False)

    def required_checks(
        self, repository: RepositoryRef, base_branch: str
    ) -> tuple[RequiredCheck, ...]:
        """Discover branch-protection and applicable ruleset checks, failing closed."""

        validate_branch(base_branch)
        checks_by_key: dict[tuple[str, int | None], RequiredCheck] = {}
        protection = self.api(
            f"repos/{repository}/branches/{base_branch}/protection/required_status_checks",
            allow_not_found=True,
        )
        if protection is not None:
            if not isinstance(protection, Mapping):
                raise GitHubError("required status-check response is invalid")
            contexts = protection.get("contexts", [])
            checks = protection.get("checks", [])
            if isinstance(contexts, list):
                for value in contexts:
                    if not isinstance(value, str):
                        raise GitHubError("required status-check context is invalid")
                    _add_required_check(checks_by_key, value, None)
            if isinstance(checks, list):
                for value in checks:
                    if not isinstance(value, Mapping) or not isinstance(value.get("context"), str):
                        raise GitHubError("required status-check descriptor is invalid")
                    app_id = value.get("app_id")
                    if app_id is not None and not isinstance(app_id, int):
                        raise GitHubError("required status-check app ID is invalid")
                    _add_required_check(checks_by_key, value["context"], app_id)
        for detail in self._applicable_active_rulesets(repository, base_branch):
            rules = detail.get("rules", [])
            if not isinstance(rules, list):
                raise GitHubError("GitHub ruleset rules are invalid")
            for rule in rules:
                if not isinstance(rule, Mapping) or not isinstance(rule.get("type"), str):
                    raise RequiredCheckError("an active ruleset contains an invalid rule")
                rule_type = rule["type"]
                if rule_type not in _DIRECT_PUSH_PROVABLE_RULE_TYPES:
                    raise RequiredCheckError(
                        f"active ruleset rule {rule_type!r} cannot be proven for direct push"
                    )
                if rule_type != "required_status_checks":
                    continue
                parameters = rule.get("parameters", {})
                required = (
                    parameters.get("required_status_checks", [])
                    if isinstance(parameters, Mapping)
                    else []
                )
                if not isinstance(required, list):
                    raise GitHubError("ruleset required status-check list is invalid")
                for check in required:
                    if not isinstance(check, Mapping) or not isinstance(check.get("context"), str):
                        raise GitHubError("ruleset required status-check descriptor is invalid")
                    integration_id = check.get("integration_id")
                    if integration_id is not None and not isinstance(integration_id, int):
                        raise GitHubError("ruleset status-check integration ID is invalid")
                    _add_required_check(checks_by_key, check["context"], integration_id)
        if not checks_by_key:
            raise RequiredCheckError("no required checks could be proven for direct push")
        return tuple(
            checks_by_key[key]
            for key in sorted(checks_by_key, key=lambda item: (item[0], item[1] or -1))
        )

    def has_direct_push_bypass(self, repository: RepositoryRef, base_branch: str) -> bool:
        """Prove a point bypass for the current user on every applicable active ruleset.

        General RepositoryRole/OrganizationAdmin bypasses intentionally do not count: the MVP
        policy requires a point actor and must not equate ADMIN with bypass permission.
        """

        validate_branch(base_branch)
        _, actor_id = self.current_user()
        applicable = self._applicable_active_rulesets(repository, base_branch)
        if not applicable:
            return False
        for ruleset in applicable:
            actors = ruleset.get("bypass_actors")
            if not isinstance(actors, list):
                return False
            point_match = any(
                isinstance(actor, Mapping)
                and actor.get("actor_type") == "User"
                and actor.get("actor_id") == actor_id
                and actor.get("bypass_mode") in {"always", "exempt"}
                for actor in actors
            )
            if not point_match:
                return False
        return True

    def commit_checks(
        self,
        repository: RepositoryRef,
        commit_sha: str,
        required: Sequence[str | RequiredCheck],
    ) -> CommitChecks:
        if not _OBJECT_ID.fullmatch(commit_sha) or not required:
            raise RequiredCheckError("commit SHA and required check set must be explicit")
        required_checks = _normalize_required_checks(required)
        check_runs = self._paginated_items(
            f"repos/{repository}/commits/{commit_sha}/check-runs", "check_runs"
        )
        statuses = self._paginated_items(
            f"repos/{repository}/commits/{commit_sha}/status", "statuses"
        )
        run_states: dict[tuple[str, int | None], list[str]] = {}
        for check in check_runs:
            if not isinstance(check, Mapping) or not isinstance(check.get("name"), str):
                raise GitHubError("GitHub check-run response is invalid")
            app = check.get("app")
            app_id = app.get("id") if isinstance(app, Mapping) else None
            if app_id is not None and not isinstance(app_id, int):
                raise GitHubError("GitHub check-run app ID is invalid")
            status, conclusion = check.get("status"), check.get("conclusion")
            state = (
                "success"
                if status == "completed" and conclusion == "success"
                else "failed"
                if status == "completed"
                else "pending"
            )
            run_states.setdefault((check["name"], app_id), []).append(state)
        # The combined-status endpoint returns newest statuses first. Keep the newest
        # legacy state, but only use it when no check-run exists for an unbound context.
        legacy_states: dict[str, str] = {}
        for status in statuses:
            if not isinstance(status, Mapping) or not isinstance(status.get("context"), str):
                raise GitHubError("GitHub commit-status response is invalid")
            raw_state = status.get("state")
            state = (
                "success"
                if raw_state == "success"
                else "pending"
                if raw_state == "pending"
                else "failed"
            )
            legacy_states.setdefault(status["context"], state)

        classified: dict[str, str | None] = {}
        for required_check in required_checks:
            if required_check.app_id is None:
                values = [
                    state
                    for (context, _), states_for_run in run_states.items()
                    if context == required_check.context
                    for state in states_for_run
                ]
                required_state = (
                    _collapse_states(values)
                    if values
                    else legacy_states.get(required_check.context)
                )
            else:
                required_state = _collapse_states(
                    run_states.get((required_check.context, required_check.app_id), [])
                )
            classified[required_check.label] = required_state
        successful = tuple(sorted(name for name, state in classified.items() if state == "success"))
        pending = tuple(sorted(name for name, state in classified.items() if state == "pending"))
        failed = tuple(sorted(name for name, state in classified.items() if state == "failed"))
        missing = tuple(sorted(name for name, state in classified.items() if state is None))
        return CommitChecks(
            commit_sha=commit_sha,
            successful=successful,
            pending=pending,
            failed=failed,
            missing=missing,
        )

    async def wait_for_checks(
        self,
        repository: RepositoryRef,
        commit_sha: str,
        required: Sequence[str | RequiredCheck],
        *,
        timeout_seconds: float = 900,
        poll_seconds: float = 10,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> CommitChecks:
        deadline = time.monotonic() + timeout_seconds
        while True:
            checks = self.commit_checks(repository, commit_sha, required)
            if checks.failed:
                raise RequiredCheckError(
                    "required candidate checks failed: " + ", ".join(checks.failed)
                )
            if checks.complete_and_successful:
                return checks
            if time.monotonic() >= deadline:
                raise RequiredCheckError(
                    "required candidate checks were missing or incomplete before timeout"
                )
            await sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))

    def _applicable_active_rulesets(
        self, repository: RepositoryRef, base_branch: str
    ) -> list[Mapping[str, object]]:
        summaries = self._paginated_items(
            f"repos/{repository}/rulesets?includes_parents=true", None
        )
        full_ref = f"refs/heads/{base_branch}"
        applicable: list[Mapping[str, object]] = []
        for summary in summaries:
            if not isinstance(summary, Mapping) or not isinstance(summary.get("id"), int):
                raise GitHubError("GitHub ruleset summary is invalid")
            detail = self.api(f"repos/{repository}/rulesets/{summary['id']}")
            if not isinstance(detail, Mapping):
                raise GitHubError("GitHub ruleset detail is invalid")
            if detail.get("enforcement") != "active":
                continue
            if _ruleset_applies(detail.get("conditions"), full_ref, base_branch):
                applicable.append(detail)
        return applicable

    def _paginated_items(self, endpoint: str, key: str | None) -> list[object]:
        result: list[object] = []
        for page in range(1, 101):
            separator = "&" if "?" in endpoint else "?"
            payload = self.api(f"{endpoint}{separator}per_page=100&page={page}")
            if key is None:
                batch = payload
            elif isinstance(payload, Mapping):
                batch = payload.get(key)
            else:
                batch = None
            if not isinstance(batch, list):
                raise GitHubError("GitHub paginated response is invalid")
            result.extend(batch)
            if len(batch) < 100:
                return result
        raise GitHubError("GitHub pagination exceeded the safe page limit")

    def api(self, endpoint: str, *, allow_not_found: bool = False) -> Any:
        if not endpoint or endpoint.startswith(("http://", "https://")) or "\x00" in endpoint:
            raise GitHubError("unsafe GitHub API endpoint")
        output, stderr, returncode = self._execute("api", "--method", "GET", endpoint)
        if returncode != 0:
            if allow_not_found and re.search(r"\bHTTP 404\b", stderr, re.I):
                return None
            message = redact_text(stderr or output, self._secrets).strip()[:2000]
            raise GitHubError(f"GitHub API request failed: {message}")
        try:
            return json.loads(output)
        except json.JSONDecodeError as exc:
            raise GitHubError("GitHub API returned invalid JSON") from exc

    def _environment(self) -> dict[str, str]:
        result = {
            "PATH": os.environ.get("PATH", ""),
            "GH_PROMPT_DISABLED": "1",
            "GH_HOST": "github.com",
        }
        for name in ("SYSTEMROOT", "WINDIR", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
            if name in os.environ:
                result[name] = os.environ[name]
        token = self._token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token:
            result["GH_TOKEN"] = token
        return result


def _parse_pull_request_match(
    value: object,
    *,
    expected_head_owner: str,
    expected_head_branch: str,
) -> _PullRequestMatch:
    if not isinstance(value, Mapping):
        raise GitHubError("GitHub returned invalid Pull Request metadata")
    number = value.get("number")
    url = value.get("url")
    state = value.get("state")
    base_branch = value.get("baseRefName")
    head_branch = value.get("headRefName")
    raw_owner = value.get("headRepositoryOwner")
    head_owner = raw_owner.get("login") if isinstance(raw_owner, Mapping) else raw_owner
    merged_at = value.get("mergedAt")
    if (
        not isinstance(number, int)
        or number < 1
        or not isinstance(url, str)
        or state not in {"OPEN", "CLOSED", "MERGED"}
        or not isinstance(base_branch, str)
        or not isinstance(head_branch, str)
        or not isinstance(head_owner, str)
        or (merged_at is not None and (not isinstance(merged_at, str) or not merged_at))
    ):
        raise GitHubError("GitHub returned invalid Pull Request metadata")
    if (
        head_branch != expected_head_branch
        or head_owner.casefold() != expected_head_owner.casefold()
    ):
        raise GitHubError("GitHub Pull Request head does not match the deterministic run branch")
    if state == "OPEN" and merged_at is not None:
        raise GitHubError("open Pull Request unexpectedly has merge metadata")
    if state == "MERGED" and merged_at is None:
        raise GitHubError("merged Pull Request is missing merge metadata")
    return _PullRequestMatch(
        url=_validate_github_url(url),
        number=number,
        state=cast(Literal["OPEN", "CLOSED", "MERGED"], state),
        base_branch=base_branch,
        merged=state == "MERGED" or merged_at is not None,
    )


def _validate_public_text(value: str, secrets: tuple[str, ...], *, single_line: bool) -> None:
    if not value or "\x00" in value or (single_line and "\n" in value):
        raise GitHubError("Pull Request text is empty or contains forbidden controls")
    if _TELEGRAM_TOKEN.search(value) or any(secret and secret in value for secret in secrets):
        raise GitHubError("credential detected in Pull Request text")
    if any(url_has_credentials(token) for token in value.split() if "://" in token):
        raise GitHubError("credential URL detected in Pull Request text")


def _validate_github_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise GitHubError("GitHub returned an invalid URL") from exc
    if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.username is not None:
        raise GitHubError("GitHub returned a non-canonical URL")
    return value


def _ruleset_applies(conditions: object, full_ref: str, default_branch: str) -> bool:
    if not isinstance(conditions, Mapping):
        return True
    ref = conditions.get("ref_name")
    if not isinstance(ref, Mapping):
        return True
    includes = ref.get("include", ["~ALL"])
    excludes = ref.get("exclude", [])
    if not isinstance(includes, list) or not isinstance(excludes, list):
        raise GitHubError("GitHub ruleset ref conditions are invalid")

    def matches(pattern: object) -> bool:
        if pattern == "~ALL":
            return True
        if pattern == "~DEFAULT_BRANCH":
            return full_ref == f"refs/heads/{default_branch}"
        return isinstance(pattern, str) and _github_fnmatch(full_ref, pattern)

    return any(matches(pattern) for pattern in includes) and not any(
        matches(pattern) for pattern in excludes
    )


_DIRECT_PUSH_PROVABLE_RULE_TYPES = frozenset(
    {
        # Required status checks are evaluated against the exact candidate commit.
        "required_status_checks",
        # These rules are either the reason for the point bypass or are made
        # irrelevant by the operation (existing branch, ordinary fast-forward,
        # no deletion). All other/unknown rules fail closed.
        "pull_request",
        "non_fast_forward",
        "creation",
        "update",
        "deletion",
    }
)


def _add_required_check(
    checks: dict[tuple[str, int | None], RequiredCheck], context: str, app_id: int | None
) -> None:
    try:
        descriptor = RequiredCheck(context=context, app_id=app_id)
    except ValueError as exc:
        raise GitHubError("required status-check descriptor is invalid") from exc
    if app_id is not None:
        checks.pop((context, None), None)
    elif any(name == context and integration is not None for name, integration in checks):
        return
    checks[(context, app_id)] = descriptor


def _normalize_required_checks(
    required: Sequence[str | RequiredCheck],
) -> tuple[RequiredCheck, ...]:
    result: dict[tuple[str, int | None], RequiredCheck] = {}
    for value in required:
        try:
            check = value if isinstance(value, RequiredCheck) else RequiredCheck(context=value)
        except (TypeError, ValueError) as exc:
            raise RequiredCheckError("required check set is invalid") from exc
        result[(check.context, check.app_id)] = check
    if not result:
        raise RequiredCheckError("required check set must not be empty")
    return tuple(result[key] for key in sorted(result, key=lambda item: (item[0], item[1] or -1)))


def _collapse_states(values: Sequence[str]) -> str | None:
    if not values:
        return None
    if "failed" in values:
        return "failed"
    if "pending" in values:
        return "pending"
    return "success"


def _github_fnmatch(value: str, pattern: str) -> bool:
    """Match GitHub ref globs using File::FNM_PATHNAME slash semantics."""

    expression: list[str] = ["^"]
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "*":
            end = index
            while end < len(pattern) and pattern[end] == "*":
                end += 1
            expression.append(".*" if end - index > 1 else "[^/]*")
            index = end
            continue
        if character == "?":
            expression.append("[^/]")
        elif character == "[":
            end = pattern.find("]", index + 1)
            if end == -1:
                expression.append(r"\[")
            else:
                content = pattern[index + 1 : end]
                if content.startswith("!"):
                    content = "^" + content[1:]
                elif content.startswith("^"):
                    content = "\\" + content
                expression.append("[" + content.replace("\\", r"\\") + "]")
                index = end
        else:
            expression.append(re.escape(character))
        index += 1
    expression.append(r"\Z")
    return re.match("".join(expression), value) is not None
