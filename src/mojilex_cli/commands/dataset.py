"""Commands that operate directly on a local canonical dataset."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from mojilex_cli.dataset import build_index, validate_dataset
from mojilex_cli.dataset.git_provenance import (
    GitSourceProvenance,
    resolve_head_revision,
    tool_source_checkout_root,
    verify_release_source,
    verify_tool_source,
)
from mojilex_cli.domain import preview_takedown, review_emoji, set_availability, takedown

from .runtime import CommandError, CommandResult, require_local_repository


def validate_command(path: Path, *, strict: bool) -> CommandResult:
    root = require_local_repository(path)
    report = validate_dataset(root, strict=strict)
    issues = [
        {"code": issue.code, "path": issue.path, "message": issue.message}
        for issue in report.issues
    ]
    if issues:
        raise CommandError(
            "VALIDATION_FAILED",
            f"Dataset validation found {len(issues)} issue(s).",
            hint="Inspect result details or run without --json to see the failing paths.",
            details={"issues": issues},
        )
    return CommandResult(result={"path": str(root), "strict": strict, "issues": 0})


def build_index_command(
    path: Path,
    output: Path | None,
    *,
    snapshot_id: str,
    source_date_epoch: int,
    git_commit: str | None = None,
    tool_commit: str | None = None,
    dependency_lock_sha256: str | None = None,
) -> CommandResult:
    root = require_local_repository(path)
    revision = git_commit if git_commit is not None else resolve_head_revision(root).commit
    provenance = verify_release_source(root, revision)
    tool_root = tool_source_checkout_root()
    tool_provenance: GitSourceProvenance | None = None
    if tool_root is None:
        if tool_commit is None or dependency_lock_sha256 is None:
            raise ValueError(
                "installed CLI builds require explicit --tool-commit and --dependency-lock-sha256"
            )
        resolved_tool_commit = tool_commit
        resolved_lock_sha256 = dependency_lock_sha256
    else:
        selected_tool_commit = (
            tool_commit if tool_commit is not None else resolve_head_revision(tool_root).commit
        )
        tool_provenance = verify_tool_source(tool_root, selected_tool_commit)
        resolved_tool_commit = tool_provenance.commit
        resolved_lock_sha256 = hashlib.sha256(
            (tool_root / "pyproject.toml").read_bytes()
        ).hexdigest()
        if dependency_lock_sha256 is not None and dependency_lock_sha256 != resolved_lock_sha256:
            raise ValueError(
                "dependency_lock_sha256 differs from the verified CLI dependency contract"
            )
    built = build_index(
        root,
        output,
        snapshot_id=snapshot_id,
        source_date_epoch=source_date_epoch,
        git_commit=provenance.commit,
        tool_commit=resolved_tool_commit,
        dependency_lock_sha256=resolved_lock_sha256,
    )
    after = verify_release_source(root, provenance.commit)
    tool_after = (
        verify_tool_source(tool_root, tool_provenance.commit)
        if tool_root is not None and tool_provenance is not None
        else None
    )
    if (
        after != provenance
        or built.git_commit != provenance.commit
        or built.git_object_format != provenance.object_format
    ):
        raise ValueError("distribution manifest Git provenance differs from verified source")
    if (
        tool_after != tool_provenance
        or built.tool_commit != resolved_tool_commit
        or built.dependency_lock_sha256 != resolved_lock_sha256
    ):
        raise ValueError("distribution manifest tool provenance differs from verified source")
    return CommandResult(
        result={
            "output_directory": str(built.output_directory),
            "git_commit": built.git_commit,
            "git_object_format": built.git_object_format,
            "files": built.file_sha256,
            "counts": built.counts,
        }
    )


def review_command(path: Path, emoji_id: str, action: str, reviewer: str | None) -> CommandResult:
    root = require_local_repository(path)
    identity = reviewer or _git_config(root, "user.name")
    if not identity:
        raise CommandError(
            "CONFIG_MISSING",
            "Reviewer identity is required.",
            hint="Pass --reviewer or configure git user.name in the data repository.",
        )
    changed = review_emoji(root, emoji_id, action, reviewer=identity)
    return CommandResult(
        status="noop" if changed.status == "noop" else "succeeded",  # type: ignore[arg-type]
        result={
            "target_id": changed.target_id,
            "reviewer": identity,
            "changed_paths": [str(item) for item in changed.changed_paths],
        },
    )


def set_status_command(
    path: Path,
    entity_id: str,
    availability: str,
    reason: str | None,
    reviewer: str | None,
) -> CommandResult:
    root = require_local_repository(path)
    identity = reviewer or _git_config(root, "user.name")
    changed = set_availability(
        root,
        entity_id,
        availability,
        reason_code=reason,
        reviewer=identity,
    )
    return CommandResult(
        status="noop" if changed.status == "noop" else "succeeded",  # type: ignore[arg-type]
        result={
            "target_id": changed.target_id,
            "availability": availability,
            "changed_paths": [str(item) for item in changed.changed_paths],
        },
    )


def takedown_preview_command(path: Path, entity_id: str, reason: str) -> dict[str, object]:
    root = require_local_repository(path)
    impact = preview_takedown(root, entity_id, reason=reason)
    return {
        "target_id": impact.target_id,
        "target_type": impact.target_type,
        "affected_ids": list(impact.affected_ids),
        "changed_paths": [str(item) for item in impact.changed_paths],
        "source_sha256": impact.source_sha256,
        "status": impact.status,
    }


def takedown_command(
    path: Path,
    entity_id: str,
    reason: str,
    *,
    expected_source_sha256: str | None = None,
) -> CommandResult:
    root = require_local_repository(path)
    changed = takedown(
        root,
        entity_id,
        reason=reason,
        expected_source_sha256=expected_source_sha256,
    )
    return CommandResult(
        status="noop" if changed.status == "noop" else "succeeded",  # type: ignore[arg-type]
        result={
            "target_id": changed.target_id,
            "reason": reason,
            "changed_paths": [str(item) for item in changed.changed_paths],
        },
    )


def _git_config(root: Path, key: str) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "config", "--get", key],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
        shell=False,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None
