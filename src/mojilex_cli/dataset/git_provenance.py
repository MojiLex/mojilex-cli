from __future__ import annotations

import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

_SOURCE_ROOTS = (
    "analysis-profiles",
    "data",
    "platforms",
    "rights",
    "schemas",
    "taxonomy",
    "tombstones",
)
_REQUIRED_SOURCE_FILES = {"dataset.json"}
_TOOL_SOURCE_ROOT = "src/mojilex_cli"
_TOOL_REQUIRED_SOURCE_FILES = {"pyproject.toml"}
_TOOL_SOURCE_SUFFIXES = {".json", ".py", ".typed"}


@dataclass(frozen=True)
class GitSourceProvenance:
    commit: str
    object_format: str


def _git(root: Path, *arguments: str, text: bool = False) -> subprocess.CompletedProcess[Any]:
    command = [
        "git",
        "-c",
        f"safe.directory={root.as_posix()}",
        "-C",
        str(root),
        *arguments,
    ]
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=text,
        encoding="utf-8" if text else None,
        errors="strict" if text else None,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
        timeout=30,
        shell=False,
    )


def _git_text(root: Path, *arguments: str) -> str:
    result = _git(root, *arguments, text=True)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f"Git provenance check failed: {detail or 'git command failed'}")
    return cast(str, result.stdout).strip()


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse_flag)


def _is_source_path(relative: str) -> bool:
    path = PurePosixPath(relative)
    parts = path.parts
    if relative in _REQUIRED_SOURCE_FILES:
        return True
    if len(parts) == 2 and parts[0] == "analysis-profiles" and path.suffix == ".json":
        return True
    if len(parts) == 2 and parts[0] == "platforms" and path.suffix == ".json":
        return True
    if relative == "rights/profiles.json":
        return True
    if len(parts) == 3 and parts[:2] == ("taxonomy", "v1") and path.suffix == ".json":
        return True
    if parts[:2] == ("schemas", "v1") and path.suffix == ".json":
        return True
    if (
        len(parts) == 4
        and parts[:3] == ("schemas", "distribution", "v1")
        and path.suffix == ".json"
    ):
        return True
    if len(parts) == 3 and parts[0] == "tombstones" and path.suffix == ".json":
        return True
    if len(parts) == 6 and parts[0] == "data" and parts[2] == "collections":
        return parts[-1] in {"collection.json", "memberships.jsonl"}
    if len(parts) == 5 and parts[0] == "data" and parts[2] == "emojis":
        return path.suffix == ".jsonl"
    return (
        len(parts) == 5 and parts[:3] == ("data", "relations", "visual") and path.suffix == ".jsonl"
    )


def _walk_source_candidates(root: Path) -> set[str]:
    candidates: set[str] = set()
    for relative in _REQUIRED_SOURCE_FILES:
        path = root / PurePosixPath(relative)
        if _is_link_or_reparse(path):
            raise ValueError(f"source path must not be a link or reparse point: {path}")
        if path.exists():
            if not path.is_file():
                raise ValueError(f"source path must be a regular file: {path}")
            candidates.add(relative)
    pending = [root / name for name in _SOURCE_ROOTS]
    while pending:
        current = pending.pop()
        if not current.exists():
            continue
        if _is_link_or_reparse(current):
            raise ValueError(f"source path must not be a link or reparse point: {current}")
        if current.is_dir():
            pending.extend(current.iterdir())
            continue
        if not current.is_file():
            raise ValueError(f"source path must be a regular file: {current}")
        relative = current.relative_to(root).as_posix()
        if _is_source_path(relative):
            candidates.add(relative)
    return candidates


def _committed_source_entries(root: Path, revision: str) -> dict[str, tuple[str, str]]:
    result = _git(root, "ls-tree", "-r", "-z", revision, "--", *_SOURCE_ROOTS)
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"cannot enumerate source paths at {revision}: {detail}")
    entries: dict[str, tuple[str, str]] = {}
    for raw_entry in result.stdout.split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ")
        relative = raw_path.decode("utf-8", errors="strict")
        if _is_source_path(relative):
            entries[relative] = (mode, f"{object_type}:{object_id}")
    for relative in _REQUIRED_SOURCE_FILES:
        if relative in entries:
            continue
        result = _git(root, "ls-tree", "-z", revision, "--", relative)
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"cannot inspect source path {relative!r}: {detail}")
        raw_entry = result.stdout.rstrip(b"\0")
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ")
        path_from_git = raw_path.decode("utf-8", errors="strict")
        entries[path_from_git] = (mode, f"{object_type}:{object_id}")
    return entries


def _is_tool_source_path(relative: str) -> bool:
    if relative in _TOOL_REQUIRED_SOURCE_FILES:
        return True
    path = PurePosixPath(relative)
    return (
        path.parts[:2] == ("src", "mojilex_cli")
        and "__pycache__" not in path.parts
        and path.suffix in _TOOL_SOURCE_SUFFIXES
    )


def _walk_tool_source_candidates(root: Path) -> set[str]:
    candidates: set[str] = set()
    for relative in _TOOL_REQUIRED_SOURCE_FILES:
        path = root / PurePosixPath(relative)
        if _is_link_or_reparse(path):
            raise ValueError(f"tool source path must not be a link or reparse point: {path}")
        if path.exists():
            if not path.is_file():
                raise ValueError(f"tool source path must be a regular file: {path}")
            candidates.add(relative)

    pending = [root / PurePosixPath(_TOOL_SOURCE_ROOT)]
    while pending:
        current = pending.pop()
        if _is_link_or_reparse(current):
            raise ValueError(f"tool source path must not be a link or reparse point: {current}")
        if not current.exists():
            continue
        if current.is_dir():
            if current.name != "__pycache__":
                pending.extend(current.iterdir())
            continue
        if not current.is_file():
            raise ValueError(f"tool source path must be a regular file: {current}")
        relative = current.relative_to(root).as_posix()
        if _is_tool_source_path(relative):
            candidates.add(relative)
    return candidates


def _committed_tool_source_entries(root: Path, revision: str) -> dict[str, tuple[str, str]]:
    result = _git(root, "ls-tree", "-r", "-z", revision, "--", _TOOL_SOURCE_ROOT)
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"cannot enumerate tool source paths at {revision}: {detail}")
    entries: dict[str, tuple[str, str]] = {}
    for raw_entry in result.stdout.split(b"\0"):
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ")
        relative = raw_path.decode("utf-8", errors="strict")
        if _is_tool_source_path(relative):
            entries[relative] = (mode, f"{object_type}:{object_id}")
    for relative in _TOOL_REQUIRED_SOURCE_FILES:
        result = _git(root, "ls-tree", "-z", revision, "--", relative)
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"cannot inspect tool source path {relative!r}: {detail}")
        raw_entry = result.stdout.rstrip(b"\0")
        if not raw_entry:
            continue
        metadata, raw_path = raw_entry.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ")
        path_from_git = raw_path.decode("utf-8", errors="strict")
        entries[path_from_git] = (mode, f"{object_type}:{object_id}")
    return entries


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.realpath(left)) == os.path.normcase(os.path.realpath(right))


def resolve_head_revision(root: Path) -> GitSourceProvenance:
    root = root.resolve()
    top = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
    if not _same_path(root, top):
        raise ValueError(f"dataset root must be the exact Git worktree top: {top}")
    object_format = _git_text(root, "rev-parse", "--show-object-format")
    if object_format not in {"sha1", "sha256"}:
        raise ValueError(f"unsupported Git object format: {object_format!r}")
    commit = _git_text(root, "rev-parse", "--verify", "HEAD^{commit}").lower()
    expected_length = 40 if object_format == "sha1" else 64
    if not re.fullmatch(rf"[0-9a-f]{{{expected_length}}}", commit):
        raise ValueError("HEAD did not resolve to a full Git commit object ID")
    return GitSourceProvenance(commit=commit, object_format=object_format)


def tool_source_checkout_root() -> Path | None:
    package_root = Path(__file__).resolve().parents[1]
    candidate = package_root.parent.parent
    expected_package_root = candidate / "src" / "mojilex_cli"
    if not _same_path(package_root, expected_package_root):
        return None
    if not (candidate / ".git").exists():
        return None
    return candidate


def verify_release_source(root: Path, revision: str) -> GitSourceProvenance:
    root = root.resolve()
    top = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
    if not _same_path(root, top):
        raise ValueError(f"dataset root must be the exact Git worktree top: {top}")
    object_format = _git_text(root, "rev-parse", "--show-object-format")
    if object_format not in {"sha1", "sha256"}:
        raise ValueError(f"unsupported Git object format: {object_format!r}")
    expected_length = 40 if object_format == "sha1" else 64
    if not re.fullmatch(rf"[0-9a-f]{{{expected_length}}}", revision):
        raise ValueError(f"revision must be a full lowercase {object_format} Git object ID")
    resolved = _git_text(
        root,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{revision}^{{commit}}",
    ).lower()
    if resolved != revision:
        raise ValueError("supplied revision does not resolve to that exact commit")

    current_paths = _walk_source_candidates(root)
    committed_entries = _committed_source_entries(root, revision)
    committed_paths = set(committed_entries)
    missing = sorted(committed_paths - current_paths)
    untracked = sorted(current_paths - committed_paths)
    if missing:
        raise ValueError(f"source paths are missing from the worktree: {missing!r}")
    if untracked:
        raise ValueError(f"source paths are absent from commit {revision}: {untracked!r}")

    for relative in sorted(current_paths, key=lambda item: item.encode("utf-8")):
        path = root / PurePosixPath(relative)
        if _is_link_or_reparse(path) or not path.is_file():
            raise ValueError(f"source path must be a regular non-link file: {path}")
        mode, typed_object = committed_entries[relative]
        object_type, object_id = typed_object.split(":", 1)
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(f"source path is not a regular Git blob at {revision}: {relative}")
        blob = _git(root, "cat-file", "blob", object_id)
        if blob.returncode:
            detail = blob.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"cannot read committed source path {relative!r}: {detail}")
        if path.read_bytes() != blob.stdout:
            raise ValueError(f"source path differs from commit {revision}: {relative}")
    return GitSourceProvenance(commit=revision, object_format=object_format)


def verify_tool_source(root: Path, revision: str) -> GitSourceProvenance:
    root = root.resolve()
    top = Path(_git_text(root, "rev-parse", "--show-toplevel")).resolve()
    if not _same_path(root, top):
        raise ValueError(f"tool root must be the exact Git worktree top: {top}")
    object_format = _git_text(root, "rev-parse", "--show-object-format")
    if object_format not in {"sha1", "sha256"}:
        raise ValueError(f"unsupported Git object format: {object_format!r}")
    expected_length = 40 if object_format == "sha1" else 64
    if not re.fullmatch(rf"[0-9a-f]{{{expected_length}}}", revision):
        raise ValueError(f"revision must be a full lowercase {object_format} Git object ID")
    resolved = _git_text(
        root,
        "rev-parse",
        "--verify",
        "--end-of-options",
        f"{revision}^{{commit}}",
    ).lower()
    if resolved != revision:
        raise ValueError("supplied revision does not resolve to that exact commit")

    current_paths = _walk_tool_source_candidates(root)
    committed_entries = _committed_tool_source_entries(root, revision)
    committed_paths = set(committed_entries)
    missing = sorted(committed_paths - current_paths)
    untracked = sorted(current_paths - committed_paths)
    if missing:
        raise ValueError(f"tool source paths are missing from the worktree: {missing!r}")
    if untracked:
        raise ValueError(f"tool source paths are absent from commit {revision}: {untracked!r}")

    for relative in sorted(current_paths, key=lambda item: item.encode("utf-8")):
        path = root / PurePosixPath(relative)
        if _is_link_or_reparse(path) or not path.is_file():
            raise ValueError(f"tool source path must be a regular non-link file: {path}")
        mode, typed_object = committed_entries[relative]
        object_type, object_id = typed_object.split(":", 1)
        if object_type != "blob" or mode not in {"100644", "100755"}:
            raise ValueError(
                f"tool source path is not a regular Git blob at {revision}: {relative}"
            )
        blob = _git(root, "cat-file", "blob", object_id)
        if blob.returncode:
            detail = blob.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"cannot read committed tool source path {relative!r}: {detail}")
        if path.read_bytes() != blob.stdout:
            raise ValueError(f"tool source path differs from commit {revision}: {relative}")
    return GitSourceProvenance(commit=revision, object_format=object_format)
