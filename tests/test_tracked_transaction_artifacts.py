from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

import mojilex_cli.dataset.validation as validation_module
from mojilex_cli.dataset import validate_dataset, validate_snapshot
from test_dataset_helpers import write_fixture


def _initialize_git_repository(root: Path) -> None:
    (root / ".gitignore").write_text(
        "/.mojilex/\n/.mojilex-atomic-write/\n",
        encoding="utf-8",
        newline="",
    )
    subprocess.run(
        ["git", "init", "--quiet", str(root)],
        check=True,
        capture_output=True,
        shell=False,
    )


def _configure_git_identity(root: Path) -> None:
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "audit@example.invalid"],
        check=True,
        capture_output=True,
        shell=False,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "MojiLex audit"],
        check=True,
        capture_output=True,
        shell=False,
    )


def test_ignored_runtime_transaction_trees_are_not_scanned(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path)
    _initialize_git_repository(tmp_path)
    runtime_payload = tmp_path / ".mojilex" / "locks" / "active.lock"
    runtime_payload.parent.mkdir(parents=True, exist_ok=True)
    runtime_payload.write_bytes(b"ignored\0github_pat_" + b"x" * 40)
    legacy_payload = tmp_path / ".mojilex-atomic-write" / "oversized-control"
    legacy_payload.parent.mkdir()
    legacy_payload.write_bytes(b"ignored\0")

    report = validate_snapshot(snapshot, repository_files=True)

    assert report.valid, report.issues


def test_untracked_known_work_tree_is_not_scanned(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path)
    _initialize_git_repository(tmp_path)
    payload = tmp_path / "work" / "addendum-dist" / "local-preview.png"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"ignored\0github_pat_" + b"x" * 40)

    report = validate_snapshot(snapshot, repository_files=True)

    assert report.valid, report.issues


@pytest.mark.parametrize("tree", [".mojilex", ".mojilex-atomic-write"])
def test_tracked_transaction_artifact_is_rejected_before_dataset_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tree: str,
) -> None:
    write_fixture(tmp_path)
    _initialize_git_repository(tmp_path)
    payload = tmp_path / tree / "transaction" / "payload"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"must-not-be-read\0")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--force", "--", payload.relative_to(tmp_path)],
        check=True,
        capture_output=True,
        shell=False,
    )

    def fail_if_dataset_is_loaded(_root: str | Path):
        raise AssertionError("tracked transaction payload reached dataset recovery")

    monkeypatch.setattr(validation_module, "load_dataset", fail_if_dataset_is_loaded)

    report = validate_dataset(tmp_path, strict=True)

    artifacts = [issue for issue in report.issues if issue.code == "LOCAL_ARTIFACT"]
    assert [issue.path for issue in artifacts] == [payload.relative_to(tmp_path).as_posix()]
    assert "tracked runtime transaction artifacts" in artifacts[0].message


def test_repository_scan_does_not_inherit_ancestor_git_excludes(tmp_path: Path) -> None:
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    (ancestor / ".gitignore").write_text("/dataset/\n", encoding="utf-8", newline="")
    subprocess.run(
        ["git", "init", "--quiet", str(ancestor)],
        check=True,
        capture_output=True,
        shell=False,
    )
    dataset = ancestor / "dataset"
    snapshot = write_fixture(dataset)
    (dataset / "leaked.png").write_bytes(b"not really an image")
    (dataset / "notes.txt").write_text(
        "synthetic credential 123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi",
        encoding="utf-8",
    )

    report = validate_snapshot(snapshot, repository_files=True)

    assert {"NO_MEDIA", "SECRET"}.issubset({issue.code for issue in report.issues})


def test_force_tracked_ignored_generated_tree_is_rejected(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path)
    _initialize_git_repository(tmp_path)
    payload = tmp_path / "dist" / "payload.txt"
    payload.parent.mkdir()
    payload.write_text("generated\n", encoding="utf-8", newline="")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--force", "--", payload.relative_to(tmp_path)],
        check=True,
        capture_output=True,
        shell=False,
    )

    report = validate_snapshot(snapshot, repository_files=True)

    assert any(
        issue.code == "LOCAL_ARTIFACT" and issue.path == "dist/payload.txt"
        for issue in report.issues
    )


@pytest.mark.parametrize(
    ("mode", "expected_code"),
    [("120000", "SYMLINK"), ("160000", "GITLINK")],
)
def test_forbidden_git_index_modes_are_rejected(
    tmp_path: Path, mode: str, expected_code: str
) -> None:
    write_fixture(tmp_path)
    _initialize_git_repository(tmp_path)
    _configure_git_identity(tmp_path)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "--quiet", "--allow-empty", "-m", "base"],
        check=True,
        capture_output=True,
        shell=False,
    )
    if mode == "120000":
        object_id = (
            subprocess.run(
                ["git", "-C", str(tmp_path), "hash-object", "-w", "--stdin"],
                input=b"target\n",
                check=True,
                capture_output=True,
                shell=False,
            )
            .stdout.decode("ascii")
            .strip()
        )
    else:
        object_id = subprocess.run(
            ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            shell=False,
            text=True,
        ).stdout.strip()
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "update-index",
            "--add",
            "--cacheinfo",
            f"{mode},{object_id},nested-entry",
        ],
        check=True,
        capture_output=True,
        shell=False,
    )

    report = validate_dataset(tmp_path, strict=False)

    assert any(
        issue.code == expected_code and issue.path == "nested-entry" for issue in report.issues
    )


def test_nested_git_repository_marker_is_rejected(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path)
    _initialize_git_repository(tmp_path)
    nested = tmp_path / "nested"
    nested.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", str(nested)],
        check=True,
        capture_output=True,
        shell=False,
    )

    report = validate_snapshot(snapshot, repository_files=True)

    assert any(
        issue.code == "NESTED_GIT" and issue.path == "nested/.git" for issue in report.issues
    )


def test_linked_worktree_git_file_preserves_transaction_preflight(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    _initialize_git_repository(base)
    _configure_git_identity(base)
    subprocess.run(
        ["git", "-C", str(base), "commit", "--quiet", "--allow-empty", "-m", "base"],
        check=True,
        capture_output=True,
        shell=False,
    )
    worktree = tmp_path / "worktree"
    subprocess.run(
        ["git", "-C", str(base), "worktree", "add", "--quiet", "-b", "audit", str(worktree)],
        check=True,
        capture_output=True,
        shell=False,
    )
    write_fixture(worktree)
    (worktree / ".gitignore").write_text(
        "/.mojilex/\n/.mojilex-atomic-write/\n",
        encoding="utf-8",
        newline="",
    )
    payload = worktree / ".mojilex" / "transactions" / "payload"
    payload.parent.mkdir(parents=True, exist_ok=True)
    payload.write_bytes(b"tracked\0")
    subprocess.run(
        ["git", "-C", str(worktree), "add", "--force", "--", payload.relative_to(worktree)],
        check=True,
        capture_output=True,
        shell=False,
    )

    report = validate_dataset(worktree, strict=False)

    assert (worktree / ".git").is_file()
    assert any(
        issue.code == "LOCAL_ARTIFACT" and issue.path == ".mojilex/transactions/payload"
        for issue in report.issues
    )


@pytest.mark.parametrize("tracked", [False, True])
def test_large_files_are_streamed_and_secret_scanned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tracked: bool,
) -> None:
    snapshot = write_fixture(tmp_path)
    _initialize_git_repository(tmp_path)
    payload = tmp_path / "large-notes.txt"
    token = b"123456789" + b":" + b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
    with payload.open("wb") as target:
        target.write(b"x" * (5 * 1024 * 1024 + 64 * 1024 - 6))
        target.write(b"\n" + token + b"\n")
    if tracked:
        subprocess.run(
            ["git", "-C", str(tmp_path), "add", "--", payload.relative_to(tmp_path)],
            check=True,
            capture_output=True,
            shell=False,
        )
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path: Path) -> bytes:
        if path == payload:
            raise AssertionError("large repository files must not be read wholesale")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    report = validate_snapshot(snapshot, repository_files=True)

    assert any(
        issue.code == "SECRET" and issue.path == "large-notes.txt" for issue in report.issues
    )


@pytest.mark.skipif(os.name != "nt", reason="NTFS junction regression is Windows-specific")
def test_untracked_junction_is_rejected_without_following_target(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("outside remains untouched\n", encoding="utf-8", newline="")
    snapshot = write_fixture(dataset)
    _initialize_git_repository(dataset)
    junction = dataset / "junction"
    subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        check=True,
        capture_output=True,
        shell=False,
    )

    report = validate_snapshot(snapshot, repository_files=True)

    assert any(issue.code == "SYMLINK" and issue.path == "junction" for issue in report.issues)
    assert sentinel.read_text(encoding="utf-8") == "outside remains untouched\n"
