import stat
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

from mojilex_cli.dataset import validation


def _plain_file_tree(tmp_path, monkeypatch):
    target = tmp_path / "notes.txt"
    target.write_text("Synthetic safe repository notes.\n", encoding="utf-8")
    monkeypatch.setattr(validation, "_tracked_transaction_issues", lambda root: ())
    monkeypatch.setattr(validation, "_git_ls_files", lambda root, **kwargs: None)
    monkeypatch.setattr(validation, "_git_root_is_exact_worktree", lambda root: None)
    return target


def test_file_metadata_has_fresh_walk_and_final_checks_without_following_stat(
    tmp_path, monkeypatch
):
    target = _plain_file_tree(tmp_path, monkeypatch)
    real_lstat = Path.lstat
    real_stat = Path.stat
    lstats = []
    following_stats = []

    def record_lstat(path):
        if path == target:
            lstats.append(path)
        return real_lstat(path)

    def record_stat(path, *, follow_symlinks=True):
        if path == target and follow_symlinks:
            following_stats.append(path)
        return real_stat(path, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(Path, "lstat", record_lstat)
    monkeypatch.setattr(Path, "stat", record_stat)
    issues = []
    validation._validate_repository_files(tmp_path, issues)

    assert issues == []
    assert len(lstats) == 2
    assert following_stats == []


@pytest.mark.parametrize(
    ("mode", "attributes"),
    [(stat.S_IFLNK, 0), (stat.S_IFREG, 0x400)],
    ids=["symlink", "non-symlink-reparse"],
)
def test_final_file_check_rejects_link_or_reparse_introduced_after_walk(
    tmp_path, monkeypatch, mode, attributes
):
    target = _plain_file_tree(tmp_path, monkeypatch)
    real_lstat = Path.lstat
    target_checks = 0
    scanned = []

    def changed_metadata(path):
        nonlocal target_checks
        if path == target:
            target_checks += 1
            if target_checks == 2:
                return SimpleNamespace(st_mode=mode, st_file_attributes=attributes)
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", changed_metadata)
    monkeypatch.setattr(
        validation, "_scan_repository_file", lambda path, relative, issues: scanned.append(path)
    )
    issues = []
    validation._validate_repository_files(tmp_path, issues)

    assert [(issue.code, issue.path) for issue in issues] == [("SYMLINK", "notes.txt")]
    assert target_checks == 2
    assert target not in scanned


@pytest.mark.parametrize("missing_error", [FileNotFoundError, NotADirectoryError])
def test_tracked_file_disappearing_after_walk_is_not_scanned(tmp_path, monkeypatch, missing_error):
    target = _plain_file_tree(tmp_path, monkeypatch)
    monkeypatch.setattr(
        validation,
        "_git_ls_files",
        lambda root, **kwargs: (PurePosixPath("notes.txt"),),
    )
    monkeypatch.setattr(validation, "_git_root_is_exact_worktree", lambda root: True)
    real_lstat = Path.lstat
    target_checks = 0
    scanned = []

    def missing_after_walk(path):
        nonlocal target_checks
        if path == target:
            target_checks += 1
            if target_checks == 2:
                raise missing_error("synthetic file disappeared")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", missing_after_walk)
    monkeypatch.setattr(
        validation, "_scan_repository_file", lambda path, relative, issues: scanned.append(path)
    )
    issues = []
    validation._validate_repository_files(tmp_path, issues)

    assert issues == []
    assert target_checks == 2
    assert target not in scanned


def test_final_file_metadata_permission_error_still_fails_closed(tmp_path, monkeypatch):
    target = _plain_file_tree(tmp_path, monkeypatch)
    real_lstat = Path.lstat
    target_checks = 0

    def denied_after_walk(path):
        nonlocal target_checks
        if path == target:
            target_checks += 1
            if target_checks == 2:
                raise PermissionError("synthetic metadata denied")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", denied_after_walk)
    with pytest.raises(PermissionError, match="metadata denied"):
        validation._validate_repository_files(tmp_path, [])
