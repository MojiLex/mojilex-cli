from dataclasses import replace
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from mojilex_cli.dataset import transaction as tx


def _entry(index=0):
    return tx.TransactionEntry(
        index, PurePosixPath(f"data/emoji-{index}.jsonl"), "a" * 64, "b" * 64, 100, 200
    )


def _transaction(tmp_path, entries):
    return tx.DurableDatasetTransaction(tmp_path, entries, SimpleNamespace(is_locked=True))


@pytest.mark.parametrize(
    "changes",
    [
        {"relative": PurePosixPath("data/foreign.jsonl")},
        {"old_sha256": "c" * 64},
        {"new_sha256": "c" * 64},
        {"old_size": 101},
        {"new_size": 201},
    ],
)
def test_foreign_entry_with_same_index_is_rejected_before_file_access(
    tmp_path, monkeypatch, changes
):
    original = _entry()
    transaction = _transaction(tmp_path, [original])
    foreign = replace(original, **changes)

    def unexpected(*args, **kwargs):
        pytest.fail("foreign entry reached filesystem validation")

    monkeypatch.setattr(tx, "_validate_payload_file", unexpected)
    monkeypatch.setattr(tx, "safe_destination", unexpected)
    with pytest.raises(tx.AtomicWriteError, match="invalid staged transaction entry"):
        transaction.staged_path(foreign)
    with pytest.raises(tx.AtomicWriteError, match="invalid dataset transaction entry"):
        transaction.sync_target_parent(foreign)


def test_entry_checks_scale_linearly_and_preserve_value_equality(tmp_path, monkeypatch):
    entries = tuple(_entry(index) for index in range(2000))
    transaction = _transaction(tmp_path, entries)
    comparisons = 0
    real_equal = tx.TransactionEntry.__eq__
    payload_checks = []
    synced_parents = []

    def count_equal(self, other):
        nonlocal comparisons
        comparisons += 1
        return real_equal(self, other)

    monkeypatch.setattr(tx.TransactionEntry, "__eq__", count_equal)
    monkeypatch.setattr(tx, "_validate_payload_file", lambda *args: payload_checks.append(args))
    monkeypatch.setattr(tx, "safe_destination", lambda root, relative: root / relative)
    monkeypatch.setattr(tx, "_fsync_directory", synced_parents.append)
    for original in entries:
        # The old tuple accepted equal values, not just the original object.
        entry = replace(original)
        staged = transaction.staged_path(entry)
        assert payload_checks[-1] == (staged, entry.new_sha256, entry.new_size)
        transaction.sync_target_parent(entry)

    assert len(payload_checks) == len(synced_parents) == len(entries)
    assert comparisons <= 4 * len(entries)


def test_original_entry_is_accepted_and_delete_has_no_staged_payload(tmp_path, monkeypatch):
    original = _entry()
    deleted = replace(_entry(1), new_sha256=None, new_size=None)
    transaction = _transaction(tmp_path, [original, deleted])
    checked = []
    monkeypatch.setattr(tx, "_validate_payload_file", lambda *args: checked.append(args))
    monkeypatch.setattr(tx, "safe_destination", lambda root, relative: root / relative)
    monkeypatch.setattr(tx, "_fsync_directory", lambda path: None)

    assert transaction.staged_path(original).name == ".mojilex-stage-000000"
    transaction.sync_target_parent(original)
    transaction.sync_target_parent(deleted)
    with pytest.raises(tx.AtomicWriteError, match="invalid staged transaction entry"):
        transaction.staged_path(deleted)
    assert len(checked) == 1
