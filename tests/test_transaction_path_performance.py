import os
from collections import Counter
from pathlib import Path, PurePosixPath

import pytest

from mojilex_cli.dataset import layout, transaction


def test_bulk_path_identities_do_not_repeat_filesystem_resolution(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    paths = [PurePosixPath(f"data/telegram/emojis/aa/{index:06x}.jsonl") for index in range(128)]
    destinations = [root.joinpath(*relative.parts) for relative in paths]
    destinations[0].parent.mkdir(parents=True)
    for path in destinations[::2]:
        path.write_bytes(b"{}\n")
    expected = [os.path.normcase(str(path.resolve())) for path in destinations]
    real_resolve = Path.resolve
    real_check = layout.is_link_or_reparse_point
    resolutions = []
    checks = Counter()

    def count_resolve(path, *args, **kwargs):
        resolutions.append(path)
        return real_resolve(path, *args, **kwargs)

    def count_check(path):
        checks[path] += 1
        return real_check(path)

    monkeypatch.setattr(Path, "resolve", count_resolve)
    monkeypatch.setattr(layout, "is_link_or_reparse_point", count_check)

    actual = [transaction.transaction_path_identity(root, relative) for relative in paths]

    assert actual == expected
    # Permit one root and one target resolution, for existing and absent files.
    assert len(resolutions) <= 2 * len(paths)
    assert all(checks[path] == 1 for path in destinations)
    # Shared ancestors still get checked afresh; there is no stale safety cache.
    assert checks[destinations[0].parent] == len(paths)


@pytest.mark.parametrize("component", ["root", "ancestor", "target"])
def test_later_path_check_rejects_new_reparse_point(tmp_path, monkeypatch, component):
    root = tmp_path.resolve()
    relative = PurePosixPath("data/telegram/emojis/aa/abcdef.jsonl")
    destination = root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"{}\n")
    transaction.transaction_path_identity(root, relative)
    changed_path = {
        "root": root,
        "ancestor": destination.parent,
        "target": destination,
    }[component]
    real_check = layout.is_link_or_reparse_point

    def new_reparse(path):
        return path == changed_path or real_check(path)

    monkeypatch.setattr(layout, "is_link_or_reparse_point", new_reparse)

    with pytest.raises(transaction.AtomicWriteError, match="unsafe path"):
        transaction.transaction_path_identity(root, relative)


def test_bulk_precondition_reads_every_file_and_detects_same_metadata_edit(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    expected = {PurePosixPath("dataset.json"): b"{}\n"}
    expected.update(
        (PurePosixPath(f"data/telegram/emojis/aa/{index:06x}.jsonl"), b"old\n")
        for index in range(128)
    )
    for relative, payload in expected.items():
        destination = root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    real_hash = transaction._file_sha256
    real_destination = transaction.safe_destination
    hashed = []
    checked = []

    def record_destination(root, relative, **kwargs):
        checked.append(relative)
        return real_destination(root, relative, **kwargs)

    def record_hash(path):
        assert root.joinpath(*checked[-1].parts) == path
        hashed.append(path)
        return real_hash(path)

    monkeypatch.setattr(transaction, "safe_destination", record_destination)
    monkeypatch.setattr(transaction, "_file_sha256", record_hash)
    transaction._verify_dataset_precondition(root, expected)
    assert set(hashed) == {root.joinpath(*relative.parts) for relative in expected}
    assert Counter(checked) == Counter({relative: 1 for relative in expected})
    changed = root.joinpath(*next(reversed(expected)).parts)
    before = changed.stat()
    changed.write_bytes(b"new\n")
    os.utime(changed, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(transaction.AtomicWriteError, match="changed since the snapshot"):
        transaction._verify_dataset_precondition(root, expected)


@pytest.mark.parametrize("component", ["root", "ancestor", "target"])
def test_precondition_rechecks_reparse_after_tree_scan(tmp_path, monkeypatch, component):
    root = tmp_path.resolve()
    relative = PurePosixPath("data/telegram/emojis/aa/abcdef.jsonl")
    target = root.joinpath(*relative.parts)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"{}\n")
    (root / "dataset.json").write_bytes(b"{}\n")
    expected = {PurePosixPath("dataset.json"): b"{}\n", relative: b"{}\n"}
    changed = {"root": root, "ancestor": target.parent, "target": target}[component]
    real_scan = transaction._current_canonical_dataset_paths
    real_check = layout.is_link_or_reparse_point
    real_hash = transaction._file_sha256
    scanned = False
    hashed = []

    def scan_then_replace(root):
        nonlocal scanned
        result = real_scan(root)
        scanned = True
        return result

    def new_reparse(path):
        return (scanned and path == changed) or real_check(path)

    def record_hash(path):
        hashed.append(path)
        return real_hash(path)

    monkeypatch.setattr(transaction, "_current_canonical_dataset_paths", scan_then_replace)
    monkeypatch.setattr(layout, "is_link_or_reparse_point", new_reparse)
    monkeypatch.setattr(transaction, "_file_sha256", record_hash)

    with pytest.raises(transaction.AtomicWriteError, match="unsafe path"):
        transaction._verify_dataset_precondition(root, expected)

    assert scanned
    assert target not in hashed


@pytest.mark.skipif(os.name != "nt", reason="Windows filename alias regression")
def test_precondition_canonical_aliases_remain_rejected(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    (root / "dataset.json").write_bytes(b"{}\n")
    target = root / "data" / "A.json"
    target.parent.mkdir()
    target.write_bytes(b"{}\n")
    if not (target.parent / "a.json").exists():
        pytest.skip("test directory uses case-sensitive filenames")
    expected = dict.fromkeys(
        (PurePosixPath("dataset.json"), PurePosixPath("data/A.json"), PurePosixPath("data/a.json")),
        b"{}\n",
    )
    # A real scan already rejects this snapshot by its path set. Force that
    # earlier check to pass to exercise canonical identity detection as well.
    monkeypatch.setattr(transaction, "_current_canonical_dataset_paths", lambda root: set(expected))

    with pytest.raises(transaction.AtomicWriteError, match="alias the same target"):
        transaction._verify_dataset_precondition(root, expected)
