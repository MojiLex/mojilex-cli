import os
from pathlib import PurePosixPath

import pytest

from mojilex_cli.dataset import layout, staging, transaction


def _write_expected(root):
    expected = {
        PurePosixPath("dataset.json"): b"{}\n",
        PurePosixPath("data/telegram/emojis/aa/abcdef.jsonl"): b"old\n",
    }
    for relative, content in expected.items():
        target = root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return expected


def test_constructor_does_not_scan_expected_files_before_locked_validation(tmp_path, monkeypatch):
    expected = _write_expected(tmp_path)
    expected.update(
        (PurePosixPath(f"data/telegram/emojis/aa/{index:06x}.jsonl"), b"{}\n")
        for index in range(128)
    )
    checked = []
    real_check = layout.is_link_or_reparse_point

    def record_check(path):
        checked.append(path)
        return real_check(path)

    monkeypatch.setattr(layout, "is_link_or_reparse_point", record_check)
    writer = staging.AtomicDatasetWriter(tmp_path, expected_files=expected)

    assert writer._expected_files == expected
    assert not set(checked) & {tmp_path.joinpath(*relative.parts) for relative in expected}


@pytest.mark.parametrize(
    "relative", ["/absolute.json", "../escape.json", "data/../escape.json", "data\\..\\escape.json"]
)
def test_constructor_rejects_lexical_expected_path_escapes(tmp_path, relative):
    with pytest.raises(ValueError, match=r"unsafe dataset path|escapes dataset root"):
        staging.AtomicDatasetWriter(tmp_path, expected_files={PurePosixPath(relative): b"{}"})


@pytest.mark.skipif(os.name != "nt", reason="Windows drive-relative path regression")
def test_constructor_rejects_expected_path_on_another_drive(tmp_path):
    other_drive = "Z:" if tmp_path.drive.upper() != "Z:" else "Y:"
    with pytest.raises(ValueError, match="escapes dataset root"):
        staging.AtomicDatasetWriter(
            tmp_path, expected_files={PurePosixPath(f"{other_drive}/escape.json"): b"{}"}
        )


def test_commit_detects_expected_content_change_before_writing(tmp_path):
    expected = _write_expected(tmp_path)
    writer = staging.AtomicDatasetWriter(tmp_path, expected_files=expected)
    writer.stage_bytes("dataset.json", b"replacement\n")
    changed = tmp_path.joinpath(*next(reversed(expected)).parts)
    changed.write_bytes(b"new\n")

    with pytest.raises(staging.AtomicWriteError, match="changed since the snapshot"):
        writer.commit()

    assert (tmp_path / "dataset.json").read_bytes() == b"{}\n"
    assert changed.read_bytes() == b"new\n"
    assert not (tmp_path / transaction.TRANSACTION_DIRECTORY_NAME).exists()


def test_commit_rechecks_expected_path_reparse_before_reading(tmp_path, monkeypatch):
    expected = _write_expected(tmp_path)
    writer = staging.AtomicDatasetWriter(tmp_path, expected_files=expected)
    writer.stage_bytes("dataset.json", b"replacement\n")
    target = tmp_path.joinpath(*next(reversed(expected)).parts)
    real_check = layout.is_link_or_reparse_point
    real_hash = transaction._file_sha256
    hashed = []

    def changed_reparse(path):
        return path == target or real_check(path)

    def record_hash(path):
        hashed.append(path)
        return real_hash(path)

    monkeypatch.setattr(layout, "is_link_or_reparse_point", changed_reparse)
    monkeypatch.setattr(transaction, "_file_sha256", record_hash)
    with pytest.raises(staging.AtomicWriteError, match="unsafe path"):
        writer.commit()

    assert target not in hashed
    assert (tmp_path / "dataset.json").read_bytes() == b"{}\n"
    assert not (tmp_path / transaction.TRANSACTION_DIRECTORY_NAME).exists()


def test_constructor_retains_owned_snapshot_bytes(tmp_path):
    expected = _write_expected(tmp_path)
    mutable = bytearray(expected[PurePosixPath("dataset.json")])
    expected[PurePosixPath("dataset.json")] = mutable
    writer = staging.AtomicDatasetWriter(tmp_path, expected_files=expected)
    mutable[:] = b"changed after construction"
    writer.stage_bytes("dataset.json", b"replacement\n")

    assert writer.commit() == (PurePosixPath("dataset.json"),)
    assert (tmp_path / "dataset.json").read_bytes() == b"replacement\n"
