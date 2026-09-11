import asyncio
import os

import pytest

from mojilex_cli.dataset.staging import AtomicDatasetWriter, AtomicWriteError


def test_atomic_writer_rejects_path_traversal_before_writing(tmp_path) -> None:
    writer = AtomicDatasetWriter(tmp_path)
    with pytest.raises(ValueError, match="unsafe dataset path"):
        writer.stage_bytes("../outside.json", b"{}\n")
    assert not (tmp_path.parent / "outside.json").exists()


def test_atomic_writer_rolls_back_every_applied_file_on_failure(tmp_path, monkeypatch) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"first-before")
    second.write_bytes(b"second-before")
    writer = AtomicDatasetWriter(tmp_path)
    writer.stage_bytes("first.json", b"first-after")
    writer.stage_bytes("second.json", b"second-after")
    real_replace = os.replace
    stage_replacements = 0

    def fail_second_staged_replace(source, destination) -> None:
        nonlocal stage_replacements
        if ".mojilex-stage-" in str(source):
            stage_replacements += 1
            if stage_replacements == 2:
                raise OSError("synthetic replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr("mojilex_cli.dataset.staging.os.replace", fail_second_staged_replace)
    with pytest.raises(AtomicWriteError, match="rolled back"):
        writer.commit()
    assert first.read_bytes() == b"first-before"
    assert second.read_bytes() == b"second-before"


@pytest.mark.parametrize(
    "interruption",
    [KeyboardInterrupt("synthetic interrupt"), asyncio.CancelledError("synthetic cancellation")],
    ids=["keyboard-interrupt", "cancelled-error"],
)
def test_atomic_writer_rolls_back_before_reraising_base_exception(
    tmp_path, monkeypatch, interruption
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b"first-before")
    second.write_bytes(b"second-before")
    writer = AtomicDatasetWriter(tmp_path)
    writer.stage_bytes("first.json", b"first-after")
    writer.stage_bytes("second.json", b"second-after")
    real_replace = os.replace
    stage_replacements = 0

    def interrupt_second_staged_replace(source, destination) -> None:
        nonlocal stage_replacements
        if ".mojilex-stage-" in str(source):
            stage_replacements += 1
            if stage_replacements == 2:
                raise interruption
        real_replace(source, destination)

    monkeypatch.setattr("mojilex_cli.dataset.staging.os.replace", interrupt_second_staged_replace)
    with pytest.raises(type(interruption), match="synthetic"):
        writer.commit()
    assert first.read_bytes() == b"first-before"
    assert second.read_bytes() == b"second-before"
