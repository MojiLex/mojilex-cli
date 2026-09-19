from pathlib import Path

import pytest

import mojilex_cli.dataset.validation as validation
from mojilex_cli.dataset import load_validated_dataset, validate_dataset
from test_dataset_helpers import write_fixture
from test_tracked_transaction_artifacts import (
    test_tracked_transaction_artifact_is_rejected_before_dataset_load as tracked_guard,
)


@pytest.mark.parametrize("strict", [False, True])
def test_load_validated_dataset_loads_once_and_validates_same_snapshot(
    tmp_path, monkeypatch, strict
) -> None:
    write_fixture(tmp_path)
    real_load = validation.load_dataset
    real_validate = validation.validate_snapshot
    loaded = []
    validated = []

    def load(root):
        snapshot = real_load(root)
        loaded.append(snapshot)
        return snapshot

    def validate(snapshot, **options):
        validated.append((snapshot, options))
        return real_validate(snapshot, **options)

    monkeypatch.setattr(validation, "load_dataset", load)
    monkeypatch.setattr(validation, "validate_snapshot", validate)
    snapshot, report = load_validated_dataset(tmp_path, strict=strict)
    assert len(loaded) == 1
    assert snapshot is loaded[0]
    assert len(validated) == 1
    assert validated[0][0] is snapshot
    assert validated[0][1] == {
        "canonical": strict,
        "schemas": strict,
        "repository_files": strict,
    }
    assert report == real_validate(snapshot, **validated[0][1])


@pytest.mark.parametrize("tree", [".mojilex", ".mojilex-atomic-write"])
def test_load_validated_dataset_blocks_tracked_transaction_before_loading(
    tmp_path, monkeypatch, tree
) -> None:
    # Exercise the same real Git index guard through the snapshot-returning API.
    import test_tracked_transaction_artifacts as guards

    def report_only(root, **options):
        snapshot, report = load_validated_dataset(root, **options)
        assert snapshot is None
        return report

    monkeypatch.setattr(guards, "validate_dataset", report_only)
    tracked_guard(tmp_path, monkeypatch, tree)


def test_load_validated_dataset_reports_load_error_without_snapshot(tmp_path) -> None:
    snapshot, report = load_validated_dataset(tmp_path)
    assert snapshot is None
    assert [issue.code for issue in report.issues] == ["LOAD"]
    assert report.issues[0].path == str(tmp_path)
    assert report == validate_dataset(tmp_path)


def test_load_validated_dataset_preserves_invalid_integrity_report(tmp_path) -> None:
    write_fixture(tmp_path)
    for path in (tmp_path / "data" / "telegram" / "emojis").rglob("*.jsonl"):
        path.write_bytes(b"")
    snapshot, report = load_validated_dataset(tmp_path, strict=False)
    assert snapshot is not None
    assert not report.valid
    assert "DANGLING" in {issue.code for issue in report.issues}
    assert report == validate_dataset(tmp_path, strict=False)


def test_validate_dataset_delegates_to_single_load_api(tmp_path, monkeypatch) -> None:
    expected = validation.ValidationReport(())
    calls = []

    def load(root: str | Path, *, strict: bool):
        calls.append((root, strict))
        return None, expected

    monkeypatch.setattr(validation, "load_validated_dataset", load)
    assert validate_dataset(tmp_path, strict=False) is expected
    assert calls == [(tmp_path, False)]
