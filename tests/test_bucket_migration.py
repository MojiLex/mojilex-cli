from pathlib import PurePosixPath

import pytest

from mojilex_cli.dataset import load_dataset, validate_snapshot
from mojilex_cli.dataset.layout import collection_shard, emoji_bucket_path, legacy_bucket_path
from mojilex_cli.dataset.repository import DatasetLoadError
from mojilex_cli.dataset.staging import apply_snapshot
from mojilex_cli.pipeline.reapply import reapply_candidate
from mojilex_cli.pipeline.runner import _changed_paths
from test_dataset_helpers import write_fixture


def _legacy_fixture(root):
    write_fixture(root)
    current = next(root.glob("data/*/emojis/*/*.jsonl"))
    legacy = current.with_name(current.stem[:2] + ".jsonl")
    current.rename(legacy)
    return load_dataset(root), PurePosixPath(legacy.relative_to(root).as_posix())


def test_saved_legacy_analysis_validates_and_migrates_without_content_changes(tmp_path):
    before, legacy = _legacy_fixture(tmp_path)
    assert validate_snapshot(before, canonical=True).valid
    assert before.to_files(preserve_legacy_paths=True) == before.source_bytes
    after = before.clone()
    expected_changes = _changed_paths(before, after)
    changed = apply_snapshot(before, after, validator=validate_snapshot)
    assert changed == expected_changes
    assert legacy in changed
    assert not (tmp_path / legacy).exists()
    loaded = load_dataset(tmp_path)
    assert loaded.emojis == before.emojis
    assert loaded.memberships == before.memberships
    assert loaded.collections == before.collections
    assert validate_snapshot(loaded, canonical=True).valid
    assert all(len(path.stem) == 6 for path in tmp_path.glob("data/*/emojis/*/*.jsonl"))
    assert apply_snapshot(loaded, loaded.clone()) == ()


def test_saved_hash_prefixed_collection_validates_and_moves_to_flat_directory(tmp_path):
    original = write_fixture(tmp_path)
    collection = next(iter(original.collections.values()))
    flat = tmp_path / "data" / collection.platform / "collections" / collection.id
    legacy = flat.parent / collection_shard(collection.id) / collection.id
    legacy.parent.mkdir()
    flat.rename(legacy)
    before = load_dataset(tmp_path)
    assert validate_snapshot(before, canonical=True).valid
    assert before.to_files(preserve_legacy_paths=True) == before.source_bytes
    changed = apply_snapshot(before, before.clone(), validator=validate_snapshot)
    assert changed
    assert not (legacy / "collection.json").exists()
    assert not (legacy / "memberships.jsonl").exists()
    assert flat.is_dir()
    after = load_dataset(tmp_path)
    assert after.collections == before.collections
    assert after.memberships == before.memberships
    assert validate_snapshot(after, canonical=True).valid


def test_duplicate_collection_in_both_layouts_is_rejected(tmp_path):
    snapshot = write_fixture(tmp_path)
    collection = next(iter(snapshot.collections.values()))
    flat = tmp_path / "data" / collection.platform / "collections" / collection.id
    legacy = flat.parent / collection_shard(collection.id) / collection.id
    legacy.mkdir(parents=True)
    for name in ("collection.json", "memberships.jsonl"):
        (legacy / name).write_bytes((flat / name).read_bytes())
    with pytest.raises(DatasetLoadError, match="duplicate entity"):
        load_dataset(tmp_path)


@pytest.mark.parametrize("width", [2, 4, 6])
def test_wrong_or_unsupported_bucket_prefix_is_rejected(tmp_path, width):
    write_fixture(tmp_path)
    path = next(tmp_path.glob("data/*/emojis/*/*.jsonl"))
    wrong = "0" * width if path.stem[:width] != "0" * width else "f" * width
    path.rename(path.with_name(wrong + ".jsonl"))
    report = validate_snapshot(load_dataset(tmp_path), canonical=True)
    assert not report.valid
    assert "PATH" in {issue.code for issue in report.issues}


def test_mixed_layout_duplicate_cannot_hide_a_record(tmp_path):
    write_fixture(tmp_path)
    path = next(tmp_path.glob("data/*/emojis/*/*.jsonl"))
    path.with_name(path.stem[:2] + ".jsonl").write_bytes(path.read_bytes())
    with pytest.raises(DatasetLoadError, match="duplicate entity"):
        load_dataset(tmp_path)


def test_independent_pr_additions_in_same_legacy_bucket_survive_reapply(tmp_path):
    base = write_fixture(tmp_path)
    template = next(iter(base.emojis.values()))
    base.emojis.clear()
    base.memberships.clear()
    candidate, latest = base.clone(), base.clone()
    # The exact two identifiers from the reported PR conflict have a shared f8e6 prefix.
    ids = ("mxe_23711c96-5c1d-5e66-a556-0c6c82499493", "mxe_3eb59b79-e082-509f-b4a5-89f487614581")
    first, second = [emoji_bucket_path("telegram", identifier) for identifier in ids]
    assert legacy_bucket_path(first) == legacy_bucket_path(second)
    assert first != second
    candidate.emojis[ids[0]] = template.model_copy(update={"id": ids[0]})
    latest.emojis[ids[1]] = template.model_copy(update={"id": ids[1]})
    merged = reapply_candidate(base, candidate, latest)
    assert set(merged.emojis) == set(ids)
    assert first in merged.to_files()
    assert second in merged.to_files()
