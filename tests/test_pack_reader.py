from pathlib import Path

import pytest

from mojilex_cli.dataset.layout import emoji_bucket_path, legacy_bucket_path
from mojilex_cli.dataset.pack_reader import load_pack_snapshot
from mojilex_cli.dataset.repository import DatasetLoadError, load_dataset
from test_dataset_helpers import write_fixture


def test_partial_pack_read_matches_full_readiness_records(tmp_path):
    write_fixture(tmp_path)
    full = load_dataset(tmp_path)
    selected = load_pack_snapshot(tmp_path, {"SuspiciousCats"})
    assert selected.collections == full.collections
    assert selected.memberships == full.memberships
    assert selected.emojis == full.emojis


def test_unrequested_emoji_buckets_are_not_read(tmp_path, monkeypatch):
    fixture = write_fixture(tmp_path)
    real_read = Path.read_bytes
    reads = []

    def read(path):
        reads.append(path)
        return real_read(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    selected = load_pack_snapshot(tmp_path, {"AnotherPack"})
    assert selected.collections == fixture.collections
    assert not selected.emojis
    assert not any("emojis" in path.parts for path in reads)


def test_legacy_bucket_is_read_and_duplicate_ids_are_rejected(tmp_path):
    fixture = write_fixture(tmp_path)
    emoji = next(iter(fixture.emojis.values()))
    relative = emoji_bucket_path(emoji.platform, emoji.id)
    current = tmp_path / relative
    legacy = tmp_path / legacy_bucket_path(relative)
    data = current.read_bytes()
    current.rename(legacy)
    assert load_pack_snapshot(tmp_path, {"SuspiciousCats"}).emojis == fixture.emojis
    current.write_bytes(data)
    with pytest.raises(DatasetLoadError, match="duplicate entity ID"):
        load_pack_snapshot(tmp_path, {"SuspiciousCats"})


def test_missing_required_emoji_remains_missing(tmp_path):
    fixture = write_fixture(tmp_path)
    emoji = next(iter(fixture.emojis.values()))
    path = tmp_path / emoji_bucket_path(emoji.platform, emoji.id)
    path.rename(path.with_suffix(".saved"))
    assert not load_pack_snapshot(tmp_path, {"SuspiciousCats"}).emojis


def test_required_bucket_link_is_rejected(tmp_path):
    root = tmp_path / "dataset"
    fixture = write_fixture(root)
    emoji = next(iter(fixture.emojis.values()))
    path = root / emoji_bucket_path(emoji.platform, emoji.id)
    saved = tmp_path / "outside.jsonl"
    path.rename(saved)
    try:
        path.symlink_to(saved)
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    with pytest.raises(DatasetLoadError, match=r"link|reparse"):
        load_pack_snapshot(root, {"SuspiciousCats"})
