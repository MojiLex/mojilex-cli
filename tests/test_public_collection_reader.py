from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mojilex_cli import schemas
from mojilex_cli.dataset import collection_reader
from mojilex_cli.dataset.collection_reader import load_collection_names
from mojilex_cli.dataset.layout import collection_path, collection_shard
from mojilex_cli.dataset.repository import DatasetLoadError
from mojilex_cli.dataset.serialization import pretty_json, serialize_collection
from mojilex_cli.domain.models import Collection
from test_dataset_helpers import write_fixture


@pytest.fixture
def catalog(tmp_path):
    snapshot = write_fixture(tmp_path)
    shutil.copytree(Path(schemas.__file__).parent / "v1", tmp_path / "schemas" / "v1")
    record = next(iter(snapshot.collections.values()))
    path = tmp_path.joinpath(*collection_path("telegram", record.id).parts)
    return tmp_path, path, record


def test_reads_only_collection_identity_not_emoji_membership_or_readme(catalog, monkeypatch):
    root, path, record = catalog
    path.with_name("memberships.jsonl").write_bytes(b"not valid memberships")
    path.parent.parent.joinpath("README.md").write_text("local draft ExtraPack")
    real_open = Path.open
    reads = []

    def opened(file, *args, **kwargs):
        reads.append(file.relative_to(root).as_posix())
        assert "emojis" not in file.parts
        assert file.name not in {"memberships.jsonl", "README.md"}
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)
    assert load_collection_names(root) == {record.native_id.casefold()}
    assert "dataset.json" in reads
    assert sum(name.endswith("/collection.json") for name in reads) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2.0.0"),
        ("native_namespace", "unrelated"),
        ("scope_id", "private"),
        ("kind", "sticker_set"),
        ("platform", "other"),
        ("canonical_url", "https://t.me/addemoji/OtherPack"),
        ("id", "mxc_00000000-0000-5000-8000-000000000000"),
        ("native_id", "OtherPack"),
        ("item_count", -1),
    ],
)
def test_rejects_invalid_public_collection_structure_or_identity(catalog, field, value):
    root, path, record = catalog
    data = record.as_dict()
    data[field] = value
    path.write_text(pretty_json(data), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError):
        load_collection_names(root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2.0.0"),
        ("canonical_repository", "https://github.com/Other/private"),
        ("id_namespace", "not-a-namespace"),
        ("dataset", "other"),
    ],
)
def test_rejects_incompatible_manifest(catalog, field, value):
    root, _, _ = catalog
    path = root / "dataset.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(pretty_json(data), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError):
        load_collection_names(root)


def test_rejects_json_schema_extension_violation(catalog):
    root, path, record = catalog
    data = record.as_dict()
    data["extensions"]["telegram"]["unknown_field"] = "forbidden"
    path.write_bytes(serialize_collection(Collection.model_validate(data)))
    with pytest.raises(ValueError, match="SCHEMA"):
        load_collection_names(root)


@pytest.mark.parametrize("value", [None, "invalid", []])
def test_malformed_telegram_extension_is_a_controlled_read_failure(catalog, value):
    root, path, record = catalog
    data = record.as_dict()
    data["extensions"]["telegram"] = value
    path.write_text(pretty_json(data), encoding="utf-8", newline="\n")
    with pytest.raises(DatasetLoadError):
        load_collection_names(root)


def test_identity_epoch_must_match_derived_collection_id(catalog):
    root, path, record = catalog
    path.write_bytes(serialize_collection(record.model_copy(update={"identity_epoch": 1})))
    with pytest.raises(ValueError, match="ID"):
        load_collection_names(root)


def test_missing_required_schema_is_not_a_valid_empty_catalog(catalog):
    root, _, _ = catalog
    root.joinpath("schemas/v1/collection.schema.json").unlink()
    with pytest.raises(ValueError, match="SCHEMA_MISSING"):
        load_collection_names(root)


def test_rejects_duplicate_id_even_in_supported_old_layout(catalog):
    root, path, record = catalog
    other = path.parent.parent / collection_shard(record.id) / record.id / "collection.json"
    other.parent.mkdir(parents=True)
    other.write_bytes(path.read_bytes())
    with pytest.raises(DatasetLoadError, match="duplicate entity ID"):
        load_collection_names(root)


def test_accepts_supported_old_collection_shard(catalog):
    root, path, record = catalog
    other = path.parent.parent / collection_shard(record.id) / record.id / "collection.json"
    other.parent.mkdir(parents=True)
    other.write_bytes(path.read_bytes())
    path.unlink()
    # Move the other legacy files too, leaving no misleading canonical pack folder.
    for entry in path.parent.iterdir():
        entry.rename(other.parent / entry.name)
    path.parent.rmdir()
    assert load_collection_names(root) == {record.native_id.casefold()}


def test_missing_collection_never_silently_becomes_an_empty_catalog(catalog):
    root, path, _ = catalog
    path.unlink()
    with pytest.raises(DatasetLoadError, match="record is missing"):
        load_collection_names(root)


def test_noncanonical_collection_path_is_rejected(catalog):
    root, path, _ = catalog
    target = path.parent.parent / "wrong-id" / "collection.json"
    path.parent.rename(target.parent)
    with pytest.raises(DatasetLoadError, match="noncanonical ID"):
        load_collection_names(root)


def test_reader_has_bounded_files(catalog, monkeypatch):
    root, _, _ = catalog
    monkeypatch.setattr(collection_reader, "_MAX_FILE_BYTES", 32)
    with pytest.raises(DatasetLoadError, match="safe limit"):
        load_collection_names(root)


def test_missing_manifest_cannot_authorize_analysis(tmp_path):
    with pytest.raises(DatasetLoadError):
        load_collection_names(tmp_path)


def test_reader_rejects_linked_collection_record(catalog, tmp_path):
    root, path, _ = catalog
    original = tmp_path / "outside.json"
    original.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(original)
    except OSError:
        pytest.skip("OS does not permit symlink creation")
    with pytest.raises(DatasetLoadError):
        load_collection_names(root)
