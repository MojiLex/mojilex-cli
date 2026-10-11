"""Private workspace paths may change; validation inputs and resolver bindings may not."""

import json
import shutil

import pytest
from referencing.exceptions import Unresolvable

from mojilex_cli.dataset import validation
from test_schema_validation_scope import check, count_validation_calls
from test_validation_compilation_cache import schema_fixture


def portable_fixture(root):
    snapshot, emoji_schema = schema_fixture(root)
    schema_root = root / "schemas" / "v1"
    for path in schema_root.glob("*.json"):
        document = json.loads(path.read_bytes())
        document["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        document["$id"] = f"https://schemas.example.test/v1/{path.name}"
        path.write_text(json.dumps(document), encoding="utf-8")
    dependency = schema_root / "definitions.schema.json"
    dependency.write_text(
        json.dumps(
            {
                "$id": "https://schemas.example.test/v1/definitions.schema.json",
                "properties": {"native_id": {"type": "string"}},
            }
        ),
        encoding="utf-8",
    )
    document = json.loads(emoji_schema.read_bytes())
    document["$ref"] = "https://schemas.example.test/v1/definitions.schema.json"
    emoji_schema.write_text(json.dumps(document), encoding="utf-8")
    return snapshot, emoji_schema


def copied_workspace(snapshot, root):
    shutil.copytree(snapshot.root / "schemas", root / "schemas")
    copied = snapshot.clone()
    copied.root = root
    return copied


def test_absolute_ids_reuse_successes_across_workspaces_but_registry_keeps_real_bindings(
    tmp_path, monkeypatch
):
    original, schema = portable_fixture(tmp_path / "one")
    moved = copied_workspace(original, tmp_path / "two")
    calls = count_validation_calls(monkeypatch)
    with validation.schema_validation_scope(persistent_cache=tmp_path / "cache"):
        first_registry, _, first_key = validation._schema_store(schema.parent)
        other_registry, _, other_key = validation._schema_store(moved.root / "schemas" / "v1")
        assert first_key == other_key
        assert first_registry is not other_registry
        assert schema.as_uri() in first_registry
        assert schema.as_uri() not in other_registry
        assert (moved.root / "schemas" / "v1" / schema.name).as_uri() in other_registry
        assert not check(original)
    before = len(calls)
    assert before > 0
    with validation.schema_validation_scope(persistent_cache=tmp_path / "cache"):
        assert not check(moved)
    assert len(calls) == before
    # Cached successes never replace semantic and identity checks.
    next(iter(moved.collections.values())).item_count += 1
    with validation.schema_validation_scope(persistent_cache=tmp_path / "cache"):
        assert any(
            issue.code == "ITEM_COUNT"
            for issue in validation.validate_snapshot(moved, schemas=True).issues
        )


def test_portable_success_still_binds_exact_schema_bytes_and_nested_record_content(
    tmp_path, monkeypatch
):
    original, schema = portable_fixture(tmp_path / "one")
    moved = copied_workspace(original, tmp_path / "two")
    calls = count_validation_calls(monkeypatch)
    with validation.schema_validation_scope(persistent_cache=tmp_path / "cache"):
        assert not check(original)
    before = len(calls)
    copied_schema = moved.root / "schemas" / "v1" / schema.name
    copied_schema.write_bytes(copied_schema.read_bytes() + b"\n")
    with validation.schema_validation_scope(persistent_cache=tmp_path / "cache"):
        assert not check(moved)
    assert len(calls) > before
    emoji = next(iter(moved.emojis.values()))
    document = json.loads(copied_schema.read_bytes())
    document["properties"]["native_id"] = {"const": emoji.native_id}
    copied_schema.write_text(json.dumps(document), encoding="utf-8")
    with validation.schema_validation_scope(persistent_cache=tmp_path / "cache"):
        assert not check(moved)
    emoji.native_id = "changed"
    with validation.schema_validation_scope(persistent_cache=tmp_path / "cache"):
        assert any(issue.code == "SCHEMA" for issue in check(moved))


@pytest.mark.parametrize(
    "change",
    [
        {"$ref": "definitions.schema.json"},
        {"$ref": "https://unknown.example.test/external.schema.json"},
        {"$id": "relative.schema.json"},
        {"$defs": {"nested": {"$id": "https://schemas.example.test/nested"}}},
        {"$schema": "https://json-schema.org/draft/2019-09/schema"},
        {"$dynamicRef": "definitions.schema.json#dynamic"},
    ],
)
def test_unknown_relative_or_changed_bases_keep_workspace_bound_key(tmp_path, change):
    original, schema = portable_fixture(tmp_path / "one")
    document = json.loads(schema.read_bytes())
    document.update(change)
    schema.write_text(json.dumps(document), encoding="utf-8")
    moved = copied_workspace(original, tmp_path / "two")
    assert (
        validation._schema_store(schema.parent)[2]
        != validation._schema_store(moved.root / "schemas" / "v1")[2]
    )


def test_file_uri_reference_cannot_reuse_success_in_different_workspace(tmp_path):
    original, schema = portable_fixture(tmp_path / "one")
    document = json.loads(schema.read_bytes())
    document["$ref"] = (schema.parent / "definitions.schema.json").as_uri()
    schema.write_text(json.dumps(document), encoding="utf-8")
    moved = copied_workspace(original, tmp_path / "two")
    with validation.schema_validation_scope():
        assert not check(original)
        assert (
            validation._schema_store(schema.parent)[2]
            != validation._schema_store(moved.root / "schemas" / "v1")[2]
        )
        # The second registry cannot resolve the first workspace's file URI.
        # Its cached success must never conceal the original resolution error.
        with pytest.raises(Unresolvable, match="Unresolvable"):
            check(moved)


def test_relative_schema_filename_changes_portable_success_identity(tmp_path):
    original, schema = portable_fixture(tmp_path / "one")
    moved = copied_workspace(original, tmp_path / "two")
    old_key = validation._schema_store(schema.parent)[2]
    copied_root = moved.root / "schemas" / "v1"
    dependency = copied_root / "definitions.schema.json"
    nested = copied_root / "nested" / dependency.name
    nested.parent.mkdir()
    dependency.rename(nested)
    assert validation._schema_store(copied_root)[2] != old_key


def test_literal_schema_key_in_const_and_default_keeps_bound_validation(tmp_path):
    original, emoji_schema = portable_fixture(tmp_path / "one")
    schema = emoji_schema.with_name("dataset.schema.json")
    document = json.loads(schema.read_bytes())
    literal = {"$schema": []}
    document["properties"] = {"probe": {"const": literal, "default": literal}}
    schema.write_text(json.dumps(document), encoding="utf-8")
    original.manifest["probe"] = literal
    moved = copied_workspace(original, tmp_path / "two")
    with validation.schema_validation_scope():
        assert (
            validation._schema_store(schema.parent)[2]
            != validation._schema_store(moved.root / "schemas" / "v1")[2]
        )
        assert not check(original)
        assert not check(moved)
        moved.manifest["probe"] = {"$schema": ["changed"]}
        assert any(issue.code == "SCHEMA" for issue in check(moved))
