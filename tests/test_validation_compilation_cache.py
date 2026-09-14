"""Compiled schemas/profiles may be reused; validation and byte checks may not."""

import hashlib
import json
from dataclasses import replace

import pytest

from mojilex_cli.analysis import profiles
from mojilex_cli.analysis.models import AnalysisError
from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.dataset.validation import _validate_json_schemas
from mojilex_cli.read import snapshot as reader
from mojilex_cli.schemas import EmbeddedSchema
from test_dataset_helpers import make_snapshot


def schema_fixture(tmp_path):
    snapshot = make_snapshot(tmp_path)
    root = tmp_path / "schemas" / "v1"
    root.mkdir(parents=True)
    for name in ("dataset", "collection", "membership"):
        (root / f"{name}.schema.json").write_text("{}", encoding="utf-8")
    path = root / "emoji.schema.json"
    path.write_text(json.dumps({"properties": {"native_id": {"type": "string"}}}), encoding="utf-8")
    return snapshot, path


def test_compiled_validator_checks_every_record_and_later_call(tmp_path):
    snapshot, path = schema_fixture(tmp_path)
    emoji = next(iter(snapshot.emojis.values()))
    snapshot.emojis["second"] = emoji.model_copy(update={"native_id": "second"})
    snapshot.emojis["third"] = emoji.model_copy(update={"native_id": "third"})
    path.write_text(
        json.dumps({"properties": {"native_id": {"const": emoji.native_id}}}), encoding="utf-8"
    )
    issues = []
    _validate_json_schemas(snapshot, issues)
    assert len([issue for issue in issues if issue.code == "SCHEMA"]) == 2
    # A previously valid instance can become invalid in the same process.
    emoji.native_id = "changed"
    later = []
    _validate_json_schemas(snapshot, later)
    assert len([issue for issue in later if issue.code == "SCHEMA"]) == 3


def test_schema_edit_is_reloaded_and_invalid_schema_is_rejected(tmp_path):
    snapshot, path = schema_fixture(tmp_path)
    issues = []
    _validate_json_schemas(snapshot, issues)
    assert issues == []
    path.write_text(json.dumps({"required": ["new_required_field"]}), encoding="utf-8")
    _validate_json_schemas(snapshot, issues)
    assert any(issue.code == "SCHEMA" and "new_required_field" in issue.message for issue in issues)
    path.write_text(json.dumps({"type": "not-a-json-schema-type"}), encoding="utf-8")
    issues = []
    _validate_json_schemas(snapshot, issues)
    assert any(issue.code == "SCHEMA_INVALID" for issue in issues)


def test_embedded_cache_is_bound_to_exact_payload_and_validates_each_instance(monkeypatch):
    uri = "mlx://schemas/test/cache-contract.schema.json"
    document = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": uri,
        "type": "object",
        "required": ["value"],
        "properties": {"value": {"type": "integer"}},
    }
    payload = json.dumps(document).encode()
    embedded = {
        uri: EmbeddedSchema(uri, "test.schema.json", payload, hashlib.sha256(payload).hexdigest())
    }
    monkeypatch.setattr(reader, "embedded_schemas", lambda: embedded)

    def validate(value):
        reader.validate_embedded_schema_instance(
            value, uri, location="test.json", invalid_code="MANIFEST_INVALID"
        )

    validate({"value": 1})
    with pytest.raises(CommandError):
        validate({"value": "invalid"})
    document["properties"]["value"]["minimum"] = 5
    # Even unchanged URI/hash metadata cannot make different bytes hit the cache.
    embedded[uri] = replace(embedded[uri], payload=json.dumps(document).encode())
    with pytest.raises(CommandError):
        validate({"value": 1})
    validate({"value": 5})
    document["type"] = "invalid-schema-type"
    embedded[uri] = replace(embedded[uri], payload=json.dumps(document).encode())
    with pytest.raises(CommandError, match="not valid Draft"):
        validate({"value": 5})


def test_warm_profile_cache_still_checks_disk_bytes_and_preserves_hash(tmp_path, monkeypatch):
    original = profiles.load_analysis_profile("color-v1")
    path = tmp_path / "color-v1.json"
    path.write_bytes(original.raw_bytes)
    monkeypatch.setattr(profiles, "files", lambda package: tmp_path)
    assert profiles.load_analysis_profile("color-v1").sha256 == original.sha256
    path.write_bytes(original.raw_bytes + b" ")
    with pytest.raises(AnalysisError, match="hash mismatch"):
        profiles.load_analysis_profile("color-v1")
    path.unlink()
    with pytest.raises(AnalysisError, match="unavailable"):
        profiles.load_analysis_profile("color-v1")
    path.write_bytes(original.raw_bytes)
    restored = profiles.load_analysis_profile("color-v1")
    assert restored.raw_bytes == original.raw_bytes
    assert restored.sha256 == original.sha256 == hashlib.sha256(restored.raw_bytes).hexdigest()
    with pytest.raises(TypeError):
        restored.data["palette"]["max_colors"] = 99
