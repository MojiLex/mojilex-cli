"""Successful exact schema memoization must not bypass fresh validation inputs."""

import json

import pytest

from mojilex_cli.dataset import validation
from test_validation_compilation_cache import schema_fixture


def check(snapshot):
    issues = []
    validation._validate_json_schemas(snapshot, issues)
    return issues


def count_validation_calls(monkeypatch):
    calls = []
    original = validation.Draft202012Validator.iter_errors

    def iter_errors(self, instance, *args, **kwargs):
        calls.append(instance)
        yield from original(self, instance, *args, **kwargs)

    monkeypatch.setattr(validation.Draft202012Validator, "iter_errors", iter_errors)
    return calls


def test_exact_success_reused_but_mutated_and_failed_instances_are_checked(tmp_path, monkeypatch):
    snapshot, path = schema_fixture(tmp_path)
    calls = count_validation_calls(monkeypatch)
    emoji = next(iter(snapshot.emojis.values()))
    native_id = emoji.native_id
    path.write_text(
        json.dumps({"properties": {"native_id": {"const": native_id}}}), encoding="utf-8"
    )
    with validation.schema_validation_scope():
        assert not check(snapshot)
        warmed = len(calls)
        assert warmed > 0
        with validation.schema_validation_scope():
            assert not check(snapshot)
            assert len(calls) == warmed
        emoji.native_id = "changed"
        assert any(issue.code == "SCHEMA" for issue in check(snapshot))
        invalid_calls = len(calls)
        assert invalid_calls > warmed
        assert any(issue.code == "SCHEMA" for issue in check(snapshot))
        assert len(calls) > invalid_calls
        emoji.native_id = native_id
        before = len(calls)
        assert not check(snapshot)
        assert len(calls) == before
    assert not check(snapshot)
    assert len(calls) > before  # No memo escapes its operation.


def test_exact_schema_bytes_and_referenced_schema_changes_invalidate(tmp_path, monkeypatch):
    snapshot, path = schema_fixture(tmp_path)
    dependency = path.parent / "definitions.schema.json"
    path.write_text(json.dumps({"$ref": "definitions.schema.json"}), encoding="utf-8")
    dependency.write_text(
        json.dumps({"properties": {"native_id": {"type": "string"}}}), encoding="utf-8"
    )
    calls = count_validation_calls(monkeypatch)
    with validation.schema_validation_scope():
        assert not check(snapshot)
        before = len(calls)
        assert not check(snapshot)
        assert len(calls) == before
        dependency.write_bytes(dependency.read_bytes() + b"\n")
        assert not check(snapshot)
        assert len(calls) > before  # Exact bytes, not just parsed semantic equality.
        dependency.write_text(json.dumps({"required": ["new_required_property"]}), encoding="utf-8")
        assert any(issue.code == "SCHEMA" for issue in check(snapshot))
        dependency.write_text("{malformed", encoding="utf-8")
        assert any(issue.code == "SCHEMA_INVALID" for issue in check(snapshot))


def test_all_integrity_checks_run_even_when_every_schema_instance_was_cached(tmp_path):
    snapshot, _ = schema_fixture(tmp_path)
    with validation.schema_validation_scope():
        assert validation.validate_snapshot(snapshot, schemas=True).valid
        # The lightweight schema fixture does not constrain item_count, but the
        # independently rerun semantic integrity checks still detect the mismatch.
        next(iter(snapshot.collections.values())).item_count += 1
        report = validation.validate_snapshot(snapshot, schemas=True)
        assert any(issue.code == "ITEM_COUNT" for issue in report.issues)


def test_schema_memo_is_bounded_and_eviction_does_not_skip_validation(tmp_path, monkeypatch):
    snapshot, _ = schema_fixture(tmp_path)
    monkeypatch.setattr(validation, "_SCHEMA_MEMO_LIMIT", 2)
    calls = count_validation_calls(monkeypatch)
    with validation.schema_validation_scope():
        for _ in range(3):
            before = len(calls)
            assert not check(snapshot)
            assert len(calls) > before
            memo = validation._SCHEMA_MEMO.get()
            assert len(memo.successes) <= 2
            assert all(isinstance(key, bytes) and len(key) == 32 for key in memo.successes)
    assert validation._SCHEMA_MEMO.get() is None


@pytest.mark.parametrize(
    "valid_value, mutated_value, schema",
    [
        ([1], (1,), {"type": "array"}),
        ({"123": 1}, {123: 1}, {"required": ["123"]}),
    ],
)
def test_non_json_mutation_cannot_hit_equivalent_json_cache(
    tmp_path, valid_value, mutated_value, schema
):
    snapshot, _ = schema_fixture(tmp_path)
    snapshot.manifest["cache_probe"] = valid_value
    path = tmp_path / "schemas" / "v1" / "dataset.schema.json"
    path.write_text(json.dumps({"properties": {"cache_probe": schema}}), encoding="utf-8")
    with validation.schema_validation_scope():
        assert not check(snapshot)
        snapshot.manifest["cache_probe"] = mutated_value
        assert any(issue.code == "SCHEMA" for issue in check(snapshot))
        assert validation._schema_instance_key(b"graph", "dataset", snapshot.manifest) is None


def test_repository_scan_still_detects_new_files_after_schema_cache_warms(tmp_path):
    snapshot, _ = schema_fixture(tmp_path)
    with validation.schema_validation_scope():
        assert validation.validate_snapshot(snapshot, schemas=True, repository_files=True).valid
        (tmp_path / "new-local-media.png").write_bytes(b"synthetic image artifact")
        report = validation.validate_snapshot(snapshot, schemas=True, repository_files=True)
        assert any(issue.code == "NO_MEDIA" for issue in report.issues)
