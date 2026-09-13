from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

import pytest
from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]

import mojilex_cli.dataset.index as index_module
from mojilex_cli.dataset import build_index
from mojilex_cli.dataset.distribution import (
    DELEGATED_PROFILE_TYPES,
    PAYLOAD_NAMES,
    PROFILE_CONTRACT_SCHEMA_FILES,
    PROFILE_FILES,
    DataError,
)
from mojilex_cli.dataset.index import _index_transaction_lock_name
from mojilex_cli.dataset.staging import AtomicDatasetWriter
from mojilex_cli.dataset.transaction import AtomicWriteError, DurableDatasetTransaction
from mojilex_cli.domain import ConceptMappingStatus, jcs_bytes
from mojilex_cli.read.snapshot import _embedded_schema_contracts
from test_dataset_helpers import write_fixture

DATA_COMMIT = "1" * 40
TOOL_COMMIT = "2" * 40
LOCK_SHA256 = "3" * 64
SNAPSHOT_ID = "data-2026.09.11.1"
SOURCE_DATE_EPOCH = 1_789_084_800
RELEASE_MANIFEST_SCHEMA_URI = "mlx://schemas/distribution/v1/release-manifest.schema.json"

_CANONICAL_SCHEMAS = (
    "collection.schema.json",
    "emoji.schema.json",
    "membership.schema.json",
    "tombstone.schema.json",
    "visual-relation.schema.json",
)
_DISTRIBUTION_SCHEMAS = (
    "agent-record.schema.json",
    "analysis-profile.schema.json",
    "artifact-descriptor.schema.json",
    "collection-facet.schema.json",
    "concept-candidate-profile.schema.json",
    "concept.schema.json",
    "concepts-registry.schema.json",
    "delegated-profile.schema.json",
    "duplicate-group-membership.schema.json",
    "duplicate-group.schema.json",
    "platform-profile.schema.json",
    "platform-profiles-registry.schema.json",
    "release-manifest.schema.json",
    "resource-descriptor.schema.json",
    "rights-profile.schema.json",
    "rights-profiles-registry.schema.json",
    "search-record.schema.json",
    "taxonomy-dictionary.schema.json",
    "taxonomy-registry.schema.json",
    "taxonomy-source.schema.json",
    *sorted(PROFILE_CONTRACT_SCHEMA_FILES.values()),
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jcs(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jcs_bytes(value))


def _prepare_distribution_fixture(root: Path) -> Path:
    snapshot = write_fixture(root)
    for emoji_id, emoji in snapshot.emojis.items():
        snapshot.emojis[emoji_id] = emoji.model_copy(
            update={
                "concept_ids": ["animal.cat"],
                "concept_mapping_status": ConceptMappingStatus.COMPLETE,
            }
        )
    writer = AtomicDatasetWriter(root)
    for path, payload in snapshot.to_files().items():
        writer.stage_bytes(path, payload)
    writer.commit()
    for name in _CANONICAL_SCHEMAS:
        _write_json(
            root / "schemas" / "v1" / name,
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": f"https://schemas.mojilex.org/v1/{name}",
                "type": "object",
            },
        )
    for name in _DISTRIBUTION_SCHEMAS:
        _write_json(
            root / "schemas" / "distribution" / "v1" / name,
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": f"mlx://schemas/distribution/v1/{name}",
                "type": "object",
            },
        )
    for field, (name, scope) in PROFILE_FILES.items():
        profile: dict[str, object] = {
            "profile_schema_version": "1.0.0",
            "profile_id": f"{field.replace('_', '-')}-test-v1",
        }
        if field in DELEGATED_PROFILE_TYPES:
            contract_filename = PROFILE_CONTRACT_SCHEMA_FILES[field]
            contract_path = root / "schemas" / "distribution" / "v1" / contract_filename
            profile.update(
                {
                    "profile_type": DELEGATED_PROFILE_TYPES[field],
                    "contract_schema_ref": (f"mlx://schemas/distribution/v1/{contract_filename}"),
                    "contract_schema_sha256": hashlib.sha256(
                        contract_path.read_bytes()
                    ).hexdigest(),
                    "body": {"root_scope": scope},
                }
            )
        else:
            profile["root_scope"] = scope
        _write_jcs(root / "analysis-profiles" / name, profile)
    dataset_manifest_path = root / "dataset.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_bytes())
    for field, id_field, sha_field in (
        ("color", "color_profile", "color_profile_sha256"),
        ("dedupe", "dedupe_profile", "dedupe_profile_sha256"),
        (
            "collection_dedupe",
            "collection_dedupe_profile",
            "collection_dedupe_profile_sha256",
        ),
    ):
        filename = PROFILE_FILES[field][0]
        profile_path = root / "analysis-profiles" / filename
        profile = json.loads(profile_path.read_bytes())
        dataset_manifest[id_field] = profile["profile_id"]
        dataset_manifest[sha_field] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    _write_json(dataset_manifest_path, dataset_manifest)
    return root


def _build(dataset: Path, output: Path, **overrides: object):
    arguments: dict[str, object] = {
        "snapshot_id": SNAPSHOT_ID,
        "source_date_epoch": SOURCE_DATE_EPOCH,
        "git_commit": DATA_COMMIT,
        "tool_commit": TOOL_COMMIT,
        "dependency_lock_sha256": LOCK_SHA256,
        "validate": False,
    }
    arguments.update(overrides)
    return build_index(dataset, output, **arguments)  # type: ignore[arg-type]


def _tree_bytes(root: Path, *, include_transaction_lock: bool = True) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and (
            include_transaction_lock
            or re.fullmatch(
                r"\.mojilex/locks/(?:dataset-transaction-v1|index-[0-9a-f]{64})\.lock",
                path.relative_to(root).as_posix(),
            )
            is None
            or path.stat().st_size != 0
        )
    }


def test_build_index_is_byte_deterministic_and_self_describing(tmp_path: Path) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    output = tmp_path / "dist"

    first = _build(dataset, output)
    first_bytes = _tree_bytes(output)
    second = _build(dataset, output)
    second_bytes = _tree_bytes(output)

    assert first_bytes == second_bytes
    assert first.file_sha256 == second.file_sha256
    manifest_bytes = first_bytes["manifest.json"]
    assert not manifest_bytes.endswith(b"\n")
    manifest = json.loads(manifest_bytes)
    _schemas, registry, _resources = _embedded_schema_contracts()
    counts_validator = Draft202012Validator(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": f"{RELEASE_MANIFEST_SCHEMA_URI}#/$defs/counts",
        },
        registry=registry,
        format_checker=FormatChecker(),
    )
    assert not list(counts_validator.iter_errors(manifest["counts"]))
    assert manifest["snapshot_id"] == SNAPSHOT_ID
    assert manifest["git"] == {
        "repository": "https://github.com/MojiLex/mojilex",
        "commit": DATA_COMMIT,
        "object_format": "sha1",
    }
    assert manifest["trust_stage"] == "pre-enforcement"
    assert manifest["required_features"] == sorted(manifest["required_features"])
    assert manifest["build"]["source_date_epoch"] == SOURCE_DATE_EPOCH
    assert manifest["build"]["tool"] == "mojilex-cli"
    assert manifest["build"]["tool_commit"] == TOOL_COMMIT
    assert manifest["build"]["dependency_lock_sha256"] == LOCK_SHA256
    availability_statuses = {"active", "unavailable", "private", "deleted", "unknown"}
    assert set(manifest["counts"]["availability_by_status"]["collections"]) == (
        availability_statuses
    )
    assert set(manifest["counts"]["availability_by_status"]["emojis"]) == (availability_statuses)
    assert set(manifest["counts"]["emoji_review_by_status"]) == {
        "unreviewed",
        "approved",
        "changes_requested",
        "rejected",
    }
    assert set(manifest["counts"]["memberships_by_status"]) == {
        "active",
        "removed_from_collection",
        "unknown",
    }
    assert {item["path"] for item in manifest["artifacts"]} == set(PAYLOAD_NAMES)
    for descriptor in manifest["artifacts"]:
        payload = first_bytes[descriptor["path"]]
        assert descriptor["object_byte_size"] == len(payload)
        assert descriptor["object_sha256"] == hashlib.sha256(payload).hexdigest()
        assert descriptor["payload_sha256"] == descriptor["object_sha256"]
        assert descriptor["compression"] == "none"
        if descriptor["content_model"] == "recordset-jsonl":
            assert "schema_ref" in descriptor
            assert "table_root_sha256" in descriptor
            assert descriptor["record_count"] == len(payload.splitlines())
    for resource in manifest["resources"]:
        if resource["resource_kind"] != "physical":
            continue
        payload = first_bytes[resource["path"]]
        assert resource["object_byte_size"] == len(payload)
        assert resource["object_sha256"] == hashlib.sha256(payload).hexdigest()
    profile_resources = {
        resource["source_path"]: resource["content_schema_ref"]
        for resource in manifest["resources"]
        if resource["resource_kind"] == "physical"
        and resource["source_path"].startswith("analysis-profiles/")
    }
    assert profile_resources["analysis-profiles/concept-candidates-v1.json"] == (
        "mlx://schemas/distribution/v1/concept-candidate-profile.schema.json"
    )
    assert {
        schema_ref
        for source_path, schema_ref in profile_resources.items()
        if source_path != "analysis-profiles/concept-candidates-v1.json"
    } == {"mlx://schemas/distribution/v1/delegated-profile.schema.json"}
    descriptors = {item["logical_name"]: item for item in manifest["artifacts"]}
    profile_selectors = {
        "emojis-active": set(),
        "duplicate-groups": {"/profiles/dedupe"},
        "duplicate-group-memberships": {"/profiles/dedupe"},
        "collection-facets": {"/profiles/collection_dedupe"},
        "search-en": {"/profiles/lexical_search"},
        "search-ru": {"/profiles/lexical_search"},
    }
    policy_selectors = {
        "/policies/platform_profiles",
        "/policies/rights_profiles",
    }
    for logical_name, specific_selectors in profile_selectors.items():
        dependency = descriptors[logical_name]["derived_from"]
        assert "/build/source_date_epoch" in {
            item["manifest_pointer"] for item in dependency["manifest_inputs"]
        }
        assert {item["manifest_pointer"] for item in dependency["selectors"]} == (
            policy_selectors | specific_selectors
        )


def test_search_record_has_exact_safe_filtering_fields(tmp_path: Path) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    output = tmp_path / "dist"
    _build(dataset, output)

    rows = (output / "search-en.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    canonical = json.loads((output / "emojis.jsonl").read_bytes())
    assert row["record_schema_version"] == "1.0.0"
    assert row["entity_type"] == "search_record"
    assert row["language"] == "en"
    assert row["canonical_locator"]["logical_name"] == "emojis"
    assert row["canonical_record_sha256"] == hashlib.sha256(jcs_bytes(canonical)).hexdigest()
    assert row["availability"] == {"status": "active", "freshness_status": "unknown"}
    assert row["review"]["attested"] is False
    assert row["review"]["model_qualification_status"] == "missing"
    assert row["review"]["generation_attestation_status"] == "missing"
    assert row["rights"]["distribution_status"] == "allowed"
    assert row["facets"]["animated"] is False
    assert row["facets"]["contains_text"] is False
    assert row["literal_text"] == []
    assert row["duplicate_group_ids"] == []
    assert row["platform_capability_refs"] == ["telegram.message-custom-emoji"]


def test_builder_never_writes_media_or_mutates_canonical_source(tmp_path: Path) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    output = tmp_path / "dist"
    (dataset / ".mojilex" / "locks" / "dataset-transaction-v1.lock").touch()
    before = _tree_bytes(dataset, include_transaction_lock=False)

    _build(dataset, output)

    assert _tree_bytes(dataset, include_transaction_lock=False) == before
    emitted = _tree_bytes(output)
    assert emitted
    assert all(path == "SHA256SUMS" or Path(path).suffix in {".json", ".jsonl"} for path in emitted)
    assert not any(
        Path(path).suffix.lower() in {".webp", ".tgs", ".webm", ".png", ".jpg", ".jpeg"}
        for path in emitted
    )


def test_rebuild_rejects_tampered_or_incomplete_managed_output(tmp_path: Path) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    output = tmp_path / "dist"
    _build(dataset, output)
    search = output / "search-en.jsonl"
    search.write_bytes(search.read_bytes() + b"{}\n")

    with pytest.raises(ValueError, match="does not match"):
        _build(dataset, output)
    assert search.read_bytes().endswith(b"{}\n")

    search.write_bytes(b"")
    (output / "search-ru.jsonl").unlink()
    with pytest.raises(ValueError, match="missing"):
        _build(dataset, output)
    assert not (output / "search-ru.jsonl").exists()


def test_builder_rejects_and_preserves_unmanaged_output(tmp_path: Path) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    output = tmp_path / "dist"
    output.mkdir()
    marker = output / "keep-user-file.txt"
    marker.write_text("do not delete", encoding="utf-8")
    foreign = output / "keep-user-directory"
    foreign.mkdir()

    with pytest.raises(ValueError, match="recognized"):
        _build(dataset, output)

    assert marker.read_text(encoding="utf-8") == "do not delete"
    assert foreign.is_dir()


@pytest.mark.parametrize(
    "protected_name",
    ["analysis-profiles", "data", "platforms", "rights", "schemas", "taxonomy"],
)
def test_builder_rejects_canonical_subtrees(tmp_path: Path, protected_name: str) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    output = dataset / protected_name / "generated-index"

    with pytest.raises(ValueError, match="canonical"):
        _build(dataset, output)

    assert not output.exists()


def test_builder_requires_immutable_release_inputs(tmp_path: Path) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    output = tmp_path / "dist"

    with pytest.raises(ValueError, match="snapshot_id"):
        _build(dataset, output, snapshot_id="latest")
    with pytest.raises(ValueError, match="invalid UTC calendar date"):
        _build(dataset, output, snapshot_id="data-2026.02.31.1")
    with pytest.raises(ValueError, match="source_date_epoch"):
        _build(dataset, output, source_date_epoch=True)
    with pytest.raises(ValueError, match="tool_commit"):
        _build(dataset, output, tool_commit="short")


def test_builder_rejects_future_canonical_evidence(tmp_path: Path) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    output = tmp_path / "dist"
    emoji_path = next((dataset / "data" / "telegram" / "emojis").rglob("*.jsonl"))
    emoji = json.loads(emoji_path.read_bytes())
    emoji["availability"]["last_verified_at"] = "2026-09-12T00:00:00Z"
    emoji_path.write_bytes(jcs_bytes(emoji) + b"\n")

    with pytest.raises(DataError, match="future evidence"):
        _build(dataset, output)


def test_missing_tool_checkout_requires_explicit_tool_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")

    def no_checkout(_root: Path, *, option: str = "--git-commit") -> str:
        raise ValueError(f"cannot resolve a full Git commit; pass {option} explicitly")

    monkeypatch.setattr(index_module, "_git_commit", no_checkout)
    with pytest.raises(ValueError, match="pass --tool-commit explicitly"):
        build_index(
            dataset,
            tmp_path / "dist",
            snapshot_id=SNAPSHOT_ID,
            source_date_epoch=SOURCE_DATE_EPOCH,
            git_commit=DATA_COMMIT,
            dependency_lock_sha256=LOCK_SHA256,
            validate=False,
        )


def test_installed_layout_requires_explicit_dependency_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    installed = tmp_path / "installed-wheel"
    installed.mkdir()
    monkeypatch.setattr(index_module, "_tool_root", lambda: installed)

    with pytest.raises(ValueError, match="pass --dependency-lock-sha256 explicitly"):
        build_index(
            dataset,
            tmp_path / "dist",
            snapshot_id=SNAPSHOT_ID,
            source_date_epoch=SOURCE_DATE_EPOCH,
            git_commit=DATA_COMMIT,
            tool_commit=TOOL_COMMIT,
            validate=False,
        )


def test_interrupted_replacement_is_recovered_before_rebuild(tmp_path: Path) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    output = tmp_path / "dist"
    _build(dataset, output)
    expected = index_module._validate_existing_output(output)

    transaction = DurableDatasetTransaction.prepare(
        output,
        {PurePosixPath("manifest.json"): b"interrupted\n"},
        expected_tree_files=expected,
        lock_root=dataset,
        lock_name=_index_transaction_lock_name(output),
    )
    entry = transaction.entries[0]
    os.replace(transaction.staged_path(entry), output / "manifest.json")
    transaction.sync_target_parent(entry)
    transaction._release_lock()

    _build(dataset, output)

    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["snapshot_id"] == SNAPSHOT_ID
    assert not (output / ".mojilex-atomic-write").exists()


def test_tree_precondition_rejects_concurrent_output_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    output = tmp_path / "dist"
    _build(dataset, output)
    original_manifest = (output / "manifest.json").read_bytes()
    real_build = index_module.build_distribution

    def inject_foreign_file(*args: object, **kwargs: object):
        (output / "foreign.txt").write_text("preserve", encoding="utf-8")
        return real_build(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(index_module, "build_distribution", inject_foreign_file)
    with pytest.raises(AtomicWriteError, match="tree changed"):
        _build(dataset, output)

    assert (output / "foreign.txt").read_text(encoding="utf-8") == "preserve"
    assert (output / "manifest.json").read_bytes() == original_manifest


def test_rebuild_removes_only_empty_directories_from_retired_resources(tmp_path: Path) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    output = tmp_path / "dist"
    _build(dataset, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    retired_path = "resources/retired-v1/only.json"
    retired_payload = b"{}"
    retired_digest = hashlib.sha256(retired_payload).hexdigest()
    retired_file = output / Path(retired_path)
    retired_file.parent.mkdir(parents=True)
    retired_file.write_bytes(retired_payload)
    manifest["resources"].append(
        {
            "uri": "mlx://test/retired-v1",
            "resource_kind": "physical",
            "source_path": "test/retired-v1.json",
            "path": retired_path,
            "media_type": "application/json",
            "compression": "none",
            "payload_sha256": retired_digest,
            "object_sha256": retired_digest,
            "uncompressed_byte_size": len(retired_payload),
            "object_byte_size": len(retired_payload),
            "bindings": [],
        }
    )
    manifest["resources"].sort(key=lambda item: item["uri"].encode())
    manifest_bytes = jcs_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    checksums = {item["path"]: item["object_sha256"] for item in manifest["artifacts"]}
    checksums.update(
        {
            item["path"]: item["object_sha256"]
            for item in manifest["resources"]
            if item["resource_kind"] == "physical"
        }
    )
    checksums["manifest.json"] = hashlib.sha256(manifest_bytes).hexdigest()
    (output / "SHA256SUMS").write_text(
        "".join(f"{digest}  {path}\n" for path, digest in sorted(checksums.items())),
        encoding="ascii",
        newline="\n",
    )

    _build(dataset, output)

    assert not retired_file.exists()
    assert not retired_file.parent.exists()
    _build(dataset, output)


def test_source_inventory_excludes_only_empty_runtime_locks(tmp_path):
    files = {
        ".mojilex/locks/dataset-transaction-v1.lock": b"",
        ".mojilex/locks/index-" + "a" * 64 + ".lock": b"",
        ".mojilex/locks/index-" + "b" * 64 + ".lock": b"unexpected content",
        ".mojilex/locks/unexpected.lock": b"",
        ".mojilex/transactions/unexpected.json": b"{}",
        "data/example.json": b"{}",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    assert _tree_bytes(tmp_path) == files
    expected = dict(files)
    del expected[".mojilex/locks/dataset-transaction-v1.lock"]
    del expected[".mojilex/locks/index-" + "a" * 64 + ".lock"]
    assert _tree_bytes(tmp_path, include_transaction_lock=False) == expected
