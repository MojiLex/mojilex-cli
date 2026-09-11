import hashlib
import json
import os
import subprocess
from pathlib import Path, PurePosixPath

import pytest

import mojilex_cli.dataset.index as index_module
import mojilex_cli.dataset.validation as validation_module
from mojilex_cli.dataset import AtomicDatasetWriter, AtomicWriteError, build_index
from mojilex_cli.dataset.index import _index_transaction_lock_name
from mojilex_cli.dataset.transaction import DurableDatasetTransaction
from mojilex_cli.dataset.validation import ValidationReport
from mojilex_cli.domain import Emoji, Membership, membership_id
from test_dataset_helpers import write_fixture
from test_visual_relations import _relation, _two_emojis


def _create_windows_junction(link: Path, target: Path) -> None:
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        pytest.skip(f"junction creation is unavailable: {completed.stderr.strip()}")


def test_build_index_is_byte_for_byte_deterministic_and_self_verifying(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "dist"
    write_fixture(dataset)
    revision = "1" * 40
    first = build_index(dataset, output, git_commit=revision, validate=False)
    first_bytes = {item.name: item.read_bytes() for item in output.iterdir() if item.is_file()}
    second = build_index(dataset, output, git_commit=revision, validate=False)
    second_bytes = {item.name: item.read_bytes() for item in output.iterdir() if item.is_file()}
    assert first_bytes == second_bytes
    assert first.file_sha256 == second.file_sha256
    manifest = json.loads(first_bytes["manifest.json"])
    for name, digest in manifest["payload_sha256"].items():
        assert hashlib.sha256(first_bytes[name]).hexdigest() == digest
    expected_sums = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in first_bytes["SHA256SUMS"].decode().splitlines()
    }
    assert "SHA256SUMS" not in expected_sums
    assert (
        expected_sums["manifest.json"] == hashlib.sha256(first_bytes["manifest.json"]).hexdigest()
    )


def test_build_index_validates_and_builds_one_loaded_snapshot(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "dist"
    write_fixture(dataset)
    real_load = index_module.load_dataset
    load_count = 0
    validated_snapshots = []

    def counted_load(root):
        nonlocal load_count
        load_count += 1
        return real_load(root)

    def accept_snapshot(snapshot, **_kwargs):
        validated_snapshots.append(snapshot)
        return ValidationReport(())

    monkeypatch.setattr(index_module, "load_dataset", counted_load)
    monkeypatch.setattr(validation_module, "load_dataset", counted_load)
    monkeypatch.setattr(index_module, "validate_snapshot", accept_snapshot)
    monkeypatch.setattr(validation_module, "validate_snapshot", accept_snapshot)

    build_index(dataset, output, git_commit="1" * 40, validate=True)

    assert load_count == 1
    assert len(validated_snapshots) == 1


def test_search_rows_are_denormalized_without_media_or_secrets(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "dist"
    write_fixture(dataset)
    build_index(dataset, output, git_commit="2" * 40, validate=False)
    row = json.loads((output / "search-en.jsonl").read_text(encoding="utf-8"))
    assert row["text"].startswith("A yellow cat")
    assert len(row["collection_ids"]) == 1
    assert row["facets"]["content_types"] == ["animal", "reaction"]
    assert row["facets"]["literal_text"] == []
    assert row["duplicate_group_ids"] == []
    assert "media" not in row and "extensions" not in row


def test_build_index_emits_addendum_groups_relations_and_collection_facets(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "dist"
    snapshot = write_fixture(dataset)
    subject, object_emoji = _two_emojis(snapshot)
    relation = _relation(subject, object_emoji)
    snapshot.relations[relation.id] = relation
    collection = next(iter(snapshot.collections.values()))
    first_membership = next(iter(snapshot.memberships.values()))
    new_emoji = next(
        emoji for emoji in (subject, object_emoji) if emoji.id != first_membership.emoji_id
    )
    second_membership_raw = first_membership.as_dict()
    second_membership_raw.update(
        {
            "id": membership_id(collection.id, new_emoji.id),
            "emoji_id": new_emoji.id,
            "position": 1,
        }
    )
    second_membership = Membership.model_validate(second_membership_raw)
    snapshot.memberships[second_membership.id] = second_membership
    collection.item_count = 2
    writer = AtomicDatasetWriter(dataset)
    for path, data in snapshot.to_files().items():
        writer.stage_bytes(path, data)
    writer.commit()

    build_index(dataset, output, git_commit="6" * 40, validate=False)

    groups = [
        json.loads(line)
        for line in (output / "duplicate-groups.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {group["group_type"] for group in groups} == {
        "binary-exact",
        "decoded-exact",
        "reviewed-same-artwork",
    }
    assert any(group["scope"] == "entity" for group in groups)
    relations = [
        json.loads(line)
        for line in (output / "visual-relations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [item["id"] for item in relations] == [relation.id]
    collection_facets = json.loads((output / "collection-facets.jsonl").read_text(encoding="utf-8"))
    assert collection_facets["active_memberships"] == 2
    assert collection_facets["exact_duplicate_group_count"] >= 2
    assert collection_facets["reviewed_visual_duplicate_group_count"] == 1
    taxonomy = json.loads((output / "taxonomy.json").read_text(encoding="utf-8"))
    assert taxonomy["taxonomy_version"] == "1.0.0"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["visual_relations"] == 1
    assert manifest["profiles"]["dedupe"]["id"] == "dedupe-v1"


def test_release_keeps_current_approved_relation_for_unavailable_endpoint(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "dist"
    snapshot = write_fixture(dataset)
    subject, object_emoji = _two_emojis(snapshot)
    relation = _relation(subject, object_emoji)
    unavailable = subject.as_dict()
    unavailable["availability"]["status"] = "unavailable"
    snapshot.emojis[subject.id] = Emoji.model_validate(unavailable)
    snapshot.relations[relation.id] = relation
    writer = AtomicDatasetWriter(dataset)
    for path, data in snapshot.to_files().items():
        writer.stage_bytes(path, data)
    writer.commit()

    build_index(dataset, output, git_commit="8" * 40, validate=False)

    relations = [
        json.loads(line)
        for line in (output / "visual-relations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [item["id"] for item in relations] == [relation.id]


@pytest.mark.parametrize(
    "protected_name",
    ["data", "schemas", "tombstones", "taxonomy", "quality", "platforms", "examples"],
)
def test_build_index_rejects_canonical_dataset_subtrees(tmp_path, protected_name) -> None:
    dataset = tmp_path / "dataset"
    write_fixture(dataset)
    output = dataset / protected_name / "generated-index"

    with pytest.raises(ValueError, match="canonical"):
        build_index(dataset, output, git_commit="3" * 40, validate=False)

    assert not output.exists()


def test_build_index_rejects_dataset_root_and_its_ancestors_without_deleting(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    write_fixture(dataset)
    dataset_marker = dataset / "keep-root.txt"
    ancestor_marker = tmp_path / "keep-ancestor.txt"
    dataset_marker.write_text("keep", encoding="utf-8")
    ancestor_marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset root"):
        build_index(dataset, dataset, git_commit="4" * 40, validate=False)
    with pytest.raises(ValueError, match="dataset root"):
        build_index(dataset, tmp_path, git_commit="4" * 40, validate=False)

    assert dataset_marker.read_text(encoding="utf-8") == "keep"
    assert ancestor_marker.read_text(encoding="utf-8") == "keep"


def test_build_index_rejects_and_preserves_unmanaged_output_entries(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "dist"
    write_fixture(dataset)
    output.mkdir()
    marker = output / "keep-user-file.txt"
    directory = output / "keep-user-directory"
    marker.write_text("do not delete", encoding="utf-8")
    directory.mkdir()

    with pytest.raises(ValueError, match="unmanaged"):
        build_index(dataset, output, git_commit="5" * 40, validate=False)

    assert marker.read_text(encoding="utf-8") == "do not delete"
    assert directory.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_build_index_rejects_output_junction_without_touching_target(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    outside = tmp_path / "outside-output"
    output = tmp_path / "dist-link"
    write_fixture(dataset)
    outside.mkdir()
    sentinel = outside / "preserve.txt"
    sentinel.write_text("keep", encoding="utf-8")
    _create_windows_junction(output, outside)

    with pytest.raises(ValueError, match="link or reparse point"):
        build_index(dataset, output, git_commit="5" * 40, validate=False)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (outside / "manifest.json").exists()


def test_build_index_replaces_only_a_recognized_managed_output(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "dist"
    write_fixture(dataset)
    build_index(dataset, output, git_commit="5" * 40, validate=False)
    first_manifest = (output / "manifest.json").read_bytes()

    build_index(dataset, output, git_commit="6" * 40, validate=False)

    assert (output / "manifest.json").read_bytes() != first_manifest
    assert {item.name for item in output.iterdir()} == {
        "manifest.json",
        "SHA256SUMS",
        *{
            "collections.jsonl",
            "emojis.jsonl",
            "memberships.jsonl",
            "tombstones.jsonl",
            "emojis-active.jsonl",
            "search-ru.jsonl",
            "search-en.jsonl",
            "collection-facets.jsonl",
            "duplicate-groups.jsonl",
            "visual-relations.jsonl",
            "taxonomy.json",
        },
    }


def test_build_index_recovers_interrupted_output_before_managed_check(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "dist"
    write_fixture(dataset)
    build_index(dataset, output, git_commit="5" * 40, validate=False)

    transaction = DurableDatasetTransaction.prepare(
        output,
        {PurePosixPath("manifest.json"): b"interrupted\n"},
        lock_root=dataset,
        lock_name=_index_transaction_lock_name(output),
    )
    entry = transaction.entries[0]
    os.replace(transaction.staged_path(entry), output / "manifest.json")
    transaction.sync_target_parent(entry)
    transaction._release_lock()

    build_index(dataset, output, git_commit="6" * 40, validate=False)

    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["git_commit"] == "6" * 40
    assert not (output / ".mojilex-atomic-write").exists()
    assert not (output / ".mojilex").exists()


def test_build_index_rechecks_exact_output_under_transaction_lock(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "dist"
    write_fixture(dataset)
    build_index(dataset, output, git_commit="5" * 40, validate=False)
    original_manifest = (output / "manifest.json").read_bytes()
    real_payloads = index_module._payloads

    def inject_foreign_file(snapshot):
        (output / "foreign.txt").write_text("preserve", encoding="utf-8")
        return real_payloads(snapshot)

    monkeypatch.setattr(index_module, "_payloads", inject_foreign_file)

    with pytest.raises(AtomicWriteError, match="changed since it was inspected"):
        build_index(dataset, output, git_commit="6" * 40, validate=False)

    assert (output / "foreign.txt").read_text(encoding="utf-8") == "preserve"
    assert (output / "manifest.json").read_bytes() == original_manifest


def test_build_index_rejects_taxonomy_registry_path_traversal_even_without_validation(
    tmp_path,
) -> None:
    dataset = tmp_path / "dataset"
    output = tmp_path / "dist"
    write_fixture(dataset)
    taxonomy_path = dataset / "taxonomy" / "v1" / "taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    taxonomy["registries"][0]["path"] = "../outside.json"
    taxonomy_path.write_text(
        json.dumps(taxonomy, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="taxonomy/v1"):
        build_index(dataset, output, git_commit="7" * 40, validate=False)

    assert not output.exists()
