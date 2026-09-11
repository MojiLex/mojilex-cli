import pytest

from mojilex_cli.dataset import AtomicDatasetWriter, load_dataset, validate_snapshot
from mojilex_cli.domain import (
    Collection,
    Membership,
    collection_id,
    membership_id,
    preview_takedown,
    takedown,
)
from test_dataset_helpers import write_fixture


def test_emoji_takedown_cascades_membership_and_leaves_anonymous_tombstone(tmp_path) -> None:
    snapshot = write_fixture(tmp_path)
    emoji_id = next(iter(snapshot.emojis))
    membership_id_value = next(iter(snapshot.memberships))
    collection_id_value = next(iter(snapshot.collections))
    before = {path: data for path, data in snapshot.to_files().items()}
    impact = preview_takedown(tmp_path, emoji_id, reason="privacy")
    assert set(impact.affected_ids) == {emoji_id, membership_id_value, collection_id_value}
    assert impact.changed_paths
    assert load_dataset(tmp_path).to_files() == before
    result = takedown(
        tmp_path,
        emoji_id,
        reason="privacy",
        withheld_at="2026-09-10T20:00:00Z",
        expected_source_sha256=impact.source_sha256,
    )
    loaded = load_dataset(tmp_path)
    assert emoji_id not in loaded.emojis
    assert not loaded.memberships
    assert next(iter(loaded.collections.values())).item_count == 0
    tombstone = loaded.tombstones[emoji_id].as_dict()
    assert set(tombstone) == {
        "schema_version",
        "entity_type",
        "target_entity_type",
        "target_id",
        "reason_code",
        "withheld_at",
        "public_note",
    }
    assert validate_snapshot(loaded, canonical=True).valid
    assert result.changed_paths


def test_takedown_is_idempotent_after_tombstone_exists(tmp_path) -> None:
    snapshot = write_fixture(tmp_path)
    emoji_id = next(iter(snapshot.emojis))
    takedown(tmp_path, emoji_id, reason="privacy", withheld_at="2026-09-10T20:00:00Z")
    second = takedown(tmp_path, emoji_id, reason="privacy")
    assert second.status == "noop"
    assert second.changed_paths == ()


def test_takedown_rejects_dataset_changed_after_preview(tmp_path) -> None:
    snapshot = write_fixture(tmp_path)
    emoji_id = next(iter(snapshot.emojis))
    impact = preview_takedown(tmp_path, emoji_id, reason="privacy")
    collection = next(iter(snapshot.collections.values()))
    collection.title = "Changed after preview"
    writer = AtomicDatasetWriter(tmp_path)
    for path, data in snapshot.to_files().items():
        writer.stage_bytes(path, data)
    writer.commit()

    with pytest.raises(RuntimeError, match="changed after takedown preview"):
        takedown(
            tmp_path,
            emoji_id,
            reason="privacy",
            expected_source_sha256=impact.source_sha256,
        )

    assert emoji_id in load_dataset(tmp_path).emojis


def test_collection_takedown_preserves_emoji_referenced_by_another_collection(tmp_path) -> None:
    snapshot = write_fixture(tmp_path)
    first_collection = next(iter(snapshot.collections.values()))
    emoji = next(iter(snapshot.emojis.values()))
    second_id = collection_id("telegram", "sticker_set.name", "global", "SecondPack")
    second = Collection.model_validate(
        {
            **first_collection.as_dict(),
            "id": second_id,
            "native_id": "SecondPack",
            "title": "Second Pack",
            "canonical_url": "https://t.me/addemoji/SecondPack",
            "extensions": {
                "telegram": {
                    **first_collection.extensions["telegram"],
                    "short_name": "SecondPack",
                }
            },
        }
    )
    second_membership = Membership(
        schema_version="1.0.0",
        entity_type="membership",
        id=membership_id(second_id, emoji.id),
        collection_id=second_id,
        emoji_id=emoji.id,
        status="active",
        position=0,
        first_seen_at="2026-09-10T18:00:00Z",
        last_changed_at="2026-09-10T18:00:00Z",
    )
    snapshot.collections[second_id] = second
    snapshot.memberships[second_membership.id] = second_membership
    writer = AtomicDatasetWriter(tmp_path)
    for path, data in snapshot.to_files().items():
        writer.stage_bytes(path, data)
    writer.commit()

    takedown(
        tmp_path,
        first_collection.id,
        reason="copyright",
        withheld_at="2026-09-10T20:00:00Z",
    )
    loaded = load_dataset(tmp_path)
    assert first_collection.id not in loaded.collections
    assert second_id in loaded.collections
    assert emoji.id in loaded.emojis
    assert next(iter(loaded.memberships.values())).collection_id == second_id
    assert validate_snapshot(loaded, canonical=True).valid
