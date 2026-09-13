import pytest

from mojilex_cli.dataset import (
    DatasetValidationError,
    apply_snapshot,
    load_dataset,
    validate_snapshot,
)
from mojilex_cli.domain import (
    ContentRating,
    Emoji,
    Membership,
    Review,
    emoji_id,
    membership_id,
    review_emoji,
    reviewed_content_sha256,
    set_availability,
    telegram_set_fingerprint,
)
from test_dataset_helpers import write_fixture


def _stage_two_sensitive_emojis(root) -> tuple[str, str]:
    write_fixture(root)
    before = load_dataset(root)
    after = before.clone()
    first = next(iter(after.emojis.values()))
    first.content.rating = ContentRating.SENSITIVE

    second_native_id = "5368324170671202287"
    second_file_unique_id = "AgADSecondExampleUniqueId"
    second_id = emoji_id("telegram", "custom_emoji.id", "global", second_native_id)
    second_raw = first.as_dict()
    second_raw["id"] = second_id
    second_raw["native_id"] = second_native_id
    second_raw["extensions"]["telegram"]["custom_emoji_id"] = second_native_id
    second_raw["extensions"]["telegram"]["file_unique_id"] = second_file_unique_id
    second = Emoji.model_validate(second_raw)
    after.emojis[second.id] = second

    collection = next(iter(after.collections.values()))
    first_membership = next(iter(after.memberships.values()))
    second_membership_raw = first_membership.as_dict()
    second_membership_raw["id"] = membership_id(collection.id, second.id)
    second_membership_raw["emoji_id"] = second.id
    second_membership_raw["position"] = 1
    second_membership = Membership.model_validate(second_membership_raw)
    after.memberships[second_membership.id] = second_membership
    collection.item_count = 2

    first_telegram = first.extensions["telegram"]
    collection.extensions["telegram"]["set_fingerprint_sha256"] = telegram_set_fingerprint(
        [
            (first.native_id, str(first_telegram["file_unique_id"])),
            (second.native_id, second_file_unique_id),
        ]
    )
    apply_snapshot(before, after)
    return first.id, second.id


def test_review_approve_rewrites_only_bucket_and_recomputes_hash(tmp_path) -> None:
    snapshot = write_fixture(tmp_path)
    emoji_id = next(iter(snapshot.emojis))
    result = review_emoji(
        tmp_path,
        emoji_id,
        "approve",
        reviewer="reviewer",
        reviewed_at="2026-09-10T19:00:00Z",
    )
    loaded = load_dataset(tmp_path)
    emoji = loaded.emojis[emoji_id]
    assert emoji.review.status.value == "approved"
    assert emoji.review.reviewed_content_sha256 == reviewed_content_sha256(emoji)
    assert len(result.changed_paths) == 1
    assert validate_snapshot(loaded, canonical=True).valid


def test_review_rejects_unknown_action_before_writing(tmp_path) -> None:
    snapshot = write_fixture(tmp_path)
    emoji_id = next(iter(snapshot.emojis))
    before = {path: data for path, data in snapshot.to_files().items()}
    try:
        review_emoji(tmp_path, emoji_id, "maybe", reviewer="reviewer")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown action must fail")
    assert load_dataset(tmp_path).to_files() == before


def test_review_can_approve_one_sensitive_item_while_another_is_pending(tmp_path) -> None:
    first_id, second_id = _stage_two_sensitive_emojis(tmp_path)

    review_emoji(
        tmp_path,
        first_id,
        "approve",
        reviewer="reviewer",
        reviewed_at="2026-09-10T19:00:00Z",
    )

    staged = load_dataset(tmp_path)
    assert staged.emojis[first_id].review.status.value == "approved"
    assert staged.emojis[second_id].review.status.value == "unreviewed"
    assert validate_snapshot(staged, canonical=True).valid

    review_emoji(
        tmp_path,
        second_id,
        "approve",
        reviewer="reviewer",
        reviewed_at="2026-09-10T19:01:00Z",
    )
    assert validate_snapshot(load_dataset(tmp_path), canonical=True).valid


def test_human_approval_is_optional_for_an_unqualified_ai_item(tmp_path) -> None:
    write_fixture(tmp_path)
    before = load_dataset(tmp_path)
    staged = before.clone()
    emoji = next(iter(staged.emojis.values()))
    emoji.provenance.qualification_id = None
    apply_snapshot(before, staged)
    assert validate_snapshot(load_dataset(tmp_path), canonical=True).valid

    review_emoji(
        tmp_path,
        emoji.id,
        "approve",
        reviewer="reviewer",
        reviewed_at="2026-09-10T19:02:00Z",
    )

    approved = load_dataset(tmp_path)
    assert approved.emojis[emoji.id].review.status.value == "approved"
    assert validate_snapshot(approved, canonical=True).valid


def test_human_review_does_not_bypass_malformed_qualification_registry(tmp_path) -> None:
    snapshot = write_fixture(tmp_path)
    emoji_id_value = next(iter(snapshot.emojis))
    registry = tmp_path / "quality" / "model-qualifications.json"
    registry.write_text("{}\n", encoding="utf-8", newline="\n")

    with pytest.raises(DatasetValidationError, match="QUALIFICATION_REGISTRY"):
        review_emoji(tmp_path, emoji_id_value, "approve", reviewer="reviewer")


def test_review_does_not_ignore_non_policy_validation_failures(tmp_path) -> None:
    first_id, _ = _stage_two_sensitive_emojis(tmp_path)
    before = load_dataset(tmp_path)
    invalid = before.clone()
    next(iter(invalid.collections.values())).item_count = 99
    apply_snapshot(before, invalid)

    with pytest.raises(DatasetValidationError, match="ITEM_COUNT"):
        review_emoji(tmp_path, first_id, "approve", reviewer="reviewer")

    assert load_dataset(tmp_path).emojis[first_id].review == Review(status="unreviewed")


def test_manual_deleted_status_requires_reviewer_and_reason(tmp_path) -> None:
    snapshot = write_fixture(tmp_path)
    collection_id = next(iter(snapshot.collections))
    try:
        set_availability(tmp_path, collection_id, "deleted")
    except ValueError:
        pass
    else:
        raise AssertionError("deleted must require reviewer evidence")
    result = set_availability(
        tmp_path,
        collection_id,
        "deleted",
        reason_code="reviewer_confirmed",
        reviewer="reviewer",
        verified_at="2026-09-10T21:00:00Z",
    )
    loaded = load_dataset(tmp_path)
    availability = loaded.collections[collection_id].availability
    assert availability.status.value == "deleted"
    assert availability.set_by == "reviewer"
    assert result.changed_paths
