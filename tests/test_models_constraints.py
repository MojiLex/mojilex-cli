import pytest
from pydantic import ValidationError

from mojilex_cli.domain import (
    Availability,
    Content,
    LocalizedDescription,
    Media,
    Membership,
    Provenance,
    ToolProvenance,
)
from test_dataset_helpers import MEDIA_HASH, NOW, make_snapshot


def test_models_reject_unknown_universal_fields(tmp_path) -> None:
    emoji = next(iter(make_snapshot(tmp_path).emojis.values())).as_dict()
    emoji["file_id"] = "must-not-be-stored"
    from mojilex_cli.domain import Emoji

    with pytest.raises(ValidationError, match="Extra inputs"):
        Emoji.model_validate(emoji)


def test_external_numeric_identifiers_must_remain_json_strings(tmp_path) -> None:
    emoji = next(iter(make_snapshot(tmp_path).emojis.values())).as_dict()
    emoji["native_id"] = 5368324170671202286
    from mojilex_cli.domain import Emoji

    with pytest.raises(ValidationError, match="string"):
        Emoji.model_validate(emoji)


def test_static_media_requires_not_applicable_motion(tmp_path) -> None:
    emoji = next(iter(make_snapshot(tmp_path).emojis.values())).as_dict()
    emoji["descriptions"]["en"]["motion_status"] = "undetermined"
    from mojilex_cli.domain import Emoji

    with pytest.raises(ValidationError, match="static-only"):
        Emoji.model_validate(emoji)


def test_description_keys_must_be_bcp47(tmp_path) -> None:
    emoji = next(iter(make_snapshot(tmp_path).emojis.values())).as_dict()
    emoji["descriptions"]["not_a_language"] = emoji["descriptions"]["en"]
    from mojilex_cli.domain import Emoji

    with pytest.raises(ValidationError, match="BCP 47"):
        Emoji.model_validate(emoji)


def test_animated_media_requires_duration_and_matching_mime() -> None:
    with pytest.raises(ValidationError, match="duration_ms"):
        Media(
            role="primary",
            kind="video",
            format="webm",
            mime_type="video/webm",
            sha256=MEDIA_HASH,
            byte_size=10,
            width=10,
            height=10,
            animated=True,
        )
    with pytest.raises(ValidationError, match="must match format"):
        Media(
            role="primary",
            kind="video",
            format="webm",
            mime_type="image/webp",
            sha256=MEDIA_HASH,
            byte_size=10,
            width=10,
            height=10,
            animated=True,
            duration_ms=100,
        )


def test_descriptions_reject_html_controls_and_duplicate_usage() -> None:
    with pytest.raises(ValidationError, match="HTML"):
        LocalizedDescription(text="<b>cat</b>", motion_status="not_applicable", usage=[])
    with pytest.raises(ValidationError, match="unique"):
        LocalizedDescription(text="Cat.", motion_status="not_applicable", usage=["same", "same"])


def test_provenance_origin_conditions() -> None:
    tool = ToolProvenance(name="mojilex-cli", version="0.1.0")
    with pytest.raises(ValidationError, match="AI fields"):
        Provenance(origin="ai", pipeline_version="1.0.0", tool=tool)
    human = Provenance(
        origin="human",
        tool=tool,
        created_at=NOW,
        creator="reviewer",
    )
    assert human.provider is None


def test_provenance_rejects_invalid_semver_and_cross_origin_fields() -> None:
    with pytest.raises(ValidationError):
        ToolProvenance(name="mojilex-cli", version="01.0.0")
    tool = ToolProvenance(name="mojilex-cli", version="0.1.0")
    with pytest.raises(ValidationError, match="human creation fields"):
        Provenance(
            origin="ai",
            provider="provider",
            model="model",
            prompt_version="1.0.0",
            pipeline_version="1.0.0",
            description_profile="standard-v1",
            prompt_sha256="1" * 64,
            request_parameters_sha256="2" * 64,
            generation_stage="primary",
            routing_policy_version="1.0.0",
            routing_reason_codes=[],
            generated_at=NOW,
            input_media_sha256=[MEDIA_HASH],
            created_at=NOW,
            creator="reviewer",
            tool=tool,
        )


def test_invalid_calendar_timestamp_and_uuid_shape_are_rejected(tmp_path) -> None:
    with pytest.raises(ValidationError):
        Availability(
            status="active",
            first_seen_at="2026-99-99T18:00:00Z",
            last_changed_at=NOW,
            last_verified_at=NOW,
        )
    membership = next(iter(make_snapshot(tmp_path).memberships.values())).as_dict()
    membership["collection_id"] = "mxc_------------------------------------"
    with pytest.raises(ValidationError):
        Membership.model_validate(membership)


def test_unknown_availability_cannot_keep_manual_reason() -> None:
    with pytest.raises(ValidationError, match="must not retain"):
        Availability(
            status="unknown",
            first_seen_at=NOW,
            last_changed_at=NOW,
            reason_code="network_error",
        )


def test_content_warning_values_are_closed_and_unique() -> None:
    with pytest.raises(ValidationError):
        Content(rating="general", warnings=["not-a-warning"])
    with pytest.raises(ValidationError, match="unique"):
        Content(rating="sensitive", warnings=["nudity", "nudity"])


def test_concept_mapping_contract_is_closed_and_sorted(tmp_path) -> None:
    from mojilex_cli.domain import Emoji

    raw = next(iter(make_snapshot(tmp_path).emojis.values())).as_dict()
    raw["concept_mapping_status"] = "complete"
    raw["concept_ids"] = ["emotion.skepticism", "animal.cat"]
    with pytest.raises(ValidationError, match="bytewise sorted"):
        Emoji.model_validate(raw)

    raw["concept_ids"] = []
    with pytest.raises(ValidationError, match="requires at least one"):
        Emoji.model_validate(raw)

    raw["concept_mapping_status"] = "pending"
    raw["concept_ids"] = ["animal.cat"]
    with pytest.raises(ValidationError, match="must have no concept_ids"):
        Emoji.model_validate(raw)
