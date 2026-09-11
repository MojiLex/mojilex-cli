"""Reviewer-only state transitions with validated atomic dataset writes."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from mojilex_cli.dataset.repository import DatasetSnapshot, load_dataset
from mojilex_cli.dataset.staging import apply_snapshot
from mojilex_cli.dataset.validation import ValidationReport, validate_snapshot

from .hashes import reviewed_content_sha256, telegram_set_fingerprint
from .models import (
    Availability,
    AvailabilityStatus,
    Review,
    ReviewStatus,
    TakedownReason,
    Tombstone,
)


@dataclass(frozen=True, slots=True)
class OperationResult:
    target_id: str
    changed_paths: tuple[PurePosixPath, ...]
    status: str = "succeeded"


@dataclass(frozen=True, slots=True)
class TakedownImpact:
    target_id: str
    target_type: str
    affected_ids: tuple[str, ...]
    changed_paths: tuple[PurePosixPath, ...]
    source_sha256: str
    status: str = "succeeded"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _without_allowed_issues(
    report: ValidationReport, allowed_issue_codes: frozenset[str]
) -> ValidationReport:
    return ValidationReport(
        tuple(issue for issue in report.issues if issue.code not in allowed_issue_codes)
    )


def _validated_apply(
    before: DatasetSnapshot,
    after: DatasetSnapshot,
    *,
    before_allowed_issue_codes: frozenset[str] = frozenset(),
    after_allowed_issue_codes: frozenset[str] = frozenset(),
) -> tuple[PurePosixPath, ...]:
    schemas_available = (before.root / "schemas" / "v1").is_dir()
    _without_allowed_issues(
        validate_snapshot(
            before,
            canonical=True,
            schemas=schemas_available,
            repository_files=True,
        ),
        before_allowed_issue_codes,
    ).raise_for_errors()
    return apply_snapshot(
        before,
        after,
        validator=lambda value: _without_allowed_issues(
            validate_snapshot(
                value,
                schemas=schemas_available,
                repository_files=True,
            ),
            after_allowed_issue_codes,
        ),
    )


def review_emoji(
    root: str | Path,
    target_id: str,
    action: str,
    *,
    reviewer: str,
    reviewed_at: str | None = None,
) -> OperationResult:
    actions = {
        "approve": ReviewStatus.APPROVED,
        "request-changes": ReviewStatus.CHANGES_REQUESTED,
        "reject": ReviewStatus.REJECTED,
    }
    if action not in actions:
        raise ValueError("action must be approve, request-changes, or reject")
    before = load_dataset(root)
    current = before.emojis.get(target_id)
    if current is None:
        raise KeyError(f"emoji not found: {target_id}")
    after = before.clone()
    emoji = after.emojis[target_id]
    emoji.review = Review(
        status=actions[action],
        reviewed_at=reviewed_at or utc_now(),
        reviewer=reviewer,
        reviewed_content_sha256=reviewed_content_sha256(emoji),
        review_hash_profile_id="semantic-review-content-v3",
    )
    return OperationResult(
        target_id,
        _validated_apply(
            before,
            after,
            before_allowed_issue_codes=frozenset(
                {
                    "POLICY_REVIEW",
                    "POLICY_REVIEW_BLOCKING",
                    "QUALIFICATION",
                    "STYLE_CONFLICT",
                }
            ),
            after_allowed_issue_codes=frozenset(
                {
                    "POLICY_REVIEW",
                    "POLICY_REVIEW_BLOCKING",
                    "QUALIFICATION",
                    "STYLE_CONFLICT",
                }
            ),
        ),
    )


def set_availability(
    root: str | Path,
    target_id: str,
    status: AvailabilityStatus | str,
    *,
    reason_code: str | None = None,
    reviewer: str | None = None,
    verified_at: str | None = None,
) -> OperationResult:
    desired = AvailabilityStatus(status)
    if desired in {AvailabilityStatus.PRIVATE, AvailabilityStatus.DELETED} and (
        not reason_code or not reviewer
    ):
        raise ValueError("private/deleted status requires reason_code and reviewer")
    before = load_dataset(root)
    after = before.clone()
    entity = after.collections.get(target_id) or after.emojis.get(target_id)
    if entity is None:
        raise KeyError(f"collection or emoji not found: {target_id}")
    old = entity.availability
    now = verified_at or utc_now()
    entity.availability = Availability(
        status=desired,
        first_seen_at=old.first_seen_at,
        last_changed_at=now if desired != old.status else old.last_changed_at,
        last_verified_at=None if desired is AvailabilityStatus.UNKNOWN else now,
        reason_code=None
        if desired in {AvailabilityStatus.ACTIVE, AvailabilityStatus.UNKNOWN}
        else reason_code,
        set_by=reviewer
        if desired in {AvailabilityStatus.PRIVATE, AvailabilityStatus.DELETED}
        else None,
    )
    return OperationResult(target_id, _validated_apply(before, after))


def _recount(snapshot: DatasetSnapshot) -> None:
    for collection in snapshot.collections.values():
        collection.item_count = sum(
            membership.collection_id == collection.id and membership.status.value == "active"
            for membership in snapshot.memberships.values()
        )


def _refresh_telegram_fingerprints(snapshot: DatasetSnapshot) -> None:
    for collection in snapshot.collections.values():
        if collection.platform != "telegram":
            continue
        extension = collection.extensions.get("telegram")
        if not isinstance(extension, dict):
            continue
        pairs: list[tuple[str, str]] = []
        for membership in snapshot.memberships.values():
            if membership.collection_id != collection.id or membership.status.value != "active":
                continue
            emoji = snapshot.emojis.get(membership.emoji_id)
            if emoji is None:
                continue
            telegram = emoji.extensions.get("telegram")
            if isinstance(telegram, dict) and isinstance(telegram.get("file_unique_id"), str):
                pairs.append((emoji.native_id, telegram["file_unique_id"]))
        extension["set_fingerprint_sha256"] = telegram_set_fingerprint(pairs)


def _snapshot_sha256(snapshot: DatasetSnapshot) -> str:
    digest = hashlib.sha256()
    for path, data in sorted(snapshot.to_files().items(), key=lambda item: str(item[0])):
        encoded_path = str(path).encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _changed_snapshot_paths(
    before: DatasetSnapshot, after: DatasetSnapshot
) -> tuple[PurePosixPath, ...]:
    old_files = before.to_files()
    new_files = after.to_files()
    return tuple(
        sorted(
            (
                path
                for path in set(old_files) | set(new_files)
                if old_files.get(path) != new_files.get(path)
            ),
            key=str,
        )
    )


def _affected_snapshot_ids(before: DatasetSnapshot, after: DatasetSnapshot) -> tuple[str, ...]:
    affected: set[str] = set()
    for old, new in (
        (before.collections, after.collections),
        (before.emojis, after.emojis),
        (before.memberships, after.memberships),
        (before.relations, after.relations),
        (before.tombstones, after.tombstones),
    ):
        for identifier in set(old) | set(new):
            if old.get(identifier) != new.get(identifier):
                affected.add(identifier)
    return tuple(sorted(affected))


def _takedown_snapshots(
    root: str | Path,
    target_id: str,
    *,
    reason: TakedownReason | str,
    withheld_at: str | None,
    public_note: str,
) -> tuple[DatasetSnapshot, DatasetSnapshot, str]:
    before = load_dataset(root)
    if target_id in before.tombstones:
        return before, before.clone(), before.tombstones[target_id].target_entity_type
    after = before.clone()
    target_type: str
    removed_emoji_ids: set[str] = set()
    if target_id in after.emojis:
        target_type = "emoji"
        del after.emojis[target_id]
        removed_emoji_ids.add(target_id)
        after.memberships = {
            key: value for key, value in after.memberships.items() if value.emoji_id != target_id
        }
    elif target_id in after.collections:
        target_type = "collection"
        affected_emoji_ids = {
            item.emoji_id for item in after.memberships.values() if item.collection_id == target_id
        }
        del after.collections[target_id]
        after.memberships = {
            key: value
            for key, value in after.memberships.items()
            if value.collection_id != target_id
        }
        still_referenced = {item.emoji_id for item in after.memberships.values()}
        for emoji_id in affected_emoji_ids - still_referenced:
            if after.emojis.pop(emoji_id, None) is not None:
                removed_emoji_ids.add(emoji_id)
    elif target_id in after.memberships:
        target_type = "membership"
        del after.memberships[target_id]
    else:
        raise KeyError(f"entity not found: {target_id}")
    if removed_emoji_ids:
        after.relations = {
            key: value
            for key, value in after.relations.items()
            if value.subject_id not in removed_emoji_ids
            and value.object_id not in removed_emoji_ids
        }
    _recount(after)
    _refresh_telegram_fingerprints(after)
    after.tombstones[target_id] = Tombstone(
        schema_version="1.0.0",
        entity_type="tombstone",
        target_entity_type=target_type,
        target_id=target_id,
        reason_code=TakedownReason(reason),
        withheld_at=withheld_at or utc_now(),
        public_note=public_note,
    )
    return before, after, target_type


def preview_takedown(
    root: str | Path,
    target_id: str,
    *,
    reason: TakedownReason | str,
    public_note: str = "Record withheld under the MojiLex takedown policy.",
) -> TakedownImpact:
    before, after, target_type = _takedown_snapshots(
        root,
        target_id,
        reason=reason,
        withheld_at="1970-01-01T00:00:00Z",
        public_note=public_note,
    )
    changed_paths = _changed_snapshot_paths(before, after)
    return TakedownImpact(
        target_id=target_id,
        target_type=target_type,
        affected_ids=_affected_snapshot_ids(before, after),
        changed_paths=changed_paths,
        source_sha256=_snapshot_sha256(before),
        status="noop" if not changed_paths else "succeeded",
    )


def takedown(
    root: str | Path,
    target_id: str,
    *,
    reason: TakedownReason | str,
    withheld_at: str | None = None,
    public_note: str = "Record withheld under the MojiLex takedown policy.",
    expected_source_sha256: str | None = None,
) -> OperationResult:
    before, after, _ = _takedown_snapshots(
        root,
        target_id,
        reason=reason,
        withheld_at=withheld_at,
        public_note=public_note,
    )
    if expected_source_sha256 is not None and _snapshot_sha256(before) != expected_source_sha256:
        raise RuntimeError("dataset changed after takedown preview; review the impact again")
    if not _changed_snapshot_paths(before, after):
        return OperationResult(target_id, (), status="noop")
    return OperationResult(target_id, _validated_apply(before, after))
