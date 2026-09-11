"""Stable MojiLex UUIDv5 identifiers."""

from __future__ import annotations

import unicodedata
from uuid import UUID, uuid5

DEFAULT_NAMESPACE = UUID("47d42c76-38da-5ab5-90fe-7af0ba6c4a27")
VISUAL_RELATION_NAMESPACE = UUID("4958ce2d-8120-5c3a-8755-a71d93c0c866")
DUPLICATE_GROUP_NAMESPACE = UUID("b6a91ddf-1126-5eee-ba5e-b8bd812883df")
RIGHTS_ASSIGNMENT_NAMESPACE = UUID("de845fef-2c83-51e1-a40a-f1bf11139639")


class IdentityError(ValueError):
    """An identity component cannot be represented by schema v1."""


def _component(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise IdentityError(f"{label} must be a string")
    if "\0" in value:
        raise IdentityError(f"{label} contains forbidden U+0000")
    return unicodedata.normalize("NFC", value)


def _epoch(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise IdentityError("identity_epoch must be a non-negative integer")
    return str(value)


def entity_identity_name(
    platform: str,
    entity_type: str,
    native_namespace: str,
    scope_id: str,
    native_id: str,
    identity_epoch: int = 0,
) -> str:
    if entity_type not in {"collection", "emoji"}:
        raise IdentityError("entity_type must be collection or emoji")
    return "\0".join(
        (
            _component(platform, "platform"),
            entity_type,
            _component(native_namespace, "native_namespace"),
            _component(scope_id, "scope_id"),
            _component(native_id, "native_id"),
            _epoch(identity_epoch),
        )
    )


def collection_id(
    platform: str,
    native_namespace: str,
    scope_id: str,
    native_id: str,
    identity_epoch: int = 0,
    *,
    namespace: UUID = DEFAULT_NAMESPACE,
) -> str:
    name = entity_identity_name(
        platform, "collection", native_namespace, scope_id, native_id, identity_epoch
    )
    return f"mxc_{uuid5(namespace, name)}"


def emoji_id(
    platform: str,
    native_namespace: str,
    scope_id: str,
    native_id: str,
    identity_epoch: int = 0,
    *,
    namespace: UUID = DEFAULT_NAMESPACE,
) -> str:
    name = entity_identity_name(
        platform, "emoji", native_namespace, scope_id, native_id, identity_epoch
    )
    return f"mxe_{uuid5(namespace, name)}"


def membership_id(
    collection: str,
    emoji: str,
    *,
    namespace: UUID = DEFAULT_NAMESPACE,
) -> str:
    name = "\0".join((_component(collection, "collection_id"), _component(emoji, "emoji_id")))
    return f"mxm_{uuid5(namespace, name)}"


def visual_relation_identity_name(
    subject_id: str,
    object_id: str,
    scope: str,
    *,
    subject_role: str | None = None,
    subject_variant_id: str | None = None,
    object_role: str | None = None,
    object_variant_id: str | None = None,
    identity_epoch: int = 0,
) -> str:
    """Return the direction-independent UUID name for one relation mapping.

    Endpoint IDs are always sorted for identity.  Their role/variant references
    move with the endpoint, while ``variant-of`` direction remains represented by
    the relation payload rather than by its ID.
    """

    if scope not in {"entity", "media-pair"}:
        raise IdentityError("visual relation scope must be entity or media-pair")
    for label, value in (
        ("subject_variant_id", subject_variant_id),
        ("object_variant_id", object_variant_id),
    ):
        if value == "":
            raise IdentityError(f"{label} must use None to encode an absent variant")
    endpoints = [
        (
            _component(subject_id, "subject_id"),
            _component(subject_role or "", "subject_role"),
            _component(subject_variant_id or "", "subject_variant_id"),
        ),
        (
            _component(object_id, "object_id"),
            _component(object_role or "", "object_role"),
            _component(object_variant_id or "", "object_variant_id"),
        ),
    ]
    if endpoints[0][0] == endpoints[1][0]:
        raise IdentityError("visual relation endpoints must be distinct")
    if scope == "entity":
        if any(
            value is not None
            for value in (
                subject_role,
                subject_variant_id,
                object_role,
                object_variant_id,
            )
        ):
            raise IdentityError("entity relation identity must not include media references")
    else:
        allowed_roles = {"primary", "light", "dark", "alternate"}
        if subject_role not in allowed_roles or object_role not in allowed_roles:
            raise IdentityError("media-pair relation identity requires both media roles")
    left, right = sorted(endpoints, key=lambda item: item[0])
    return "\0".join(
        (
            "visual-relation",
            left[0],
            right[0],
            scope,
            left[1],
            left[2],
            right[1],
            right[2],
            _epoch(identity_epoch),
        )
    )


def visual_relation_id(
    subject_id: str,
    object_id: str,
    scope: str,
    *,
    subject_role: str | None = None,
    subject_variant_id: str | None = None,
    object_role: str | None = None,
    object_variant_id: str | None = None,
    identity_epoch: int = 0,
    namespace: UUID = VISUAL_RELATION_NAMESPACE,
) -> str:
    name = visual_relation_identity_name(
        subject_id,
        object_id,
        scope,
        subject_role=subject_role,
        subject_variant_id=subject_variant_id,
        object_role=object_role,
        object_variant_id=object_variant_id,
        identity_epoch=identity_epoch,
    )
    return f"mxr_{uuid5(namespace, name)}"


def parse_namespace(value: str | UUID) -> UUID:
    try:
        return value if isinstance(value, UUID) else UUID(value)
    except (ValueError, AttributeError) as exc:
        raise IdentityError("id_namespace must be a valid UUID") from exc
