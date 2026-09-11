"""Entity-aware three-way reapplication of a staged dataset change set."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from pydantic import BaseModel

from mojilex_cli.dataset import DatasetSnapshot

_Entity = TypeVar("_Entity", bound=BaseModel)


class ReapplyConflictError(RuntimeError):
    """A staged entity and the latest base both changed incompatibly."""

    code = "GIT_CONFLICT"

    def __init__(self, entity_ids: tuple[str, ...]) -> None:
        self.entity_ids = entity_ids
        joined = ", ".join(entity_ids[:8])
        if len(entity_ids) > 8:
            joined += f", and {len(entity_ids) - 8} more"
        super().__init__(f"staged changes conflict with the current base: {joined}")


def reapply_candidate(
    base: DatasetSnapshot,
    candidate: DatasetSnapshot,
    latest: DatasetSnapshot,
) -> DatasetSnapshot:
    """Apply ``base -> candidate`` once onto ``latest`` at entity granularity."""

    result = latest.clone()
    conflicts: list[str] = []
    result.manifest = _merge_manifest(base.manifest, candidate.manifest, latest.manifest, conflicts)
    result.collections = _merge_entities(
        "collection", base.collections, candidate.collections, latest.collections, conflicts
    )
    result.emojis = _merge_entities(
        "emoji", base.emojis, candidate.emojis, latest.emojis, conflicts
    )
    result.memberships = _merge_entities(
        "membership", base.memberships, candidate.memberships, latest.memberships, conflicts
    )
    result.relations = _merge_entities(
        "visual_relation", base.relations, candidate.relations, latest.relations, conflicts
    )
    result.tombstones = _merge_entities(
        "tombstone", base.tombstones, candidate.tombstones, latest.tombstones, conflicts
    )
    if conflicts:
        raise ReapplyConflictError(tuple(sorted(conflicts)))
    return result


def _merge_manifest(
    base: dict[str, object],
    candidate: dict[str, object],
    latest: dict[str, object],
    conflicts: list[str],
) -> dict[str, object]:
    if candidate == base:
        return dict(latest)
    if latest == base or latest == candidate:
        return dict(candidate)
    conflicts.append("dataset.json")
    return dict(latest)


def _merge_entities(
    kind: str,
    base: Mapping[str, _Entity],
    candidate: Mapping[str, _Entity],
    latest: Mapping[str, _Entity],
    conflicts: list[str],
) -> dict[str, _Entity]:
    result = {key: value.model_copy(deep=True) for key, value in latest.items()}
    for entity_id in sorted(set(base) | set(candidate)):
        original = base.get(entity_id)
        staged = candidate.get(entity_id)
        if _same(original, staged):
            continue
        current = latest.get(entity_id)
        if not (_same(current, original) or _same(current, staged)):
            conflicts.append(f"{kind}:{entity_id}")
            continue
        if staged is None:
            result.pop(entity_id, None)
        else:
            result[entity_id] = staged.model_copy(deep=True)
    return result


def _same(left: BaseModel | None, right: BaseModel | None) -> bool:
    if left is None or right is None:
        return left is right
    return left.model_dump(mode="json", exclude_none=True) == right.model_dump(
        mode="json", exclude_none=True
    )
