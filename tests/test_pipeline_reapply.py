from __future__ import annotations

import pytest

from mojilex_cli.pipeline.reapply import ReapplyConflictError, reapply_candidate
from test_dataset_helpers import make_snapshot
from test_visual_relations import _relation, _two_emojis


def test_reapply_preserves_unrelated_latest_entity_change(tmp_path) -> None:
    base = make_snapshot(tmp_path.resolve())
    candidate = base.clone()
    latest = base.clone()
    collection = next(iter(candidate.collections.values()))
    collection.title = "staged title"
    emoji = next(iter(latest.emojis.values()))
    emoji.semantic_tags.append("latest-only")

    merged = reapply_candidate(base, candidate, latest)

    assert merged.collections[collection.id].title == "staged title"
    assert "latest-only" in merged.emojis[emoji.id].semantic_tags


def test_reapply_rejects_concurrent_change_to_same_entity(tmp_path) -> None:
    base = make_snapshot(tmp_path.resolve())
    candidate = base.clone()
    latest = base.clone()
    entity_id = next(iter(candidate.collections))
    candidate.collections[entity_id].title = "staged title"
    latest.collections[entity_id].title = "latest title"

    with pytest.raises(ReapplyConflictError) as caught:
        reapply_candidate(base, candidate, latest)

    assert caught.value.entity_ids == (f"collection:{entity_id}",)


def test_reapply_is_idempotent_when_candidate_already_present(tmp_path) -> None:
    base = make_snapshot(tmp_path.resolve())
    candidate = base.clone()
    entity_id = next(iter(candidate.collections))
    candidate.collections[entity_id].title = "staged title"

    merged = reapply_candidate(base, candidate, candidate.clone())

    assert merged.to_files() == candidate.to_files()


def test_reapply_preserves_staged_visual_relation(tmp_path) -> None:
    base = make_snapshot(tmp_path.resolve())
    candidate = base.clone()
    subject, object_emoji = _two_emojis(candidate)
    relation = _relation(subject, object_emoji)
    candidate.relations[relation.id] = relation

    merged = reapply_candidate(base, candidate, base.clone())

    assert merged.relations == {relation.id: relation}
