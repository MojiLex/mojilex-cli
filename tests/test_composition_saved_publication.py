from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from mojilex_cli.composition.publication import (
    defer_fragment_overflow_for_legacy_schema,
    mark_saved_fragments,
    strip_legacy_fragment_tags,
)
from mojilex_cli.domain import emoji_id, membership_id
from mojilex_cli.pipeline.batch import _add_missing_packs
from mojilex_cli.pipeline.reapply import reapply_candidate
from mojilex_cli.runs import ElementCheckpoint
from test_dataset_helpers import make_snapshot


def _saved(tmp_path):
    snapshot = make_snapshot(tmp_path.resolve())
    first = next(iter(snapshot.emojis.values()))
    second = first.model_copy(deep=True)
    second.native_id = "12345"
    second.id = emoji_id("telegram", "custom_emoji.id", "global", second.native_id)
    snapshot.emojis[second.id] = second
    original_membership = next(iter(snapshot.memberships.values()))
    membership = original_membership.model_copy(deep=True)
    membership.emoji_id = second.id
    membership.id = membership_id(membership.collection_id, second.id)
    membership.position = 1
    snapshot.memberships[membership.id] = membership
    name = next(iter(snapshot.collections.values())).native_id
    group = {
        "detector": "composition-v3",
        "columns": 2,
        "rows": 1,
        "verified": True,
        "verifier_model": "test",
        "verification_passes": 3,
        "members": [
            {
                "native_id": emoji.native_id,
                "media_sha256": emoji.media[0].sha256,
                "tile_sha256": "c" * 64,
            }
            for emoji in snapshot.emojis.values()
        ],
    }
    checkpoint = SimpleNamespace(
        safe_parameters={
            "composition_evidence": {name: [group]},
            "source_memberships": {name: [first.native_id, second.native_id]},
        },
        elements={
            emoji.native_id: ElementCheckpoint(
                stage="validated", media_sha256=(emoji.media[0].sha256,)
            )
            for emoji in snapshot.emojis.values()
        },
    )
    return snapshot, checkpoint, group


@pytest.mark.parametrize("detector", ["composition-v2", "composition-v3"])
def test_saved_projection_preserves_twelve_concrete_tags_and_is_idempotent(tmp_path, detector):
    snapshot, checkpoint, group = _saved(tmp_path)
    group["detector"] = detector
    for emoji in snapshot.emojis.values():
        emoji.semantic_tags = [f"tag-{i}" for i in range(12)]
    assert mark_saved_fragments(snapshot, checkpoint) == set(snapshot.emojis)
    assert all(len(emoji.semantic_tags) == 13 for emoji in snapshot.emojis.values())
    assert mark_saved_fragments(snapshot, checkpoint) == set()


@pytest.mark.parametrize(
    "damage",
    [
        "unverified",
        "old",
        "passes",
        "missing-model",
        "coerced",
        "malformed",
        "overlap",
        "missing-checkpoint-member",
        "stale-checkpoint-hash",
        "missing-public-member",
        "stale-public-hash",
        "inactive-membership",
        "wrong-pack",
        "missing-allowed-member",
        "repainting",
        "animated",
        "duplicate-active-native",
        "oversized",
        "bad-container",
    ],
)
def test_saved_projection_rejects_entire_stale_or_malformed_group(tmp_path, damage):
    snapshot, checkpoint, group = _saved(tmp_path)
    first = next(iter(snapshot.emojis.values()))
    name = next(iter(snapshot.collections.values())).native_id
    if damage == "unverified":
        group["verified"] = False
    elif damage == "old":
        group.update(detector="composition-v1", verification_passes=0)
    elif damage == "passes":
        group["verification_passes"] = 2
    elif damage == "missing-model":
        group["verifier_model"] = None
    elif damage == "coerced":
        group["verified"] = "true"
    elif damage == "malformed":
        group["members"] = []
    elif damage == "overlap":
        checkpoint.safe_parameters["composition_evidence"][name].append(copy.deepcopy(group))
    elif damage == "missing-checkpoint-member":
        checkpoint.elements.pop(first.native_id)
    elif damage == "stale-checkpoint-hash":
        checkpoint.elements[first.native_id] = ElementCheckpoint(
            stage="validated", media_sha256=("0" * 64,)
        )
    elif damage == "missing-public-member":
        snapshot.emojis.pop(first.id)
    elif damage == "stale-public-hash":
        first.media[0].sha256 = "0" * 64
    elif damage == "inactive-membership":
        membership = next(iter(snapshot.memberships.values()))
        membership.status = type(membership.status).REMOVED
    elif damage == "wrong-pack":
        next(iter(snapshot.collections.values())).native_id = "OtherPack"
    elif damage == "missing-allowed-member":
        checkpoint.safe_parameters["source_memberships"][name].remove(first.native_id)
    elif damage == "repainting":
        first.extensions["telegram"]["needs_repainting"] = True
    elif damage == "animated":
        first.media[0] = first.media[0].model_copy(update={"animated": True})
    elif damage == "duplicate-active-native":
        list(snapshot.emojis.values())[1].native_id = first.native_id
    elif damage == "oversized":
        checkpoint.safe_parameters["composition_evidence"][name] = [group] * 65
    elif damage == "bad-container":
        checkpoint.safe_parameters["composition_evidence"] = []
    before = {key: list(value.semantic_tags) for key, value in snapshot.emojis.items()}
    assert not mark_saved_fragments(snapshot, checkpoint)
    assert {key: value.semantic_tags for key, value in snapshot.emojis.items()} == before


def test_projection_is_kept_in_candidate_and_reapplied_to_latest_root(tmp_path):
    import json
    import shutil
    from pathlib import Path

    from mojilex_cli.dataset.validation import _validate_json_schemas

    candidate, checkpoint, _ = _saved(tmp_path / "old-schema")
    for emoji in candidate.emojis.values():
        emoji.semantic_tags = [f"tag-{i}" for i in range(12)]
    base = candidate.clone()
    latest = base.clone()
    latest.root = (tmp_path / "current-schema").resolve()
    schemas = Path(__file__).parents[1] / "src" / "mojilex_cli" / "schemas" / "v1"
    for root in (candidate.root, latest.root):
        shutil.copytree(schemas, root / "schemas" / "v1")
    old_common = candidate.root / "schemas" / "v1" / "common.schema.json"
    old_schema = json.loads(old_common.read_text(encoding="utf-8"))
    old_schema["$defs"]["semanticTags"]["maxItems"] = 12
    for extra in ("description", "if", "then", "else"):
        old_schema["$defs"]["semanticTags"].pop(extra, None)
    old_common.write_text(json.dumps(old_schema), encoding="utf-8")
    original_issues = []
    _validate_json_schemas(candidate, original_issues)
    assert original_issues == []
    mark_saved_fragments(candidate, checkpoint)
    old_issues = []
    _validate_json_schemas(candidate, old_issues)
    assert old_issues
    assert defer_fragment_overflow_for_legacy_schema(candidate) == set(candidate.emojis)
    deferred_issues = []
    _validate_json_schemas(candidate, deferred_issues)
    assert deferred_issues == []
    assert mark_saved_fragments(candidate, checkpoint) == set(candidate.emojis)
    merged = reapply_candidate(base, candidate, latest)
    current_issues = []
    _validate_json_schemas(merged, current_issues)
    assert current_issues == []
    assert merged.root == latest.root
    assert all(len(emoji.semantic_tags) == 13 for emoji in merged.emojis.values())
    assert all(len(emoji.semantic_tags) == 12 for emoji in base.emojis.values())


@pytest.mark.parametrize("schema_kind", ["new", "missing", "unknown", "malformed"])
def test_overflow_deferral_never_changes_unrecognized_or_current_schema(tmp_path, schema_kind):
    import json
    from pathlib import Path

    snapshot, checkpoint, _ = _saved(tmp_path)
    for emoji in snapshot.emojis.values():
        emoji.semantic_tags = [f"tag-{i}" for i in range(12)]
    mark_saved_fragments(snapshot, checkpoint)
    if schema_kind != "missing":
        source = Path(__file__).parents[1] / "src/mojilex_cli/schemas/v1/common.schema.json"
        schema = json.loads(source.read_text(encoding="utf-8"))
        if schema_kind == "unknown":
            schema["$defs"]["semanticTags"]["maxItems"] = 12
            schema["$defs"]["semanticTags"]["items"]["maxLength"] = 49
        target = snapshot.root / "schemas/v1/common.schema.json"
        target.parent.mkdir(parents=True)
        target.write_text(
            "{" if schema_kind == "malformed" else json.dumps(schema), encoding="utf-8"
        )
    before = {key: value.as_dict() for key, value in snapshot.emojis.items()}
    assert defer_fragment_overflow_for_legacy_schema(snapshot) == set()
    assert {key: value.as_dict() for key, value in snapshot.emojis.items()} == before


def test_sync_keeps_published_shared_emoji_authoritative(tmp_path):
    candidate, checkpoint, _ = _saved(tmp_path)
    base = candidate.clone()
    base.collections.clear()
    base.memberships.clear()
    base.emojis.clear()
    latest = base.clone()
    shared = next(iter(candidate.emojis.values())).model_copy(deep=True)
    latest.emojis[shared.id] = shared
    mark_saved_fragments(candidate, checkpoint)
    merged, added, skipped = _add_missing_packs(latest, base, candidate)
    assert added and not skipped
    assert "fragment" not in merged.emojis[shared.id].semantic_tags
    assert all(
        "fragment" in emoji.semantic_tags
        for key, emoji in merged.emojis.items()
        if key != shared.id
    )


@pytest.mark.asyncio
async def test_local_submit_projects_saved_evidence_without_rewriting_staging(
    tmp_path, monkeypatch
):
    from contextlib import nullcontext

    from mojilex_cli.pipeline import runner

    snapshot, checkpoint, _ = _saved(tmp_path / "staging")
    checkpoint.command = "describe"
    checkpoint.status = "succeeded"
    checkpoint.run_id = "mlxrun_" + "a" * 32
    checkpoint.target_repository = "owner/repo"
    checkpoint.base_revision = "f" * 40
    original_tags = {key: list(value.semantic_tags) for key, value in snapshot.emojis.items()}
    config = SimpleNamespace(
        runs_dir=tmp_path / "runs", repository=SimpleNamespace(publish="local")
    )
    monkeypatch.setattr(runner, "load_config", lambda **kwargs: config)
    monkeypatch.setattr(runner, "load_credentials", lambda: SimpleNamespace())
    monkeypatch.setattr(
        runner, "RunStore", lambda _: SimpleNamespace(load_for_resume=lambda *a, **kw: checkpoint)
    )
    monkeypatch.setattr(runner, "_staging_path_from_checkpoint", lambda _: snapshot.root)
    monkeypatch.setattr(runner, "snapshot_at_revision", lambda *a: nullcontext(tmp_path / "base"))
    monkeypatch.setattr(runner, "load_dataset", lambda _: snapshot.clone())
    validated = []

    def validate(path, **kwargs):
        validated.append(path)
        return SimpleNamespace(raise_for_errors=lambda: None)

    monkeypatch.setattr(runner, "validate_dataset", validate)

    def routing(candidate):
        assert all("fragment" in emoji.semantic_tags for emoji in candidate.emojis.values())
        return SimpleNamespace(as_dict=lambda: {})

    monkeypatch.setattr(runner, "official_submission_report", routing)
    result = await runner._run_submit(
        checkpoint.run_id,
        repository=None,
        publish="local",
        direct_push=False,
        base=None,
        confirmation=None,
    )
    assert validated == [snapshot.root]
    assert result.result["fragment_markers_projected"] == 2
    assert result.result["projection_persisted"] is False
    assert result.result["changed_paths"]
    assert {key: value.semantic_tags for key, value in snapshot.emojis.items()} == original_tags


def _legacy(tmp_path):
    snapshot, checkpoint, group = _saved(tmp_path)
    first = next(iter(snapshot.emojis.values()))
    checkpoint.created_at = "2026-09-10T17:00:00Z"
    checkpoint.updated_at = "2026-09-10T19:00:00Z"
    checkpoint.safe_parameters.update(
        provider=first.provenance.provider, model=first.provenance.model
    )
    for emoji in snapshot.emojis.values():
        emoji.semantic_tags = sorted({*emoji.semantic_tags, "fragment"})
        checkpoint.elements[emoji.native_id] = checkpoint.elements[emoji.native_id].model_copy(
            update={"ai_facets_complete": True, "source_descriptor_sha256": "a" * 64}
        )
    return snapshot, checkpoint, group


def test_legacy_ai_word_is_removed_without_verified_composition(tmp_path):
    snapshot, checkpoint, group = _legacy(tmp_path)
    group["verified"] = False
    assert mark_saved_fragments(snapshot, checkpoint) == set(snapshot.emojis)
    assert all("fragment" not in emoji.semantic_tags for emoji in snapshot.emojis.values())
    assert mark_saved_fragments(snapshot, checkpoint) == set()


def test_legacy_verified_fragments_are_net_idempotent(tmp_path):
    snapshot, checkpoint, _ = _legacy(tmp_path)
    original = {key: value.as_dict() for key, value in snapshot.emojis.items()}
    assert mark_saved_fragments(snapshot, checkpoint) == set()
    assert mark_saved_fragments(snapshot, checkpoint) == set()
    assert {key: value.as_dict() for key, value in snapshot.emojis.items()} == original


@pytest.mark.parametrize(
    "protected",
    [
        "marker-version",
        "future-marker-version",
        "inherited",
        "outside-run",
        "missing-time",
        "provider",
        "model",
        "media",
        "incomplete-ai",
        "missing-descriptor",
        "missing-membership",
        "human",
        "mixed",
        "approved",
        "rejected",
        "changes_requested",
    ],
)
def test_legacy_cleanup_does_not_remove_unowned_or_reviewed_marker(tmp_path, protected):
    snapshot, checkpoint, _ = _legacy(tmp_path)
    for emoji in snapshot.emojis.values():
        if protected == "marker-version":
            checkpoint.safe_parameters["public_fragment_marker_version"] = 1
        elif protected == "future-marker-version":
            checkpoint.safe_parameters["public_fragment_marker_version"] = 2
        elif protected == "inherited":
            checkpoint.created_at = "2026-09-10T18:01:00Z"
        elif protected == "outside-run":
            checkpoint.updated_at = "2026-09-10T17:01:00Z"
        elif protected == "missing-time":
            checkpoint.created_at = None
        elif protected == "provider":
            checkpoint.safe_parameters["provider"] = "different"
        elif protected == "model":
            checkpoint.safe_parameters["model"] = "different"
        elif protected in {"media", "incomplete-ai", "missing-descriptor"}:
            update = {
                "media": {"media_sha256": ("0" * 64,)},
                "incomplete-ai": {"ai_facets_complete": False},
                "missing-descriptor": {"source_descriptor_sha256": None},
            }[protected]
            checkpoint.elements[emoji.native_id] = checkpoint.elements[emoji.native_id].model_copy(
                update=update
            )
        elif protected == "missing-membership":
            checkpoint.safe_parameters["source_memberships"] = {}
        elif protected in {"human", "mixed"}:
            snapshot.emojis[emoji.id] = emoji.model_copy(
                update={
                    "provenance": emoji.provenance.model_copy(
                        update={"origin": type(emoji.provenance.origin)(protected)}
                    )
                }
            )
        else:
            snapshot.emojis[emoji.id] = emoji.model_copy(
                update={
                    "review": emoji.review.model_copy(
                        update={"status": type(emoji.review.status)(protected)}
                    )
                }
            )
    assert strip_legacy_fragment_tags(snapshot, checkpoint) == set()
    assert all("fragment" in emoji.semantic_tags for emoji in snapshot.emojis.values())
