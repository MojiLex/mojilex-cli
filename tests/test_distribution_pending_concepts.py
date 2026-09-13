import json
from pathlib import Path

import pytest

from mojilex_cli.dataset import load_dataset
from mojilex_cli.dataset.distribution import _eligible_search
from mojilex_cli.dataset.staging import AtomicDatasetWriter
from mojilex_cli.domain import ConceptMappingStatus, Review, reviewed_content_sha256
from test_build_index_determinism import _build, _prepare_distribution_fixture, _tree_bytes


def _set_mapping(dataset: Path, *, complete: bool, approved: bool) -> None:
    snapshot = load_dataset(dataset)
    emoji = next(iter(snapshot.emojis.values())).model_copy(
        update={
            "concept_ids": ["animal.cat"] if complete else [],
            "concept_mapping_status": (
                ConceptMappingStatus.COMPLETE if complete else ConceptMappingStatus.PENDING
            ),
        }
    )
    snapshot.emojis[emoji.id] = emoji
    if approved:
        emoji.review = Review(
            status="approved",
            reviewer="qa-reviewer",
            reviewed_at="2026-09-10T18:00:00Z",
            reviewed_content_sha256=reviewed_content_sha256(emoji),
            review_hash_profile_id="semantic-review-content-v3",
        )
    writer = AtomicDatasetWriter(dataset)
    for path, payload in snapshot.to_files().items():
        writer.stage_bytes(path, payload)
    writer.commit()


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("complete", [False, True])
def test_release_retains_pending_canonical_records_until_search_mapping_is_complete(
    tmp_path: Path, approved: bool, complete: bool
) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    _set_mapping(dataset, complete=complete, approved=approved)
    output = tmp_path / "dist"

    before = _tree_bytes(dataset, include_transaction_lock=False)
    _build(dataset, output)
    canonical = json.loads((output / "emojis.jsonl").read_bytes())
    active = json.loads((output / "emojis-active.jsonl").read_bytes())
    assert active == canonical
    assert canonical["concept_mapping_status"] == ("complete" if complete else "pending")
    assert canonical["concept_ids"] == (["animal.cat"] if complete else [])
    assert canonical["review"]["status"] == ("approved" if approved else "unreviewed")
    assert (output / "memberships.jsonl").read_bytes()
    for language in ("en", "ru"):
        search = (output / f"search-{language}.jsonl").read_bytes()
        if complete:
            assert json.loads(search)["semantic"]["concept_ids"] == ["animal.cat"]
        else:
            assert search == b""
    assert _tree_bytes(dataset, include_transaction_lock=False) == before


def test_rebuild_removes_stale_search_rows_without_discarding_pending_records(
    tmp_path: Path,
) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    _set_mapping(dataset, complete=True, approved=True)
    output = tmp_path / "dist"
    _build(dataset, output)
    assert (output / "search-en.jsonl").read_bytes()
    _set_mapping(dataset, complete=False, approved=True)

    _build(dataset, output)
    assert not (output / "search-en.jsonl").read_bytes()
    assert json.loads((output / "emojis-active.jsonl").read_bytes())["concept_mapping_status"] == (
        "pending"
    )


@pytest.mark.parametrize(
    ("status", "concept_ids", "expected"),
    [("complete", ["animal.cat"], True), ("complete", [], False), ("pending", [], False)],
)
def test_search_requires_both_complete_mapping_and_nonempty_concepts(
    status: str, concept_ids: list[str], expected: bool
) -> None:
    assert (
        _eligible_search({"concept_mapping_status": status, "concept_ids": concept_ids}) is expected
    )
