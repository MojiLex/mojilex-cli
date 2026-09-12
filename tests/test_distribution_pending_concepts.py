from pathlib import Path

import pytest

from mojilex_cli.dataset import load_dataset
from mojilex_cli.dataset.distribution import DataError
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
def test_pending_concepts_fail_before_creating_snapshot(tmp_path: Path, approved: bool) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    _set_mapping(dataset, complete=False, approved=approved)
    output = tmp_path / "dist"

    with pytest.raises(DataError, match="complete concept mapping before building"):
        _build(dataset, output)

    assert not output.exists()


def test_pending_concepts_preserve_existing_release(tmp_path: Path) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    _set_mapping(dataset, complete=True, approved=True)
    output = tmp_path / "dist"
    _build(dataset, output)
    original = _tree_bytes(output)
    _set_mapping(dataset, complete=False, approved=True)

    with pytest.raises(DataError, match="incomplete concept mapping"):
        _build(dataset, output)

    assert _tree_bytes(output) == original
