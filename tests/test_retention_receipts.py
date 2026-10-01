"""Startup pressure uses exact completed receipts, never active cache entries."""

import copy
import json
from pathlib import Path

import pytest

from mojilex_cli.concurrency import ByteBudgetExceeded, batch_limits
from mojilex_cli.media.models import MediaLimits
from mojilex_cli.media.resume import RetainedMediaStore, get_retained_store
from mojilex_cli.media.temporary import TemporaryMediaRun
from mojilex_cli.pipeline.retention import (
    COMPLETED_MEDIA_PARAMETER,
    completed_media_receipt,
    restore_completed_media_receipts,
)
from mojilex_cli.runs import ElementCheckpoint, new_checkpoint
from test_media_resume_frames import KEY, _expected, _media


def _checkpoint(media, *, stage="validated"):
    return new_checkpoint(
        command="add",
        safe_parameters={COMPLETED_MEDIA_PARAMETER: {"item": completed_media_receipt(KEY, media)}},
        cli_version="0.2.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    ).model_copy(
        update={
            "elements": {
                "item": ElementCheckpoint(
                    stage=stage,
                    source_descriptor_sha256=KEY,
                    media_sha256=(media.metadata.sha256,),
                )
            }
        }
    )


def test_completed_receipt_roundtrip_has_no_media_paths(tmp_path: Path) -> None:
    media = _media(tmp_path)
    checkpoint = _checkpoint(media)
    receipt = checkpoint.safe_parameters[COMPLETED_MEDIA_PARAMETER]["item"]
    assert str(tmp_path) not in json.dumps(receipt)
    assert "frame_paths" not in receipt
    restored = restore_completed_media_receipts(checkpoint)
    assert restored[KEY].metadata == media.metadata
    assert restored[KEY].frame_paths == restored[KEY].dark_frame_paths == ()
    assert restored[KEY].semantic_frame_count == 1
    assert restored[KEY].semantic_has_dark_render
    store = RetainedMediaStore(tmp_path / "retained", 1_000_000)
    assert store.put(KEY, media)
    assert store.mark_completed(KEY, restored[KEY])
    assert store.reclaim_completed(1) > 0
    assert not (store.root / KEY).exists()


@pytest.mark.parametrize("stage", ["ai_cached", "fingerprint_ready", "failed", "discovered"])
def test_unfinished_checkpoint_does_not_grant_reclamation(tmp_path, stage) -> None:
    assert not restore_completed_media_receipts(_checkpoint(_media(tmp_path), stage=stage))


@pytest.mark.parametrize("status", ["succeeded", "noop", "interrupted", "partial", "running"])
def test_final_candidate_stage_requires_completed_run(tmp_path, status) -> None:
    checkpoint = _checkpoint(_media(tmp_path), stage="candidate_scanned").model_copy(
        update={"status": status}
    )
    assert bool(restore_completed_media_receipts(checkpoint)) is (status in {"succeeded", "noop"})


@pytest.mark.parametrize(
    "change",
    [
        "version_bool",
        "descriptor",
        "hash",
        "count_bool",
        "count_zero",
        "count_large",
        "static_count",
        "dark_int",
        "metadata_int_string",
        "extra_paths",
        "deep",
        "large",
    ],
)
def test_malformed_or_mismatched_receipts_are_ignored(tmp_path, change) -> None:
    checkpoint = _checkpoint(_media(tmp_path))
    parameters = copy.deepcopy(checkpoint.safe_parameters)
    receipt = parameters[COMPLETED_MEDIA_PARAMETER]["item"]
    if change == "version_bool":
        receipt["format_version"] = True
    elif change == "descriptor":
        receipt["source_descriptor_sha256"] = "b" * 64
    elif change == "hash":
        receipt["metadata"]["sha256"] = "b" * 64
    elif change.startswith("count_"):
        receipt["frame_count"] = {"count_bool": True, "count_zero": 0, "count_large": 17}[change]
    elif change == "static_count":
        receipt["frame_count"] = 2
    elif change == "dark_int":
        receipt["has_dark_render"] = 1
    elif change == "metadata_int_string":
        receipt["metadata"]["width"] = "512"
    elif change == "extra_paths":
        receipt["frame_paths"] = ["personal.txt"]
    elif change == "deep":
        nested = {}
        for _ in range(100):
            nested = {"nested": nested}
        receipt["analysis"] = nested
    else:
        receipt["analysis"] = "x" * 20_000
    assert not restore_completed_media_receipts(
        checkpoint.model_copy(update={"safe_parameters": parameters})
    )


def test_initial_admission_reclaims_only_receipted_completed_bytes_before_charging(
    tmp_path: Path,
) -> None:
    media = _media(tmp_path)
    root = tmp_path / "retained"
    seed = RetainedMediaStore(root, 1_000_000)
    assert seed.put(KEY, media)
    assert seed.put("b" * 64, media)
    total = seed.size_bytes
    completed = restore_completed_media_receipts(_checkpoint(media))
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=total) as limits:
        limits.temp_budget.adjust(total // 2)
        with TemporaryMediaRun(root=tmp_path) as run:
            store = get_retained_store(root, max_bytes=total, run=run, completed=completed)
            assert not (root / KEY).exists()
            assert (root / ("b" * 64)).is_dir()
            assert store.size_bytes == total // 2
            assert limits.temp_budget.used == total
        limits.temp_budget.adjust(-total // 2)
        assert limits.temp_budget.used == store.size_bytes


def test_oversized_store_startup_can_reclaim_exact_receipt_without_negative_accounting(
    tmp_path: Path,
) -> None:
    media = _media(tmp_path)
    root = tmp_path / "retained"
    seed = RetainedMediaStore(root, 1_000_000)
    assert seed.put(KEY, media)
    assert seed.put("b" * 64, media)
    maximum = seed.size_bytes // 2
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=maximum) as limits:
        with TemporaryMediaRun(root=tmp_path) as run:
            store = get_retained_store(
                root,
                max_bytes=maximum,
                run=run,
                completed=restore_completed_media_receipts(_checkpoint(media)),
            )
            assert limits.temp_budget.used == store.size_bytes == maximum
            assert (root / ("b" * 64)).is_dir()


def test_stale_startup_receipts_never_remark_active_admitted_paths(tmp_path: Path) -> None:
    media = _media(tmp_path)
    root = tmp_path / "retained"
    seed = RetainedMediaStore(root, 1_000_000)
    assert seed.put(KEY, media)
    completed = restore_completed_media_receipts(_checkpoint(media))
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=seed.size_bytes) as limits:
        with TemporaryMediaRun(root=tmp_path) as run:
            store = get_retained_store(
                root, max_bytes=seed.size_bytes, run=run, completed=completed
            )
            assert store.get(KEY, _expected(media)) is not None
            assert (
                get_retained_store(root, max_bytes=seed.size_bytes, run=run, completed=completed)
                is store
            )
            with pytest.raises(ByteBudgetExceeded):
                limits.temp_budget.adjust(1)
            assert (root / KEY).is_dir()


def test_non_batch_initial_admission_reclaims_before_individual_reservation(tmp_path: Path) -> None:
    media = _media(tmp_path)
    root = tmp_path / "retained"
    seed = RetainedMediaStore(root, 1_000_000)
    assert seed.put(KEY, media)
    assert seed.put("b" * 64, media)
    total = seed.size_bytes
    with TemporaryMediaRun(root=tmp_path, limits=MediaLimits(max_run_temp_bytes=total)) as run:
        run.reserve_retained_bytes(total // 2)
        store = get_retained_store(
            root,
            max_bytes=total,
            run=run,
            completed=restore_completed_media_receipts(_checkpoint(media)),
        )
        assert run.available_temp_bytes == 0
        assert store.size_bytes == total // 2
        assert (root / ("b" * 64)).is_dir()
