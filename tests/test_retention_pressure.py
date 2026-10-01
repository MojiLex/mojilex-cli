"""Only durable completed previews are optional when the shared disk fills."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from mojilex_cli.concurrency import ByteBudget, ByteBudgetExceeded, batch_limits
from mojilex_cli.media.models import MediaLimitError
from mojilex_cli.media.resume import RetainedMediaStore, get_retained_store
from mojilex_cli.media.temporary import TemporaryMediaRun
from test_media_resume_frames import KEY, _expected, _media


def test_budget_reclaims_outside_accounting_lock_and_only_under_pressure() -> None:
    budget = ByteBudget(100)
    budget.adjust(90)
    calls = []

    def reclaim(required: int) -> int:
        calls.append(required)
        assert budget.used == 90
        budget.adjust(-20)
        return 20

    budget.register_reclaimer(reclaim)
    budget.register_reclaimer(reclaim)
    budget.adjust(-10)
    budget.adjust(10)
    assert not calls
    budget.adjust(15)
    assert calls == [5]
    assert budget.used == 85


def test_impossible_reservation_does_not_remove_previews() -> None:
    budget = ByteBudget(10)
    budget.adjust(5)
    calls = []
    budget.register_reclaimer(lambda required: calls.append(required) or 0)
    with pytest.raises(ByteBudgetExceeded):
        budget.adjust(11)
    assert not calls
    assert budget.used == 5


def test_reclaimers_continue_until_real_accounting_has_enough_space() -> None:
    budget = ByteBudget(10)
    budget.adjust(10)
    calls = []

    def first(required):
        calls.append(required)
        budget.adjust(-2)
        return 2

    def second(required):
        calls.append(required)
        budget.adjust(-3)
        return 3

    budget.register_reclaimer(first)
    budget.register_reclaimer(second)
    budget.adjust(4)
    assert calls == [4, 2]
    assert budget.used == 9


def test_observed_outputs_reclaim_previews_before_recording_real_overflow() -> None:
    budget = ByteBudget(10)
    budget.adjust(10)

    def reclaim(required):
        assert required == 5
        budget.adjust(-3)
        return 3

    budget.register_reclaimer(reclaim)
    budget.adjust(5, observed=True)
    assert budget.used == 12


def test_pressure_preserves_active_and_unknown_entries_and_stops_after_enough_bytes(
    tmp_path: Path,
) -> None:
    value = _media(tmp_path)
    root = tmp_path / "retained"
    seed = RetainedMediaStore(root, 1_000_000)
    for key in (KEY, "b" * 64, "c" * 64, "d" * 64):
        assert seed.put(key, value)
    unknown = root / ("e" * 64)
    unknown.mkdir()
    (unknown / "personal.txt").write_bytes(b"keep")
    total = seed.size_bytes + 4
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=total) as limits:
        with TemporaryMediaRun(root=tmp_path) as run:
            store = get_retained_store(root, max_bytes=total, run=run)
            assert store.mark_completed("e" * 64, _expected(value))
            assert store.mark_completed("b" * 64, _expected(value))
            assert store.mark_completed("c" * 64, _expected(value))
            assert store.mark_completed("d" * 64, _expected(value))
            # Reading a previously completed item marks its paths active again.
            assert store.get("d" * 64, _expected(value)) is not None
            assert (root / ("b" * 64)).is_dir()  # Marking never removes previews.
            limits.temp_budget.adjust(1)
            assert not (root / ("b" * 64)).exists()
            assert (root / KEY).is_dir()
            assert (root / ("c" * 64)).is_dir()
            assert (root / ("d" * 64)).is_dir()
            assert (unknown / "personal.txt").read_bytes() == b"keep"
            assert limits.temp_budget.used == store.size_bytes + 1
            limits.temp_budget.adjust(-1)


def test_busy_store_reclaimer_skips_lock_owned_by_reserving_thread(tmp_path: Path) -> None:
    value = _media(tmp_path)
    root = tmp_path / "retained"
    seed = RetainedMediaStore(root, 1_000_000)
    assert seed.put(KEY, value)
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=seed.size_bytes) as limits:
        with TemporaryMediaRun(root=tmp_path) as run:
            store = get_retained_store(root, max_bytes=seed.size_bytes, run=run)
            assert store.mark_completed(KEY, _expected(value))
            with store._lock, ThreadPoolExecutor(max_workers=1) as pool:
                # The worker cannot wait for our store lock while holding the
                # serial pressure lock. It must fail safely instead of deadlock.
                future = pool.submit(limits.temp_budget.adjust, 1)
                with pytest.raises(ByteBudgetExceeded):
                    future.result(timeout=2)
            assert (root / KEY).exists()
            limits.temp_budget.adjust(1)
            assert not (root / KEY).exists()
            assert limits.temp_budget.used == 1
            limits.temp_budget.adjust(-1)


def test_rejected_store_admission_does_not_register_uncharged_reclaimer(tmp_path: Path) -> None:
    value = _media(tmp_path)
    root = tmp_path / "retained"
    seed = RetainedMediaStore(root, 1_000_000)
    assert seed.put(KEY, value)
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=1) as limits:
        with TemporaryMediaRun(root=tmp_path) as run:
            with pytest.raises(MediaLimitError, match="temporary disk limit"):
                get_retained_store(root, max_bytes=seed.size_bytes, run=run)
            assert not limits.temp_budget._reclaimers
            assert limits.temp_budget.used == 0
            assert (root / KEY).is_dir()


def test_concurrent_reservations_preserve_exact_accounting() -> None:
    budget = ByteBudget(32)
    budget.adjust(32)
    reclaimed = []

    def reclaim(required):
        if not reclaimed:
            budget.adjust(-32)
            reclaimed.append(32)
            return 32
        return 0

    budget.register_reclaimer(reclaim)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: budget.adjust(1), range(32)))
    assert reclaimed == [32]
    assert budget.used == 32
    with pytest.raises(ByteBudgetExceeded):
        budget.adjust(1)
    assert budget.used == 32


def test_long_run_reclaims_completed_previews_instead_of_accumulating_disk(
    tmp_path: Path,
) -> None:
    value = _media(tmp_path)
    seed = RetainedMediaStore(tmp_path / "seed", 1_000_000)
    assert seed.put(KEY, value)
    maximum = 3 * seed.size_bytes
    root = tmp_path / "retained"
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=maximum) as limits:
        for index in range(40):
            key = f"{index:064x}"
            with TemporaryMediaRun(root=tmp_path) as run:
                assert run.path is not None
                store = get_retained_store(root, max_bytes=maximum, run=run)
                # A new pack's generated pixels create actual shared pressure.
                frame = run.path / "rendered.png"
                frame.write_bytes(value.frame_paths[0].read_bytes())
                dark = run.path / "rendered-dark.png"
                dark.write_bytes(value.dark_frame_paths[0].read_bytes())
                run.account_outputs([frame, dark])
                current = value.model_copy(
                    update={"frame_paths": (frame,), "dark_frame_paths": (dark,)}
                )
                assert store.put(key, current)
                assert limits.temp_budget.used <= maximum
            # Mirrors the runner boundary: final public records and checkpoint
            # have been saved; the temporary media run has fully cleaned up.
            assert store.mark_completed(key, current)
        assert len(list(root.iterdir())) <= 3
        assert limits.temp_budget.used == store.size_bytes <= maximum
