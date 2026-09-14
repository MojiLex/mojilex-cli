import asyncio
import threading

import pytest

from mojilex_cli.concurrency import batch_limits, current_batch_limits, run_blocking
from mojilex_cli.media import resume


async def test_cancelled_disk_work_finishes_before_owner_can_cleanup(tmp_path):
    started = threading.Event()
    finish = threading.Event()
    destination = tmp_path / "result"

    def write():
        started.set()
        assert finish.wait(3)
        destination.write_text("durable")

    task = asyncio.create_task(run_blocking(write))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert destination.read_text() == "durable"


async def test_blocking_work_inherits_operation_resource_limits():
    with batch_limits(downloads=1, renders=1, ai=1, max_temp_bytes=100) as limits:
        assert await run_blocking(current_batch_limits) is limits


def test_retained_inventory_checks_directories_once_and_counts_all_bytes(tmp_path, monkeypatch):
    for index in range(3):
        directory = tmp_path / str(index)
        directory.mkdir()
        for frame in range(20):
            (directory / str(frame)).write_bytes(b"frames")
    checked = []
    original = resume._safe_path

    def check(path):
        checked.append(path)
        return original(path)

    monkeypatch.setattr(resume, "_safe_path", check)
    store = resume.RetainedMediaStore(tmp_path, 10000)
    assert store.size_bytes == 3 * 20 * 6
    assert len(checked) <= 6
