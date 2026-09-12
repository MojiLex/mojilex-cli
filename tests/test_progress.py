from __future__ import annotations

import asyncio
import json

import pytest

from mojilex_cli.commands import progress, runtime


async def test_heartbeat_reports_active_work_without_inventing_completion(monkeypatch):
    reports = []
    monkeypatch.setattr(progress, "report_progress", reports.append)
    async with progress.BatchProgress("Media", 2, interval=0.01) as counter:
        counter.phase("first", "download")
        await asyncio.sleep(0.025)
        assert any("0/2" in line and "downloading: 1" in line for line in reports)
        counter.phase("first", "render")
        counter.finish("first")
        counter.phase("second", "download")
        counter.finish("second", failed=True)
    assert "1/2 (50%)" in reports[-1]
    assert "errors: 1" in reports[-1]
    count = len(reports)
    await asyncio.sleep(0.02)
    assert len(reports) == count


async def test_cancelled_progress_stops_heartbeat(monkeypatch):
    reports = []
    monkeypatch.setattr(progress, "report_progress", reports.append)
    with pytest.raises(asyncio.CancelledError):
        async with progress.BatchProgress("Media", 10, interval=0.01):
            raise asyncio.CancelledError
    assert "0/10" in reports[-1] and "stopped" in reports[-1]
    count = len(reports)
    await asyncio.sleep(0.02)
    assert len(reports) == count


@pytest.mark.parametrize("quiet", (False, True))
def test_progress_preserves_json_stdout_and_quiet(capsys, quiet):
    async def operation():
        async with progress.BatchProgress("Media", 1) as counter:
            counter.phase("item", "render")
            counter.finish("item")
        return runtime.CommandResult()

    runtime.execute("test", lambda: asyncio.run(operation()), json_output=True, quiet=quiet)
    captured = capsys.readouterr()
    assert json.loads(captured.out)["ok"] is True
    assert ("1/1" in captured.err) is (not quiet)
