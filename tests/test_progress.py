from __future__ import annotations

import asyncio
import json
from io import StringIO

import pytest
from rich.console import Console
from rich.text import Text

from mojilex_cli.commands import progress, runtime


async def test_heartbeat_reports_active_work_without_inventing_completion(monkeypatch):
    reports = []
    active_reported = asyncio.Event()

    def report(line):
        reports.append(line)
        if "0/2" in line and "downloading: 1" in line:
            active_reported.set()

    monkeypatch.setattr(progress, "report_progress", report)
    async with progress.BatchProgress("Media", 2, interval=0.01) as counter:
        counter.phase("first", "download")
        await asyncio.wait_for(active_reported.wait(), timeout=1)
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


async def test_ai_progress_distinguishes_batches_items_and_unstarted_queue(monkeypatch):
    reports = []
    monkeypatch.setattr(progress, "report_progress", reports.append)
    async with progress.BatchProgress("AI", 100, batch_total=13, interval=0.01) as counter:
        counter.phase("batch-1", "request", count=8)
        counter._report()
        assert "0/100 emojis" in reports[-1]
        assert "batches: 0/13" in reports[-1]
        assert "queued: 92 emojis" in reports[-1]
        assert "active batches: 1" in reports[-1]
        assert "waiting for AI response: 1" in reports[-1]
        counter.phase("batch-1", "retry")
        assert counter.active_counts["batch-1"] == 8
        counter.stop_queue()
        counter.finish("batch-1", count=8, failed=True)
    assert "errors: 8" in reports[-1]
    assert "not started: 92 emojis" in reports[-1]
    assert counter.failed == 8


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


async def test_live_panel_counts_items_and_distinguishes_retries_from_failures(monkeypatch):
    views = []

    def capture(view):
        output = StringIO()
        Console(file=output, width=100).print(view)
        views.append(output.getvalue())
        return True

    monkeypatch.setattr(progress, "update_live_progress", capture)
    requests_used = 61
    async with progress.BatchProgress(
        "AI", 12, batch_total=2, request_budget=lambda: (requests_used, 100)
    ) as counter:
        counter.phase("first", "request", count=8)
        counter.phase("second", "request", count=4)
        counter.phase("first", "transport_retry")
        assert "Processing 4" in " ".join(views[-1].split())
        assert "Retrying 8" in " ".join(views[-1].split())
        assert "Unresolved failures" not in views[-1]
        assert "AI requests 61 / 100" in " ".join(views[-1].split())
        requests_used = 62
        counter.phase("first", "request")
        counter.advance("first", count=3)
        counter.finish("first", count=5, failed=True)
        counter.finish("second", count=4)
    assert "7 / 12" in views[-1]
    assert "Unresolved failures" in views[-1]
    assert counter.completed == 7 and counter.failed == 5
    assert counter.retry_events == 1
    assert "AI requests 62 / 100" in " ".join(views[-1].split())


def test_live_rendering_pauses_for_confirmation_and_keeps_disclosures(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = StringIO()
    terminal = Console(file=output, force_terminal=True, width=100)
    monkeypatch.setattr(runtime, "Console", lambda **kwargs: terminal)
    context = runtime._CommandContext("test")
    token = runtime._COMMAND_CONTEXT.set(context)
    try:
        assert runtime.update_live_progress(Text("initial progress"))
        runtime.report_progress("Provider processing terms: test disclosure")
        runtime.report_progress("AI batch 1/2: 8 emojis — AI request 1/100; waiting for response")
        assert "AI request 1/100" in context.progress_note
        runtime.pause_live_progress(True)
        assert context.live is None
        terminal.print("Confirm spending?")
        before = output.getvalue()
        assert runtime.update_live_progress(Text("heartbeat while waiting"))
        runtime.report_progress("AI batch 2/2: 4 emojis — AI request 2/100; waiting for response")
        assert output.getvalue() == before
        runtime.pause_live_progress(False)
        assert runtime.update_live_progress(Text("completed progress"))
        runtime.finish_live_progress()
        assert context.live is None
    finally:
        runtime.finish_live_progress()
        runtime._COMMAND_CONTEXT.reset(token)
    assert "test disclosure" in output.getvalue()
    assert "Confirm spending?" in output.getvalue()
    assert "completed progress" in output.getvalue()


@pytest.mark.parametrize("json_output,quiet", [(True, False), (False, True)])
def test_live_rendering_never_starts_for_json_or_quiet(monkeypatch, json_output, quiet):
    terminal = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(runtime, "Console", lambda **kwargs: terminal)
    context = runtime._CommandContext("test", json_output=json_output, quiet=quiet)
    token = runtime._COMMAND_CONTEXT.set(context)
    try:
        assert not runtime.update_live_progress(Text("hidden"))
        assert context.live is None
    finally:
        runtime._COMMAND_CONTEXT.reset(token)
