from __future__ import annotations

import asyncio
import json
from io import StringIO

import pytest
from rich.console import Console
from rich.text import Text

from mojilex_cli.commands import progress, runtime


def test_saved_descriptions_display_full_pack_in_compact_view():
    counter = progress.BatchProgress("AI", 0, batch_total=0, pack_total=200)
    counter.cached = 200
    assert "200/200" in counter.compact_view().plain
    assert "0/0" not in counter.compact_view().plain


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


async def test_media_retry_is_visible_without_ai_label(monkeypatch):
    reports = []
    monkeypatch.setattr(progress, "report_progress", reports.append)
    async with progress.BatchProgress("Media", 2) as counter:
        counter.phase("item", "media_retry")
        counter._report()
        assert "retrying download / processing: 1" in reports[-1]
        assert "AI request" not in reports[-1]
        assert "retry 1" in counter.compact_view().plain
        assert counter.retry_events == 1


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

    def capture(view, *, key=None):
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


def test_parallel_panels_finish_independently_and_preserve_other_approval(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    terminal = Console(file=StringIO(), force_terminal=True, width=100)
    monkeypatch.setattr(runtime, "Console", lambda **kwargs: terminal)
    context = runtime._CommandContext("test")
    token = runtime._COMMAND_CONTEXT.set(context)
    try:
        runtime.update_live_progress(Text("pack A"), key="a")
        runtime.update_live_progress(Text("pack B"), key="b")
        assert set(context.progress_views) == {"a", "b"}
        runtime.pause_live_progress(True, key="a")
        runtime.pause_live_progress(False, key="b")
        assert context.progress_paused
        with runtime.suspend_progress():
            assert context.progress_paused
        assert context.progress_pause_keys == {"a"}
        runtime.finish_live_progress(key="b")
        assert set(context.progress_views) == {"a"}
        assert context.progress_paused
        runtime.pause_live_progress(False, key="a")
        runtime.update_live_progress(Text("pack A done"), key="a")
        assert context.live is not None
        runtime.finish_live_progress(key="a")
        assert context.live is None
        assert context.progress_view is None
        assert context.progress_views == {}
    finally:
        runtime.finish_live_progress()
        runtime._COMMAND_CONTEXT.reset(token)


def test_finishing_one_parallel_panel_does_not_stop_live_peer(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    terminal = Console(file=StringIO(), force_terminal=True, width=100)
    monkeypatch.setattr(runtime, "Console", lambda **kwargs: terminal)
    context = runtime._CommandContext("test")
    token = runtime._COMMAND_CONTEXT.set(context)
    try:
        runtime.update_live_progress(Text("pack A"), key="a")
        live = context.live
        runtime.update_live_progress(Text("pack B"), key="b")
        runtime.finish_live_progress(key="a")
        assert context.live is live
        assert set(context.progress_views) == {"b"}
    finally:
        runtime.finish_live_progress()
        runtime._COMMAND_CONTEXT.reset(token)


def test_finishing_nonterminal_peer_keeps_other_approval_paused():
    context = runtime._CommandContext("test")
    token = runtime._COMMAND_CONTEXT.set(context)
    try:
        runtime.pause_live_progress(True, key="a")
        runtime.finish_live_progress(key="b")
        assert context.progress_paused
        assert context.progress_pause_keys == {"a"}
    finally:
        runtime.finish_live_progress()
        runtime._COMMAND_CONTEXT.reset(token)


def test_many_pack_stages_fit_short_terminal_and_resize(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    terminal = Console(file=StringIO(), force_terminal=True, width=90, height=12)
    monkeypatch.setattr(runtime, "Console", lambda **kwargs: terminal)
    context = runtime._CommandContext("test")
    token = runtime._COMMAND_CONTEXT.set(context)
    try:
        counters = [progress.BatchProgress(f"Pack {index}", 100) for index in range(30)]
        for counter in counters:
            counter.phase("item", "download")
        lines = terminal.render_lines(context.progress_view, terminal.options, pad=False)
        assert len(lines) <= 7
        rendered = "\n".join("".join(segment.text for segment in line) for line in lines)
        assert "Pack 0: 0/100" in rendered
        assert "active 1" in rendered
        assert "Active stages: 30" in rendered
        terminal.size = (60, 8)
        lines = terminal.render_lines(context.progress_view, terminal.options, pad=False)
        assert len(lines) <= 3
        assert len(context.progress_views) == 30  # Hidden rows keep their real state.
    finally:
        runtime.finish_live_progress()
        runtime._COMMAND_CONTEXT.reset(token)


def test_background_completion_and_messages_wait_until_prompt_returns(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = StringIO()
    terminal = Console(file=output, force_terminal=True, width=100)
    monkeypatch.setattr(runtime, "Console", lambda **kwargs: terminal)
    context = runtime._CommandContext("test")
    token = runtime._COMMAND_CONTEXT.set(context)
    try:
        runtime.update_live_progress(Text("Pack A"), key="a")
        runtime.update_live_progress(Text("Pack B"), key="b")
        with runtime.suspend_progress():
            terminal.print("Confirm?")
            before = output.getvalue()
            runtime.report_progress("Pack A finished while waiting")
            runtime.finish_live_progress(key="a")
            runtime.update_live_progress(Text("Pack B working"), key="b")
            assert output.getvalue() == before
        assert "Pack A finished while waiting" in output.getvalue()
        assert not context.pending_progress
    finally:
        runtime.finish_live_progress()
        runtime._COMMAND_CONTEXT.reset(token)


async def test_finished_batch_keeps_one_line_in_scrollback(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = StringIO()
    terminal = Console(file=output, force_terminal=True, width=100)
    monkeypatch.setattr(runtime, "Console", lambda **kwargs: terminal)
    context = runtime._CommandContext("test")
    token = runtime._COMMAND_CONTEXT.set(context)
    try:
        async with progress.BatchProgress("Media", 2) as counter:
            counter.finish("a")
            counter.finish("b")
        # Rich erases the transient panel and retains its compact final summary.
        assert (
            output.getvalue()
            .rstrip()
            .endswith("Media: 2/2 | active 0 | retry 0 | errors 0 | 00:00")
        )
    finally:
        runtime.finish_live_progress()
        runtime._COMMAND_CONTEXT.reset(token)
