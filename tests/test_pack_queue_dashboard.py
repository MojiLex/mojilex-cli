import json
from io import StringIO

import pytest
from rich.console import Console
from rich.text import Text

from mojilex_cli.commands import runtime
from mojilex_cli.commands.progress import BatchProgress
from mojilex_cli.commands.queue_progress import PACK, PackQueue, pack_queue_scope
from mojilex_cli.i18n import use_ui_language


def test_unique_packs_counted_in_each_actual_active_stage():
    queue = PackQueue()
    queue.register(["A", "B", "C", "A"])
    media = BatchProgress("media", 200)
    media.active = {"one": "download", "two": "render", "three": "download"}
    ai = BatchProgress("AI", 20)
    ai.active = {"request": "approval"}
    queue.batches = {media: "A", ai: "B"}
    queue.stages["C"] = "finalize"
    groups = queue.groups()
    assert groups["download"] == ["A"]
    assert groups["render"] == ["A"]
    assert groups["ai"] == ["B"]
    assert groups["ready"] == []
    queue.stages["C"] = "ready"
    assert queue.groups()["ready"] == ["C"]


@pytest.mark.parametrize("height,width", [(24, 80), (40, 120), (18, 45)])
def test_dashboard_keeps_summary_visible_when_many_packs_active(height, width):
    queue = PackQueue()
    for stage in ("download", "render", "ai", "ai_wait", "finalize", "ready", "failed"):
        for i in range(30):
            queue.stages[f"https://t.me/addemoji/{stage}_{i}"] = stage
    output = StringIO()
    console = Console(file=output, width=width, height=height)
    with use_ui_language("en"):
        console.print(queue)
    text = output.getvalue()
    assert "Ready to send to GitHub: 30" in text
    assert len(text.splitlines()) <= height - 5
    assert "Total packs: 210" in text


def test_confirmation_stays_still_while_background_packs_change(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    output = StringIO()
    terminal = Console(file=output, force_terminal=True, width=100, height=30)
    monkeypatch.setattr(runtime, "Console", lambda **kw: terminal)
    context = runtime._CommandContext("test")
    token = runtime._COMMAND_CONTEXT.set(context)
    try:
        runtime.begin_pack_queue(["A", "B"])
        runtime.pause_live_progress(True)
        terminal.print("Confirm spending?")
        before = output.getvalue()
        runtime.report_pack_stage("B", "render")
        runtime.report_progress("Source B: 200 media files")
        runtime.update_live_progress(Text("background"), key="batch")
        runtime.finish_live_progress(key="batch")
        assert output.getvalue() == before
        assert context.pending_progress == []
        runtime.pause_live_progress(False)
        runtime.report_progress("Derived contact-sheet PNG disclosure")
    finally:
        runtime.finish_live_progress()
        runtime._COMMAND_CONTEXT.reset(token)
    assert "Derived contact-sheet PNG disclosure" in output.getvalue()


def test_scoped_queue_preserves_counts_across_command_contexts(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    terminal = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(runtime, "Console", lambda **kw: terminal)
    with pack_queue_scope():
        for source in ("A", "B"):
            context = runtime._CommandContext(source)
            token = runtime._COMMAND_CONTEXT.set(context)
            try:
                runtime.begin_pack_queue([source])
                runtime.report_pack_stage(source, "ready")
                assert len(context.pack_queue.groups()["ready"]) == (1 if source == "A" else 2)
            finally:
                runtime.finish_live_progress()
                runtime._COMMAND_CONTEXT.reset(token)


def test_batch_captures_own_pack_identity():
    token = PACK.set("PackAlpha")
    try:
        batch = BatchProgress("AI", 200)
    finally:
        PACK.reset(token)
    assert batch.pack == "PackAlpha"


@pytest.mark.parametrize("machine", [False, True])
def test_analysis_selector_list_is_machine_only(capsys, machine):
    selectors = [f"mlxrun_parent:Pack{i}" for i in range(188)]
    with use_ui_language("en"):
        runtime.execute(
            "import",
            lambda: runtime.CommandResult(
                result={"analysis_selectors": selectors, "reused_packs": 188}
            ),
            json_output=machine,
        )
    output = capsys.readouterr().out
    if machine:
        assert json.loads(output)["result"]["analysis_selectors"] == selectors
    else:
        assert "Packs queued for analysis: 188" in output
        assert "mlxrun_parent" not in output
        assert "analysis selectors" not in output
