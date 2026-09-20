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
    for stage in (
        "download",
        "render",
        "ai",
        "ai_wait",
        "composition",
        "merge_wait",
        "finalize",
        "ready",
        "failed",
    ):
        for i in range(30):
            queue.stages[f"https://t.me/addemoji/{stage}_{i}"] = stage
    output = StringIO()
    console = Console(file=output, width=width, height=height)
    with use_ui_language("en"):
        console.print(queue)
    text = output.getvalue()
    assert "Ready to send to GitHub: 30" in text
    assert len(text.splitlines()) <= height - 5
    assert "Total packs: 270" in text


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


def test_dashboard_does_not_construct_hidden_batch_tables(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    terminal = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(runtime, "Console", lambda **kw: terminal)
    context = runtime._CommandContext("test")
    token = runtime._COMMAND_CONTEXT.set(context)
    try:
        runtime.begin_pack_queue(["A"])
        batch = BatchProgress("Media", 200)
        batch.pack = "A"
        from types import SimpleNamespace

        from mojilex_cli.commands import progress

        monkeypatch.setattr(
            progress,
            "Table",
            SimpleNamespace(grid=lambda **kw: pytest.fail("Hidden table must not be built")),
        )
        for i in range(200):
            batch.phase(str(i), "download")
        assert context.pack_queue.groups()["download"] == ["A"]
    finally:
        runtime.finish_live_progress()
        runtime._COMMAND_CONTEXT.reset(token)


def test_intermediate_import_does_not_print_a_duplicate_dashboard(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    terminal = Console(file=StringIO(), force_terminal=True)
    monkeypatch.setattr(runtime, "Console", lambda **kw: terminal)
    printed = []
    original_print = terminal.print

    def capture(*args, **kwargs):
        printed.extend(value for value in args if isinstance(value, runtime._ProgressDisplay))
        original_print(*args, **kwargs)

    monkeypatch.setattr(terminal, "print", capture)
    with pack_queue_scope():
        context = runtime._CommandContext("test", command="import")
        token = runtime._COMMAND_CONTEXT.set(context)
        try:
            runtime.begin_pack_queue(["A"])
            runtime.finish_live_progress()
            assert printed == []
        finally:
            runtime._COMMAND_CONTEXT.reset(token)


def test_media_download_count_is_independent_of_decode_and_retries():
    queue = PackQueue()
    queue.register(["A"])
    media = BatchProgress("Media", 10, unit="media")
    media.cached = 2
    media.completed = 2
    media.downloaded_item("one")
    media.downloaded_item("one")  # Decoder retry must not count the same file twice.
    media.downloaded_item("two")
    media.active = {"one": "render", "three": "download"}
    queue.batches = {media: "A"}
    output = StringIO()
    with use_ui_language("en"):
        Console(file=output, width=100, height=40).print(queue)
    text = output.getvalue()
    assert "A — downloading · 2/8 · cached: 2" in text
    assert "A — processing on pc · 2/10" in text


def test_cache_backend_probe_is_not_reported_as_media_download_count():
    queue = PackQueue()
    queue.register(["A"])
    queue.stages["A"] = "render"
    queue.report_counts("A", "render", 0, 200)
    checking = BatchProgress("Checking cache", 1, unit="backends")
    checking.completed = 1
    queue.batches = {checking: "A"}
    output = StringIO()
    with use_ui_language("en"):
        Console(file=output, width=100, height=40).print(queue)
    text = output.getvalue()
    assert "Downloading: 0" in text
    assert "A — processing on pc · 0/200 · checking tools 1/1" in text


def test_intermediate_import_summary_is_silent_but_standalone_remains_visible(capsys):
    from mojilex_cli.output import OutputEnvelope

    envelope = OutputEnvelope(
        ok=True,
        command="import",
        status="noop",
        run_id="mlxrun_" + "a" * 32,
        result={"analysis_selectors": ["saved:Pack"]},
    )
    with pack_queue_scope():
        runtime._render_human(envelope)
    assert capsys.readouterr().out == ""
    runtime._render_human(envelope)
    assert "MojiLex import" in capsys.readouterr().out


def test_pack_counts_survive_backend_and_media_batch_gaps(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    terminal = Console(file=StringIO(), force_terminal=True, width=120, height=40)
    monkeypatch.setattr(runtime, "Console", lambda **kw: terminal)
    context = runtime._CommandContext("test")
    token = runtime._COMMAND_CONTEXT.set(context)
    try:
        runtime.begin_pack_queue(["Alpha", "Beta"])
        runtime.report_pack_stage("Alpha", "render")
        runtime.report_pack_counts("Alpha", "download", 0, 200)
        runtime.report_pack_counts("Alpha", "render", 25, 200, detail="checking saved media")
        assert "25/200 · checking saved media" in context.pack_queue.suffix(
            "Alpha", "render", ru=False
        )
        backend = BatchProgress("Checking cache", 2, unit="backends")
        backend.pack = "Alpha"
        backend.completed = 1
        with use_ui_language("en"):
            runtime.update_pack_progress(backend)
            runtime.finish_live_progress(key=backend)
        assert "25/200 · checking tools 1/2" in context.pack_queue.suffix(
            "Alpha", "render", ru=False
        )
        runtime.report_pack_counts("Alpha", "render", 42, 200, detail="checking saved media")
        assert "42/200 · checking saved media" in context.pack_queue.suffix(
            "Alpha", "render", ru=False
        )
        media = BatchProgress("Media", 200, unit="media")
        media.pack = "Alpha"
        media.cached = 42
        media.completed = 100
        media.downloaded = {str(i) for i in range(75)}
        runtime.update_pack_progress(media)
        runtime.finish_live_progress(key=media)
        assert "75/158" in context.pack_queue.suffix("Alpha", "download", ru=False)
        assert "100/200" in context.pack_queue.suffix("Alpha", "render", ru=False)
        assert "200" not in context.pack_queue.suffix("Beta", "download", ru=False)
        assert "fetching file list" in context.pack_queue.suffix("Beta", "download", ru=False)
    finally:
        runtime.finish_live_progress()
        runtime._COMMAND_CONTEXT.reset(token)


def test_retained_ai_count_survives_transition_to_final_validation():
    queue = PackQueue()
    queue.register(["A"])
    queue.report_counts("A", "ai", 133, 133)
    queue.stages["A"] = "finalize"
    assert "133/133" in queue.suffix("A", "finalize", ru=False)
    queue.report_counts("A", "finalize", 25, 133, detail="validating")
    assert "25/133 · validating" in queue.suffix("A", "finalize", ru=False)


@pytest.mark.parametrize("saved,completed,work", [(200, 0, 0), (170, 10, 30), (0, 10, 200)])
def test_ai_progress_uses_full_pack_and_only_verified_saved_descriptions(saved, completed, work):
    queue = PackQueue()
    queue.register(["A"])
    queue.report_counts("A", "render", 200, 200)
    ai = BatchProgress("AI", work, batch_total=0 if not work else 2, pack_total=200)
    ai.cached = saved
    ai.completed = completed
    with use_ui_language("en"):
        queue.remember_batch(ai, "A")
    for stage in ("ai", "composition", "merge_wait", "finalize"):
        suffix = queue.suffix("A", stage, ru=False)
        assert f"{saved + completed}/200" in suffix
        if saved:
            assert f"saved descriptions: {saved}" in suffix
        if stage != "ai":
            assert "descriptions " in suffix
    assert ai.total == work  # Pending-work accounting remains unchanged.


def test_zero_ai_work_does_not_invent_completed_descriptions():
    queue = PackQueue()
    queue.register(["A"])
    queue.report_counts("A", "render", 200, 200)
    ai = BatchProgress("AI", 0, batch_total=0)
    with use_ui_language("en"):
        queue.remember_batch(ai, "A")
    suffix = queue.suffix("A", "finalize", ru=False)
    assert "descriptions 0/200" in suffix
    assert "no new description work" in suffix
    assert "200/200" not in suffix
    assert " 0/0" not in suffix


def test_puzzle_and_save_wait_stages_are_visible_and_not_ready():
    queue = PackQueue()
    queue.register(["A", "B"])
    queue.stages.update(A="composition", B="merge_wait")
    queue.report_counts("A", "composition", 3, 7, detail="candidate groups")
    queue.report_counts("B", "ai", 200, 200)
    output = StringIO()
    with use_ui_language("en"):
        Console(file=output, width=120, height=40).print(queue)
    text = output.getvalue()
    assert "Checking puzzles: 1" in text
    assert "A — checking puzzles · 3/7 · candidate groups" in text
    assert "B — waiting to save · descriptions 200/200" in text
    assert "Ready to send to GitHub: 0" in text
    assert queue.groups()["ai"] == []


def test_streaming_ai_counts_accumulate_without_reset_or_double_count():
    queue = PackQueue()
    queue.register(["A"])
    first = BatchProgress("AI", 2, batch_total=1, pack_total=10)
    first.completed_item_ids = {"1", "2"}
    queue.remember_batch(first, "A")
    second = BatchProgress("AI", 2, batch_total=1, pack_total=10)
    second.completed_item_ids = set()
    second.active = {"request": "request"}
    queue.batches = {second: "A"}
    assert "2/10" in queue.suffix("A", "ai", ru=False)
    assert "requests awaiting response: 1" in queue.suffix("A", "ai", ru=False)
    second.completed_item_ids.update({"2", "3"})
    assert "3/10" in queue.suffix("A", "ai", ru=False)
    final = BatchProgress("AI", 7, batch_total=1, pack_total=10)
    final.completed_item_ids = {"1", "2", "3"}
    queue.remember_batch(final, "A")
    assert queue.counts["A"]["ai"].completed == 3


def test_overflow_uses_explicit_header_not_plus_suffix():
    queue = PackQueue()
    for index in range(60):
        queue.stages[f"Pack{index}"] = "render"
    output = StringIO()
    with use_ui_language("en"):
        Console(file=output, width=100, height=24).print(queue)
    text = output.getvalue()
    assert "Processing on PC: 60 · shown" in text
    assert "(+" not in text


@pytest.mark.asyncio
async def test_throttled_counter_update_is_not_lost(monkeypatch):
    import asyncio
    import time
    from types import SimpleNamespace

    updates = []
    context = runtime._CommandContext("test")
    context.pack_queue = PackQueue()
    context.progress_view = Text("progress")
    context.live = SimpleNamespace(update=lambda *args, **kwargs: updates.append(1))
    context.last_pack_refresh = time.monotonic()
    runtime._refresh_live(context)
    assert updates == []
    await asyncio.sleep(0.25)
    assert updates == [1]
    assert context.pack_refresh_handle is None
