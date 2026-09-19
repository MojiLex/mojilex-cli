from io import StringIO
from types import SimpleNamespace

from rich.console import Console
from rich.text import Text

from mojilex_cli.commands import runtime
from mojilex_cli.commands.progress import BatchProgress
from mojilex_cli.commands.queue_progress import PackQueue, pack_queue_scope
from mojilex_cli.i18n import use_ui_language


def test_footer_follows_requests_then_save_then_no_active_work():
    output = StringIO()
    console = Console(file=output, width=120)
    queue = PackQueue(stages={"A": "ai"})
    batch = BatchProgress("AI", 2, batch_total=2)
    batch.active = {"one": "request", "two": "request"}
    queue.batches[batch] = "A"
    context = runtime._CommandContext(
        "test",
        pack_queue=queue,
        progress_view=Text("dashboard"),
        progress_note="AI batch 2/2 — waiting for response",
        live=SimpleNamespace(update=lambda view, **kwargs: console.print(view)),
    )

    def render():
        output.seek(0)
        output.truncate()
        context.last_pack_refresh = 0
        runtime._refresh_live(context)
        return output.getvalue()

    with use_ui_language("en"):
        assert "awaiting response: 2" in render()
        assert queue.groups()["ai"] == ["A"]
        batch.active = {"one": "save"}
        assert "saving results: 1" in render()
        assert "awaiting response" not in render()
        assert queue.groups()["ai"] == []
        assert queue.groups()["render"] == []
        assert queue.groups()["finalize"] == ["A"]
        batch.active.clear()
        queue.batches.clear()
        queue.stages["A"] = "merge_wait"
        assert "response" not in render()


def test_provider_disclosure_once_per_operation_and_per_model(monkeypatch):
    output = StringIO()
    console = Console(file=output, width=200)
    monkeypatch.setattr(runtime, "Console", lambda **kwargs: console)
    message = "Derived contact-sheet PNG provider=gemini model=first"
    for _ in range(2):
        with pack_queue_scope():
            for run in range(3):
                token = runtime._COMMAND_CONTEXT.set(runtime._CommandContext(str(run)))
                try:
                    runtime.report_progress(message)
                    runtime.report_progress(message.replace("first", "second"))
                finally:
                    runtime._COMMAND_CONTEXT.reset(token)
    assert output.getvalue().count("model=first") == 2
    assert output.getvalue().count("model=second") == 2
