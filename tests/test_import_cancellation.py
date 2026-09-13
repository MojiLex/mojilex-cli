from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

import pytest

from mojilex_cli.config import MojiLexConfig, TelegramConfig
from mojilex_cli.dataset.validation import validate_dataset
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import RunStore
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import (
    _PAYLOAD,
    _analysis,
    _collection,
    _CountingAdapter,
    _CountingProcessor,
    _item,
)


async def test_cancelled_import_preserves_completed_item_and_resumes(tmp_path_factory, monkeypatch):
    tmp_path = tmp_path_factory.mktemp("ic")
    snapshot = write_fixture(tmp_path / "data")
    (snapshot.root / ".gitignore").write_text(".mojilex/\n", encoding="utf-8")
    # Exercise the persistent POSIX lock artifact on Windows too.
    (snapshot.root / ".mojilex" / "locks" / "dataset-transaction-v1.lock").touch()
    for args in (
        ("init", "-b", "main"),
        ("add", "."),
        ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "base"),
        ("remote", "add", "origin", "https://github.com/MojiLex/mojilex.git"),
    ):
        await asyncio.to_thread(
            subprocess.run,
            ["git", "-C", str(snapshot.root), *args],
            capture_output=True,
            check=True,
        )
    source = _collection(
        tuple(_item(name, unique_id=name, file_id=name) for name in ("first", "second")),
        native_id="ImportCancellationTest",
    )
    blocked = asyncio.Event()
    release = asyncio.Event()
    processors = []
    downloads = []
    run_ids = []

    class Adapter(_CountingAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__({"first": _PAYLOAD, "second": _PAYLOAD})

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def validate_credentials(self):
            pass

        def canonicalize(self, value):
            return value

        async def fetch_collection(self, _reference):
            return source

        async def fetch_media(self, item):
            downloads.append(item.native_id)
            if item.native_id == "second" and not release.is_set():
                blocked.set()
                await release.wait()
            async for chunk in super().fetch_media(item):
                yield chunk

    def processor(temporary):
        value = _CountingProcessor(temporary, _analysis(snapshot))
        processors.append(value)
        return value

    config = MojiLexConfig(
        repository={"target": str(snapshot.root), "publish": "local"},
        telegram=TelegramConfig(download_concurrency=1),
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )
    monkeypatch.setattr(runner, "load_config", lambda: config)
    monkeypatch.setattr(runner, "_resolved_config", lambda _: config)
    # Synthetic domain fixture deliberately omits the external JSON Schema files.
    monkeypatch.setattr(
        runner, "validate_dataset", lambda root, **_: validate_dataset(root, strict=False)
    )
    monkeypatch.setattr(
        runner,
        "load_credentials",
        lambda: SimpleNamespace(telegram_bot_token="fixture", github_token=None),
    )
    monkeypatch.setattr(runner, "TelegramBotAPI", Adapter)
    monkeypatch.setattr(runner, "MediaProcessor", processor)
    monkeypatch.setattr(runner, "report_run_id", run_ids.append)
    task = asyncio.create_task(
        runner._run_import((source.canonical_url,), runner.PipelineOptions())
    )
    try:
        await asyncio.wait_for(blocked.wait(), timeout=10)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    checkpoint = RunStore(config.runs_dir).load(run_ids[0])
    assert checkpoint.status == "interrupted"
    assert checkpoint.issues[-1].code == "INTERRUPTED"
    assert checkpoint.issues[-1].retryable is True
    assert set(checkpoint.elements) == {"first"}
    assert checkpoint.elements["first"].fingerprint_complete
    assert downloads == ["first", "second"]

    release.set()
    result = await runner.run_resume(checkpoint.run_id)
    assert not result.errors
    assert result.status.value == "succeeded"
    assert processors[-1].decode_calls == 1
    assert downloads == ["first", "second", "first", "second"]
    checkpoint = RunStore(config.runs_dir).load(result.run_id)
    assert checkpoint.status == "succeeded"
    assert set(checkpoint.elements) == {"first", "second"}
