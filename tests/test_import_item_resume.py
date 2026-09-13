from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

import pytest

from mojilex_cli.config import MojiLexConfig, TelegramConfig
from mojilex_cli.dataset.validation import validate_dataset
from mojilex_cli.media.models import MediaError, MediaRenderError
from mojilex_cli.media.processor import SourceChangedDuringRunError
from mojilex_cli.media.temporary import TemporaryMediaRun
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


@pytest.mark.parametrize("failure_type", [MediaRenderError, MediaError])
async def test_import_persists_each_file_and_resumes_without_reanalysis(
    tmp_path_factory, monkeypatch, failure_type
):
    tmp_path = tmp_path_factory.mktemp("ir")
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
    first = _item("first", unique_id="first", file_id="first")
    second = _item("second", unique_id="second", file_id="second")
    third = _item("third", unique_id="third", file_id="third")
    source = _collection((first, second, third), native_id="ImportResumeTest")
    payloads = {"first": _PAYLOAD, "third": _PAYLOAD}
    processors = []
    downloads = []
    backend_checks = []
    current_backends = runner._decoder_backend_candidates

    def check_backend(media_format, processor):
        backend_checks.append(media_format)
        return current_backends(media_format, processor)

    monkeypatch.setattr(runner, "_decoder_backend_candidates", check_backend)

    class Adapter(_CountingAdapter):
        def __init__(self, *args, **kwargs):
            super().__init__(payloads)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            downloads.extend(self.media_calls)

        async def validate_credentials(self):
            pass

        def canonicalize(self, value):
            return value

        async def fetch_collection(self, _reference):
            return source

        async def fetch_media(self, item):
            if item.file_id not in self.payloads:
                self.media_calls.append(item.file_id)
                raise failure_type("synthetic media failure")
            async for chunk in super().fetch_media(item):
                yield chunk

    def processor(temporary, **_kwargs):
        value = _CountingProcessor(temporary, _analysis(snapshot))
        processors.append(value)
        return value

    config = MojiLexConfig(
        repository={"target": str(snapshot.root), "publish": "local"},
        telegram=TelegramConfig(download_concurrency=1),
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )
    monkeypatch.setattr(runner, "_resolved_config", lambda _: config)
    # This data fixture has domain records, but no external JSON Schema files.
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
    result = await runner._run_import((source.canonical_url,), runner.PipelineOptions())
    checkpoint = RunStore(config.runs_dir).load(result.run_id)
    assert result.errors
    assert checkpoint.elements["first"].fingerprint_complete
    assert "second" not in checkpoint.elements
    assert checkpoint.elements["third"].fingerprint_complete
    assert downloads == ["first", "second", "third"]
    assert checkpoint.ai_requests_used == 0
    assert all(not element.ai_facets_complete for element in checkpoint.elements.values())
    assert checkpoint.safe_parameters["source_memberships"][source.native_id] == [
        "first",
        "second",
        "third",
    ]

    payloads["second"] = _PAYLOAD
    result = await runner._run_import(
        (source.canonical_url,),
        runner.PipelineOptions(),
        resume_checkpoint=checkpoint,
        expected_hashes={key: element.media_sha256 for key, element in checkpoint.elements.items()},
    )
    assert not result.errors
    assert processors[-1].decode_calls == 1
    assert downloads == ["first", "second", "third", "first", "second", "third"]
    checkpoint = RunStore(config.runs_dir).load(result.run_id)
    assert len(checkpoint.elements) == 3
    assert backend_checks == ["webp"]


@pytest.mark.parametrize("interrupted", [False, True])
async def test_terminal_media_failure_does_not_start_queued_items(tmp_path, interrupted):
    snapshot = write_fixture(tmp_path / "data")
    source = _collection(
        tuple(_item(name, unique_id=name, file_id=name) for name in ("first", "bad", "last"))
    )
    saved = []

    class Adapter(_CountingAdapter):
        async def fetch_media(self, item):
            if item.file_id == "bad":
                self.media_calls.append(item.file_id)
                if interrupted:
                    raise asyncio.CancelledError
                raise SourceChangedDuringRunError("source changed during saved import")
            async for chunk in super().fetch_media(item):
                yield chunk

    async def save(item, _value):
        saved.append(item.native_id)

    adapter = Adapter({"first": _PAYLOAD, "last": _PAYLOAD})
    expected_error = asyncio.CancelledError if interrupted else SourceChangedDuringRunError
    with TemporaryMediaRun(root=tmp_path) as temporary:
        with pytest.raises(expected_error):
            await runner._process_media(
                adapter,
                source,
                _CountingProcessor(temporary, _analysis(snapshot)),
                concurrency=1,
                on_item_completed=save,
            )
    assert saved == ["first"]
    assert adapter.media_calls == ["first", "bad"]


@pytest.mark.parametrize("interrupted", [False, True])
async def test_terminal_media_error_takes_priority_over_earlier_recoverable_error(
    tmp_path, interrupted
):
    snapshot = write_fixture(tmp_path / "data")
    source = _collection(
        tuple(_item(name, unique_id=name, file_id=name) for name in ("bad", "stop", "last"))
    )

    class Adapter(_CountingAdapter):
        async def fetch_media(self, item):
            if item.file_id in {"bad", "stop"}:
                self.media_calls.append(item.file_id)
                if item.file_id == "bad":
                    raise MediaRenderError("one unrenderable file")
                if interrupted:
                    raise asyncio.CancelledError
                raise SourceChangedDuringRunError("saved source identity changed")
            async for chunk in super().fetch_media(item):
                yield chunk

    adapter = Adapter({"last": _PAYLOAD})
    expected_error = asyncio.CancelledError if interrupted else SourceChangedDuringRunError
    with TemporaryMediaRun(root=tmp_path) as temporary:
        with pytest.raises(expected_error):
            await runner._process_media(
                adapter,
                source,
                _CountingProcessor(temporary, _analysis(snapshot)),
                concurrency=1,
            )
    assert adapter.media_calls == ["bad", "stop"]
