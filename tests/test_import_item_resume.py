from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

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


async def test_import_persists_each_file_and_resumes_without_reanalysis(
    tmp_path_factory, monkeypatch
):
    tmp_path = tmp_path_factory.mktemp("ir")
    snapshot = write_fixture(tmp_path / "data")
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
    payloads = {"first": _PAYLOAD}
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
    assert "third" not in checkpoint.elements
    assert downloads == ["first", "second"]  # Stop queued work after the first failure.
    assert checkpoint.safe_parameters["source_memberships"][source.native_id] == [
        "first",
        "second",
        "third",
    ]

    payloads["second"] = _PAYLOAD
    payloads["third"] = _PAYLOAD
    result = await runner._run_import(
        (source.canonical_url,),
        runner.PipelineOptions(),
        resume_checkpoint=checkpoint,
        expected_hashes={key: element.media_sha256 for key, element in checkpoint.elements.items()},
    )
    assert not result.errors
    assert processors[-1].decode_calls == 2
    assert downloads == ["first", "second", "first", "second", "third"]
    checkpoint = RunStore(config.runs_dir).load(result.run_id)
    assert len(checkpoint.elements) == 3
    assert backend_checks == ["webp"]
