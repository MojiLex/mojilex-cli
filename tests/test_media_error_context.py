import asyncio

import pytest

from mojilex_cli.commands.runtime import CommandError, public_source_error, structured_exception
from mojilex_cli.media import TemporaryMediaRun
from mojilex_cli.media.models import MediaRenderError
from mojilex_cli.media.processor import SourceChangedDuringRunError
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import RunStore, new_checkpoint
from mojilex_cli.sources.base import SourceRateLimitError
from test_add_run_staging import saved_add  # noqa: F401
from test_dataset_helpers import write_fixture
from test_pack_describe_pipeline import pipeline  # noqa: F401
from test_pipeline_resume_cache import _analysis, _collection, _CountingProcessor, _item


@pytest.mark.parametrize("error_type", [MediaRenderError, SourceRateLimitError])
async def test_failed_media_preserves_exception_and_saves_public_identity(
    tmp_path, monkeypatch, error_type
):
    snapshot = write_fixture(tmp_path / "dataset")
    item = _item("7000000000000000001", unique_id="transient-unique", file_id="transient-file")
    source = _collection((item,), native_id="PublicPack")
    error = error_type("synthetic media failure")
    calls = 0
    messages = []
    real_sleep = asyncio.sleep

    async def no_backoff(_):
        await real_sleep(0)

    class Adapter:
        async def fetch_media(self, _item):
            nonlocal calls
            calls += 1
            raise error
            yield b""  # pragma: no cover

    monkeypatch.setattr(runner.asyncio, "sleep", no_backoff)
    monkeypatch.setattr(runner, "report_progress", messages.append)
    with TemporaryMediaRun(root=tmp_path) as temporary:
        with pytest.raises(error_type) as caught:
            await runner._process_media(
                Adapter(),
                source,
                _CountingProcessor(temporary, _analysis(snapshot)),
                concurrency=1,
                max_attempts=2,
            )
    assert caught.value is error
    assert calls == (2 if error_type is MediaRenderError else 1)
    diagnostic = structured_exception(error)
    assert diagnostic.source == source.canonical_url
    assert diagnostic.entity_id == item.native_id
    assert diagnostic.code == error.code
    assert diagnostic.retryable == bool(getattr(error, "retryable", False))
    assert any(f"PublicPack {item.native_id}: {error.code}:" in text for text in messages)

    checkpoint = new_checkpoint(
        command="import",
        safe_parameters={},
        cli_version="0.2.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    checkpoint = runner._checkpoint_issue(checkpoint, diagnostic)
    store = RunStore(tmp_path / "runs")
    path = store.save(checkpoint)
    issue = store.load(checkpoint.run_id).issues[-1]
    assert (issue.source, issue.entity_id) == (source.canonical_url, item.native_id)
    assert "transient-unique" not in path.read_text()
    assert "transient-file" not in path.read_text()


@pytest.mark.parametrize("terminal", [False, True])
async def test_pack_failure_before_media_identification_has_public_source(
    request, monkeypatch, terminal
):
    state = request.getfixturevalue("pipeline")
    error_type = SourceChangedDuringRunError if terminal else MediaRenderError

    async def fail(*_args, **_kwargs):
        raise error_type("synthetic pack preparation failure")

    monkeypatch.setattr(runner, "_prepare_collection_media", fail)
    if terminal:
        with pytest.raises(error_type) as caught:
            await state.run()
        diagnostic = structured_exception(caught.value)
        assert diagnostic.source == state.sources[0].canonical_url
    else:
        result = await state.run()
        assert result.errors[0].source == state.sources[0].canonical_url
        assert result.errors[0].entity_id is None
    issue = state.latest_checkpoint().issues[-1]
    assert issue.source == state.sources[0].canonical_url
    assert issue.entity_id is None


def test_command_error_context_preserves_explicit_fields():
    error = CommandError(
        "MEDIA_RENDER_FAILED",
        "synthetic",
        hint="specific retry hint",
        retryable=True,
        source="https://t.me/addemoji/OriginalPack",
        entity_id="7000000000000000002",
        details={"phase": "render"},
    )
    error._mojilex_source = "https://t.me/addemoji/OtherPack"
    error._mojilex_entity_id = "7000000000000000001"
    assert structured_exception(error) == error.error


@pytest.mark.parametrize(
    "source",
    [
        "https://private.example/download",
        "https://" + "fixture:fixture@" + "t.me/addemoji/PublicPack",
        "https://t.me/addemoji/PublicPack?credential=private",
        "https://t.me/addemoji/" + "a" * 65,
        "not a pack",
    ],
)
def test_error_context_rejects_private_or_invalid_sources(source):
    error = MediaRenderError("synthetic")
    error._mojilex_source = source
    error._mojilex_entity_id = "7000000000000000001"
    diagnostic = structured_exception(error)
    assert diagnostic.source is None
    assert diagnostic.entity_id is None


@pytest.mark.parametrize("entity_id", ["transient-file", "1" * 21, "١٢٣", 123])
def test_error_context_rejects_nondecimal_or_unbounded_identifiers(entity_id):
    original = structured_exception(MediaRenderError("synthetic"))
    diagnostic = public_source_error(original, "PublicPack", entity_id)
    assert diagnostic.source == "https://t.me/addemoji/PublicPack"
    assert diagnostic.entity_id is None
