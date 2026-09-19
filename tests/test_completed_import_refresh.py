import asyncio
from types import SimpleNamespace

import pytest

from mojilex_cli.commands import import_reuse
from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.config import MojiLexConfig
from mojilex_cli.pipeline.runner import _source_descriptor_sha256
from mojilex_cli.runs import ElementCheckpoint, RunStore, new_checkpoint
from mojilex_cli.runs.pack_scope import source_state
from mojilex_cli.runs.store import RunLockedError
from test_dataset_helpers import FILE_UNIQUE_ID, MEDIA_HASH, NATIVE_EMOJI_ID, write_fixture
from test_pipeline_resume_cache import _collection, _item


@pytest.fixture
def ready_pack(tmp_path, monkeypatch):
    snapshot = write_fixture(tmp_path / "dataset")
    item = _item(NATIVE_EMOJI_ID, unique_id=FILE_UNIQUE_ID, file_id="transient")
    collection = _collection((item,), native_id="SuspiciousCats")
    source = collection.canonical_url
    sibling = "https://t.me/addemoji/OtherPack"
    config = MojiLexConfig(runs_dir=tmp_path / "runs")
    checkpoint = new_checkpoint(
        command="describe",
        safe_parameters={
            "sources": [source, sibling],
            "source_memberships": {collection.native_id: [item.native_id], "OtherPack": ["other"]},
            "staging_repository": str(snapshot.root),
            "public_fragment_marker_version": 1,
            "source_states": {
                value: {"phase": "describe", "status": "succeeded"} for value in (source, sibling)
            },
            "composition_evidence": {"OtherPack": ["preserved"]},
        },
        cli_version="0.1.0",
        schema_version="1.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    ).model_copy(
        update={
            "status": "succeeded",
            "ai_requests_used": 19,
            "elements": {
                item.native_id: ElementCheckpoint(
                    stage="candidate_scanned",
                    source_descriptor_sha256=_source_descriptor_sha256(item),
                    media_sha256=(MEDIA_HASH,),
                    deterministic_cache_key="b" * 64,
                    ai_cache_key="c" * 64,
                    ai_facets_complete=True,
                    fingerprint_complete=True,
                )
            },
        }
    )
    store = RunStore(config.runs_dir)
    store.save(checkpoint)
    state = SimpleNamespace(
        checkpoint=checkpoint,
        config=config,
        store=store,
        source=source,
        sibling=sibling,
        fresh=collection,
        item=item,
        snapshot=snapshot,
        calls=[],
    )

    async def fetch(sources, config, *, concurrency):
        state.calls.append(tuple(sources))
        return {source: state.fresh}

    monkeypatch.setattr(import_reuse, "_fetch_completed_metadata", fetch)
    return state


def refresh(state):
    return import_reuse.refresh_completed_imports(
        {state.source: (state.checkpoint, state.source)}, state.config, download_concurrency=4
    )[state.source][0]


def test_ready_unchanged_only_checks_metadata_without_media_or_ai(ready_pack, monkeypatch):
    state = ready_pack

    def unexpected_save(*args):
        pytest.fail("Unchanged ready pack must not write or requeue work")

    monkeypatch.setattr(RunStore, "save", unexpected_save)
    assert refresh(state) == state.checkpoint
    assert state.calls == [(state.source,)]
    assert source_state(state.checkpoint, state.source)["phase"] == "describe"


def test_append_preserves_paid_work_and_unrequested_sibling(ready_pack):
    state = ready_pack
    new = _item("123456789", unique_id="new", file_id="new")
    state.fresh = _collection((state.item, new), native_id=state.fresh.native_id)
    updated = refresh(state)
    assert updated == state.store.load(updated.run_id)
    assert updated.elements == state.checkpoint.elements
    assert updated.ai_requests_used == 19
    assert updated.safe_parameters["composition_evidence"] == {"OtherPack": ["preserved"]}
    assert updated.safe_parameters["source_memberships"] == {
        "SuspiciousCats": [state.item.native_id, new.native_id],
        "OtherPack": ["other"],
    }
    assert source_state(updated, state.source) == {"phase": "import", "status": "succeeded"}
    assert source_state(updated, state.sibling) == {"phase": "describe", "status": "succeeded"}
    assert updated.status == "partial"


@pytest.mark.parametrize("change", ["removed", "reordered", "descriptor"])
def test_changed_existing_emoji_preserves_checkpoint(ready_pack, change):
    state = ready_pack
    if change == "removed":
        items = ()
    elif change == "reordered":
        items = (_item("123456789", unique_id="new", file_id="new"), state.item)
    else:
        items = (state.item.model_copy(update={"file_unique_id": "changed"}),)
    state.fresh = _collection(items, native_id=state.fresh.native_id)
    with pytest.raises(CommandError, match="changed, removed or reordered"):
        refresh(state)
    assert state.store.load(state.checkpoint.run_id) == state.checkpoint


@pytest.mark.parametrize("proof", ["marker", "public_media", "public_membership"])
def test_missing_final_proof_resumes_validation_preserving_cache(ready_pack, monkeypatch, proof):
    state = ready_pack
    if proof == "marker":
        safe = dict(state.checkpoint.safe_parameters)
        safe.pop("public_fragment_marker_version")
        state.checkpoint = state.checkpoint.model_copy(update={"safe_parameters": safe})
        state.store.save(state.checkpoint)
    else:
        if proof == "public_media":
            state.snapshot.emojis.clear()
        else:
            state.snapshot.memberships.clear()
        monkeypatch.setattr(import_reuse, "load_dataset", lambda _: state.snapshot)
    updated = refresh(state)
    assert updated.elements == state.checkpoint.elements
    assert source_state(updated, state.source) == {"phase": "import", "status": "succeeded"}
    assert source_state(updated, state.sibling)["status"] == "succeeded"


def test_change_during_metadata_request_rejected(ready_pack, monkeypatch):
    state = ready_pack
    changed = state.checkpoint.model_copy(update={"ai_requests_used": 20})

    async def fetch(*args, **kwargs):
        state.store.save(changed)
        return {state.source: state.fresh}

    monkeypatch.setattr(import_reuse, "_fetch_completed_metadata", fetch)
    with pytest.raises(CommandError, match="changed while checking"):
        refresh(state)
    assert state.store.load(changed.run_id) == changed


@pytest.mark.parametrize("kind", ["execution", "collection"])
def test_busy_run_or_collection_cannot_be_modified(ready_pack, kind):
    state = ready_pack
    lock = (
        state.store.execution_lock(state.checkpoint.run_id)
        if kind == "execution"
        else state.store.collection_lock("telegram", state.fresh.native_id)
    )
    with lock, pytest.raises(RunLockedError):
        refresh(state)
    assert state.store.load(state.checkpoint.run_id) == state.checkpoint


def test_metadata_adapter_never_downloads_media(monkeypatch):
    source = _collection((_item("123", unique_id="unique", file_id="transient"),))
    calls = []

    class MetadataAdapter:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def canonicalize(self, url):
            return url

        async def fetch_collection(self, reference):
            calls.append(reference)
            return source

        async def fetch_media(self, *args):
            pytest.fail("Ready pack metadata check must not download media")

    monkeypatch.setattr(import_reuse, "TelegramBotAPI", MetadataAdapter)
    monkeypatch.setattr(
        import_reuse, "load_credentials", lambda: SimpleNamespace(telegram_bot_token="fixture")
    )
    result = asyncio.run(
        import_reuse._fetch_completed_metadata(
            (source.canonical_url,), MojiLexConfig(), concurrency=4
        )
    )
    assert result == {source.canonical_url: source}
    assert calls == [source.canonical_url]


def test_transient_file_id_does_not_invalidate_ready_pack(ready_pack):
    state = ready_pack
    state.fresh = _collection(
        (state.item.model_copy(update={"file_id": "rotated"}),), native_id=state.fresh.native_id
    )
    assert refresh(state) == state.checkpoint


def test_legacy_without_membership_is_revalidated_without_inventing_plan(ready_pack):
    state = ready_pack
    safe = dict(state.checkpoint.safe_parameters)
    safe.pop("source_memberships")
    state.checkpoint = state.checkpoint.model_copy(update={"safe_parameters": safe})
    state.store.save(state.checkpoint)
    updated = refresh(state)
    assert updated.safe_parameters["source_memberships"] == {}
    assert updated.elements == state.checkpoint.elements
    assert source_state(updated, state.source)["phase"] == "import"


def test_shared_snapshot_loaded_once_and_collection_lock_deduplicated(ready_pack, monkeypatch):
    state = ready_pack
    second = state.checkpoint.model_copy(update={"run_id": "mlxrun_" + "f" * 32})
    state.store.save(second)
    alias = state.source.replace("t.me", "telegram.me")
    calls = []
    original_load = import_reuse.load_dataset

    def load(path):
        calls.append(path)
        return original_load(path)

    async def fetch(*args, **kwargs):
        return {state.source: state.fresh, alias: state.fresh}

    monkeypatch.setattr(import_reuse, "load_dataset", load)
    monkeypatch.setattr(import_reuse, "_fetch_completed_metadata", fetch)
    existing = {state.source: (state.checkpoint, state.source), alias: (second, state.source)}
    assert (
        import_reuse.refresh_completed_imports(existing, state.config, download_concurrency=4)
        == existing
    )
    assert calls == [state.snapshot.root]
