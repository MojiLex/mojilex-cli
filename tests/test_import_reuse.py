from types import SimpleNamespace

import pytest

from mojilex_cli.commands import import_reuse, interactive, packs, runtime, workflow
from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.config import MojiLexConfig
from mojilex_cli.output import RunStatus
from mojilex_cli.runs import ElementCheckpoint, new_checkpoint

A = "https://t.me/addemoji/PackAlpha"
B = "https://t.me/addemoji/PackBravo"
C = "https://t.me/addemoji/PackCharlie"


@pytest.fixture
def saved_runs(tmp_path, monkeypatch):
    runs = []
    config = MojiLexConfig(repository={"target": "MojiLex/mojilex"})
    monkeypatch.setattr(import_reuse, "_runs", lambda config: (runs, 0))
    monkeypatch.setattr(workflow, "load_config", lambda **kwargs: config)

    def create(sources=(A, B), *, phase="import", status="succeeded", repo="MojiLex/mojilex"):
        staging = tmp_path / f"staging-{len(runs)}"
        staging.mkdir()
        checkpoint = new_checkpoint(
            command=phase,
            safe_parameters={
                "sources": list(sources),
                "max_items": None,
                "staging_repository": str(staging),
                "source_states": {source: {"phase": phase, "status": status} for source in sources},
            },
            cli_version="0.1.0",
            schema_version="1.0",
            target_repository=repo,
            base_revision="a" * 40,
        ).model_copy(update={"status": status})
        runs.insert(0, checkpoint)
        return checkpoint

    return create, runs, config


def do_import(sources, **kwargs):
    return workflow.import_command(
        sources,
        repo="MojiLex/mojilex",
        platform="telegram",
        max_items=None,
        download_concurrency=4,
        check_media=True,
        fail_fast=False,
        official_pack_policy="allow",
        **kwargs,
    )


def forbid(*args, **kwargs):
    pytest.fail("A completed saved pack must not be downloaded or resumed again")


def test_completed_file_import_reused_for_individual_url(saved_runs, monkeypatch):
    create, _, _ = saved_runs
    checkpoint = create()
    monkeypatch.setattr(workflow, "run_import", forbid)
    monkeypatch.setattr(workflow, "run_resume_sync", forbid)
    result = do_import([A])
    assert result.result["analysis_selectors"] == [f"{checkpoint.run_id}:PackAlpha"]
    assert result.result["reused_packs"] == 1


def test_partial_file_import_resumes_only_requested_pack(saved_runs, monkeypatch):
    create, _, _ = saved_runs
    checkpoint = create(status="interrupted")
    monkeypatch.setattr(workflow, "run_import", forbid)
    calls = []

    def resume(run_id, **kwargs):
        calls.append((run_id, kwargs))
        return CommandResult(run_id=run_id)

    monkeypatch.setattr(workflow, "run_resume_sync", resume)
    result = do_import([B])
    assert calls[0][0] == checkpoint.run_id
    assert calls[0][1]["selected_sources"] == (B,)
    assert result.result["analysis_selectors"] == [f"{checkpoint.run_id}:PackBravo"]


def test_other_repository_and_max_items_do_not_reuse(saved_runs):
    create, runs, config = saved_runs
    create(repo="Other/dataset")
    assert import_reuse.reusable_imports([A], config, max_items=None) == {}
    checkpoint = create()
    assert import_reuse.reusable_imports([A], config, max_items=10) == {}
    runs[0] = checkpoint.model_copy(
        update={"safe_parameters": {**checkpoint.safe_parameters, "staging_repository": ""}}
    )
    # An empty staging value means cwd and must not qualify as a durable workspace.
    assert import_reuse.reusable_imports([A], config, max_items=None) == {}


def test_completed_analysis_does_not_start_ai_again(saved_runs, monkeypatch):
    create, _, _ = saved_runs
    create(phase="describe")
    # Metadata/proof refresh is covered with real saved records in its own tests.
    monkeypatch.setattr(
        import_reuse, "refresh_completed_imports", lambda existing, *a, **k: existing
    )
    monkeypatch.setattr(workflow, "run_import", forbid)
    monkeypatch.setattr(workflow, "run_resume_sync", forbid)
    result = do_import([A])
    assert result.result["analysis_selectors"] == []
    dispatched = []
    monkeypatch.setattr(interactive, "_notice", lambda message: None)

    def dispatch(arguments):
        dispatched.append(arguments)
        runtime._RESULT_COLLECTOR.get().append(result)
        return True

    interactive._analyze(A, dispatch)
    assert dispatched == [["import", A, "--preparation", "metadata"]]


def test_completed_analysis_is_preferred_over_newer_metadata_import(saved_runs):
    create, _, config = saved_runs
    completed = create(phase="describe")
    create(phase="import")
    assert import_reuse.reusable_imports([A], config, max_items=None)[A][0] == completed


def test_complete_copy_preferred_over_accidental_interrupted_reimport(saved_runs):
    create, _, config = saved_runs
    completed = create(phase="describe")
    create(status="interrupted")
    assert import_reuse.reusable_imports([A], config, max_items=None)[A][0] == completed


def test_mixed_saved_and_new_links_only_download_new_source(saved_runs, monkeypatch):
    create, _, _ = saved_runs
    checkpoint = create()
    calls = []
    fresh_id = "mlxrun_" + "b" * 32

    def download(sources, options):
        calls.append(sources)
        return CommandResult(run_id=fresh_id)

    monkeypatch.setattr(workflow, "run_import", download)
    monkeypatch.setattr(workflow, "run_resume_sync", forbid)
    result = do_import([A, C])
    assert calls == [(C,)]
    assert result.result["analysis_selectors"] == [f"{checkpoint.run_id}:PackAlpha", fresh_id]


def test_failed_resume_stops_before_downloading_new_pack(saved_runs, monkeypatch):
    create, _, _ = saved_runs
    checkpoint = create(status="interrupted")
    monkeypatch.setattr(workflow, "run_import", forbid)
    monkeypatch.setattr(
        workflow,
        "run_resume_sync",
        lambda *args, **kwargs: CommandResult(run_id=checkpoint.run_id, status=RunStatus.PARTIAL),
    )
    assert do_import([A, C]).status is RunStatus.PARTIAL


def test_failed_new_pack_stops_before_resuming_later_saved_pack(saved_runs, monkeypatch):
    create, _, _ = saved_runs
    create(status="interrupted")
    calls = []

    def download(sources, options):
        calls.append(sources)
        return CommandResult(status=RunStatus.PARTIAL)

    monkeypatch.setattr(workflow, "run_import", download)
    monkeypatch.setattr(workflow, "run_resume_sync", forbid)
    assert do_import([C, A]).status is RunStatus.PARTIAL
    assert calls == [(C,)]


def test_mixed_sources_keep_input_order_even_with_shared_saved_parent(saved_runs, monkeypatch):
    create, _, _ = saved_runs
    checkpoint = create((A, C), status="interrupted")
    fresh_id = "mlxrun_" + "b" * 32
    calls = []

    def download(sources, options):
        calls.append(("download", sources))
        return CommandResult(run_id=fresh_id)

    def resume(run_id, **kwargs):
        assert run_id == checkpoint.run_id
        calls.append(("resume", kwargs["selected_sources"]))
        return CommandResult(run_id=run_id)

    monkeypatch.setattr(workflow, "run_import", download)
    monkeypatch.setattr(workflow, "run_resume_sync", resume)
    result = do_import([A, B, C])
    assert calls == [("resume", (A,)), ("download", (B,)), ("resume", (C,))]
    assert result.result["analysis_selectors"] == [
        f"{checkpoint.run_id}:PackAlpha",
        fresh_id,
        f"{checkpoint.run_id}:PackCharlie",
    ]


def test_explicit_refresh_bypasses_saved_download(saved_runs, monkeypatch):
    create, _, _ = saved_runs
    create(phase="describe")
    calls = []

    def download(sources, options):
        calls.append(sources)
        return CommandResult()

    monkeypatch.setattr(workflow, "run_import", download)
    do_import([A], refresh=True)
    assert calls == [(A,)]


def test_import_resume_does_not_include_sibling_with_downloaded_media(saved_runs, monkeypatch):
    create, runs, _ = saved_runs
    checkpoint = create(status="interrupted")
    safe = {
        **checkpoint.safe_parameters,
        "source_states": {
            A: {"phase": "import", "status": "interrupted"},
            B: {"phase": "describe", "status": "interrupted"},
        },
        "source_memberships": {"PackBravo": ["media-b"]},
    }
    checkpoint = checkpoint.model_copy(
        update={
            "safe_parameters": safe,
            "elements": {"media-b": SimpleNamespace(fingerprint_complete=True)},
        }
    )
    runs[0] = checkpoint
    calls = []

    def resume(run_id, **kwargs):
        calls.append(kwargs["selected_sources"])
        return CommandResult(run_id=run_id)

    monkeypatch.setattr(workflow, "run_resume_sync", resume)
    monkeypatch.setattr(workflow, "run_import", forbid)
    result = do_import([A, B])
    assert calls == [(A,)]
    assert result.result["analysis_selectors"] == [
        f"{checkpoint.run_id}:PackAlpha",
        f"{checkpoint.run_id}:PackBravo",
    ]


@pytest.mark.parametrize("mode", ["sequential", "fast"])
def test_multi_parent_analysis_retains_selected_pack_scope_and_one_approval(
    saved_runs, monkeypatch, mode
):
    create, _, config = saved_runs
    config = config.model_copy(
        update={"processing": config.processing.model_copy(update={"file_analysis_mode": mode})}
    )
    monkeypatch.setattr(workflow, "load_config", lambda **_: config)
    first = create()
    second = create((B, C))
    lookup = {first.run_id: first, second.run_id: second}
    resolutions = []

    def resolve(selector, **kw):
        resolutions.append(selector)
        return lookup[selector.split(":")[0]]

    monkeypatch.setattr(packs, "resolve_pack_run", resolve)
    calls = []
    approvals = []

    def describe(run_id, options):
        calls.append((run_id, options.selected_sources))
        assert options.unknown_cost_confirmation(None)
        return CommandResult(run_id=run_id, status=RunStatus.SUCCEEDED)

    monkeypatch.setattr(workflow, "run_describe", describe)

    def describe_many(groups, options):
        assert mode == "fast"
        for run_id, values in groups:
            from dataclasses import replace

            describe(run_id, replace(options, selected_sources=tuple(values)))
        return CommandResult(status=RunStatus.SUCCEEDED)

    monkeypatch.setattr(workflow, "run_describe_many", describe_many)
    workflow.describe_command(
        [f"{first.run_id}:PackAlpha", f"{second.run_id}:PackCharlie", f"{first.run_id}:PackBravo"],
        provider=None,
        model=None,
        max_ai_requests=None,
        max_cost_usd=None,
        allow_unknown_cost=False,
        unknown_cost_confirmation=lambda limit: approvals.append(limit) or True,
    )
    assert calls == [(first.run_id, (A,)), (second.run_id, (C,)), (first.run_id, (B,))]
    assert approvals == [None]
    assert len(resolutions) == 2


@pytest.mark.parametrize("phase", ["import", "describe"])
def test_metadata_reuse_enters_analysis_without_finishing_import(saved_runs, monkeypatch, phase):
    create, _, _ = saved_runs
    checkpoint = create(phase=phase, status="interrupted")
    monkeypatch.setattr(workflow, "run_import", forbid)
    monkeypatch.setattr(workflow, "run_resume_sync", forbid)
    result = do_import([A, B], preparation="metadata")
    assert result.result["analysis_selectors"] == [
        f"{checkpoint.run_id}:PackAlpha",
        f"{checkpoint.run_id}:PackBravo",
    ]


def test_reuse_indexes_sources_once_per_run(saved_runs, monkeypatch):
    create, _, config = saved_runs
    sources = tuple(f"https://t.me/addemoji/Pack{index}" for index in range(250))
    checkpoint = create(sources)
    calls = 0
    original = import_reuse._source_name

    def counted(source):
        nonlocal calls
        calls += 1
        return original(source)

    monkeypatch.setattr(import_reuse, "_source_name", counted)
    result = import_reuse.reusable_imports(sources, config, max_items=None)
    assert len(result) == len(sources)
    assert all(run is checkpoint for run, _ in result.values())
    assert calls <= len(sources) * 2


def test_startup_progress_visible_before_scanning_checkpoints(saved_runs, monkeypatch):
    from contextlib import contextmanager

    create, _, _ = saved_runs
    checkpoint = create()
    active = []
    stages = []

    @contextmanager
    def progress(label):
        active.append(label)
        stages.append(label)
        try:
            yield
        finally:
            active.pop()

    def restore(*args, **kwargs):
        assert active, "Saved-run discovery must never run without startup progress"
        return {A: (checkpoint, A)}

    monkeypatch.setattr(runtime, "operation_progress", progress)
    monkeypatch.setattr(import_reuse, "reusable_imports", restore)
    monkeypatch.setattr(workflow, "run_import", forbid)
    result = do_import([A], preparation="metadata")
    assert result.result["reused_packs"] == 1
    assert len(stages) == 3


def test_cached_selector_index_still_rejects_foreign_pack(saved_runs, monkeypatch):
    create, _, config = saved_runs
    checkpoint = create()
    monkeypatch.setattr(packs, "load_config", lambda: config)

    def resolve(selector, **kwargs):
        if selector.endswith(":ForeignPack"):
            raise runtime.CommandError(
                "CONFIG_INVALID", "Foreign pack", hint="Choose a saved pack."
            )
        return checkpoint

    monkeypatch.setattr(packs, "resolve_pack_run", resolve)
    monkeypatch.setattr(workflow, "run_describe_many", forbid)
    with pytest.raises(runtime.CommandError, match="Foreign pack"):
        workflow.describe_command(
            [f"{checkpoint.run_id}:PackAlpha", f"{checkpoint.run_id}:ForeignPack"],
            provider=None,
            model=None,
            max_ai_requests=None,
            max_cost_usd=None,
            allow_unknown_cost=False,
        )


def _with_saved_work(checkpoint, pack, *, media, ai=0):
    ids = [f"{pack}-{index}" for index in range(media)]
    return checkpoint.model_copy(
        update={
            "safe_parameters": {
                **checkpoint.safe_parameters,
                "source_memberships": {pack: ids},
            },
            "elements": {
                identifier: ElementCheckpoint(
                    stage="ai_cached" if index < ai else "fingerprint_ready",
                    fingerprint_complete=True,
                    deterministic_cache_key="a" * 64,
                    ai_facets_complete=index < ai,
                    ai_cache_key="b" * 64 if index < ai else None,
                )
                for index, identifier in enumerate(ids)
            },
        }
    )


@pytest.mark.parametrize("ai", [0, 128])
def test_interrupted_saved_media_beats_newer_successful_metadata_import(
    saved_runs, monkeypatch, ai
):
    create, runs, config = saved_runs
    older = _with_saved_work(
        create(phase="describe", status="interrupted"), "PackAlpha", media=200, ai=ai
    )
    runs[0] = older
    imported = create(phase="import", status="succeeded")
    assert imported.updated_at >= older.updated_at
    selected = import_reuse.reusable_imports([A], config, max_items=None)
    assert selected[A][0] is older
    monkeypatch.setattr(workflow, "run_import", forbid)
    monkeypatch.setattr(workflow, "run_resume_sync", forbid)
    result = do_import([A], preparation="metadata")
    assert result.result["analysis_selectors"] == [f"{older.run_id}:PackAlpha"]


def test_reuse_prefers_saved_ai_then_media_before_newer_run_status(saved_runs):
    create, runs, config = saved_runs
    older = _with_saved_work(
        create(phase="describe", status="interrupted"), "PackAlpha", media=100, ai=90
    )
    runs[0] = older
    newer = _with_saved_work(
        create(phase="import", status="succeeded"), "PackAlpha", media=200, ai=20
    )
    runs[0] = newer
    assert import_reuse.reusable_imports([A], config, max_items=None)[A][0] is older


def test_sibling_progress_does_not_influence_pack_selection(saved_runs):
    create, runs, config = saved_runs
    alpha = _with_saved_work(
        create(phase="describe", status="interrupted"), "PackAlpha", media=10, ai=10
    )
    runs[0] = alpha
    bravo = _with_saved_work(
        create(phase="describe", status="interrupted"), "PackBravo", media=200, ai=200
    )
    runs[0] = bravo
    selected = import_reuse.reusable_imports([A, B], config, max_items=None)
    assert selected[A][0] is alpha
    assert selected[B][0] is bravo


def test_reuse_progress_index_is_built_once_per_run(saved_runs, monkeypatch):
    create, runs, config = saved_runs
    sources = tuple(f"https://t.me/addemoji/Pack{index}" for index in range(100))
    create(sources, phase="describe", status="interrupted")
    create(sources, phase="import")
    calls = []
    original = import_reuse._source_progress_index

    def indexed(checkpoint, names):
        calls.append(checkpoint.run_id)
        return original(checkpoint, names)

    monkeypatch.setattr(import_reuse, "_source_progress_index", indexed)
    result = import_reuse.reusable_imports(sources, config, max_items=None)
    assert len(result) == len(sources)
    assert len(calls) == len(runs)
