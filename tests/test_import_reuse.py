from types import SimpleNamespace

import pytest

from mojilex_cli.commands import import_reuse, interactive, packs, runtime, workflow
from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.config import MojiLexConfig
from mojilex_cli.output import RunStatus
from mojilex_cli.runs import new_checkpoint

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


def test_newer_completed_download_not_shadowed_by_old_analysis(saved_runs):
    create, _, config = saved_runs
    create(phase="describe")
    imported = create(phase="import")
    assert import_reuse.reusable_imports([A], config, max_items=None)[A][0] == imported


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


def test_multi_parent_analysis_retains_selected_pack_scope_and_one_approval(
    saved_runs, monkeypatch
):
    create, _, _ = saved_runs
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
