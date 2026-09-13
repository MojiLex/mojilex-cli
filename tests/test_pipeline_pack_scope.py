"""Selected packs retain the parent plan, exact cache and shared budget."""

from decimal import Decimal

import pytest

from mojilex_cli.commands.runtime import CommandError, CommandResult
from mojilex_cli.config import MojiLexConfig, ProcessingConfig
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import RunStore, new_checkpoint
from mojilex_cli.runs.pack_scope import (
    initialize_source_states,
    overall_status,
    record_source_state,
    selected_source_entries,
    source_state,
)
from test_add_run_staging import saved_add  # noqa: F401
from test_pack_describe_pipeline import pipeline  # noqa: F401
from test_pack_import_pipeline import _harness, _pack

SOURCES = ("https://t.me/addemoji/ScopeA", "https://t.me/addemoji/ScopeB")


def _checkpoint(tmp_path, *, command="import", status="succeeded"):
    return new_checkpoint(
        command=command,
        target_repository=str(tmp_path),
        cli_version="0.2.0",
        schema_version=runner.SCHEMA_VERSION,
        base_revision="a" * 40,
        safe_parameters={
            "sources": list(SOURCES),
            "languages": ["ru", "en"],
            "staging_repository": str(tmp_path),
            "max_ai_requests": 37,
            "max_cost_usd": "1.25",
        },
    ).model_copy(
        update={"status": status, "ai_requests_used": 14, "ai_cost_reserved_usd": Decimal("0.25")}
    )


def test_scope_preserves_original_source_indexes_and_rejects_foreign_source():
    assert selected_source_entries(SOURCES, (SOURCES[1],)) == ((1, SOURCES[1]),)
    assert selected_source_entries(SOURCES) == tuple(enumerate(SOURCES))
    with pytest.raises(CommandError):
        selected_source_entries(SOURCES, ("https://t.me/addemoji/Other",))


def test_legacy_phases_survive_one_pack_becoming_described(tmp_path):
    checkpoint = initialize_source_states(_checkpoint(tmp_path))
    checkpoint = checkpoint.model_copy(update={"command": "describe", "status": "running"})
    checkpoint = record_source_state(checkpoint, SOURCES[0], "describe", "succeeded")
    assert source_state(checkpoint, SOURCES[1]) == {"phase": "import", "status": "succeeded"}
    assert overall_status(checkpoint, "describe", "succeeded") == "partial"
    checkpoint = record_source_state(checkpoint, SOURCES[1], "describe", "running")
    assert overall_status(checkpoint, "describe", "succeeded") == "partial"
    checkpoint = record_source_state(checkpoint, SOURCES[1], "describe", "succeeded")
    assert overall_status(checkpoint, "describe", "succeeded") == "succeeded"
    assert checkpoint.ai_requests_used == 14


async def test_selected_import_fetches_only_selected_pack_and_keeps_full_parent(
    tmp_path_factory, monkeypatch
):
    async def ready(_item):
        pass

    state = await _harness(
        tmp_path_factory.mktemp("sc"),
        monkeypatch,
        (_pack("ScopeA", "one"), _pack("ScopeB", "two")),
        ready,
    )
    result = await runner._run_import(
        state.sources, runner.PipelineOptions(selected_sources=(state.sources[0],))
    )
    checkpoint = RunStore(state.config.runs_dir).load(result.run_id)
    assert state.fetched == ["ScopeA"]
    assert state.downloads == ["one"]
    assert checkpoint.safe_parameters["sources"] == list(state.sources)
    assert checkpoint.status == "partial"
    assert source_state(checkpoint, state.sources[0]) == {"phase": "import", "status": "succeeded"}
    assert source_state(checkpoint, state.sources[1])["status"] != "succeeded"


async def test_import_of_sibling_preserves_ai_budget_and_other_pack_state(
    tmp_path_factory, monkeypatch
):
    async def ready(_item):
        pass

    state = await _harness(
        tmp_path_factory.mktemp("sb"),
        monkeypatch,
        (_pack("ScopeA", "one"), _pack("ScopeB", "two")),
        ready,
    )
    imported = await runner._run_import(state.sources, runner.PipelineOptions())
    store = RunStore(state.config.runs_dir)
    checkpoint = initialize_source_states(store.load(imported.run_id))
    checkpoint = record_source_state(checkpoint, state.sources[0], "describe", "succeeded")
    checkpoint = checkpoint.model_copy(
        update={
            "command": "describe",
            "status": "partial",
            "ai_requests_used": 9,
            "ai_cost_reserved_usd": Decimal("0.125"),
        }
    )
    store.save(checkpoint)
    previous = checkpoint.elements["one"]
    state.fetched.clear()
    result = await runner._run_import(
        state.sources,
        runner.PipelineOptions(selected_sources=(state.sources[1],)),
        resume_id=checkpoint.run_id,
        resume_checkpoint=checkpoint,
    )
    after = store.load(result.run_id)
    assert state.fetched == ["ScopeB"]
    assert after.elements["one"] == previous
    assert after.ai_requests_used == 9
    assert after.ai_cost_reserved_usd == Decimal("0.125")
    assert after.status == "partial"
    assert source_state(after, state.sources[0]) == {"phase": "describe", "status": "succeeded"}
    assert after.safe_parameters["sources"] == list(state.sources)


@pytest.mark.parametrize("operation", ["describe", "resume"])
async def test_scoped_analysis_keeps_budget_and_prompts_only_for_selection(
    tmp_path, monkeypatch, operation
):
    checkpoint = _checkpoint(tmp_path, command="describe", status="partial")
    checkpoint = record_source_state(checkpoint, SOURCES[0], "describe", "failed")
    checkpoint = record_source_state(checkpoint, SOURCES[1], "describe", "succeeded")
    config = MojiLexConfig(
        runs_dir=tmp_path / "runs", processing=ProcessingConfig(official_pack_policy="allow")
    )
    RunStore(config.runs_dir).save(checkpoint)
    monkeypatch.setattr(runner, "load_config", lambda: config)
    captured = {}

    from mojilex_cli.commands import official_packs

    def select(sources, **_kwargs):
        captured["checked_sources"] = tuple(sources)
        return official_packs.SourceSelection(tuple(sources))

    async def execute(sources, options, **kwargs):
        captured.update(sources=sources, options=options, checkpoint=kwargs["resume_checkpoint"])
        return CommandResult()

    monkeypatch.setattr(official_packs, "select_sources", select)
    monkeypatch.setattr(runner, "_run_add", execute)
    if operation == "describe":
        await runner._run_describe(
            checkpoint.run_id, runner.PipelineOptions(selected_sources=(SOURCES[0],))
        )
    else:
        await runner.run_resume(checkpoint.run_id, selected_sources=(SOURCES[0],))
    assert captured["checked_sources"] == (SOURCES[0],)
    assert captured["sources"] == SOURCES
    assert captured["options"].selected_sources == (SOURCES[0],)
    assert captured["checkpoint"].ai_requests_used == 14
    assert captured["checkpoint"].ai_cost_reserved_usd == Decimal("0.25")
    assert source_state(captured["checkpoint"], SOURCES[1]) == {
        "phase": "describe",
        "status": "succeeded",
    }


async def test_selected_analysis_persists_membership_and_can_finish_other_pack(request):
    state = request.getfixturevalue("pipeline")
    sources = tuple(source.canonical_url for source in state.sources)
    result = await runner._run_add(
        sources, runner.PipelineOptions(selected_sources=(sources[1],)), stage_only=True
    )
    store = RunStore(state.config.runs_dir)
    checkpoint = store.load(result.run_id)
    assert state.merges == ["PackBeta"]
    assert set(checkpoint.elements) == {"PackBeta"}
    assert checkpoint.status == "partial"
    assert checkpoint.safe_parameters["source_memberships"] == {"PackBeta": ["PackBeta"]}
    assert checkpoint.safe_parameters["sources"] == list(sources)
    assert source_state(checkpoint, sources[1]) == {"phase": "describe", "status": "succeeded"}
    checkpoint = checkpoint.model_copy(
        update={"ai_requests_used": 7, "ai_cost_reserved_usd": Decimal("0.15")}
    )
    store.save(checkpoint)
    previous_beta = checkpoint.elements["PackBeta"]
    result = await runner._run_add(
        sources,
        runner.PipelineOptions(selected_sources=(sources[0],)),
        resume_id=checkpoint.run_id,
        resume_checkpoint=checkpoint,
        stage_only=True,
    )
    after = store.load(result.run_id)
    assert state.merges == ["PackBeta", "PackAlpha"]
    assert after.elements["PackBeta"] == previous_beta
    assert after.status in {"succeeded", "noop"}
    assert after.ai_requests_used == 7
    assert after.ai_cost_reserved_usd == Decimal("0.15")
    assert after.safe_parameters["sources"] == list(sources)


async def test_failed_composition_verification_never_marks_selected_pack_ready(request):
    state = request.getfixturevalue("pipeline")

    async def fail_verify(self, **_kwargs):
        raise CommandError(
            "AI_OUTPUT_INVALID", "Synthetic composition failure", hint="Resume later"
        )

    state.queue.verify = fail_verify
    sources = tuple(source.canonical_url for source in state.sources)
    with pytest.raises(CommandError):
        await runner._run_add(
            sources, runner.PipelineOptions(selected_sources=(sources[0],)), stage_only=True
        )
    checkpoint = state.latest_checkpoint()
    assert source_state(checkpoint, sources[0])["status"] != "succeeded"
    assert source_state(checkpoint, sources[1])["status"] != "succeeded"


@pytest.mark.parametrize("status", ["interrupted", "failed", "budget_exceeded"])
def test_idle_source_uses_terminal_parent_status_without_changing_evidence(tmp_path, status):
    checkpoint = record_source_state(
        _checkpoint(tmp_path, status=status), SOURCES[0], "describe", "running"
    )
    assert source_state(checkpoint, SOURCES[0])["status"] == status
    assert checkpoint.safe_parameters["source_states"][SOURCES[0]]["status"] == "running"
