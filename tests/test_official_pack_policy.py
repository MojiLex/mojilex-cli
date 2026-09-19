from __future__ import annotations

from contextlib import contextmanager
from decimal import Decimal
from types import SimpleNamespace

import pytest
import typer
from typer.testing import CliRunner

from mojilex_cli import cli
from mojilex_cli.commands import official_packs, workflow
from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.config import MojiLexConfig, load_config
from mojilex_cli.i18n import use_ui_language
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import RunStore, new_checkpoint
from mojilex_cli.runs.store import ElementCheckpoint

OLD = "https://t.me/addemoji/Published"
NEW = "https://t.me/addemoji/NewPack"


@pytest.fixture
def official(monkeypatch):
    calls = []

    def names():
        calls.append(True)
        return frozenset({"published", "another"})

    monkeypatch.setattr(official_packs, "official_pack_names", names)
    monkeypatch.setattr(official_packs, "load_config", MojiLexConfig)
    return calls


@pytest.mark.parametrize("approved", [False, True])
def test_explicit_ask_uses_one_decision_for_whole_mixed_batch(official, approved):
    prompts = []
    other = "tg://addemoji?set=Another"
    result = official_packs.select_sources(
        [OLD, NEW, other],
        platform="auto",
        policy="ask",
        confirmation=lambda message: prompts.append(message) or approved,
    )
    assert len(prompts) == 1
    assert len(official) == 1
    assert result.selected == ((OLD, NEW, other) if approved else (NEW,))
    assert result.skipped == (() if approved else (OLD, other))


@pytest.mark.parametrize("policy", ["skip", "ask"])
def test_missing_confirmation_never_permits_official_pack(official, policy):
    selected = official_packs.select_sources([OLD], platform="auto", policy=policy)
    assert selected.selected == ()
    assert selected.empty_result().run_id is None
    assert selected.empty_result().result["official_packs_skipped"] == [OLD]


def test_skip_never_prompts_and_allow_never_checks(official):
    def fail(_):
        pytest.fail("must not ask")

    result = official_packs.select_sources(
        [OLD, NEW], platform="auto", policy="skip", confirmation=fail
    )
    assert result.selected == (NEW,)
    assert len(official) == 1
    official.clear()
    assert official_packs.select_sources(
        [OLD, NEW], platform="auto", policy="allow", confirmation=fail
    ).selected == (OLD, NEW)
    assert official == []


def test_bare_names_aliases_and_case_match_official_identity(official):
    values = ["published", "https://telegram.me/addemoji/PUBLISHED", NEW]
    result = official_packs.select_sources(values, platform="telegram", policy="skip")
    assert result.selected == (NEW,)
    assert len(result.skipped) == 2


def test_check_failure_does_not_silently_authorize_analysis(monkeypatch):
    def fail():
        raise RuntimeError("GitHub unavailable")

    monkeypatch.setattr(official_packs, "official_pack_names", fail)
    with pytest.raises(RuntimeError, match="GitHub unavailable"):
        official_packs.select_sources([OLD], platform="auto", policy="ask")


def test_index_uses_only_official_main_and_correct_namespace(tmp_path, monkeypatch):
    opened = []

    @contextmanager
    def workspace(target, branch):
        opened.append((target, branch))
        yield SimpleNamespace(root=tmp_path)

    monkeypatch.setattr(official_packs, "repository_workspace", workspace)
    collections = [
        SimpleNamespace(
            platform="telegram",
            native_namespace="sticker_set.name",
            scope_id="global",
            native_id="Published",
        ),
        SimpleNamespace(
            platform="telegram", native_namespace="other", scope_id="global", native_id="Unrelated"
        ),
    ]
    monkeypatch.setattr(
        official_packs,
        "load_validated_dataset",
        lambda *_args, **_kwargs: (
            SimpleNamespace(collections=dict(enumerate(collections))),
            SimpleNamespace(raise_for_errors=lambda: None),
        ),
    )
    assert official_packs.official_pack_names() == {"published"}
    assert opened == [("MojiLex/mojilex", "main")]


@pytest.mark.parametrize("answer,allowed", [("\n", False), ("нет\n", False), ("да\n", True)])
def test_actual_russian_prompt_enter_is_no_and_only_explicit_yes_allows(
    monkeypatch, answer, allowed
):
    app = typer.Typer()

    @app.command()
    def check():
        monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
        with use_ui_language("ru"):
            result = cli._official_confirmation_callback(
                non_interactive=False, json_output=False, quiet=False
            )("Повторить обработку?")
        typer.echo(f"allowed={result}")

    result = CliRunner().invoke(app, [], input=answer)
    assert result.exit_code == 0, result.output
    assert "[да/Нет]" in result.output
    assert f"allowed={allowed}" in result.output


@pytest.mark.parametrize("flag", ["non_interactive", "json_output", "quiet"])
def test_machine_modes_skip_without_prompt(monkeypatch, flag):
    monkeypatch.setattr(cli, "ui_confirm", lambda *_a, **_k: pytest.fail("must not prompt"))
    kwargs = dict.fromkeys(["non_interactive", "json_output", "quiet"], False)
    kwargs[flag] = True
    assert cli._official_confirmation_callback(**kwargs)("prompt") is False


@pytest.mark.parametrize("sources", [(OLD,), (OLD, NEW)])
def test_import_filters_before_any_media_or_ai_operation(official, monkeypatch, sources):
    imports = []
    monkeypatch.setattr(
        workflow,
        "run_import",
        lambda selected, options: imports.append(selected) or CommandResult(),
    )
    result = workflow.import_command(
        sources,
        repo=None,
        platform="auto",
        max_items=None,
        download_concurrency=None,
        check_media=True,
        fail_fast=False,
        official_confirmation=lambda _: False,
    )
    assert imports == ([] if sources == (OLD,) else [(NEW,)])
    assert result.result["official_packs_skipped"] == [OLD]


def test_policy_and_render_settings_preserve_other_config(tmp_path):
    from mojilex_cli.commands.settings import update_setting_command

    path = tmp_path / "settings.toml"
    path.write_text('[ai]\nmodel="keep-model"\nmax_ai_requests=123\n')
    for key, value in [("official_pack_policy", "skip"), ("render_concurrency", "4")]:
        update_setting_command(
            key, value, user_path=path, project_path=tmp_path / "absent", environment={}
        )
    config = load_config(user_path=path, project_path=tmp_path / "absent", environment={})
    assert config.processing.official_pack_policy == "skip"
    assert config.processing.render_concurrency == 4
    assert config.ai.model == "keep-model" and config.ai.max_ai_requests == 123
    overridden = load_config(
        user_path=path,
        project_path=tmp_path / "absent",
        environment={"MOJILEX_OFFICIAL_PACK_POLICY": "allow", "MOJILEX_RENDER_CONCURRENCY": "3"},
    )
    assert overridden.processing.official_pack_policy == "allow"
    assert overridden.processing.render_concurrency == 3


def test_receipt_is_source_specific_and_explicit_skip_overrides_it(official):
    prompts = []
    receipt = official_packs.select_sources(
        [OLD], platform="auto", policy="ask", confirmation=lambda _: True
    ).approved_sources
    assert receipt == (OLD,)
    other = "https://t.me/addemoji/Another"
    selection = official_packs.select_sources(
        [OLD, other, NEW],
        platform="auto",
        policy="ask",
        approved_sources=receipt,
        confirmation=lambda message: prompts.append(message) or False,
    )
    assert len(prompts) == 1 and "Another" in prompts[0]
    assert selection.selected == (OLD, NEW)
    assert selection.skipped == (other,)
    assert selection.approved_sources == (OLD,)
    skipped = official_packs.select_sources(
        [OLD], platform="auto", policy="skip", approved_sources=receipt
    )
    assert skipped.selected == ()


@pytest.fixture
def saved_import(tmp_path, monkeypatch):
    config = MojiLexConfig(
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
        processing={"official_pack_policy": "ask"},
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    options = runner.PipelineOptions(
        max_ai_requests=37, max_cost_usd=Decimal("1.25"), ai_concurrency=3
    )
    checkpoint = new_checkpoint(
        command="import",
        safe_parameters={
            **runner._safe_parameters((OLD, NEW), options),
            "staging_repository": str(staging),
            "source_memberships": {"Published": ["old"], "NewPack": ["new"]},
        },
        cli_version="0.2.0",
        schema_version=runner.SCHEMA_VERSION,
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    ).model_copy(
        update={
            "elements": {
                name: ElementCheckpoint(stage="fingerprint_ready", media_sha256=("b" * 64,))
                for name in ("old", "new")
            },
            "ai_requests_used": 5,
            "ai_cost_reserved_usd": Decimal("0.25"),
        }
    )
    store = RunStore(config.runs_dir)
    store.save(checkpoint)
    monkeypatch.setattr(runner, "load_config", lambda: config)
    return SimpleNamespace(config=config, checkpoint=checkpoint, store=store)


async def test_old_import_decline_preserves_plan_media_budget_and_durable_exclusions(
    official, saved_import, monkeypatch
):
    state = saved_import
    calls = []

    async def execute(sources, options, **kwargs):
        calls.append((sources, options, kwargs))
        return CommandResult(run_id=state.checkpoint.run_id)

    monkeypatch.setattr(runner, "_run_add", execute)
    result = await runner._run_describe(
        state.checkpoint.run_id,
        runner.PipelineOptions(official_confirmation=lambda _: False),
    )
    saved = state.store.load(state.checkpoint.run_id)
    assert saved.safe_parameters["sources"] == [OLD, NEW]
    assert saved.elements == state.checkpoint.elements
    assert saved.ai_requests_used == 5 and saved.ai_cost_reserved_usd == Decimal("0.25")
    assert saved.safe_parameters["official_excluded_sources"] == [OLD]
    assert result.result["official_packs_skipped"] == [OLD]
    sources, options, kwargs = calls[0]
    assert sources == (OLD, NEW)  # Preserve source indexes and original plan.
    assert options.official_excluded_sources == (OLD,)
    assert options.max_ai_requests == 37 and options.max_cost_usd == Decimal("1.25")
    assert kwargs["expected_hashes"] == {"old": ("b" * 64,), "new": ("b" * 64,)}
    assert kwargs["stage_only"] is True
    assert kwargs["resume_checkpoint"] == saved

    # A real _run_add marks the stage describe before it starts any AI work.
    state.store.save(saved.model_copy(update={"command": "describe"}))
    official.clear()
    await runner.run_resume(saved.run_id)
    assert official == []
    assert calls[-1][1].official_excluded_sources == (OLD,)


async def test_all_official_saved_import_skips_without_ai_or_losing_media(
    official, saved_import, monkeypatch
):
    state = saved_import
    checkpoint = state.checkpoint.model_copy(
        update={"safe_parameters": {**state.checkpoint.safe_parameters, "sources": [OLD]}}
    )
    state.store.save(checkpoint)
    monkeypatch.setattr(runner, "_run_add", lambda *_a, **_k: pytest.fail("must not call AI"))
    result = await runner._run_describe(checkpoint.run_id)
    assert result.status == "noop"
    assert result.run_id == checkpoint.run_id
    saved = state.store.load(checkpoint.run_id)
    assert saved.safe_parameters["sources"] == [OLD]
    assert saved.elements == checkpoint.elements


async def test_import_approval_is_saved_and_immediate_describe_does_not_prompt_again(
    official, saved_import, monkeypatch
):
    state = saved_import
    prompts = []

    def import_run(sources, options):
        checkpoint = state.checkpoint.model_copy(
            update={
                "safe_parameters": {
                    **state.checkpoint.safe_parameters,
                    **runner._safe_parameters(sources, options),
                }
            }
        )
        state.store.save(checkpoint)
        return CommandResult(run_id=checkpoint.run_id)

    async def execute(*_args, **_kwargs):
        return CommandResult(run_id=state.checkpoint.run_id)

    monkeypatch.setattr(workflow, "run_import", import_run)
    monkeypatch.setattr(runner, "_run_add", execute)
    workflow.import_command(
        [OLD],
        repo=None,
        platform="auto",
        max_items=None,
        download_concurrency=None,
        check_media=True,
        fail_fast=False,
        official_pack_policy="ask",
        official_confirmation=lambda message: prompts.append(message) or True,
    )
    saved = state.store.load(state.checkpoint.run_id)
    assert saved.safe_parameters["official_approved_sources"] == [OLD]
    await runner._run_describe(
        saved.run_id,
        runner.PipelineOptions(official_confirmation=lambda _: pytest.fail("already approved")),
    )
    assert len(prompts) == 1 and len(official) == 1


def test_direct_describe_selectors_cannot_bypass_official_guard(official, monkeypatch):
    monkeypatch.setattr(workflow, "load_config", lambda **_: MojiLexConfig())
    monkeypatch.setattr(workflow, "_sources_for_selectors", lambda *_a, **_k: (OLD, NEW))
    calls = []
    monkeypatch.setattr(
        workflow, "run_add", lambda sources, options: calls.append(sources) or CommandResult()
    )
    result = workflow.describe_command(
        [OLD, NEW],
        provider=None,
        model=None,
        max_ai_requests=None,
        max_cost_usd=None,
        allow_unknown_cost=False,
        official_confirmation=lambda _: False,
    )
    assert calls == [(NEW,)]
    assert result.result["official_packs_skipped"] == [OLD]


def test_excluded_saved_sources_are_not_scheduled_and_original_indexes_stay_stable():
    sources = (OLD, NEW, "https://t.me/addemoji/Another", "https://t.me/addemoji/Last")
    assert runner._remaining_source_entries(sources, {3}, excluded_sources=(OLD, sources[2])) == (
        (1, NEW),
    )
    assert sources[0] == OLD and len(sources) == 4


@pytest.mark.parametrize("command", ["describe", "add"])
async def test_legacy_describe_resume_uses_dedicated_official_confirmation(
    official, saved_import, monkeypatch, command
):
    state = saved_import
    parameters = {
        key: value
        for key, value in state.checkpoint.safe_parameters.items()
        if key not in {"official_approved_sources", "official_excluded_sources"}
    }
    checkpoint = state.checkpoint.model_copy(
        update={"command": command, "safe_parameters": parameters}
    )
    state.store.save(checkpoint)
    calls = []

    async def execute(sources, options, **kwargs):
        calls.append(options)
        return CommandResult(run_id=checkpoint.run_id)

    prompts = []
    monkeypatch.setattr(runner, "_run_add", execute)
    await runner.run_resume(
        checkpoint.run_id,
        confirmation=lambda _: pytest.fail("publication approval must not be reused"),
        official_confirmation=lambda prompt: prompts.append(prompt) or False,
    )
    assert len(prompts) == 1
    assert calls[0].official_excluded_sources == (OLD,)
    assert state.store.load(checkpoint.run_id).safe_parameters["sources"] == [OLD, NEW]


@pytest.mark.parametrize("dry_run", [False, True])
def test_update_all_checks_once_before_ai_but_preview_does_not_check(
    official, monkeypatch, dry_run
):
    monkeypatch.setattr(workflow, "load_config", lambda **_: MojiLexConfig())
    monkeypatch.setattr(workflow, "_sources_for_selectors", lambda *_a, **_k: (OLD, NEW))
    calls = []
    prompts = []
    monkeypatch.setattr(
        workflow,
        "run_add",
        lambda sources, options: calls.append((sources, options)) or CommandResult(),
    )
    workflow.update_command(
        None,
        all_collections=True,
        repo=None,
        dry_run=dry_run,
        official_pack_policy="ask",
        official_confirmation=lambda message: prompts.append(message) or False,
    )
    assert calls[0][0] == ((OLD, NEW) if dry_run else (NEW,))
    assert calls[0][1].dry_run is dry_run
    assert len(official) == (0 if dry_run else 1)
    assert len(prompts) == (0 if dry_run else 1)


def test_update_cli_forwards_official_policy_and_noninteractive_callback(monkeypatch):
    calls = []
    monkeypatch.setattr(
        workflow,
        "update_command",
        lambda selector, **kwargs: calls.append(kwargs) or CommandResult(),
    )
    result = CliRunner().invoke(
        cli.app, ["update", OLD, "--official-packs", "skip", "--non-interactive"]
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["official_pack_policy"] == "skip"
    assert calls[0]["official_confirmation"]("must not prompt") is False


def test_default_skips_official_packs_without_confirmation(official):
    selected = official_packs.select_sources(
        [OLD, NEW],
        platform="auto",
        confirmation=lambda _: pytest.fail("default must not ask"),
    )
    assert selected.selected == (NEW,)
    assert selected.skipped == (OLD,)
    assert len(official) == 1


def test_official_snapshot_is_shared_for_import_and_saved_runs_only_within_scope(monkeypatch):
    from mojilex_cli.commands.queue_progress import pack_queue_scope

    calls = []

    def load_names():
        calls.append(True)
        return frozenset({"published"} if len(calls) == 1 else {"published", "newpack"})

    monkeypatch.setattr(official_packs, "_load_official_pack_names", load_names)
    with pack_queue_scope():
        assert official_packs.select_sources(
            [OLD, NEW], platform="auto", policy="skip"
        ).selected == (NEW,)
        with pack_queue_scope():
            for _ in range(4):
                assert official_packs.select_sources(
                    [NEW], platform="auto", policy="skip"
                ).selected == (NEW,)
        assert len(calls) == 1
    with pack_queue_scope():
        assert official_packs.select_sources([NEW], platform="auto", policy="skip").selected == ()
    assert len(calls) == 2
    official_packs.official_pack_names()
    official_packs.official_pack_names()
    assert len(calls) == 4  # Standalone calls do not retain a stale main-branch index.


def test_official_snapshot_failure_is_not_cached_and_validation_stays_strict(tmp_path, monkeypatch):
    attempts = []
    validations = []

    @contextmanager
    def workspace(target, branch):
        attempts.append((target, branch))
        yield SimpleNamespace(root=tmp_path)

    def validate(root, *, strict):
        validations.append((root, strict))

        def raise_for_errors():
            if len(validations) == 1:
                raise RuntimeError("invalid official dataset")

        return SimpleNamespace(raise_for_errors=raise_for_errors)

    monkeypatch.setattr(official_packs, "repository_workspace", workspace)
    monkeypatch.setattr(
        official_packs,
        "load_validated_dataset",
        lambda root, strict: (SimpleNamespace(collections={}), validate(root, strict=strict)),
    )
    with official_packs.official_pack_scope():
        with pytest.raises(RuntimeError, match="invalid official dataset"):
            official_packs.official_pack_names()
        assert official_packs.official_pack_names() == frozenset()
        assert official_packs.official_pack_names() == frozenset()
    assert attempts == [("MojiLex/mojilex", "main")] * 2
    assert validations == [(tmp_path, True)] * 2


def test_official_snapshot_is_shared_with_worker_threads(monkeypatch):
    import asyncio

    calls = []

    def load_names():
        calls.append(True)
        return frozenset({"published"})

    monkeypatch.setattr(official_packs, "_load_official_pack_names", load_names)

    async def lookup():
        with official_packs.official_pack_scope():
            return await asyncio.gather(
                *(asyncio.to_thread(official_packs.official_pack_names) for _ in range(8))
            )

    assert asyncio.run(lookup()) == [frozenset({"published"})] * 8
    assert len(calls) == 1
