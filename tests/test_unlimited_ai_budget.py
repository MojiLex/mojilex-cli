"""Unlimited totals preserve cost, consent, persistence and resume boundaries."""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

from mojilex_cli.ai import BudgetExceededError, CostEstimate, RequestBudget
from mojilex_cli.ai.base import UnknownCostError
from mojilex_cli.commands import packs
from mojilex_cli.commands.runtime import CommandError, CommandResult
from mojilex_cli.config import AIConfig, MojiLexConfig, load_config
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import new_checkpoint


def estimate(amount="0.01"):
    return CostEstimate(
        upper_bound_usd=Decimal(amount) if amount is not None else None, note="test estimate"
    )


async def test_unlimited_concurrent_reservations_count_and_persist_every_request():
    records = []
    budget = RequestBudget(
        max_requests=None,
        requests_used=120,
        cost_reserved=Decimal("1.20"),
        reservation_recorder=lambda requests, cost: records.append((requests, cost)),
    )
    await asyncio.gather(*(budget.reserve(estimate()) for _ in range(250)))
    assert budget.requests_used == 370
    assert budget.cost_reserved == Decimal("3.70")
    assert [count for count, _ in records] == list(range(121, 371))
    assert records[-1] == (370, Decimal("3.70"))


async def test_zero_still_disables_requests():
    budget = RequestBudget(max_requests=0)
    with pytest.raises(BudgetExceededError):
        await budget.reserve(estimate("0"))
    assert budget.requests_used == 0


async def test_unlimited_shared_budget_still_enforces_cost_across_pack_workers():
    budget = RequestBudget(max_requests=None, max_cost_usd=Decimal("0.03"))

    async def pack():
        return await asyncio.gather(
            *(budget.reserve(estimate()) for _ in range(20)), return_exceptions=True
        )

    outcomes = [
        outcome for results in await asyncio.gather(pack(), pack(), pack()) for outcome in results
    ]
    assert outcomes.count(None) == 3
    assert sum(isinstance(value, BudgetExceededError) for value in outcomes) == 57
    assert budget.requests_used == 3
    assert budget.cost_reserved == Decimal("0.03")


@pytest.mark.parametrize("approval", [True, False])
async def test_unlimited_unknown_cost_authorization_is_explicit_and_once(approval):
    prompts = []

    def authorize(remaining):
        prompts.append(remaining)
        return approval

    budget = RequestBudget(max_requests=None, unknown_cost_authorizer=authorize)
    outcomes = await asyncio.gather(
        *(budget.reserve(estimate(None)) for _ in range(20)), return_exceptions=True
    )
    assert prompts == [None]
    if approval:
        assert outcomes == [None] * 20
        assert budget.requests_used == 20
    else:
        assert all(isinstance(value, UnknownCostError) for value in outcomes)
        assert budget.requests_used == 0


async def test_unlimited_recorder_failure_never_consumes_an_unpersisted_reservation():
    def fail_record(requests, cost):
        assert (requests, cost) == (151, Decimal("1.51"))
        raise OSError("checkpoint write failed")

    budget = RequestBudget(
        max_requests=None,
        requests_used=150,
        cost_reserved=Decimal("1.50"),
        reservation_recorder=fail_record,
    )
    with pytest.raises(OSError, match="checkpoint write failed"):
        await budget.reserve(estimate())
    assert budget.requests_used == 150
    assert budget.cost_reserved == Decimal("1.50")


async def test_unlimited_resume_keeps_previously_spent_cost_inside_total_cap():
    budget = RequestBudget(
        max_requests=None,
        requests_used=150,
        cost_reserved=Decimal("1.50"),
        max_cost_usd=Decimal("1.51"),
    )
    await budget.reserve(estimate())
    with pytest.raises(BudgetExceededError, match="cost limit"):
        await budget.reserve(estimate())
    assert budget.requests_used == 151
    assert budget.cost_reserved == Decimal("1.51")


@pytest.mark.parametrize("saved,current", [(None, 100), (37, None), (0, None)])
def test_materialized_limit_roundtrip_wins_over_changed_local_configuration(
    tmp_path, monkeypatch, saved, current
):
    initial = MojiLexConfig(ai=AIConfig(max_ai_requests=saved))
    options = runner._materialized_options(runner.PipelineOptions(), initial)
    parameters = runner._safe_parameters([], options)
    assert parameters["max_ai_requests"] == ("unlimited" if saved is None else saved)
    restored = runner._options_from_safe(parameters)
    user = tmp_path / "config.toml"
    user.write_text(
        '[ai]\nmax_ai_requests="unlimited"\n'
        if current is None
        else f"[ai]\nmax_ai_requests={current}\n",
        encoding="utf-8",
    )

    def changed_config(*, cli):
        return load_config(
            cli=cli, environment={}, user_path=user, project_path=tmp_path / "missing.toml"
        )

    monkeypatch.setattr(runner, "load_config", changed_config)
    assert runner._resolved_config(restored).ai.max_ai_requests == saved


def test_pack_summary_reports_saved_unlimited_even_with_current_finite_config():
    checkpoint = SimpleNamespace(
        safe_parameters={
            "sources": ["https://t.me/addemoji/Example"],
            "max_ai_requests": "unlimited",
        },
        run_id="mlxrun_test",
        command="describe",
        status="interrupted",
        updated_at=datetime.now(UTC),
        elements={},
        ai_requests_used=151,
    )
    summary = packs._summary(checkpoint, MojiLexConfig(ai=AIConfig(max_ai_requests=100)))
    assert summary["max_ai_requests"] is None
    assert summary["max_ai_requests_source"] == "checkpoint"
    assert summary["requests_used"] == 151


@pytest.mark.parametrize(
    "saved,override,expected",
    [
        ("unlimited", None, "unlimited"),
        (37, None, 37),
        (37, "unlimited", "unlimited"),
        ("unlimited", 300, 300),
        ("unlimited", 0, 0),
    ],
)
async def test_resume_explicit_limit_override_preserves_counters(
    tmp_path, monkeypatch, saved, override, expected
):
    checkpoint = SimpleNamespace(
        run_id="mlxrun_test",
        command="add",
        target_repository=str(tmp_path),
        elements={},
        safe_parameters={
            "sources": [],
            "languages": ["ru", "en"],
            "max_ai_requests": saved,
            "max_cost_usd": "3.00",
            "ai_concurrency": 4,
            "download_concurrency": 15,
        },
        ai_requests_used=151,
        ai_cost_reserved_usd=Decimal("1.51"),
    )
    captured = {}
    monkeypatch.setattr(runner, "load_config", lambda: SimpleNamespace(runs_dir=tmp_path))
    monkeypatch.setattr(
        runner,
        "RunStore",
        lambda path: SimpleNamespace(load_for_resume=lambda *a, **kw: checkpoint),
    )

    async def execute(sources, options, **kwargs):
        captured["options"] = options
        assert kwargs["resume_checkpoint"] is checkpoint
        return CommandResult()

    monkeypatch.setattr(runner, "_run_add", execute)
    await runner.run_resume(checkpoint.run_id, max_ai_requests=override)
    assert captured["options"].max_ai_requests == expected
    assert captured["options"].max_cost_usd == Decimal("3.00")
    assert captured["options"].ai_concurrency == 4
    assert checkpoint.ai_requests_used == 151
    assert checkpoint.ai_cost_reserved_usd == Decimal("1.51")


@pytest.mark.parametrize("operation", ["_run_add", "_run_import"])
@pytest.mark.parametrize(
    "request_limit,cost_limit,allowed",
    [
        (150, Decimal("2"), False),
        ("unlimited", Decimal("1.50"), False),
        (151, Decimal("1.51"), True),
        ("unlimited", None, True),
    ],
)
async def test_resume_budget_validation_precedes_credentials_workspace_and_checkpoint_writes(
    tmp_path, monkeypatch, operation, request_limit, cost_limit, allowed
):
    checkpoint = new_checkpoint(
        command="import" if operation == "_run_import" else "add",
        safe_parameters={"max_ai_requests": "unlimited", "max_cost_usd": "2"},
        cli_version="0.2.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    ).model_copy(update={"ai_requests_used": 151, "ai_cost_reserved_usd": Decimal("1.51")})
    before = checkpoint.model_dump(mode="json")
    reached_credentials = []

    class CredentialsBoundaryReached(Exception):
        pass

    def credentials_boundary():
        reached_credentials.append(True)
        raise CredentialsBoundaryReached

    def unexpected_side_effect(*args, **kwargs):
        pytest.fail("budget validation must happen before repository or checkpoint operations")

    def isolated_config(*, cli):
        return load_config(
            cli=cli,
            environment={},
            user_path=tmp_path / "absent-user.toml",
            project_path=tmp_path / "absent-project.toml",
        )

    monkeypatch.setattr(runner, "load_config", isolated_config)
    monkeypatch.setattr(runner, "load_credentials", credentials_boundary)
    monkeypatch.setattr(runner, "repository_workspace", unexpected_side_effect)
    monkeypatch.setattr(runner, "RunStore", unexpected_side_effect)
    options = runner.PipelineOptions(max_ai_requests=request_limit, max_cost_usd=cost_limit)
    if allowed:
        with pytest.raises(CredentialsBoundaryReached):
            await getattr(runner, operation)(
                ["https://t.me/addemoji/Example"], options, resume_checkpoint=checkpoint
            )
        assert reached_credentials == [True]
    else:
        with pytest.raises(CommandError) as caught:
            await getattr(runner, operation)(
                ["https://t.me/addemoji/Example"], options, resume_checkpoint=checkpoint
            )
        assert caught.value.error.code == "CONFIG_INVALID"
        assert not reached_credentials
    assert checkpoint.model_dump(mode="json") == before
