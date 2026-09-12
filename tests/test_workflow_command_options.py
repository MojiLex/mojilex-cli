from decimal import Decimal

from mojilex_cli.commands import workflow
from mojilex_cli.commands.runtime import CommandResult
from mojilex_cli.output import RunStatus


def test_describe_run_accepts_and_forwards_ai_stage_overrides(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_describe(run_id, overrides):  # type: ignore[no-untyped-def]
        captured["run_id"] = run_id
        captured["overrides"] = overrides
        return CommandResult(run_id=run_id, status=RunStatus.SUCCEEDED)

    monkeypatch.setattr(workflow, "run_describe", fake_run_describe)

    result = workflow.describe_command(
        ["mlxrun_0123456789abcdef"],
        provider="gemini",
        model="gemini-explicit",
        max_ai_requests=12,
        max_cost_usd=Decimal("1.25"),
        allow_unknown_cost=True,
    )

    overrides = captured["overrides"]
    assert result.status is RunStatus.SUCCEEDED
    assert captured["run_id"] == "mlxrun_0123456789abcdef"
    assert overrides.provider == "gemini"  # type: ignore[union-attr]
    assert overrides.model == "gemini-explicit"  # type: ignore[union-attr]
    assert overrides.max_ai_requests == 12  # type: ignore[union-attr]
    assert overrides.max_cost_usd == Decimal("1.25")  # type: ignore[union-attr]
    assert overrides.allow_unknown_cost is True  # type: ignore[union-attr]


def test_submit_forwards_late_confirmation_callback(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def confirm(message: str) -> bool:
        captured["message"] = message
        return True

    def fake_run_submit(target, **kwargs):  # type: ignore[no-untyped-def]
        captured["target"] = target
        captured.update(kwargs)
        return CommandResult(status=RunStatus.SUCCEEDED)

    monkeypatch.setattr(workflow, "run_submit", fake_run_submit)

    workflow.submit_command(
        "mlxrun_0123456789abcdef",
        repo="MojiLex/mojilex",
        publish="pr",
        direct_push=True,
        base="main",
        confirmation=confirm,
    )

    assert captured["confirmation"] is confirm
    assert captured["publish"] == "pr"
    assert "message" not in captured


def test_resume_forwards_runtime_only_confirmation_callbacks(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def confirm(message: str) -> bool:
        return bool(message)

    def confirm_unknown_cost(requests: int) -> bool:
        return requests > 0

    def fake_run_resume_sync(run_id, **kwargs):  # type: ignore[no-untyped-def]
        captured["run_id"] = run_id
        captured.update(kwargs)
        return CommandResult(status=RunStatus.SUCCEEDED)

    monkeypatch.setattr(workflow, "run_resume_sync", fake_run_resume_sync)

    workflow.resume_command(
        "mlxrun_0123456789abcdef0123456789abcdef",
        confirmation=confirm,
        unknown_cost_confirmation=confirm_unknown_cost,
    )

    assert captured == {
        "run_id": "mlxrun_0123456789abcdef0123456789abcdef",
        "confirmation": confirm,
        "unknown_cost_confirmation": confirm_unknown_cost,
    }
