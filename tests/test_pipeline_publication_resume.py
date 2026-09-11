from __future__ import annotations

from types import SimpleNamespace

import pytest

from mojilex_cli.commands.runtime import CommandError, CommandResult
from mojilex_cli.output import RunStatus
from mojilex_cli.pipeline import runner
from mojilex_cli.runs import PublicationCheckpoint, RunCheckpoint, new_checkpoint


class _FakeGit:
    def __init__(self, candidate_sha: str | None) -> None:
        self.candidate_sha = candidate_sha

    def run(self, *arguments: str) -> SimpleNamespace:
        assert arguments == ("remote",)
        return SimpleNamespace(stdout="origin\n")

    def optional_remote_sha(self, remote: str, branch: str) -> str | None:
        assert remote == "origin"
        assert branch == "mojilex/batch/0123456789ab"
        return self.candidate_sha


def _partial_checkpoint(*, phase: str = "prepared") -> tuple[RunCheckpoint, str, str]:
    old_base = "a" * 40
    candidate = "b" * 40
    publication = PublicationCheckpoint(
        mode="direct",
        remote="origin",
        base_branch="main",
        expected_old_base=old_base,
        candidate_sha=candidate,
        candidate_branch="mojilex/batch/0123456789ab",
        phase=phase,
        completed_source_indexes=(0,),
    )
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters=runner._safe_parameters(
            ("https://t.me/addemoji/Done", "https://t.me/addemoji/Retry"),
            runner.PipelineOptions(
                repository="MojiLex/mojilex",
                publish="pr",
                direct_push=True,
                base="main",
            ),
        ),
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision=old_base,
    ).model_copy(update={"publication": publication})
    return checkpoint, old_base, candidate


def test_partial_direct_publication_advances_base_and_resume_keeps_only_failures() -> None:
    checkpoint, _, candidate = _partial_checkpoint()
    assert checkpoint.publication is not None
    completed = checkpoint.publication.model_copy(update={"phase": "completed"})

    persisted = runner._checkpoint_publication_progress(checkpoint, completed)
    resumed, completed_indexes = runner._reconcile_publication_for_resume(
        persisted,
        git=_FakeGit(candidate),  # type: ignore[arg-type]
        remote_base=candidate,
        base_branch="main",
        expected_mode="direct",
        source_count=2,
    )
    remaining = runner._remaining_source_entries(
        ("https://t.me/addemoji/Done", "https://t.me/addemoji/Retry"),
        completed_indexes,
    )

    assert persisted.base_revision == candidate
    assert resumed.base_revision == candidate
    assert resumed.publication is not None
    assert resumed.publication.phase == "completed"
    assert remaining == ((1, "https://t.me/addemoji/Retry"),)


def test_resume_continues_when_base_is_old_and_candidate_is_same() -> None:
    checkpoint, old_base, candidate = _partial_checkpoint(phase="candidate_pushed")

    resumed, completed_indexes = runner._reconcile_publication_for_resume(
        checkpoint,
        git=_FakeGit(candidate),  # type: ignore[arg-type]
        remote_base=old_base,
        base_branch="main",
        expected_mode="direct",
        source_count=2,
    )

    assert resumed == checkpoint
    assert completed_indexes == set()


@pytest.mark.parametrize("remote_base", ["c" * 40, "d" * 64])
def test_resume_fails_closed_when_remote_base_is_neither_expected_nor_candidate(
    remote_base: str,
) -> None:
    checkpoint, _, candidate = _partial_checkpoint()

    with pytest.raises(CommandError, match="neither the expected old base"):
        runner._reconcile_publication_for_resume(
            checkpoint,
            git=_FakeGit(candidate),  # type: ignore[arg-type]
            remote_base=remote_base,
            base_branch="main",
            expected_mode="direct",
            source_count=2,
        )


def test_resume_fails_closed_when_saved_candidate_branch_was_replaced() -> None:
    checkpoint, old_base, _ = _partial_checkpoint(phase="candidate_pushed")

    with pytest.raises(CommandError, match="candidate branch no longer matches"):
        runner._reconcile_publication_for_resume(
            checkpoint,
            git=_FakeGit("c" * 40),  # type: ignore[arg-type]
            remote_base=old_base,
            base_branch="main",
            expected_mode="direct",
            source_count=2,
        )


@pytest.mark.asyncio
async def test_sensitive_new_identity_resume_uses_only_runtime_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    confirmation_messages: list[str] = []

    def confirm(message: str) -> bool:
        confirmation_messages.append(message)
        return True

    def confirm_unknown_cost(requests: int) -> bool:
        return requests == 1

    safe_parameters = runner._safe_parameters(
        ("https://t.me/addemoji/Sensitive",),
        runner.PipelineOptions(
            repository="MojiLex/mojilex",
            publish="pr",
            base="main",
            new_identity=True,
            allow_unknown_cost=True,
        ),
    )
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters=safe_parameters,
        cli_version="0.1.0",
        schema_version=runner.SCHEMA_VERSION,
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )

    class FakeStore:
        def load_for_resume(self, run_id: str, *, schema_version: str) -> RunCheckpoint:
            assert run_id == checkpoint.run_id
            assert schema_version == runner.SCHEMA_VERSION
            return checkpoint

    captured: dict[str, object] = {}

    async def fake_run_add(sources, options, **kwargs):  # type: ignore[no-untyped-def]
        captured["sources"] = tuple(sources)
        captured["options"] = options
        captured.update(kwargs)
        return CommandResult(run_id=checkpoint.run_id, status=RunStatus.SUCCEEDED)

    monkeypatch.setattr(runner, "load_config", lambda: SimpleNamespace(runs_dir=tmp_path))
    monkeypatch.setattr(runner, "RunStore", lambda *_args, **_kwargs: FakeStore())
    monkeypatch.setattr(runner, "_run_add", fake_run_add)

    result = await runner.run_resume(
        checkpoint.run_id,
        confirmation=confirm,
        unknown_cost_confirmation=confirm_unknown_cost,
    )

    options = captured["options"]
    assert isinstance(options, runner.PipelineOptions)
    assert result.status is RunStatus.SUCCEEDED
    assert options.new_identity is True
    assert options.confirmation is confirm
    assert options.unknown_cost_confirmation is confirm_unknown_cost
    assert "confirmation" not in checkpoint.safe_parameters
    assert "unknown_cost_confirmation" not in checkpoint.safe_parameters
    assert "yes" not in checkpoint.safe_parameters
    assert confirmation_messages == []
