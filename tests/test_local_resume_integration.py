from contextlib import contextmanager
from pathlib import Path

import pytest

from mojilex_cli.ai import BudgetExceededError
from mojilex_cli.dataset import load_dataset, validate_dataset
from mojilex_cli.pipeline import runner
from test_add_run_staging import saved_add  # noqa: F401
from test_ai_recovery_checkpoint import _prepare_exact_request, _RecoveryProvider
from test_incremental_pack_readiness import _real_packs
from test_pack_describe_pipeline import pipeline  # noqa: F401
from test_pipeline_resume_cache import _descriptions_for_collection


@pytest.mark.parametrize("legacy_checkpoint", [False, True])
async def test_local_add_budget_resume_persists_and_reuses_paid_response(
    request, monkeypatch, legacy_checkpoint
):
    state = request.getfixturevalue("pipeline")
    _real_packs(state, monkeypatch)
    state.config = state.config.model_copy(
        update={
            "ai": state.config.ai.model_copy(
                update={"max_ai_requests": 1, "ai_concurrency": 1, "model_routing": "off"}
            ),
            "processing": state.config.processing.model_copy(
                update={"pack_concurrency": 1, "file_analysis_mode": "sequential"}
            ),
        }
    )
    destinations = []

    @contextmanager
    def local_workspace(target, branch, **kwargs):
        # Reject accidental remote resolution without permitting network access.
        assert Path(target).resolve() == state.root.resolve()
        destinations.append(Path(target).resolve())
        yield runner.RepositoryWorkspace(root=state.root, target=state.target, temporary=False)

    def resolve(options):
        return state.config.model_copy(
            update={
                "repository": state.config.repository.model_copy(
                    update={"target": options.repository or str(state.root)}
                ),
                "ai": state.config.ai.model_copy(
                    update={
                        "max_ai_requests": (
                            options.max_ai_requests
                            if options.max_ai_requests is not None
                            else state.config.ai.max_ai_requests
                        )
                    }
                ),
            }
        )

    provider = _RecoveryProvider([], None)

    async def fixed_provider(*args, **kwargs):
        return provider

    monkeypatch.setattr(runner, "repository_workspace", local_workspace)
    monkeypatch.setattr(runner, "_resolved_config", resolve)
    monkeypatch.setattr(runner, "_descriptions_for_collection", _descriptions_for_collection)
    monkeypatch.setattr(runner, "_prepare_ai_request", _prepare_exact_request)
    monkeypatch.setattr(runner, "_provider_for_model", fixed_provider)
    before = load_dataset(state.root).to_files()
    with pytest.raises(BudgetExceededError, match="request limit"):
        await runner._run_add(
            tuple(source.canonical_url for source in state.sources),
            runner.PipelineOptions(repository=str(state.root), publish="local", max_ai_requests=1),
        )
    checkpoint = state.latest_checkpoint()
    assert checkpoint.ai_requests_used == 1
    assert len(provider.calls) == 1
    first_paid = provider.calls[0]
    if legacy_checkpoint:
        # Older local add runs saved the explicit destination but no staging binding.
        parameters = dict(checkpoint.safe_parameters)
        parameters.pop("staging_repository")
        state.store.save(checkpoint.model_copy(update={"safe_parameters": parameters}))

    result = await runner.run_resume(checkpoint.run_id, max_ai_requests=2)

    assert result.status == "succeeded"
    assert Path(result.publication["path"]) == state.root.resolve()
    assert state.root.is_dir()
    assert destinations == [state.root.resolve(), state.root.resolve()]
    assert len(provider.calls) == 2
    assert provider.calls.count(first_paid) == 1
    assert state.latest_checkpoint().ai_requests_used == 2
    assert load_dataset(state.root).to_files() != before
    assert validate_dataset(state.root, strict=True).valid
    snapshot = load_dataset(state.root)
    added_ids = {item.native_id for source in state.sources for item in source.items}
    assert added_ids <= {emoji.native_id for emoji in snapshot.emojis.values()}
