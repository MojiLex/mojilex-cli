import pytest

from mojilex_cli.commands.runtime import CommandError, CommandResult
from mojilex_cli.pipeline import runner
from test_pack_import_pipeline import (
    test_shared_emoji_or_duplicate_pack_reuses_completed_owner_before_next_pack as _scenario,
)


@pytest.mark.parametrize("raises", [False, True])
async def test_real_import_scenario_reports_early_failure_not_timeout(
    tmp_path_factory, monkeypatch, raises
):
    async def fail_before_media(*args, **kwargs):
        if raises:
            raise RuntimeError("synthetic startup failure")
        return CommandResult(
            status="failed",
            errors=[
                CommandError(
                    "GIT_CONFLICT", "synthetic startup failure", hint="Synthetic test failure."
                ).error
            ],
        )

    monkeypatch.setattr(runner, "_run_import", fail_before_media)
    expected = RuntimeError if raises else AssertionError
    with pytest.raises(expected, match="synthetic startup failure"):
        await _scenario(tmp_path_factory, monkeypatch, duplicate_pack=False)
