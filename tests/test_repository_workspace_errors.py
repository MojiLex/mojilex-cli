from __future__ import annotations

from pathlib import Path

import pytest

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.pipeline.runner import repository_workspace


def test_missing_local_repository_path_has_specific_configuration_error(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing-mojilex"

    with pytest.raises(CommandError) as captured:
        with repository_workspace(str(missing), "main"):
            pytest.fail("a missing local repository must not open a workspace")

    error = captured.value.error
    assert error.code == "CONFIG_INVALID"
    assert "local path" in error.message
    assert error.details["repository_target"] == str(missing)
    assert "--repo OWNER/REPO" in error.hint


def test_owner_repository_is_not_misclassified_as_local_path() -> None:
    from mojilex_cli.pipeline import runner

    assert runner._looks_like_local_repository_path("MojiLex/mojilex") is False
    assert runner._looks_like_local_repository_path(r"..\mojilex") is True
