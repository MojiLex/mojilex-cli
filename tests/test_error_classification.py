import pytest

from mojilex_cli.commands.runtime import structured_exception
from mojilex_cli.dataset import (
    DatasetLoadError,
    DatasetValidationError,
    ValidationIssue,
    ValidationReport,
)
from mojilex_cli.runs import ResumeIncompatibleError, RunLockedError, RunStoreError


def test_dataset_validation_has_a_stable_public_exception() -> None:
    report = ValidationReport((ValidationIssue("SCHEMA", "dataset.json", "invalid"),))

    with pytest.raises(DatasetValidationError) as captured:
        report.raise_for_errors()

    assert structured_exception(captured.value).code == "VALIDATION_FAILED"
    assert structured_exception(DatasetLoadError("malformed dataset")).code == "VALIDATION_FAILED"


@pytest.mark.parametrize(
    ("error", "expected_code", "retryable"),
    [
        (RunStoreError("invalid run ID"), "CONFIG_INVALID", False),
        (RunLockedError("run is locked"), "GIT_CONFLICT", True),
        (
            ResumeIncompatibleError("base revision changed"),
            "SOURCE_CHANGED_DURING_RUN",
            False,
        ),
    ],
)
def test_run_store_errors_have_stable_classification(
    error: RunStoreError, expected_code: str, retryable: bool
) -> None:
    structured = structured_exception(error)
    assert structured.code == expected_code
    assert structured.retryable is retryable
