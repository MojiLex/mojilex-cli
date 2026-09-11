from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from mojilex_cli.policy import (
    ModelQualification,
    ModelQualificationRegistry,
    QualificationQuery,
    QualificationStatus,
    match_qualification,
)
from test_dataset_helpers import (
    NOW,
    PROMPT_HASH,
    QUALIFICATION_ID,
    REQUEST_PARAMETERS_HASH,
    write_fixture,
)


def _query(**updates: object) -> QualificationQuery:
    values: dict[str, object] = {
        "provider": "gemini",
        "model": "test-model",
        "model_revision": "test-revision",
        "description_profile": "standard-v1",
        "prompt_sha256": PROMPT_HASH,
        "request_parameters_sha256": REQUEST_PARAMETERS_HASH,
        "schema_version": "1.0.0",
        "taxonomy_version": "1.0.0",
        "pipeline_version": "1.0.0",
        "routing_policy_version": "1.0.0",
        "languages": ("ru", "en"),
        "generated_at": NOW,
    }
    values.update(updates)
    return QualificationQuery.model_validate(values)


def test_exact_qualification_lookup_matches_every_provenance_field(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    registry = ModelQualificationRegistry.load(tmp_path)

    match = match_qualification(registry, _query())
    assert match.qualified
    assert match.qualification_id == QUALIFICATION_ID

    mismatch = match_qualification(registry, _query(prompt_sha256="9" * 64))
    assert mismatch.status is QualificationStatus.NOT_FOUND
    by_id = match_qualification(
        registry,
        _query(prompt_sha256="9" * 64),
        qualification_id=QUALIFICATION_ID,
    )
    assert by_id.status is QualificationStatus.MISMATCH


def test_qualification_uses_generation_time_half_interval_and_revocation(
    tmp_path: Path,
) -> None:
    write_fixture(tmp_path)
    loaded = ModelQualificationRegistry.load(tmp_path)
    raw = loaded.entries[0].model_dump(mode="json")
    raw["valid_until"] = "2026-07-01T00:00:00Z"
    expiring = ModelQualification.model_validate(raw)
    registry = loaded.model_copy(update={"entries": (expiring,)})

    assert match_qualification(
        registry,
        _query(generated_at="2026-06-30T23:59:59Z"),
    ).qualified
    assert match_qualification(registry, _query()).status is (QualificationStatus.OUTSIDE_VALIDITY)

    revoked = expiring.model_copy(update={"status": "revoked"})
    registry = loaded.model_copy(update={"entries": (revoked,)})
    assert (
        match_qualification(
            registry,
            _query(generated_at="2026-06-30T23:59:59Z"),
        ).status
        is QualificationStatus.REVOKED
    )


def test_revisionless_qualification_requires_valid_until(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    entry = ModelQualificationRegistry.load(tmp_path).entries[0]
    raw = entry.model_dump(mode="json", exclude={"valid_until"})
    raw["model_revision"] = None

    with pytest.raises(ValidationError, match="valid_until is required"):
        ModelQualification.model_validate(raw)
