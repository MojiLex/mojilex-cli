from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mojilex_cli.media import MediaMetadata
from mojilex_cli.media.models import HARD_MAX_DURATION_MS
from mojilex_cli.runs import (
    AIRequestCheckpoint,
    ElementCheckpoint,
    RunStore,
    RunStoreError,
    new_checkpoint,
)


def _media_row(**updates: object) -> dict[str, Any]:
    row: dict[str, Any] = {
        "role": "primary",
        "kind": "static",
        "format": "webp",
        "mime_type": "image/webp",
        "sha256": "a" * 64,
        "byte_size": 128,
        "width": 32,
        "height": 32,
        "animated": False,
    }
    row.update(updates)
    return row


@pytest.mark.parametrize(
    "row",
    [
        _media_row(),
        _media_row(
            kind="animation",
            format="tgs",
            mime_type="application/x-tgsticker",
            animated=True,
            duration_ms=1,
        ),
        _media_row(
            kind="video",
            format="webm",
            mime_type="video/webm",
            animated=True,
            duration_ms=HARD_MAX_DURATION_MS,
        ),
    ],
)
def test_cache_media_metadata_accepts_only_canonical_format_contracts(
    row: dict[str, Any],
) -> None:
    assert MediaMetadata.model_validate(row).format == row["format"]


@pytest.mark.parametrize(
    "updates",
    [
        {"kind": "animation"},
        {"mime_type": "video/webm"},
        {"animated": True, "duration_ms": 100},
        {
            "format": "tgs",
            "kind": "static",
            "mime_type": "application/x-tgsticker",
            "animated": True,
            "duration_ms": 100,
        },
        {
            "format": "tgs",
            "kind": "animation",
            "mime_type": "video/webm",
            "animated": True,
            "duration_ms": 100,
        },
        {
            "format": "webm",
            "kind": "animation",
            "mime_type": "video/webm",
            "animated": True,
            "duration_ms": 100,
        },
        {
            "format": "webm",
            "kind": "video",
            "mime_type": "video/webm",
            "animated": False,
        },
    ],
)
def test_cache_media_metadata_rejects_schema_shaped_but_inconsistent_rows(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="must match format"):
        MediaMetadata.model_validate(_media_row(**updates))


@pytest.mark.parametrize(
    "row",
    [
        _media_row(duration_ms=1),
        _media_row(
            kind="animation",
            format="tgs",
            mime_type="application/x-tgsticker",
            animated=True,
        ),
        _media_row(
            kind="video",
            format="webm",
            mime_type="video/webm",
            animated=True,
            duration_ms=0,
        ),
        _media_row(
            kind="video",
            format="webm",
            mime_type="video/webm",
            animated=True,
            duration_ms=HARD_MAX_DURATION_MS + 1,
        ),
    ],
)
def test_cache_media_metadata_rejects_invalid_duration_contracts(
    row: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        MediaMetadata.model_validate(row)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("byte_size", "128"),
        ("width", "32"),
        ("height", 32.0),
        ("animated", 0),
        ("duration_ms", "100"),
    ],
)
def test_cache_media_metadata_does_not_coerce_technical_facts(
    field: str,
    value: object,
) -> None:
    row = _media_row(
        kind="video",
        format="webm",
        mime_type="video/webm",
        animated=True,
        duration_ms=100,
    )
    row[field] = value
    with pytest.raises(ValueError):
        MediaMetadata.model_validate(row)


def _ai_request(**updates: object) -> dict[str, object]:
    request: dict[str, object] = {
        "stage": "primary",
        "model": "gemini-test-model",
        "model_revision": "revision-1",
        "cache_key": "b" * 64,
        "plan_sha256": "c" * 64,
        "request_sha256": "d" * 64,
        "shown_media_sha256": ["e" * 64],
        "item_label": "E001",
    }
    request.update(updates)
    return request


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cache_key", "B" * 64),
        ("cache_key", "b" * 63),
        ("cache_key", "b" * 64 + "\n"),
        ("plan_sha256", "C" * 64),
        ("request_sha256", "d" * 65),
        ("shown_media_sha256", ["E" * 64]),
        ("shown_media_sha256", ["e" * 64 + "\n"]),
        ("item_label", "E01"),
        ("item_label", "E001\n"),
        ("model", " test-model"),
        ("model", "test\nmodel"),
        ("model", "m" * 257),
        ("model_revision", ""),
        ("model_revision", "r" * 257),
        ("shown_media_sha256", ["e" * 64] * 33),
    ],
)
def test_ai_request_checkpoint_rejects_unbounded_or_noncanonical_trace_fields(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        AIRequestCheckpoint.model_validate(_ai_request(**{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_descriptor_sha256", "A" * 64),
        ("source_descriptor_sha256", "a" * 64 + "\n"),
        ("media_sha256", ["A" * 64]),
        ("media_sha256", ["a" * 63]),
        ("media_sha256", ["a" * 64 + "\n"]),
        ("deterministic_cache_key", "B" * 64),
        ("deterministic_cache_key", "b" * 65),
        ("ai_cache_key", "C" * 64),
        ("ai_cache_key", "c" * 64 + "\n"),
    ],
)
def test_element_checkpoint_rejects_noncanonical_hashes(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        ElementCheckpoint.model_validate({"stage": "fingerprint_ready", field: value})


def test_run_store_rejects_corrupt_checkpoint_hash_on_load(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    request = AIRequestCheckpoint.model_validate(_ai_request())
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={"sources": ["https://t.me/addemoji/Pack"]},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    ).model_copy(
        update={
            "elements": {
                "native-id": ElementCheckpoint(
                    stage="ai_cached",
                    source_descriptor_sha256="a" * 64,
                    media_sha256=("a" * 64,),
                    deterministic_cache_key="f" * 64,
                    ai_cache_key=request.cache_key,
                    ai_requests=(request,),
                    palette_complete=True,
                    fingerprint_complete=True,
                    ai_facets_complete=True,
                )
            }
        }
    )
    path = store.save(checkpoint)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["elements"]["native-id"]["media_sha256"] = ["A" * 64]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RunStoreError, match="checkpoint is malformed"):
        store.load(checkpoint.run_id)
