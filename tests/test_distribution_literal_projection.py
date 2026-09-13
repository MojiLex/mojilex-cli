from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]

from mojilex_cli.dataset import load_dataset
from mojilex_cli.dataset.staging import AtomicDatasetWriter
from mojilex_cli.domain import ContentType, TextContent, TextKind
from mojilex_cli.read.snapshot import _embedded_schema_contracts
from test_build_index_determinism import _build, _prepare_distribution_fixture, _tree_bytes


def test_all_canonical_literal_kinds_project_to_the_existing_search_schema(tmp_path: Path) -> None:
    dataset = _prepare_distribution_fixture(tmp_path / "dataset")
    snapshot = load_dataset(dataset)
    emoji = next(iter(snapshot.emojis.values()))
    kinds = list(TextKind)
    emoji.facets.content_types.extend([ContentType.NUMBER, ContentType.TEXT])
    emoji.facets.content_types.sort(key=str)
    emoji.facets.text_content = TextContent.model_validate(
        {
            "status": "recognized",
            "dynamics": "stable",
            "items": [
                {
                    "value": f"literal {index}",
                    "kind": kind.value,
                    "script": "Latn",
                    "language": "en",
                    "temporal_scope": "persistent",
                    "media_refs": [{"role": "primary"}],
                }
                for index, kind in enumerate(kinds)
            ],
        }
    )
    expected_kinds = ["symbol", "number", "word", "phrase", "symbol", "mixed", "symbol", "mixed"]
    assert len(kinds) == len(expected_kinds) == 8
    writer = AtomicDatasetWriter(dataset)
    for path, payload in snapshot.to_files().items():
        writer.stage_bytes(path, payload)
    writer.commit()
    before = _tree_bytes(dataset, include_transaction_lock=False)
    output = tmp_path / "dist"
    _build(dataset, output)
    canonical = json.loads((output / "emojis.jsonl").read_bytes())
    assert [item["kind"] for item in canonical["facets"]["text_content"]["items"]] == [
        kind.value for kind in kinds
    ]
    _schemas, registry, _resources = _embedded_schema_contracts()
    validator = Draft202012Validator(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$ref": "mlx://schemas/distribution/v1/search-record.schema.json",
        },
        registry=registry,
        format_checker=FormatChecker(),
    )
    for language in ("en", "ru"):
        row = json.loads((output / f"search-{language}.jsonl").read_bytes())
        assert not list(validator.iter_errors(row))
        assert [item["kind"] for item in row["literal_text"]] == expected_kinds
        for original, projected in zip(
            canonical["facets"]["text_content"]["items"], row["literal_text"], strict=True
        ):
            assert {key: value for key, value in projected.items() if key != "kind"} == {
                key: value for key, value in original.items() if key not in {"kind", "media_refs"}
            }
    assert _tree_bytes(dataset, include_transaction_lock=False) == before
