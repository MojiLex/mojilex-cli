"""Bounded collection identities, without reading descriptions or memberships.

This partial view is for official-pack selection, never publication validation.
The caller supplies a fresh checkout of the public main branch; local authoring
workspaces must not be used as the official source.
"""

from pathlib import Path, PurePosixPath

from mojilex_cli.domain.models import Collection

from .layout import assert_no_link_or_reparse, collection_path, collection_shard
from .repository import DatasetLoadError, DatasetSnapshot, _insert
from .serialization import parse_json, serialize_collection
from .validation import (
    DatasetValidationError,
    ValidationIssue,
    ValidationReport,
    _validate_ids_and_paths,
    _validate_json_schemas,
    _validate_manifest,
)

_MAX_FILE_BYTES = 1024 * 1024
_MAX_TOTAL_BYTES = 16 * 1024 * 1024
_MAX_COLLECTIONS = 10_000
_MAX_SCHEMAS = 128


def load_collection_names(root: Path) -> frozenset[str]:
    """Validate public Telegram identities without materializing emoji records."""
    try:
        assert_no_link_or_reparse(root)
        root = root.resolve()
        total_bytes = 0

        def read(path: Path) -> bytes:
            nonlocal total_bytes
            assert_no_link_or_reparse(path, boundary=root)
            with path.open("rb") as stream:
                payload = stream.read(_MAX_FILE_BYTES + 1)
            total_bytes += len(payload)
            if len(payload) > _MAX_FILE_BYTES or total_bytes > _MAX_TOTAL_BYTES:
                raise DatasetLoadError("official collection metadata exceeds its safe limit")
            return payload

        manifest = parse_json(read(root / "dataset.json"), source="dataset.json")
        snapshot = DatasetSnapshot(root, manifest)
        schema_root = root / "schemas" / "v1"
        assert_no_link_or_reparse(schema_root, boundary=root)
        # Bound the same schema graph consumed by the existing strict validator.
        for count, path in enumerate(schema_root.rglob("*.json"), start=1):
            if count > _MAX_SCHEMAS:
                raise DatasetLoadError("official schema graph exceeds its safe limit")
            read(path)
        catalog = root / "data" / "telegram" / "collections"
        assert_no_link_or_reparse(catalog, boundary=root)
        paths: set[Path] = set()
        for pattern in ("*/collection.json", "*/*/collection.json"):
            for path in catalog.glob(pattern):
                paths.add(path)
                if len(paths) > _MAX_COLLECTIONS:
                    raise DatasetLoadError("official collection count exceeds its safe limit")
        # A deleted record must not make its surviving pack silently disappear.
        for path in catalog.glob("*"):
            assert_no_link_or_reparse(path, boundary=root)
            if path.name.startswith("mxc_") and not (path / "collection.json").is_file():
                raise DatasetLoadError("official collection record is missing")
            if path.is_dir() and len(path.name) == 2:
                for child in path.iterdir():
                    assert_no_link_or_reparse(child, boundary=root)
                    if child.name.startswith("mxc_") and not (child / "collection.json").is_file():
                        raise DatasetLoadError("official collection record is missing")
        for path in sorted(paths):
            raw = read(path)
            collection = Collection.model_validate(parse_json(raw, source=str(path)))
            _insert(snapshot.collections, collection, path)
            relative = PurePosixPath(path.relative_to(root).as_posix())
            canonical = collection_path("telegram", collection.id)
            legacy = canonical.parent.parent / collection_shard(collection.id) / collection.id
            if relative not in {canonical, legacy / "collection.json"}:
                raise DatasetLoadError("official collection is stored under a noncanonical ID")
            if serialize_collection(collection) != raw:
                raise DatasetLoadError("official collection JSON is not canonical")
            extension = collection.extensions.get("telegram")
            if (
                collection.platform != "telegram"
                or collection.kind != "custom_emoji_set"
                or collection.native_namespace != "sticker_set.name"
                or collection.scope_id != "global"
                or collection.canonical_url != f"https://t.me/addemoji/{collection.native_id}"
                or not isinstance(extension, dict)
                or extension.get("short_name") != collection.native_id
            ):
                raise DatasetLoadError("official Telegram collection identity is invalid")
        issues: list[ValidationIssue] = []
        namespace = _validate_manifest(snapshot, issues)
        _validate_ids_and_paths(snapshot, namespace, issues, canonical=False)
        _validate_json_schemas(snapshot, issues)
        ValidationReport(tuple(issues)).raise_for_errors()
        return frozenset(value.native_id.casefold() for value in snapshot.collections.values())
    except (OSError, ValueError) as exc:
        if isinstance(exc, (DatasetLoadError, DatasetValidationError)):
            raise
        raise DatasetLoadError("cannot read valid official collection identities") from exc
