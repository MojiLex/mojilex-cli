"""Read public readiness records without materializing unrelated emoji buckets.

This is a partial view for readiness checks, never a publication validation result.
"""

from collections.abc import Collection as Names
from pathlib import Path, PurePosixPath

from mojilex_cli.domain.models import Collection, Emoji, Membership

from .layout import assert_no_link_or_reparse, emoji_bucket_path, legacy_bucket_path
from .repository import DatasetLoadError, DatasetSnapshot, _insert, _read
from .serialization import parse_json, parse_jsonl
from .transaction import locked_dataset_transaction_view


def load_pack_snapshot(root: str | Path, names: Names[str]) -> DatasetSnapshot:
    """Read shared membership metadata once and only requested Telegram emoji buckets.

    Both current and legacy buckets are examined so duplicate IDs cannot silently
    overwrite one another. Missing emoji remain absent for the caller's exact
    readiness checks. The transaction view and path guards match full loading.
    """
    try:
        unresolved = Path(root)
        assert_no_link_or_reparse(unresolved)
        root_path = unresolved.resolve()
        with locked_dataset_transaction_view(root_path):
            source: dict[PurePosixPath, bytes] = {}
            manifest = parse_json(
                _read(root_path / "dataset.json", root_path, source), source="dataset.json"
            )
            snapshot = DatasetSnapshot(root_path, manifest, source_bytes=source)
            data_root = root_path / "data"
            assert_no_link_or_reparse(data_root, boundary=root_path)
            catalog_file = data_root / "telegram" / "collections" / "README.md"
            if catalog_file.is_file():
                _read(catalog_file, root_path, source)
            collection_files = set(data_root.glob("*/collections/*/collection.json"))
            collection_files.update(data_root.glob("*/collections/*/*/collection.json"))
            for path in sorted(collection_files):
                collection = Collection.model_validate(
                    parse_json(_read(path, root_path, source), source=str(path))
                )
                _insert(snapshot.collections, collection, path)
                memberships = path.with_name("memberships.jsonl")
                for raw in parse_jsonl(
                    _read(memberships, root_path, source), source=str(memberships)
                ):
                    _insert(snapshot.memberships, Membership.model_validate(raw), memberships)
            selected = {
                value.id
                for value in snapshot.collections.values()
                if value.platform == "telegram" and value.native_id in names
            }
            buckets: set[PurePosixPath] = set()
            for row in snapshot.memberships.values():
                if row.collection_id in selected and row.status == "active":
                    bucket = emoji_bucket_path("telegram", row.emoji_id)
                    buckets.update((bucket, legacy_bucket_path(bucket)))
            for relative in sorted(buckets):
                path = root_path.joinpath(*relative.parts)
                # Check even absent paths: a dangling link must not look like a miss.
                assert_no_link_or_reparse(path, boundary=root_path)
                if not path.exists():
                    continue
                for raw in parse_jsonl(_read(path, root_path, source), source=str(path)):
                    _insert(snapshot.emojis, Emoji.model_validate(raw), path)
            return snapshot
    except (OSError, ValueError) as exc:
        if isinstance(exc, DatasetLoadError):
            raise
        raise DatasetLoadError(str(exc)) from exc
