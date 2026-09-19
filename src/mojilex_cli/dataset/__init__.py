"""Dataset API used by commands, CI, and integration consumers."""

from .index import IndexBuildResult, build_index
from .layout import (
    collection_path,
    collection_shard,
    emoji_bucket_path,
    emoji_shards,
    memberships_path,
    tombstone_path,
    tombstone_shard,
    visual_relation_shards,
    visual_relations_path,
)
from .merge import (
    IdentityContinuity,
    assess_identity_continuity,
    merge_collection,
    merge_emoji,
    merge_memberships,
)
from .repository import DatasetLoadError, DatasetSnapshot, load_dataset
from .serialization import (
    canonical_entity,
    compact_json,
    parse_json,
    parse_jsonl,
    pretty_json,
    serialize_collection,
    serialize_emojis,
    serialize_memberships,
    serialize_tombstone,
    serialize_visual_relations,
)
from .staging import AtomicDatasetWriter, AtomicWriteError, apply_snapshot
from .validation import (
    DatasetValidationError,
    ValidationIssue,
    ValidationReport,
    load_validated_dataset,
    validate_dataset,
    validate_snapshot,
)

__all__ = [
    "AtomicDatasetWriter",
    "AtomicWriteError",
    "DatasetLoadError",
    "DatasetSnapshot",
    "DatasetValidationError",
    "IdentityContinuity",
    "IndexBuildResult",
    "ValidationIssue",
    "ValidationReport",
    "apply_snapshot",
    "assess_identity_continuity",
    "build_index",
    "canonical_entity",
    "collection_path",
    "collection_shard",
    "compact_json",
    "emoji_bucket_path",
    "emoji_shards",
    "load_dataset",
    "load_validated_dataset",
    "memberships_path",
    "merge_collection",
    "merge_emoji",
    "merge_memberships",
    "parse_json",
    "parse_jsonl",
    "pretty_json",
    "serialize_collection",
    "serialize_emojis",
    "serialize_memberships",
    "serialize_tombstone",
    "serialize_visual_relations",
    "tombstone_path",
    "tombstone_shard",
    "validate_dataset",
    "validate_snapshot",
    "visual_relation_shards",
    "visual_relations_path",
]
