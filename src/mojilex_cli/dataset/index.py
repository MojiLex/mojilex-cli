"""Deterministic release-index builder."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from mojilex_cli.domain.hashes import media_digest
from mojilex_cli.domain.models import ContentRating, ReviewStatus

from .distribution import build_distribution
from .layout import assert_no_link_or_reparse, is_link_or_reparse_point
from .repository import DatasetSnapshot, load_dataset
from .serialization import canonical_entity, parse_json, pretty_json, serialize_jsonl
from .staging import AtomicDatasetWriter
from .transaction import recover_pending_dataset_transaction
from .validation import validate_snapshot

_GIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_PAYLOAD_NAMES = (
    "collections.jsonl",
    "emojis.jsonl",
    "memberships.jsonl",
    "tombstones.jsonl",
    "emojis-active.jsonl",
    "search-ru.jsonl",
    "search-en.jsonl",
    "collection-facets.jsonl",
    "duplicate-groups.jsonl",
    "visual-relations.jsonl",
    "taxonomy.json",
)
_PROTECTED_DATASET_DIRECTORIES = (
    ".git",
    ".github",
    "analysis-profiles",
    "data",
    "examples",
    "platforms",
    "quality",
    "rights",
    "schemas",
    "taxonomy",
    "tests",
    "tombstones",
    "tools",
)
_MANAGED_OUTPUT_NAMES = frozenset((*_PAYLOAD_NAMES, "manifest.json", "SHA256SUMS"))
_MAX_EXISTING_OUTPUT_BYTES = 1024 * 1024 * 1024
_MAX_EXISTING_MANIFEST_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class IndexBuildResult:
    output_directory: Path
    git_commit: str
    git_object_format: str
    tool_commit: str
    dependency_lock_sha256: str
    file_sha256: dict[str, str]
    counts: dict[str, Any]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_commit(root: Path, *, option: str = "--git-commit") -> str:
    if not (root / ".git").exists():
        raise ValueError(f"cannot resolve a Git checkout at {root}; pass {option} explicitly")
    result = subprocess.run(
        ["git", "-c", f"safe.directory={root}", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
        timeout=10,
        shell=False,
    )
    value = result.stdout.strip().lower()
    if result.returncode != 0 or not _GIT_SHA_RE.fullmatch(value):
        raise ValueError(f"cannot resolve a full Git commit at {root}; pass {option} explicitly")
    return value


def _dependency_contract_sha256(tool_root: Path, explicit: str | None) -> str:
    if explicit is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", explicit):
            raise ValueError("dependency_lock_sha256 must be a lowercase SHA-256 digest")
        return explicit
    contract = tool_root / "pyproject.toml"
    try:
        return _sha(contract.read_bytes())
    except OSError as exc:
        raise ValueError(
            "cannot resolve the CLI dependency contract; pass --dependency-lock-sha256 explicitly"
        ) from exc


def _tool_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _counter(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _status_counts(snapshot: DatasetSnapshot) -> dict[str, Any]:
    return {
        "collections": _counter(
            [item.availability.status.value for item in snapshot.collections.values()]
        ),
        "emojis": _counter([item.availability.status.value for item in snapshot.emojis.values()]),
        "memberships": _counter([item.status.value for item in snapshot.memberships.values()]),
        "reviews": _counter([item.review.status.value for item in snapshot.emojis.values()]),
        "ratings": _counter([item.content.rating.value for item in snapshot.emojis.values()]),
    }


def _publishable(emoji: Any) -> bool:
    if emoji.review.status is ReviewStatus.APPROVED:
        return True
    return (
        emoji.review.status is ReviewStatus.UNREVIEWED
        and emoji.content.rating is ContentRating.GENERAL
        and not emoji.content.warnings
    )


def _search_row(
    emoji: Any,
    collection_ids: list[str],
    duplicate_group_ids: list[str],
    language: str,
) -> dict[str, Any]:
    description = emoji.descriptions[language]
    result: dict[str, Any] = {
        "emoji_id": emoji.id,
        "platform": emoji.platform,
        "native_namespace": emoji.native_namespace,
        "scope_id": emoji.scope_id,
        "native_id": emoji.native_id,
        "collection_ids": collection_ids,
        "text": description.text,
    }
    if description.motion is not None:
        result["motion"] = description.motion
    result["usage"] = description.usage
    result["semantic_tags"] = emoji.semantic_tags
    rendering = emoji.facets.rendering.items
    result["facets"] = {
        "animated": any(item.animated for item in emoji.media),
        "media_kinds": sorted({item.kind.value for item in emoji.media}),
        "color_behaviors": sorted({item.color_behavior.value for item in rendering}),
        "alpha_modes": sorted({item.alpha_mode.value for item in rendering}),
        "color_families": sorted(
            {color.family.value for item in rendering for color in (item.dominant_colors or ())}
        ),
        "text_status": emoji.facets.text_content.status.value,
        "literal_text": [item.value for item in emoji.facets.text_content.items],
        "content_types": [item.value for item in emoji.facets.content_types],
        "styles": [item.value for item in emoji.facets.styles],
        "suggested_uses": [item.value for item in emoji.facets.suggested_uses],
        "uncertainties": [item.value for item in emoji.facets.uncertainties],
    }
    result["duplicate_group_ids"] = duplicate_group_ids
    result["review_status"] = emoji.review.status.value
    return result


def _current_approved_relations(snapshot: DatasetSnapshot) -> list[Any]:
    """Select public relations and independently reject stale evidence."""

    result: list[Any] = []
    tombstoned = set(snapshot.tombstones)
    for relation in sorted(snapshot.relations.values(), key=lambda item: item.id):
        if relation.review.status is not ReviewStatus.APPROVED:
            continue
        if relation.subject_id in tombstoned or relation.object_id in tombstoned:
            continue
        subject = snapshot.emojis.get(relation.subject_id)
        object_emoji = snapshot.emojis.get(relation.object_id)
        if subject is None or object_emoji is None:
            continue
        if relation.evidence.subject_media_digest != media_digest(subject.media):
            continue
        if relation.evidence.object_media_digest != media_digest(object_emoji.media):
            continue
        result.append(relation)
    return result


def _reviewed_same_artwork_groups(
    snapshot: DatasetSnapshot, relations: list[Any]
) -> list[dict[str, Any]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for relation in relations:
        if relation.relation_type.value != "same-artwork" or relation.scope.value != "entity":
            continue
        graph[relation.subject_id].add(relation.object_id)
        graph[relation.object_id].add(relation.subject_id)
    namespace = uuid.UUID(str(snapshot.manifest["visual_relation_namespace"]))
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for start in sorted(graph):
        if start in seen:
            continue
        pending = [start]
        component: list[str] = []
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            pending.extend(sorted(graph[current] - seen, reverse=True))
        members = sorted(component)
        if len(members) < 2:
            continue
        name = "\0".join(("duplicate-group", "reviewed-same-artwork", "entity", "", *members))
        result.append(
            {
                "id": f"mxdg_{uuid.uuid5(namespace, name)}",
                "group_type": "reviewed-same-artwork",
                "scope": "entity",
                "members": members,
            }
        )
    return result


def _duplicate_groups(
    snapshot: DatasetSnapshot,
    eligible_emojis: dict[str, Any],
    relations: list[Any],
) -> list[dict[str, Any]]:
    # Deferred to avoid the dataset package importing the dedupe package while
    # DatasetSnapshot itself is still being initialized.
    from mojilex_cli.dedupe import scan_snapshot

    public_snapshot = snapshot.clone()
    public_snapshot.emojis = dict(eligible_emojis)
    exact = list(scan_snapshot(public_snapshot, mode="exact").exact_groups)
    reviewed = _reviewed_same_artwork_groups(snapshot, relations)
    return sorted((*exact, *reviewed), key=lambda item: str(item["id"]))


def _group_ids_by_emoji(groups: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        for member in group["members"]:
            emoji_id = str(member["emoji_id"] if isinstance(member, dict) else member)
            result[emoji_id].append(str(group["id"]))
    for identifiers in result.values():
        identifiers.sort()
    return result


def _collection_facet_rows(
    collections: list[Any],
    memberships: list[Any],
    emojis: dict[str, Any],
    group_ids: dict[str, list[str]],
    groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    emoji_ids_by_collection: dict[str, list[str]] = defaultdict(list)
    for membership in memberships:
        emoji_ids_by_collection[membership.collection_id].append(membership.emoji_id)
    groups_by_id = {str(group["id"]): group for group in groups}
    result: list[dict[str, Any]] = []
    for collection in collections:
        emoji_ids = sorted(emoji_ids_by_collection.get(collection.id, ()))
        values = [emojis[emoji_id] for emoji_id in emoji_ids if emoji_id in emojis]
        media_kinds: Counter[str] = Counter()
        content_types: Counter[str] = Counter()
        styles: Counter[str] = Counter()
        suggested_uses: Counter[str] = Counter()
        color_families: Counter[str] = Counter()
        adaptive = fixed = recognized_text = numbers = 0
        collection_groups: set[str] = set()
        for emoji in values:
            primary = next(item for item in emoji.media if item.role.value == "primary")
            media_kinds[primary.kind.value] += 1
            content_types.update(item.value for item in emoji.facets.content_types)
            styles.update(item.value for item in emoji.facets.styles)
            suggested_uses.update(item.value for item in emoji.facets.suggested_uses)
            behaviors = {item.color_behavior.value for item in emoji.facets.rendering.items}
            if behaviors & {"platform-adaptive", "mixed"}:
                adaptive += 1
            if "fixed" in behaviors:
                fixed += 1
            color_families.update(
                {
                    color.family.value
                    for item in emoji.facets.rendering.items
                    for color in (item.dominant_colors or ())
                }
            )
            if emoji.facets.text_content.status.value in {
                "recognized",
                "partially-recognized",
            }:
                recognized_text += 1
            if any(item.value == "number" for item in emoji.facets.content_types):
                numbers += 1
            collection_groups.update(group_ids.get(emoji.id, ()))
        count = len(values)
        exact_count = sum(
            groups_by_id[group_id]["group_type"] in {"binary-exact", "decoded-exact"}
            for group_id in collection_groups
        )
        reviewed_count = sum(
            groups_by_id[group_id]["group_type"] == "reviewed-same-artwork"
            for group_id in collection_groups
        )
        result.append(
            {
                "collection_id": collection.id,
                "active_memberships": count,
                "media_kind_counts": dict(sorted(media_kinds.items())),
                "adaptive_share_bp": round(adaptive * 10_000 / count) if count else 0,
                "fixed_share_bp": round(fixed * 10_000 / count) if count else 0,
                "content_type_counts": dict(sorted(content_types.items())),
                "style_counts": dict(sorted(styles.items())),
                "recognized_text_count": recognized_text,
                "number_count": numbers,
                "color_family_counts": dict(sorted(color_families.items())),
                "suggested_use_counts": dict(sorted(suggested_uses.items())),
                "exact_duplicate_group_count": exact_count,
                "reviewed_visual_duplicate_group_count": reviewed_count,
            }
        )
    return result


def _taxonomy_snapshot(root: Path, version: str) -> dict[str, Any]:
    taxonomy_root = root / "taxonomy" / "v1"
    manifest = parse_json(
        (taxonomy_root / "taxonomy.json").read_bytes(),
        source=str(taxonomy_root / "taxonomy.json"),
    )
    registries: dict[str, Any] = {}
    for entry in manifest["registries"]:
        raw_path = Path(str(entry["path"]))
        candidate = (
            root / raw_path
            if raw_path.parts[:2] == ("taxonomy", "v1")
            else taxonomy_root / raw_path
        )
        path = candidate.resolve()
        if candidate.is_symlink() or path.parent != taxonomy_root.resolve():
            raise ValueError("taxonomy registry path must stay directly inside taxonomy/v1")
        registry = parse_json(path.read_bytes(), source=str(path))
        registries[str(entry["facet"])] = registry["entries"]
    return {"taxonomy_version": version, "registries": dict(sorted(registries.items()))}


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _payloads(snapshot: DatasetSnapshot) -> tuple[dict[str, bytes], dict[str, int]]:
    active_collections = {
        item.id
        for item in snapshot.collections.values()
        if item.availability.status.value == "active"
    }
    eligible_emojis = {
        item.id: item
        for item in snapshot.emojis.values()
        if item.availability.status.value == "active" and _publishable(item)
    }
    active_memberships = sorted(
        (
            item
            for item in snapshot.memberships.values()
            if item.status.value == "active"
            and item.collection_id in active_collections
            and item.emoji_id in eligible_emojis
        ),
        key=lambda item: item.id,
    )
    collections_for_emoji: dict[str, list[str]] = {}
    for membership in active_memberships:
        collections_for_emoji.setdefault(membership.emoji_id, []).append(membership.collection_id)
    active_emojis = [eligible_emojis[item] for item in sorted(collections_for_emoji)]
    for values in collections_for_emoji.values():
        values.sort()
    approved_relations = _current_approved_relations(snapshot)
    duplicate_groups = _duplicate_groups(snapshot, eligible_emojis, approved_relations)
    group_ids = _group_ids_by_emoji(duplicate_groups)
    public_collections = sorted(
        (snapshot.collections[item] for item in active_collections),
        key=lambda item: item.id,
    )
    collection_facets = _collection_facet_rows(
        public_collections,
        active_memberships,
        eligible_emojis,
        group_ids,
        duplicate_groups,
    )
    taxonomy = _taxonomy_snapshot(
        snapshot.root,
        str(snapshot.manifest["taxonomy_version"]),
    )
    tombstones = [canonical_entity(item) for item in snapshot.tombstones.values()]
    payloads = {
        "collections.jsonl": serialize_jsonl(
            canonical_entity(item) for item in snapshot.collections.values()
        ),
        "emojis.jsonl": serialize_jsonl(
            canonical_entity(item) for item in snapshot.emojis.values()
        ),
        "memberships.jsonl": serialize_jsonl(
            canonical_entity(item) for item in snapshot.memberships.values()
        ),
        "tombstones.jsonl": serialize_jsonl(tombstones, sort_key="target_id"),
        "emojis-active.jsonl": serialize_jsonl(canonical_entity(item) for item in active_emojis),
        "search-ru.jsonl": serialize_jsonl(
            (
                _search_row(
                    item,
                    collections_for_emoji[item.id],
                    group_ids.get(item.id, []),
                    "ru",
                )
                for item in active_emojis
            ),
            sort_key="emoji_id",
        ),
        "search-en.jsonl": serialize_jsonl(
            (
                _search_row(
                    item,
                    collections_for_emoji[item.id],
                    group_ids.get(item.id, []),
                    "en",
                )
                for item in active_emojis
            ),
            sort_key="emoji_id",
        ),
        "collection-facets.jsonl": serialize_jsonl(
            collection_facets,
            sort_key="collection_id",
        ),
        "duplicate-groups.jsonl": serialize_jsonl(duplicate_groups),
        "visual-relations.jsonl": serialize_jsonl(
            (canonical_entity(item) for item in approved_relations),
        ),
        "taxonomy.json": pretty_json(taxonomy).encode("utf-8"),
    }
    counts = {
        "collections": len(snapshot.collections),
        "emojis": len(snapshot.emojis),
        "memberships": len(snapshot.memberships),
        "tombstones": len(snapshot.tombstones),
        "active_emojis": len(active_emojis),
        "active_memberships": len(active_memberships),
        "search_ru": len(active_emojis),
        "search_en": len(active_emojis),
        "collection_facets": len(collection_facets),
        "duplicate_groups": len(duplicate_groups),
        "visual_relations": len(approved_relations),
    }
    return payloads, counts


def _safe_output_directory(dataset_root: Path, output_directory: str | Path | None) -> Path:
    """Resolve an index output without ever targeting canonical dataset content."""

    requested = Path(output_directory) if output_directory is not None else dataset_root / "dist"
    requested = requested.expanduser()
    try:
        assert_no_link_or_reparse(requested)
    except ValueError as exc:
        raise ValueError("index output path must not contain a link or reparse point") from exc
    output = requested.resolve()
    if output == dataset_root or dataset_root.is_relative_to(output):
        raise ValueError("index output must not equal or contain the dataset root")
    for name in _PROTECTED_DATASET_DIRECTORIES:
        protected = (dataset_root / name).resolve()
        if output == protected or output.is_relative_to(protected):
            raise ValueError(f"index output must not be inside canonical {name}/")
    return output


def _validate_existing_output(output: Path) -> dict[PurePosixPath, tuple[int, str]]:
    """Fail closed instead of mixing release payloads with untracked artifacts."""

    if not output.exists():
        return {}
    if not output.is_dir():
        raise ValueError("index output must be a directory")
    children = list(output.iterdir())
    if not children:
        return {}
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("index output is not a recognized MojiLex release directory")
    if manifest_path.stat().st_size > _MAX_EXISTING_MANIFEST_BYTES:
        raise ValueError("index output manifest exceeds the 64 MiB safety limit")
    try:
        manifest = parse_json(manifest_path.read_bytes(), source=str(manifest_path))
    except (OSError, ValueError) as exc:
        raise ValueError("index output is not a recognized MojiLex release directory") from exc
    if (
        manifest.get("dataset") != "mojilex"
        or manifest.get("manifest_version") != "1.0.0"
        or not isinstance(manifest.get("artifacts"), list)
        or not isinstance(manifest.get("resources"), list)
    ):
        raise ValueError("index output is not a recognized MojiLex release directory")
    expected_descriptors: dict[str, tuple[int, str]] = {}

    def add_descriptor(value: Any) -> None:
        if not isinstance(value, dict):
            raise ValueError("index manifest contains an invalid descriptor")
        path = value.get("path")
        size = value.get("object_byte_size")
        digest = value.get("object_sha256")
        if (
            not isinstance(path, str)
            or not path
            or PurePosixPath(path).is_absolute()
            or "\\" in path
            or "%" in path
            or any(part in {"", ".", ".."} for part in PurePosixPath(path).parts)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or path in expected_descriptors
        ):
            raise ValueError("index manifest contains an invalid or duplicate physical path")
        expected_descriptors[path] = (size, digest)

    for descriptor in manifest["artifacts"]:
        add_descriptor(descriptor)
    for resource in manifest["resources"]:
        if not isinstance(resource, dict):
            raise ValueError("index manifest contains an invalid resource descriptor")
        if resource.get("resource_kind") == "physical":
            add_descriptor(resource)
    expected_paths = {"manifest.json", "SHA256SUMS", *expected_descriptors}
    if len(expected_paths) > 10_000:
        raise ValueError("index output path count exceeds its safety limit")
    folded: set[str] = set()
    for expected_path in expected_paths:
        identity = expected_path.casefold() if os.name == "nt" else expected_path
        if identity in folded:
            raise ValueError("index manifest paths alias on this filesystem")
        folded.add(identity)
    actual_files: dict[str, Path] = {}
    actual_directories: set[str] = set()
    total_bytes = 0
    for child in output.rglob("*"):
        if is_link_or_reparse_point(child):
            raise ValueError("index output contains a link or reparse point")
        relative = child.relative_to(output).as_posix()
        if child.is_file():
            actual_files[relative] = child
            try:
                total_bytes += child.stat().st_size
            except OSError as exc:
                raise ValueError("index output could not be inspected safely") from exc
        elif child.is_dir():
            actual_directories.add(relative)
        else:
            raise ValueError("index output contains a special filesystem entry")
        if total_bytes > _MAX_EXISTING_OUTPUT_BYTES:
            raise ValueError("index output exceeds the 1 GiB safety limit")
    expected_directories = {
        parent.as_posix()
        for relative in expected_paths
        for parent in PurePosixPath(relative).parents
        if parent != PurePosixPath(".")
    }
    if set(actual_files) != expected_paths or actual_directories != expected_directories:
        raise ValueError("index output contains unmanaged, missing, or stale entries")
    result: dict[PurePosixPath, tuple[int, str]] = {}
    for relative, file_path in actual_files.items():
        size = file_path.stat().st_size
        digest = _file_sha256(file_path)
        if digest is None:
            raise ValueError("index output could not be inspected safely")
        descriptor = expected_descriptors.get(relative)
        if descriptor is not None and descriptor != (size, digest):
            raise ValueError(f"index output artifact does not match its manifest: {relative}")
        result[PurePosixPath(relative)] = (size, digest)
    expected_sums = "".join(
        f"{digest}  {path}\n"
        for path, (_size, digest) in sorted(
            {
                **expected_descriptors,
                "manifest.json": (
                    manifest_path.stat().st_size,
                    result[PurePosixPath("manifest.json")][1],
                ),
            }.items()
        )
    ).encode("ascii")
    if (output / "SHA256SUMS").read_bytes() != expected_sums:
        raise ValueError("index output SHA256SUMS does not match its manifest")
    return result


def _index_transaction_lock_name(output: Path) -> str:
    identity = os.path.normcase(str(output.resolve())).encode("utf-8")
    return f"index-{hashlib.sha256(identity).hexdigest()}.lock"


def _remove_empty_managed_directories(output: Path, deleted_paths: set[PurePosixPath]) -> None:
    parents = {
        parent
        for relative in deleted_paths
        for parent in relative.parents
        if parent != PurePosixPath(".")
    }
    for relative in sorted(parents, key=lambda item: (-len(item.parts), item.as_posix())):
        directory = output.joinpath(*relative.parts)
        assert_no_link_or_reparse(directory, boundary=output)
        try:
            directory.rmdir()
        except FileNotFoundError:
            continue
        except OSError:
            # A non-empty directory may contain another current resource or a
            # concurrent foreign file. Never recurse or remove its contents.
            continue


def build_index(
    root: str | Path,
    output_directory: str | Path | None = None,
    *,
    snapshot_id: str,
    source_date_epoch: int,
    git_commit: str | None = None,
    tool_commit: str | None = None,
    dependency_lock_sha256: str | None = None,
    validate: bool = True,
) -> IndexBuildResult:
    dataset_root = Path(root).resolve()
    output = _safe_output_directory(dataset_root, output_directory)
    index_lock_name = _index_transaction_lock_name(output)
    recover_pending_dataset_transaction(
        output,
        lock_root=dataset_root,
        lock_name=index_lock_name,
    )
    expected_output_files = _validate_existing_output(output)
    snapshot = load_dataset(dataset_root)
    if validate:
        validate_snapshot(
            snapshot,
            canonical=True,
            schemas=True,
            repository_files=True,
        ).raise_for_errors()
    revision = (git_commit or _git_commit(dataset_root, option="--git-commit")).lower()
    if not _GIT_SHA_RE.fullmatch(revision):
        raise ValueError("git_commit must be a full lowercase Git object ID")
    tool_root = _tool_root()
    resolved_tool_commit = (tool_commit or _git_commit(tool_root, option="--tool-commit")).lower()
    if not _GIT_SHA_RE.fullmatch(resolved_tool_commit):
        raise ValueError("tool_commit must be a full lowercase Git object ID")
    resolved_lock_sha256 = _dependency_contract_sha256(tool_root, dependency_lock_sha256)
    manifest, files = build_distribution(
        dataset_root,
        snapshot=snapshot,
        revision=revision,
        snapshot_id=snapshot_id,
        source_date_epoch=source_date_epoch,
        tool_commit=resolved_tool_commit,
        dependency_lock_sha256=resolved_lock_sha256,
    )
    writer = AtomicDatasetWriter(
        output,
        expected_tree_files=expected_output_files,
        transaction_lock_root=dataset_root,
        transaction_lock_name=index_lock_name,
    )
    generated_paths = {PurePosixPath(path) for path in files}
    deleted_paths = set(expected_output_files) - generated_paths
    for relative in sorted(deleted_paths, key=str):
        writer.stage_delete(relative)
    for output_name, payload in sorted(files.items()):
        writer.stage_bytes(output_name, payload)
    writer.commit()
    _remove_empty_managed_directories(output, deleted_paths)
    file_hashes = {name: _sha(payload) for name, payload in sorted(files.items())}
    counts = dict(manifest["counts"])
    git_metadata = manifest.get("git")
    object_format = git_metadata.get("object_format") if isinstance(git_metadata, dict) else None
    if object_format not in {"sha1", "sha256"}:
        raise ValueError("distribution manifest has an invalid Git object format")
    build_metadata = manifest.get("build")
    manifest_tool_commit = (
        build_metadata.get("tool_commit") if isinstance(build_metadata, dict) else None
    )
    manifest_lock_sha256 = (
        build_metadata.get("dependency_lock_sha256") if isinstance(build_metadata, dict) else None
    )
    if manifest_tool_commit != resolved_tool_commit:
        raise ValueError("distribution manifest tool commit differs from build input")
    if manifest_lock_sha256 != resolved_lock_sha256:
        raise ValueError("distribution manifest dependency contract differs from build input")
    return IndexBuildResult(
        output,
        revision,
        object_format,
        manifest_tool_commit,
        manifest_lock_sha256,
        file_hashes,
        counts,
    )
