"""Bounded, hash-first reader for an explicitly selected local monolith snapshot."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, cast

import rfc8785
from jsonschema import Draft202012Validator, FormatChecker  # type: ignore[import-untyped]
from referencing import Registry, Resource

from mojilex_cli import __version__
from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.dataset.layout import assert_no_link_or_reparse, is_link_or_reparse_point
from mojilex_cli.dataset.serialization import compact_json
from mojilex_cli.schemas import EmbeddedSchema, EmbeddedSchemaError, embedded_schemas

MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_JSONL_LINE_BYTES = 1024 * 1024
MAX_OBJECT_BYTES = 1536 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_STRING_BYTES = 1024 * 1024
MAX_OBJECT_PROPERTIES = 4096
MAX_ARRAY_ITEMS = 1_000_000
MAX_DESCRIPTORS = 1_000_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_RELEASE_MANIFEST_SCHEMA_URI = "mlx://schemas/distribution/v1/release-manifest.schema.json"
_DELEGATED_PROFILE_SCHEMA_URI = "mlx://schemas/distribution/v1/delegated-profile.schema.json"
_ANALYSIS_PROFILE_SCHEMA_URI = "mlx://schemas/distribution/v1/analysis-profile.schema.json"
_PROFILE_WRAPPER_SCHEMA_URIS = {
    _ANALYSIS_PROFILE_SCHEMA_URI,
    _DELEGATED_PROFILE_SCHEMA_URI,
}
_SEMVER_RE = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


def _error(code: str, message: str, hint: str, **details: object) -> CommandError:
    return CommandError(code, message, hint=hint, details=dict(details) or None)


def _read_bounded(path: Path, limit: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise _error(
            "SNAPSHOT_NOT_FOUND",
            f"Snapshot object cannot be inspected: {path}",
            "Pass an existing readable local snapshot directory.",
        ) from exc
    if size > limit:
        raise _error(
            "RESOURCE_LIMIT_EXCEEDED",
            f"Snapshot object exceeds its built-in {limit}-byte limit.",
            "Use a distribution-v1 object within the reader safety limits.",
            path=str(path),
            object_byte_size=size,
            maximum_byte_size=limit,
        )
    try:
        return path.read_bytes()
    except OSError as exc:
        raise _error(
            "SNAPSHOT_NOT_FOUND",
            f"Snapshot object cannot be read: {path}",
            "Check local file permissions and retry.",
        ) from exc


def _preflight_json(data: bytes, *, source: str) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        raise _error(
            "MANIFEST_INVALID",
            f"UTF-8 BOM is forbidden in {source}.",
            "Use exact UTF-8 bytes without BOM.",
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _error(
            "MANIFEST_INVALID",
            f"Invalid UTF-8 in {source}.",
            "Regenerate the snapshot as strict UTF-8.",
        ) from exc
    depth = 0
    in_string = False
    escaped = False
    string_start = 0
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                if len(text[string_start:index].encode("utf-8")) > MAX_STRING_BYTES:
                    raise _error(
                        "RESOURCE_LIMIT_EXCEEDED",
                        f"A JSON string in {source} exceeds 1 MiB.",
                        "Use bounded distribution-v1 records.",
                    )
                in_string = False
            continue
        if char == '"':
            in_string = True
            string_start = index + 1
        elif char in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise _error(
                    "RESOURCE_LIMIT_EXCEEDED",
                    f"JSON nesting in {source} exceeds depth 64.",
                    "Use a bounded distribution-v1 document.",
                )
        elif char in "]}":
            depth -= 1
            if depth < 0:
                break
    if in_string or depth != 0:
        raise _error(
            "MANIFEST_INVALID",
            f"Malformed JSON structure in {source}.",
            "Regenerate the snapshot object.",
        )
    return text


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON property: {key}")
        result[key] = value
    return result


def _walk_limits(value: object, *, source: str, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise _error(
            "RESOURCE_LIMIT_EXCEEDED",
            f"JSON nesting in {source} exceeds depth 64.",
            "Use a bounded distribution-v1 document.",
        )
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_STRING_BYTES:
            raise _error(
                "RESOURCE_LIMIT_EXCEEDED",
                f"A JSON string in {source} exceeds 1 MiB.",
                "Use bounded distribution-v1 records.",
            )
    elif isinstance(value, dict):
        if len(value) > MAX_OBJECT_PROPERTIES:
            raise _error(
                "RESOURCE_LIMIT_EXCEEDED",
                f"A JSON object in {source} exceeds 4096 properties.",
                "Use a bounded distribution-v1 document.",
            )
        for key, item in value.items():
            _walk_limits(key, source=source, depth=depth + 1)
            _walk_limits(item, source=source, depth=depth + 1)
    elif isinstance(value, list):
        if len(value) > MAX_ARRAY_ITEMS:
            raise _error(
                "RESOURCE_LIMIT_EXCEEDED",
                f"A JSON array in {source} exceeds 1000000 items.",
                "Use a bounded distribution-v1 document.",
            )
        for item in value:
            _walk_limits(item, source=source, depth=depth + 1)


def parse_bounded_json(data: bytes, *, source: str) -> dict[str, Any]:
    text = _preflight_json(data, source=source)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid number: {token}")
            ),
        )
    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise _error(
            "MANIFEST_INVALID",
            f"Invalid JSON in {source}: {exc}",
            "Regenerate the snapshot object.",
        ) from exc
    try:
        _walk_limits(value, source=source)
    except UnicodeEncodeError as exc:
        raise _error(
            "MANIFEST_INVALID",
            f"A JSON string in {source} contains an invalid Unicode surrogate.",
            "Regenerate the snapshot object as valid Unicode JSON.",
        ) from exc
    if not isinstance(value, dict):
        raise _error(
            "MANIFEST_INVALID",
            f"Top-level JSON in {source} must be an object.",
            "Regenerate the snapshot object.",
        )
    return cast(dict[str, Any], value)


def _hash_file(path: Path, *, maximum: int) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                total += len(chunk)
                if total > maximum:
                    raise _error(
                        "RESOURCE_LIMIT_EXCEEDED",
                        f"Snapshot object exceeds the built-in {maximum}-byte limit.",
                        "Use a distribution-v1 object within the reader safety limits.",
                        path=str(path),
                    )
                digest.update(chunk)
    except CommandError:
        raise
    except OSError as exc:
        raise _error(
            "SNAPSHOT_NOT_FOUND",
            f"Snapshot object cannot be read: {path}",
            "Check local file permissions and retry.",
        ) from exc
    return total, digest.hexdigest()


def _safe_child(root: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or "\\" in raw_path:
        raise _error(
            "MANIFEST_INVALID",
            "Artifact path must be a safe relative POSIX path.",
            "Regenerate the release manifest.",
        )
    relative = PurePosixPath(raw_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise _error(
            "MANIFEST_INVALID",
            f"Unsafe artifact path: {raw_path!r}",
            "Regenerate the release manifest.",
        )
    candidate = Path(os.path.abspath(root.joinpath(*relative.parts)))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise _error(
            "MANIFEST_INVALID",
            f"Artifact path escapes snapshot root: {raw_path!r}",
            "Regenerate the release manifest.",
        ) from exc
    try:
        assert_no_link_or_reparse(candidate, boundary=root)
    except ValueError as exc:
        raise _error(
            "MANIFEST_INVALID",
            f"Artifact path crosses a link or reparse point: {raw_path!r}",
            "Use a regular-file snapshot tree.",
        ) from exc
    if not candidate.is_file() or is_link_or_reparse_point(candidate):
        raise _error(
            "SNAPSHOT_NOT_FOUND",
            f"Snapshot artifact is missing: {raw_path}",
            "Restore the exact artifact declared by manifest.json.",
        )
    return candidate


def _resource_bindings(descriptor: dict[str, Any]) -> list[dict[str, Any]]:
    raw = descriptor.get("bindings")
    if isinstance(raw, list):
        return [cast(dict[str, Any], item) for item in raw if isinstance(item, dict)]
    binding = descriptor.get("binding")
    return [cast(dict[str, Any], binding)] if isinstance(binding, dict) else []


def _verify_artifact_payload(
    descriptor: dict[str, Any],
    *,
    actual_size: int,
    actual_digest: str,
    logical_name: str,
) -> None:
    for size_key in ("object_byte_size", "uncompressed_byte_size"):
        expected_size = descriptor.get(size_key)
        if isinstance(expected_size, int) and expected_size != actual_size:
            raise _error(
                "CHECKSUM_MISMATCH",
                f"Artifact size changed while reading: {logical_name}",
                "Restore the exact immutable snapshot object.",
                logical_name=logical_name,
            )
    for digest_key in ("object_sha256", "payload_sha256"):
        expected_digest = descriptor.get(digest_key)
        if not isinstance(expected_digest, str) or not hmac.compare_digest(
            expected_digest, actual_digest
        ):
            raise _error(
                "CHECKSUM_MISMATCH",
                f"Artifact bytes changed while reading: {logical_name}",
                "Restore the exact immutable snapshot object.",
                logical_name=logical_name,
            )


@dataclass(slots=True)
class LoadedSnapshot:
    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    descriptors: dict[str, dict[str, Any]]
    verified_paths: dict[str, Path]
    resource_descriptors: tuple[dict[str, Any], ...] = ()
    verified_resource_paths: dict[str, Path] = field(default_factory=dict)
    verified_resource_bytes: dict[str, bytes] = field(default_factory=dict)
    _rows_cache: dict[str, tuple[dict[str, Any], ...]] = field(default_factory=dict)
    _document_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def snapshot_id(self) -> str:
        value = self.manifest.get("snapshot_id")
        if isinstance(value, str) and value:
            return value
        revision = self.manifest.get("git_commit")
        if isinstance(revision, str) and revision:
            return f"git-{revision}"
        raise _error(
            "MANIFEST_INVALID",
            "manifest.json does not identify a snapshot.",
            "Use a distribution-v1 manifest with snapshot_id.",
        )

    @property
    def source_canonical_state_root_sha256(self) -> str:
        value = self.manifest.get("canonical_state_root_sha256")
        if isinstance(value, str) and _SHA256_RE.fullmatch(value):
            return value
        # Legacy monoliths predate semantic state roots. Bind projections to the
        # exact manifest without claiming that it is a canonical state root.
        return self.manifest_sha256

    @property
    def release_verification_status(self) -> str:
        if self.manifest.get("trust_stage") == "pre-enforcement":
            return "integrity-only-unsigned"
        return "unverified"

    def dataset_context(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "manifest_sha256": self.manifest_sha256,
            "release_verification_status": self.release_verification_status,
            "catalog_status": "unknown",
            "revocation_status": "unknown",
            "control_state_status": "offline-unknown",
            "errata_status": "unknown",
            "catalog_checkpoint": {"present": False},
        }

    def pinned_state(self) -> dict[str, str]:
        return {"snapshot_id": self.snapshot_id, "manifest_sha256": self.manifest_sha256}

    def require_diagnostic_opt_in(self, allow_unverified: bool) -> list[dict[str, Any]]:
        if not allow_unverified:
            raise _error(
                "SIGNATURE_MISSING",
                "The local snapshot has integrity checks but no enforceable release signature.",
                (
                    "Rerun with --allow-unverified only for diagnostic use "
                    "of this exact local snapshot."
                ),
                release_verification_status=self.release_verification_status,
            )
        return [
            {
                "code": "INTEGRITY_ONLY_UNSIGNED",
                "message": (
                    "Diagnostic read from an integrity-checked unsigned snapshot; "
                    "safe_eligible is always false."
                ),
            }
        ]

    def has(self, logical_name: str) -> bool:
        return logical_name in self.descriptors

    def descriptor(self, logical_name: str) -> dict[str, Any]:
        try:
            return self.descriptors[logical_name]
        except KeyError as exc:
            raise _error(
                "INDEX_CORRUPT",
                f"Required artifact is absent: {logical_name}",
                "Rebuild or restore the complete snapshot.",
            ) from exc

    def rows(self, logical_name: str) -> tuple[dict[str, Any], ...]:
        if logical_name in self._rows_cache:
            return self._rows_cache[logical_name]
        descriptor = self.descriptor(logical_name)
        path = self.verified_paths[logical_name]
        rows: list[dict[str, Any]] = []
        digest = hashlib.sha256()
        total_size = 0
        try:
            with path.open("rb") as stream:
                line_number = 0
                while True:
                    line = stream.readline(MAX_JSONL_LINE_BYTES + 1)
                    if not line:
                        break
                    total_size += len(line)
                    if total_size > MAX_OBJECT_BYTES:
                        raise _error(
                            "RESOURCE_LIMIT_EXCEEDED",
                            f"Artifact {logical_name} exceeds the 1.5 GiB object limit.",
                            "Use a supported monolith snapshot.",
                        )
                    digest.update(line)
                    line_number += 1
                    if len(line) > MAX_JSONL_LINE_BYTES:
                        raise _error(
                            "RESOURCE_LIMIT_EXCEEDED",
                            f"JSONL line {line_number} in {logical_name} exceeds 1 MiB.",
                            "Rebuild the snapshot with bounded records.",
                        )
                    if (
                        not line.endswith(b"\n")
                        or line in {b"\n", b"\r\n"}
                        or line.endswith(b"\r\n")
                    ):
                        raise _error(
                            "INDEX_CORRUPT",
                            f"JSONL framing is invalid at {logical_name}:{line_number}.",
                            "Restore the exact LF-delimited artifact.",
                        )
                    row = parse_bounded_json(
                        line[:-1], source=f"{descriptor['path']}:{line_number}"
                    )
                    if compact_json(row).encode("utf-8") != line[:-1]:
                        raise _error(
                            "INDEX_CORRUPT",
                            (
                                "JSONL row does not use the canonical compact formatter "
                                f"at {logical_name}:{line_number}."
                            ),
                            "Restore the canonical LF-delimited artifact.",
                        )
                    rows.append(row)
                    if len(rows) > MAX_ARRAY_ITEMS:
                        raise _error(
                            "RESOURCE_LIMIT_EXCEEDED",
                            f"Artifact {logical_name} exceeds 1000000 rows.",
                            "Use a scale layout reader for a larger snapshot.",
                        )
        except CommandError:
            raise
        except OSError as exc:
            raise _error(
                "SNAPSHOT_NOT_FOUND",
                f"Snapshot artifact cannot be read: {path}",
                "Check local file permissions and retry.",
            ) from exc
        _verify_artifact_payload(
            descriptor,
            actual_size=total_size,
            actual_digest=digest.hexdigest(),
            logical_name=logical_name,
        )
        declared = descriptor.get("logical_record_count", descriptor.get("record_count"))
        if isinstance(declared, int) and declared != len(rows):
            raise _error(
                "INDEX_CORRUPT",
                f"Artifact {logical_name} record count does not match its descriptor.",
                "Restore the exact snapshot objects.",
            )
        result = tuple(rows)
        self._rows_cache[logical_name] = result
        return result

    def document(self, logical_name: str) -> dict[str, Any]:
        if logical_name in self._document_cache:
            return self._document_cache[logical_name]
        descriptor = self.descriptor(logical_name)
        path = self.verified_paths[logical_name]
        payload = _read_bounded(path, MAX_JSON_BYTES)
        _verify_artifact_payload(
            descriptor,
            actual_size=len(payload),
            actual_digest=hashlib.sha256(payload).hexdigest(),
            logical_name=logical_name,
        )
        document = parse_bounded_json(payload, source=str(descriptor["path"]))
        if rfc8785.dumps(document) != payload:
            raise _error(
                "INDEX_CORRUPT",
                f"Singleton artifact {logical_name} is not exact JCS.",
                "Restore the canonical singleton artifact.",
            )
        self._document_cache[logical_name] = document
        return document

    def bound_document(self, container_name: str, field_name: str) -> dict[str, Any]:
        """Resolve one exact-content profile/policy without any network fallback."""

        container = self.manifest.get(container_name)
        selector = container.get(field_name) if isinstance(container, dict) else None
        if not isinstance(selector, dict):
            raise _error(
                "SCHEMA_UNSUPPORTED",
                f"Manifest selector {container_name}.{field_name} is unavailable.",
                "Use a complete distribution-v1 snapshot.",
            )
        expected = selector.get("sha256")
        pointer = f"/{container_name}/{field_name}/sha256"
        matches: list[dict[str, Any]] = []
        for descriptor in self.resource_descriptors:
            bindings = _resource_bindings(descriptor)
            if any(
                isinstance(binding, dict)
                and binding.get("kind") == "exact-content"
                and binding.get("manifest_pointer") == pointer
                for binding in bindings
            ):
                matches.append(descriptor)
        if len(matches) != 1:
            raise _error(
                "SCHEMA_UNSUPPORTED",
                f"Selector {pointer} does not resolve to exactly one local resource.",
                "Use a self-contained distribution-v1 snapshot.",
            )
        descriptor = matches[0]
        if not isinstance(expected, str) or descriptor.get("payload_sha256") != expected:
            raise _error(
                "MANIFEST_INVALID",
                f"Resource binding for {pointer} has a digest mismatch.",
                "Restore the exact manifest and resources.",
            )
        if descriptor.get("resource_kind") == "artifact-alias":
            document = self.document(str(descriptor.get("artifact_logical_name")))
        else:
            uri = descriptor.get("uri")
            data = self.verified_resource_bytes.get(str(uri))
            if data is None:
                raise _error(
                    "SCHEMA_UNSUPPORTED",
                    f"Physical resource for {pointer} is unavailable.",
                    "Use a self-contained distribution-v1 snapshot.",
                )
            document = parse_bounded_json(data, source=str(descriptor.get("path")))
            if rfc8785.dumps(document) != data:
                raise _error(
                    "MANIFEST_INVALID",
                    f"Exact JSON resource for {pointer} is not JCS encoded.",
                    "Restore the exact canonical resource bytes.",
                )
            if _delegated_contract_ref(document) is not None:
                body = document.get("body")
                profile_id = document.get("profile_id")
                if not isinstance(body, dict) or not isinstance(profile_id, str):
                    raise _error(
                        "SCHEMA_UNSUPPORTED",
                        f"Delegated profile for {pointer} is incomplete.",
                        "Use a snapshot containing the exact delegated profile wrapper.",
                    )
                effective = dict(body)
                effective["profile_id"] = profile_id
                document = effective
        return document


def load_snapshot(
    selected: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> LoadedSnapshot:
    """Load and integrity-check an explicit local monolith snapshot, without network I/O."""

    text = str(selected)
    if "://" in text:
        raise _error(
            "SNAPSHOT_NOT_FOUND",
            "Snapshot selection must be an explicit local path.",
            "Download separately, then pass the local directory with --snapshot.",
        )
    selected_path = Path(os.path.abspath(Path(selected).expanduser()))
    manifest_path = selected_path / "manifest.json" if selected_path.is_dir() else selected_path
    root = manifest_path.parent
    if manifest_path.name != "manifest.json" or not manifest_path.is_file():
        raise _error(
            "SNAPSHOT_NOT_FOUND",
            f"Local snapshot manifest was not found: {manifest_path}",
            "Pass a directory containing manifest.json or that exact file.",
        )
    try:
        assert_no_link_or_reparse(root)
        assert_no_link_or_reparse(manifest_path, boundary=root)
    except ValueError as exc:
        raise _error(
            "MANIFEST_INVALID",
            "Snapshot path crosses a link or reparse point.",
            "Use an explicit regular-file snapshot directory.",
        ) from exc
    manifest_bytes = _read_bounded(manifest_path, MAX_MANIFEST_BYTES)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_manifest_sha256 is not None:
        if not _SHA256_RE.fullmatch(expected_manifest_sha256):
            raise _error(
                "QUERY_INVALID",
                "--manifest-sha256 must be lowercase SHA-256 hex.",
                "Pass the exact 64-character manifest digest.",
            )
        if not hmac.compare_digest(expected_manifest_sha256, manifest_sha256):
            raise _error(
                "CHECKSUM_MISMATCH",
                "manifest.json does not match --manifest-sha256.",
                "Select the exact pinned snapshot pair.",
            )
    manifest = parse_bounded_json(manifest_bytes, source="manifest.json")
    if manifest.get("dataset") != "mojilex":
        raise _error(
            "MANIFEST_INVALID",
            "manifest.json is not a MojiLex release manifest.",
            "Pass a MojiLex snapshot directory.",
        )
    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, str) or schema_version.split(".", 1)[0] != "1":
        raise _error(
            "SCHEMA_UNSUPPORTED",
            "The snapshot schema major version is unsupported.",
            "Use a 1.x distribution snapshot.",
        )
    _validate_manifest_against_embedded_schema(manifest)
    _validate_minimum_reader_version(manifest)
    artifacts = manifest.get("artifacts")
    descriptors: list[dict[str, Any]] = []
    if isinstance(artifacts, list):
        if manifest.get("layout_profile") != "monolith-v1":
            raise _error(
                "SCHEMA_UNSUPPORTED",
                "Only layout_profile=monolith-v1 is supported by this reader.",
                "Use a monolith-v1 snapshot.",
            )
        try:
            if rfc8785.dumps(manifest) != manifest_bytes:
                raise _error(
                    "MANIFEST_INVALID",
                    "distribution-v1 manifest bytes are not exact JCS.",
                    "Use the immutable canonical manifest bytes.",
                )
        except (TypeError, ValueError) as exc:
            raise _error(
                "MANIFEST_INVALID",
                "manifest.json cannot be serialized as JCS.",
                "Regenerate the release manifest.",
            ) from exc
        for item in artifacts:
            if not isinstance(item, dict):
                raise _error(
                    "MANIFEST_INVALID",
                    "Every artifact descriptor must be an object.",
                    "Regenerate the release manifest.",
                )
            descriptors.append(cast(dict[str, Any], item))
    else:
        raise _error(
            "SCHEMA_UNSUPPORTED",
            "The selected manifest is not a distribution-v1 artifact manifest.",
            "Build and select a complete monolith-v1 snapshot.",
        )
    if len(descriptors) > MAX_DESCRIPTORS:
        raise _error(
            "RESOURCE_LIMIT_EXCEEDED",
            "Manifest declares more than 1000000 descriptors.",
            "Use a bounded distribution-v1 manifest.",
        )
    by_name: dict[str, dict[str, Any]] = {}
    paths_seen: set[str] = set()
    folded_paths_seen: set[str] = set()
    verified_paths: dict[str, Path] = {}
    for descriptor in descriptors:
        logical_name = descriptor.get("logical_name")
        raw_path = descriptor.get("path")
        if not isinstance(logical_name, str) or not logical_name or logical_name in by_name:
            raise _error(
                "MANIFEST_INVALID",
                "Artifact logical_name values must be non-empty and unique.",
                "Regenerate the release manifest.",
            )
        if (
            not isinstance(raw_path, str)
            or raw_path in paths_seen
            or raw_path.casefold() in folded_paths_seen
        ):
            raise _error(
                "MANIFEST_INVALID",
                "Artifact paths must be strings and unique.",
                "Regenerate the release manifest.",
            )
        if descriptor.get("compression", "none") not in {"none", "identity"}:
            raise _error(
                "SCHEMA_UNSUPPORTED",
                f"Compressed artifact {logical_name} is unsupported in local monolith MVP.",
                "Use the uncompressed monolith-v1 layout.",
            )
        artifact_path = _safe_child(root, raw_path)
        declared_size = descriptor.get("object_byte_size")
        if isinstance(declared_size, bool) or (
            declared_size is not None and not isinstance(declared_size, int)
        ):
            raise _error(
                "MANIFEST_INVALID",
                f"Artifact {logical_name} has an invalid object_byte_size.",
                "Regenerate the release manifest.",
            )
        if isinstance(declared_size, int) and (
            declared_size < 0 or declared_size > MAX_OBJECT_BYTES
        ):
            raise _error(
                "RESOURCE_LIMIT_EXCEEDED",
                f"Artifact {logical_name} exceeds the 1.5 GiB object limit.",
                "Use a supported monolith snapshot.",
            )
        actual_size, actual_digest = _hash_file(artifact_path, maximum=MAX_OBJECT_BYTES)
        if isinstance(declared_size, int) and actual_size != declared_size:
            raise _error(
                "CHECKSUM_MISMATCH",
                f"Artifact size mismatch: {logical_name}",
                "Restore the exact object declared by manifest.json.",
            )
        uncompressed_size = descriptor.get("uncompressed_byte_size")
        if isinstance(uncompressed_size, int) and uncompressed_size != actual_size:
            raise _error(
                "CHECKSUM_MISMATCH",
                f"Artifact payload size mismatch: {logical_name}",
                "Restore the exact object declared by manifest.json.",
            )
        for digest_key in ("object_sha256", "payload_sha256"):
            expected = descriptor.get(digest_key)
            if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
                raise _error(
                    "MANIFEST_INVALID",
                    f"Artifact {logical_name} lacks valid {digest_key}.",
                    "Regenerate the release manifest.",
                )
            if not hmac.compare_digest(expected, actual_digest):
                raise _error(
                    "CHECKSUM_MISMATCH",
                    f"Artifact digest mismatch: {logical_name}",
                    "Restore the exact object declared by manifest.json.",
                )
        by_name[logical_name] = descriptor
        paths_seen.add(raw_path)
        folded_paths_seen.add(raw_path.casefold())
        verified_paths[logical_name] = artifact_path
    resource_descriptors: list[dict[str, Any]] = []
    verified_resource_paths: dict[str, Path] = {}
    verified_resource_bytes: dict[str, bytes] = {}
    resources = manifest.get("resources", [])
    if isinstance(artifacts, list) and not isinstance(resources, list):
        raise _error(
            "MANIFEST_INVALID",
            "Manifest resources must be an array.",
            "Regenerate the release manifest.",
        )
    if isinstance(resources, list):
        seen_uris: set[str] = set()
        for raw_resource in resources:
            if not isinstance(raw_resource, dict):
                raise _error(
                    "MANIFEST_INVALID",
                    "Every resource descriptor must be an object.",
                    "Regenerate the release manifest.",
                )
            resource = cast(dict[str, Any], raw_resource)
            uri = resource.get("uri")
            kind = resource.get("resource_kind")
            if (
                not isinstance(uri, str)
                or not (
                    uri.startswith("mlx://") or (kind == "physical" and uri.startswith("https://"))
                )
                or uri in seen_uris
            ):
                raise _error(
                    "MANIFEST_INVALID",
                    "Resource URIs must be unique permitted absolute identifiers.",
                    "Regenerate the release manifest.",
                )
            if kind == "artifact-alias":
                logical = resource.get("artifact_logical_name")
                target = by_name.get(str(logical))
                if (
                    target is None
                    or target.get("content_model") != "singleton-json"
                    or resource.get("payload_sha256") != target.get("payload_sha256")
                ):
                    raise _error(
                        "MANIFEST_INVALID",
                        f"Resource alias {uri} does not bind one singleton artifact.",
                        "Regenerate the release manifest.",
                    )
            elif kind == "physical":
                raw_resource_path = resource.get("path")
                if (
                    not isinstance(raw_resource_path, str)
                    or raw_resource_path in paths_seen
                    or raw_resource_path.casefold() in folded_paths_seen
                ):
                    raise _error(
                        "MANIFEST_INVALID",
                        "Artifact/resource paths collide or are not unique.",
                        "Regenerate the release manifest.",
                    )
                if resource.get("compression") != "none":
                    raise _error(
                        "SCHEMA_UNSUPPORTED",
                        f"Compressed resource {uri} is unsupported in local monolith MVP.",
                        "Use an uncompressed monolith-v1 snapshot.",
                    )
                resource_path = _safe_child(root, raw_resource_path)
                declared_size = resource.get("object_byte_size")
                if (
                    isinstance(declared_size, bool)
                    or not isinstance(declared_size, int)
                    or not 0 <= declared_size <= MAX_JSON_BYTES
                ):
                    raise _error(
                        "RESOURCE_LIMIT_EXCEEDED",
                        f"Resource {uri} has an invalid or oversized object_byte_size.",
                        "Use a bounded distribution-v1 resource.",
                    )
                resource_payload = _read_bounded(resource_path, MAX_JSON_BYTES)
                actual_size = len(resource_payload)
                actual_digest = hashlib.sha256(resource_payload).hexdigest()
                if (
                    actual_size != declared_size
                    or resource.get("uncompressed_byte_size") != actual_size
                ):
                    raise _error(
                        "CHECKSUM_MISMATCH",
                        f"Resource size mismatch: {uri}",
                        "Restore the exact resource declared by manifest.json.",
                    )
                for digest_key in ("object_sha256", "payload_sha256"):
                    expected = resource.get(digest_key)
                    if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
                        raise _error(
                            "MANIFEST_INVALID",
                            f"Resource {uri} lacks valid {digest_key}.",
                            "Regenerate the release manifest.",
                        )
                    if not hmac.compare_digest(expected, actual_digest):
                        raise _error(
                            "CHECKSUM_MISMATCH",
                            f"Resource digest mismatch: {uri}",
                            "Restore the exact resource declared by manifest.json.",
                        )
                media_type = resource.get("media_type")
                if isinstance(media_type, str) and (
                    media_type == "application/json" or media_type.endswith("+json")
                ):
                    resource_document = parse_bounded_json(
                        resource_payload, source=raw_resource_path
                    )
                    if (
                        any(
                            binding.get("kind") == "exact-content"
                            for binding in _resource_bindings(resource)
                        )
                        and rfc8785.dumps(resource_document) != resource_payload
                    ):
                        raise _error(
                            "MANIFEST_INVALID",
                            f"Exact-content resource is not JCS encoded: {uri}",
                            "Restore the exact canonical resource bytes.",
                        )
                paths_seen.add(raw_resource_path)
                folded_paths_seen.add(raw_resource_path.casefold())
                verified_resource_paths[uri] = resource_path
                verified_resource_bytes[uri] = resource_payload
            else:
                raise _error(
                    "MANIFEST_INVALID",
                    f"Unknown resource_kind for {uri}.",
                    "Regenerate the release manifest.",
                )
            seen_uris.add(uri)
            resource_descriptors.append(resource)
    snapshot = LoadedSnapshot(
        root=root,
        manifest_path=manifest_path,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        descriptors=by_name,
        verified_paths=verified_paths,
        resource_descriptors=tuple(resource_descriptors),
        verified_resource_paths=verified_resource_paths,
        verified_resource_bytes=verified_resource_bytes,
    )
    _validate_release_invariants(snapshot)
    return snapshot


def _pointer_component(record: dict[str, Any], pointer: object) -> dict[str, Any]:
    if not isinstance(pointer, str) or not pointer.startswith("/") or pointer == "/":
        raise _error(
            "MANIFEST_INVALID",
            "Primary/sort keys must contain canonical JSON Pointers.",
            "Regenerate the release manifest.",
        )
    current: object = record
    present = True
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            present = False
            break
    if present:
        if isinstance(current, (dict, list)):
            raise _error(
                "MANIFEST_INVALID",
                f"Sort-key pointer {pointer} resolves to a container.",
                "Use scalar key components.",
            )
        return {"pointer": pointer, "present": True, "value": current}
    return {"pointer": pointer, "present": False}


def _key_envelope(record: dict[str, Any], pointers: list[object]) -> list[dict[str, Any]]:
    return [_pointer_component(record, pointer) for pointer in pointers]


def _validate_recordset(snapshot: LoadedSnapshot, logical: str, descriptor: dict[str, Any]) -> None:
    required = {
        "record_count",
        "logical_record_count",
        "primary_key",
        "sort_key",
        "table_root_sha256",
    }
    if not required.issubset(descriptor):
        raise _error(
            "MANIFEST_INVALID",
            f"Recordset descriptor {logical} is incomplete.",
            "Regenerate the release manifest.",
        )
    primary_key = descriptor.get("primary_key")
    sort_key = descriptor.get("sort_key")
    if (
        not isinstance(primary_key, list)
        or not primary_key
        or not isinstance(sort_key, list)
        or not sort_key
        or descriptor.get("record_count") != descriptor.get("logical_record_count")
    ):
        raise _error(
            "MANIFEST_INVALID",
            f"Monolith recordset keys/counts are invalid: {logical}",
            "Regenerate the release manifest.",
        )
    rows = snapshot.rows(logical)
    primary_lines: list[tuple[bytes, bytes]] = []
    effective_sort = [*sort_key, *(pointer for pointer in primary_key if pointer not in sort_key)]
    prior_sort: bytes | None = None
    seen_primary: set[bytes] = set()
    for row in rows:
        primary = _key_envelope(row, primary_key)
        primary_bytes = rfc8785.dumps(primary)
        if primary_bytes in seen_primary:
            raise _error(
                "INDEX_CORRUPT",
                f"Duplicate primary key in {logical}.",
                "Restore or rebuild the snapshot.",
            )
        seen_primary.add(primary_bytes)
        sort_bytes = rfc8785.dumps(_key_envelope(row, effective_sort))
        if prior_sort is not None and sort_bytes <= prior_sort:
            raise _error(
                "INDEX_CORRUPT",
                f"Rows in {logical} are not in canonical sort order.",
                "Restore or rebuild the snapshot.",
            )
        prior_sort = sort_bytes
        record_sha256 = hashlib.sha256(rfc8785.dumps(row)).hexdigest()
        primary_lines.append((primary_bytes, rfc8785.dumps([primary, record_sha256]) + b"\n"))
    primary_lines.sort(key=lambda item: item[0])
    table_root = hashlib.sha256(b"".join(line for _, line in primary_lines)).hexdigest()
    if table_root != descriptor.get("table_root_sha256"):
        raise _error(
            "INDEX_CORRUPT",
            f"Logical table root mismatch: {logical}",
            "Restore or rebuild the exact snapshot.",
        )


def _state_entries(manifest: dict[str, Any], roles: set[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for descriptor in cast(list[dict[str, Any]], manifest["artifacts"]):
        role = descriptor.get("semantic_role")
        if role not in roles:
            continue
        content_model = descriptor.get("content_model")
        digest_kind = "table-root" if content_model == "recordset-jsonl" else "payload"
        digest = descriptor.get(
            "table_root_sha256" if digest_kind == "table-root" else "payload_sha256"
        )
        entries.append(
            {
                "entry_type": "artifact",
                "entry_name": descriptor["logical_name"],
                "semantic_role": role,
                "digest_kind": digest_kind,
                "sha256": digest,
            }
        )
    for container_name, prefix in (("profiles", "profile"), ("policies", "policy")):
        container = manifest.get(container_name)
        if not isinstance(container, dict):
            continue
        for field_name, selector in container.items():
            if not isinstance(selector, dict):
                continue
            root_scope = selector.get("root_scope")
            semantic_role = (
                "normative"
                if root_scope == "canonical"
                else "derived"
                if root_scope == "derived"
                else None
            )
            if semantic_role not in roles:
                continue
            if prefix == "profile":
                selector_id = selector.get("id")
                if not isinstance(selector_id, str):
                    raise _error(
                        "MANIFEST_INVALID",
                        f"Profile selector {field_name} lacks id.",
                        "Regenerate the release manifest.",
                    )
                entry_name = f"profile:{field_name}:{selector_id}"
            else:
                entry_name = f"policy:{field_name}"
            digest = selector.get("sha256")
            if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
                raise _error(
                    "MANIFEST_INVALID",
                    f"Selector {container_name}.{field_name} lacks a valid hash.",
                    "Regenerate the release manifest.",
                )
            entries.append(
                {
                    "entry_type": "input",
                    "entry_name": entry_name,
                    "semantic_role": semantic_role,
                    "digest_kind": "input",
                    "sha256": digest,
                }
            )
    entries.sort(
        key=lambda entry: rfc8785.dumps(
            [entry["semantic_role"], entry["entry_type"], entry["entry_name"]]
        )
    )
    return entries


def _manifest_pointer(document: dict[str, Any], pointer: object) -> tuple[bool, object]:
    if not isinstance(pointer, str) or not pointer.startswith("/") or pointer == "/":
        return False, None
    current: object = document
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _external_schema_dependencies(value: object) -> set[str]:
    dependencies: set[str] = set()
    if isinstance(value, dict):
        reference = value.get("$ref")
        if isinstance(reference, str) and not reference.startswith("#"):
            target = reference.split("#", 1)[0]
            if target:
                dependencies.add(target)
        for item in value.values():
            dependencies.update(_external_schema_dependencies(item))
    elif isinstance(value, list):
        for item in value:
            dependencies.update(_external_schema_dependencies(item))
    return dependencies


def _delegated_contract_ref(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    reference = value.get("contract_schema_ref")
    return reference if isinstance(reference, str) else None


def _embedded_schema_contracts() -> tuple[
    dict[str, dict[str, Any]], Registry[Any], Mapping[str, EmbeddedSchema]
]:
    try:
        embedded = embedded_schemas()
    except EmbeddedSchemaError as exc:
        raise _error(
            "SCHEMA_UNSUPPORTED",
            "The installed CLI schema trust root is unavailable.",
            "Reinstall this exact mojilex-cli release.",
        ) from exc
    return _compiled_embedded_schema_contracts(tuple(embedded.items()))


@lru_cache(maxsize=2)
def _compiled_embedded_schema_contracts(
    entries: tuple[tuple[str, EmbeddedSchema], ...],
) -> tuple[dict[str, dict[str, Any]], Registry[Any], Mapping[str, EmbeddedSchema]]:
    # Keyed by exact immutable schema bytes, including test/update trust-root changes.
    embedded = dict(entries)
    schemas: dict[str, dict[str, Any]] = {}
    registry: Registry[Any] = Registry()
    for uri, item in embedded.items():
        try:
            schema = parse_bounded_json(
                item.payload,
                source=f"embedded:{item.relative_path}",
            )
        except CommandError as exc:
            raise _error(
                "SCHEMA_UNSUPPORTED",
                f"The installed CLI schema is invalid: {uri}",
                "Reinstall this exact mojilex-cli release.",
            ) from exc
        if schema.get("$id") != uri:
            raise _error(
                "SCHEMA_UNSUPPORTED",
                f"The installed CLI schema identity is inconsistent: {uri}",
                "Reinstall this exact mojilex-cli release.",
            )
        try:
            Draft202012Validator.check_schema(schema)
            registry = registry.with_resource(uri, Resource.from_contents(schema))
        except Exception as exc:
            raise _error(
                "SCHEMA_UNSUPPORTED",
                f"The installed CLI schema is not valid Draft 2020-12: {uri}",
                "Reinstall this exact mojilex-cli release.",
            ) from exc
        schemas[uri] = schema
    for uri, schema in schemas.items():
        unknown = _external_schema_dependencies(schema).difference(schemas)
        if unknown:
            raise _error(
                "SCHEMA_UNSUPPORTED",
                f"The installed CLI schema graph is not closed at {uri}.",
                "Reinstall this exact mojilex-cli release.",
                unknown_schema_refs=sorted(unknown),
            )
    return schemas, registry, embedded


def _required_schema_uris(
    snapshot: LoadedSnapshot,
    schemas: Mapping[str, dict[str, Any]],
) -> set[str]:
    pending: list[str] = [_RELEASE_MANIFEST_SCHEMA_URI]
    pending.extend(
        str(descriptor.get("schema_ref"))
        for descriptor in snapshot.descriptors.values()
        if isinstance(descriptor.get("schema_ref"), str)
    )
    pending.extend(
        str(resource.get("content_schema_ref"))
        for resource in snapshot.resource_descriptors
        if isinstance(resource.get("content_schema_ref"), str)
    )
    for resource in snapshot.resource_descriptors:
        if resource.get("content_schema_ref") not in _PROFILE_WRAPPER_SCHEMA_URIS:
            continue
        payload = snapshot.verified_resource_bytes.get(str(resource.get("uri")))
        if payload is None:
            continue
        wrapper = parse_bounded_json(payload, source=str(resource.get("path")))
        contract_ref = _delegated_contract_ref(wrapper)
        if contract_ref is not None:
            pending.append(contract_ref)
    required: set[str] = set()
    while pending:
        reference = pending.pop()
        uri = reference.split("#", 1)[0]
        if not uri or uri in required:
            continue
        schema = schemas.get(uri)
        if schema is None:
            raise _error(
                "SCHEMA_UNSUPPORTED",
                f"The snapshot uses a schema unknown to this CLI: {uri}",
                "Upgrade mojilex-cli or select a compatible snapshot.",
                schema_uri=uri,
            )
        required.add(uri)
        pending.extend(_external_schema_dependencies(schema))
    return required


def _require_exact_published_schemas(
    snapshot: LoadedSnapshot,
    required: set[str],
    embedded: Mapping[str, EmbeddedSchema],
) -> None:
    published = {
        str(resource.get("uri")): resource
        for resource in snapshot.resource_descriptors
        if resource.get("resource_kind") == "physical"
        and resource.get("media_type") == "application/schema+json"
    }
    for uri in sorted(required):
        descriptor = published.get(uri)
        expected = embedded[uri]
        payload = snapshot.verified_resource_bytes.get(uri)
        if descriptor is None or payload is None:
            raise _error(
                "SCHEMA_UNSUPPORTED",
                f"The snapshot does not publish a required schema: {uri}",
                "Use a self-contained snapshot with the exact supported schema resources.",
                schema_uri=uri,
            )
        if (
            descriptor.get("payload_sha256") != expected.sha256
            or descriptor.get("object_sha256") != expected.sha256
            or not hmac.compare_digest(payload, expected.payload)
        ):
            raise _error(
                "SCHEMA_UNSUPPORTED",
                f"Snapshot schema bytes do not match this CLI's trust root: {uri}",
                "Upgrade mojilex-cli or select the exact compatible snapshot.",
                schema_uri=uri,
                expected_sha256=expected.sha256,
                snapshot_sha256=descriptor.get("payload_sha256"),
            )


def _schema_contracts(
    snapshot: LoadedSnapshot,
) -> tuple[dict[str, dict[str, Any]], Registry[Any]]:
    schemas, registry, embedded = _embedded_schema_contracts()
    required = _required_schema_uris(snapshot, schemas)
    _require_exact_published_schemas(snapshot, required, embedded)
    return schemas, registry


def _validate_manifest_against_embedded_schema(manifest: dict[str, Any]) -> None:
    validate_embedded_schema_instance(
        manifest,
        _RELEASE_MANIFEST_SCHEMA_URI,
        location="manifest.json",
        invalid_code="MANIFEST_INVALID",
    )


def validate_embedded_schema_instance(
    value: object,
    schema_uri: str,
    *,
    location: str,
    invalid_code: str,
) -> None:
    """Validate against only the immutable schema graph bundled with the CLI."""

    schemas, registry, _ = _embedded_schema_contracts()
    _validate_schema_instance(
        value,
        schema_uri,
        schemas=schemas,
        registry=registry,
        location=location,
        invalid_code=invalid_code,
    )


def _parse_semver(value: object) -> tuple[int, int, int, tuple[str, ...] | None]:
    if not isinstance(value, str):
        raise ValueError("semantic version must be a string")
    match = _SEMVER_RE.fullmatch(value)
    if match is None:
        raise ValueError("invalid semantic version")
    prerelease = match.group(4)
    identifiers = tuple(prerelease.split(".")) if prerelease is not None else None
    if identifiers is not None and any(
        identifier.isdigit() and len(identifier) > 1 and identifier.startswith("0")
        for identifier in identifiers
    ):
        raise ValueError("numeric prerelease identifiers cannot have leading zeroes")
    return int(match.group(1)), int(match.group(2)), int(match.group(3)), identifiers


def _prerelease_compare(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    for left_item, right_item in zip(left, right, strict=False):
        if left_item == right_item:
            continue
        left_numeric = left_item.isdigit()
        right_numeric = right_item.isdigit()
        if left_numeric and right_numeric:
            return -1 if int(left_item) < int(right_item) else 1
        if left_numeric != right_numeric:
            return -1 if left_numeric else 1
        return -1 if left_item < right_item else 1
    if len(left) == len(right):
        return 0
    return -1 if len(left) < len(right) else 1


def _semver_less(left: object, right: object) -> bool:
    left_major, left_minor, left_patch, left_pre = _parse_semver(left)
    right_major, right_minor, right_patch, right_pre = _parse_semver(right)
    left_core = left_major, left_minor, left_patch
    right_core = right_major, right_minor, right_patch
    if left_core != right_core:
        return left_core < right_core
    if left_pre is None:
        return False
    if right_pre is None:
        return True
    return _prerelease_compare(left_pre, right_pre) < 0


def _validate_minimum_reader_version(manifest: dict[str, Any]) -> None:
    minimum = manifest.get("minimum_reader_version")
    try:
        too_old = _semver_less(__version__, minimum)
    except ValueError as exc:
        raise _error(
            "MANIFEST_INVALID",
            "minimum_reader_version is not valid SemVer.",
            "Regenerate the release manifest with a valid reader version gate.",
        ) from exc
    if too_old:
        raise _error(
            "SCHEMA_UNSUPPORTED",
            f"Snapshot requires mojilex-cli {minimum} or newer; installed is {__version__}.",
            "Upgrade mojilex-cli before reading this snapshot.",
            minimum_reader_version=minimum,
            reader_version=__version__,
        )


def _validate_schema_instance(
    value: object,
    schema_ref: object,
    *,
    schemas: dict[str, dict[str, Any]],
    registry: Registry[Any],
    location: str,
    invalid_code: str,
) -> None:
    if not isinstance(schema_ref, str) or schema_ref not in schemas:
        raise _error(
            "SCHEMA_UNSUPPORTED",
            f"Schema reference is unavailable offline for {location}: {schema_ref!r}",
            "Use a snapshot containing every exact schema resource.",
        )
    try:
        validator = Draft202012Validator(
            schemas[schema_ref], registry=registry, format_checker=FormatChecker()
        )
        errors = sorted(
            validator.iter_errors(value),
            key=lambda error: tuple(str(item) for item in error.absolute_path),
        )
    except Exception as exc:
        raise _error(
            "SCHEMA_UNSUPPORTED",
            f"Offline schema resolution failed for {location}.",
            "Restore a closed, self-contained schema graph.",
        ) from exc
    if errors:
        first = errors[0]
        path = "/".join(str(item) for item in first.absolute_path) or "<root>"
        raise _error(
            invalid_code,
            f"Schema validation failed for {location} at {path}: {first.message}",
            "Restore or rebuild the snapshot from schema-valid source records.",
        )


def _validate_delegated_profile(
    wrapper: dict[str, Any],
    *,
    schemas: dict[str, dict[str, Any]],
    registry: Registry[Any],
    location: str,
) -> None:
    contract_ref = wrapper.get("contract_schema_ref")
    declared_digest = wrapper.get("contract_schema_sha256")
    body = wrapper.get("body")
    embedded = embedded_schemas()
    contract = embedded.get(str(contract_ref))
    if contract is None:
        raise _error(
            "SCHEMA_UNSUPPORTED",
            f"Delegated profile uses an unknown contract at {location}: {contract_ref!r}",
            "Upgrade mojilex-cli or select a compatible snapshot.",
        )
    if declared_digest != contract.sha256:
        raise _error(
            "SCHEMA_UNSUPPORTED",
            f"Delegated profile contract hash does not match this CLI at {location}.",
            "Select the exact compatible snapshot and reader pair.",
            schema_uri=contract.uri,
            expected_sha256=contract.sha256,
            snapshot_sha256=declared_digest,
        )
    _validate_schema_instance(
        body,
        contract.uri,
        schemas=schemas,
        registry=registry,
        location=f"{location}/body",
        invalid_code="MANIFEST_INVALID",
    )


def _validate_schemas(snapshot: LoadedSnapshot) -> None:
    schemas, registry = _schema_contracts(snapshot)
    _validate_schema_instance(
        snapshot.manifest,
        _RELEASE_MANIFEST_SCHEMA_URI,
        schemas=schemas,
        registry=registry,
        location="manifest.json",
        invalid_code="MANIFEST_INVALID",
    )
    for descriptor in snapshot.descriptors.values():
        logical = str(descriptor["logical_name"])
        if descriptor.get("content_model") == "recordset-jsonl":
            for index, row in enumerate(snapshot.rows(logical), 1):
                _validate_schema_instance(
                    row,
                    descriptor.get("schema_ref"),
                    schemas=schemas,
                    registry=registry,
                    location=f"{logical}:{index}",
                    invalid_code="INDEX_CORRUPT",
                )
        elif descriptor.get("content_model") == "singleton-json":
            _validate_schema_instance(
                snapshot.document(logical),
                descriptor.get("schema_ref"),
                schemas=schemas,
                registry=registry,
                location=logical,
                invalid_code="INDEX_CORRUPT",
            )
    for resource in snapshot.resource_descriptors:
        schema_ref = resource.get("content_schema_ref")
        if not isinstance(schema_ref, str):
            continue
        uri = str(resource.get("uri"))
        payload = snapshot.verified_resource_bytes.get(uri)
        if payload is None:
            raise _error(
                "SCHEMA_UNSUPPORTED",
                f"Schema-bound resource is unavailable: {uri}",
                "Use a self-contained distribution-v1 snapshot.",
            )
        value = parse_bounded_json(payload, source=str(resource.get("path")))
        _validate_schema_instance(
            value,
            schema_ref,
            schemas=schemas,
            registry=registry,
            location=uri,
            invalid_code="MANIFEST_INVALID",
        )
        if (
            schema_ref in _PROFILE_WRAPPER_SCHEMA_URIS
            and _delegated_contract_ref(value) is not None
        ):
            _validate_delegated_profile(
                value,
                schemas=schemas,
                registry=registry,
                location=uri,
            )


def _validate_resource_aggregates(snapshot: LoadedSnapshot) -> None:
    groups: dict[str, list[dict[str, str]]] = {}
    for resource in snapshot.resource_descriptors:
        for binding in _resource_bindings(resource):
            if binding.get("kind") not in {"aggregate-member", "source-bundle-member"}:
                continue
            pointer = binding.get("manifest_pointer")
            source_path = resource.get("source_path")
            digest = resource.get("payload_sha256")
            if not all(isinstance(item, str) for item in (pointer, source_path, digest)):
                raise _error(
                    "MANIFEST_INVALID",
                    "Aggregate resource binding is incomplete.",
                    "Regenerate the release manifest.",
                )
            groups.setdefault(cast(str, pointer), []).append(
                {"path": cast(str, source_path), "sha256": cast(str, digest)}
            )
    required_pointers: set[str] = set()
    profiles = snapshot.manifest.get("profiles")
    if isinstance(profiles, dict):
        for field in ("canonical_schemas", "derived_schemas", "transport_schemas"):
            required_pointers.add(f"/profiles/{field}/sha256")
        taxonomy = profiles.get("taxonomy")
        if isinstance(taxonomy, dict) and "source_bundle_sha256" in taxonomy:
            required_pointers.add("/profiles/taxonomy/source_bundle_sha256")
    policies = snapshot.manifest.get("policies")
    platform_profiles = policies.get("platform_profiles") if isinstance(policies, dict) else None
    if isinstance(platform_profiles, dict) and "source_bundle_sha256" in platform_profiles:
        required_pointers.add("/policies/platform_profiles/source_bundle_sha256")
    if set(groups) != required_pointers:
        raise _error(
            "MANIFEST_INVALID",
            "Aggregate resource bindings do not cover the exact required selector set.",
            "Regenerate the self-contained release resources.",
        )
    for pointer, members in groups.items():
        members.sort(key=lambda item: item["path"].encode("utf-8"))
        calculated = hashlib.sha256(rfc8785.dumps(members)).hexdigest()
        present, expected = _manifest_pointer(snapshot.manifest, pointer)
        if not present or expected != calculated:
            raise _error(
                "MANIFEST_INVALID",
                f"Aggregate resource digest mismatch: {pointer}",
                "Restore the exact manifest and resource set.",
            )


def _validate_physical_set(snapshot: LoadedSnapshot) -> None:
    expected = {"manifest.json", "SHA256SUMS"}
    expected.update(str(item["path"]) for item in snapshot.descriptors.values())
    expected.update(
        str(item["path"])
        for item in snapshot.resource_descriptors
        if item.get("resource_kind") == "physical"
    )
    actual: set[str] = set()
    for path in snapshot.root.rglob("*"):
        if is_link_or_reparse_point(path):
            raise _error(
                "MANIFEST_INVALID",
                f"Snapshot contains a link or reparse point: {path}",
                "Use a regular-file snapshot tree.",
            )
        if path.is_file():
            actual.add(path.relative_to(snapshot.root).as_posix())
        elif not path.is_dir():
            raise _error(
                "MANIFEST_INVALID",
                f"Snapshot contains a special filesystem entry: {path}",
                "Use a regular-file snapshot tree.",
            )
    if actual != expected:
        raise _error(
            "MANIFEST_INVALID",
            "Physical snapshot file set differs from manifest descriptors.",
            "Remove unlisted objects or restore missing declared objects.",
            missing_paths=sorted(expected - actual),
            unexpected_paths=sorted(actual - expected),
        )
    checksum_path = _safe_child(snapshot.root, "SHA256SUMS")
    expected_lines = b"".join(
        (
            f"{_hash_file(_safe_child(snapshot.root, path), maximum=MAX_OBJECT_BYTES)[1]}  {path}\n"
        ).encode("ascii")
        for path in sorted(expected - {"SHA256SUMS"}, key=lambda item: item.encode("utf-8"))
    )
    if _read_bounded(checksum_path, 256 * 1024 * 1024) != expected_lines:
        raise _error(
            "CHECKSUM_MISMATCH",
            "SHA256SUMS is incomplete, unsorted, or inconsistent.",
            "Restore the checksum file generated for this exact snapshot.",
        )


def _validate_derived_from(
    descriptor: dict[str, Any], manifest: dict[str, Any], descriptors: dict[str, dict[str, Any]]
) -> None:
    dependency = descriptor.get("derived_from")
    if not isinstance(dependency, dict) or dependency.get("profile") != "derived-from-v1":
        raise _error(
            "MANIFEST_INVALID",
            f"Derived artifact has an invalid dependency profile: {descriptor.get('logical_name')}",
            "Regenerate the release manifest.",
        )
    if dependency.get("source_canonical_state_root_sha256") != manifest.get(
        "canonical_state_root_sha256"
    ):
        raise _error(
            "MANIFEST_INVALID",
            f"Derived artifact pins the wrong canonical root: {descriptor.get('logical_name')}",
            "Regenerate all derived artifacts from one canonical state.",
        )
    for item in dependency.get("manifest_inputs", []):
        present, value = _manifest_pointer(manifest, item.get("manifest_pointer"))
        if not present or hashlib.sha256(rfc8785.dumps(cast(Any, value))).hexdigest() != item.get(
            "value_sha256"
        ):
            raise _error(
                "MANIFEST_INVALID",
                f"Derived manifest input mismatch: {descriptor.get('logical_name')}",
                "Regenerate the derived artifact.",
            )
    for item in dependency.get("selectors", []):
        pointer = item.get("manifest_pointer")
        present, value = _manifest_pointer(manifest, pointer)
        expected_kind = (
            "profile" if isinstance(pointer, str) and pointer.startswith("/profiles/") else "policy"
        )
        if (
            not present
            or item.get("selector_kind") != expected_kind
            or hashlib.sha256(rfc8785.dumps(cast(Any, value))).hexdigest()
            != item.get("selector_value_sha256")
        ):
            raise _error(
                "MANIFEST_INVALID",
                f"Derived selector input mismatch: {descriptor.get('logical_name')}",
                "Regenerate the derived artifact.",
            )
    for item in dependency.get("derived_artifacts", []):
        source = descriptors.get(str(item.get("logical_name")))
        if source is None:
            raise _error(
                "MANIFEST_INVALID",
                f"Derived dependency is missing: {descriptor.get('logical_name')}",
                "Restore the complete artifact set.",
            )
        recordset = source.get("content_model") == "recordset-jsonl"
        digest_kind = "table-root" if recordset else "payload"
        digest = source.get("table_root_sha256" if recordset else "payload_sha256")
        if item.get("digest_kind") != digest_kind or item.get("sha256") != digest:
            raise _error(
                "MANIFEST_INVALID",
                f"Derived dependency digest mismatch: {descriptor.get('logical_name')}",
                "Regenerate the derived artifact.",
            )
    if dependency.get("physical_input_roots") != []:
        raise _error(
            "MANIFEST_INVALID",
            f"Stage-A derived artifact has physical inputs: {descriptor.get('logical_name')}",
            "Use the closed Stage-A derived-from profile.",
        )


def _validate_build_tuple(manifest: dict[str, Any]) -> None:
    features = [
        "concepts-v1",
        "rights-v1",
        "search-records-v1",
        "semantic-roles-v1",
        "state-roots-v1",
    ]
    if manifest.get("required_features") != features or manifest.get("trust_stage") != (
        "pre-enforcement"
    ):
        raise _error(
            "MANIFEST_INVALID",
            "Snapshot does not use the exact Stage-A feature/trust branch.",
            "Use a pre-enforcement distribution-v1 snapshot.",
        )
    build = cast(dict[str, Any], manifest["build"])
    profiles = cast(dict[str, Any], manifest["profiles"])
    expected_parameters = {
        "distribution_profile": profiles["distribution"]["id"],
        "layout_profile": manifest["layout_profile"],
        "storage_mode": cast(dict[str, Any], manifest["storage"])["mode"],
        "compression_profile": "none",
        "part_packing_profile": profiles["part_packing"]["id"],
        "bundle_mode": "none",
        "bundle_profile": profiles["bundling"]["id"],
        "partition_overrides": {},
    }
    parameters = build.get("parameters")
    if (
        parameters != expected_parameters
        or build.get("parameters_sha256") != hashlib.sha256(rfc8785.dumps(parameters)).hexdigest()
    ):
        raise _error(
            "MANIFEST_INVALID",
            "Closed build parameters or parameters_sha256 do not match.",
            "Regenerate the release with the exact build tuple.",
        )
    empty_root = hashlib.sha256(rfc8785.dumps([])).hexdigest()
    migrations_root = hashlib.sha256(rfc8785.dumps(manifest["migrations"])).hexdigest()
    if (
        build.get("evidence_inputs_root_sha256") != empty_root
        or build.get("migration_inputs_root_sha256") != migrations_root
    ):
        raise _error(
            "MANIFEST_INVALID",
            "Build evidence or migration input root is inconsistent.",
            "Regenerate the release with complete immutable inputs.",
        )

    def present(value: object, exists: bool) -> dict[str, Any]:
        return {"present": True, "value": value} if exists else {"present": False}

    build_input = {
        "build_input_profile_id": "release-build-input-v1",
        "manifest_header": {
            key: manifest[key]
            for key in (
                "manifest_version",
                "dataset",
                "snapshot_id",
                "schema_version",
                "layout_profile",
                "trust_stage",
                "minimum_reader_version",
                "required_features",
                "git",
                "languages",
                "storage",
            )
        },
        "selectors": {"profiles": manifest["profiles"], "policies": manifest["policies"]},
        "lineage": {
            "previous_snapshot": present(
                manifest.get("previous_snapshot"), "previous_snapshot" in manifest
            ),
            "change_set_batch": present(
                manifest.get("change_set_batch"), "change_set_batch" in manifest
            ),
            "migration_inputs_root_sha256": build["migration_inputs_root_sha256"],
        },
        "builder": {
            key: build[key]
            for key in (
                "tool",
                "tool_version",
                "tool_repository",
                "tool_commit",
                "dependency_lock_sha256",
                "distribution_profile",
                "parameters",
                "parameters_sha256",
                "evidence_inputs_root_sha256",
                "source_date_epoch",
            )
        },
    }
    if (
        build.get("build_inputs_sha256")
        != hashlib.sha256(rfc8785.dumps(cast(Any, build_input))).hexdigest()
    ):
        raise _error(
            "MANIFEST_INVALID",
            "release-build-input-v1 hash does not match the full build tuple.",
            "Regenerate the release from its immutable inputs.",
        )


def _validate_release_invariants(snapshot: LoadedSnapshot) -> None:
    manifest = snapshot.manifest
    required = {
        "manifest_version",
        "dataset",
        "snapshot_id",
        "schema_version",
        "layout_profile",
        "trust_stage",
        "minimum_reader_version",
        "required_features",
        "git",
        "languages",
        "profiles",
        "policies",
        "migrations",
        "storage",
        "artifacts",
        "resources",
        "bundles",
        "counts",
        "canonical_state_root_sha256",
        "derived_views_root_sha256",
        "artifact_set_sha256",
        "build",
    }
    if not required.issubset(manifest) or manifest.get("bundles") != []:
        raise _error(
            "MANIFEST_INVALID",
            "The local monolith manifest is incomplete or declares unsupported bundles.",
            "Use a complete unbundled monolith-v1 release.",
        )
    _validate_schemas(snapshot)
    _validate_resource_aggregates(snapshot)
    _validate_physical_set(snapshot)
    _validate_build_tuple(manifest)
    artifacts = cast(list[dict[str, Any]], manifest["artifacts"])
    artifact_order = [str(item.get("logical_name")) for item in artifacts]
    if artifact_order != sorted(artifact_order) or len(set(artifact_order)) != len(artifact_order):
        raise _error(
            "MANIFEST_INVALID",
            "Monolith artifact descriptors are not uniquely sorted by logical_name.",
            "Regenerate the release manifest.",
        )
    resource_order = [str(item.get("uri")) for item in snapshot.resource_descriptors]
    if resource_order != sorted(resource_order) or len(set(resource_order)) != len(resource_order):
        raise _error(
            "MANIFEST_INVALID",
            "Resource descriptors are not uniquely sorted by URI.",
            "Regenerate the release manifest.",
        )
    resource_uris = set(resource_order)
    exact_bindings: dict[str, int] = {}
    for resource in snapshot.resource_descriptors:
        uri = str(resource.get("uri"))
        kind = resource.get("resource_kind")
        if kind == "physical":
            required_physical = {
                "uri",
                "resource_kind",
                "source_path",
                "path",
                "media_type",
                "compression",
                "payload_sha256",
                "object_sha256",
                "uncompressed_byte_size",
                "object_byte_size",
            }
            if not required_physical.issubset(resource):
                raise _error(
                    "MANIFEST_INVALID",
                    f"Physical resource descriptor is incomplete: {uri}",
                    "Regenerate the release manifest.",
                )
            media_type = resource.get("media_type")
            if media_type != "application/schema+json":
                schema_ref = resource.get("content_schema_ref")
                if not isinstance(schema_ref, str) or schema_ref not in resource_uris:
                    raise _error(
                        "MANIFEST_INVALID",
                        f"JSON resource schema binding is unresolved: {uri}",
                        "Publish its exact content schema as a snapshot resource.",
                    )
        elif kind == "artifact-alias":
            if not {
                "uri",
                "resource_kind",
                "artifact_logical_name",
                "payload_sha256",
            }.issubset(resource):
                raise _error(
                    "MANIFEST_INVALID",
                    f"Artifact-alias resource descriptor is incomplete: {uri}",
                    "Regenerate the release manifest.",
                )
        bindings = _resource_bindings(resource)
        if not bindings:
            raise _error(
                "MANIFEST_INVALID",
                f"Resource has no semantic binding: {uri}",
                "Regenerate the release manifest.",
            )
        for binding in bindings:
            if binding.get("kind") == "exact-content":
                pointer = binding.get("manifest_pointer")
                if not isinstance(pointer, str):
                    raise _error(
                        "MANIFEST_INVALID",
                        f"Exact-content resource has no manifest pointer: {uri}",
                        "Regenerate the release manifest.",
                    )
                exact_bindings[pointer] = exact_bindings.get(pointer, 0) + 1
                parts = pointer.split("/")
                if (
                    len(parts) != 4
                    or parts[1] not in {"profiles", "policies"}
                    or parts[3] != "sha256"
                ):
                    raise _error(
                        "MANIFEST_INVALID",
                        f"Forbidden exact-content manifest pointer: {pointer}",
                        "Regenerate the release manifest.",
                    )
                container = manifest.get(parts[1])
                selector = container.get(parts[2]) if isinstance(container, dict) else None
                if not isinstance(selector, dict) or selector.get("sha256") != resource.get(
                    "payload_sha256"
                ):
                    raise _error(
                        "MANIFEST_INVALID",
                        f"Exact-content binding does not match {pointer}.",
                        "Restore the exact manifest and resources.",
                    )
    for container_name in ("profiles", "policies"):
        container = manifest.get(container_name)
        if not isinstance(container, dict):
            continue
        for field_name, selector in container.items():
            if not isinstance(selector, dict) or field_name in {
                "canonical_schemas",
                "derived_schemas",
                "transport_schemas",
            }:
                continue
            pointer = f"/{container_name}/{field_name}/sha256"
            if exact_bindings.get(pointer) != 1:
                raise _error(
                    "MANIFEST_INVALID",
                    f"Selector {pointer} must have exactly one exact-content resource.",
                    "Publish a self-contained offline snapshot.",
                )
    artifact_set = hashlib.sha256(
        rfc8785.dumps(
            {
                "artifacts": artifacts,
                "resources": list(snapshot.resource_descriptors),
                "bundles": [],
            }
        )
    ).hexdigest()
    if artifact_set != manifest.get("artifact_set_sha256"):
        raise _error(
            "MANIFEST_INVALID",
            "artifact_set_sha256 does not match descriptors.",
            "Restore the exact manifest or objects.",
        )
    semantic_names: set[str] = set()
    for descriptor in artifacts:
        logical = str(descriptor.get("logical_name"))
        model = descriptor.get("content_model")
        role = descriptor.get("semantic_role")
        common_descriptor_fields = {
            "logical_name",
            "semantic_role",
            "content_model",
            "path",
            "media_type",
            "schema_ref",
            "compression",
            "payload_sha256",
            "object_sha256",
            "uncompressed_byte_size",
            "object_byte_size",
        }
        schema_ref = descriptor.get("schema_ref")
        if (
            not common_descriptor_fields.issubset(descriptor)
            or not isinstance(schema_ref, str)
            or schema_ref not in resource_uris
        ):
            raise _error(
                "MANIFEST_INVALID",
                f"Artifact descriptor/schema binding is incomplete: {logical}",
                "Publish every referenced schema as a snapshot resource.",
            )
        if role not in {"canonical", "normative", "derived"} or model not in {
            "recordset-jsonl",
            "singleton-json",
        }:
            raise _error(
                "SCHEMA_UNSUPPORTED",
                f"Unsupported descriptor semantics for {logical}.",
                "Use the uncompressed JSON/JSONL monolith MVP.",
            )
        if role == "derived" and not isinstance(descriptor.get("derived_from"), dict):
            raise _error(
                "MANIFEST_INVALID",
                f"Derived artifact {logical} lacks derived_from.",
                "Regenerate the release manifest.",
            )
        if role == "derived":
            _validate_derived_from(descriptor, manifest, snapshot.descriptors)
        if role != "derived" and "derived_from" in descriptor:
            raise _error(
                "MANIFEST_INVALID",
                f"Non-derived artifact {logical} declares derived_from.",
                "Regenerate the release manifest.",
            )
        if logical in semantic_names:
            raise _error(
                "MANIFEST_INVALID",
                f"Duplicate logical artifact: {logical}",
                "Regenerate the release manifest.",
            )
        semantic_names.add(logical)
        if model == "recordset-jsonl":
            _validate_recordset(snapshot, logical, descriptor)
        else:
            snapshot.document(logical)
    canonical_entries = _state_entries(manifest, {"canonical", "normative"})
    canonical_root = hashlib.sha256(
        rfc8785.dumps({"profile": "state-roots-v1", "entries": canonical_entries})
    ).hexdigest()
    if canonical_root != manifest.get("canonical_state_root_sha256"):
        raise _error(
            "MANIFEST_INVALID",
            "canonical_state_root_sha256 does not match semantic entries.",
            "Restore the exact manifest or rebuild the snapshot.",
        )
    derived_entries = _state_entries(manifest, {"derived"})
    derived_root = hashlib.sha256(
        rfc8785.dumps(
            {
                "profile": "state-roots-v1",
                "source_canonical_state_root_sha256": canonical_root,
                "entries": derived_entries,
            }
        )
    ).hexdigest()
    if derived_root != manifest.get("derived_views_root_sha256"):
        raise _error(
            "MANIFEST_INVALID",
            "derived_views_root_sha256 does not match semantic entries.",
            "Restore the exact manifest or rebuild the snapshot.",
        )
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise _error(
            "MANIFEST_INVALID",
            "Manifest counts must be an object.",
            "Regenerate the release manifest.",
        )
    for logical in ("collections", "emojis", "memberships", "tombstones"):
        if (
            logical in snapshot.descriptors
            and logical in counts
            and counts[logical] != len(snapshot.rows(logical))
        ):
            raise _error(
                "MANIFEST_INVALID",
                f"Manifest count mismatch for {logical}.",
                "Restore or rebuild the snapshot.",
            )
    if "active_emojis" in counts and counts["active_emojis"] != sum(
        1 for row in snapshot.rows("emojis") if _record_status(row.get("availability")) == "active"
    ):
        raise _error(
            "MANIFEST_INVALID",
            "Manifest active_emojis count is inconsistent.",
            "Restore or rebuild the snapshot.",
        )
    expected_availability = {
        "collections": _status_histogram(
            snapshot.rows("collections"),
            "availability",
            ("active", "deleted", "private", "unavailable", "unknown"),
        ),
        "emojis": _status_histogram(
            snapshot.rows("emojis"),
            "availability",
            ("active", "deleted", "private", "unavailable", "unknown"),
        ),
    }
    expected_review = _status_histogram(
        snapshot.rows("emojis"),
        "review",
        ("approved", "changes_requested", "rejected", "unreviewed"),
    )
    expected_memberships = _status_histogram(
        snapshot.rows("memberships"),
        "status",
        ("active", "removed_from_collection", "unknown"),
    )
    for name, expected in (
        ("availability_by_status", expected_availability),
        ("emoji_review_by_status", expected_review),
        ("memberships_by_status", expected_memberships),
    ):
        if counts.get(name) != expected:
            raise _error(
                "MANIFEST_INVALID",
                f"Manifest {name} histogram is inconsistent.",
                "Restore or rebuild the snapshot.",
            )


def _record_status(value: object) -> str | None:
    if isinstance(value, dict) and isinstance(value.get("status"), str):
        return cast(str, value["status"])
    return value if isinstance(value, str) else None


def _status_histogram(
    rows: tuple[dict[str, Any], ...], field: str, statuses: tuple[str, ...]
) -> dict[str, int]:
    result = {status: 0 for status in statuses}
    for row in rows:
        status = _record_status(row.get(field))
        if status not in result:
            raise _error(
                "MANIFEST_INVALID",
                f"Record status is unsupported in the manifest histogram: {status!r}.",
                "Restore or rebuild the snapshot.",
            )
        result[status] += 1
    return result
