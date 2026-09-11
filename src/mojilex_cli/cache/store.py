"""Persistent metadata/AI cache which structurally rejects media and secrets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import rfc8785
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mojilex_cli.ai import DescriptionResult, VisionContext
from mojilex_cli.config.secrets import contains_secret_text, is_secret_key, url_has_credentials

_FORBIDDEN_KEYS = frozenset(
    {
        "file_id",
        "download_url",
        "raw_download_url",
        "local_path",
        "source_path",
        "image",
        "image_bytes",
        "frames",
        "frame_paths",
        "contact_sheet",
        "contact_sheets",
        "raw_media",
    }
)
_TELEGRAM_TOKEN = re.compile(r"\b[0-9]{5,12}:[A-Za-z0-9_-]{20,}\b")
_DOWNLOAD_TOKEN_URL = re.compile(r"(?i)https://api\.telegram\.org/(?:file/)?bot[^/\s]+")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON_BYTES = 4 * 1024 * 1024
_CACHE_SCHEMA_VERSION = 2
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


class CacheError(RuntimeError):
    pass


class CachedAIResult(BaseModel):
    """One immutable provider result and its real generation instant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[2] = 2
    generated_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    result: DescriptionResult

    @field_validator("generated_at")
    @classmethod
    def real_utc_second(cls, value: str) -> str:
        try:
            datetime.strptime(value, _UTC_FORMAT)
        except ValueError as exc:
            raise ValueError("generated_at must be a real UTC second ending in Z") from exc
        return value


@dataclass(frozen=True, slots=True)
class AICacheWrite:
    """One immutable row participating in an atomic AI request result write."""

    key: str
    result: DescriptionResult
    generated_at: str
    aliases: tuple[str, ...] = ()


class CacheStore:
    def __init__(
        self,
        path: Path,
        *,
        repository_root: Path | None = None,
        read_only: bool = False,
    ) -> None:
        self.path = path.resolve()
        if repository_root is not None and self.path.is_relative_to(repository_root.resolve()):
            raise CacheError("persistent cache must be outside the target repository")
        self.read_only = read_only
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if read_only:
                uri = f"file:{self.path.as_posix()}?mode=ro"
                self._connection = sqlite3.connect(uri, uri=True, timeout=5)
                self._cache_schema_version = int(
                    self._connection.execute("PRAGMA user_version").fetchone()[0]
                )
            else:
                self._connection = sqlite3.connect(self.path, timeout=5)
                self._initialize()
                try:
                    os.chmod(self.path, 0o600)
                except OSError:
                    pass
        except sqlite3.Error as exc:
            raise CacheError(f"cannot open metadata cache: {exc}") from exc
        self._connection.row_factory = sqlite3.Row

    def __enter__(self) -> CacheStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def _initialize(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version > _CACHE_SCHEMA_VERSION:
            raise CacheError(
                f"cache schema {version} is newer than supported {_CACHE_SCHEMA_VERSION}"
            )
        self._connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA trusted_schema=OFF;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS metadata_cache (
                cache_key TEXT PRIMARY KEY,
                base_sha TEXT,
                payload_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                accessed_at INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS ai_cache (
                cache_key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                accessed_at INTEGER NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS ai_cache_alias (
                alias_key TEXT PRIMARY KEY,
                cache_key TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                accessed_at INTEGER NOT NULL,
                FOREIGN KEY(cache_key) REFERENCES ai_cache(cache_key) ON DELETE CASCADE
            ) WITHOUT ROWID;
            PRAGMA user_version=2;
            """
        )
        self._connection.commit()
        self._cache_schema_version = _CACHE_SCHEMA_VERSION

    def get_metadata(self, key: str, *, base_sha: str | None = None) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT payload_json, base_sha FROM metadata_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        if row is None or (base_sha is not None and row["base_sha"] != base_sha):
            return None
        if not self.read_only:
            self._connection.execute(
                "UPDATE metadata_cache SET accessed_at = ? WHERE cache_key = ?",
                (int(time.time()), key),
            )
            self._connection.commit()
        value = cast(dict[str, Any], json.loads(row["payload_json"]))
        _assert_cache_safe(value)
        return value

    def put_metadata(
        self, key: str, payload: Mapping[str, Any], *, base_sha: str | None = None
    ) -> None:
        self._ensure_writable()
        serialized = _serialize_safe(payload)
        now = int(time.time())
        with self._connection:
            self._connection.execute(
                """INSERT INTO metadata_cache(
                     cache_key, base_sha, payload_json, created_at, accessed_at
                   )
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET
                     base_sha=excluded.base_sha, payload_json=excluded.payload_json,
                     created_at=excluded.created_at, accessed_at=excluded.accessed_at""",
                (key, base_sha, serialized, now, now),
            )

    def get_ai(self, key: str) -> DescriptionResult | None:
        hit = self.get_ai_entry(key)
        return hit[1].result if hit is not None else None

    def get_ai_entry(
        self, key: str, *, follow_aliases: bool = False
    ) -> tuple[str, CachedAIResult] | None:
        """Return the resolved key and versioned entry, optionally following one alias."""

        row = self._connection.execute(
            "SELECT cache_key, payload_json FROM ai_cache WHERE cache_key = ?", (key,)
        ).fetchone()
        used_alias = False
        if row is None and follow_aliases and self._cache_schema_version >= 2:
            row = self._connection.execute(
                """SELECT ai_cache.cache_key, ai_cache.payload_json
                   FROM ai_cache_alias
                   JOIN ai_cache ON ai_cache.cache_key = ai_cache_alias.cache_key
                   WHERE ai_cache_alias.alias_key = ?""",
                (key,),
            ).fetchone()
            used_alias = row is not None
        if row is None:
            return None
        entry = _decode_ai_entry(row["payload_json"])
        # Version-1 payloads contain no trustworthy generation instant. Treat
        # them as misses instead of assigning the current time and potentially
        # granting a qualification that did not exist when they were produced.
        if entry is None:
            return None
        if not self.read_only:
            now = int(time.time())
            self._connection.execute(
                "UPDATE ai_cache SET accessed_at = ? WHERE cache_key = ?",
                (now, row["cache_key"]),
            )
            if used_alias:
                self._connection.execute(
                    "UPDATE ai_cache_alias SET accessed_at = ? WHERE alias_key = ?",
                    (now, key),
                )
            self._connection.commit()
        return str(row["cache_key"]), entry

    def put_ai(
        self,
        key: str,
        result: DescriptionResult,
        *,
        generated_at: str | None = None,
        aliases: Sequence[str] = (),
    ) -> CachedAIResult:
        generated = generated_at or datetime.now(UTC).strftime(_UTC_FORMAT)
        return self.put_ai_batch(
            (AICacheWrite(key=key, result=result, generated_at=generated, aliases=tuple(aliases)),)
        )[key]

    def put_ai_batch(
        self,
        writes: Sequence[AICacheWrite],
        *,
        envelope_key: str | None = None,
        envelope: Mapping[str, Any] | None = None,
        repair_request_envelope: bool = False,
    ) -> dict[str, CachedAIResult]:
        """Write every result, alias, and optional request envelope in one transaction."""

        self._ensure_writable()
        if not writes:
            raise CacheError("atomic AI cache batch requires at least one result")
        if (envelope_key is None) != (envelope is None):
            raise CacheError("AI cache envelope key and payload must be provided together")
        if repair_request_envelope and envelope_key is None:
            raise CacheError("AI cache envelope repair requires a complete envelope")

        prepared: list[tuple[AICacheWrite, CachedAIResult, str]] = []
        keys: set[str] = set()
        alias_targets: dict[str, str] = {}
        for write in writes:
            if write.key in keys:
                raise CacheError("atomic AI cache batch contains a duplicate key")
            keys.add(write.key)
            candidate = CachedAIResult(
                generated_at=write.generated_at,
                result=write.result,
            )
            serialized = _serialize_safe(candidate.model_dump(mode="json"))
            prepared.append((write, candidate, serialized))
            for alias in set(write.aliases):
                previous = alias_targets.setdefault(alias, write.key)
                if previous != write.key:
                    raise CacheError("AI cache alias maps to multiple batch results")

        envelope_serialized = _serialize_safe(envelope) if envelope is not None else None
        now = int(time.time())
        stored: dict[str, CachedAIResult] = {}
        try:
            with self._connection:
                for write, candidate, serialized in prepared:
                    existing = self._connection.execute(
                        "SELECT payload_json FROM ai_cache WHERE cache_key = ?", (write.key,)
                    ).fetchone()
                    if existing is None:
                        self._connection.execute(
                            """INSERT INTO ai_cache(
                                 cache_key, payload_json, created_at, accessed_at
                               ) VALUES (?, ?, ?, ?)""",
                            (write.key, serialized, now, now),
                        )
                        value = candidate
                    else:
                        try:
                            decoded = _decode_ai_entry(existing["payload_json"])
                        except CacheError:
                            # A provider response produced for this exact request is a
                            # trustworthy replacement for a corrupt row. Cache probes
                            # remain fail-closed, but corruption must not permanently
                            # poison an otherwise recoverable cache key.
                            decoded = None
                        if decoded is None:
                            self._connection.execute(
                                """UPDATE ai_cache
                                   SET payload_json = ?, created_at = ?, accessed_at = ?
                                   WHERE cache_key = ?""",
                                (serialized, now, now, write.key),
                            )
                            value = candidate
                        else:
                            value = decoded
                            # Cache entries are immutable. In particular, a later write
                            # must never move generated_at across a qualification boundary.
                            self._connection.execute(
                                "UPDATE ai_cache SET accessed_at = ? WHERE cache_key = ?",
                                (now, write.key),
                            )
                    stored[write.key] = value
                    for alias in sorted(set(write.aliases)):
                        if alias == write.key:
                            continue
                        if repair_request_envelope:
                            self._connection.execute(
                                """INSERT INTO ai_cache_alias(
                                     alias_key, cache_key, created_at, accessed_at
                                   ) VALUES (?, ?, ?, ?)
                                   ON CONFLICT(alias_key) DO UPDATE SET
                                     cache_key=excluded.cache_key,
                                     accessed_at=excluded.accessed_at""",
                                (alias, write.key, now, now),
                            )
                        else:
                            self._connection.execute(
                                """INSERT INTO ai_cache_alias(
                                     alias_key, cache_key, created_at, accessed_at
                                   ) VALUES (?, ?, ?, ?)
                                   ON CONFLICT(alias_key) DO NOTHING""",
                                (alias, write.key, now, now),
                            )

                if envelope_key is not None and envelope_serialized is not None:
                    existing_envelope = self._connection.execute(
                        "SELECT payload_json FROM metadata_cache WHERE cache_key = ?",
                        (envelope_key,),
                    ).fetchone()
                    if existing_envelope is None:
                        self._connection.execute(
                            """INSERT INTO metadata_cache(
                                 cache_key, base_sha, payload_json, created_at, accessed_at
                               ) VALUES (?, NULL, ?, ?, ?)""",
                            (envelope_key, envelope_serialized, now, now),
                        )
                    elif (
                        existing_envelope["payload_json"] != envelope_serialized
                        and not repair_request_envelope
                    ):
                        raise CacheError("AI request envelope is immutable")
                    elif existing_envelope["payload_json"] != envelope_serialized:
                        # A complete fresh provider response may repair a malformed
                        # or partial run envelope. Rows remain content-addressed and
                        # immutable; envelope + run aliases move together atomically.
                        self._connection.execute(
                            """UPDATE metadata_cache
                               SET base_sha = NULL, payload_json = ?,
                                   created_at = ?, accessed_at = ?
                               WHERE cache_key = ?""",
                            (envelope_serialized, now, now, envelope_key),
                        )
                    else:
                        self._connection.execute(
                            "UPDATE metadata_cache SET accessed_at = ? WHERE cache_key = ?",
                            (now, envelope_key),
                        )
        except sqlite3.Error as exc:
            raise CacheError(f"cannot atomically write AI cache batch: {exc}") from exc
        return stored

    def info(self) -> dict[str, int]:
        metadata = self._connection.execute("SELECT COUNT(*) FROM metadata_cache").fetchone()[0]
        ai = self._connection.execute("SELECT COUNT(*) FROM ai_cache").fetchone()[0]
        size = self.path.stat().st_size if self.path.exists() else 0
        return {"metadata_entries": metadata, "ai_entries": ai, "database_bytes": size}

    def prune(self, *, older_than_epoch: int) -> dict[str, int]:
        self._ensure_writable()
        with self._connection:
            metadata = self._connection.execute(
                "DELETE FROM metadata_cache WHERE accessed_at < ?", (older_than_epoch,)
            ).rowcount
            ai = self._connection.execute(
                "DELETE FROM ai_cache WHERE accessed_at < ?", (older_than_epoch,)
            ).rowcount
        return {"metadata_removed": metadata, "ai_removed": ai}

    def _ensure_writable(self) -> None:
        if self.read_only:
            raise CacheError("cache is read-only (dry-run mode)")


def media_digest(media: Sequence[Mapping[str, object]]) -> str:
    normalized: list[dict[str, str]] = []
    identities: set[tuple[str, str | None]] = set()
    for item in media:
        role, sha256 = item.get("role"), item.get("sha256")
        variant = item.get("variant_id")
        if (
            not isinstance(role, str)
            or not isinstance(sha256, str)
            or not _SHA256.fullmatch(sha256)
        ):
            raise CacheError("media digest requires role and lowercase SHA-256")
        if variant is not None and not isinstance(variant, str):
            raise CacheError("media variant_id must be a string")
        identity = (role, variant)
        if identity in identities:
            raise CacheError("media role + variant_id must be unique")
        identities.add(identity)
        value = {"role": role, "sha256": sha256}
        if variant is not None:
            value["variant_id"] = variant
        normalized.append(value)
    if not normalized:
        raise CacheError("media digest requires at least one variant")
    normalized.sort(key=lambda value: (value["role"], value.get("variant_id", ""), value["sha256"]))
    return hashlib.sha256(rfc8785.dumps(normalized)).hexdigest()


def canonical_context_hash(context: VisionContext | Mapping[str, object]) -> str:
    value = context.model_dump(mode="json") if isinstance(context, BaseModel) else dict(context)
    allowed = {
        "fallback_emoji",
        "needs_repainting",
        "requested_languages",
        "frame_count",
        "canvas_size",
        "background_variants",
    }
    if set(value) - allowed:
        raise CacheError("canonical AI context contains non-approved fields")
    _assert_cache_safe(value)
    return hashlib.sha256(rfc8785.dumps(cast(Any, value))).hexdigest()


def deterministic_analysis_key(
    *,
    media_digest_value: str,
    pipeline_version: str,
    color_profile: str,
    color_profile_sha256: str,
    dedupe_profile: str,
    dedupe_profile_sha256: str,
    decoder_backend_fingerprint: str,
) -> str:
    hashes = (
        media_digest_value,
        color_profile_sha256,
        dedupe_profile_sha256,
        decoder_backend_fingerprint,
    )
    if any(not _SHA256.fullmatch(value) for value in hashes):
        raise CacheError("deterministic analysis key requires canonical SHA-256 components")
    if not all((pipeline_version, color_profile, dedupe_profile)):
        raise CacheError("deterministic analysis key components must be non-empty")
    return hashlib.sha256(
        rfc8785.dumps(
            {
                "media_digest": media_digest_value,
                "pipeline_version": pipeline_version,
                "color_profile": color_profile,
                "color_profile_sha256": color_profile_sha256,
                "dedupe_profile": dedupe_profile,
                "dedupe_profile_sha256": dedupe_profile_sha256,
                "decoder_backend_fingerprint": decoder_backend_fingerprint,
            }
        )
    ).hexdigest()


def ai_cache_key(
    *,
    media_digest_value: str,
    provider: str,
    model: str,
    model_revision: str | None,
    prompt_version: str,
    prompt_sha256: str,
    schema_version: str,
    pipeline_version: str,
    languages: Sequence[str],
    canonical_context_hash_value: str,
    description_profile: str,
    taxonomy_version: str,
    routing_policy_version: str,
    shown_media_sha256: Sequence[str],
    request_parameters_sha256: str,
    request_identity_sha256: str | None = None,
    item_label: str = "E001",
) -> str:
    if request_identity_sha256 is None:
        request_identity_sha256 = hashlib.sha256(
            rfc8785.dumps(
                {
                    "canonical_context_hash": canonical_context_hash_value,
                    "shown_media_sha256": list(shown_media_sha256),
                    "item_label": item_label,
                }
            )
        ).hexdigest()
    hashes = (
        media_digest_value,
        prompt_sha256,
        canonical_context_hash_value,
        request_parameters_sha256,
        request_identity_sha256,
        *shown_media_sha256,
    )
    if any(not _SHA256.fullmatch(value) for value in hashes):
        raise CacheError("AI cache key requires canonical SHA-256 components")
    if not shown_media_sha256 or len(shown_media_sha256) > 32:
        raise CacheError("AI cache key requires 1-32 shown-media hashes")
    if not re.fullmatch(r"E[0-9]{3}", item_label):
        raise CacheError("AI cache key requires a canonical item label")
    if not all(
        (
            provider,
            model,
            prompt_version,
            schema_version,
            pipeline_version,
            description_profile,
            taxonomy_version,
            routing_policy_version,
        )
    ):
        raise CacheError("AI cache key components must be non-empty")
    value = {
        "media_digest": media_digest_value,
        "provider": provider,
        "model": model,
        # JSON null is the normative marker when the provider reports no revision.
        "model_revision": model_revision,
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256,
        "schema_version": schema_version,
        "pipeline_version": pipeline_version,
        "languages": sorted(set(languages)),
        "canonical_context_hash": canonical_context_hash_value,
        "request_identity_sha256": request_identity_sha256,
        "item_label": item_label,
        "description_profile": description_profile,
        "taxonomy_version": taxonomy_version,
        "routing_policy_version": routing_policy_version,
        "shown_media_sha256": list(shown_media_sha256),
        "request_parameters_sha256": request_parameters_sha256,
    }
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def _serialize_safe(value: object) -> str:
    _assert_cache_safe(value)
    result = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(result.encode("utf-8")) > _MAX_JSON_BYTES:
        raise CacheError("cache payload exceeds 4 MiB")
    return result


def _decode_ai_entry(payload_json: str) -> CachedAIResult | None:
    try:
        value = json.loads(payload_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CacheError("AI cache payload is not valid JSON") from exc
    _assert_cache_safe(value)
    if not isinstance(value, Mapping):
        raise CacheError("AI cache payload must be an object")
    if "format_version" not in value:
        # The only pre-v2 representation was a bare DescriptionResult. Validate
        # its shape before classifying it as a safely ignored legacy entry.
        try:
            DescriptionResult.model_validate(value)
        except ValueError as exc:
            raise CacheError("AI cache payload has an unknown unversioned format") from exc
        return None
    try:
        return CachedAIResult.model_validate(value)
    except ValueError as exc:
        raise CacheError("AI cache payload has an unsupported or invalid format") from exc


def _assert_cache_safe(value: object, *, path: str = "cache") -> None:
    if isinstance(value, bytes):
        raise CacheError(f"binary media is forbidden in persistent {path}")
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.lower() in _FORBIDDEN_KEYS or is_secret_key(key):
                raise CacheError(f"unsafe field is forbidden in persistent {path}: {key}")
            _assert_cache_safe(child, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _assert_cache_safe(child, path=path)
        return
    if isinstance(value, str):
        if len(value) > 20_000:
            raise CacheError(f"oversized string is forbidden in persistent {path}")
        if (
            _TELEGRAM_TOKEN.search(value)
            or _DOWNLOAD_TOKEN_URL.search(value)
            or contains_secret_text(value)
        ):
            raise CacheError(f"credential is forbidden in persistent {path}")
        if "://" in value and url_has_credentials(value):
            raise CacheError(f"credential URL is forbidden in persistent {path}")
