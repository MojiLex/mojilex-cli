"""Versioned rebuildable local index for dedupe scan results."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import rfc8785

from mojilex_cli.analysis import load_analysis_profile, profile_sha256
from mojilex_cli.dataset import DatasetSnapshot

from .engine import DedupeCandidate, DedupeScanReport, scan_snapshot

_FORMAT_VERSION = 2
_STATIC_METADATA_KEYS = frozenset({"format_version", "dedupe_profile_sha256"})
_DYNAMIC_METADATA_KEYS = frozenset(
    {"candidate_limit", "relation_state_sha256", "topology_state_sha256"}
)


class DedupeIndexError(RuntimeError):
    code = "DEDUPE_INDEX_ERROR"


class DedupeIndex:
    def __init__(self, path: Path, *, repository_root: Path | None = None) -> None:
        self.path = path.expanduser().resolve()
        if repository_root is not None and self.path.is_relative_to(repository_root.resolve()):
            raise DedupeIndexError("dedupe index must be outside the canonical repository")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.connection = sqlite3.connect(self.path, timeout=10)
            self.connection.row_factory = sqlite3.Row
            self._initialize()
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        except sqlite3.Error as exc:
            raise DedupeIndexError("cannot open the local dedupe index") from exc

    def __enter__(self) -> DedupeIndex:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def update(
        self,
        snapshot: DatasetSnapshot,
        *,
        rebuild: bool = False,
        selected_emoji_ids: set[str] | None = None,
        max_candidates: int | None = None,
    ) -> DedupeScanReport:
        profile = load_analysis_profile("dedupe-v1")
        thresholds = profile.data["candidate_thresholds"]
        limit = (
            int(thresholds["candidate_limit_default"]) if max_candidates is None else max_candidates
        )
        hard_limit = int(thresholds["candidate_limit_hard_max"])
        if not 1 <= limit <= hard_limit:
            raise DedupeIndexError(f"max dedupe candidates must be in the range 1..{hard_limit}")

        current = {
            emoji.id: _emoji_state_sha256(emoji.as_dict())
            for emoji in snapshot.emojis.values()
            if emoji.availability.status.value == "active"
            and emoji.fingerprints.status.value == "complete"
        }
        previous = {
            str(row["emoji_id"]): str(row["state_digest"])
            for row in self.connection.execute("SELECT emoji_id, state_digest FROM emoji_state")
        }
        if selected_emoji_ids is not None:
            unknown = selected_emoji_ids - set(snapshot.emojis)
            if unknown:
                raise DedupeIndexError("selected emoji IDs are not present in the dataset")

        dynamic_metadata = {
            "candidate_limit": str(limit),
            "relation_state_sha256": _relation_state_sha256(snapshot),
            "topology_state_sha256": _topology_state_sha256(snapshot),
        }
        stored_metadata = self._metadata()
        dynamic_changed = any(
            stored_metadata.get(key) != value for key, value in dynamic_metadata.items()
        )
        force_rebuild = rebuild or dynamic_changed or (not previous and bool(current))
        changed = (
            set(current)
            if force_rebuild
            else {
                key for key in set(current) | set(previous) if current.get(key) != previous.get(key)
            }
        )
        if selected_emoji_ids is not None:
            changed.update(selected_emoji_ids)

        affected = set(changed)
        if force_rebuild:
            report = scan_snapshot(snapshot, max_candidates=limit, mode="near")
            affected.update(previous)
        elif changed:
            old_neighbors = self._candidate_neighbors(changed)
            probe = scan_snapshot(
                snapshot,
                selected_emoji_ids=changed,
                max_candidates=limit,
                mode="near",
            )
            new_neighbors = set(probe.candidate_neighbors)
            affected.update(old_neighbors)
            affected.update(new_neighbors)
            report = scan_snapshot(
                snapshot,
                selected_emoji_ids=affected & set(snapshot.emojis),
                max_candidates=limit,
                mode="near",
            )
        else:
            report = scan_snapshot(
                snapshot,
                selected_emoji_ids=set(),
                max_candidates=limit,
                mode="near",
            )

        with self.connection:
            if force_rebuild:
                self.connection.execute("DELETE FROM candidates")
            elif affected:
                self._delete_candidate_sources(affected)
            for source_id, candidates in report.candidates.items():
                for rank, candidate in enumerate(candidates, start=1):
                    self.connection.execute(
                        "INSERT INTO candidates(source_emoji_id, rank, against_emoji_id, payload) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            source_id,
                            rank,
                            candidate.against_emoji_id,
                            _candidate_payload(candidate),
                        ),
                    )
            self.connection.execute("DELETE FROM emoji_state")
            self.connection.executemany(
                "INSERT INTO emoji_state(emoji_id, state_digest) VALUES (?, ?)",
                sorted(current.items()),
            )
            self.connection.execute("DELETE FROM exact_groups")
            self.connection.executemany(
                "INSERT INTO exact_groups(group_id, payload) VALUES (?, ?)",
                [
                    (
                        str(group["id"]),
                        json.dumps(
                            group,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                    )
                    for group in report.exact_groups
                ],
            )
            self.connection.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                sorted(dynamic_metadata.items()),
            )
        return report

    def explain(self, emoji_id: str, against_emoji_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT payload FROM candidates WHERE source_emoji_id = ? AND against_emoji_id = ?",
            (emoji_id, against_emoji_id),
        ).fetchone()
        return json.loads(str(row[0])) if row is not None else None

    def _metadata(self) -> dict[str, str]:
        return {
            str(row[0]): str(row[1])
            for row in self.connection.execute("SELECT key, value FROM metadata")
        }

    def _load_temp_ids(self, values: set[str]) -> None:
        self.connection.execute("DELETE FROM affected_ids")
        self.connection.executemany(
            "INSERT INTO affected_ids(emoji_id) VALUES (?)",
            ((value,) for value in sorted(values)),
        )

    def _candidate_neighbors(self, values: set[str]) -> set[str]:
        if not values:
            return set()
        self._load_temp_ids(values)
        rows = self.connection.execute(
            "SELECT source_emoji_id, against_emoji_id FROM candidates AS candidate "
            "WHERE EXISTS (SELECT 1 FROM affected_ids AS affected "
            "WHERE affected.emoji_id = candidate.source_emoji_id) "
            "OR EXISTS (SELECT 1 FROM affected_ids AS affected "
            "WHERE affected.emoji_id = candidate.against_emoji_id)"
        )
        return {str(value) for row in rows for value in row}

    def _delete_candidate_sources(self, values: set[str]) -> None:
        if not values:
            return
        self._load_temp_ids(values)
        self.connection.execute(
            "DELETE FROM candidates AS candidate WHERE EXISTS "
            "(SELECT 1 FROM affected_ids AS affected "
            "WHERE affected.emoji_id = candidate.source_emoji_id)"
        )

    def _initialize(self) -> None:
        actual_profile = profile_sha256("dedupe-v1")
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA trusted_schema=OFF;
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
        stored = self._metadata()
        expected = {
            "format_version": str(_FORMAT_VERSION),
            "dedupe_profile_sha256": actual_profile,
        }
        static_mismatch = any(stored.get(key) != value for key, value in expected.items())
        existing_data_tables = {
            str(row[0])
            for row in self.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('candidates', 'exact_groups', 'emoji_state')"
            )
        }
        if static_mismatch and (stored or existing_data_tables):
            with self.connection:
                self.connection.executescript(
                    """
                    DROP TABLE IF EXISTS candidates;
                    DROP TABLE IF EXISTS exact_groups;
                    DROP TABLE IF EXISTS emoji_state;
                    DELETE FROM metadata;
                    """
                )
            stored = {}
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS emoji_state (
                emoji_id TEXT PRIMARY KEY,
                state_digest TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS candidates (
                source_emoji_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                against_emoji_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY(source_emoji_id, rank)
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS exact_groups (
                group_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE TEMP TABLE IF NOT EXISTS affected_ids (
                emoji_id TEXT PRIMARY KEY
            ) WITHOUT ROWID;
            """
        )
        if not stored:
            with self.connection:
                self.connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)", sorted(expected.items())
                )
        elif {key: stored.get(key) for key in _STATIC_METADATA_KEYS} != expected:
            raise DedupeIndexError("dedupe index static metadata is inconsistent")
        unexpected_dynamic = _DYNAMIC_METADATA_KEYS - set(stored)
        if unexpected_dynamic and unexpected_dynamic != _DYNAMIC_METADATA_KEYS:
            # A partially written dynamic header is never trusted; its first
            # update will rebuild all rows and replace the complete header.
            with self.connection:
                self.connection.execute("DELETE FROM candidates")
                self.connection.execute("DELETE FROM exact_groups")
                self.connection.execute("DELETE FROM emoji_state")
                self.connection.executemany(
                    "DELETE FROM metadata WHERE key = ?",
                    ((key,) for key in _DYNAMIC_METADATA_KEYS),
                )


def _sha256_jcs(value: Any) -> str:
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


def _emoji_state_sha256(raw: dict[str, Any]) -> str:
    return _sha256_jcs(
        {
            "availability_status": raw["availability"]["status"],
            "media": raw["media"],
            "fingerprints": raw["fingerprints"],
            "rendering": raw["facets"]["rendering"],
            "text_content": raw["facets"]["text_content"],
        }
    )


def _relation_state_sha256(snapshot: DatasetSnapshot) -> str:
    return _sha256_jcs([snapshot.relations[key].as_dict() for key in sorted(snapshot.relations)])


def _topology_state_sha256(snapshot: DatasetSnapshot) -> str:
    return _sha256_jcs(
        {
            "collections": [
                {
                    "id": collection.id,
                    "availability_status": collection.availability.status.value,
                }
                for collection in sorted(snapshot.collections.values(), key=lambda item: item.id)
            ],
            "memberships": [
                {
                    "id": membership.id,
                    "collection_id": membership.collection_id,
                    "emoji_id": membership.emoji_id,
                    "status": membership.status.value,
                }
                for membership in sorted(snapshot.memberships.values(), key=lambda item: item.id)
            ],
        }
    )


def _candidate_payload(candidate: DedupeCandidate) -> str:
    return json.dumps(
        {
            "emoji_id": candidate.emoji_id,
            "against_emoji_id": candidate.against_emoji_id,
            "role": candidate.role,
            "variant_id": candidate.variant_id,
            "against_role": candidate.against_role,
            "against_variant_id": candidate.against_variant_id,
            "signals": list(candidate.signals),
            "primary_distance": candidate.primary_distance,
            "alpha_distance": candidate.alpha_distance,
            "edge_distance": candidate.edge_distance,
            "p90_distance": candidate.p90_distance,
            "speed_ratio": candidate.speed_ratio,
            "duration_compatible": candidate.duration_compatible,
            "text_match": candidate.text_match,
            "text_mismatch": candidate.text_mismatch,
            "high_priority": candidate.high_priority,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
