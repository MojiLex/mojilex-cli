"""Exact, offline concept candidates for SPEC-003 generation and cache inputs."""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.dataset.serialization import parse_json
from mojilex_cli.domain import jcs_bytes
from mojilex_cli.read.snapshot import validate_embedded_schema_instance


class ConceptContextError(ValueError):
    """The exact concept input contract could not be established."""

    code = "POLICY_INVALID"


@dataclass(frozen=True, slots=True)
class ConceptContext:
    """Immutable hash-bound inputs; returned prompt dictionaries are fresh copies."""

    registry_id: str
    registry_jcs: bytes
    profile_id: str
    profile_jcs: bytes
    candidate_ids: tuple[str, ...]

    @property
    def provenance_fields(self) -> dict[str, str]:
        return {
            "concept_registry_id": self.registry_id,
            "concept_registry_sha256": hashlib.sha256(self.registry_jcs).hexdigest(),
            "concept_candidate_profile_id": self.profile_id,
            "concept_candidate_profile_sha256": hashlib.sha256(self.profile_jcs).hexdigest(),
            "concept_candidate_set_sha256": hashlib.sha256(
                jcs_bytes(list(self.candidate_ids))
            ).hexdigest(),
        }

    @property
    def candidate_records(self) -> tuple[dict[str, Any], ...]:
        registry = json.loads(self.registry_jcs)
        return tuple(row for row in registry["concepts"] if row["status"] == "active")

    @property
    def prompt_context(self) -> dict[str, Any]:
        return {
            **self.provenance_fields,
            "candidate_ids": list(self.candidate_ids),
            "candidates": list(self.candidate_records),
        }

    def validate_selection(
        self, concept_ids: Iterable[str], *, require_complete: bool = True
    ) -> tuple[str, ...]:
        selected = tuple(concept_ids)
        if (
            any(not isinstance(value, str) for value in selected)
            or len(selected) > 16
            or (require_complete and not selected)
            or len(selected) != len(set(selected))
            or tuple(sorted(selected)) != selected
            or not set(selected).issubset(self.candidate_ids)
        ):
            raise ConceptContextError(
                "Concept selection must contain 1..16 unique sorted IDs from the exact active "
                "candidate set; unknown proposals belong only in staging."
            )
        return selected


def _validate_registry_order_and_links(registry: dict[str, Any]) -> None:
    rows = registry["concepts"]
    identifiers = [row["id"] for row in rows]
    if identifiers != sorted(set(identifiers), key=lambda value: value.encode("utf-8")):
        raise ConceptContextError("Concept registry IDs must be unique and bytewise sorted.")
    known = set(identifiers)
    incoming = dict.fromkeys(identifiers, 0)
    children: dict[str, list[str]] = {identifier: [] for identifier in identifiers}
    replacements: dict[str, str] = {}
    for row in rows:
        parents = row["parent_ids"]
        if parents != sorted(parents) or not set(parents).issubset(known):
            raise ConceptContextError("Concept parents must be sorted existing IDs.")
        for parent in parents:
            incoming[row["id"]] += 1
            children[parent].append(row["id"])
        if "replaced_by" in row:
            replacement = row["replaced_by"]
            if replacement not in known or replacement == row["id"]:
                raise ConceptContextError("Concept replacement must name another existing ID.")
            replacements[row["id"]] = replacement
    queue = deque(identifier for identifier, count in incoming.items() if count == 0)
    visited = 0
    while queue:
        visited += 1
        for child in children[queue.popleft()]:
            incoming[child] -= 1
            if incoming[child] == 0:
                queue.append(child)
    if visited != len(identifiers):
        raise ConceptContextError("Concept parent graph contains a cycle.")
    for identifier in replacements:
        path: set[str] = set()
        current = identifier
        while current in replacements:
            if current in path:
                raise ConceptContextError("Concept replacement graph contains a cycle.")
            path.add(current)
            current = replacements[current]


def concept_context_from_documents(
    registry: dict[str, Any], profile: dict[str, Any]
) -> ConceptContext:
    """Validate exact versioned documents; never truncate or guess a candidate set."""

    try:
        for value, schema in (
            (registry, "concepts-registry.schema.json"),
            (profile, "concept-candidate-profile.schema.json"),
        ):
            validate_embedded_schema_instance(
                value,
                f"mlx://schemas/distribution/v1/{schema}",
                location=schema,
                invalid_code="POLICY_INVALID",
            )
        _validate_registry_order_and_links(registry)
        candidates = tuple(row["id"] for row in registry["concepts"] if row["status"] == "active")
        if not candidates or len(candidates) > profile["max_candidates"]:
            raise ConceptContextError("Active concept candidates must number 1..65536.")
        return ConceptContext(
            registry_id=registry["registry_id"],
            registry_jcs=jcs_bytes(registry),
            profile_id=profile["profile_id"],
            profile_jcs=jcs_bytes(profile),
            candidate_ids=candidates,
        )
    except (CommandError, TypeError, ValueError) as exc:
        if isinstance(exc, ConceptContextError):
            raise
        raise ConceptContextError("Concept registry or candidate profile is invalid.") from exc


def load_concept_context(dataset_root: Path) -> ConceptContext:
    """Read the active registry and exact JCS candidate profile without network access."""

    try:
        registry_path = dataset_root / "taxonomy" / "v1" / "concepts.json"
        profile_path = dataset_root / "analysis-profiles" / "concept-candidates-v1.json"
        registry = parse_json(registry_path.read_bytes(), source=str(registry_path))
        profile_bytes = profile_path.read_bytes()
        profile = parse_json(profile_bytes, source=str(profile_path))
        if not isinstance(registry, dict) or not isinstance(profile, dict):
            raise ConceptContextError("Concept input documents must be JSON objects.")
        result = concept_context_from_documents(registry, profile)
        if profile_bytes != result.profile_jcs:
            raise ConceptContextError("Concept candidate profile must use exact JCS bytes.")
        return result
    except OSError as exc:
        raise ConceptContextError("Concept registry or candidate profile is unavailable.") from exc
