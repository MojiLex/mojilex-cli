"""Pure, offline query service over a verified local monolith snapshot."""

from __future__ import annotations

import copy
import hashlib
import re
import unicodedata
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import rfc8785

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.read.cursor import decode_cursor, encode_cursor, pagination_domain
from mojilex_cli.read.local import local_text
from mojilex_cli.read.snapshot import LoadedSnapshot

_BCP47_RE = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")
_CANONICAL_INSTANT_RE = re.compile(
    r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]Z"
)
_EXTERNAL_REFERENCE_NAMESPACE = uuid.UUID("81c8f223-b8d8-5124-8ede-473b4d847f58")
_PRESENT_FALSE: dict[str, bool] = {"present": False}
_SIMILAR_RANKS = {
    "binary-exact": 10,
    "decoded-exact": 20,
    "reviewed-same-artwork": 30,
    "same-artwork": 40,
    "variant-of": 50,
    "related-series": 60,
}


def _failure(code: str, message: str, hint: str, **details: object) -> CommandError:
    return CommandError(code, message, hint=hint, details=dict(details) or None)


def _present(value: object | None = None, *, is_present: bool = False) -> dict[str, Any]:
    return {"present": True, "value": value} if is_present else {"present": False}


def _status(value: object, default: str = "unknown") -> str:
    if isinstance(value, dict):
        candidate = value.get("status")
        return candidate if isinstance(candidate, str) else default
    return value if isinstance(value, str) else default


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _mapping(value: object) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _registry_entries(document: dict[str, Any], *keys: str) -> list[dict[str, Any]]:
    for key in keys:
        value = document.get(key)
        if isinstance(value, list):
            return [cast(dict[str, Any], item) for item in value if isinstance(item, dict)]
    return []


def _jcs_sha256(value: object) -> str:
    try:
        return hashlib.sha256(rfc8785.dumps(cast(Any, value))).hexdigest()
    except (TypeError, ValueError) as exc:
        raise _failure(
            "INDEX_CORRUPT",
            "A snapshot record cannot be serialized as JCS.",
            "Restore the exact snapshot artifacts.",
        ) from exc


def _normalize_query(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _canonical_language(value: str) -> str:
    if "_" in value or not _BCP47_RE.fullmatch(value):
        raise _failure(
            "LANGUAGE_UNAVAILABLE",
            f"Invalid BCP 47 language tag: {value!r}",
            "Use a canonical tag such as en, ru, en-US, or ru-RU.",
        )
    parts = value.split("-")
    canonical = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            canonical.append(part.title())
        elif (len(part) == 2 and part.isalpha()) or (len(part) == 3 and part.isdigit()):
            canonical.append(part.upper())
        else:
            canonical.append(part.lower())
    return "-".join(canonical)


def _parse_instant(value: str) -> datetime:
    if not _CANONICAL_INSTANT_RE.fullmatch(value):
        raise _failure(
            "QUERY_INVALID",
            "--as-of must be a canonical UTC RFC 3339 instant.",
            "Use a value such as 2026-09-11T00:00:00Z.",
        )
    try:
        result = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise _failure(
            "QUERY_INVALID",
            "--as-of is not a valid RFC 3339 instant.",
            "Use a value such as 2026-09-11T00:00:00Z.",
        ) from exc
    if result.tzinfo is None or result.utcoffset() != UTC.utcoffset(result):
        raise _failure(
            "QUERY_INVALID", "--as-of must use the Z UTC suffix.", "Use a canonical UTC instant."
        )
    return result


@dataclass(slots=True)
class Page:
    items: list[Any]
    next_cursor: dict[str, Any]


class SnapshotReader:
    """Snapshot-bound service. It does not import network, AI, Telegram, or media modules."""

    def __init__(self, snapshot: LoadedSnapshot) -> None:
        self.snapshot = snapshot
        self._emoji_rows = snapshot.rows("emojis") if snapshot.has("emojis") else ()
        self._collection_rows = snapshot.rows("collections") if snapshot.has("collections") else ()
        self._membership_rows = snapshot.rows("memberships") if snapshot.has("memberships") else ()
        self._tombstone_rows = snapshot.rows("tombstones") if snapshot.has("tombstones") else ()
        self.emojis = {
            str(row.get("id")): row for row in self._emoji_rows if isinstance(row.get("id"), str)
        }
        self.collections = {
            str(row.get("id")): row
            for row in self._collection_rows
            if isinstance(row.get("id"), str)
        }
        self.tombstones = {
            str(row.get("target_id"))
            for row in self._tombstone_rows
            if isinstance(row.get("target_id"), str)
        }
        self.memberships_by_collection: dict[str, list[dict[str, Any]]] = {}
        self.memberships_by_emoji: dict[str, list[dict[str, Any]]] = {}
        for row in self._membership_rows:
            collection_id = row.get("collection_id")
            emoji_id = row.get("emoji_id")
            if isinstance(collection_id, str):
                self.memberships_by_collection.setdefault(collection_id, []).append(row)
            if isinstance(emoji_id, str):
                self.memberships_by_emoji.setdefault(emoji_id, []).append(row)

    def resolve_language(self, requested: str | None) -> tuple[str, str, bool]:
        requested_language = _canonical_language(requested or "en")
        artifact_languages = {
            logical[7:]
            for logical in self.snapshot.descriptors
            if logical.startswith("search-") and len(logical) > 7
        }
        languages = self.snapshot.manifest.get("languages")
        if not isinstance(languages, dict):
            raise _failure(
                "SCHEMA_UNSUPPORTED",
                "Snapshot language metadata is unavailable.",
                "Use a complete distribution-v1 snapshot.",
            )
        profile = self._bound_profile("language_fallback", "language-fallback-v1")
        if (
            languages.get("fallback_profile") != "language-fallback-v1"
            or profile.get("lookup_algorithm") != "rfc4647-lookup"
            or profile.get("machine_translation") != "forbidden"
        ):
            raise _failure(
                "SCHEMA_UNSUPPORTED",
                "The manifest-bound language fallback profile is unsupported.",
                "Use the exact language-fallback-v1 profile.",
            )
        declared = languages.get("available")
        if not isinstance(declared, list) or any(not isinstance(item, str) for item in declared):
            raise _failure(
                "SCHEMA_UNSUPPORTED",
                "Snapshot languages.available is invalid.",
                "Use a complete distribution-v1 snapshot.",
            )
        available = artifact_languages.intersection(cast(list[str], declared))
        chains = profile.get("chains")
        if not isinstance(chains, dict):
            raise _failure(
                "SCHEMA_UNSUPPORTED",
                "The language fallback profile has no closed chains map.",
                "Use the exact language-fallback-v1 profile.",
            )
        lookup = requested_language
        candidates: list[str] = []
        while lookup:
            chain = chains.get(lookup)
            if isinstance(chain, list):
                for candidate in chain:
                    if not isinstance(candidate, str):
                        raise _failure(
                            "SCHEMA_UNSUPPORTED",
                            "A language fallback chain contains a non-string value.",
                            "Use the exact language-fallback-v1 profile.",
                        )
                    canonical = _canonical_language(candidate)
                    if canonical not in candidates:
                        candidates.append(canonical)
                break
            if lookup not in candidates:
                candidates.append(lookup)
            lookup = lookup.rsplit("-", 1)[0] if "-" in lookup else ""
        for candidate in candidates:
            if candidate in available:
                return requested_language, candidate, candidate != requested_language
        raise _failure(
            "LANGUAGE_UNAVAILABLE",
            f"No search artifact is available for {requested_language}.",
            (
                "Select a language declared by this pinned snapshot; "
                "runtime AI translation is disabled."
            ),
            available_languages=sorted(available),
        )

    def _bound_profile(self, field: str, expected_id: str) -> dict[str, Any]:
        profile = self.snapshot.bound_document("profiles", field)
        if profile.get("profile_id") != expected_id:
            raise _failure(
                "SCHEMA_UNSUPPORTED",
                f"Unsupported manifest-bound {field} profile.",
                f"Use the exact {expected_id} profile.",
            )
        return profile

    def search(
        self,
        *,
        query: str,
        language: str | None,
        filters: dict[str, Any],
        sort: str,
        view: str,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            raise _failure(
                "QUERY_INVALID",
                "Search query must contain 1 to 4096 Unicode characters.",
                "Pass a bounded non-empty lexical query.",
            )
        requested, effective, fallback = self.resolve_language(language)
        rows = list(self.snapshot.rows(f"search-{effective}"))
        normalized = _normalize_query(query)
        scored: list[tuple[int, str, dict[str, Any]]] = []
        for row in rows:
            emoji_id = row.get("emoji_id")
            if not isinstance(emoji_id, str) or row.get("language") != effective:
                raise _failure(
                    "INDEX_CORRUPT",
                    f"Invalid row in search-{effective}.",
                    "Restore the exact search artifact.",
                )
            if not self._matches_filters(row, filters):
                continue
            if view == "agent" and not self._safe_agent_record(row):
                continue
            score = self._lexical_score(row, normalized, effective)
            if score <= 0:
                continue
            scored.append((score, emoji_id, row))
        if sort == "relevance":
            scored.sort(key=lambda item: (-item[0], item[1].encode("utf-8")))

            def tuple_for(item: tuple[int, str, dict[str, Any]]) -> list[object]:
                return [-item[0], item[1]]

        else:
            scored.sort(key=lambda item: item[1].encode("utf-8"))

            def tuple_for(item: tuple[int, str, dict[str, Any]]) -> list[object]:
                return [item[1]]

        arguments = {
            "query": query,
            "language": requested,
            "filters": filters,
            "sort": sort,
            "view": view,
            "limit": limit,
        }
        page = self._paginate("search", scored, tuple_for, arguments, cursor, limit)
        projected = [self._project_for_view(item[2], view, effective) for item in page.items]
        return {
            "requested_language": requested,
            "effective_language": effective,
            "fallback_used": fallback,
            "effective_filters": copy.deepcopy(filters),
            "items": projected,
            "next_cursor": page.next_cursor,
        }

    def _paginate(
        self,
        command: str,
        items: list[Any],
        tuple_for: Callable[[Any], list[object]],
        arguments: dict[str, Any],
        cursor: str | None,
        limit: int,
    ) -> Page:
        domain = pagination_domain(
            command=command, pinned_state=self.snapshot.pinned_state(), semantic_arguments=arguments
        )
        start = 0
        if cursor is not None:
            last = decode_cursor(cursor, domain)
            for index, item in enumerate(items):
                if tuple_for(item) == last:
                    start = index + 1
                    break
            else:
                raise _failure(
                    "CURSOR_INVALID",
                    "The cursor tuple is absent from this immutable result domain.",
                    "Restart pagination without --cursor.",
                )
        selected = items[start : start + limit]
        more = start + len(selected) < len(items)
        next_cursor = (
            _present(encode_cursor(domain, tuple_for(selected[-1])), is_present=True)
            if more and selected
            else {"present": False}
        )
        return Page(selected, next_cursor)

    def _search_rows(self, language: str) -> tuple[dict[str, Any], ...]:
        return self.snapshot.rows(f"search-{language}")

    def _search_row(self, emoji_id: str, language: str) -> dict[str, Any]:
        for row in self._search_rows(language):
            if row.get("emoji_id") == emoji_id:
                self._verify_join(row)
                return row
        raise _failure(
            "CONTENT_POLICY_BLOCKED",
            "No eligible search record exists for this emoji and language.",
            "Use canonical diagnostic view if policy permits.",
        )

    def _any_search_row(self, emoji_id: str) -> dict[str, Any] | None:
        for logical in sorted(self.snapshot.descriptors):
            if not logical.startswith("search-"):
                continue
            for row in self.snapshot.rows(logical):
                if row.get("emoji_id") == emoji_id:
                    self._verify_join(row)
                    return row
        return None

    def _canonical_emoji(self, emoji_id: str) -> dict[str, Any]:
        record = self.emojis.get(emoji_id)
        if record is None or emoji_id in self.tombstones:
            raise _failure(
                "ENTITY_NOT_FOUND",
                "Emoji was not found in the pinned snapshot.",
                "Check the MojiLex emoji ID and snapshot selection.",
                entity_id=emoji_id,
            )
        return record

    def _verify_join(self, row: dict[str, Any]) -> None:
        emoji_id = row.get("emoji_id")
        canonical = self.emojis.get(str(emoji_id))
        expected = row.get("canonical_record_sha256")
        if canonical is None or not isinstance(expected, str) or _jcs_sha256(canonical) != expected:
            raise _failure(
                "INDEX_CORRUPT",
                "Search-to-canonical record hash join failed.",
                "Restore artifacts from one exact snapshot.",
                emoji_id=emoji_id,
            )

    def _project_for_view(self, row: dict[str, Any], view: str, language: str) -> dict[str, Any]:
        self._verify_join(row)
        trust_source = row
        if view == "search":
            record = copy.deepcopy(row)
        elif view == "agent":
            if not self._safe_agent_record(row):
                raise _failure(
                    "CONTENT_POLICY_BLOCKED",
                    "A result is outside the safe agent subset.",
                    "Use an explicit broader canonical diagnostic request.",
                )
            record = self._agent_record(row, language)
        elif view == "canonical":
            canonical = self._canonical_emoji(str(row["emoji_id"]))
            record = copy.deepcopy(canonical)
            trust_source = canonical
        else:
            raise _failure(
                "FILTER_INVALID", f"Unsupported view: {view}", "Use canonical, search, or agent."
            )
        return {"record": record, "runtime_trust": self._runtime_trust(trust_source)}

    def _agent_record(self, row: dict[str, Any], language: str) -> dict[str, Any]:
        description = _mapping(row.get("description"))
        semantic = _mapping(row.get("semantic"))
        review = _mapping(row.get("review"))
        result: dict[str, Any] = {
            "record_schema_version": "1.0.0",
            "entity_type": "agent_record",
            "snapshot_id": self.snapshot.snapshot_id,
            "source_canonical_state_root_sha256": self.snapshot.source_canonical_state_root_sha256,
            "emoji_id": row["emoji_id"],
            "canonical_record_sha256": row["canonical_record_sha256"],
            "language": language,
            "platform": row.get("platform"),
            "description": description.get("text", ""),
            "motion": description.get("motion", ""),
            "usage": copy.deepcopy(description.get("usage", [])),
            "concept_ids": copy.deepcopy(semantic.get("concept_ids", [])),
            "facets": copy.deepcopy(row.get("facets", {})),
            "literal_text": copy.deepcopy(row.get("literal_text", [])),
            "availability": copy.deepcopy(row.get("availability", {})),
            "semantic_trust": self._semantic_trust(review),
            "content": copy.deepcopy(row.get("content", {})),
            "rights": copy.deepcopy(row.get("rights", {})),
            "collection_count": row.get("collection_count", 0),
            "collection_ids": copy.deepcopy(row.get("collection_ids", [])),
            "collections_truncated": row.get("collections_truncated", False),
            "duplicate_group_count": row.get("duplicate_group_count", 0),
            "duplicate_group_ids": copy.deepcopy(row.get("duplicate_group_ids", [])),
            "duplicate_groups_truncated": row.get("duplicate_groups_truncated", False),
            "native_reference_count": row.get("native_reference_count", 0),
            "native_references": copy.deepcopy(row.get("native_references", [])),
            "native_references_truncated": row.get("native_references_truncated", False),
            "platform_capability_refs": copy.deepcopy(row.get("platform_capability_refs", [])),
            "canonical_locator": copy.deepcopy(row.get("canonical_locator", {})),
            "visible_text_is_untrusted": True,
        }
        return result

    def _semantic_trust(self, review: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "review_status": review.get("status", "unreviewed"),
            "review_attested": bool(review.get("attested")),
            "provenance_origin": review.get("provenance_origin", "human"),
            "model_qualification_status": review.get(
                "model_qualification_status", "not-applicable"
            ),
            "model_qualification": copy.deepcopy(
                review.get("model_qualification", {"present": False})
            ),
            "generation_attestation_status": review.get(
                "generation_attestation_status", "not-applicable"
            ),
        }
        for key in ("review_attestation_id", "generation_attestation_id"):
            if key in review:
                result[key] = copy.deepcopy(review[key])
        return result

    def _runtime_trust(self, record: dict[str, Any]) -> dict[str, Any]:
        review = _mapping(record.get("review"))
        origin = review.get("provenance_origin", _mapping(record.get("provenance")).get("origin"))
        default_ai_status = "missing" if origin == "ai" else "not-applicable"
        review_status = _status(review, "unreviewed")
        attested = bool(review.get("attested"))
        if attested:
            attestation_status = "unverified"
            hash_profile_status = "valid"
        elif review_status == "unreviewed":
            attestation_status = "legacy-unattested"
            hash_profile_status = "not-applicable"
        elif review.get("review_hash_profile_id"):
            attestation_status = "legacy-unattested"
            hash_profile_status = "valid"
        else:
            attestation_status = "legacy-unprofiled"
            hash_profile_status = "legacy-unprofiled"
        result: dict[str, Any] = {
            "control_state_status": "offline-unknown",
            "review_attestation_status": attestation_status,
            "review_hash_profile_status": hash_profile_status,
            "model_qualification_status": review.get(
                "model_qualification_status", default_ai_status
            ),
            "generation_attestation_status": review.get(
                "generation_attestation_status", default_ai_status
            ),
            "safe_eligible": False,
        }
        for key in ("review_attestation_id", "generation_attestation_id", "model_qualification"):
            if key in review:
                result[key] = copy.deepcopy(review[key])
        return result

    def _safe_agent_record(self, row: dict[str, Any]) -> bool:
        dataset = self.snapshot.dataset_context()
        if not (
            dataset["release_verification_status"] == "verified"
            and dataset["catalog_status"] in {"current", "superseded"}
            and dataset["revocation_status"] == "clear"
            and dataset["control_state_status"] == "current"
            and dataset["errata_status"] in {"none", "applicable"}
        ):
            return False
        content = _mapping(row.get("content"))
        rights = _mapping(row.get("rights"))
        review = _mapping(row.get("review"))
        if _status(row.get("availability")) != "active" or content.get("rating") != "general":
            return False
        if _string_list(content.get("warnings")) or rights.get("distribution_status") != "allowed":
            return False
        if str(row.get("emoji_id")) in self.tombstones:
            return False
        review_status = review.get("status")
        approved = review_status == "approved" and bool(review.get("attested"))
        # A Stage-C verifier is required before qualified-AI can enter a safe
        # agent subset. Baked statuses in an unsigned snapshot are claims only.
        qualified_ai = False
        if not (approved or qualified_ai):
            return False
        return self._has_active_collection(str(row.get("emoji_id")))

    def _has_active_collection(self, emoji_id: str) -> bool:
        return bool(self._active_collection_ids(emoji_id))

    def _active_collection_ids(self, emoji_id: str) -> set[str]:
        result: set[str] = set()
        for membership in self.memberships_by_emoji.get(emoji_id, []):
            collection = self.collections.get(str(membership.get("collection_id")))
            if (
                _status(membership.get("status")) == "active"
                and str(membership.get("id")) not in self.tombstones
                and collection is not None
                and _status(collection.get("availability")) == "active"
                and str(collection.get("id")) not in self.tombstones
            ):
                result.add(str(collection["id"]))
        return result

    def _matches_filters(self, row: dict[str, Any], filters: dict[str, Any]) -> bool:
        canonical = self.emojis.get(str(row.get("emoji_id")))
        if canonical is None:
            raise _failure(
                "INDEX_CORRUPT",
                "A search row does not resolve to a canonical emoji.",
                "Restore artifacts from one exact snapshot.",
            )
        expected_rights = self._effective_rights_summary(
            str(canonical.get("platform", "")), require_generated_annotations=True
        )
        self._assert_search_rights(row, expected_rights)
        content = _mapping(row.get("content"))
        facets = _mapping(row.get("facets"))
        semantic = _mapping(row.get("semantic"))
        rights = _mapping(row.get("rights"))
        checks: list[bool] = [
            not filters.get("platform") or row.get("platform") in filters["platform"],
            not filters.get("availability")
            or _status(row.get("availability")) in filters["availability"],
            not filters.get("rating") or content.get("rating") in filters["rating"],
            not filters.get("rights") or rights.get("distribution_status") in filters["rights"],
            not filters.get("review") or self._review_matches(row, filters["review"]),
            not filters.get("exclude_warning")
            or not set(_string_list(content.get("warnings"))).intersection(
                filters["exclude_warning"]
            ),
            not filters.get("media_kind")
            or bool(
                set(_string_list(facets.get("media_kinds"))).intersection(filters["media_kind"])
            ),
            not filters.get("color_behavior")
            or bool(
                set(_string_list(facets.get("color_behaviors"))).intersection(
                    filters["color_behavior"]
                )
            ),
            not filters.get("color_family")
            or bool(
                set(_string_list(facets.get("color_families"))).intersection(
                    filters["color_family"]
                )
            ),
            not filters.get("content_type")
            or bool(
                set(_string_list(facets.get("content_types"))).intersection(filters["content_type"])
            ),
            not filters.get("style")
            or bool(set(_string_list(facets.get("styles"))).intersection(filters["style"])),
            not filters.get("suggested_use")
            or bool(
                set(_string_list(facets.get("suggested_uses"))).intersection(
                    filters["suggested_use"]
                )
            ),
            not filters.get("uncertainty")
            or bool(
                set(_string_list(facets.get("uncertainties"))).intersection(filters["uncertainty"])
            ),
            not filters.get("concept")
            or bool(
                set(_string_list(semantic.get("concept_ids"))).intersection(filters["concept"])
            ),
        ]
        collection_filter = filters.get("collection") or []
        if collection_filter:
            active_collections = self._active_collection_ids(str(row.get("emoji_id")))
            checks.append(bool(active_collections.intersection(collection_filter)))
        animated = filters.get("animated", "any")
        if animated != "any":
            checks.append(bool(facets.get("animated")) is (animated == "yes"))
        contains_text = filters.get("contains_text", "any")
        if contains_text != "any":
            checks.append(bool(facets.get("contains_text")) is (contains_text == "yes"))
        if filters.get("require_no_warnings", False):
            checks.append(not _string_list(content.get("warnings")))
        return all(checks)

    def _review_matches(self, row: dict[str, Any], accepted: list[str]) -> bool:
        review = _mapping(row.get("review"))
        if review.get("status") in accepted:
            return True
        # Stage C is intentionally absent from this MVP. Baked qualification
        # strings in an unsigned artifact are claims, not verified evidence.
        return False

    def _lexical_score(self, row: dict[str, Any], query: str, language: str) -> int:
        profile = self._bound_profile("lexical_search", "lexical-search-v1")
        if (
            profile.get("query_normalization") != "NFKC"
            or profile.get("case_folding") != "locale-aware"
            or profile.get("whitespace_policy") != "collapse"
            or profile.get("token_boundary_policy") != "unicode-punctuation-v1"
            or profile.get("tie_break") != "emoji-id-utf8-bytewise-ascending"
            or profile.get("stop_word_policy") != "none"
            or profile.get("stemming_policy") != "none"
            or profile.get("ai_translation") != "forbidden"
            or profile.get("embeddings") is not False
        ):
            raise _failure(
                "SCHEMA_UNSUPPORTED",
                "The manifest-bound lexical search algorithm is unsupported.",
                "Use the exact lexical-search-v1 profile.",
            )
        expected_rules = {
            "field_evaluation": "best-tier-per-field-v1",
            "exact": "normalized-full-value-v1",
            "phrase": "normalized-contiguous-substring-v1",
            "prefix": "normalized-token-prefix-v1",
            "token": "normalized-token-equality-v1",
            "concept_alias": "normalized-full-value-v1",
        }
        rules = profile.get("match_rules")
        scores = profile.get("match_scores")
        score_names = {
            "exact_mojilex_id",
            "exact_native_reference",
            "exact_concept_id",
            "field_exact_multiplier",
            "field_phrase_multiplier",
            "field_prefix_multiplier",
            "field_token_multiplier",
            "concept_alias_bonus_multiplier",
        }
        if (
            rules != expected_rules
            or not isinstance(scores, dict)
            or set(scores) != score_names
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in scores.values()
            )
        ):
            raise _failure(
                "SCHEMA_UNSUPPORTED",
                "The lexical search match rules or scores are unsupported.",
                "Use the exact lexical-search-v1 profile.",
            )
        score_values = cast(dict[str, int], scores)
        fields = profile.get("fields")
        if not isinstance(fields, list):
            raise _failure(
                "SCHEMA_UNSUPPORTED",
                "The lexical search profile has no fields array.",
                "Use the exact lexical-search-v1 profile.",
            )
        weights: dict[str, int] = {}
        for field in fields:
            name = field.get("field") if isinstance(field, dict) else None
            weight = field.get("weight") if isinstance(field, dict) else None
            if (
                not isinstance(name, str)
                or isinstance(weight, bool)
                or not isinstance(weight, int)
                or weight <= 0
                or name in weights
            ):
                raise _failure(
                    "SCHEMA_UNSUPPORTED",
                    "The lexical search field weights are invalid.",
                    "Use the exact lexical-search-v1 profile.",
                )
            weights[name] = weight
        expected_fields = {
            "description.text",
            "description.motion",
            "description.usage",
            "semantic.concept_ids",
            "semantic.semantic_tags",
            "literal_text.value",
        }
        if set(weights) != expected_fields:
            raise _failure(
                "SCHEMA_UNSUPPORTED",
                "The lexical search profile declares unsupported fields.",
                "Use the exact lexical-search-v1 field set.",
            )
        if _normalize_query(str(row.get("emoji_id", ""))) == query:
            return score_values["exact_mojilex_id"]
        native_ids = [
            str(item.get("native_id"))
            for item in row.get("native_references", [])
            if isinstance(item, dict)
        ]
        if query in {_normalize_query(value) for value in native_ids}:
            return score_values["exact_native_reference"]
        semantic = _mapping(row.get("semantic"))
        concepts = _string_list(semantic.get("concept_ids"))
        if query in {_normalize_query(value) for value in concepts}:
            return score_values["exact_concept_id"]
        description = _mapping(row.get("description"))
        field_values: dict[str, list[str]] = {
            "description.text": [str(description.get("text", ""))],
            "description.motion": [str(description.get("motion", ""))],
            "description.usage": _string_list(description.get("usage")),
            "semantic.concept_ids": list(concepts),
            "semantic.semantic_tags": _string_list(semantic.get("semantic_tags")),
            "literal_text.value": [],
        }
        for item in row.get("literal_text", []):
            if isinstance(item, dict) and isinstance(item.get("value"), str):
                field_values["literal_text.value"].append(str(item["value"]))
        aliases: list[str] = []
        if not self.snapshot.has("concepts"):
            raise _failure(
                "SCHEMA_UNSUPPORTED",
                "Concept labels and aliases are unavailable.",
                "Use a complete concepts-v1 snapshot.",
            )
        registry = self.snapshot.bound_document("profiles", "concepts")
        concept_rows = _registry_entries(registry, "concepts")
        by_id = {
            str(item.get("id")): item for item in concept_rows if isinstance(item.get("id"), str)
        }
        for concept_id in concepts:
            concept = by_id.get(concept_id)
            if concept is None:
                raise _failure(
                    "INDEX_CORRUPT",
                    "A search record references an unknown concept ID.",
                    "Restore artifacts from one exact snapshot.",
                    concept_id=concept_id,
                )
            labels = concept.get("labels")
            if isinstance(labels, dict) and isinstance(labels.get(language), str):
                field_values["semantic.concept_ids"].append(str(labels[language]))
            alias_map = concept.get("aliases")
            if isinstance(alias_map, dict):
                aliases.extend(_string_list(alias_map.get(language)))
        total = 0
        for field_name, values in field_values.items():
            best_tier = 0
            for raw_value in values:
                value = _normalize_query(raw_value)
                if value == query:
                    best_tier = max(best_tier, score_values["field_exact_multiplier"])
                elif query in value:
                    best_tier = max(best_tier, score_values["field_phrase_multiplier"])
                elif any(token.startswith(query) for token in re.findall(r"\w+", value)):
                    best_tier = max(best_tier, score_values["field_prefix_multiplier"])
                elif query in re.findall(r"\w+", value):
                    best_tier = max(best_tier, score_values["field_token_multiplier"])
            total += weights[field_name] * best_tier
        if any(_normalize_query(alias) == query for alias in aliases):
            total += (
                weights["semantic.concept_ids"] * score_values["concept_alias_bonus_multiplier"]
            )
        return total

    def _guard_content_rights(
        self,
        canonical: dict[str, Any],
        row: dict[str, Any] | None,
        include_sensitive: bool,
        *,
        require_generated_annotations: bool,
    ) -> None:
        content = _mapping(canonical.get("content"))
        if not include_sensitive and (
            content.get("rating") != "general" or _string_list(content.get("warnings"))
        ):
            raise _failure(
                "CONTENT_POLICY_BLOCKED",
                "Emoji content requires explicit sensitive-content opt-in.",
                "Rerun with --include-sensitive for permitted diagnostic use.",
            )
        expected = self._require_effective_rights(
            str(canonical.get("platform", "")),
            require_generated_annotations=require_generated_annotations,
        )
        if row is not None:
            self._assert_search_rights(row, expected)

    def _require_platform_metadata_rights(self, platform: str) -> None:
        self._require_effective_rights(platform, require_generated_annotations=False)

    def _assert_search_rights(self, row: dict[str, Any], expected: dict[str, Any]) -> None:
        rights = row.get("rights")
        if rights != expected:
            raise _failure(
                "INDEX_CORRUPT",
                "A search rights summary does not match canonical rights inputs.",
                "Restore artifacts from one exact snapshot.",
                emoji_id=row.get("emoji_id"),
            )

    def _require_effective_rights(
        self, platform: str, *, require_generated_annotations: bool
    ) -> dict[str, Any]:
        summary = self._effective_rights_summary(
            platform, require_generated_annotations=require_generated_annotations
        )
        if summary["distribution_status"] != "allowed":
            required = (
                "publish-metadata and publish-generated-annotations"
                if require_generated_annotations
                else "publish-metadata"
            )
            raise _failure(
                "RIGHTS_POLICY_BLOCKED",
                f"Effective rights do not allow {required}.",
                "Use a snapshot with an allowed active rights profile.",
            )
        return summary

    def _effective_rights_summary(
        self, platform: str, *, require_generated_annotations: bool
    ) -> dict[str, Any]:
        if self.snapshot.has("rights-assignments"):
            raise _failure(
                "SCHEMA_UNSUPPORTED",
                "This reader cannot safely evaluate explicit rights assignments.",
                "Use a reader with the exact rights-assignment evaluator.",
            )
        if not self.snapshot.has("platform-profiles") or not self.snapshot.has("rights-profiles"):
            raise _failure(
                "RIGHTS_POLICY_BLOCKED",
                "Platform rights cannot be resolved from this snapshot.",
                "Use a complete rights-v1 snapshot.",
            )
        platform_entries = _registry_entries(
            self.snapshot.document("platform-profiles"), "entries", "profiles", "platforms"
        )
        platform_matches = [item for item in platform_entries if item.get("platform") == platform]
        if len(platform_matches) != 1:
            raise _failure(
                "RIGHTS_POLICY_BLOCKED",
                "Platform rights selector is missing or ambiguous.",
                "Use a complete rights-v1 snapshot.",
            )
        platform_profile = platform_matches[0]
        has_default = isinstance(platform_profile.get("default_rights_profile_id"), str)
        inherits = platform_profile.get("inherit_project_default") is True
        if has_default == inherits:
            raise _failure(
                "RIGHTS_POLICY_BLOCKED",
                "Platform rights default selection is invalid.",
                "Use a complete rights-v1 snapshot.",
            )
        profile_id = platform_profile.get("default_rights_profile_id")
        rights_document = self.snapshot.document("rights-profiles")
        if inherits:
            profile_id = rights_document.get("project_default_profile_id")
        profiles = _registry_entries(rights_document, "profiles", "entries")
        matches = [item for item in profiles if item.get("rights_profile_id") == profile_id]
        if len(matches) != 1 or matches[0].get("status") != "active":
            raise _failure(
                "RIGHTS_POLICY_BLOCKED",
                "Effective rights profile is missing, ambiguous, or inactive.",
                "Use a complete rights-v1 snapshot.",
            )
        profile = matches[0]
        build_time = datetime.fromtimestamp(
            self.snapshot.manifest["build"]["source_date_epoch"], UTC
        )
        effective_from = datetime.fromisoformat(profile["effective_from"].replace("Z", "+00:00"))
        effective_until = (
            datetime.fromisoformat(profile["effective_until"].replace("Z", "+00:00"))
            if "effective_until" in profile
            else None
        )
        in_effective_interval = build_time >= effective_from and (
            effective_until is None or build_time < effective_until
        )
        operations = _mapping(profile.get("operations"))
        operation_names = ["publish-metadata"]
        if require_generated_annotations:
            operation_names.append("publish-generated-annotations")
        decisions = [
            _mapping(operations.get(name)).get("decision", "unknown") for name in operation_names
        ]
        if not in_effective_interval:
            distribution_status = "unknown"
        elif profile.get("withdrawn_from_distribution") is True:
            distribution_status = "withdrawn"
        elif any(decision in {"deny", "conditional", "not-granted"} for decision in decisions):
            distribution_status = "restricted"
        elif any(
            decision not in {"allow", "deny", "conditional", "not-granted"}
            for decision in decisions
        ):
            distribution_status = "unknown"
        elif all(decision == "allow" for decision in decisions):
            distribution_status = "allowed"
        else:
            distribution_status = "restricted"
        attribution_required = profile.get("attribution_required")
        if not isinstance(profile_id, str) or not isinstance(attribution_required, bool):
            raise _failure(
                "RIGHTS_POLICY_BLOCKED",
                "Effective rights profile has an invalid public summary.",
                "Use a complete rights-v1 snapshot.",
            )
        result: dict[str, Any] = {
            "rights_profile_id": profile_id,
            "distribution_status": distribution_status,
            "attribution_required": attribution_required,
        }
        if attribution_required:
            locator = profile.get("attribution_locator")
            if not isinstance(locator, str):
                raise _failure(
                    "RIGHTS_POLICY_BLOCKED",
                    "Attribution is required but its locator is unavailable.",
                    "Use a complete rights-v1 snapshot.",
                )
            result["attribution_locator"] = locator
        return result

    def _identity_references(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        logical = (
            "identity-references"
            if self.snapshot.has("identity-references")
            else "external-references"
            if self.snapshot.has("external-references")
            else None
        )
        if logical is not None:
            for row in self.snapshot.rows(logical):
                locator = _mapping(row.get("locator"))
                result.append(
                    {
                        "reference_id": row.get("id"),
                        "platform": row.get("platform"),
                        "native_namespace": locator.get("native_namespace"),
                        "scope_id": locator.get("scope_id"),
                        "native_id": str(locator.get("native_id")),
                        "identity_epoch": locator.get("identity_epoch"),
                        "reference_status": row.get("status"),
                        "reference_role": row.get("reference_role"),
                        "observed_from": row.get("observed_from"),
                        "observed_until": row.get("observed_until"),
                        "target_entity_type": row.get("target_entity_type"),
                        "target_id": row.get("target_id"),
                        "identity_attestation_id": row.get("identity_attestation_id"),
                        "identity_link_id": row.get("identity_link_id"),
                    }
                )
        else:
            for entity in (*self._emoji_rows, *self._collection_rows):
                fields = [
                    entity.get(key)
                    for key in (
                        "platform",
                        "native_namespace",
                        "scope_id",
                        "native_id",
                        "identity_epoch",
                    )
                ]
                if not all(
                    isinstance(item, (str, int)) and not isinstance(item, bool) for item in fields
                ):
                    continue
                observed_from = cast(dict[str, Any], entity.get("availability", {})).get(
                    "first_seen_at"
                )
                if not isinstance(observed_from, str):
                    continue
                components = [
                    "external-reference",
                    str(entity["platform"]),
                    str(entity["entity_type"]),
                    str(entity["id"]),
                    str(entity["native_namespace"]),
                    str(entity["scope_id"]),
                    str(entity["native_id"]),
                    str(entity["identity_epoch"]),
                    "primary",
                    observed_from,
                ]
                reference_id = "mxi_" + str(
                    uuid.uuid5(_EXTERNAL_REFERENCE_NAMESPACE, "\x00".join(components))
                )
                result.append(
                    {
                        "reference_id": reference_id,
                        "platform": entity["platform"],
                        "native_namespace": entity["native_namespace"],
                        "scope_id": entity["scope_id"],
                        "native_id": str(entity["native_id"]),
                        "identity_epoch": entity["identity_epoch"],
                        "reference_status": "current",
                        "reference_role": "primary",
                        "observed_from": observed_from,
                        "observed_until": None,
                        "target_entity_type": entity["entity_type"],
                        "target_id": entity["id"],
                        "identity_attestation_id": None,
                        "identity_link_id": None,
                    }
                )
        return result

    def _reference_contains(self, row: dict[str, Any], instant: datetime) -> bool:
        start = row.get("observed_from")
        if not isinstance(start, str) or _parse_instant(start) > instant:
            return False
        end = row.get("observed_until")
        return not isinstance(end, str) or instant < _parse_instant(end)

    def _reference_visible(self, row: dict[str, Any], include_history: bool) -> bool:
        target_id = str(row.get("target_id"))
        if target_id in self.tombstones:
            return False
        target = self.emojis.get(target_id) or self.collections.get(target_id)
        if target is None or _status(target.get("availability")) == "private":
            return False
        if _status(target.get("availability")) != "active" and not include_history:
            return False
        if row.get("reference_status") != "current" and not include_history:
            return False
        if target.get("entity_type") == "emoji":
            content = _mapping(target.get("content"))
            if content.get("rating") != "general" or _string_list(content.get("warnings")):
                return False
        # Resolve is metadata-only. Search rows summarize the stricter
        # metadata+annotation decision and are never the rights authority.
        self._require_platform_metadata_rights(str(target.get("platform", "")))
        return True

    def _resolution_candidate(self, row: dict[str, Any]) -> dict[str, Any]:
        target = (
            self.emojis.get(str(row["target_id"]))
            or self.collections.get(str(row["target_id"]))
            or {}
        )
        attestation = row.get("identity_attestation_id")
        link = row.get("identity_link_id")
        return {
            "reference_id": row["reference_id"],
            "identity_epoch": row["identity_epoch"],
            "reference_status": row["reference_status"],
            "reference_role": row["reference_role"],
            "observed_from": row["observed_from"],
            "observed_until": _present(
                row.get("observed_until"), is_present=isinstance(row.get("observed_until"), str)
            ),
            "target_entity_type": row["target_entity_type"],
            "target_id": row["target_id"],
            "path_evidence": {
                "identity_link_id": _present(link, is_present=isinstance(link, str)),
                "attestation_id": _present(attestation, is_present=isinstance(attestation, str)),
            },
            "runtime_trust": self._runtime_trust(target),
        }

    def _dedupe_profile(self) -> tuple[str, str]:
        profiles = self.snapshot.manifest.get("profiles")
        candidate = profiles.get("dedupe") if isinstance(profiles, dict) else None
        if (
            isinstance(candidate, dict)
            and isinstance(candidate.get("id"), str)
            and isinstance(candidate.get("sha256"), str)
        ):
            return str(candidate["id"]), str(candidate["sha256"])
        raise _failure(
            "SCHEMA_UNSUPPORTED",
            "Snapshot does not bind an exact dedupe evidence profile.",
            "Use a complete distribution-v1 manifest.",
        )

    def _suppressed_pairs(self) -> set[frozenset[str]]:
        result: set[frozenset[str]] = set()
        if not self.snapshot.has("visual-relations"):
            return result
        for relation in self.snapshot.rows("visual-relations"):
            if (
                relation.get("relation_type") == "not-duplicate"
                and _status(relation.get("review")) == "approved"
            ):
                subject, object_id = relation.get("subject_id"), relation.get("object_id")
                if isinstance(subject, str) and isinstance(object_id, str):
                    result.add(frozenset((subject, object_id)))
        return result

    def _keep_similarity(
        self,
        matches: dict[str, tuple[int, str, dict[str, Any]]],
        target_id: str,
        similarity: dict[str, Any],
        evidence_id: str,
    ) -> None:
        rank = int(similarity["match_rank"])
        current = matches.get(target_id)
        if current is None or (rank, evidence_id.encode("utf-8")) < (
            current[0],
            current[1].encode("utf-8"),
        ):
            matches[target_id] = (rank, evidence_id, similarity)

    def _safe_target(self, emoji_id: str, view: str) -> bool:
        row = self._any_search_row(emoji_id)
        if row is None:
            return False
        if view == "agent":
            return self._safe_agent_record(row)
        content = row.get("content")
        canonical = self.emojis.get(emoji_id)
        if canonical is None:
            raise _failure(
                "INDEX_CORRUPT",
                "A similarity target does not resolve to a canonical emoji.",
                "Restore artifacts from one exact snapshot.",
            )
        expected_rights = self._require_effective_rights(
            str(canonical.get("platform", "")),
            require_generated_annotations=view != "canonical",
        )
        if view != "canonical":
            self._assert_search_rights(row, expected_rights)
        return (
            _status(row.get("availability")) == "active"
            and isinstance(content, dict)
            and content.get("rating") == "general"
            and not _string_list(content.get("warnings"))
            and self._has_active_collection(emoji_id)
        )

    def _target_for_view(self, emoji_id: str, view: str, language: str | None) -> dict[str, Any]:
        if view == "canonical":
            canonical = self._canonical_emoji(emoji_id)
            self._guard_content_rights(
                canonical,
                None,
                False,
                require_generated_annotations=False,
            )
            return {
                "record": copy.deepcopy(canonical),
                "runtime_trust": self._runtime_trust(canonical),
            }
        if language is None:
            raise _failure(
                "LANGUAGE_UNAVAILABLE",
                "A language is required for this view.",
                "Pass --language or use the en default.",
            )
        return self._project_for_view(self._search_row(emoji_id, language), view, language)

    def get(
        self,
        emoji_id: str,
        *,
        view: str,
        language: str | None,
        include_sensitive: bool,
    ) -> dict[str, Any]:
        if re.fullmatch(r"[0-9]{1,20}", emoji_id):
            resolved = self.resolve(
                platform="telegram",
                namespace="custom_emoji.id",
                scope="global",
                native_id=emoji_id,
                identity_epoch=None,
                as_of=None,
                include_history=False,
                limit=2,
                cursor=None,
            )
            candidates = resolved["candidates"]
            if len(candidates) != 1 or candidates[0]["target_entity_type"] != "emoji":
                raise _failure(
                    "ENTITY_NOT_FOUND",
                    local_text(
                        "Telegram emoji ID was not found in this snapshot.",
                        "Telegram ID эмодзи не найден в этом снимке.",
                    ),
                    local_text(
                        "Check the ID and local snapshot. Import/AI drafts are not automatically "
                        "included in release snapshots.",
                        "Проверьте ID и выбранный снимок. "
                        "Импорт и AI-черновик не добавляются в релиз автоматически.",
                    ),
                )
            emoji_id = candidates[0]["target_id"]
        canonical = self._canonical_emoji(emoji_id)
        language_component: dict[str, Any] = {"present": False}
        if view == "canonical":
            self._guard_content_rights(
                canonical,
                None,
                include_sensitive,
                require_generated_annotations=False,
            )
            item = {
                "record": copy.deepcopy(canonical),
                "runtime_trust": self._runtime_trust(canonical),
            }
        else:
            requested, effective, fallback = self.resolve_language(language)
            language_component = _present(
                {
                    "requested_language": requested,
                    "effective_language": effective,
                    "fallback_used": fallback,
                },
                is_present=True,
            )
            row = self._search_row(emoji_id, effective)
            self._guard_content_rights(
                canonical,
                row,
                include_sensitive,
                require_generated_annotations=True,
            )
            if view == "agent" and not self._safe_agent_record(row):
                raise _failure(
                    "CONTENT_POLICY_BLOCKED",
                    "The emoji is outside the safe agent subset.",
                    "Use canonical diagnostic view with explicit content opt-in when appropriate.",
                )
            item = self._project_for_view(row, view, effective)
        return {"item": item, "view": view, "language": language_component}

    def get_collection(
        self,
        collection_id: str,
        *,
        include_history: bool,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        collection = self.collections.get(collection_id)
        if collection is None or collection_id in self.tombstones:
            raise _failure(
                "ENTITY_NOT_FOUND",
                "Collection was not found in the pinned snapshot.",
                "Check the collection ID and snapshot selection.",
            )
        collection_status = _status(collection.get("availability"))
        if collection_status != "active" and not include_history:
            raise _failure(
                "ENTITY_NOT_FOUND",
                "Collection is not active in the pinned snapshot.",
                "Use --include-history for permitted diagnostic history.",
            )
        if collection_status == "private":
            raise _failure(
                "CONTENT_POLICY_BLOCKED",
                "Collection metadata is not publishable.",
                "Do not disclose private collection metadata.",
            )
        self._require_platform_metadata_rights(str(collection.get("platform", "")))
        candidates: list[dict[str, Any]] = []
        for membership in self.memberships_by_collection.get(collection_id, []):
            membership_id = membership.get("id")
            emoji_id = membership.get("emoji_id")
            if not isinstance(membership_id, str) or not isinstance(emoji_id, str):
                raise _failure(
                    "INDEX_CORRUPT",
                    "Membership record lacks canonical identity fields.",
                    "Restore the memberships artifact.",
                )
            if membership_id in self.tombstones or emoji_id in self.tombstones:
                continue
            if _status(membership.get("status")) != "active" and not include_history:
                continue
            target = self.emojis.get(emoji_id)
            if target is None or _status(target.get("availability")) == "private":
                continue
            if _status(target.get("availability")) != "active" and not include_history:
                continue
            self._require_platform_metadata_rights(str(target.get("platform", "")))
            content = cast(dict[str, Any], target.get("content", {}))
            if content.get("rating") != "general" or _string_list(content.get("warnings")):
                continue
            candidates.append(membership)
        candidates.sort(
            key=lambda row: (int(row.get("position", 0)), str(row.get("id", "")).encode("utf-8"))
        )
        arguments = {
            "collection_id": collection_id,
            "include_history": include_history,
            "limit": limit,
        }
        page = self._paginate(
            "get-collection",
            candidates,
            lambda row: [int(row.get("position", 0)), str(row.get("id", ""))],
            arguments,
            cursor,
            limit,
        )
        return {
            "collection": {
                "record": copy.deepcopy(collection),
                "runtime_trust": self._runtime_trust(collection),
            },
            "memberships": [
                {"record": copy.deepcopy(row), "runtime_trust": self._runtime_trust(row)}
                for row in page.items
            ],
            "next_cursor": page.next_cursor,
        }

    def resolve(
        self,
        *,
        platform: str,
        namespace: str,
        scope: str,
        native_id: str,
        identity_epoch: int | None,
        as_of: str | None,
        include_history: bool,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        rows = self._identity_references()
        matches = [
            row
            for row in rows
            if row["platform"] == platform
            and row["native_namespace"] == namespace
            and row["scope_id"] == scope
            and row["native_id"] == native_id
        ]
        if identity_epoch is not None:
            matches = [row for row in matches if row["identity_epoch"] == identity_epoch]
        if as_of is not None:
            instant = _parse_instant(as_of)
            build = self.snapshot.manifest.get("build")
            source_epoch = build.get("source_date_epoch") if isinstance(build, dict) else None
            if isinstance(source_epoch, int) and instant.timestamp() > source_epoch:
                raise _failure(
                    "QUERY_INVALID",
                    "--as-of is later than the pinned snapshot build instant.",
                    "Choose an instant at or before manifest.build.source_date_epoch.",
                )
            matches = [row for row in matches if self._reference_contains(row, instant)]
        elif identity_epoch is None:
            matches = [row for row in matches if row["reference_status"] == "current"]
        visible = [row for row in matches if self._reference_visible(row, include_history)]
        if as_of is None and identity_epoch is None and len(visible) > 1:
            raise _failure(
                "AMBIGUOUS_REFERENCE",
                "Multiple permitted current identities match this native locator.",
                "Pass an exact --identity-epoch; automatic epoch selection is forbidden.",
                candidate_count=len(visible),
            )
        if not visible:
            return {
                "resolution_status": "not_found",
                "resolution_path": "none",
                "candidate_count": 0,
                "candidates": [],
                "next_cursor": {"present": False},
            }
        visible.sort(
            key=lambda row: (
                row["identity_epoch"],
                row["reference_id"].encode("utf-8"),
                row["target_id"].encode("utf-8"),
            )
        )
        if len(visible) > 1:
            status, path = "ambiguous", "none"
        else:
            status = "current" if visible[0]["reference_status"] == "current" else "former"
            path = "alias" if visible[0]["reference_role"] == "alias" else "direct"
        arguments = {
            "platform": platform,
            "namespace": namespace,
            "scope": scope,
            "native_id": native_id,
            "identity_epoch": identity_epoch,
            "as_of": as_of,
            "include_history": include_history,
            "limit": limit,
        }
        page = self._paginate(
            "resolve",
            visible,
            lambda row: [row["identity_epoch"], row["reference_id"], row["target_id"]],
            arguments,
            cursor,
            limit,
        )
        return {
            "resolution_status": status,
            "resolution_path": path,
            "candidate_count": len(visible),
            "candidates": [self._resolution_candidate(row) for row in page.items],
            "next_cursor": page.next_cursor,
        }

    def similar(
        self,
        emoji_id: str,
        *,
        view: str,
        language: str | None,
        limit: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        self._canonical_emoji(emoji_id)
        language_component: dict[str, Any] = {"present": False}
        effective: str | None = None
        if view != "canonical":
            requested, effective, fallback = self.resolve_language(language)
            language_component = _present(
                {
                    "requested_language": requested,
                    "effective_language": effective,
                    "fallback_used": fallback,
                },
                is_present=True,
            )
        suppressed = self._suppressed_pairs()
        evidence_profile_id, evidence_profile_sha256 = self._dedupe_profile()
        matches: dict[str, tuple[int, str, dict[str, Any]]] = {}
        groups = {
            str(row.get("group_id")): row
            for row in (
                self.snapshot.rows("duplicate-groups")
                if self.snapshot.has("duplicate-groups")
                else ()
            )
            if isinstance(row.get("group_id"), str)
        }
        memberships = (
            list(self.snapshot.rows("duplicate-group-memberships"))
            if self.snapshot.has("duplicate-group-memberships")
            else []
        )
        source_groups = {
            str(row.get("group_id")) for row in memberships if row.get("emoji_id") == emoji_id
        }
        for member in memberships:
            target_id = member.get("emoji_id")
            group_id = member.get("group_id")
            if (
                not isinstance(target_id, str)
                or target_id == emoji_id
                or group_id not in source_groups
            ):
                continue
            group = groups.get(str(group_id))
            group_type = group.get("group_type") if group else None
            if group_type not in {"binary-exact", "decoded-exact"}:
                continue
            assert isinstance(group_type, str)
            similarity = {
                "match_type": group_type,
                "match_rank": _SIMILAR_RANKS[group_type],
                "group_id": _present(str(group_id), is_present=True),
                "relation_id": {"present": False},
                "evidence_profile_id": evidence_profile_id,
                "evidence_profile_sha256": evidence_profile_sha256,
                "relation_review": {
                    "review_attestation_id": {"present": False},
                    "status": "not-applicable",
                },
            }
            self._keep_similarity(matches, target_id, similarity, str(group_id))
        ordered: list[tuple[int, str, str, dict[str, Any]]] = []
        for target_id, (rank, evidence_id, similarity) in matches.items():
            if frozenset((emoji_id, target_id)) in suppressed or not self._safe_target(
                target_id, view
            ):
                continue
            ordered.append((rank, target_id, evidence_id, similarity))
        ordered.sort(key=lambda item: (item[0], item[1].encode("utf-8"), item[2].encode("utf-8")))
        arguments = {
            "emoji_id": emoji_id,
            "view": view,
            "language": effective,
            "limit": limit,
            "policy": "confirmed-only-v1",
        }
        page = self._paginate(
            "similar", ordered, lambda item: [item[0], item[1], item[2]], arguments, cursor, limit
        )
        result_items: list[dict[str, Any]] = []
        for _, target_id, _, similarity in page.items:
            wrapper = self._target_for_view(target_id, view, effective)
            wrapper["similarity"] = copy.deepcopy(similarity)
            result_items.append(wrapper)
        return {
            "source_emoji_id": emoji_id,
            "view": view,
            "language": language_component,
            "items": result_items,
            "next_cursor": page.next_cursor,
        }
