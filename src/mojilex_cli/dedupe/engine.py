"""Scalable exact grouping and bounded perceptual candidate generation."""

from __future__ import annotations

import base64
import hashlib
import itertools
import math
import statistics
import struct
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from functools import lru_cache
from typing import Any, Literal

import rfc8785

from mojilex_cli.analysis import load_analysis_profile
from mojilex_cli.dataset.repository import DatasetSnapshot


@dataclass(frozen=True, slots=True)
class DedupeCandidate:
    emoji_id: str
    against_emoji_id: str
    role: str
    variant_id: str | None
    against_role: str
    against_variant_id: str | None
    signals: tuple[str, ...]
    primary_distance: int
    alpha_distance: int
    edge_distance: int
    p90_distance: int
    speed_ratio: str
    duration_compatible: bool
    text_match: bool
    text_mismatch: bool
    high_priority: bool


@dataclass(frozen=True, slots=True)
class CollectionCandidate:
    collection_id: str
    against_collection_id: str
    candidate_type: Literal["clone-candidate", "subset-candidate"]
    matched_count: int
    collection_size: int
    against_collection_size: int
    collection_coverage: str
    against_collection_coverage: str


@dataclass(frozen=True, slots=True)
class DedupeScanReport:
    profile: str
    scan_mode: Literal["exact", "near"]
    exact_groups: tuple[dict[str, Any], ...]
    candidates: dict[str, tuple[DedupeCandidate, ...]]
    candidate_overflow: dict[str, bool]
    candidate_count_before_limit: dict[str, int]
    suppressed_bucket_count: int
    comparisons: int
    collection_candidates: tuple[CollectionCandidate, ...] = ()
    candidate_neighbors: dict[str, frozenset[str]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "scan_mode": self.scan_mode,
            "exact_groups": list(self.exact_groups),
            "candidates": {
                key: [asdict(item) for item in value]
                for key, value in sorted(self.candidates.items())
            },
            "candidate_overflow": dict(sorted(self.candidate_overflow.items())),
            "candidate_count_before_limit": dict(sorted(self.candidate_count_before_limit.items())),
            "suppressed_bucket_count": self.suppressed_bucket_count,
            "comparisons": self.comparisons,
            "collection_candidates": [asdict(item) for item in self.collection_candidates],
        }


@dataclass(frozen=True, slots=True)
class _Ref:
    emoji_id: str
    profile_id: str
    role: str
    variant_id: str | None
    media_sha256: str
    byte_size: int
    duration_ms: int
    animated: bool
    alpha_mode: str
    color_behavior: str
    decoded_sha256: str
    canonical_sha256: str
    shape_sha256: str
    low_information: bool
    layout: tuple[int, ...]
    content: tuple[int, ...]
    alpha: tuple[int, ...]
    edge: tuple[int, ...]

    @property
    def media_key(self) -> tuple[str, str]:
        return self.role, self.variant_id or ""


@dataclass(slots=True)
class _CandidateIndexStats:
    """Bounded diagnostics for the production candidate index."""

    bucket_entries: int = 0
    bucket_probes: int = 0
    candidate_hits: int = 0
    yielded_pairs: int = 0
    low_information_refs: int = 0


@dataclass(slots=True)
class _CandidateIndex:
    refs: Sequence[_Ref]
    lsh_buckets: Mapping[tuple[str, int, int], Sequence[int]]
    lsh_values: Mapping[tuple[str, int], tuple[int, ...]]
    canonical_buckets: Mapping[tuple[str, str], Sequence[int]]
    shape_buckets: Mapping[str, Sequence[int]]
    shape_secondary_buckets: Mapping[tuple[str, str, int], Sequence[int]]
    oversized_shapes: frozenset[str]
    secondary_keys: tuple[str, ...]
    thresholds: Mapping[str, Any]
    band_bits: int
    suppressed_bucket_count: int
    stats: _CandidateIndexStats

    def pairs(self) -> Iterator[tuple[int, int]]:
        """Yield candidate pairs without retaining a dataset-wide pair set."""

        bands = 64 // self.band_bits
        static_threshold = max(
            int(self.thresholds["static_primary_hamming_max"]),
            int(self.thresholds["adaptive_alpha_hamming_max"]),
            int(self.thresholds["adaptive_edge_hamming_max"]),
        )
        animation_threshold = int(self.thresholds["animation_median_hamming_max"])
        for index, ref in enumerate(self.refs):
            neighbors: set[int] = set()
            if not ref.low_information:
                threshold = animation_threshold if ref.animated else static_threshold
                masks = _probe_masks(self.band_bits, threshold // bands)
                for label, band, value in _lsh_keys(ref, band_bits=self.band_bits):
                    self._add_lsh_neighbors(
                        neighbors,
                        index=index,
                        label=label,
                        band=band,
                        value=value,
                        masks=masks,
                        radius=threshold // bands,
                    )

            self._add_bucket_neighbors(
                neighbors,
                index=index,
                members=self.canonical_buckets.get(("canonical", ref.canonical_sha256), ()),
            )
            if not ref.low_information:
                direct_shape = self.shape_buckets.get(ref.shape_sha256)
                if direct_shape is not None:
                    self._add_bucket_neighbors(
                        neighbors,
                        index=index,
                        members=direct_shape,
                    )
                elif ref.shape_sha256 in self.oversized_shapes:
                    sequences = dict(_hash_sequences(ref))
                    for specification in self.secondary_keys:
                        label_with_suffix, _, raw_width = specification.rsplit("-", 2)
                        label = label_with_suffix.removesuffix("-phash")
                        width = int(raw_width)
                        prefix = _majority_hash(sequences[label]) >> (64 - width)
                        self._add_bucket_neighbors(
                            neighbors,
                            index=index,
                            members=self.shape_secondary_buckets.get(
                                (ref.shape_sha256, specification, prefix), ()
                            ),
                        )

            for other in sorted(neighbors):
                candidate = self.refs[other]
                if (
                    ref.emoji_id != candidate.emoji_id
                    and ref.animated == candidate.animated
                    and ref.profile_id == candidate.profile_id
                ):
                    self.stats.yielded_pairs += 1
                    yield index, other

    def _add_lsh_neighbors(
        self,
        neighbors: set[int],
        *,
        index: int,
        label: str,
        band: int,
        value: int,
        masks: Sequence[int],
        radius: int,
    ) -> None:
        active_values = self.lsh_values.get((label, band), ())
        if len(active_values) < len(masks):
            for candidate_value in active_values:
                self.stats.bucket_probes += 1
                if (value ^ candidate_value).bit_count() > radius:
                    continue
                self._add_bucket_neighbors(
                    neighbors,
                    index=index,
                    members=self.lsh_buckets[(label, band, candidate_value)],
                )
            return
        for mask in masks:
            self.stats.bucket_probes += 1
            self._add_bucket_neighbors(
                neighbors,
                index=index,
                members=self.lsh_buckets.get((label, band, value ^ mask), ()),
            )

    def _add_bucket_neighbors(
        self,
        neighbors: set[int],
        *,
        index: int,
        members: Sequence[int],
    ) -> None:
        self.stats.candidate_hits += len(members)
        neighbors.update(other for other in members if other > index)


def scan_snapshot(
    snapshot: DatasetSnapshot,
    *,
    selected_emoji_ids: set[str] | None = None,
    max_candidates: int | None = None,
    mode: Literal["exact", "near"] = "near",
) -> DedupeScanReport:
    """Scan without an all-pairs pass; candidate buckets have a hard ceiling."""

    profile = load_analysis_profile("dedupe-v1")
    thresholds = profile.data["candidate_thresholds"]
    bucket_policy = profile.data["oversized_bucket_policy"]
    default_limit = int(thresholds["candidate_limit_default"])
    hard_limit = int(thresholds["candidate_limit_hard_max"])
    limit = default_limit if max_candidates is None else max_candidates
    if not 1 <= limit <= hard_limit:
        raise ValueError(f"max dedupe candidates must be in the range 1..{hard_limit}")
    refs = _references(snapshot)
    exact_groups = _exact_groups(snapshot, refs)
    collection_candidates = _collection_candidates(
        snapshot,
        exact_groups,
        selected_emoji_ids=selected_emoji_ids,
    )
    if mode == "exact":
        if selected_emoji_ids is not None:
            exact_groups = [
                group
                for group in exact_groups
                if any(
                    _group_member_emoji_id(member) in selected_emoji_ids
                    for member in group["members"]
                )
            ]
        return DedupeScanReport(
            profile=profile.profile_id,
            scan_mode="exact",
            exact_groups=tuple(exact_groups),
            candidates={},
            candidate_overflow={},
            candidate_count_before_limit={},
            suppressed_bucket_count=0,
            comparisons=0,
            collection_candidates=collection_candidates,
        )
    suppressed = _approved_not_duplicate_pairs(snapshot)
    bucket_limit = int(bucket_policy["posting_bucket_cap"])
    secondary_keys = tuple(str(value) for value in bucket_policy["secondary_keys"])
    band_bits = _lsh_band_bits(secondary_keys)
    candidate_index = _build_candidate_index(
        refs,
        thresholds=thresholds,
        bucket_limit=bucket_limit,
        secondary_keys=secondary_keys,
        band_bits=band_bits,
    )

    by_emoji: dict[str, list[DedupeCandidate]] = defaultdict(list)
    comparisons = 0
    for left_index, right_index in candidate_index.pairs():
        left_ref, right_ref = refs[left_index], refs[right_index]
        if selected_emoji_ids is not None and not (
            left_ref.emoji_id in selected_emoji_ids or right_ref.emoji_id in selected_emoji_ids
        ):
            continue
        if left_ref.profile_id != right_ref.profile_id:
            continue
        endpoint_pair = tuple(sorted((left_ref.emoji_id, right_ref.emoji_id)))
        if endpoint_pair in suppressed:
            continue
        comparisons += 1
        candidate = _compare(left_ref, right_ref, snapshot, thresholds)
        if candidate is None:
            continue
        reverse = DedupeCandidate(
            emoji_id=candidate.against_emoji_id,
            against_emoji_id=candidate.emoji_id,
            role=candidate.against_role,
            variant_id=candidate.against_variant_id,
            against_role=candidate.role,
            against_variant_id=candidate.variant_id,
            signals=candidate.signals,
            primary_distance=candidate.primary_distance,
            alpha_distance=candidate.alpha_distance,
            edge_distance=candidate.edge_distance,
            p90_distance=candidate.p90_distance,
            speed_ratio=candidate.speed_ratio,
            duration_compatible=candidate.duration_compatible,
            text_match=candidate.text_match,
            text_mismatch=candidate.text_mismatch,
            high_priority=candidate.high_priority,
        )
        by_emoji[candidate.emoji_id].append(candidate)
        by_emoji[reverse.emoji_id].append(reverse)

    targets = (
        set(by_emoji) if selected_emoji_ids is None else selected_emoji_ids & set(snapshot.emojis)
    )
    output: dict[str, tuple[DedupeCandidate, ...]] = {}
    overflows: dict[str, bool] = {}
    counts: dict[str, int] = {}
    candidate_neighbors = {
        emoji_id: frozenset(candidate.against_emoji_id for candidate in candidates)
        for emoji_id, candidates in by_emoji.items()
    }
    for emoji_id in sorted(targets):
        ranked_candidates = sorted(by_emoji.get(emoji_id, ()), key=_candidate_sort_key)
        counts[emoji_id] = len(ranked_candidates)
        overflows[emoji_id] = len(ranked_candidates) > limit
        output[emoji_id] = tuple(ranked_candidates[:limit])
    return DedupeScanReport(
        profile=profile.profile_id,
        scan_mode="near",
        exact_groups=tuple(exact_groups),
        candidates=output,
        candidate_overflow=overflows,
        candidate_count_before_limit=counts,
        suppressed_bucket_count=candidate_index.suppressed_bucket_count,
        comparisons=comparisons,
        collection_candidates=collection_candidates,
        candidate_neighbors=candidate_neighbors,
    )


def explain_pair(
    snapshot: DatasetSnapshot,
    emoji_id: str,
    against_emoji_id: str,
) -> dict[str, Any]:
    """Explain exact, perceptual, text, and human-decision evidence for one pair."""

    if emoji_id == against_emoji_id:
        raise ValueError("dedupe explanation requires two distinct emoji IDs")
    if emoji_id not in snapshot.emojis or against_emoji_id not in snapshot.emojis:
        raise KeyError("dedupe explanation endpoint is not present in the dataset")
    left_emoji = snapshot.emojis[emoji_id]
    right_emoji = snapshot.emojis[against_emoji_id]
    profiles_compatible = left_emoji.fingerprints.profile == right_emoji.fingerprints.profile
    left_refs = [value for value in _references(snapshot) if value.emoji_id == emoji_id]
    right_refs = [value for value in _references(snapshot) if value.emoji_id == against_emoji_id]
    thresholds = load_analysis_profile("dedupe-v1").data["candidate_thresholds"]
    comparisons: list[dict[str, Any]] = []
    candidates: list[DedupeCandidate] = []
    for left_ref in left_refs:
        for right_ref in right_refs:
            binary_exact = (
                left_ref.media_sha256 == right_ref.media_sha256
                and left_ref.byte_size == right_ref.byte_size
            )
            decoded_exact = left_ref.decoded_sha256 == right_ref.decoded_sha256
            candidate = (
                _compare(left_ref, right_ref, snapshot, thresholds) if profiles_compatible else None
            )
            if candidate is not None:
                candidates.append(candidate)
            comparisons.append(
                {
                    "subject": {"role": left_ref.role, "variant_id": left_ref.variant_id},
                    "object": {"role": right_ref.role, "variant_id": right_ref.variant_id},
                    "binary_exact": binary_exact,
                    "decoded_exact": decoded_exact,
                    "canonical_render_match": (
                        left_ref.canonical_sha256 == right_ref.canonical_sha256
                    ),
                    "shape_match": left_ref.shape_sha256 == right_ref.shape_sha256,
                    "animated_compatible": left_ref.animated == right_ref.animated,
                    "candidate": asdict(candidate) if candidate is not None else None,
                }
            )
    decisions = sorted(
        (
            relation.as_dict()
            for relation in snapshot.relations.values()
            if {relation.subject_id, relation.object_id} == {emoji_id, against_emoji_id}
        ),
        key=lambda value: (value["scope"] != "entity", value["id"]),
    )
    decision = decisions[0] if decisions else None
    exclusions: list[str] = []
    if not left_refs or not right_refs:
        exclusions.append("missing-active-complete-fingerprint")
    if not profiles_compatible:
        exclusions.append("profile-mismatch")
    if comparisons and not any(value["animated_compatible"] for value in comparisons):
        exclusions.append("static-animation-mismatch")
    if not candidates and not exclusions:
        exclusions.append("near-thresholds-not-met")
    if tuple(sorted((emoji_id, against_emoji_id))) in _approved_not_duplicate_pairs(snapshot):
        exclusions.append("approved-not-duplicate")
    best = min(candidates, key=_candidate_sort_key) if candidates else None
    return {
        "emoji_id": emoji_id,
        "against_emoji_id": against_emoji_id,
        "profile": left_emoji.fingerprints.profile,
        "against_profile": right_emoji.fingerprints.profile,
        "profile_compatible": profiles_compatible,
        "input_media_digest_match": (
            left_emoji.fingerprints.input_media_digest
            == right_emoji.fingerprints.input_media_digest
        ),
        "literal_text": list(_literal_text(snapshot, emoji_id)),
        "against_literal_text": list(_literal_text(snapshot, against_emoji_id)),
        "comparisons": comparisons,
        "best_candidate": asdict(best) if best is not None else None,
        "human_decision": decision,
        "human_decisions": decisions,
        "exclusion_reasons": sorted(set(exclusions)),
    }


def best_cyclic_alignment(
    snapshot: DatasetSnapshot,
    emoji_id: str,
    against_emoji_id: str,
) -> tuple[int, int]:
    """Return best right-side layout shift and normative sample count."""

    refs = _references(snapshot)
    left = next((value for value in refs if value.emoji_id == emoji_id), None)
    right = next((value for value in refs if value.emoji_id == against_emoji_id), None)
    if left is None or right is None or not left.animated or not right.animated:
        return 0, 1
    thresholds = load_analysis_profile("dedupe-v1").data["candidate_thresholds"]
    if not bool(thresholds["cyclic_alignment"]):
        return 0, len(right.layout) or 1
    if len(left.layout) != len(right.layout) or not left.layout:
        return 0, max(len(left.layout), len(right.layout), 1)
    ranked: list[tuple[int, int]] = []
    for shift in range(len(right.layout)):
        shifted = right.layout[shift:] + right.layout[:shift]
        ranked.append(
            (sum((a ^ b).bit_count() for a, b in zip(left.layout, shifted, strict=True)), shift)
        )
    _, shift = min(ranked)
    return shift, len(right.layout)


def _collection_candidates(
    snapshot: DatasetSnapshot,
    exact_groups: list[dict[str, Any]],
    *,
    selected_emoji_ids: set[str] | None,
) -> tuple[CollectionCandidate, ...]:
    """Score only collection pairs sharing a proven-equivalence component."""

    active_emojis = {
        emoji.id
        for emoji in snapshot.emojis.values()
        if emoji.availability.status.value == "active"
        and emoji.fingerprints.status.value == "complete"
    }
    parent = {emoji_id: emoji_id for emoji_id in active_emojis}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parent[second] = first

    for group in exact_groups:
        if group["scope"] != "entity":
            continue
        members = [
            _group_member_emoji_id(member)
            for member in group["members"]
            if _group_member_emoji_id(member) in parent
        ]
        for member in members[1:]:
            union(members[0], member)
    for relation in snapshot.relations.values():
        subject = snapshot.emojis.get(relation.subject_id)
        object_emoji = snapshot.emojis.get(relation.object_id)
        if (
            relation.review.status.value == "approved"
            and relation.relation_type.value == "same-artwork"
            and relation.scope.value == "entity"
            and subject is not None
            and object_emoji is not None
            and relation.subject_id in parent
            and relation.object_id in parent
            and relation.evidence.subject_media_digest == subject.fingerprints.input_media_digest
            and relation.evidence.object_media_digest
            == object_emoji.fingerprints.input_media_digest
        ):
            union(relation.subject_id, relation.object_id)

    active_collections = {
        collection.id
        for collection in snapshot.collections.values()
        if collection.availability.status.value == "active"
    }
    component_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    collection_sizes: dict[str, int] = defaultdict(int)
    selected_collections: set[str] = set()
    for membership in snapshot.memberships.values():
        if (
            membership.status.value != "active"
            or membership.collection_id not in active_collections
            or membership.emoji_id not in parent
        ):
            continue
        collection_sizes[membership.collection_id] += 1
        component_counts[find(membership.emoji_id)][membership.collection_id] += 1
        if selected_emoji_ids is not None and membership.emoji_id in selected_emoji_ids:
            selected_collections.add(membership.collection_id)

    pair_matches: dict[tuple[str, str], int] = defaultdict(int)
    for counts in component_counts.values():
        collections = sorted(counts)
        for position, left in enumerate(collections):
            for right in collections[position + 1 :]:
                pair_matches[(left, right)] += min(counts[left], counts[right])

    profile = load_analysis_profile("collection-dedupe-v1").data
    clone_minimum = Decimal(str(profile["clone_candidate"]["minimum_each_coverage"]))
    subset_minimum = Decimal(str(profile["subset_candidate"]["minimum_smaller_coverage"]))
    subset_maximum = Decimal(str(profile["subset_candidate"]["maximum_larger_coverage_exclusive"]))
    clone_matches = int(profile["clone_candidate"]["minimum_matches"])
    subset_matches = int(profile["subset_candidate"]["minimum_matches"])
    result: list[CollectionCandidate] = []
    for (left, right), matched in sorted(pair_matches.items()):
        if selected_emoji_ids is not None and not ({left, right} & selected_collections):
            continue
        left_size, right_size = collection_sizes[left], collection_sizes[right]
        left_coverage = Decimal(matched) / left_size
        right_coverage = Decimal(matched) / right_size
        candidate_type: Literal["clone-candidate", "subset-candidate"] | None = None
        if (
            matched >= clone_matches
            and left_coverage >= clone_minimum
            and right_coverage >= clone_minimum
        ):
            candidate_type = "clone-candidate"
        elif (
            matched >= subset_matches
            and Decimal(matched) / min(left_size, right_size) >= subset_minimum
            and Decimal(matched) / max(left_size, right_size) < subset_maximum
        ):
            candidate_type = "subset-candidate"
        if candidate_type is None:
            continue
        result.append(
            CollectionCandidate(
                collection_id=left,
                against_collection_id=right,
                candidate_type=candidate_type,
                matched_count=matched,
                collection_size=left_size,
                against_collection_size=right_size,
                collection_coverage=_decimal_ratio(left_coverage),
                against_collection_coverage=_decimal_ratio(right_coverage),
            )
        )
    return tuple(result)


def _decimal_ratio(value: Decimal) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _references(snapshot: DatasetSnapshot) -> list[_Ref]:
    result: list[_Ref] = []
    for emoji in sorted(snapshot.emojis.values(), key=lambda item: item.id):
        if emoji.availability.status.value != "active":
            continue
        media = {(item.role.value, item.variant_id or ""): item for item in emoji.media}
        rendering = {item.key: item for item in emoji.facets.rendering.items}
        if emoji.fingerprints.status.value != "complete":
            continue
        for item in emoji.fingerprints.items:
            key = item.key
            source_media = media[key]
            render = rendering[key]
            perceptual = item.perceptual
            result.append(
                _Ref(
                    emoji_id=emoji.id,
                    profile_id=emoji.fingerprints.profile,
                    role=item.role.value,
                    variant_id=item.variant_id,
                    media_sha256=source_media.sha256,
                    byte_size=source_media.byte_size,
                    duration_ms=source_media.duration_ms or 0,
                    animated=source_media.animated,
                    alpha_mode=render.alpha_mode.value,
                    color_behavior=render.color_behavior.value,
                    decoded_sha256=item.decoded_payload_sha256,
                    canonical_sha256=item.canonical_render_sha256,
                    shape_sha256=item.shape_sha256,
                    low_information=perceptual.low_information,
                    layout=_decode_hashes(perceptual.layout_phash64),
                    content=_decode_hashes(perceptual.content_phash64),
                    alpha=_decode_hashes(perceptual.alpha_phash64),
                    edge=_decode_hashes(perceptual.edge_phash64),
                )
            )
    return result


def _exact_groups(snapshot: DatasetSnapshot, refs: list[_Ref]) -> list[dict[str, Any]]:
    namespace = uuid.UUID(str(snapshot.manifest["visual_relation_namespace"]))
    groups: list[dict[str, Any]] = []
    binary: dict[tuple[str, int], list[_Ref]] = defaultdict(list)
    decoded: dict[tuple[str, str], list[_Ref]] = defaultdict(list)
    for ref in refs:
        binary[(ref.media_sha256, ref.byte_size)].append(ref)
        decoded[(ref.profile_id, ref.decoded_sha256)].append(ref)
    for (sha256, byte_size), members in sorted(binary.items()):
        if len(members) < 2:
            continue
        digest = hashlib.sha256(
            rfc8785.dumps({"byte_size": byte_size, "sha256": sha256})
        ).hexdigest()
        group = _group(namespace, "binary-exact", "media", "", digest, members)
        group["source_sha256"] = sha256
        group["byte_size"] = byte_size
        groups.append(group)
    for (profile, digest), members in sorted(decoded.items()):
        if len(members) >= 2:
            groups.append(_group(namespace, "decoded-exact", "media", profile, digest, members))

    refs_by_emoji: dict[str, list[_Ref]] = defaultdict(list)
    for ref in refs:
        refs_by_emoji[ref.emoji_id].append(ref)
    for group_type in ("binary-exact", "decoded-exact"):
        entity_buckets: dict[tuple[str, bytes], list[str]] = defaultdict(list)
        signature_payloads: dict[tuple[str, bytes], list[dict[str, Any]]] = {}
        for emoji_id, entity_refs in refs_by_emoji.items():
            profiles = {ref.profile_id for ref in entity_refs}
            if group_type == "decoded-exact" and len(profiles) != 1:
                continue
            profile = "" if group_type == "binary-exact" else min(profiles)
            payload = []
            for ref in sorted(entity_refs, key=lambda item: item.media_key):
                value: dict[str, Any] = {"role": ref.role}
                if ref.variant_id is not None:
                    value["variant_id"] = ref.variant_id
                if group_type == "binary-exact":
                    value.update({"sha256": ref.media_sha256, "byte_size": ref.byte_size})
                else:
                    value["decoded_payload_sha256"] = ref.decoded_sha256
                payload.append(value)
            signature = rfc8785.dumps(payload)
            bucket_key = (profile, signature)
            entity_buckets[bucket_key].append(emoji_id)
            signature_payloads[bucket_key] = payload
        for bucket_key, emoji_ids in sorted(entity_buckets.items(), key=lambda item: item[0]):
            if len(emoji_ids) < 2:
                continue
            profile, _ = bucket_key
            digest = hashlib.sha256(rfc8785.dumps(signature_payloads[bucket_key])).hexdigest()
            groups.append(_entity_group(namespace, group_type, profile, digest, sorted(emoji_ids)))
    return sorted(groups, key=lambda item: item["id"])


def _group(
    namespace: uuid.UUID,
    group_type: str,
    scope: str,
    profile: str,
    digest: str,
    members: list[_Ref],
) -> dict[str, Any]:
    name = "\0".join(("duplicate-group", group_type, scope, profile, digest))
    result: dict[str, Any] = {
        "id": "mxdg_" + str(uuid.uuid5(namespace, name)),
        "group_type": group_type,
        "scope": scope,
        "content_digest": digest,
        "members": [_member(value) for value in sorted(members, key=_ref_sort_key)],
    }
    if profile:
        result["profile"] = profile
    return result


def _entity_group(
    namespace: uuid.UUID,
    group_type: str,
    profile: str,
    digest: str,
    emoji_ids: list[str],
) -> dict[str, Any]:
    name = "\0".join(("duplicate-group", group_type, "entity", profile, digest))
    result: dict[str, Any] = {
        "id": "mxdg_" + str(uuid.uuid5(namespace, name)),
        "group_type": group_type,
        "scope": "entity",
        "content_digest": digest,
        "members": emoji_ids,
    }
    if profile:
        result["profile"] = profile
    return result


def _group_member_emoji_id(member: Any) -> str:
    return str(member["emoji_id"] if isinstance(member, dict) else member)


def _member(ref: _Ref) -> dict[str, str]:
    result = {"emoji_id": ref.emoji_id, "role": ref.role}
    if ref.variant_id is not None:
        result["variant_id"] = ref.variant_id
    return result


def _bounded_insert(
    buckets: dict[Any, list[int]],
    suppressed: set[Any],
    key: Any,
    index: int,
    limit: int,
) -> None:
    if key in suppressed:
        return
    values = buckets.setdefault(key, [])
    values.append(index)
    if len(values) > limit:
        buckets.pop(key, None)
        suppressed.add(key)


def _lsh_band_bits(secondary_keys: Sequence[str]) -> int:
    widths: set[int] = set()
    for value in secondary_keys:
        try:
            label, marker, raw_width = value.rsplit("-", 2)
            width = int(raw_width)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid dedupe secondary key: {value}") from exc
        if label not in {"layout-phash", "content-phash", "alpha-phash", "edge-phash"}:
            raise ValueError(f"unsupported dedupe secondary key: {value}")
        if marker != "prefix" or width <= 0 or 64 % width:
            raise ValueError(f"invalid dedupe secondary key width: {value}")
        widths.add(width)
    if len(widths) != 1:
        raise ValueError("dedupe secondary keys must use one 64-bit divisor width")
    return next(iter(widths))


def _hash_sequences(ref: _Ref) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return (
        ("layout", ref.layout),
        ("content", ref.content),
        ("alpha", ref.alpha),
        ("edge", ref.edge),
    )


def _lsh_keys(ref: _Ref, *, band_bits: int) -> set[tuple[str, int, int]]:
    result: set[tuple[str, int, int]] = set()
    mask = (1 << band_bits) - 1
    for label, values in _hash_sequences(ref):
        # Index every animation sample: a majority hash alone can miss a pair
        # whose median aligned distance satisfies the profile threshold.
        for value in set(values):
            for band in range(64 // band_bits):
                result.add((label, band, (value >> (band * band_bits)) & mask))
    return result


@lru_cache(maxsize=32)
def _probe_masks(width: int, radius: int) -> tuple[int, ...]:
    if radius < 0 or radius > width:
        raise ValueError("LSH probe radius is outside the band width")
    result = [0]
    for distance in range(1, radius + 1):
        result.extend(
            sum(1 << bit for bit in positions)
            for positions in itertools.combinations(range(width), distance)
        )
    return tuple(result)


def _build_candidate_index(
    references: Iterable[_Ref],
    *,
    thresholds: Mapping[str, Any],
    bucket_limit: int,
    secondary_keys: Sequence[str],
    band_bits: int,
) -> _CandidateIndex:
    """Build bounded indexes shared by full and incremental near scans."""

    refs: Sequence[_Ref]
    if isinstance(references, Sequence):
        refs = references
    else:
        refs = tuple(references)
    lsh_buckets: dict[tuple[str, int, int], list[int]] = {}
    canonical_buckets: dict[tuple[str, str], list[int]] = {}
    shape_members: dict[str, list[int]] = defaultdict(list)
    suppressed: set[Any] = set()
    for index, ref in enumerate(refs):
        if not ref.low_information:
            for key in _lsh_keys(ref, band_bits=band_bits):
                _bounded_insert(lsh_buckets, suppressed, key, index, bucket_limit)
        _bounded_insert(
            canonical_buckets,
            suppressed,
            ("canonical", ref.canonical_sha256),
            index,
            bucket_limit,
        )
        if not ref.low_information:
            shape_members[ref.shape_sha256].append(index)

    shape_buckets: dict[str, Sequence[int]] = {}
    shape_secondary_buckets: dict[tuple[str, str, int], list[int]] = {}
    shape_secondary_suppressed: set[tuple[str, str, int]] = set()
    oversized_shapes: set[str] = set()
    for digest, values in sorted(shape_members.items()):
        if len(values) <= bucket_limit:
            shape_buckets[digest] = values
            continue
        oversized_shapes.add(digest)
        suppressed.add(("shape", digest))
        for index in values:
            ref = refs[index]
            sequences = dict(_hash_sequences(ref))
            for specification in secondary_keys:
                label_with_suffix, _, raw_width = specification.rsplit("-", 2)
                label = label_with_suffix.removesuffix("-phash")
                width = int(raw_width)
                aggregate = _majority_hash(sequences[label])
                prefix = aggregate >> (64 - width)
                shape_key = (digest, specification, prefix)
                _bounded_insert(
                    shape_secondary_buckets,
                    shape_secondary_suppressed,
                    shape_key,
                    index,
                    bucket_limit,
                )
    suppressed.update(("shape-secondary", *key) for key in shape_secondary_suppressed)

    lsh_values: dict[tuple[str, int], list[int]] = defaultdict(list)
    for label, band, value in lsh_buckets:
        lsh_values[(label, band)].append(value)
    frozen_lsh_values = {key: tuple(sorted(values)) for key, values in lsh_values.items()}
    stats = _CandidateIndexStats(
        bucket_entries=sum(len(values) for values in lsh_buckets.values())
        + sum(len(values) for values in canonical_buckets.values())
        + sum(len(values) for values in shape_buckets.values())
        + sum(len(values) for values in shape_secondary_buckets.values()),
        low_information_refs=sum(ref.low_information for ref in refs),
    )
    return _CandidateIndex(
        refs=refs,
        lsh_buckets=lsh_buckets,
        lsh_values=frozen_lsh_values,
        canonical_buckets=canonical_buckets,
        shape_buckets=shape_buckets,
        shape_secondary_buckets=shape_secondary_buckets,
        oversized_shapes=frozenset(oversized_shapes),
        secondary_keys=tuple(secondary_keys),
        thresholds=thresholds,
        band_bits=band_bits,
        suppressed_bucket_count=len(suppressed),
        stats=stats,
    )


def _majority_hash(values: tuple[int, ...]) -> int:
    threshold = (len(values) + 1) // 2
    result = 0
    for bit in range(64):
        if sum((value >> bit) & 1 for value in values) >= threshold:
            result |= 1 << bit
    return result


def _compare(
    left: _Ref,
    right: _Ref,
    snapshot: DatasetSnapshot,
    thresholds: Any,
) -> DedupeCandidate | None:
    if left.animated != right.animated:
        return None
    animated = left.animated
    cyclic = bool(thresholds["cyclic_alignment"])
    dtw_band = int(thresholds["dtw_band_frames"])
    reverse = bool(thresholds["reverse_playback_match"])
    layout = _aligned_distances(
        left.layout,
        right.layout,
        animated=animated,
        cyclic=cyclic,
        dtw_band=dtw_band,
        reverse=reverse,
    )
    content = _aligned_distances(
        left.content,
        right.content,
        animated=animated,
        cyclic=cyclic,
        dtw_band=dtw_band,
        reverse=reverse,
    )
    alpha = _aligned_distances(
        left.alpha,
        right.alpha,
        animated=animated,
        cyclic=cyclic,
        dtw_band=dtw_band,
        reverse=reverse,
    )
    edge = _aligned_distances(
        left.edge,
        right.edge,
        animated=animated,
        cyclic=cyclic,
        dtw_band=dtw_band,
        reverse=reverse,
    )
    primary_values = min((layout, content), key=lambda values: (statistics.median(values), values))
    primary = int(statistics.median(primary_values))
    alpha_distance = int(statistics.median(alpha))
    edge_distance = int(statistics.median(edge))
    p90 = _percentile90(primary_values)
    signals: list[str] = []
    if left.canonical_sha256 == right.canonical_sha256:
        signals.append("canonical-render-match")
    shape_eligible = _shape_candidate_eligible(left, right)
    if left.shape_sha256 == right.shape_sha256 and shape_eligible:
        signals.extend(("possible-recolor", "shape-match"))
    high_priority = False
    if left.animated:
        if primary <= int(thresholds["animation_median_hamming_max"]) and p90 <= int(
            thresholds["animation_p90_hamming_max"]
        ):
            signals.append("cyclic-phash-match")
    else:
        alpha_required = left.alpha_mode != "opaque" or right.alpha_mode != "opaque"
        if primary <= int(thresholds["static_primary_hamming_max"]) and (
            not alpha_required or alpha_distance <= int(thresholds["static_alpha_hamming_max"])
        ):
            signals.append("phash-match")
        high_priority = primary <= int(thresholds["static_high_priority_primary_hamming_max"]) and (
            not alpha_required
            or alpha_distance <= int(thresholds["static_high_priority_alpha_hamming_max"])
        )
    if (
        shape_eligible
        and alpha_distance <= int(thresholds["adaptive_alpha_hamming_max"])
        and edge_distance <= int(thresholds["adaptive_edge_hamming_max"])
    ):
        signals.append("possible-recolor")
    if not signals:
        return None
    speed = _speed_ratio(left.duration_ms, right.duration_ms)
    duration_compatible = speed <= float(thresholds["speed_change_ratio"])
    if not duration_compatible:
        signals.append("speed-change")
    left_text = _literal_text(snapshot, left.emoji_id)
    right_text = _literal_text(snapshot, right.emoji_id)
    text_match = bool(left_text) and left_text == right_text
    text_mismatch = left_text != right_text and bool(left_text) and bool(right_text)
    if text_mismatch:
        signals.append("possible-variant")
    return DedupeCandidate(
        emoji_id=left.emoji_id,
        against_emoji_id=right.emoji_id,
        role=left.role,
        variant_id=left.variant_id,
        against_role=right.role,
        against_variant_id=right.variant_id,
        signals=tuple(sorted(set(signals))),
        primary_distance=primary,
        alpha_distance=alpha_distance,
        edge_distance=edge_distance,
        p90_distance=p90,
        speed_ratio=f"{speed:.6f}".rstrip("0").rstrip("."),
        duration_compatible=duration_compatible,
        text_match=text_match,
        text_mismatch=text_mismatch,
        high_priority=high_priority,
    )


def _aligned_distances(
    left: tuple[int, ...],
    right: tuple[int, ...],
    *,
    animated: bool,
    cyclic: bool,
    dtw_band: int,
    reverse: bool,
) -> tuple[int, ...]:
    if len(left) != len(right) or not left:
        return (64,)
    if not animated or len(left) == 1:
        return tuple((a ^ b).bit_count() for a, b in zip(left, right, strict=True))
    candidates: list[tuple[int, tuple[int, ...]]] = []
    orientations = (right, tuple(reversed(right))) if reverse else (right,)
    shifts = range(len(right)) if cyclic else range(1)
    for orientation in orientations:
        for shift in shifts:
            shifted = orientation[shift:] + orientation[:shift]
            direct = tuple((a ^ b).bit_count() for a, b in zip(left, shifted, strict=True))
            candidates.append((sum(direct), direct))
            dtw = _banded_dtw(left, shifted, band=dtw_band)
            candidates.append((sum(dtw), dtw))
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def _shape_candidate_eligible(left: _Ref, right: _Ref) -> bool:
    if left.low_information or right.low_information:
        return False
    return any(
        ref.color_behavior in {"platform-adaptive", "mixed"}
        or ref.alpha_mode in {"binary", "translucent"}
        for ref in (left, right)
    )


def _banded_dtw(left: tuple[int, ...], right: tuple[int, ...], *, band: int) -> tuple[int, ...]:
    size = len(left)
    paths: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {(0, 0): (0, ())}
    for i in range(1, size + 1):
        for j in range(max(1, i - band), min(size, i + band) + 1):
            previous = [
                paths[key] for key in ((i - 1, j), (i, j - 1), (i - 1, j - 1)) if key in paths
            ]
            if not previous:
                continue
            best_cost, best_values = min(previous, key=lambda item: (item[0], len(item[1])))
            distance = (left[i - 1] ^ right[j - 1]).bit_count()
            paths[(i, j)] = (best_cost + distance, (*best_values, distance))
    return paths.get((size, size), (64 * size, (64,) * size))[1]


def _decode_hashes(value: str) -> tuple[int, ...]:
    raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    return tuple(item[0] for item in struct.iter_unpack(">Q", raw))


def _literal_text(snapshot: DatasetSnapshot, emoji_id: str) -> tuple[str, ...]:
    emoji = snapshot.emojis[emoji_id]
    return tuple(item.value for item in emoji.facets.text_content.items)


def _approved_not_duplicate_pairs(snapshot: DatasetSnapshot) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for value in snapshot.relations.values():
        if value.relation_type.value != "not-duplicate" or value.review.status.value != "approved":
            continue
        subject = snapshot.emojis.get(value.subject_id)
        object_emoji = snapshot.emojis.get(value.object_id)
        if subject is None or object_emoji is None:
            continue
        if (
            value.evidence.subject_media_digest != subject.fingerprints.input_media_digest
            or value.evidence.object_media_digest != object_emoji.fingerprints.input_media_digest
        ):
            continue
        left_id, right_id = sorted((value.subject_id, value.object_id))
        result.add((left_id, right_id))
    return result


def _speed_ratio(left: int, right: int) -> float:
    if left <= 0 or right <= 0:
        return 1.0
    return max(left, right) / min(left, right)


def _percentile90(values: tuple[int, ...]) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.9) - 1)]


def _candidate_sort_key(value: DedupeCandidate) -> tuple[Any, ...]:
    priority = (
        0
        if "canonical-render-match" in value.signals
        else 1
        if "shape-match" in value.signals
        else 2
    )
    return (
        priority,
        not value.high_priority,
        value.primary_distance,
        value.p90_distance,
        value.alpha_distance,
        not value.duration_compatible,
        Decimal(value.speed_ratio),
        not value.text_match,
        value.text_mismatch,
        value.against_emoji_id,
        value.against_role,
        value.against_variant_id or "",
    )


def _ref_sort_key(value: _Ref) -> tuple[str, str, str]:
    return value.emoji_id, value.role, value.variant_id or ""
