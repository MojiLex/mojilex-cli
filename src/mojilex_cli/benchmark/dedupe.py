"""Manifest-driven benchmark for exact and bounded near-dedupe behavior."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import rfc8785
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mojilex_cli.analysis import load_analysis_profile
from mojilex_cli.dataset import load_dataset, validate_dataset
from mojilex_cli.dedupe import scan_snapshot

from .common import (
    ID_PATTERN,
    SEMVER_PATTERN,
    SHA256_PATTERN,
    BenchmarkError,
    DeclaredFile,
    RightsRecord,
    declared_files_sha256,
    finalize_report,
    load_manifest,
    ratio_bp,
    resolve_declared_file,
    validate_canonical_ids,
    wilson_interval_bp,
)

MANDATORY_DEDUPE_STRATA = (
    "adaptive",
    "animation",
    "low-information",
    "static",
    "text-digit-arrow",
    "transparent-outline",
)


class DedupeBenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=ID_PATTERN)
    split: Literal["development", "holdout"]
    left_emoji_id: str = Field(min_length=1, max_length=100)
    right_emoji_id: str = Field(min_length=1, max_length=100)
    rights_id: str = Field(pattern=ID_PATTERN)
    strata: tuple[str, ...] = Field(min_length=1)
    isolation_groups: tuple[str, ...] = Field(min_length=1)
    pair_label: Literal[
        "binary-exact",
        "decoded-exact",
        "canonical-render-match",
        "same-artwork",
        "variant",
        "not-duplicate",
    ]
    binary_exact_expected: bool
    decoded_exact_expected: bool
    near_candidate_expected: bool

    @model_validator(mode="after")
    def canonical_case(self) -> DedupeBenchmarkCase:
        if self.left_emoji_id >= self.right_emoji_id:
            raise ValueError("dedupe benchmark pair endpoints must be distinct and sorted")
        validate_canonical_ids(self.strata, label="dedupe strata")
        validate_canonical_ids(self.isolation_groups, label="split isolation groups")
        if self.pair_label == "binary-exact" and not self.binary_exact_expected:
            raise ValueError("binary-exact label requires binary exact ground truth")
        if self.pair_label == "decoded-exact" and (
            self.binary_exact_expected or not self.decoded_exact_expected
        ):
            raise ValueError("decoded-exact label requires only decoded exact ground truth")
        if self.pair_label in {
            "canonical-render-match",
            "same-artwork",
            "variant",
            "not-duplicate",
        } and (self.binary_exact_expected or self.decoded_exact_expected):
            raise ValueError("non-exact pair label cannot claim an exact match")
        if self.pair_label == "not-duplicate" and self.near_candidate_expected:
            raise ValueError("not-duplicate cannot be a required near candidate")
        if (
            self.pair_label
            in {
                "canonical-render-match",
                "same-artwork",
                "variant",
            }
            and not self.near_candidate_expected
        ):
            raise ValueError("visual match labels must be required near candidates")
        return self


class RuntimeComponent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    component_id: str = Field(pattern=ID_PATTERN)
    version: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def safe_version(self) -> RuntimeComponent:
        if self.version != self.version.strip() or any(
            ord(character) < 32 for character in self.version
        ):
            raise ValueError("runtime component version is invalid")
        return self


def dedupe_benchmark_split_sha256(cases: tuple[DedupeBenchmarkCase, ...]) -> str:
    payload = [
        {
            "case_id": item.case_id,
            "isolation_groups": list(item.isolation_groups),
            "left_emoji_id": item.left_emoji_id,
            "pair_label": item.pair_label,
            "right_emoji_id": item.right_emoji_id,
            "split": item.split,
        }
        for item in cases
    ]
    return hashlib.sha256(rfc8785.dumps(payload)).hexdigest()


class DedupeBenchmarkManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_type: Literal["mojilex-dedupe-benchmark-v1"]
    schema_version: Literal["1.0.0"]
    benchmark_id: str = Field(pattern=ID_PATTERN)
    benchmark_version: str = Field(pattern=SEMVER_PATTERN)
    dedupe_profile: Literal["dedupe-v1"]
    dedupe_profile_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_path: str = Field(pattern=ID_PATTERN)
    dataset_sha256: str = Field(pattern=SHA256_PATTERN)
    split_sha256: str = Field(pattern=SHA256_PATTERN)
    dataset_files: tuple[DeclaredFile, ...] = Field(min_length=1)
    cli_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    decoder_backend_fingerprint: str = Field(pattern=SHA256_PATTERN)
    runtime_components: tuple[RuntimeComponent, ...] = Field(min_length=1)
    exact_command: Literal["mojilex benchmark-dedupe --manifest <PATH>"]
    rights: tuple[RightsRecord, ...] = Field(min_length=1)
    cases: tuple[DedupeBenchmarkCase, ...] = Field(min_length=1)
    maximum_comparisons: int = Field(ge=0)
    production_all_pairs_allowed: Literal[False] = False

    @model_validator(mode="after")
    def manifest_integrity(self) -> DedupeBenchmarkManifest:
        rights_ids = [item.rights_id for item in self.rights]
        case_ids = [item.case_id for item in self.cases]
        file_paths = [item.path for item in self.dataset_files]
        validate_canonical_ids(rights_ids, label="rights records")
        validate_canonical_ids(case_ids, label="dedupe cases")
        validate_canonical_ids(
            [item.component_id for item in self.runtime_components],
            label="runtime components",
        )
        if file_paths != sorted(file_paths) or len(file_paths) != len(set(file_paths)):
            raise ValueError("dataset files must be unique and sorted")
        if any(not item.redistribution_allowed for item in self.rights):
            raise ValueError("all benchmark fixtures must permit redistribution")
        known_rights = set(rights_ids)
        if any(item.rights_id not in known_rights for item in self.cases):
            raise ValueError("dedupe case references unknown rights metadata")
        pairs = [(item.left_emoji_id, item.right_emoji_id) for item in self.cases]
        if len(pairs) != len(set(pairs)):
            raise ValueError("dedupe benchmark pairs must be unique")
        split_by_group: dict[str, str] = {}
        for item in self.cases:
            for group in (
                *item.isolation_groups,
                f"emoji:{item.left_emoji_id}",
                f"emoji:{item.right_emoji_id}",
            ):
                previous = split_by_group.setdefault(group, item.split)
                if previous != item.split:
                    raise ValueError("collection/exact/artwork group crosses benchmark split")
        if declared_files_sha256(self.dataset_files) != self.dataset_sha256:
            raise ValueError("dataset_sha256 does not match the declared file manifest")
        if dedupe_benchmark_split_sha256(self.cases) != self.split_sha256:
            raise ValueError("split_sha256 does not match benchmark split declarations")
        return self


class DedupeObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=ID_PATTERN)
    binary_exact_observed: bool
    decoded_exact_observed: bool
    near_candidate_observed: bool


@dataclass(frozen=True, slots=True)
class DedupeEvidence:
    observations: tuple[DedupeObservation, ...]
    comparisons: int
    indexed_items: int
    suppressed_bucket_count: int
    candidate_overflow_count: int = 0
    candidate_count_before_limit_total: int = 0


DedupeScanner = Callable[[DedupeBenchmarkManifest, Path], DedupeEvidence]


def run_dedupe_benchmark(
    manifest_path: Path,
    *,
    scanner: DedupeScanner | None = None,
) -> dict[str, Any]:
    manifest, manifest_hash, _ = load_manifest(manifest_path, DedupeBenchmarkManifest)
    profile = load_analysis_profile(manifest.dedupe_profile)
    if profile.sha256 != manifest.dedupe_profile_sha256:
        raise BenchmarkError("dedupe benchmark profile hash does not match this CLI")
    evidence = (scanner or _scan_dataset)(manifest, manifest_path.resolve().parent)
    return evaluate_dedupe_benchmark(manifest, evidence, manifest_sha256=manifest_hash)


def evaluate_dedupe_benchmark(
    manifest: DedupeBenchmarkManifest,
    evidence: DedupeEvidence,
    *,
    manifest_sha256: str,
) -> dict[str, Any]:
    if any(
        value < 0
        for value in (
            evidence.comparisons,
            evidence.indexed_items,
            evidence.suppressed_bucket_count,
            evidence.candidate_overflow_count,
            evidence.candidate_count_before_limit_total,
        )
    ):
        raise BenchmarkError("dedupe evidence counters must be non-negative")
    observations = {item.case_id: item for item in evidence.observations}
    expected_ids = {item.case_id for item in manifest.cases}
    if set(observations) != expected_ids or len(observations) != len(evidence.observations):
        raise BenchmarkError("dedupe observations must cover every case exactly once")
    binary = _confusion(
        manifest.cases, observations, "binary_exact_expected", "binary_exact_observed"
    )
    decoded = _confusion(
        manifest.cases, observations, "decoded_exact_expected", "decoded_exact_observed"
    )
    near = _confusion(
        manifest.cases, observations, "near_candidate_expected", "near_candidate_observed"
    )
    near["precision_at_20_bp"] = near["precision_bp"]
    near["precision_at_20_ci95_bp"] = near["precision_ci95_bp"]
    strata: dict[str, dict[str, Any]] = {}
    for stratum in sorted({value for case in manifest.cases for value in case.strata}):
        selected = tuple(case for case in manifest.cases if stratum in case.strata)
        metric = _confusion(
            selected, observations, "near_candidate_expected", "near_candidate_observed"
        )
        metric["precision_at_20_bp"] = metric["precision_bp"]
        metric["precision_at_20_ci95_bp"] = metric["precision_ci95_bp"]
        strata[stratum] = {"case_count": len(selected), "near_candidate": metric}
    possible_pairs = evidence.indexed_items * (evidence.indexed_items - 1) // 2
    mandatory_counts = {
        name: sum(name in item.strata for item in manifest.cases)
        for name in MANDATORY_DEDUPE_STRATA
    }
    mandatory_positive_counts = {
        name: sum(name in item.strata and item.near_candidate_expected for item in manifest.cases)
        for name in MANDATORY_DEDUPE_STRATA
    }
    pair_label_counts = {
        name: sum(item.pair_label == name for item in manifest.cases)
        for name in (
            "binary-exact",
            "decoded-exact",
            "canonical-render-match",
            "same-artwork",
            "variant",
            "not-duplicate",
        )
    }
    gates = {
        "benchmark_has_at_least_500_pairs": len(manifest.cases) >= 500,
        "development_and_holdout_present": {item.split for item in manifest.cases}
        == {"development", "holdout"},
        "mandatory_strata_have_30_cases": all(value >= 30 for value in mandatory_counts.values()),
        "mandatory_strata_have_30_positive_pairs": all(
            value >= 30 for value in mandatory_positive_counts.values()
        ),
        "all_required_pair_labels_present": all(value > 0 for value in pair_label_counts.values()),
        "binary_exact_false_positives_zero": binary["false_positive"] == 0,
        "decoded_exact_false_positives_zero": decoded["false_positive"] == 0,
        "binary_exact_recall_100_percent": binary["positive_count"] > 0
        and binary["recall_bp"] == 10_000,
        "decoded_exact_recall_100_percent": decoded["positive_count"] > 0
        and decoded["recall_bp"] == 10_000,
        "near_candidate_recall_at_least_98_percent": near["positive_count"] > 0
        and near["recall_bp"] >= 9_800,
        "near_candidate_recall_each_stratum_at_least_95_percent": all(
            strata[name]["near_candidate"]["recall_bp"] >= 9_500
            for name in MANDATORY_DEDUPE_STRATA
            if name in strata
        )
        and all(name in strata for name in MANDATORY_DEDUPE_STRATA),
        "bounded_comparison_budget": evidence.comparisons <= manifest.maximum_comparisons,
        "not_all_pairs": possible_pairs == 0 or evidence.comparisons < possible_pairs,
    }
    return finalize_report(
        {
            "report_type": "mojilex-dedupe-benchmark-report-v1",
            "benchmark_id": manifest.benchmark_id,
            "benchmark_version": manifest.benchmark_version,
            "manifest_sha256": manifest_sha256,
            "dedupe_profile": manifest.dedupe_profile,
            "dedupe_profile_sha256": manifest.dedupe_profile_sha256,
            "dataset_sha256": manifest.dataset_sha256,
            "split_sha256": manifest.split_sha256,
            "cli_commit": manifest.cli_commit,
            "decoder_backend_fingerprint": manifest.decoder_backend_fingerprint,
            "runtime_components": [
                item.model_dump(mode="json") for item in manifest.runtime_components
            ],
            "exact_command": manifest.exact_command,
            "case_count": len(manifest.cases),
            "indexed_items": evidence.indexed_items,
            "comparisons": evidence.comparisons,
            "possible_all_pairs": possible_pairs,
            "suppressed_bucket_count": evidence.suppressed_bucket_count,
            "candidate_overflow_count": evidence.candidate_overflow_count,
            "candidate_overflow_rate_bp": ratio_bp(
                evidence.candidate_overflow_count, evidence.indexed_items
            ),
            "candidate_count_before_limit_total": (evidence.candidate_count_before_limit_total),
            "metrics": {"binary_exact": binary, "decoded_exact": decoded, "near": near},
            "strata": strata,
            "mandatory_strata_counts": mandatory_counts,
            "mandatory_strata_positive_counts": mandatory_positive_counts,
            "pair_label_counts": pair_label_counts,
            "gates": gates,
            "passed": all(gates.values()),
        }
    )


def _confusion(
    cases: tuple[DedupeBenchmarkCase, ...],
    observations: Mapping[str, DedupeObservation],
    expected_field: str,
    observed_field: str,
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for case in cases:
        expected = bool(getattr(case, expected_field))
        observed = bool(getattr(observations[case.case_id], observed_field))
        if expected and observed:
            tp += 1
        elif observed:
            fp += 1
        elif expected:
            fn += 1
        else:
            tn += 1
    positives = tp + fn
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "positive_count": positives,
        "precision_bp": ratio_bp(tp, tp + fp),
        "precision_ci95_bp": wilson_interval_bp(tp, tp + fp),
        "recall_bp": ratio_bp(tp, positives),
        "recall_ci95_bp": wilson_interval_bp(tp, positives),
    }


def _scan_dataset(manifest: DedupeBenchmarkManifest, root: Path) -> DedupeEvidence:
    dataset_root = root / manifest.dataset_path
    try:
        dataset_root = dataset_root.resolve(strict=True)
        dataset_root.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise BenchmarkError("benchmark dataset path is missing or unsafe") from exc
    for declaration in manifest.dataset_files:
        resolve_declared_file(dataset_root, declaration)
    validate_dataset(dataset_root, strict=False).raise_for_errors()
    snapshot = load_dataset(dataset_root)
    declared = {PurePosixPath(item.path): item.sha256 for item in manifest.dataset_files}
    if set(snapshot.source_bytes) != set(declared) or any(
        hashlib.sha256(value).hexdigest() != declared[path]
        for path, value in snapshot.source_bytes.items()
    ):
        raise BenchmarkError("loaded dataset inputs do not exactly match declared files")
    report = scan_snapshot(snapshot, mode="near", max_candidates=20)
    binary_membership, decoded_membership = _exact_memberships(report.exact_groups)
    near_pairs = {
        tuple(sorted((emoji_id, candidate.against_emoji_id)))
        for emoji_id, candidates in report.candidates.items()
        for candidate in candidates
    }
    observations = tuple(
        DedupeObservation(
            case_id=case.case_id,
            binary_exact_observed=_same_group(
                binary_membership, case.left_emoji_id, case.right_emoji_id
            ),
            decoded_exact_observed=_same_group(
                decoded_membership, case.left_emoji_id, case.right_emoji_id
            ),
            near_candidate_observed=(case.left_emoji_id, case.right_emoji_id) in near_pairs,
        )
        for case in manifest.cases
    )
    return DedupeEvidence(
        observations=observations,
        comparisons=report.comparisons,
        indexed_items=len(snapshot.emojis),
        suppressed_bucket_count=report.suppressed_bucket_count,
        candidate_overflow_count=sum(report.candidate_overflow.values()),
        candidate_count_before_limit_total=sum(report.candidate_count_before_limit.values()),
    )


def _exact_memberships(
    groups: tuple[dict[str, Any], ...],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Index entity group membership without expanding groups into all pairs."""

    binary: dict[str, set[str]] = {}
    decoded: dict[str, set[str]] = {}
    for group in groups:
        if group.get("scope") != "entity":
            continue
        group_type = group.get("group_type")
        if group_type not in {"binary-exact", "decoded-exact"}:
            continue
        target = binary if group_type == "binary-exact" else decoded
        group_id = str(group.get("id", ""))
        if not group_id:
            raise BenchmarkError("dedupe exact group has no stable ID")
        for member in group.get("members", ()):
            emoji_id = str(member["emoji_id"] if isinstance(member, dict) else member)
            target.setdefault(emoji_id, set()).add(group_id)
    return binary, decoded


def _same_group(membership: Mapping[str, set[str]], left: str, right: str) -> bool:
    return bool(membership.get(left, set()) & membership.get(right, set()))
