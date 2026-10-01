from __future__ import annotations

import hashlib
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterator
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mojilex_cli.analysis import load_analysis_profile
from mojilex_cli.dedupe.engine import (
    _build_candidate_index,
    _compare,
    _lsh_band_bits,
    _Ref,
)

_FINGERPRINT_COUNT = 100_000
_OVERFLOW_CLUSTER_SIZE = 22
_MAX_WALL_SECONDS = 60.0
_MAX_TRACED_PEAK_BYTES = 256 * 1024 * 1024


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _synthetic_references() -> Iterator[_Ref]:
    common_low_information_render = "f" * 64
    for index in range(_FINGERPRINT_COUNT):
        in_overflow_cluster = index < _OVERFLOW_CLUSTER_SIZE
        yield _Ref(
            emoji_id=f"mxe_scalability_{index:06d}",
            profile_id="dedupe-v1",
            role="primary",
            variant_id=None,
            media_sha256="1" * 64,
            byte_size=1,
            duration_ms=0,
            animated=False,
            alpha_mode="opaque",
            color_behavior="fixed",
            decoded_sha256="2" * 64,
            canonical_sha256=(
                _digest(f"cluster-render:{index}")
                if in_overflow_cluster
                else common_low_information_render
            ),
            shape_sha256=(_digest(f"cluster-shape:{index}") if in_overflow_cluster else "e" * 64),
            low_information=not in_overflow_cluster,
            layout=(0,),
            content=(0,),
            alpha=(0,),
            edge=(0,),
        )


def _run_scalability_scenario() -> tuple[float, int, int, int, int, int]:
    profile = load_analysis_profile("dedupe-v1").data
    thresholds = profile["candidate_thresholds"]
    bucket_policy = profile["oversized_bucket_policy"]
    secondary_keys = tuple(str(value) for value in bucket_policy["secondary_keys"])
    band_bits = _lsh_band_bits(secondary_keys)

    tracemalloc.start()
    started = time.perf_counter()
    try:
        index = _build_candidate_index(
            _synthetic_references(),
            thresholds=thresholds,
            bucket_limit=int(bucket_policy["posting_bucket_cap"]),
            secondary_keys=secondary_keys,
            band_bits=band_bits,
        )
        textless = SimpleNamespace(facets=SimpleNamespace(text_content=SimpleNamespace(items=())))
        snapshot = SimpleNamespace(
            emojis={ref.emoji_id: textless for ref in index.refs[:_OVERFLOW_CLUSTER_SIZE]}
        )
        candidate_counts: Counter[str] = Counter()
        comparisons = 0
        for left_index, right_index in index.pairs():
            comparisons += 1
            left = index.refs[left_index]
            right = index.refs[right_index]
            candidate = _compare(left, right, snapshot, thresholds)
            assert candidate is not None
            candidate_counts[left.emoji_id] += 1
            candidate_counts[right.emoji_id] += 1
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        elapsed = time.perf_counter() - started
        tracemalloc.stop()

    default_limit = int(thresholds["candidate_limit_default"])
    assert len(index.refs) == _FINGERPRINT_COUNT
    assert index.suppressed_bucket_count >= 1
    assert index.stats.low_information_refs == _FINGERPRINT_COUNT - _OVERFLOW_CLUSTER_SIZE
    assert comparisons == 231
    assert index.stats.yielded_pairs == comparisons
    assert len(candidate_counts) == _OVERFLOW_CLUSTER_SIZE
    assert set(candidate_counts.values()) == {_OVERFLOW_CLUSTER_SIZE - 1}
    assert all(count > default_limit for count in candidate_counts.values())
    assert index.stats.bucket_entries < 1_000
    assert index.stats.bucket_probes < 10_000
    assert index.stats.candidate_hits < 20_000
    return (
        elapsed,
        peak_bytes,
        index.stats.bucket_entries,
        index.stats.bucket_probes,
        index.stats.candidate_hits,
        comparisons,
    )


def test_100k_fingerprints_use_bounded_production_candidate_index() -> None:
    elapsed, peak_bytes, *_ = _run_scalability_scenario()

    assert elapsed < _MAX_WALL_SECONDS
    assert peak_bytes < _MAX_TRACED_PEAK_BYTES


@pytest.mark.parametrize("positions", [(), (0,), (10,), (21,), (24,), (25,), (0, 10, 21)])
def test_incremental_candidate_discovery_matches_full_scan_in_both_directions(positions):
    profile = load_analysis_profile("dedupe-v1").data
    policy = profile["oversized_bucket_policy"]
    keys = tuple(policy["secondary_keys"])
    refs = [ref for _, ref in zip(range(30), _synthetic_references(), strict=False)]
    # Exercise canonical-only low-information matches as well as LSH/shape
    # matches, and multiple media records belonging to one selected identity.
    refs[24] = replace(refs[24], canonical_sha256=refs[10].canonical_sha256)
    refs[25] = replace(refs[0], role="variant", variant_id="alternate")

    def build():
        return _build_candidate_index(
            refs,
            thresholds=profile["candidate_thresholds"],
            bucket_limit=int(policy["posting_bucket_cap"]),
            secondary_keys=keys,
            band_bits=_lsh_band_bits(keys),
        )

    selected = {refs[position].emoji_id for position in positions}
    full = build()
    full_pairs = set(full.pairs())
    assert not refs[10].low_information and refs[24].low_information
    assert (10, 24) in full_pairs
    expected = {
        pair for pair in full_pairs if any(refs[index].emoji_id in selected for index in pair)
    }
    incremental = build()
    actual = list(incremental.pairs(selected_emoji_ids=selected))
    assert set(actual) == expected
    assert len(actual) == len(set(actual))
    assert all(left < right for left, right in actual)
    assert incremental.stats.bucket_probes <= full.stats.bucket_probes
