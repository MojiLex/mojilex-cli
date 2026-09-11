from __future__ import annotations

import hashlib
import time
import tracemalloc
from collections import Counter
from collections.abc import Iterator
from types import SimpleNamespace

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
