"""Compatibility oracle for the original cost/path-length DTW contract."""

from random import Random

import pytest

from mojilex_cli.dedupe import engine


def _reference_dtw(left, right, *, band):
    paths = {(0, 0): (0, ())}
    size = len(left)
    for i in range(1, size + 1):
        for j in range(max(1, i - band), min(size, i + band) + 1):
            candidates = [
                paths[key] for key in ((i - 1, j), (i, j - 1), (i - 1, j - 1)) if key in paths
            ]
            if candidates:
                cost, values = min(candidates, key=lambda item: (item[0], len(item[1])))
                distance = (left[i - 1] ^ right[j - 1]).bit_count()
                paths[(i, j)] = cost + distance, (*values, distance)
    return paths.get((size, size), (64 * size, (64,) * size))[1]


def _reference_alignment(left, right, *, animated, cyclic, dtw_band, reverse):
    if len(left) != len(right) or not left:
        return (64,)
    if not animated or len(left) == 1:
        return tuple((a ^ b).bit_count() for a, b in zip(left, right, strict=True))
    candidates = []
    orientations = (right, tuple(reversed(right))) if reverse else (right,)
    for orientation in orientations:
        for shift in range(len(right)) if cyclic else range(1):
            shifted = orientation[shift:] + orientation[:shift]
            direct = tuple((a ^ b).bit_count() for a, b in zip(left, shifted, strict=True))
            dtw = _reference_dtw(left, shifted, band=dtw_band)
            candidates.extend(((sum(direct), direct), (sum(dtw), dtw)))
    return min(candidates, key=lambda item: (item[0], item[1]))[1]


def test_dtw_preserves_full_distance_paths_and_stable_ties():
    random = Random(716)
    for size in (0, 1, 2, 3, 8, 16):
        for band in (-1, 0, 1, 2, 4, 16):
            pairs = [((0,) * size, (0,) * size), ((0,) * size, (1,) * size)]
            pairs.extend(
                (
                    tuple(random.getrandbits(64) for _ in range(size)),
                    tuple(random.getrandbits(64) for _ in range(size)),
                )
                for _ in range(60)
            )
            for left, right in pairs:
                assert engine._banded_dtw(left, right, band=band) == _reference_dtw(
                    left, right, band=band
                )
    # The internal helper's exact behavior also survives wider integer inputs.
    assert engine._banded_dtw((0, 0), (1 << 512, (1 << 513) - 1), band=1) == _reference_dtw(
        (0, 0), (1 << 512, (1 << 513) - 1), band=1
    )


@pytest.mark.parametrize("animated", [False, True])
@pytest.mark.parametrize("cyclic", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtw_band", [0, 2, 16])
def test_alignment_preserves_original_full_outputs(animated, cyclic, reverse, dtw_band):
    random = Random(717)
    pairs = [
        ((), ()),
        ((1,), (2,)),
        ((1, 2), (1,)),
        ((0,) * 16, (0,) * 16),
        ((1, 2) * 8, (2, 1) * 8),
        (tuple(range(16)), (2,) * 16),
        (
            tuple(random.getrandbits(64) for _ in range(16)),
            tuple(random.getrandbits(64) for _ in range(16)),
        ),
    ]
    options = dict(animated=animated, cyclic=cyclic, reverse=reverse, dtw_band=dtw_band)
    for left, right in pairs:
        assert engine._aligned_distances(left, right, **options) == _reference_alignment(
            left, right, **options
        )


@pytest.mark.parametrize("right,unique_rotations", [((1,) * 16, 1), ((1, 2) * 8, 2)])
def test_periodic_hashes_compute_each_distinct_rotation_once(monkeypatch, right, unique_rotations):
    left = tuple(range(16))
    expected = _reference_alignment(
        left, right, animated=True, cyclic=True, dtw_band=2, reverse=True
    )
    original = engine._banded_dtw
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "_banded_dtw", counted)
    assert (
        engine._aligned_distances(left, right, animated=True, cyclic=True, dtw_band=2, reverse=True)
        == expected
    )
    assert calls == unique_rotations
