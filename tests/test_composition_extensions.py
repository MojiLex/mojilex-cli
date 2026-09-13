import hashlib
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from mojilex_cli.ai.base import CostEstimate, RequestBudget
from mojilex_cli.composition import service
from mojilex_cli.composition.detector import Composition, _extensions
from test_composition_v2 import _split, _tile


def _groups(tiles):
    return [
        Composition(columns=2, rows=1, members=[tile.member for tile in tiles[start : start + 2]])
        for start in (0, 2)
    ]


def _verified(group):
    return group.model_copy(
        update={
            "verified": True,
            "verifier_model": "test",
            "verification_passes": 3,
            "detector": "composition-v2",
        }
    )


def _queue():
    tiles = _split(2, 2)
    old = [_verified(group) for group in _groups(tiles)]
    unrelated = _verified(
        Composition(
            columns=2,
            rows=1,
            members=[
                tiles[0].member.model_copy(update={"native_id": "90"}),
                tiles[1].member.model_copy(update={"native_id": "91"}),
            ],
        )
    )
    large = Composition(columns=2, rows=2, members=[tile.member for tile in tiles])
    queue = service.CompositionQueue(model="test")
    queue.accepted["pack"] = [*old, unrelated]
    queue._pending = [("pack", large, b"only-a-mocked-verifier-reads-this")]
    return queue, old, unrelated, large


def test_two_supported_rows_extend_to_one_complete_picture() -> None:
    tiles = _split(2, 2)
    found = _extensions(_groups(tiles), {tile.member.native_id: tile for tile in tiles})
    assert any(
        group.columns == group.rows == 2
        and [member.native_id for member in group.members] == ["1", "2", "3", "4"]
        for group in found
    )


def test_unrelated_icon_rows_cannot_extend_by_blank_borders() -> None:
    tiles = []
    for index in range(4):
        image = Image.new("RGBA", (64, 64), "white")
        ImageDraw.Draw(image).ellipse((8, 8, 56, 56), fill=(index * 60, 30, 90, 255))
        tiles.append(_tile(image, index + 1))
    assert _extensions(_groups(tiles), {tile.member.native_id: tile for tile in tiles}) == []


@pytest.mark.parametrize("failure", [0, 1, 2, "exception"])
async def test_each_veto_failure_keeps_the_original_verified_groups(monkeypatch, failure) -> None:
    queue, old, unrelated, _ = _queue()
    calls = []

    async def veto(image, **kwargs):
        assert queue.accepted["pack"] == [*old, unrelated]
        await kwargs["budget"].reserve(CostEstimate(upper_bound_usd=None, note="test"))
        calls.append(kwargs["audit"])
        if failure == "exception":
            raise RuntimeError("provider unavailable")
        return len(calls) - 1 != failure

    monkeypatch.setattr(service, "verify_composition", veto)
    result = await queue.verify(
        api_key=None, budget=RequestBudget(max_requests=3, allow_unknown_cost=True)
    )
    assert result["pack"] == [*old, unrelated]
    assert len(calls) == (1 if failure == "exception" else failure + 1)
    assert queue._pending == []


@pytest.mark.parametrize("limit", [0, 1, 2])
async def test_incomplete_audit_budget_preserves_baseline(monkeypatch, limit: int) -> None:
    queue, old, unrelated, _ = _queue()

    async def veto(image, **kwargs):
        await kwargs["budget"].reserve(CostEstimate(upper_bound_usd=None, note="test"))
        return True

    monkeypatch.setattr(service, "verify_composition", veto)
    budget = RequestBudget(max_requests=limit, allow_unknown_cost=True)
    assert (await queue.verify(api_key=None, budget=budget))["pack"] == [*old, unrelated]
    assert budget.requests_used == limit


async def test_three_successes_atomically_replace_only_contained_groups(monkeypatch) -> None:
    queue, old, unrelated, large = _queue()
    calls = []

    async def veto(image, **kwargs):
        assert queue.accepted["pack"] == [*old, unrelated]
        calls.append(kwargs["audit"])
        await kwargs["budget"].reserve(CostEstimate(upper_bound_usd=None, note="test"))
        return True

    monkeypatch.setattr(service, "verify_composition", veto)
    accepted = (
        await queue.verify(
            api_key=None, budget=RequestBudget(max_requests=3, allow_unknown_cost=True)
        )
    )["pack"]
    assert calls == ["continuity", "independent_objects", "layout"]
    assert accepted[0] == unrelated
    assert accepted[1].members == large.members
    assert accepted[1].verified and accepted[1].verification_passes == 3
    ids = [member.native_id for group in accepted for member in group.members]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("change", ["partial", "reordered", "hash"])
async def test_conflicting_extension_never_replaces_old_layout(monkeypatch, change: str) -> None:
    queue, old, unrelated, large = _queue()
    members = list(large.members)
    if change == "partial":
        members[3] = members[3].model_copy(update={"native_id": "99"})
    elif change == "reordered":
        members[1], members[2] = members[2], members[1]
    else:
        members[0] = members[0].model_copy(update={"media_sha256": "f" * 64})
    queue._pending = [("pack", large.model_copy(update={"members": members}), b"mock")]

    async def forbidden(*args, **kwargs):
        pytest.fail("incompatible layout must not spend API budget")

    monkeypatch.setattr(service, "verify_composition", forbidden)
    assert (await queue.verify(api_key=None, budget=RequestBudget(max_requests=5)))["pack"] == [
        *old,
        unrelated,
    ]


@pytest.mark.parametrize(
    "changed,failure",
    [(False, None), (True, None), (False, "candidates"), (False, "assemble")],
)
async def test_prepare_restores_current_old_groups_without_old_candidates(
    tmp_path,
    monkeypatch,
    changed: bool,
    failure: str | None,
) -> None:
    tiles = _split(2, 2)
    processed = {}
    for tile in tiles:
        path = tmp_path / f"{tile.member.native_id}.png"
        tile.image.save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        object.__setattr__(tile, "member", tile.member.model_copy(update={"tile_sha256": digest}))
        processed[tile.member.native_id] = SimpleNamespace(
            metadata=SimpleNamespace(sha256=tile.member.media_sha256),
            composition_tile_path=path,
            composition_tile_sha256=digest,
        )
    old = [_verified(group) for group in _groups(tiles)]
    if changed:
        processed["1"].metadata.sha256 = "f" * 64
    source = SimpleNamespace(
        items=[
            SimpleNamespace(
                native_id=tile.member.native_id,
                animated=False,
                video=False,
                needs_repainting=False,
            )
            for tile in tiles
        ]
    )
    large = Composition(columns=2, rows=2, members=[tile.member for tile in tiles])
    monkeypatch.setattr(service, "candidates", lambda tiles: [large])
    if failure is not None:

        def unavailable(*args, **kwargs):
            raise RuntimeError("optional composition processing failed")

        monkeypatch.setattr(service, failure, unavailable)
    queue = service.CompositionQueue(model="test")
    accepted = await queue.prepare(
        "pack", source, processed, previous=[g.model_dump() for g in old]
    )
    assert accepted == (old[1:] if changed else old)
    assert len(queue._pending) == (0 if failure else 1)
