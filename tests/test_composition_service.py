import hashlib
from types import SimpleNamespace

import pytest

from mojilex_cli.ai.base import CostEstimate, RequestBudget
from mojilex_cli.composition import service
from test_composition_detector import _split


def setup_tiles(tmp_path):
    _, tiles = _split()
    processed = {}
    for tile in tiles:
        path = tmp_path / f"{tile.member.native_id}.png"
        tile.image.save(path)
        processed[tile.member.native_id] = SimpleNamespace(
            metadata=SimpleNamespace(sha256=tile.member.media_sha256),
            composition_tile_path=path,
            composition_tile_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    source = SimpleNamespace(
        items=[
            SimpleNamespace(
                native_id=t.member.native_id, animated=False, video=False, needs_repainting=False
            )
            for t in tiles
        ]
    )
    return source, processed


@pytest.mark.parametrize("approve", [True, False])
async def test_geometry_never_becomes_verified_without_veto_approval(
    tmp_path, monkeypatch, approve
):
    source, processed = setup_tiles(tmp_path)
    budget = RequestBudget(max_requests=5, allow_unknown_cost=True)
    calls = []

    async def verify(image, **kwargs):
        await kwargs["budget"].reserve(CostEstimate(upper_bound_usd=None, note="test"))
        calls.append(kwargs)
        return approve

    monkeypatch.setattr(service, "verify_composition", verify)
    result = await service.analyze_compositions(
        source, processed, model="test", api_key=None, budget=budget
    )
    assert len(calls) == (3 if approve else 1)
    assert calls[0]["columns"] == 3 and calls[0]["rows"] == 2
    assert budget.requests_used == (3 if approve else 1)
    assert bool(result) == approve
    if result:
        assert result[0].verified and result[0].verifier_model == "test"


async def test_unchanged_verified_evidence_reuses_no_paid_requests(tmp_path, monkeypatch):
    source, processed = setup_tiles(tmp_path)

    async def yes(*args, **kwargs):
        return True

    monkeypatch.setattr(service, "verify_composition", yes)
    first = await service.analyze_compositions(
        source,
        processed,
        model="test",
        api_key=None,
        budget=RequestBudget(max_requests=5, allow_unknown_cost=True),
    )

    async def forbidden(*args, **kwargs):
        pytest.fail("unnecessary paid verification")

    monkeypatch.setattr(service, "verify_composition", forbidden)
    reused = await service.analyze_compositions(
        source,
        processed,
        model="test",
        api_key=None,
        budget=RequestBudget(max_requests=0),
        previous=[g.model_dump() for g in first],
    )
    assert reused == first
    processed["1"].metadata.sha256 = "f" * 64
    invalidated = await service.analyze_compositions(
        source,
        processed,
        model="test",
        api_key=None,
        budget=RequestBudget(max_requests=0),
        previous=[g.model_dump() for g in first],
    )
    assert invalidated == []


@pytest.mark.parametrize("condition", ["video", "repainting", "missing", "hash", "budget"])
async def test_ineligible_or_unverifiable_group_has_no_tags_or_calls(
    tmp_path, monkeypatch, condition
):
    source, processed = setup_tiles(tmp_path)
    if condition == "video":
        for item in source.items:
            item.video = True
    elif condition == "repainting":
        for item in source.items:
            item.needs_repainting = True
    elif condition == "missing":
        for p in processed.values():
            p.composition_tile_path = None
    elif condition == "hash":
        for p in processed.values():
            p.composition_tile_sha256 = "0" * 64

    async def forbidden(*a, **kw):
        pytest.fail("ineligible paid call")

    monkeypatch.setattr(service, "verify_composition", forbidden)
    result = await service.analyze_compositions(
        source,
        processed,
        model="test",
        api_key=None,
        budget=RequestBudget(
            max_requests=0 if condition == "budget" else 5, allow_unknown_cost=True
        ),
    )
    assert result == []


async def test_batch_descriptions_get_budget_before_optional_checks(tmp_path, monkeypatch):
    source, processed = setup_tiles(tmp_path)
    budget = RequestBudget(max_requests=2, allow_unknown_cost=True)
    queue = service.CompositionQueue(model="test")
    calls = []

    async def verify(*args, **kwargs):
        calls.append(True)
        return True

    monkeypatch.setattr(service, "verify_composition", verify)
    await budget.reserve(CostEstimate(upper_bound_usd=None, note="first pack description"))
    await queue.prepare("first", source, processed)
    assert budget.requests_used == 1 and not calls
    await budget.reserve(CostEstimate(upper_bound_usd=None, note="second pack description"))
    await queue.prepare("second", source, processed)
    assert await queue.verify(api_key=None, budget=budget) == {"first": [], "second": []}
    assert not calls


async def test_deferred_check_survives_temporary_tile_cleanup(tmp_path, monkeypatch):
    source, processed = setup_tiles(tmp_path)
    queue = service.CompositionQueue(model="test")
    await queue.prepare("first", source, processed)
    for media in processed.values():
        media.composition_tile_path.unlink()

    async def verify(png, **kwargs):
        assert png.startswith(b"\x89PNG")
        return True

    monkeypatch.setattr(service, "verify_composition", verify)
    result = await queue.verify(
        api_key=None, budget=RequestBudget(max_requests=2, allow_unknown_cost=True)
    )
    assert len(result["first"]) == 1
    assert result["first"][0].verified
    assert queue._pending == [] and queue._bytes == 0


async def test_deferred_png_memory_bound(tmp_path, monkeypatch):
    source, processed = setup_tiles(tmp_path)
    monkeypatch.setattr(service, "assemble", lambda *args: b"x" * (8 * 1024 * 1024))
    queue = service.CompositionQueue(model="test")
    for key in ("one", "two", "three"):
        await queue.prepare(key, source, processed)
    assert len(queue._pending) == 2
    assert queue._bytes == 16 * 1024 * 1024


async def test_deferred_group_count_bound(tmp_path, monkeypatch):
    source, processed = setup_tiles(tmp_path)
    queue = service.CompositionQueue(model="test")
    await queue.prepare("first", source, processed)
    proposal = queue._pending[0][1]

    async def prepare(source, processed, *, enqueue, **kwargs):
        for _ in range(70):
            enqueue(proposal, b"small")
        return []

    monkeypatch.setattr(service, "_prepare_compositions", prepare)
    await queue.prepare("second", source, processed)
    assert len(queue._pending) == 64


@pytest.mark.parametrize("rejected_audit", ["independent_objects", "layout"])
async def test_each_independent_veto_is_binding(tmp_path, monkeypatch, rejected_audit):
    source, processed = setup_tiles(tmp_path)
    calls = []

    async def verify(*args, **kwargs):
        calls.append(kwargs["audit"])
        return kwargs["audit"] != rejected_audit

    monkeypatch.setattr(service, "verify_composition", verify)
    result = await service.analyze_compositions(
        source,
        processed,
        model="test",
        api_key=None,
        budget=RequestBudget(max_requests=10, allow_unknown_cost=True),
    )
    assert result == []
    assert calls[-1] == rejected_audit
    assert len(calls) == len(set(calls))


async def test_budget_exhaustion_cannot_turn_partial_audits_into_approval(tmp_path, monkeypatch):
    source, processed = setup_tiles(tmp_path)
    budget = RequestBudget(max_requests=2, allow_unknown_cost=True)

    async def verify(*args, **kwargs):
        await kwargs["budget"].reserve(CostEstimate(upper_bound_usd=None, note="audit"))
        return True

    monkeypatch.setattr(service, "verify_composition", verify)
    assert (
        await service.analyze_compositions(
            source,
            processed,
            model="test",
            api_key=None,
            budget=budget,
        )
        == []
    )
    assert budget.requests_used == 2


@pytest.mark.parametrize("large_approved", [True, False])
async def test_alternatives_only_overlap_until_verified(tmp_path, monkeypatch, large_approved):
    source, processed = setup_tiles(tmp_path)
    queue = service.CompositionQueue(model="test")
    await queue.prepare("source", source, processed)
    key, large, image = queue._pending[0]
    small = large.model_copy(update={"rows": 1, "members": large.members[:3]})
    queue._pending.append((key, small, image))
    calls = []

    async def verify(*args, **kwargs):
        calls.append(kwargs["rows"])
        return large_approved if kwargs["rows"] == 2 else True

    monkeypatch.setattr(service, "verify_composition", verify)
    result = (await queue.verify(api_key=None, budget=RequestBudget(max_requests=10)))[key]
    assert len(result) == 1
    assert result[0].rows == (2 if large_approved else 1)
    assert result[0].verification_passes == 3
    assert calls == ([2, 2, 2] if large_approved else [2, 1, 1, 1])


async def test_old_single_audit_evidence_is_not_reused(tmp_path, monkeypatch):
    source, processed = setup_tiles(tmp_path)
    queue = service.CompositionQueue(model="test")
    await queue.prepare("source", source, processed)
    old = queue._pending[0][1].model_dump()
    old.update(detector="composition-v1", verified=True, verifier_model="test")
    result = await service.analyze_compositions(
        source,
        processed,
        model="test",
        api_key=None,
        previous=[old],
        budget=RequestBudget(max_requests=0),
    )
    assert result == []


async def test_shared_emoji_across_packs_has_global_ten_audit_limit(tmp_path, monkeypatch):
    source, processed = setup_tiles(tmp_path)
    queue = service.CompositionQueue(model="test")
    for key in ("one", "two", "three", "four", "five"):
        await queue.prepare(key, source, processed)
    budget = RequestBudget(max_requests=100, allow_unknown_cost=True)

    async def verify(*args, **kwargs):
        await kwargs["budget"].reserve(CostEstimate(upper_bound_usd=None, note="audit"))
        return True

    monkeypatch.setattr(service, "verify_composition", verify)
    result = await queue.verify(api_key=None, budget=budget)
    assert budget.requests_used == 10
    assert [len(groups) for groups in result.values()] == [1, 1, 1, 0, 0]
