"""Optional post-description composition checks sharing the existing AI budget."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping

from mojilex_cli.ai.base import RequestBudget
from mojilex_cli.concurrency import run_blocking
from mojilex_cli.media.models import ProcessedMedia
from mojilex_cli.sources.base import SourceCollection

from .detector import VERSION, Composition, Tile, assemble, candidates, load_tile
from .verifier import verify_composition


async def _prepare_compositions(
    source: SourceCollection,
    processed: Mapping[str, ProcessedMedia],
    *,
    model: str,
    enqueue: Callable[[Composition, bytes], None],
    previous: object = None,
) -> list[Composition]:
    """Fail closed without changing normal descriptions or requiring new consent."""
    tiles: list[Tile] = []
    accepted: list[Composition] = []
    try:
        if len(source.items) > 256:
            return []
        for item in source.items:
            media = processed.get(item.native_id)
            if item.animated or item.video or item.needs_repainting or media is None:
                continue
            if media.composition_tile_path is None or media.composition_tile_sha256 is None:
                continue
            tile = load_tile(
                item.native_id,
                media.metadata.sha256,
                media.composition_tile_path,
                media.composition_tile_sha256,
            )
            if tile is not None:
                tiles.append(tile)
        mapping = {tile.member.native_id: tile for tile in tiles}
        reusable: list[Composition] = []
        if isinstance(previous, list) and len(previous) <= 64:
            for raw in previous:
                try:
                    old = Composition.model_validate(raw)
                    if (
                        old.verified
                        and old.verifier_model == model
                        and old.detector in {"composition-v2", VERSION}
                        and old.verification_passes == 3
                        and all(
                            member.native_id in mapping
                            and mapping[member.native_id].member == member
                            for member in old.members
                        )
                    ):
                        reusable.append(old)
                except ValueError:
                    continue
        occurrences: dict[str, int] = {}
        for old in reusable:
            for member in old.members:
                occurrences[member.native_id] = occurrences.get(member.native_id, 0) + 1
        accepted = [
            old for old in reusable if all(occurrences[m.native_id] == 1 for m in old.members)
        ]
        # Drain the worker on cancellation before closing the images it reads.
        proposals = await run_blocking(candidates, tiles)
        for proposal in proposals:
            prior = next(
                (
                    old
                    for old in accepted
                    if old.columns == proposal.columns
                    and old.rows == proposal.rows
                    and old.members == proposal.members
                ),
                None,
            )
            if prior is not None:
                continue
            enqueue(proposal, assemble(proposal, mapping))
        return accepted
    except Exception:
        # Composition is optional; decoder/cache/model trouble must not break a pack.
        return accepted
    finally:
        for tile in tiles:
            tile.image.close()


def _preserves_layout(old: Composition, new: Composition) -> bool:
    """An extension must preserve every old tile, hash and relative coordinate."""
    if len(old.members) >= len(new.members):
        return False
    positions = {
        m.native_id: (i % new.columns, i // new.columns, m) for i, m in enumerate(new.members)
    }
    offsets = set()
    for i, member in enumerate(old.members):
        point = positions.get(member.native_id)
        if point is None or point[2] != member:
            return False
        offsets.add((point[0] - i % old.columns, point[1] - i // old.columns))
    return len(offsets) == 1


class CompositionQueue:
    """Bounded proposals checked as soon as their source descriptions finish."""

    def __init__(self, *, model: str) -> None:
        self.model = model
        self._pending: list[tuple[str, Composition, bytes]] = []
        self._bytes = 0
        self._locks: dict[str, asyncio.Lock] = {}
        self._attempts: dict[str, int] = {}
        self.accepted: dict[str, list[Composition]] = {}

    def _lock(self, key: str) -> asyncio.Lock:
        return self._locks.setdefault(key, asyncio.Lock())

    def _release(self, selected: list[tuple[str, Composition, bytes]]) -> None:
        # In-flight proposals stay reserved in _pending until their owner exits.
        # Identity matters: equal proposals prepared later have another owner.
        identities = {id(item) for item in selected}
        self._pending = [item for item in self._pending if id(item) not in identities]
        self._bytes = sum(len(png) for _, _, png in self._pending)

    async def prepare(
        self,
        key: str,
        source: SourceCollection,
        processed: Mapping[str, ProcessedMedia],
        *,
        previous: object = None,
    ) -> list[Composition]:
        async with self._lock(key):
            added: list[tuple[str, Composition, bytes]] = []

            def enqueue(group: Composition, png: bytes) -> None:
                if len(self._pending) >= 64 or self._bytes + len(png) > 16 * 1024 * 1024:
                    return
                item = (key, group, png)
                self._pending.append(item)
                added.append(item)
                self._bytes += len(png)

            try:
                self.accepted[key] = await _prepare_compositions(
                    source, processed, model=self.model, previous=previous, enqueue=enqueue
                )
            except BaseException:
                self._release(added)
                raise
            return self.accepted[key]

    async def verify(
        self, *, api_key: str | None, budget: RequestBudget, key: str | None = None
    ) -> dict[str, list[Composition]]:
        """Check one ready source, or drain the currently queued sources in order.

        Different keys can run concurrently through the existing shared AI slots.
        A source's preparation and verification serialize to protect its evidence.
        The ten-audit limit belongs to this queue, including repeated calls.
        """
        if key is not None:
            await self._verify_key(key, api_key=api_key, budget=budget)
        else:
            for source_key in dict.fromkeys(item[0] for item in self._pending):
                await self._verify_key(source_key, api_key=api_key, budget=budget)
        return self.accepted

    async def _verify_key(self, key: str, *, api_key: str | None, budget: RequestBudget) -> None:
        async with self._lock(key):
            selected = [item for item in self._pending if item[0] == key]
            await self._verify_selected(selected, api_key=api_key, budget=budget)

    async def _verify_selected(
        self,
        selected: list[tuple[str, Composition, bytes]],
        *,
        api_key: str | None,
        budget: RequestBudget,
    ) -> None:
        try:
            for key, proposal, png in selected:
                ids = {m.native_id for m in proposal.members}
                overlapping = [
                    old
                    for old in self.accepted[key]
                    if ids.intersection(m.native_id for m in old.members)
                ]
                if any(not _preserves_layout(old, proposal) for old in overlapping):
                    continue
                if budget.max_requests is not None and budget.requests_used >= budget.max_requests:
                    break
                try:
                    approved = True
                    for audit in ("continuity", "independent_objects", "layout"):
                        ids = {member.native_id for member in proposal.members}
                        if any(self._attempts.get(native_id, 0) >= 10 for native_id in ids):
                            approved = False
                            break
                        for native_id in ids:
                            self._attempts[native_id] = self._attempts.get(native_id, 0) + 1
                        if not await verify_composition(
                            png,
                            model=self.model,
                            api_key=api_key,
                            budget=budget,
                            columns=proposal.columns,
                            rows=proposal.rows,
                            audit=audit,
                        ):
                            approved = False
                            break
                    if approved:
                        verified = Composition.model_validate(
                            {
                                **proposal.model_dump(),
                                "verified": True,
                                "verifier_model": self.model,
                                "verification_passes": 3,
                            }
                        )
                        self.accepted[key] = [
                            old for old in self.accepted[key] if old not in overlapping
                        ] + [verified]
                except Exception:
                    continue
        finally:
            self._release(selected)


async def analyze_compositions(
    source: SourceCollection,
    processed: Mapping[str, ProcessedMedia],
    *,
    model: str,
    api_key: str | None,
    budget: RequestBudget,
    previous: object = None,
) -> list[Composition]:
    """Single-source convenience API using the same bounded verification queue."""
    queue = CompositionQueue(model=model)
    await queue.prepare("source", source, processed, previous=previous)
    return (await queue.verify(api_key=api_key, budget=budget))["source"]
