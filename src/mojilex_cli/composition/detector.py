"""Find unambiguous rectangular assemblies from verified, unpadded static tiles.

Geometry only proposes candidates. Callers must obtain an independent veto check
before exposing a composition. No candidate is an emoji-level fragment label.
"""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mojilex_cli.media.resume import _read

VERSION = "composition-v3"
MAX_TILES = 256


class Member(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    native_id: str = Field(pattern=r"^[0-9]{1,32}$")
    media_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    tile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Composition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    detector: str = Field(default=VERSION, pattern=r"^composition-v[123]$")
    columns: int = Field(ge=1, le=8)
    rows: int = Field(ge=1, le=8)
    members: list[Member] = Field(min_length=2, max_length=24)
    verified: bool = False
    verifier_model: str | None = Field(default=None, max_length=128)
    verification_passes: int = Field(default=0, ge=0, le=3)

    @model_validator(mode="after")
    def valid_grid(self) -> Composition:
        if len(self.members) != self.columns * self.rows:
            raise ValueError("incomplete grid")
        if len({m.native_id for m in self.members}) != len(self.members):
            raise ValueError("duplicate tile")
        if self.verified and not self.verifier_model:
            raise ValueError("verified group requires verifier model")
        if self.verified and self.detector != "composition-v1" and self.verification_passes != 3:
            raise ValueError("composition requires all three verification passes")
        return self


@dataclass(frozen=True)
class Tile:
    member: Member
    image: Image.Image


def load_tile(native_id: str, media_sha256: str, path: Path, expected: str) -> Tile | None:
    """Read only a bounded checksummed PNG produced by the sandbox worker."""
    try:
        raw = _read(path, 512 * 1024)
        if hashlib.sha256(raw).hexdigest() != expected:
            return None
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != "PNG" or im.mode != "RGBA" or im.width != im.height:
                return None
            if not 32 <= im.width <= 256 or getattr(im, "n_frames", 1) != 1:
                return None
            im.load()
            image = im.copy()
        return Tile(
            Member(native_id=native_id, media_sha256=media_sha256, tile_sha256=expected), image
        )
    except (OSError, ValueError, Image.DecompressionBombError):
        return None


def _edge(image: Image.Image, side: str) -> list[tuple[int, int, int, int]]:
    n = image.width
    pixels = image.load()
    assert pixels is not None
    points = [min(n - 1, int((i + 0.5) * n / 48)) for i in range(48)]
    coords = {
        "left": [(0, t) for t in points],
        "right": [(n - 1, t) for t in points],
        "top": [(t, 0) for t in points],
        "bottom": [(t, n - 1) for t in points],
    }[side]
    return [cast(tuple[int, int, int, int], pixels[x, y]) for x, y in coords]


def seam_score(
    a: Sequence[tuple[int, int, int, int]], b: Sequence[tuple[int, int, int, int]]
) -> float:
    """Transparent, featureless and poorly aligned boundaries cannot be evidence."""
    active = [(x, y) for x, y in zip(a, b, strict=True) if min(x[3], y[3]) >= 128]
    if len(active) < len(a) * 0.125:
        return math.inf
    alpha_error = sum(abs(x[3] - y[3]) for x, y in zip(a, b, strict=True)) / (len(a) * 255)
    if alpha_error > 0.25:
        return math.inf
    # Variation must exist along BOTH edges, not just across different flat colors.
    for side in (0, 1):
        values = [sum(pair[side][:3]) / 3 for pair in active]
        mean = sum(values) / len(values)
        deviation = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if deviation < 5:
            return math.inf
    color_error = sum(sum(abs(x[c] - y[c]) for c in range(3)) for x, y in active) / (
        len(active) * 3 * 255
    )
    if color_error > 0.10:
        return math.inf
    return color_error + alpha_error * 0.75


def _candidates(tiles: Sequence[Tile], *, smooth: bool) -> list[Composition]:
    """Propose connected grids; resolve weak edges using the whole layout.

    Every member needs a measured seam. No blank padding, rotation, source-order
    assumption or invented tile is allowed. The independent AI veto is mandatory.
    """
    if not 2 <= len(tiles) <= MAX_TILES:
        return []
    if len({t.member.native_id for t in tiles}) != len(tiles):
        return []
    edges = [
        {side: _edge(tile.image, side) for side in ("left", "right", "top", "bottom")}
        for tile in tiles
    ]
    if smooth:
        # A second scale suppresses pixel/compression noise in leaf/rock detail.
        # This affects proposals only; the verifier always sees original pixels.
        edges = [
            {
                side: [
                    cast(
                        tuple[int, int, int, int],
                        tuple(
                            round(
                                sum(p[c] for p in values[max(0, i - 1) : i + 2])
                                / len(values[max(0, i - 1) : i + 2])
                            )
                            for c in range(4)
                        ),
                    )
                    for i in range(len(values))
                ]
                for side, values in tile_edges.items()
            }
            for tile_edges in edges
        ]
    links: list[tuple[float, int, int, int, int, bool]] = []
    for outgoing, incoming in (("right", "left"), ("bottom", "top")):
        scores: dict[tuple[int, int], float] = {}
        for i, tile in enumerate(tiles):
            for j, other in enumerate(tiles):
                if i != j and tile.image.size == other.image.size:
                    scores[i, j] = seam_score(edges[i][outgoing], edges[j][incoming])
        forward: dict[int, tuple[int, bool]] = {}
        backward: dict[int, tuple[int, bool]] = {}
        for reverse, output in ((False, forward), (True, backward)):
            for i in range(len(tiles)):
                ranked = sorted(
                    (s, a if reverse else b)
                    for (a, b), s in scores.items()
                    if (b if reverse else a) == i and math.isfinite(s)
                )
                if not ranked:
                    continue
                best, index = ranked[0]
                unique = len(ranked) == 1 or ranked[1][0] >= best * 1.6 + 0.012
                # Exact/near ties provide no directional evidence, even in a grid.
                if len(ranked) > 1 and ranked[1][0] - best < 0.005:
                    continue
                output[i] = (index, unique)
        for i, (j, unique) in forward.items():
            reciprocal = backward.get(j)
            if reciprocal is not None and reciprocal[0] == i:
                links.append(
                    (
                        scores[i, j],
                        i,
                        j,
                        int(outgoing == "right"),
                        int(outgoing == "bottom"),
                        unique and reciprocal[1],
                    )
                )

    # Merge the strongest links first. A conflicting coordinate/cycle is rejected,
    # rather than sacrificing a consistent layout to an uncertain dark boundary.
    layouts = {i: {i: (0, 0)} for i in range(len(tiles))}
    owners = list(range(len(tiles)))
    strong = {i: True for i in range(len(tiles))}
    joined: list[tuple[int, int, bool]] = []
    proposals: dict[tuple[int, ...], tuple[int, int]] = {}
    for score, i, j, dx, dy, unique in sorted(links):
        left, right = owners[i], owners[j]
        if left == right:
            continue
        a, b = layouts[left], layouts[right]
        ox, oy = a[i][0] + dx - b[j][0], a[i][1] + dy - b[j][1]
        shifted = {k: (x + ox, y + oy) for k, (x, y) in b.items()}
        if set(a.values()) & set(shifted.values()):
            continue
        merged = {**a, **shifted}
        xs, ys = zip(*merged.values(), strict=True)
        width, height = max(xs) - min(xs) + 1, max(ys) - min(ys) + 1
        if width > 8 or height > 8 or width * height > 24:
            continue
        layouts[left] = merged
        joined.append((i, j, unique and score <= 0.13))
        for k in b:
            owners[k] = left
        strong[left] = strong[left] and strong[right] and unique and score <= 0.13
        # A strip has no perpendicular context: retain unique detailed seams.
        if len(merged) == width * height and (min(width, height) > 1 or strong[left]):
            grid = tuple(sorted(merged, key=lambda k: (merged[k][1], merged[k][0])))
            proposals[grid] = width, height

    # An irregular silhouette or an uncertain extension must not erase a complete
    # rectangle inside the component. Enumerate bounded connected subrectangles.
    for owner in set(owners):
        layout = layouts[owner]
        cells = {point: index for index, point in layout.items()}
        for x, y in cells:
            for width in range(1, 9):
                for height in range(1, 9):
                    if not 2 <= width * height <= 24:
                        continue
                    coords = [(x + dx, y + dy) for dy in range(height) for dx in range(width)]
                    if not all(point in cells for point in coords):
                        continue
                    grid = tuple(cells[point] for point in coords)
                    members = set(grid)
                    adjacency: dict[int, set[int]] = {k: set() for k in grid}
                    for first, second, reliable in joined:
                        if (
                            first in members
                            and second in members
                            and (min(width, height) > 1 or reliable)
                        ):
                            adjacency[first].add(second)
                            adjacency[second].add(first)
                    reached, todo = set(), [grid[0]]
                    while todo:
                        k = todo.pop()
                        if k not in reached:
                            reached.add(k)
                            todo.extend(adjacency[k] - reached)
                    if reached == members:
                        proposals[grid] = width, height

    # Keep disjoint maximal proposals within this scale. The public search
    # retains alternatives across scales and proposes extensions of these groups.
    chosen: list[Composition] = []
    used: set[int] = set()
    for grid, (columns, rows) in sorted(proposals.items(), key=lambda p: (-len(p[0]), p[0])):
        if used.intersection(grid):
            continue
        used.update(grid)
        chosen.append(
            Composition(columns=columns, rows=rows, members=[tiles[i].member for i in grid])
        )
    return chosen[:64]


def _extensions(groups: Sequence[Composition], tiles: Mapping[str, Tile]) -> list[Composition]:
    """Join complete subassemblies using multiple corresponding boundary seams.

    An individual edge need not win a pack-wide nearest-neighbour contest: the
    already established row supplies context. Never infer gaps or add blank tiles.
    """
    found: list[Composition] = []
    for first in groups:
        first_ids = {m.native_id for m in first.members}
        for second in groups:
            if first_ids.intersection(m.native_id for m in second.members):
                continue
            for vertical in (False, True):
                if vertical:
                    if first.columns != second.columns or first.columns < 2:
                        continue
                    columns, rows = first.columns, first.rows + second.rows
                    boundary = zip(first.members[-columns:], second.members[:columns], strict=True)
                    members = first.members + second.members
                    outgoing, incoming = "bottom", "top"
                else:
                    if first.rows != second.rows or first.rows < 2:
                        continue
                    columns, rows = first.columns + second.columns, first.rows
                    boundary = zip(
                        first.members[first.columns - 1 :: first.columns],
                        second.members[:: second.columns],
                        strict=True,
                    )
                    members = [
                        member
                        for row in range(rows)
                        for member in (
                            first.members[row * first.columns : (row + 1) * first.columns]
                            + second.members[row * second.columns : (row + 1) * second.columns]
                        )
                    ]
                    outgoing, incoming = "right", "left"
                if columns > 8 or rows > 8 or columns * rows > 24:
                    continue
                scores = []
                for a, b in boundary:
                    left, right = tiles[a.native_id].image, tiles[b.native_id].image
                    if left.size != right.size:
                        break
                    edges = [_edge(left, outgoing), _edge(right, incoming)]
                    smooth = [
                        [
                            cast(
                                tuple[int, int, int, int],
                                tuple(
                                    round(
                                        sum(p[c] for p in edge[max(0, i - 1) : i + 2])
                                        / len(edge[max(0, i - 1) : i + 2])
                                    )
                                    for c in range(4)
                                ),
                            )
                            for i in range(len(edge))
                        ]
                        for edge in edges
                    ]
                    score = min(seam_score(*edges), seam_score(*smooth))
                    if not math.isfinite(score):
                        break
                    scores.append(score)
                expected = columns if vertical else rows
                if len(scores) == expected and sum(scores) / expected <= 0.18:
                    found.append(Composition(columns=columns, rows=rows, members=members))
    return found


def candidates(tiles: Sequence[Tile]) -> list[Composition]:
    """Search two scales, retaining bounded alternatives for the AI veto.

    A rejected large assembly may have a valid smaller alternative. Membership
    becomes exclusive only after verification, never during speculative search.
    """
    ordered = sorted(tiles, key=lambda tile: tile.member.native_id)
    found = {
        (group.columns, group.rows, tuple(m.native_id for m in group.members)): group
        for smooth in (False, True)
        for group in _candidates(ordered, smooth=smooth)
    }

    def order(group: Composition) -> tuple[int, tuple[str, ...], int]:
        return -len(group.members), tuple(m.native_id for m in group.members), group.columns

    base = sorted(found.values(), key=order)[:64]
    counts: dict[str, int] = {}
    for group in base:
        for member in group.members:
            counts[member.native_id] = counts.get(member.native_id, 0) + 1
    additions = _extensions(base, {tile.member.native_id: tile for tile in ordered})
    for group in sorted(additions, key=order):
        key = group.columns, group.rows, tuple(m.native_id for m in group.members)
        if key in found or len(base) >= 64 or any(counts[m.native_id] >= 3 for m in group.members):
            continue
        found[key] = group
        base.append(group)
        for member in group.members:
            counts[member.native_id] += 1
    # Preserve the original alternatives for a veto/budget fallback.
    return sorted(base, key=order)


def assemble(group: Composition, tiles: Mapping[str, Tile]) -> bytes:
    if len(group.members) != group.columns * group.rows:
        raise ValueError("composition grid is incomplete")
    selected = [tiles[m.native_id] for m in group.members]
    size = selected[0].image.width
    if any(tile.image.size != (size, size) for tile in selected):
        raise ValueError("composition tile sizes differ")
    canvas = Image.new("RGBA", (group.columns * size, group.rows * size), (0, 0, 0, 0))
    for index, tile in enumerate(selected):
        canvas.paste(tile.image, ((index % group.columns) * size, (index // group.columns) * size))
    output = io.BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()
