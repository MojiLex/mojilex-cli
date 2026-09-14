"""Find durable imports before creating another download operation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path

from mojilex_cli.config import MojiLexConfig
from mojilex_cli.git import GitRunner
from mojilex_cli.pipeline.runner import _reference_from_remote
from mojilex_cli.runs import ElementCheckpoint, RunCheckpoint
from mojilex_cli.runs.pack_scope import source_state

from .packs import _pack_elements, _runs, _source_name, _sources


def import_complete(checkpoint: RunCheckpoint, source: str) -> bool:
    state = source_state(checkpoint, source)
    if state["status"] in {"succeeded", "noop"}:
        return True
    if state["phase"] != "describe":
        return False
    elements = _pack_elements(checkpoint, _source_name(source))
    return bool(elements) and all(item.fingerprint_complete for item in elements.values())


def _source_progress_index(
    checkpoint: RunCheckpoint, names: Sequence[str]
) -> dict[str, tuple[int, int, bool]]:
    """Count only this pack's durable work, visiting its membership IDs once."""
    raw = checkpoint.safe_parameters.get("source_memberships", {})
    memberships = (
        {name.casefold(): ids for name, ids in raw.items() if isinstance(name, str)}
        if isinstance(raw, dict)
        else {}
    )
    progress: dict[str, tuple[int, int, bool]] = {}
    for name in names:
        members = memberships.get(name)
        if isinstance(members, (list, tuple)):
            identifiers = {value for value in members if isinstance(value, str)}
            elements: Iterable[ElementCheckpoint | None] = (
                checkpoint.elements.get(identifier) for identifier in identifiers
            )
            total = len(identifiers)
        elif len(names) == 1:
            elements = checkpoint.elements.values()
            total = len(checkpoint.elements)
        else:
            elements = ()
            total = 0
        ai = 0
        media = 0
        for element in elements:
            if element is None:
                continue
            ai += int(
                bool(getattr(element, "ai_facets_complete", False))
                and bool(getattr(element, "ai_cache_key", None))
            )
            media += int(element.fingerprint_complete)
        progress[name] = ai, media, total > 0 and media == total
    return progress


def reusable_imports(
    sources: Sequence[str], config: MojiLexConfig, *, max_items: int | None
) -> dict[str, tuple[RunCheckpoint, str]]:
    """Match the repository and source, keeping the original run and its ledger."""
    runs, _ = _runs(config)
    if not runs:
        return {}
    target = config.repository.target
    path = Path(target).expanduser()
    if path.is_dir():
        target = str(_reference_from_remote(GitRunner(path.resolve()).remote_url()))
    # Index each durable run once. Large input files must not repeat filesystem
    # probes and scan all of a run's sources for every requested link.
    candidates: dict[str, tuple[RunCheckpoint, str]] = {}
    ranks: dict[str, tuple[bool, int, int, bool, datetime]] = {}
    for run in runs:
        if run.target_repository.casefold() != target.casefold():
            continue
        if run.safe_parameters.get("max_items") != max_items:
            continue
        staging = run.safe_parameters.get("staging_repository")
        if (
            not isinstance(staging, str)
            or not Path(staging).is_absolute()
            or not Path(staging).is_dir()
        ):
            continue
        saved_sources: dict[str, str] = {}
        for saved in _sources(run):
            name = _source_name(saved)
            if name is not None:
                saved_sources.setdefault(name.casefold(), saved)
        progress = _source_progress_index(run, tuple(saved_sources))
        for name, saved in saved_sources.items():
            state = source_state(run, saved)
            complete = state["status"] in {"succeeded", "noop"}
            ai, media, media_complete = progress[name]
            rank = (
                state["phase"] == "describe" and complete,
                ai,
                media,
                media_complete or complete,
                run.updated_at,
            )
            if name not in ranks or rank > ranks[name]:
                ranks[name] = rank
                candidates[name] = run, saved
    found: dict[str, tuple[RunCheckpoint, str]] = {}
    for source in sources:
        name = _source_name(source)
        match = candidates.get(name.casefold()) if name is not None else None
        if match is not None:
            # An import's succeeded status may mean metadata only. Keep the copy
            # with the most durable per-pack work before considering recency.
            found[source] = match
    return found
