"""Find durable imports before creating another download operation."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

from mojilex_cli.concurrency import bounded_map
from mojilex_cli.config import MojiLexConfig, load_credentials
from mojilex_cli.dataset import load_dataset
from mojilex_cli.dataset.repository import DatasetSnapshot
from mojilex_cli.git import GitRunner
from mojilex_cli.pipeline.runner import _reference_from_remote, _source_descriptor_sha256
from mojilex_cli.runs import ElementCheckpoint, RunCheckpoint, RunStore
from mojilex_cli.runs.pack_scope import (
    initialize_source_states,
    record_source_state,
    source_state,
)
from mojilex_cli.sources.base import SourceCollection, SourceError
from mojilex_cli.sources.telegram import TelegramBotAPI

from .packs import _pack_elements, _runs, _source_name, _sources
from .runtime import CommandError


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


def _completed_source(checkpoint: RunCheckpoint, source: str) -> bool:
    state = source_state(checkpoint, source)
    return state["phase"] == "describe" and state["status"] in {"succeeded", "noop"}


def _saved_members(checkpoint: RunCheckpoint, source: str) -> tuple[str, ...] | None:
    raw = checkpoint.safe_parameters.get("source_memberships")
    members = raw.get(_source_name(source)) if isinstance(raw, dict) else None
    if not isinstance(members, (list, tuple)) or not members:
        return None
    if any(not isinstance(item, str) for item in members) or len(set(members)) != len(members):
        return None
    return tuple(members)


def _ready_members(
    checkpoint: RunCheckpoint, source: str, snapshot: DatasetSnapshot
) -> tuple[str, ...] | None:
    """Require the final public records, not just cached AI completion flags."""
    if (
        not _completed_source(checkpoint, source)
        or type(checkpoint.safe_parameters.get("public_fragment_marker_version")) is not int
        or checkpoint.safe_parameters["public_fragment_marker_version"] != 1
    ):
        return None
    name = _source_name(source)
    members = _saved_members(checkpoint, source)
    if members is None:
        return None
    collections = [
        value
        for value in snapshot.collections.values()
        if value.platform == "telegram" and value.native_id == name
    ]
    if len(collections) != 1 or collections[0].item_count != len(members):
        return None
    rows = sorted(
        (
            row
            for row in snapshot.memberships.values()
            if row.collection_id == collections[0].id and row.status == "active"
        ),
        key=lambda row: row.position,
    )
    if len(rows) != len(members):
        return None
    for position, (row, native_id) in enumerate(zip(rows, members, strict=True)):
        emoji = snapshot.emojis.get(row.emoji_id)
        element = checkpoint.elements.get(native_id)
        if (
            row.position != position
            or emoji is None
            or emoji.platform != "telegram"
            or emoji.native_id != native_id
            or element is None
            or not element.source_descriptor_sha256
            or not element.media_sha256
            or tuple(sorted(media.sha256 for media in emoji.media))
            != tuple(sorted(element.media_sha256))
            or any(language not in emoji.descriptions for language in ("ru", "en"))
        ):
            return None
    return tuple(members)


async def _fetch_completed_metadata(
    sources: Sequence[str], config: MojiLexConfig, *, concurrency: int
) -> dict[str, SourceCollection]:
    credentials = load_credentials()
    if not credentials.telegram_bot_token:
        raise CommandError(
            "CREDENTIAL_MISSING",
            "TELEGRAM_BOT_TOKEN is not available.",
            hint="Set it in the current process environment.",
        )
    try:
        async with TelegramBotAPI(
            credentials.telegram_bot_token,
            timeout_seconds=config.telegram.timeout_seconds,
            max_attempts=config.telegram.max_attempts,
            max_download_bytes=config.processing.max_download_bytes,
        ) as adapter:

            async def fetch(source: str) -> SourceCollection:
                return await adapter.fetch_collection(adapter.canonicalize(source))

            collections = await bounded_map(
                sources, fetch, concurrency=min(len(sources), concurrency)
            )
    except SourceError as exc:
        raise CommandError(
            exc.code,
            "Could not check the current contents of completed Telegram packs.",
            hint="Retry when Telegram is available; saved progress has not been changed.",
        ) from exc
    return dict(zip(sources, collections, strict=True))


def refresh_completed_imports(
    existing: dict[str, tuple[RunCheckpoint, str]],
    config: MojiLexConfig,
    *,
    download_concurrency: int | None,
) -> dict[str, tuple[RunCheckpoint, str]]:
    """Refresh append-only membership without discarding paid work or media hints.

    This is an explicit new import, not resume: only its verified metadata plan
    may extend. Every later resume still checks the exact updated membership.
    """
    completed = {source: match for source, match in existing.items() if _completed_source(*match)}
    if not completed:
        return existing
    fresh = asyncio.run(
        _fetch_completed_metadata(
            tuple(completed),
            config,
            concurrency=download_concurrency or config.telegram.download_concurrency,
        )
    )
    if config.runs_dir is None:
        raise CommandError(
            "CONFIG_INVALID",
            "The saved run directory is unavailable.",
            hint="Configure the saved run directory and retry.",
        )
    store = RunStore(config.runs_dir)
    originals = {checkpoint.run_id: checkpoint for checkpoint, _ in completed.values()}
    snapshots: dict[str, DatasetSnapshot] = {}
    snapshots_by_path: dict[Path, DatasetSnapshot] = {}
    updates: dict[str, RunCheckpoint] = {}
    with ExitStack() as locks:
        for run_id in sorted(originals):
            locks.enter_context(store.execution_lock(run_id))
        for name in sorted({fresh[source].native_id for source in completed}):
            locks.enter_context(store.collection_lock("telegram", name))
        for run_id, checkpoint in originals.items():
            if store.load(run_id) != checkpoint or (
                checkpoint.publication is not None and checkpoint.publication.phase != "completed"
            ):
                raise CommandError(
                    "CONFIG_INVALID",
                    "The saved run changed while checking Telegram or is being published.",
                    hint="Finish the active operation and retry the import.",
                )
            staging = checkpoint.safe_parameters.get("staging_repository")
            if not isinstance(staging, str):
                raise CommandError(
                    "CONFIG_INVALID",
                    "The saved public snapshot is unavailable.",
                    hint="Restore the saved staging repository before continuing.",
                )
            staging_path = Path(staging)
            if staging_path not in snapshots_by_path:
                snapshots_by_path[staging_path] = load_dataset(staging_path)
            snapshots[run_id] = snapshots_by_path[staging_path]
        for source, (original, saved_source) in completed.items():
            ready = _ready_members(original, saved_source, snapshots[original.run_id])
            members = _saved_members(original, saved_source)
            collection = fresh[source]
            fresh_members = tuple(item.native_id for item in collection.items)
            max_items = original.safe_parameters.get("max_items")
            if type(max_items) is int and collection.item_count > max_items:
                raise CommandError(
                    "CONFIG_INVALID",
                    "The updated pack exceeds --max-items.",
                    hint="Start a fresh import with a higher item limit.",
                )
            if (
                collection.native_id != _source_name(saved_source)
                or collection.item_count != len(fresh_members)
                or len(set(fresh_members)) != len(fresh_members)
                or (members is not None and fresh_members[: len(members)] != members)
                or (
                    members is not None
                    and any(
                        (element := original.elements.get(item.native_id)) is None
                        or element.source_descriptor_sha256 != _source_descriptor_sha256(item)
                        for item in collection.items[: len(members)]
                    )
                )
            ):
                raise CommandError(
                    "SOURCE_CHANGED_DURING_RUN",
                    "Existing emoji were changed, removed or reordered in a completed pack.",
                    hint="Saved work is preserved. Use import --refresh for a fresh plan.",
                    source=saved_source,
                )
            if fresh_members == members and ready is not None:
                continue
            if original.publication is not None:
                raise CommandError(
                    "CONFIG_INVALID",
                    "Updates to a published run require a fresh import.",
                    hint="Use import --refresh; the previously published checkpoint is preserved.",
                    source=saved_source,
                )
            checkpoint = initialize_source_states(updates.get(original.run_id, original))
            raw_memberships = checkpoint.safe_parameters.get("source_memberships")
            memberships = dict(raw_memberships) if isinstance(raw_memberships, dict) else {}
            # Legacy records without a source plan need normal resume validation.
            # Never invent a membership plan or erase their cached work here.
            if members is not None:
                memberships[collection.native_id] = list(fresh_members)
            checkpoint = checkpoint.model_copy(
                update={
                    "safe_parameters": {
                        **checkpoint.safe_parameters,
                        "source_memberships": memberships,
                    },
                    "status": "partial",
                    "updated_at": datetime.now(UTC).replace(microsecond=0),
                    "dedupe_scan": None,
                }
            )
            updates[original.run_id] = record_source_state(
                checkpoint, saved_source, "import", "succeeded"
            )
        # Complete every compatibility check before changing any checkpoint.
        for checkpoint in updates.values():
            store.save(checkpoint)
    return {
        source: (updates.get(checkpoint.run_id, checkpoint), saved_source)
        for source, (checkpoint, saved_source) in existing.items()
    }
