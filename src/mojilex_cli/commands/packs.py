"""Read-only pack discovery and saved-description browsing by public pack name."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from mojilex_cli.cache import CacheError, CacheStore
from mojilex_cli.composition.detector import Composition
from mojilex_cli.config import MojiLexConfig, load_config
from mojilex_cli.dataset import DatasetLoadError
from mojilex_cli.dataset.layout import assert_no_link_or_reparse
from mojilex_cli.dataset.repository import _load_dataset_unlocked
from mojilex_cli.dataset.transaction import TRANSACTION_DIRECTORY_NAME
from mojilex_cli.runs import RunCheckpoint, RunStore, RunStoreError

from .runtime import CommandError, CommandResult, operation_progress

_RUN_ID = re.compile(r"mlxrun_[0-9a-f]{32}\Z")
_PACK_NAME = re.compile(r"[A-Za-z0-9_]{1,64}\Z")
_MAX_RUNS = 10_000
_Purpose = Literal["latest", "view", "publish", "resume", "describe"]
_COMPLETE = {"succeeded", "noop"}


def _source_name(source: object) -> str | None:
    if not isinstance(source, str):
        return None
    if _PACK_NAME.fullmatch(source):
        return source
    try:
        parsed = urlsplit(source)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {
        "t.me",
        "telegram.me",
        "www.t.me",
        "www.telegram.me",
    }:
        return None
    pieces = parsed.path.strip("/").split("/")
    if len(pieces) != 2 or pieces[0].lower() not in {"addemoji", "addstickers"}:
        return None
    return pieces[1] if _PACK_NAME.fullmatch(pieces[1]) else None


def _names(checkpoint: RunCheckpoint) -> tuple[str, ...]:
    sources = checkpoint.safe_parameters.get("sources")
    if not isinstance(sources, (list, tuple)):
        return ()
    names = {
        name.casefold(): name for source in sources if (name := _source_name(source)) is not None
    }
    return tuple(names[key] for key in sorted(names))


def _group(checkpoint: RunCheckpoint) -> tuple[tuple[str, ...], str]:
    return tuple(name.casefold() for name in _names(checkpoint)), checkpoint.target_repository


def _selector_name(selector: str) -> str | None:
    if selector.startswith("mlxrun_"):
        run_id, separator, name = selector.partition(":")
        if separator and _RUN_ID.fullmatch(run_id) and _PACK_NAME.fullmatch(name):
            return name
        return None
    return _source_name(selector)


def _pack_selector(checkpoint: RunCheckpoint, name: str) -> str:
    return f"{checkpoint.run_id}:{name}"


def _sources(checkpoint: RunCheckpoint) -> tuple[str, ...]:
    sources = checkpoint.safe_parameters.get("sources", ())
    return (
        tuple(value for value in sources if isinstance(value, str))
        if isinstance(sources, (list, tuple))
        else ()
    )


def selected_pack_sources(checkpoint: RunCheckpoint, selector: str) -> tuple[str, ...]:
    """An exact RunID keeps batch semantics; a named pack always selects one source."""
    name = _selector_name(selector)
    if name is None or len(_names(checkpoint)) <= 1:
        return ()
    return tuple(
        source
        for source in _sources(checkpoint)
        if isinstance(source, str) and (_source_name(source) or "").casefold() == name.casefold()
    )


def _pack_elements(checkpoint: RunCheckpoint, name: str | None) -> dict[str, Any]:
    if name is None or len(_names(checkpoint)) == 1:
        return checkpoint.elements
    memberships = checkpoint.safe_parameters.get("source_memberships", {})
    members: Any = (
        next(
            (values for key, values in memberships.items() if key.casefold() == name.casefold()),
            [],
        )
        if isinstance(memberships, dict)
        else []
    )
    return (
        {
            identifier: checkpoint.elements[identifier]
            for identifier in members
            if isinstance(identifier, str) and identifier in checkpoint.elements
        }
        if isinstance(members, list)
        else {}
    )


def _pack_phase_status(checkpoint: RunCheckpoint, name: str | None) -> tuple[str, str]:
    if name is None or len(_names(checkpoint)) == 1:
        return checkpoint.command, checkpoint.status
    from mojilex_cli.runs.pack_scope import source_state

    sources = selected_pack_sources(checkpoint, name)
    state = source_state(checkpoint, sources[0]) if sources else {}
    return state.get("phase", "import"), state.get("status", "pending")


def _runs(config: MojiLexConfig) -> tuple[list[RunCheckpoint], int]:
    if config.runs_dir is None:
        return [], 0
    store = RunStore(config.runs_dir, write_enabled=False)
    if not store.root.is_dir():
        return [], 0
    checkpoints = []
    skipped = 0
    paths = sorted(store.root.glob("mlxrun_*.json"))
    for index, path in enumerate(paths):
        if index >= _MAX_RUNS:
            raise CommandError(
                "CONFIG_INVALID",
                "There are too many saved runs to list safely.",
                hint="Use an exact RunID to open the required run.",
            )
        if not _RUN_ID.fullmatch(path.stem):
            continue
        try:
            from mojilex_cli.i18n import current_ui_language

            label = (
                "Чтение сохранённых запусков"
                if current_ui_language() == "ru"
                else "Reading saved runs"
            )
            with operation_progress(f"{label}: {index + 1}/{len(paths)}"):
                checkpoint = store.load(path.stem)
        except (RunStoreError, OSError, ValueError):
            skipped += 1
            continue
        if _names(checkpoint) and not checkpoint.safe_parameters.get("publication_source_run"):
            checkpoints.append(checkpoint)
    checkpoints.sort(key=lambda run: (run.updated_at, run.run_id), reverse=True)
    return checkpoints, skipped


def _select(
    matches: list[RunCheckpoint], purpose: _Purpose, name: str | None = None
) -> RunCheckpoint:
    if purpose in {"latest", "describe"}:
        return matches[0]
    if purpose == "resume":
        return next(
            (run for run in matches if _pack_phase_status(run, name)[1] not in _COMPLETE),
            matches[0],
        )
    if purpose == "publish":
        completed = [
            run
            for run in matches
            if _pack_phase_status(run, name)[0] == "describe"
            and _pack_phase_status(run, name)[1] in _COMPLETE
        ]
        if not completed:
            raise CommandError(
                "CONFIG_MISSING",
                "No completed description run is ready to publish for this pack.",
                hint="Complete mojilex describe for this pack first, or choose an exact RunID.",
            )
        return completed[0]
    if purpose == "view":
        ready = [
            run
            for run in matches
            if any(element.ai_facets_complete for element in _pack_elements(run, name).values())
        ]
        completed = [run for run in ready if _pack_phase_status(run, name)[1] in _COMPLETE]
        if completed:
            return completed[0]
        if ready:
            return ready[0]
        for run in matches:
            staging = run.safe_parameters.get("staging_repository")
            if isinstance(staging, str) and Path(staging).is_dir():
                return run
        return matches[0]
    raise ValueError("unknown pack selection purpose")


def _resolve(
    selector: str,
    config: MojiLexConfig,
    *,
    purpose: _Purpose = "latest",
) -> RunCheckpoint:
    if selector.startswith("mlxrun_"):
        if config.runs_dir is None:
            raise RunStoreError("run storage is not configured")
        run_id, separator, scoped_name = selector.partition(":")
        checkpoint = RunStore(config.runs_dir, write_enabled=False).load(run_id)
        if separator and (
            not _PACK_NAME.fullmatch(scoped_name)
            or scoped_name.casefold() not in {value.casefold() for value in _names(checkpoint)}
        ):
            raise CommandError(
                "CONFIG_INVALID",
                "The selected pack is not part of this saved run.",
                hint="Choose an existing pack from mojilex list.",
            )
        return checkpoint
    name = _source_name(selector)
    if name is None:
        raise CommandError(
            "CONFIG_INVALID",
            "Use a pack name or an exact RunID.",
            hint="Run mojilex list to see saved packs.",
        )
    runs, _ = _runs(config)
    matches = [run for run in runs if name.casefold() in {n.casefold() for n in _names(run)}]
    if not matches:
        raise CommandError(
            "CONFIG_MISSING",
            "No saved run matches this pack name.",
            hint="Run mojilex list to see saved packs, or pass an exact RunID.",
        )
    if len({run.target_repository for run in matches}) != 1:
        raise CommandError(
            "CONFIG_INVALID",
            "This pack name matches different repositories.",
            hint="Choose an exact RunID from mojilex list.",
            details={"runs": [run.run_id for run in matches]},
        )
    return _select(matches, purpose, name)


def resolve_pack_run(selector: str, *, purpose: _Purpose = "latest") -> RunCheckpoint:
    """Resolve a pack for one operation; an explicit RunID is never redirected."""
    return _resolve(selector, load_config(), purpose=purpose)


def _summary(
    checkpoint: RunCheckpoint, config: MojiLexConfig, name: str | None = None
) -> dict[str, Any]:
    maximum = checkpoint.safe_parameters.get("max_ai_requests")
    checkpoint_maximum = maximum == "unlimited" or (type(maximum) is int and maximum >= 0)
    elements = _pack_elements(checkpoint, name)
    phase, status = _pack_phase_status(checkpoint, name)
    members = checkpoint.safe_parameters.get("source_memberships", {})
    member_count = (
        next(
            (
                len(values)
                for key, values in members.items()
                if name is not None
                and key.casefold() == name.casefold()
                and isinstance(values, list)
            ),
            len(elements),
        )
        if isinstance(members, dict)
        else len(elements)
    )
    return {
        "run_id": checkpoint.run_id,
        "selector": _pack_selector(checkpoint, name) if name else checkpoint.run_id,
        "names": [name] if name else list(_names(checkpoint)),
        "status": status,
        "phase": phase,
        "budget_scope": "batch" if len(_names(checkpoint)) > 1 else "pack",
        "updated_at": checkpoint.updated_at.isoformat(),
        "items": member_count,
        "ai_ready": sum(element.ai_facets_complete for element in elements.values()),
        "requests_used": checkpoint.ai_requests_used,
        "max_ai_requests": (None if maximum == "unlimited" else maximum)
        if checkpoint_maximum
        else config.ai.max_ai_requests,
        "max_ai_requests_source": "checkpoint" if checkpoint_maximum else "config",
    }


def list_packs_command() -> CommandResult:
    config = load_config()
    runs, skipped = _runs(config)
    groups: dict[tuple[str, str], list[RunCheckpoint]] = {}
    for run in runs:
        for name in _names(run):
            groups.setdefault((name.casefold(), run.target_repository), []).append(run)
    rows = []
    for (name_key, _), group in groups.items():
        visible = _select(group, "view", name_key)
        name = next(value for value in _names(visible) if value.casefold() == name_key)
        row = _summary(visible, config, name)
        unfinished = next(
            (run for run in group if _pack_phase_status(run, name)[1] not in _COMPLETE), None
        )
        if unfinished is not None and unfinished.run_id != visible.run_id:
            row["latest_unfinished"] = _summary(unfinished, config, name)
        rows.append(row)
    return CommandResult(
        result={
            "packs": rows,
            "saved_runs": len(runs),
        },
        warnings=["Some saved run files could not be read."] if skipped else [],
    )


def _semantic_fields(value: Any) -> dict[str, Any]:
    facets = value.facets.model_dump(mode="json")
    return {
        "descriptions": {
            language: description.model_dump(mode="json")
            for language, description in (
                value.descriptions.items()
                if isinstance(value.descriptions, dict)
                else (("ru", value.descriptions.ru), ("en", value.descriptions.en))
            )
        },
        "semantic_tags": list(value.semantic_tags),
        "content": value.content.model_dump(mode="json"),
        "concept_ids": list(value.concept_ids),
        "facets": {
            key: facets[key]
            for key in (
                "text_content",
                "content_types",
                "styles",
                "suggested_uses",
                "uncertainties",
            )
        },
    }


def _staging_items(
    checkpoint: RunCheckpoint,
    names: tuple[str, ...],
    warnings: list[str],
) -> dict[str, dict[str, Any]]:
    staging = checkpoint.safe_parameters.get("staging_repository")
    if not isinstance(staging, str) or not Path(staging).is_dir():
        return {}
    try:
        # Public load_dataset recovers transactions and creates a writer lock.
        # Browsing must not do either, so refuse pending writes and use the
        # parsing-only loader with before/after consistency checks.
        root = Path(staging)
        assert_no_link_or_reparse(root)
        root = root.resolve()
        transaction = root / TRANSACTION_DIRECTORY_NAME
        if transaction.exists() or transaction.is_symlink():
            raise DatasetLoadError("staging has an unfinished transaction")
        snapshot = _load_dataset_unlocked(root)
        if transaction.exists() or transaction.is_symlink():
            raise DatasetLoadError("staging changed while being read")
        for relative, original in snapshot.source_bytes.items():
            path = root / relative
            assert_no_link_or_reparse(path, boundary=root)
            if path.read_bytes() != original:
                raise DatasetLoadError("staging changed while being read")
    except (OSError, DatasetLoadError, ValueError):
        warnings.append("The saved staging dataset could not be read; using exact cached results.")
        return {}
    wanted = {name.casefold() for name in names}
    collections = {
        item.id for item in snapshot.collections.values() if item.native_id.casefold() in wanted
    }
    member_ids = {
        item.emoji_id
        for item in snapshot.memberships.values()
        if item.collection_id in collections and item.status.value == "active"
    }
    result = {}
    for emoji in snapshot.emojis.values():
        if emoji.id not in member_ids:
            continue
        result[emoji.native_id] = {
            "native_id": emoji.native_id,
            **_semantic_fields(emoji),
            "review_status": emoji.review.status.value,
            "source": "staging",
        }
    return result


def _saved_compositions(checkpoint: RunCheckpoint, names: tuple[str, ...]) -> list[dict[str, Any]]:
    """Expose only current, verified and non-overlapping local composition evidence."""
    evidence = checkpoint.safe_parameters.get("composition_evidence")
    memberships = checkpoint.safe_parameters.get("source_memberships")
    if not isinstance(evidence, dict) or not isinstance(memberships, dict):
        return []
    selected = {name.casefold() for name in names}
    groups: list[Composition] = []
    occurrences: dict[str, int] = {}
    for name, values in evidence.items():
        if not isinstance(name, str) or name.casefold() not in selected:
            continue
        members = memberships.get(name)
        if not isinstance(members, list) or not isinstance(values, list) or len(values) > 64:
            continue
        allowed = {member for member in members if isinstance(member, str)}
        for raw in values:
            try:
                group = Composition.model_validate(raw, strict=True)
            except (ValueError, TypeError):
                continue
            ids = {member.native_id for member in group.members}
            if (
                not group.verified
                or len(group.members) != group.columns * group.rows
                or len(ids) != len(group.members)
                or not ids <= allowed
                or any(
                    (element := checkpoint.elements.get(member.native_id)) is None
                    or element.media_sha256 != (member.media_sha256,)
                    for member in group.members
                )
            ):
                continue
            groups.append(group)
            for native_id in ids:
                occurrences[native_id] = occurrences.get(native_id, 0) + 1
    return [
        group.model_dump(mode="json")
        for group in groups
        if all(occurrences[member.native_id] == 1 for member in group.members)
    ]


def show_pack_command(selector: str, *, review: bool = False) -> CommandResult:
    """Browse saved text only. Review mode never approves or changes an item."""
    config = load_config()
    checkpoint = _resolve(selector, config, purpose="view")
    all_names = _names(checkpoint)
    selected_name = _selector_name(selector)
    names = tuple(
        name
        for name in all_names
        if selected_name is None or name.casefold() == selected_name.casefold()
    )
    warnings: list[str] = []
    staging = _staging_items(checkpoint, names, warnings)
    membership_map = checkpoint.safe_parameters.get("source_memberships")
    native_ids: set[str] = set()
    if isinstance(membership_map, dict):
        for name, members in membership_map.items():
            if name.casefold() in {item.casefold() for item in names} and isinstance(members, list):
                native_ids.update(member for member in members if isinstance(member, str))
    if selected_name is None or len(all_names) == 1:
        native_ids.update(checkpoint.elements)
    native_ids.update(staging)
    counts = {"ready": 0, "pending": 0, "missing": 0, "invalid": 0}
    items = []
    cache: CacheStore | None = None
    needs_cache = any(native_id not in staging for native_id in native_ids)
    cache_path = config.cache_dir / "cache-v1.sqlite3" if config.cache_dir is not None else None
    if needs_cache and cache_path is not None and cache_path.is_file():
        try:
            cache = CacheStore(cache_path, read_only=True)
        except (CacheError, OSError):
            warnings.append(
                "The cache cannot be read without changing it; "
                "retry after the running command ends."
            )
    try:
        for native_id in sorted(native_ids):
            if native_id in staging:
                items.append(staging[native_id])
                counts["ready"] += 1
                continue
            element = checkpoint.elements.get(native_id)
            if element is None or element.ai_cache_key is None:
                counts["pending"] += 1
                continue
            if cache is None:
                counts["missing"] += 1
                continue
            try:
                hit = cache.get_ai_entry(element.ai_cache_key, follow_aliases=False)
                if hit is None:
                    counts["missing"] += 1
                    continue
                result = hit[1].result
                if len(result.batch.items) != 1:
                    counts["invalid"] += 1
                    continue
                items.append(
                    {
                        "native_id": native_id,
                        **_semantic_fields(result.batch.items[0]),
                        "review_status": "unreviewed",
                        "source": "ai_cache",
                        "generated_at": hit[1].generated_at,
                    }
                )
                counts["ready"] += 1
            except (CacheError, sqlite3.Error, ValueError, TypeError):
                counts["invalid"] += 1
    finally:
        if cache is not None:
            cache.close()
    if counts["missing"]:
        warnings.append(
            "Some exact cached descriptions are unavailable; saved progress was not changed."
        )
    summary = _summary(checkpoint, config, selected_name)
    summary.update(
        {
            "names": list(names),
            "items": len(native_ids),
            "ai_ready": sum(
                element.ai_facets_complete
                for native_id, element in checkpoint.elements.items()
                if native_id in native_ids
            ),
        }
    )
    return CommandResult(
        run_id=checkpoint.run_id,
        result={
            "pack": summary,
            "items": items,
            "counts": counts,
            "review": review,
            "compositions": _saved_compositions(checkpoint, names),
        },
        warnings=list(warnings),
    )
