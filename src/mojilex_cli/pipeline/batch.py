"""Publish all saved, completed pack descriptions as one additive synchronization."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from mojilex_cli.commands.packs import _runs
from mojilex_cli.commands.runtime import CommandError, CommandResult
from mojilex_cli.composition.publication import mark_saved_fragments
from mojilex_cli.config import load_config
from mojilex_cli.dataset import DatasetSnapshot, load_dataset, validate_dataset
from mojilex_cli.domain import MembershipStatus, media_digest, telegram_set_fingerprint
from mojilex_cli.git import GitRunner
from mojilex_cli.github import RepositoryRef
from mojilex_cli.i18n import current_ui_language
from mojilex_cli.output import RunStatus
from mojilex_cli.runs import RunStore
from mojilex_cli.runs.store import RunLockedError

from .pack_publication import ready_source_names, scoped_candidate
from .runner import (
    _apply_with_rollback,
    _changed_paths,
    _reference_from_remote,
    _run_submit,
    _staging_path_from_checkpoint,
    repository_workspace,
)
from .workspaces import snapshot_at_revision


def _add_missing_packs(
    latest: DatasetSnapshot, base: DatasetSnapshot, candidate: DatasetSnapshot
) -> tuple[DatasetSnapshot, list[str], list[str]]:
    """Keep repository entities authoritative, including shared emoji descriptions."""
    result = latest.clone()
    added: list[str] = []
    skipped: list[str] = []
    for identifier, collection in candidate.collections.items():
        memberships = [
            item for item in candidate.memberships.values() if item.collection_id == identifier
        ]
        if base.collections.get(identifier) == collection and all(
            base.memberships.get(item.id) == item
            and base.emojis.get(item.emoji_id) == candidate.emojis.get(item.emoji_id)
            for item in memberships
        ):
            continue
        if identifier in result.tombstones:
            skipped.append(identifier)
            continue
        missing = [
            item
            for item in memberships
            if item.id not in result.memberships
            and item.id not in result.tombstones
            and item.emoji_id not in result.tombstones
        ]
        descriptions = {
            item.emoji_id: {
                language: description.model_copy(deep=True)
                for language, description in candidate.emojis[item.emoji_id].descriptions.items()
                if language not in result.emojis[item.emoji_id].descriptions
            }
            for item in memberships
            if item.emoji_id in result.emojis
            and item.emoji_id not in result.tombstones
            and media_digest(result.emojis[item.emoji_id].media)
            == media_digest(candidate.emojis[item.emoji_id].media)
        }
        if identifier in result.collections and not missing and not any(descriptions.values()):
            skipped.append(identifier)
            continue
        # Never resurrect deleted shared entities when adding a new collection.
        if identifier not in result.collections and any(
            item.emoji_id in result.tombstones or item.id in result.tombstones
            for item in memberships
        ):
            skipped.append(identifier)
            continue
        if identifier not in result.collections:
            result.collections[identifier] = collection.model_copy(deep=True)
        occupied = {
            item.position
            for item in result.memberships.values()
            if item.collection_id == identifier and item.status is MembershipStatus.ACTIVE
        }
        for membership in sorted(missing, key=lambda item: (item.position, item.id)):
            member = membership.model_copy(deep=True)
            if member.status is MembershipStatus.ACTIVE:
                if member.position in occupied:
                    member.position = max(occupied, default=-1) + 1
                occupied.add(member.position)
            result.memberships[member.id] = member
            if member.emoji_id not in result.emojis:
                result.emojis[member.emoji_id] = candidate.emojis[member.emoji_id].model_copy(
                    deep=True
                )
        for emoji_identifier, missing_descriptions in descriptions.items():
            result.emojis[emoji_identifier].descriptions.update(missing_descriptions)
        merged_collection = result.collections[identifier]
        active = [
            result.emojis[item.emoji_id]
            for item in result.memberships.values()
            if item.collection_id == identifier and item.status is MembershipStatus.ACTIVE
        ]
        merged_collection.item_count = len(active)
        if merged_collection.platform == "telegram":
            merged_collection.extensions["telegram"]["set_fingerprint_sha256"] = (
                telegram_set_fingerprint(
                    [
                        (item.native_id, item.extensions["telegram"]["file_unique_id"])
                        for item in active
                    ]
                )
            )
        added.append(identifier)
    new_emojis = result.emojis.keys() - latest.emojis.keys()
    for identifier, relation in candidate.relations.items():
        if (
            identifier not in result.relations
            and relation.subject_id in result.emojis
            and relation.object_id in result.emojis
            and relation.subject_id in new_emojis
            and relation.object_id in new_emojis
        ):
            result.relations[identifier] = relation.model_copy(deep=True)
    return result, added, skipped


def _sync_identifier(
    target: str, base_branch: str, packs: list[str], snapshot: DatasetSnapshot
) -> str:
    """Reuse an unchanged PR, but give later additions a new publication identity."""
    identity = "\n".join(
        [
            target,
            base_branch,
            *sorted(packs),
            *sorted(
                member.id
                + ":"
                + hashlib.sha256(
                    snapshot.emojis[member.emoji_id].model_dump_json().encode()
                ).hexdigest()
                for member in snapshot.memberships.values()
                if member.collection_id in packs
            ),
        ]
    )
    return "mlxrun_" + hashlib.sha256(identity.encode()).hexdigest()[:32]


def sync_packs_command(
    *, local: bool = False, confirmation: Callable[[str], bool] | None = None
) -> CommandResult:
    config = load_config()
    target_path = Path(config.repository.target).expanduser()
    target = str(
        _reference_from_remote(GitRunner(target_path).remote_url())
        if target_path.is_dir()
        else RepositoryRef.parse(config.repository.target)
    )
    checkpoints, unreadable = _runs(config)
    ready = [
        run
        for run in checkpoints
        if run.command == "describe"
        and not getattr(run, "safe_parameters", {}).get("publication_source_run")
        and (run.status in {"succeeded", "noop"} or ready_source_names(run))
        and run.target_repository.casefold() == target.casefold()
    ]
    warnings: list[dict[str, Any] | str] = (
        ["Some saved run files could not be read."] if unreadable else []
    )
    if not ready:
        return CommandResult(
            status=RunStatus.NOOP,
            result={"added_packs": [], "skipped_packs": [], "changed_paths": []},
            warnings=warnings,
        )
    with repository_workspace(
        config.repository.target, config.repository.base_branch, isolated=True
    ) as workspace:
        validate_dataset(workspace.root, strict=True).raise_for_errors()
        latest = load_dataset(workspace.root)
        merged = latest
        added: list[str] = []
        skipped: set[str] = set()
        selected_runs: list[str] = []
        store = RunStore(cast(Path, config.runs_dir))
        # _runs sorts newest first: duplicates in older runs cannot overwrite them.
        for checkpoint in ready:
            try:
                with store.execution_lock(checkpoint.run_id):
                    checkpoint = store.load(checkpoint.run_id)
                    staging = _staging_path_from_checkpoint(checkpoint)
                    validate_dataset(staging, strict=True).raise_for_errors()
                    candidate = load_dataset(staging)
                    mark_saved_fragments(candidate, checkpoint)
                    with snapshot_at_revision(staging, checkpoint.base_revision) as base_root:
                        base_snapshot = load_dataset(base_root)
                        # Completion belongs to a source, not to the entire batch.
                        if (
                            getattr(checkpoint, "safe_parameters", {}).get("source_states")
                            is not None
                        ):
                            completed_names = ready_source_names(checkpoint)
                            if not completed_names:
                                continue
                            candidate = scoped_candidate(base_snapshot, candidate, completed_names)
                        elif checkpoint.command != "describe" or checkpoint.status not in {
                            "succeeded",
                            "noop",
                        }:
                            continue
                        merged, additions, omissions = _add_missing_packs(
                            merged, base_snapshot, candidate
                        )
            except RunLockedError:
                warnings.append("Skipped a saved run that is currently being processed.")
                continue
            except CommandError:
                warnings.append(
                    "Skipped a saved run whose staging workspace is unavailable or unsafe."
                )
                continue
            added.extend(item for item in additions if item not in added)
            skipped.update(omissions)
            if additions:
                selected_runs.append(checkpoint.run_id)
        paths = _changed_paths(latest, merged)
        summary = {
            "added_packs": added,
            "skipped_packs": sorted(skipped - set(added)),
            "selected_runs": selected_runs,
            "changed_paths": [str(path) for path in paths],
        }
        if not paths:
            return CommandResult(status=RunStatus.NOOP, result=summary, warnings=warnings)
        _apply_with_rollback(latest, merged)
        if local:
            return CommandResult(
                result={**summary, "validated": True},
                publication={"mode": "local", "preview": True},
                warnings=warnings,
            )
        names = [merged.collections[identifier].native_id for identifier in added]
        message = (
            f"Отправить {len(added)} паков одним PR в {workspace.target}: "
            if current_ui_language() == "ru"
            else f"Publish {len(added)} packs in one GitHub PR to {workspace.target}: "
        ) + ", ".join(names)
        if confirmation is not None and not confirmation(message):
            raise CommandError(
                "CONFIG_INVALID",
                "Batch publication was not confirmed.",
                hint="Run synchronization again when ready to publish.",
            )
        # A stable identity reuses the same pending PR on a repeated synchronization.
        identifier = _sync_identifier(
            str(workspace.target), config.repository.base_branch, added, merged
        )
        store = RunStore(cast(Path, config.runs_dir))
        with store.execution_lock(identifier):
            result = asyncio.run(
                _run_submit(
                    str(workspace.root),
                    repository=str(workspace.root),
                    publish="pr",
                    direct_push=False,
                    base=config.repository.base_branch,
                    confirmation=confirmation,
                    batch_identifier=identifier,
                )
            )
        result.result.update(summary)
        result.warnings.extend(warnings)
        return result
