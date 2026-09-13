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
from mojilex_cli.git import GitRunner
from mojilex_cli.github import RepositoryRef
from mojilex_cli.i18n import current_ui_language
from mojilex_cli.output import RunStatus
from mojilex_cli.runs import RunStore

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
        if base.collections.get(identifier) == collection:
            continue
        if identifier in result.collections or identifier in result.tombstones:
            skipped.append(identifier)
            continue
        memberships = [
            item for item in candidate.memberships.values() if item.collection_id == identifier
        ]
        # A deleted shared entity must not be silently resurrected by synchronization.
        if any(item.emoji_id in result.tombstones for item in memberships):
            skipped.append(identifier)
            continue
        result.collections[identifier] = collection.model_copy(deep=True)
        for membership in memberships:
            result.memberships[membership.id] = membership.model_copy(deep=True)
            if membership.emoji_id not in result.emojis:
                result.emojis[membership.emoji_id] = candidate.emojis[
                    membership.emoji_id
                ].model_copy(deep=True)
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
        and run.status in {"succeeded", "noop"}
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
        # _runs sorts newest first: duplicates in older runs cannot overwrite them.
        for checkpoint in ready:
            try:
                staging = _staging_path_from_checkpoint(checkpoint)
            except CommandError:
                warnings.append(
                    "Skipped a saved run whose staging workspace is unavailable or unsafe."
                )
                continue
            validate_dataset(staging, strict=True).raise_for_errors()
            candidate = load_dataset(staging)
            mark_saved_fragments(candidate, checkpoint)
            with snapshot_at_revision(staging, checkpoint.base_revision) as base_root:
                merged, additions, omissions = _add_missing_packs(
                    merged, load_dataset(base_root), candidate
                )
            added.extend(additions)
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
        identity = "\n".join([str(workspace.target), config.repository.base_branch, *sorted(added)])
        identifier = "mlxrun_" + hashlib.sha256(identity.encode()).hexdigest()[:32]
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
