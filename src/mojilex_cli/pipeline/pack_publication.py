"""Build a publication candidate for one pack without changing its batch run."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import TYPE_CHECKING

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.dataset import DatasetSnapshot, load_dataset, validate_dataset, validate_snapshot
from mojilex_cli.github import RepositoryRef
from mojilex_cli.runs import RunCheckpoint, RunStore
from mojilex_cli.runs.pack_scope import source_name, source_state

from .workspaces import prepare_staging_workspace, snapshot_at_revision

if TYPE_CHECKING:
    from mojilex_cli.config import MojiLexConfig


def ready_source_names(checkpoint: RunCheckpoint) -> set[str]:
    """Return individually completed descriptions, including those in a partial batch."""
    sources = getattr(checkpoint, "safe_parameters", {}).get("sources")
    if not isinstance(sources, (list, tuple)):
        return set()
    return {
        name.casefold()
        for source in sources
        if isinstance(source, str)
        and (name := source_name(source)) is not None
        and source_state(checkpoint, source) == {"phase": "describe", "status": "succeeded"}
    }


def scoped_candidate(
    base: DatasetSnapshot, candidate: DatasetSnapshot, names: set[str]
) -> DatasetSnapshot:
    """Keep unrelated records authoritative and carry only the selected pack changes.

    Existing emoji shared with an unselected pack retain their base descriptions.
    New shared emoji are included when required by the selected pack's membership.
    Deletions of global emoji or tombstones are deliberately never inferred here.
    """
    result = base.clone()
    selected = {
        identifier
        for identifier, collection in candidate.collections.items()
        if collection.platform == "telegram" and collection.native_id.casefold() in names
    }
    if not selected:
        raise CommandError(
            "CONFIG_MISSING",
            "The saved workspace does not contain the selected pack.",
            hint="Finish describing this pack before publishing it.",
        )
    selected_memberships = {
        identifier: member
        for identifier, member in candidate.memberships.items()
        if member.collection_id in selected
    }
    selected_emojis = {member.emoji_id for member in selected_memberships.values()}
    shared = {
        member.emoji_id
        for snapshot in (base, candidate)
        for member in snapshot.memberships.values()
        if member.collection_id not in selected
    }
    editable = selected_emojis - (shared & base.emojis.keys())
    for identifier in selected:
        result.collections[identifier] = candidate.collections[identifier].model_copy(deep=True)
    result.memberships = {
        identifier: member
        for identifier, member in result.memberships.items()
        if member.collection_id not in selected
    }
    result.memberships.update(
        {
            identifier: member.model_copy(deep=True)
            for identifier, member in selected_memberships.items()
        }
    )
    for identifier in editable:
        if identifier not in candidate.emojis:
            raise CommandError(
                "CONFIG_INVALID",
                "The selected pack references a missing saved emoji.",
                hint="Validate the saved workspace before publishing.",
            )
        result.emojis[identifier] = candidate.emojis[identifier].model_copy(deep=True)
    # Relations may point to an unchanged existing emoji, but must never pull in
    # an otherwise unrelated new emoji from another pack in the same batch.
    for identifier, relation in candidate.relations.items():
        if (
            relation.subject_id in result.emojis
            and relation.object_id in result.emojis
            and (relation.subject_id in editable or relation.object_id in editable)
            and (
                identifier not in base.relations
                or {relation.subject_id, relation.object_id} <= editable
            )
        ):
            result.relations[identifier] = relation.model_copy(deep=True)
    for identifier, relation in base.relations.items():
        if (
            identifier not in candidate.relations
            and {relation.subject_id, relation.object_id} <= editable
        ):
            del result.relations[identifier]
    return result


def prepare_pack_publication(
    checkpoint: RunCheckpoint, name: str, config: MojiLexConfig
) -> RunCheckpoint:
    """Persist a stable, isolated submit target; leave the parent history untouched."""
    from mojilex_cli.composition.publication import mark_saved_fragments

    from .runner import _apply_with_rollback, _staging_path_from_checkpoint

    if config.runs_dir is None:
        raise CommandError("CONFIG_MISSING", "Run storage is not configured.", hint="Set runs_dir.")
    store = RunStore(config.runs_dir)
    with store.execution_lock(checkpoint.run_id):
        checkpoint = store.load(checkpoint.run_id)
        if name.casefold() not in ready_source_names(checkpoint):
            raise CommandError(
                "CONFIG_MISSING",
                "The selected pack does not have a completed description.",
                hint="Finish describing this pack before publishing it.",
            )
        staging = _staging_path_from_checkpoint(checkpoint)
        validate_dataset(staging, strict=True).raise_for_errors()
        candidate = load_dataset(staging)
        parameters = deepcopy(checkpoint.safe_parameters)
        sources = parameters.get("sources")
        assert isinstance(sources, (list, tuple))
        parameters["sources"] = [
            source
            for source in sources
            if isinstance(source, str) and (source_name(source) or "").casefold() == name.casefold()
        ]
        for key in ("source_memberships", "composition_evidence", "source_states"):
            values = parameters.get(key)
            if isinstance(values, dict):
                parameters[key] = {
                    key_name: value
                    for key_name, value in values.items()
                    if isinstance(key_name, str)
                    and (source_name(key_name) or "").casefold() == name.casefold()
                }
        parameters["publication_source_run"] = checkpoint.run_id
        # Retain original timestamps for legacy fragment evidence verification.
        projection = checkpoint.model_copy(update={"safe_parameters": parameters})
        mark_saved_fragments(candidate, projection)
        with snapshot_at_revision(staging, checkpoint.base_revision) as base_root:
            scoped = scoped_candidate(load_dataset(base_root), candidate, {name.casefold()})
            validate_snapshot(scoped, schemas=True, repository_files=True).raise_for_errors()
        content = hashlib.sha256()
        for path, payload in sorted(scoped.to_files().items()):
            content.update(str(path).encode())
            content.update(b"\0")
            content.update(hashlib.sha256(payload).digest())
        identifier = (
            "mlxrun_"
            + hashlib.sha256(
                f"publish-pack-v1\0{checkpoint.run_id}\0{name.casefold()}\0{content.hexdigest()}".encode()
            ).hexdigest()[:32]
        )
        with store.execution_lock(identifier):
            previous = (
                store.load(identifier) if (store.root / f"{identifier}.json").exists() else None
            )
            base_branch = parameters.get("base")
            workspace = prepare_staging_workspace(
                staging,
                target=RepositoryRef.parse(checkpoint.target_repository),
                runs_dir=config.runs_dir,
                run_id=identifier,
                base_branch=base_branch
                if isinstance(base_branch, str) and base_branch
                else config.repository.base_branch,
                base_revision=checkpoint.base_revision,
            )
            current = load_dataset(workspace)
            scoped.root = workspace
            _apply_with_rollback(current, scoped)
            parameters["staging_repository"] = str(workspace)
            parameters["repository"] = str(workspace)
            selected_ids = {
                member.emoji_id
                for member in scoped.memberships.values()
                if scoped.collections[member.collection_id].native_id.casefold() == name.casefold()
            }
            selected_native_ids = {scoped.emojis[key].native_id for key in selected_ids}
            derived = projection.model_copy(
                update={
                    "run_id": identifier,
                    "command": "describe",
                    "status": "succeeded",
                    "safe_parameters": parameters,
                    "elements": {
                        key: value
                        for key, value in checkpoint.elements.items()
                        if key in selected_native_ids
                    },
                    "dedupe_scan": None,
                    "publication": previous.publication if previous is not None else None,
                    "issues": previous.issues if previous is not None else (),
                }
            )
            store.save(derived)
            return derived
