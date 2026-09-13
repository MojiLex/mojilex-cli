"""Find durable imports before creating another download operation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mojilex_cli.config import MojiLexConfig
from mojilex_cli.git import GitRunner
from mojilex_cli.pipeline.runner import _reference_from_remote
from mojilex_cli.runs import RunCheckpoint
from mojilex_cli.runs.pack_scope import source_state

from .packs import _pack_elements, _runs, _source_name, _sources


def import_complete(checkpoint: RunCheckpoint, source: str) -> bool:
    state = source_state(checkpoint, source)
    elements = _pack_elements(checkpoint, _source_name(source))
    return state["status"] in {"succeeded", "noop"} or (
        state["phase"] == "describe"
        and bool(elements)
        and all(item.fingerprint_complete for item in elements.values())
    )


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
    found: dict[str, tuple[RunCheckpoint, str]] = {}
    for source in sources:
        name = _source_name(source)
        if name is None:
            continue
        matches = []
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
            saved = next(
                (
                    item
                    for item in _sources(run)
                    if (_source_name(item) or "").casefold() == name.casefold()
                ),
                None,
            )
            if saved is not None:
                matches.append((run, saved))
        if matches:
            # Prefer a completed copy over a newer accidental interrupted reimport.
            found[source] = next(
                (
                    item
                    for item in matches
                    if source_state(*item)["status"] in {"succeeded", "noop"}
                ),
                matches[0],
            )
    return found
