"""Per-source progress within a shared, budgeted run checkpoint."""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlsplit

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.runs.store import RunCheckpoint


def source_name(source: str) -> str:
    return urlsplit(source).path.rstrip("/").rsplit("/", 1)[-1]


def selected_source_entries(
    sources: Sequence[str], selected: Sequence[str] = ()
) -> tuple[tuple[int, str], ...]:
    wanted = set(selected)
    if wanted - set(sources):
        raise CommandError(
            "CONFIG_INVALID",
            "The selected pack is not part of this saved run.",
            hint="Choose a pack from the saved run without editing its source plan.",
        )
    return tuple((i, source) for i, source in enumerate(sources) if not wanted or source in wanted)


def source_state(checkpoint: RunCheckpoint, source: str) -> dict[str, str]:
    states = checkpoint.safe_parameters.get("source_states")
    state = states.get(source) if isinstance(states, dict) else None
    if isinstance(state, dict) and state.get("phase") in {"import", "describe"}:
        status = str(state.get("status", "pending"))
        if status == "running" and checkpoint.status != "running":
            status = (
                checkpoint.status if checkpoint.status not in {"succeeded", "noop"} else "pending"
            )
        return {"phase": str(state["phase"]), "status": status}
    phase = "import" if checkpoint.command == "import" else "describe"
    if checkpoint.status in {"succeeded", "noop"}:
        return {"phase": phase, "status": "succeeded"}
    memberships = checkpoint.safe_parameters.get("source_memberships")
    members = memberships.get(source_name(source)) if isinstance(memberships, dict) else None
    if (
        isinstance(members, list)
        and members
        and all(
            isinstance(native_id, str)
            and (element := checkpoint.elements.get(native_id)) is not None
            and element.fingerprint_complete
            for native_id in members
        )
    ):
        return {"phase": "import", "status": "succeeded"}
    return {"phase": phase, "status": "pending"}


def initialize_source_states(checkpoint: RunCheckpoint) -> RunCheckpoint:
    """Capture legacy phases before changing the run's aggregate command/status."""
    sources = checkpoint.safe_parameters.get("sources", ())
    if not isinstance(sources, (list, tuple)) or len(sources) < 2:
        return checkpoint
    states = {
        source: source_state(checkpoint, source) for source in sources if isinstance(source, str)
    }
    return checkpoint.model_copy(
        update={"safe_parameters": {**checkpoint.safe_parameters, "source_states": states}}
    )


def record_source_state(
    checkpoint: RunCheckpoint, source: str, phase: str, status: str
) -> RunCheckpoint:
    states = checkpoint.safe_parameters.get("source_states", {})
    states = dict(states) if isinstance(states, dict) else {}
    states[source] = {"phase": phase, "status": status}
    return checkpoint.model_copy(
        update={"safe_parameters": {**checkpoint.safe_parameters, "source_states": states}}
    )


def overall_status(checkpoint: RunCheckpoint, phase: str, fallback: str) -> str:
    if fallback not in {"succeeded", "noop"}:
        return fallback
    sources = checkpoint.safe_parameters.get("sources", ())
    raw_excluded = checkpoint.safe_parameters.get("official_excluded_sources", ())
    excluded = raw_excluded if isinstance(raw_excluded, (list, tuple)) else ()
    for source in sources if isinstance(sources, (list, tuple)) else ():
        if not isinstance(source, str) or source in excluded:
            continue
        state = source_state(checkpoint, source)
        if state["status"] != "succeeded" or (phase == "describe" and state["phase"] != phase):
            return "partial"
    return fallback
