"""Disposable run discovery summaries; checkpoints remain the source of truth."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict

from .store import (
    ElementCheckpoint,
    RunCheckpoint,
    RunStore,
    RunStoreError,
    _assert_safe,
    _atomic_write,
)


class SourceProgress(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    source: str
    name: str
    rank: tuple[bool, int, int, bool, datetime]


class RunIndex(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    summary_sha256: str = ""
    checkpoint_sha256: str
    run_id: str
    schema_version: str
    target_repository: str
    staging_repository: str | None
    max_items: int | None
    sources: tuple[SourceProgress, ...]


def source_progress_index(
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


def build_index(checkpoint: RunCheckpoint, payload: bytes) -> RunIndex:
    from .pack_scope import source_state

    saved: dict[str, str] = {}
    if not checkpoint.safe_parameters.get("publication_source_run"):
        sources = checkpoint.safe_parameters.get("sources", ())
        for source in sources if isinstance(sources, (list, tuple)) else ():
            if not isinstance(source, str):
                continue
            name = source
            if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", name):
                try:
                    parsed = urlsplit(source)
                except ValueError:
                    continue
                pieces = parsed.path.strip("/").split("/")
                if (
                    parsed.scheme not in {"http", "https"}
                    or parsed.netloc.lower()
                    not in {"t.me", "telegram.me", "www.t.me", "www.telegram.me"}
                    or len(pieces) != 2
                    or pieces[0].lower() not in {"addemoji", "addstickers"}
                ):
                    continue
                name = pieces[1]
            if re.fullmatch(r"[A-Za-z0-9_]{1,64}", name):
                saved.setdefault(name.casefold(), source)
    progress = source_progress_index(checkpoint, tuple(saved))
    entries = []
    for name, source in saved.items():
        state = source_state(checkpoint, source)
        ai, media, ready = progress[name]
        complete = state["status"] in {"succeeded", "noop"}
        entries.append(
            SourceProgress(
                source=source,
                name=name,
                rank=(
                    state["phase"] == "describe" and complete,
                    ai,
                    media,
                    ready or complete,
                    checkpoint.updated_at,
                ),
            )
        )
    staging = checkpoint.safe_parameters.get("staging_repository")
    maximum = checkpoint.safe_parameters.get("max_items")
    index = RunIndex(
        checkpoint_sha256=hashlib.sha256(payload).hexdigest(),
        run_id=checkpoint.run_id,
        schema_version=checkpoint.schema_version,
        target_repository=checkpoint.target_repository,
        staging_repository=staging if isinstance(staging, str) else None,
        max_items=maximum if isinstance(maximum, int) else None,
        sources=tuple(entries),
    )

    return index.model_copy(update={"summary_sha256": _summary_digest(index)})


def _summary_digest(index: RunIndex) -> str:
    canonical = json.dumps(
        index.model_dump(mode="json", exclude={"summary_sha256"}),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def save_index(root: Path, checkpoint: RunCheckpoint, payload: bytes) -> RunIndex:
    index = build_index(checkpoint, payload)
    directory = root / "indexes"
    directory.mkdir(exist_ok=True)
    _atomic_write(directory / f"{checkpoint.run_id}.json", index.model_dump_json().encode())
    return index


def read_index(store: RunStore, run_id: str) -> RunIndex:
    """Validate exact checkpoint bytes before trusting discovery metadata.

    No mtime shortcut: externally edited files, even of identical size and date,
    invalidate the summary. A cache miss falls back to the ordinary safe loader.
    """
    from .store import _MAX_CHECKPOINT_BYTES

    path = store._checkpoint_path(run_id)
    if path.is_symlink():
        raise RunStoreError("checkpoint symlinks are forbidden")
    with path.open("rb") as stream:
        payload = stream.read(_MAX_CHECKPOINT_BYTES + 1)
    if len(payload) > _MAX_CHECKPOINT_BYTES:
        raise RunStoreError("checkpoint exceeds the safe size limit")
    sidecar = store.root / "indexes" / f"{run_id}.json"
    try:
        if sidecar.is_symlink():
            raise ValueError("index symlink")
        with sidecar.open("rb") as stream:
            raw = stream.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("index too large")
        index = RunIndex.model_validate_json(raw)
        _assert_safe(json.loads(raw))
        if (
            index.run_id == run_id
            and index.summary_sha256 == _summary_digest(index)
            and index.checkpoint_sha256 == hashlib.sha256(payload).hexdigest()
        ):
            return index
    except (OSError, ValueError, RunStoreError):
        pass
    checkpoint = store._load_payload(payload)
    index = build_index(checkpoint, payload)
    if store.write_enabled:
        try:
            sidecar.parent.mkdir(exist_ok=True)
            _atomic_write(sidecar, index.model_dump_json().encode())
        except OSError:
            pass  # A missing optimization must not prevent resuming paid work.
    return index
