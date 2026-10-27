"""Project-local emoji pack choices shared by Codex and Claude Code."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from mojilex_cli.dataset.layout import assert_no_link_or_reparse
from mojilex_cli.read.lookup import LookupError

_RELATIVE_PATH = Path(".mojilex") / "emoji-packs.json"


def preferences_path(project: Path | None = None) -> Path:
    """Use an explicit project, or the closest choice within the nearest Git project."""
    directory = Path(os.path.abspath((project or Path.cwd()).expanduser()))
    if not directory.is_dir():
        raise LookupError("LOOKUP_PROJECT", "The project must be an existing directory.")
    if project is not None:
        return directory / _RELATIVE_PATH
    for parent in (directory, *directory.parents):
        candidate = parent / _RELATIVE_PATH
        if candidate.exists() or candidate.is_symlink() or (parent / ".git").exists():
            return candidate
    return directory / _RELATIVE_PATH


def _safe_path(path: Path) -> None:
    try:
        assert_no_link_or_reparse(path)
    except ValueError as exc:
        raise LookupError("LOOKUP_PROJECT", "Project preferences cannot traverse links.") from exc


def read_preferences(project: Path | None = None) -> dict[str, Any]:
    path = preferences_path(project)
    _safe_path(path)
    if not path.exists():
        return {"configured": False, "path": str(path), "packs": []}
    try:
        if path.stat().st_size > 64 * 1024:
            raise ValueError("oversize preferences")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("version") != 1
            or not isinstance(value.get("packs"), list)
            or not value["packs"]
            or any(not isinstance(pack, str) or not pack.strip() for pack in value["packs"])
        ):
            raise ValueError("invalid preferences")
    except (OSError, ValueError, RecursionError) as exc:
        raise LookupError(
            "LOOKUP_PROJECT",
            "The project's emoji pack preferences are invalid.",
            "Check .mojilex/emoji-packs.json; lookup never falls back to all packs on this error.",
        ) from exc
    return {"configured": True, "path": str(path), "packs": value["packs"]}


def save_preferences(packs: list[str], project: Path | None = None) -> dict[str, Any]:
    if not packs or any(not pack.strip() for pack in packs):
        raise LookupError("LOOKUP_PROJECT", "Choose at least one published emoji pack.")
    path = preferences_path(project)
    _safe_path(path)
    lock_path = path.with_suffix(".lock")
    _safe_path(lock_path)
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(lock_path, timeout=10):
            _safe_path(path)
            payload = (
                json.dumps(
                    {"version": 1, "packs": list(dict.fromkeys(packs))},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
            descriptor, name = tempfile.mkstemp(prefix=".emoji-packs-", dir=path.parent)
            temporary = Path(name)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            if path.exists():
                os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
            os.replace(temporary, path)
    except Timeout as exc:
        raise LookupError("LOOKUP_BUSY", "Another process is saving project preferences.") from exc
    except OSError as exc:
        raise LookupError(
            "LOOKUP_PROJECT", "Could not save the project's emoji pack choice."
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return read_preferences(path.parent.parent)
