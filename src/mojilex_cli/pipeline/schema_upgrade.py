"""Narrow, additive schema upgrades for checkpoints' private staging trees."""

from __future__ import annotations

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.dataset.layout import assert_no_link_or_reparse
from mojilex_cli.git import GitError, GitRunner
from mojilex_cli.github import GitHubError, RepositoryRef

from .workspaces import staging_workspace_path

_SCHEMA = Path("schemas/v1/emoji.schema.json")
_BUNDLED_SCHEMA = Path(__file__).resolve().parents[1] / _SCHEMA
_MAX_SCHEMA_BYTES = 256 * 1024
_PNG_VARIANT = {
    "properties": {
        "kind": {"const": "static"},
        "format": {"const": "png"},
        "mime_type": {"const": "image/png"},
        "animated": {"const": False},
    },
    "required": ["kind", "format", "mime_type", "animated"],
}


def _official(value: str) -> bool:
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    try:
        return str(RepositoryRef.parse(value)).casefold() == "mojilex/mojilex"
    except (GitHubError, ValueError):
        return False


def _read_schema(path: Path) -> tuple[bytes, Any]:
    with path.open("rb") as stream:
        raw = stream.read(_MAX_SCHEMA_BYTES + 1)
    if len(raw) > _MAX_SCHEMA_BYTES:
        raise ValueError("schema exceeds the bounded upgrade size")
    return raw, json.loads(raw, object_pairs_hook=_unique_object)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate schema object key")
        result[key] = value
    return result


def _canonical(value: Any) -> str:
    # Python object equality conflates false with 0; schema const does not.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _without_png(schema: Any) -> Any:
    previous = deepcopy(schema)
    variants = previous["allOf"][-1]["then"]["properties"]["media"]["items"]["allOf"][-1]["oneOf"]
    if not isinstance(variants, list) or variants.count(_PNG_VARIANT) != 1:
        raise ValueError("bundled schema does not have the known additive PNG variant")
    variants.remove(_PNG_VARIANT)
    return previous


def upgrade_private_staging_schema(
    staging: Path,
    *,
    runs_dir: Path,
    run_id: str,
    base_revision: str,
    target_repository: str,
) -> bool:
    """Upgrade only the known pre-PNG contract in this run's exact private tree.

    The caller must hold the run execution lock. Custom repositories, edited or
    future schemas, and mismatched workspaces are left unchanged. HEAD and the
    checkpoint base never change: publication derives only metadata differences.
    """
    if not _official(target_repository):
        return False
    try:
        runs_root = runs_dir.expanduser()
        assert_no_link_or_reparse(runs_root)
        expected = staging_workspace_path(runs_root, run_id)
        candidate = Path(os.path.abspath(staging.expanduser()))
        if candidate != expected:
            return False
        assert_no_link_or_reparse(candidate, boundary=runs_root)
        assert_no_link_or_reparse(candidate / ".git", boundary=candidate)
        schema_path = candidate / _SCHEMA
        assert_no_link_or_reparse(schema_path, boundary=candidate)
        git = GitRunner(candidate)
        if (
            Path(git.run("rev-parse", "--show-toplevel").stdout.strip()).resolve() != candidate
            or git.current_sha() != base_revision
            or not _official(git.remote_url())
        ):
            return False
        original, current = _read_schema(schema_path)
        replacement, bundled = _read_schema(_BUNDLED_SCHEMA)
        if _canonical(current) == _canonical(bundled) or _canonical(current) != _canonical(
            _without_png(bundled)
        ):
            return False
    except (
        CommandError,
        GitError,
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        RecursionError,
    ):
        return False

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=schema_path.parent, prefix=".mojilex-schema-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(replacement)
            stream.flush()
            os.fsync(stream.fileno())
        # A second check also preserves edits made after the initial inspection.
        assert_no_link_or_reparse(schema_path, boundary=runs_root)
        if _read_schema(schema_path)[0] != original or git.current_sha() != base_revision:
            return False
        os.replace(temporary, schema_path)
        return True
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
