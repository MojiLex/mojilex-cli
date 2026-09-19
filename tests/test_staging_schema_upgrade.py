from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from mojilex_cli.dataset.repository import DatasetSnapshot
from mojilex_cli.pipeline import schema_upgrade
from mojilex_cli.pipeline.runner import _changed_paths
from mojilex_cli.pipeline.schema_upgrade import upgrade_private_staging_schema
from mojilex_cli.pipeline.workspaces import staging_workspace_path

RUN_ID = "mlxrun_" + "a" * 32
SCHEMA = Path("schemas/v1/emoji.schema.json")
BUNDLED = Path(schema_upgrade.__file__).resolve().parents[1] / SCHEMA


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def staging(tmp_path: Path) -> tuple[Path, Path, str]:
    runs = tmp_path / "runs"
    root = staging_workspace_path(runs, RUN_ID)
    schema = root / SCHEMA
    schema.parent.mkdir(parents=True)
    previous = json.loads(BUNDLED.read_bytes())
    variants = previous["allOf"][-1]["then"]["properties"]["media"]["items"]["allOf"][-1]["oneOf"]
    variants[:] = [entry for entry in variants if entry["properties"]["format"]["const"] != "png"]
    schema.write_text(json.dumps(previous, indent=4, sort_keys=True) + "\n", encoding="utf-8")
    _git(root, "init", "--initial-branch=main")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "base",
    )
    _git(root, "remote", "add", "origin", "https://github.com/MojiLex/mojilex.git")
    return runs, root, _git(root, "rev-parse", "HEAD")


def _upgrade(staging: tuple[Path, Path, str], **overrides: str | Path) -> bool:
    runs, root, revision = staging
    return upgrade_private_staging_schema(
        Path(overrides.get("staging", root)),
        runs_dir=Path(overrides.get("runs_dir", runs)),
        run_id=str(overrides.get("run_id", RUN_ID)),
        base_revision=str(overrides.get("base_revision", revision)),
        target_repository=str(overrides.get("target_repository", "MojiLex/mojilex")),
    )


def test_only_known_private_schema_is_upgraded_without_changing_head_or_other_files(staging):
    _, root, revision = staging
    (root / "saved-progress.json").write_bytes(b"unchanged progress")
    assert _upgrade(staging)
    assert (root / SCHEMA).read_bytes() == BUNDLED.read_bytes()
    assert _git(root, "rev-parse", "HEAD") == revision
    assert (root / "saved-progress.json").read_bytes() == b"unchanged progress"
    assert _git(root, "diff", "--name-only") == SCHEMA.as_posix()
    assert _git(root, "diff", "--cached", "--name-only") == ""
    modified = (root / SCHEMA).stat().st_mtime_ns
    assert not _upgrade(staging)
    assert (root / SCHEMA).stat().st_mtime_ns == modified


@pytest.mark.parametrize("future", [False, True])
def test_custom_and_future_schemas_are_never_overwritten(staging, future):
    _, root, _ = staging
    schema = root / SCHEMA
    payload = json.loads((BUNDLED if future else schema).read_bytes())
    payload["title"] = "A deliberately different contract"
    original = json.dumps(payload).encode()
    schema.write_bytes(original)
    assert not _upgrade(staging)
    assert schema.read_bytes() == original


def test_numeric_zero_is_not_treated_as_schema_false(staging):
    _, root, _ = staging
    schema = root / SCHEMA
    payload = json.loads(schema.read_bytes())
    payload["additionalProperties"] = 0
    original = json.dumps(payload).encode()
    schema.write_bytes(original)
    assert not _upgrade(staging)
    assert schema.read_bytes() == original


@pytest.mark.parametrize(
    "override",
    [
        {"run_id": "mlxrun_" + "b" * 32},
        {"run_id": "../another-run"},
        {"base_revision": "0" * 40},
        {"target_repository": "someone/mojilex"},
        {"target_repository": "https://example.invalid/MojiLex/mojilex"},
    ],
)
def test_wrong_checkpoint_binding_leaves_schema_untouched(staging, override):
    schema = staging[1] / SCHEMA
    original = schema.read_bytes()
    assert not _upgrade(staging, **override)
    assert schema.read_bytes() == original


def test_repository_outside_exact_private_workspace_is_not_upgraded(staging, tmp_path):
    runs, root, revision = staging
    other = tmp_path / "user-repository"
    root.rename(other)
    original = (other / SCHEMA).read_bytes()
    assert not _upgrade((runs, other, revision))
    assert (other / SCHEMA).read_bytes() == original


def test_different_git_origin_is_not_upgraded(staging):
    _, root, _ = staging
    _git(root, "remote", "set-url", "origin", "https://github.com/another/repository.git")
    original = (root / SCHEMA).read_bytes()
    assert not _upgrade(staging)
    assert (root / SCHEMA).read_bytes() == original


@pytest.mark.parametrize(
    "raw",
    [b"not json", b'{"title":"one","title":"two"}', b" " * 262145],
    ids=["invalid-json", "duplicate-key", "oversized"],
)
def test_invalid_duplicate_or_oversized_schema_is_preserved(staging, raw):
    schema = staging[1] / SCHEMA
    schema.write_bytes(raw)
    assert not _upgrade(staging)
    assert schema.read_bytes() == raw


def test_schema_symlink_is_not_followed(staging):
    _, root, _ = staging
    schema = root / SCHEMA
    original = schema.read_bytes()
    other = root / "original-schema.json"
    schema.rename(other)
    try:
        schema.symlink_to(other)
    except OSError:
        pytest.skip("symbolic links are unavailable")
    assert not _upgrade(staging)
    assert other.read_bytes() == original


def test_failed_atomic_replace_preserves_original_schema_and_cleans_own_temporary(
    staging, monkeypatch
):
    schema = staging[1] / SCHEMA
    original = schema.read_bytes()

    def fail(*args):
        raise OSError("synthetic replacement failure")

    monkeypatch.setattr(schema_upgrade.os, "replace", fail)
    with pytest.raises(OSError, match="synthetic replacement"):
        _upgrade(staging)
    assert schema.read_bytes() == original
    assert list(schema.parent.glob(".mojilex-schema-*")) == []


def test_schema_upgrade_does_not_enter_publication_metadata_diff(staging):
    _, root, _ = staging
    relative = PurePosixPath(SCHEMA.as_posix())
    before = DatasetSnapshot(
        root=root, manifest={}, source_bytes={relative: (root / SCHEMA).read_bytes()}
    )
    assert _upgrade(staging)
    after = before.clone()
    after.source_bytes[relative] = (root / SCHEMA).read_bytes()
    assert _changed_paths(before, after) == ()
    after.manifest["description"] = "changed metadata"
    assert _changed_paths(before, after) == (PurePosixPath("dataset.json"),)


def _old_literal_schema(staging):
    relative = Path("schemas/v1/facets.schema.json")
    bundled = Path(schema_upgrade.__file__).resolve().parents[1] / relative
    target = staging[1] / relative
    payload = json.loads(bundled.read_bytes())
    value = payload["$defs"]["textItem"]["properties"]["value"]
    value.pop("not")
    value["pattern"] = r"^[^\u0000-\u001f\u007f<>]+$"
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target, bundled


def test_existing_staging_literal_contract_is_upgraded_independently(staging):
    assert _upgrade(staging)  # PNG schema already current on the second call.
    target, bundled = _old_literal_schema(staging)
    assert _upgrade(staging)
    assert target.read_bytes() == bundled.read_bytes()
    assert _git(staging[1], "rev-parse", "HEAD") == staging[2]
    assert not _upgrade(staging)


@pytest.mark.parametrize("custom", [True, False])
def test_custom_literal_schema_and_custom_repository_are_preserved(staging, custom):
    assert _upgrade(staging)
    target, _ = _old_literal_schema(staging)
    if custom:
        payload = json.loads(target.read_bytes())
        payload["title"] = "Custom literal policy"
        target.write_text(json.dumps(payload), encoding="utf-8")
    original = target.read_bytes()
    overrides = {} if custom else {"target_repository": "someone/mojilex"}
    assert not _upgrade(staging, **overrides)
    assert target.read_bytes() == original
