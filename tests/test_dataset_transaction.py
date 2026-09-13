import errno
import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path, PurePosixPath

import pytest

import mojilex_cli.dataset.transaction as transaction_module
from mojilex_cli.dataset.repository import DatasetLoadError, load_dataset
from mojilex_cli.dataset.staging import AtomicDatasetWriter, AtomicWriteError, apply_snapshot
from mojilex_cli.dataset.transaction import (
    TRANSACTION_DIRECTORY_NAME,
    DurableDatasetTransaction,
    _transaction_lock_path,
    recover_pending_dataset_transaction,
)
from test_dataset_helpers import write_fixture


def _create_windows_junction(link: Path, target: Path) -> None:
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        pytest.skip(f"junction creation is unavailable: {completed.stderr.strip()}")


def test_successful_commit_removes_durable_transaction(tmp_path) -> None:
    target = tmp_path / "value.json"
    target.write_bytes(b"before")
    writer = AtomicDatasetWriter(tmp_path)
    writer.stage_bytes("value.json", b"after")

    assert writer.commit() == (PurePosixPath("value.json"),)

    assert target.read_bytes() == b"after"
    assert not (tmp_path / TRANSACTION_DIRECTORY_NAME).exists()


def test_load_dataset_recovers_hard_crash_after_first_replacement(tmp_path) -> None:
    root = tmp_path / "dataset"
    original = write_fixture(root)
    original_manifest = (root / "dataset.json").read_bytes()
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path

        import mojilex_cli.dataset.staging as staging

        root = Path(sys.argv[1])
        writer = staging.AtomicDatasetWriter(root)
        writer.stage_bytes("dataset.json", b"{invalid")
        writer.stage_bytes("zz-crash.json", b"created-after-crash")
        real_replace = os.replace

        def crash_after_first_target(source, destination):
            real_replace(source, destination)
            if ".mojilex-stage-" in str(source):
                os._exit(91)

        staging.os.replace = crash_after_first_target
        writer.commit()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 91, completed.stderr
    assert (root / "dataset.json").read_bytes() == b"{invalid"
    assert (root / TRANSACTION_DIRECTORY_NAME).is_dir()

    recovered = load_dataset(root)

    assert recovered.manifest == original.manifest
    assert (root / "dataset.json").read_bytes() == original_manifest
    assert not (root / "zz-crash.json").exists()
    assert not (root / TRANSACTION_DIRECTORY_NAME).exists()


@pytest.mark.parametrize(
    "phase",
    [
        "after-directory",
        "preparing-temp",
        "after-preparing",
        "after-directories",
        "after-first-payload",
        "after-payloads",
        "manifest-temp",
        "after-manifest",
        "state-temp",
        "after-ready",
    ],
)
def test_prepare_crash_at_every_durable_phase_is_cleaned(tmp_path, phase) -> None:
    root = tmp_path / phase
    root.mkdir()
    target = root / "value.json"
    target.write_bytes(b"before")
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path

        import mojilex_cli.dataset.transaction as transaction_module
        from mojilex_cli.dataset.staging import AtomicDatasetWriter

        root = Path(sys.argv[1])
        phase = sys.argv[2]
        transaction_directory = root / transaction_module.TRANSACTION_DIRECTORY_NAME
        real_fsync_directory = transaction_module._fsync_directory
        real_replace = os.replace
        real_write = transaction_module._write_durable_file
        payload_writes = 0

        def crash():
            os._exit(93)

        def fsync_directory(path):
            real_fsync_directory(path)
            path = Path(path)
            if (
                phase == "after-directory"
                and path == root
                and transaction_directory.is_dir()
                and not any(transaction_directory.iterdir())
            ):
                crash()
            if (
                phase == "after-directories"
                and path == transaction_directory
                and (transaction_directory / "preparing.json").is_file()
                and (transaction_directory / "backups").is_dir()
                and (transaction_directory / "staged").is_dir()
                and not any((transaction_directory / "backups").iterdir())
                and not any((transaction_directory / "staged").iterdir())
            ):
                crash()
            if phase == "after-payloads" and path == transaction_directory / "staged":
                crash()

        def write_durable_file(path, data):
            global payload_writes
            path = Path(path)
            if phase == "preparing-temp" and path.name == ".preparing.tmp":
                with path.open("xb") as handle:
                    handle.write(data[: max(1, len(data) // 2)])
                    handle.flush()
                    os.fsync(handle.fileno())
                crash()
            if phase == "manifest-temp" and path.name == ".manifest.tmp":
                with path.open("xb") as handle:
                    handle.write(data[: max(1, len(data) // 2)])
                    handle.flush()
                    os.fsync(handle.fileno())
                crash()
            if phase == "state-temp" and path.name == ".state.tmp":
                with path.open("xb") as handle:
                    handle.write(data[:2])
                    handle.flush()
                    os.fsync(handle.fileno())
                crash()
            real_write(path, data)
            if path.name.startswith((".mojilex-backup-", ".mojilex-stage-")):
                payload_writes += 1
                if phase == "after-first-payload" and payload_writes == 1:
                    crash()

        def replace(source, destination):
            real_replace(source, destination)
            destination = Path(destination)
            if phase == "after-preparing" and destination.name == "preparing.json":
                crash()
            if phase == "after-manifest" and destination.name == "manifest.json":
                crash()
            if phase == "after-ready" and destination.name == "state":
                crash()

        transaction_module._fsync_directory = fsync_directory
        transaction_module._write_durable_file = write_durable_file
        transaction_module.os.replace = replace

        writer = AtomicDatasetWriter(root)
        writer.stage_bytes("value.json", b"after")
        writer.stage_bytes("created.json", b"created")
        writer.commit()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(root), phase],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 93, completed.stderr
    assert recover_pending_dataset_transaction(root)
    assert target.read_bytes() == b"before"
    assert not (root / "created.json").exists()
    assert not (root / TRANSACTION_DIRECTORY_NAME).exists()


def test_active_writer_lock_prevents_concurrent_load_or_recovery(tmp_path) -> None:
    root = tmp_path / "dataset"
    original = write_fixture(root)
    replacement_manifest = dict(original.manifest)
    replacement_manifest["transaction_test"] = "committed"
    replacement = tmp_path / "replacement.json"
    replacement.write_text(
        json.dumps(replacement_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paused = tmp_path / "writer-paused"
    release = tmp_path / "writer-release"
    script = textwrap.dedent(
        """
        import os
        import sys
        import time
        from pathlib import Path

        import mojilex_cli.dataset.staging as staging

        root, replacement, paused, release = map(Path, sys.argv[1:])
        writer = staging.AtomicDatasetWriter(root)
        writer.stage_bytes("dataset.json", replacement.read_bytes())
        writer.stage_bytes("zz-lock.json", b"second-target")
        real_replace = os.replace
        did_pause = False

        def pause_after_first_target(source, destination):
            global did_pause
            real_replace(source, destination)
            if ".mojilex-stage-" in str(source) and not did_pause:
                did_pause = True
                paused.write_text("paused", encoding="ascii")
                deadline = time.monotonic() + 30
                while not release.exists():
                    if time.monotonic() >= deadline:
                        os._exit(92)
                    time.sleep(0.02)

        staging.os.replace = pause_after_first_target
        writer.commit()
        """
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(root),
            str(replacement),
            str(paused),
            str(release),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not paused.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert paused.exists(), process.communicate(timeout=5)[1]
        partially_applied = (root / "dataset.json").read_bytes()
        assert partially_applied == replacement.read_bytes()
        assert not (root / "zz-lock.json").exists()

        with pytest.raises(AtomicWriteError, match="locked by another active"):
            load_dataset(root)
        with pytest.raises(AtomicWriteError, match="locked by another active"):
            AtomicDatasetWriter(root)

        assert (root / "dataset.json").read_bytes() == partially_applied
        assert not (root / "zz-lock.json").exists()
    finally:
        release.write_text("release", encoding="ascii")
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, f"stdout={stdout!r}\nstderr={stderr!r}"

    recovered = load_dataset(root)
    assert recovered.manifest["transaction_test"] == "committed"
    assert (root / "zz-lock.json").read_bytes() == b"second-target"
    assert not (root / TRANSACTION_DIRECTORY_NAME).exists()


@pytest.mark.parametrize(
    "phase",
    [
        "unlink-backup",
        "unlink-staged",
        "rmdir-backups",
        "rmdir-staged",
        "unlink-manifest-temp",
        "unlink-preparing-temp",
        "unlink-state-temp",
        "unlink-manifest",
        "fsync-1",
        "unlink-state",
        "fsync-2",
        "unlink-preparing",
        "rmdir-journal",
        "fsync-3",
    ],
)
def test_committed_cleanup_crash_at_every_filesystem_step_recovers(tmp_path, phase) -> None:
    root = tmp_path / phase
    root.mkdir()
    target = root / "value.json"
    target.write_bytes(b"before")
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path, PurePosixPath

        import mojilex_cli.dataset.transaction as transaction_module

        root = Path(sys.argv[1])
        phase = sys.argv[2]
        target = root / "value.json"
        transaction = transaction_module.DurableDatasetTransaction.prepare(
            root,
            {PurePosixPath("value.json"): b"after"},
        )
        entry = transaction.entries[0]
        target.write_bytes(transaction.staged_path(entry).read_bytes())
        transaction.sync_target_parent(entry)
        transaction.mark_committed()

        real_unlink = Path.unlink
        real_rmdir = Path.rmdir
        real_fsync_directory = transaction_module._fsync_directory
        fsync_count = 0
        unlink_phases = {
            ".mojilex-backup-000000": "unlink-backup",
            ".mojilex-stage-000000": "unlink-staged",
            ".manifest.tmp": "unlink-manifest-temp",
            ".preparing.tmp": "unlink-preparing-temp",
            ".state.tmp": "unlink-state-temp",
            "manifest.json": "unlink-manifest",
            "state": "unlink-state",
            "preparing.json": "unlink-preparing",
        }
        rmdir_phases = {
            "backups": "rmdir-backups",
            "staged": "rmdir-staged",
            transaction_module.TRANSACTION_DIRECTORY_NAME: "rmdir-journal",
        }

        def crash():
            os._exit(95)

        def unlink(path, *args, **kwargs):
            result = real_unlink(path, *args, **kwargs)
            if unlink_phases.get(path.name) == phase:
                crash()
            return result

        def rmdir(path, *args, **kwargs):
            result = real_rmdir(path, *args, **kwargs)
            if rmdir_phases.get(path.name) == phase:
                crash()
            return result

        def fsync_directory(path):
            global fsync_count
            real_fsync_directory(path)
            fsync_count += 1
            if phase == f"fsync-{fsync_count}":
                crash()

        Path.unlink = unlink
        Path.rmdir = rmdir
        transaction_module._fsync_directory = fsync_directory
        transaction.cleanup()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(root), phase],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 95, completed.stderr
    assert target.read_bytes() == b"after"
    recover_pending_dataset_transaction(root)
    assert target.read_bytes() == b"after"
    assert not (root / TRANSACTION_DIRECTORY_NAME).exists()


def test_overlapping_stale_snapshot_fails_without_losing_first_commit(tmp_path) -> None:
    write_fixture(tmp_path)
    first_before = load_dataset(tmp_path)
    second_before = load_dataset(tmp_path)
    first_after = first_before.clone()
    second_after = second_before.clone()
    collection_id = next(iter(first_after.collections))
    first_after.collections[collection_id].title = "first writer"
    second_after.collections[collection_id].title = "stale second writer"

    apply_snapshot(first_before, first_after)
    with pytest.raises(AtomicWriteError, match="changed since the snapshot was loaded"):
        apply_snapshot(second_before, second_after)

    persisted = load_dataset(tmp_path)
    assert persisted.collections[collection_id].title == "first writer"


def test_nonoverlapping_stale_snapshot_fails_without_losing_first_commit(tmp_path) -> None:
    write_fixture(tmp_path)
    first_before = load_dataset(tmp_path)
    second_before = load_dataset(tmp_path)
    first_after = first_before.clone()
    second_after = second_before.clone()
    collection_id = next(iter(first_after.collections))
    first_after.collections[collection_id].title = "first writer"
    second_after.manifest["nonoverlapping_second_writer"] = True

    apply_snapshot(first_before, first_after)
    with pytest.raises(AtomicWriteError, match="changed since the snapshot was loaded"):
        apply_snapshot(second_before, second_after)

    persisted = load_dataset(tmp_path)
    assert persisted.collections[collection_id].title == "first writer"
    assert "nonoverlapping_second_writer" not in persisted.manifest


@pytest.mark.parametrize("mutation", ["add", "delete"])
def test_stale_snapshot_detects_canonical_path_set_changes(tmp_path, mutation) -> None:
    write_fixture(tmp_path)
    stale_before = load_dataset(tmp_path)
    stale_after = stale_before.clone()
    stale_after.manifest["must_not_be_written"] = True
    writer = AtomicDatasetWriter(tmp_path)
    if mutation == "add":
        changed_path = PurePosixPath("data/telegram/emojis/aa/bb.jsonl")
        writer.stage_bytes(changed_path, b"{}\n")
    else:
        changed_path = next(
            path for path in stale_before.source_bytes if path.name == "memberships.jsonl"
        )
        writer.stage_delete(changed_path)
    writer.commit()

    with pytest.raises(AtomicWriteError, match="canonical path set differs"):
        apply_snapshot(stale_before, stale_after)

    destination = tmp_path.joinpath(*changed_path.parts)
    assert destination.exists() is (mutation == "add")
    assert "must_not_be_written" not in json.loads((tmp_path / "dataset.json").read_bytes())


def test_transaction_lock_path_is_root_local_stable_and_ignored(tmp_path) -> None:
    root = (tmp_path / "dataset").resolve()
    root.mkdir()

    lock_path = _transaction_lock_path(root)

    assert lock_path == root / ".mojilex" / "locks" / "dataset-transaction-v1.lock"
    assert lock_path.is_relative_to(root)


def test_environment_changes_cannot_select_an_alternate_lock_namespace(tmp_path) -> None:
    root = (tmp_path / "dataset").resolve()
    root.mkdir()
    primary_lock = transaction_module._acquire_transaction_lock(root)
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        import mojilex_cli.dataset.transaction as transaction_module

        root = Path(sys.argv[1])
        try:
            transaction_module.recover_pending_dataset_transaction(root)
        except transaction_module.AtomicWriteError as exc:
            print(exc)
            raise SystemExit(73)
        raise SystemExit(0)
        """
    )
    environment = os.environ.copy()
    alternate = str(tmp_path / "alternate-environment-root")
    environment.update(
        {
            "HOME": alternate,
            "LOCALAPPDATA": alternate,
            "TEMP": alternate,
            "TMP": alternate,
            "XDG_CACHE_HOME": alternate,
        }
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script, str(root)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    finally:
        primary_lock.release()

    assert completed.returncode == 73, completed.stderr
    assert "locked by another active atomic writer" in completed.stdout


def test_native_lock_failure_never_falls_back_to_a_soft_lock(tmp_path, monkeypatch) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    options = {}

    class NativeLockUnavailable:
        def __init__(self, path, **kwargs):
            options["path"] = path
            options.update(kwargs)

        def acquire(self, *, timeout):
            raise OSError(errno.ENOSYS, "native locking unavailable")

    monkeypatch.setattr(transaction_module, "FileLock", NativeLockUnavailable)

    with pytest.raises(AtomicWriteError, match="could not be acquired safely"):
        transaction_module._acquire_transaction_lock(root)

    assert options["fallback_to_soft"] is False


def test_external_lock_name_cannot_escape_lock_namespace(tmp_path) -> None:
    root = tmp_path / "dataset"
    lock_root = tmp_path / "lock-root"
    root.mkdir()
    lock_root.mkdir()

    with pytest.raises(AtomicWriteError, match="lock name is unsafe"):
        AtomicDatasetWriter(
            root,
            transaction_lock_root=lock_root,
            transaction_lock_name="../outside.lock",
        )

    assert not (tmp_path / "outside.lock").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_external_lock_root_junction_is_rejected(tmp_path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside-lock-state"
    outside.mkdir()
    alias = tmp_path / "lock-root-alias"
    _create_windows_junction(alias, outside)

    with pytest.raises(AtomicWriteError, match="link or reparse point"):
        AtomicDatasetWriter(root, transaction_lock_root=alias)

    assert not (outside / ".mojilex").exists()


def test_load_fails_closed_with_explicit_diagnostic_when_lock_storage_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "dataset"
    write_fixture(root)
    # POSIX filelock keeps its inode after release; Windows removes it.
    (root / ".mojilex" / "locks" / "dataset-transaction-v1.lock").unlink(missing_ok=True)
    (root / ".mojilex" / "locks").rmdir()
    (root / ".mojilex").rmdir()
    real_mkdir = Path.mkdir

    def deny_local_state(path, *args, **kwargs):
        if path == root / ".mojilex":
            raise PermissionError("synthetic read-only dataset")
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny_local_state)

    with pytest.raises(
        AtomicWriteError,
        match="deterministic dataset transaction lock directory is unavailable",
    ):
        load_dataset(root)


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_local_lock_namespace_junction_is_rejected_before_outside_creation(tmp_path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside-state"
    outside.mkdir()
    sentinel = outside / "user-file"
    sentinel.write_bytes(b"preserve")
    _create_windows_junction(root / ".mojilex", outside)

    with pytest.raises(AtomicWriteError, match="lock directory is unavailable"):
        AtomicDatasetWriter(root)

    assert sentinel.read_bytes() == b"preserve"
    assert not (outside / "locks").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_top_level_transaction_junction_fails_closed(tmp_path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside-journal"
    outside.mkdir()
    sentinel = outside / "user-file"
    sentinel.write_bytes(b"preserve")
    _create_windows_junction(root / TRANSACTION_DIRECTORY_NAME, outside)

    with pytest.raises(AtomicWriteError, match="link or reparse point"):
        recover_pending_dataset_transaction(root)

    assert sentinel.read_bytes() == b"preserve"
    assert (root / TRANSACTION_DIRECTORY_NAME).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
@pytest.mark.parametrize("payload_directory", ["backups", "staged"])
def test_payload_directory_junction_never_unlinks_outside_file(tmp_path, payload_directory) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    target = root / "value.json"
    target.write_bytes(b"before")
    directory = root / TRANSACTION_DIRECTORY_NAME
    directory.mkdir()
    outside = tmp_path / f"outside-{payload_directory}"
    outside.mkdir()
    payload_name = (
        ".mojilex-backup-000000" if payload_directory == "backups" else ".mojilex-stage-000000"
    )
    outside_payload = outside / payload_name
    outside_payload.write_bytes(b"before" if payload_directory == "backups" else b"after")
    other = "staged" if payload_directory == "backups" else "backups"
    (directory / other).mkdir()
    _create_windows_junction(directory / payload_directory, outside)
    manifest = {
        "entries": [
            {
                "new_sha256": hashlib.sha256(b"after").hexdigest(),
                "new_size": len(b"after"),
                "old_sha256": hashlib.sha256(b"before").hexdigest(),
                "old_size": len(b"before"),
                "path": "value.json",
            }
        ],
        "format_version": 1,
        "kind": "mojilex-dataset-transaction",
        "root_sha256": hashlib.sha256(
            os.path.normcase(str(root.resolve())).encode("utf-8")
        ).hexdigest(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "state").write_bytes(b"ready\n")

    with pytest.raises(AtomicWriteError, match="link or reparse point"):
        recover_pending_dataset_transaction(root)

    assert outside_payload.read_bytes() in {b"before", b"after"}
    assert target.read_bytes() == b"before"
    assert directory.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_data_ancestor_junction_is_rejected_for_load_and_write(tmp_path) -> None:
    root = tmp_path / "dataset"
    write_fixture(root)
    outside_data = tmp_path / "outside-data"
    (root / "data").rename(outside_data)
    _create_windows_junction(root / "data", outside_data)

    with pytest.raises(DatasetLoadError, match="reparse-point traversal is forbidden"):
        load_dataset(root)
    writer = AtomicDatasetWriter(root)
    with pytest.raises(ValueError, match="reparse-point traversal is forbidden"):
        writer.stage_bytes("data/probe.json", b"must-not-be-written")

    assert not (outside_data / "probe.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_internal_transaction_directory_case_alias_is_rejected_early(tmp_path) -> None:
    writer = AtomicDatasetWriter(tmp_path)

    with pytest.raises(AtomicWriteError, match="non-canonical path"):
        writer.stage_bytes(".MOJILEX-ATOMIC-WRITE/probe", b"must-not-be-written")

    assert not (tmp_path / TRANSACTION_DIRECTORY_NAME).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_case_alias_targets_are_rejected_before_prepare(tmp_path) -> None:
    writer = AtomicDatasetWriter(tmp_path)
    writer.stage_bytes("A.json", b"first")

    with pytest.raises(AtomicWriteError, match="alias the same target"):
        writer.stage_bytes("a.json", b"second")

    assert not (tmp_path / "A.json").exists()
    assert not (tmp_path / TRANSACTION_DIRECTORY_NAME).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_case_aliases_in_recovery_manifest_fail_closed(tmp_path) -> None:
    directory = tmp_path / TRANSACTION_DIRECTORY_NAME
    (directory / "backups").mkdir(parents=True)
    (directory / "staged").mkdir()
    entries = [
        {
            "new_sha256": None,
            "new_size": None,
            "old_sha256": None,
            "old_size": None,
            "path": path,
        }
        for path in ("A.json", "a.json")
    ]
    manifest = {
        "entries": entries,
        "format_version": 1,
        "kind": "mojilex-dataset-transaction",
        "root_sha256": hashlib.sha256(
            os.path.normcase(str(tmp_path.resolve())).encode("utf-8")
        ).hexdigest(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "state").write_bytes(b"ready\n")

    with pytest.raises(AtomicWriteError, match="alias the same target"):
        recover_pending_dataset_transaction(tmp_path)

    assert directory.is_dir()


@pytest.mark.skipif(os.name != "nt", reason="Windows applies Win32 path normalization")
@pytest.mark.parametrize(
    "unsafe_path",
    [
        "x.json.",
        "x.json ",
        "directory./x.json",
        "x.json:alternate-stream",
        "NUL.json",
        "data/aux",
        "data/CoM1.txt",
        "LPT9.any-extension",
        "COM¹.json",
        "aardva~1.jso",
    ],
)
def test_windows_ambiguous_or_reserved_paths_are_rejected_before_journal(
    tmp_path, unsafe_path
) -> None:
    writer = AtomicDatasetWriter(tmp_path)

    with pytest.raises(AtomicWriteError, match="Windows path"):
        writer.stage_bytes(unsafe_path, b"must-not-be-written")

    assert not (tmp_path / TRANSACTION_DIRECTORY_NAME).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows applies Win32 path normalization")
def test_windows_path_policy_preserves_valid_unicode(tmp_path) -> None:
    writer = AtomicDatasetWriter(tmp_path)
    writer.stage_bytes("данные/эмодзи.json", b"{}\n")

    writer.commit()

    assert (tmp_path / "данные" / "эмодзи.json").read_bytes() == b"{}\n"


@pytest.mark.skipif(os.name != "nt", reason="Windows 8.3 alias regression")
def test_real_windows_short_name_alias_is_rejected_when_available(tmp_path) -> None:
    import ctypes

    long_path = tmp_path / "aardvark-transaction-value.json"
    long_path.write_bytes(b"before")
    buffer = ctypes.create_unicode_buffer(32_768)
    length = ctypes.windll.kernel32.GetShortPathNameW(str(long_path), buffer, len(buffer))
    if length == 0 or length >= len(buffer):
        pytest.skip("GetShortPathNameW did not return a usable path")
    short_name = Path(buffer.value).name
    if "~" not in short_name or short_name.casefold() == long_path.name.casefold():
        pytest.skip("8.3 names are disabled on this test volume")
    writer = AtomicDatasetWriter(tmp_path)
    writer.stage_bytes(long_path.name, b"first")

    with pytest.raises(AtomicWriteError, match="DOS short-name shape"):
        writer.stage_bytes(short_name, b"second")

    assert long_path.read_bytes() == b"before"
    assert not (tmp_path / TRANSACTION_DIRECTORY_NAME).exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows applies Win32 path normalization")
@pytest.mark.parametrize("unsafe_path", ["x.json.", "x.json:stream", "NUL.json", "aardva~1.jso"])
def test_windows_unsafe_paths_in_recovery_manifest_fail_closed(tmp_path, unsafe_path) -> None:
    directory = tmp_path / TRANSACTION_DIRECTORY_NAME
    (directory / "backups").mkdir(parents=True)
    (directory / "staged").mkdir()
    manifest = {
        "entries": [
            {
                "new_sha256": None,
                "new_size": None,
                "old_sha256": None,
                "old_size": None,
                "path": unsafe_path,
            }
        ],
        "format_version": 1,
        "kind": "mojilex-dataset-transaction",
        "root_sha256": hashlib.sha256(
            os.path.normcase(str(tmp_path.resolve())).encode("utf-8")
        ).hexdigest(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "state").write_bytes(b"ready\n")

    with pytest.raises(AtomicWriteError, match="Windows path"):
        recover_pending_dataset_transaction(tmp_path)

    assert directory.is_dir()


@pytest.mark.parametrize("phase", ["before", "after"])
def test_hard_crash_during_direct_backup_restore_recovers_without_orphan(tmp_path, phase) -> None:
    root = tmp_path / phase
    root.mkdir()
    target = root / "value.json"
    target.write_bytes(b"before")
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path, PurePosixPath

        import mojilex_cli.dataset.transaction as transaction_module

        root = Path(sys.argv[1])
        phase = sys.argv[2]
        target = root / "value.json"
        transaction = transaction_module.DurableDatasetTransaction.prepare(
            root,
            {PurePosixPath("value.json"): b"after"},
        )
        entry = transaction.entries[0]
        os.replace(transaction.staged_path(entry), target)
        transaction.sync_target_parent(entry)
        real_replace = os.replace

        def crash_during_backup_restore(source, destination):
            if ".mojilex-backup-" in str(source):
                if phase == "before":
                    os._exit(96)
                real_replace(source, destination)
                os._exit(96)
            real_replace(source, destination)

        transaction_module.os.replace = crash_during_backup_restore
        transaction.rollback()
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(root), phase],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 96, completed.stderr
    assert recover_pending_dataset_transaction(root)
    assert target.read_bytes() == b"before"
    assert not (root / TRANSACTION_DIRECTORY_NAME).exists()
    assert not list(root.rglob(".mojilex-recovery-*"))


def test_recovery_refuses_to_overwrite_unrelated_target_edit(tmp_path) -> None:
    target = tmp_path / "value.json"
    target.write_bytes(b"before")
    transaction = DurableDatasetTransaction.prepare(
        tmp_path,
        {PurePosixPath("value.json"): b"transaction-value"},
    )
    entry = transaction.entries[0]
    os.replace(transaction.staged_path(entry), target)
    target.write_bytes(b"user-edit")
    transaction._release_lock()

    with pytest.raises(AtomicWriteError, match="changed outside"):
        recover_pending_dataset_transaction(tmp_path)

    assert target.read_bytes() == b"user-edit"
    assert (tmp_path / TRANSACTION_DIRECTORY_NAME).is_dir()


def test_recovery_finalizes_a_durable_committed_transaction(tmp_path) -> None:
    updated = tmp_path / "updated.json"
    deleted = tmp_path / "deleted.json"
    updated.write_bytes(b"before")
    deleted.write_bytes(b"delete-me")
    transaction = DurableDatasetTransaction.prepare(
        tmp_path,
        {
            PurePosixPath("created.json"): b"created",
            PurePosixPath("deleted.json"): None,
            PurePosixPath("updated.json"): b"after",
        },
    )
    for entry in transaction.entries:
        destination = tmp_path.joinpath(*entry.relative.parts)
        if entry.new_sha256 is None:
            destination.unlink(missing_ok=True)
        else:
            os.replace(transaction.staged_path(entry), destination)
        transaction.sync_target_parent(entry)
    transaction.mark_committed()
    transaction._release_lock()

    assert recover_pending_dataset_transaction(tmp_path)

    assert updated.read_bytes() == b"after"
    assert not deleted.exists()
    assert (tmp_path / "created.json").read_bytes() == b"created"
    assert not (tmp_path / TRANSACTION_DIRECTORY_NAME).exists()


def test_markerless_pure_delete_is_rolled_back(tmp_path) -> None:
    target = tmp_path / "delete.json"
    target.write_bytes(b"must-survive")
    transaction = DurableDatasetTransaction.prepare(
        tmp_path,
        {PurePosixPath("delete.json"): None},
    )
    target.unlink()
    (transaction.directory / "state").unlink()
    transaction._release_lock()

    assert recover_pending_dataset_transaction(tmp_path)

    assert target.read_bytes() == b"must-survive"
    assert not (tmp_path / TRANSACTION_DIRECTORY_NAME).exists()


def test_markerless_mixed_delete_first_is_rolled_back(tmp_path) -> None:
    deleted = tmp_path / "a-delete.json"
    updated = tmp_path / "z-update.json"
    deleted.write_bytes(b"deleted-before")
    updated.write_bytes(b"updated-before")
    transaction = DurableDatasetTransaction.prepare(
        tmp_path,
        {
            PurePosixPath("a-delete.json"): None,
            PurePosixPath("z-update.json"): b"updated-after",
        },
    )
    deleted.unlink()
    (transaction.directory / "state").unlink()
    transaction._release_lock()

    assert recover_pending_dataset_transaction(tmp_path)

    assert deleted.read_bytes() == b"deleted-before"
    assert updated.read_bytes() == b"updated-before"
    assert not (tmp_path / TRANSACTION_DIRECTORY_NAME).exists()


@pytest.mark.parametrize("phase", ["state-temp", "after-ready"])
def test_markerless_recovery_crash_while_restoring_ready_marker_retries(tmp_path, phase) -> None:
    root = tmp_path / phase
    root.mkdir()
    target = root / "value.json"
    target.write_bytes(b"before")
    script = textwrap.dedent(
        """
        import os
        import sys
        from pathlib import Path, PurePosixPath

        import mojilex_cli.dataset.transaction as transaction_module

        root = Path(sys.argv[1])
        phase = sys.argv[2]
        target = root / "value.json"
        transaction = transaction_module.DurableDatasetTransaction.prepare(
            root,
            {PurePosixPath("value.json"): b"after"},
        )
        entry = transaction.entries[0]
        os.replace(transaction.staged_path(entry), target)
        transaction.sync_target_parent(entry)
        (transaction.directory / "state").unlink()
        transaction._release_lock()
        real_write = transaction_module._write_durable_file
        real_replace = os.replace

        def crash():
            os._exit(97)

        def write_durable_file(path, data):
            path = Path(path)
            if phase == "state-temp" and path.name == ".state.tmp":
                with path.open("xb") as handle:
                    handle.write(b"rea")
                    handle.flush()
                    os.fsync(handle.fileno())
                crash()
            real_write(path, data)

        def replace(source, destination):
            real_replace(source, destination)
            if phase == "after-ready" and Path(destination).name == "state":
                crash()

        transaction_module._write_durable_file = write_durable_file
        transaction_module.os.replace = replace
        transaction_module.recover_pending_dataset_transaction(root)
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(root), phase],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 97, completed.stderr
    assert recover_pending_dataset_transaction(root)
    assert target.read_bytes() == b"before"
    assert not (root / TRANSACTION_DIRECTORY_NAME).exists()


def test_sparse_oversized_manifest_is_rejected_before_reading_payload(tmp_path) -> None:
    directory = tmp_path / TRANSACTION_DIRECTORY_NAME
    directory.mkdir()
    manifest = directory / "manifest.json"
    with manifest.open("wb") as handle:
        handle.seek(4 * 1024 * 1024)
        handle.write(b"x")

    with pytest.raises(AtomicWriteError, match="manifest exceeds its size limit"):
        recover_pending_dataset_transaction(tmp_path)

    assert manifest.stat().st_size == 4 * 1024 * 1024 + 1
    assert directory.is_dir()


def test_sparse_oversized_state_marker_is_rejected_before_reading_it(tmp_path) -> None:
    target = tmp_path / "value.json"
    target.write_bytes(b"before")
    transaction = DurableDatasetTransaction.prepare(
        tmp_path,
        {PurePosixPath("value.json"): b"after"},
    )
    state = transaction.directory / "state"
    with state.open("wb") as handle:
        handle.seek(1024 * 1024)
        handle.write(b"x")
    transaction._release_lock()

    with pytest.raises(AtomicWriteError, match="state marker exceeds its size limit"):
        recover_pending_dataset_transaction(tmp_path)

    assert target.read_bytes() == b"before"
    assert transaction.directory.is_dir()


def test_sparse_oversized_preparation_marker_is_rejected_before_reading_it(tmp_path) -> None:
    directory = tmp_path / TRANSACTION_DIRECTORY_NAME
    directory.mkdir()
    preparing = directory / "preparing.json"
    with preparing.open("wb") as handle:
        handle.seek(1024)
        handle.write(b"x")

    with pytest.raises(AtomicWriteError, match="preparation marker exceeds its size limit"):
        recover_pending_dataset_transaction(tmp_path)

    assert preparing.stat().st_size == 1025
    assert directory.is_dir()


def test_corrupt_transaction_manifest_fails_closed(tmp_path) -> None:
    target = tmp_path / "dataset.json"
    target.write_bytes(b"untouched")
    directory = tmp_path / TRANSACTION_DIRECTORY_NAME
    directory.mkdir()
    (directory / "manifest.json").write_bytes(b"{not-json")
    (directory / "state").write_bytes(b"ready\n")

    with pytest.raises(AtomicWriteError, match="not valid JSON"):
        recover_pending_dataset_transaction(tmp_path)

    assert target.read_bytes() == b"untouched"
    assert directory.is_dir()


def test_path_traversal_transaction_manifest_fails_closed(tmp_path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    target = root / "dataset.json"
    target.write_bytes(b"untouched")
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"user-data")
    directory = root / TRANSACTION_DIRECTORY_NAME
    (directory / "backups").mkdir(parents=True)
    (directory / "staged").mkdir()
    replacement = b"malicious"
    manifest = {
        "entries": [
            {
                "new_sha256": hashlib.sha256(replacement).hexdigest(),
                "new_size": len(replacement),
                "old_sha256": None,
                "old_size": None,
                "path": "../outside.json",
            }
        ],
        "format_version": 1,
        "kind": "mojilex-dataset-transaction",
        "root_sha256": hashlib.sha256(
            os.path.normcase(str(root.resolve())).encode("utf-8")
        ).hexdigest(),
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (directory / "state").write_bytes(b"ready\n")

    with pytest.raises(AtomicWriteError, match=r"unsafe (?:Windows )?path"):
        load_dataset(root)

    assert target.read_bytes() == b"untouched"
    assert outside.read_bytes() == b"user-data"
    assert directory.is_dir()
