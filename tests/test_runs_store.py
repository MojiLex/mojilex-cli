from pathlib import Path

import pytest

from mojilex_cli.runs import (
    PublicationCheckpoint,
    ResumeIncompatibleError,
    RunLockedError,
    RunStore,
    RunStoreError,
    new_checkpoint,
)
from mojilex_cli.runs import store as store_module


def test_checkpoint_roundtrip_is_atomic_and_secret_free(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={"source": "https://t.me/addemoji/Pack"},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    path = store.save(checkpoint)
    assert path.read_bytes().endswith(b"\n")
    assert store.load(checkpoint.run_id) == checkpoint
    with pytest.raises(ResumeIncompatibleError):
        store.load_for_resume(checkpoint.run_id, schema_version="2.0.0")


def test_large_legacy_checkpoint_remains_readable_and_saves_compactly(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    checkpoint = new_checkpoint(
        command="import",
        safe_parameters={"sources": ["https://t.me/addemoji/Pack"]},
        cli_version="0.2.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    element = store_module.ElementCheckpoint(
        stage="fingerprint_ready",
        source_descriptor_sha256="b" * 64,
        media_sha256=("c" * 64,),
        deterministic_cache_key="d" * 64,
        fingerprint_complete=True,
        palette_complete=True,
    )
    checkpoint = checkpoint.model_copy(
        update={"elements": {str(index): element for index in range(20_000)}}
    )
    path = store.root / f"{checkpoint.run_id}.json"
    historical = checkpoint.model_dump_json(indent=2).encode("utf-8") + b"\n"
    assert len(historical) > 8 * 1024 * 1024
    path.write_bytes(historical)
    assert store.load(checkpoint.run_id) == checkpoint

    store.save(checkpoint)
    compact = path.read_bytes()
    assert 8 * 1024 * 1024 < len(compact) < len(historical)
    assert compact.count(b"\n") == 1
    assert store.load(checkpoint.run_id) == checkpoint


def test_oversized_checkpoint_write_preserves_previous_state(tmp_path: Path, monkeypatch) -> None:
    store = RunStore(tmp_path / "runs")
    checkpoint = new_checkpoint(
        command="import",
        safe_parameters={},
        cli_version="0.2.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    path = store.save(checkpoint)
    original = path.read_bytes()
    monkeypatch.setattr(store_module, "_MAX_CHECKPOINT_BYTES", len(original))
    assert store.load(checkpoint.run_id) == checkpoint
    larger = checkpoint.model_copy(update={"safe_parameters": {"sources": ["Pack"]}})
    with pytest.raises(RunStoreError, match="safe size limit"):
        store.save(larger)
    assert path.read_bytes() == original
    assert store.load(checkpoint.run_id) == checkpoint


def test_checkpoint_reader_uses_bounded_read(tmp_path: Path, monkeypatch) -> None:
    import io

    store = RunStore(tmp_path / "runs")
    checkpoint = new_checkpoint(
        command="import",
        safe_parameters={},
        cli_version="0.2.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )
    reads = []

    class BoundedStream(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            assert size == 65
            return super().read(size)

    monkeypatch.setattr(store_module, "_MAX_CHECKPOINT_BYTES", 64)
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: BoundedStream(b"x" * 256))
    with pytest.raises(RunStoreError, match="safe size limit"):
        store.load(checkpoint.run_id)
    assert reads == [65]


def test_checkpoint_rejects_credentials_and_dry_run_writes(tmp_path: Path) -> None:
    with pytest.raises(RunStoreError, match="unsafe field"):
        new_checkpoint(
            command="add",
            safe_parameters={"telegram_bot_token": "secret"},
            cli_version="0.1.0",
            schema_version="1.0.0",
            target_repository="MojiLex/mojilex",
            base_revision="a" * 40,
        )
    store = RunStore(tmp_path / "dry", write_enabled=False)
    with pytest.raises(RunStoreError, match="disabled"):
        store.save(
            new_checkpoint(
                command="add",
                safe_parameters={},
                cli_version="0.1.0",
                schema_version="1.0.0",
                target_repository="MojiLex/mojilex",
                base_revision="a" * 40,
            )
        )


def test_collection_locks_fail_closed(tmp_path: Path) -> None:
    first = RunStore(tmp_path / "runs")
    second = RunStore(tmp_path / "runs")
    with first.collection_lock("telegram", "Pack"):
        with pytest.raises(RunLockedError):
            with second.collection_lock("telegram", "Pack"):
                pass


def test_execution_lock_allows_checkpoint_progress_but_rejects_parallel_run(
    tmp_path: Path,
) -> None:
    first = RunStore(tmp_path / "runs")
    second = RunStore(tmp_path / "runs")
    checkpoint = new_checkpoint(
        command="import",
        safe_parameters={},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    )

    with first.execution_lock(checkpoint.run_id):
        first.save(checkpoint)
        with pytest.raises(RunLockedError):
            with second.execution_lock(checkpoint.run_id):
                pass


def test_publication_checkpoint_roundtrips_bounded_remote_intent(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs")
    publication = PublicationCheckpoint(
        mode="direct",
        remote="origin",
        base_branch="main",
        expected_old_base="a" * 40,
        candidate_sha="b" * 40,
        candidate_branch="mojilex/candidate/0123456789ab",
        phase="prepared",
        completed_source_indexes=(0, 2),
    )
    checkpoint = new_checkpoint(
        command="add",
        safe_parameters={"sources": ["https://t.me/addemoji/One"]},
        cli_version="0.1.0",
        schema_version="1.0.0",
        target_repository="MojiLex/mojilex",
        base_revision="a" * 40,
    ).model_copy(update={"publication": publication})

    store.save(checkpoint)

    assert store.load(checkpoint.run_id).publication == publication


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("remote", "https://token@github.com/MojiLex/mojilex"),
        ("base_branch", "../main"),
        ("candidate_branch", "main"),
        ("candidate_sha", "B" * 40),
        ("candidate_sha", "b" * 41),
        ("completed_source_indexes", (1, 0)),
        ("completed_source_indexes", (-1,)),
    ],
)
def test_publication_checkpoint_rejects_corrupt_but_typed_fields(field: str, value: object) -> None:
    values: dict[str, object] = {
        "mode": "direct",
        "remote": "origin",
        "base_branch": "main",
        "expected_old_base": "a" * 40,
        "candidate_sha": "b" * 40,
        "candidate_branch": "mojilex/candidate/0123456789ab",
        "phase": "prepared",
        "completed_source_indexes": (0,),
    }
    values[field] = value

    with pytest.raises(ValueError):
        PublicationCheckpoint.model_validate(values)


def test_pr_publication_checkpoint_rejects_direct_only_phase() -> None:
    with pytest.raises(ValueError, match="no direct-checks phase"):
        PublicationCheckpoint(
            mode="pr",
            remote="mojilex-fork",
            base_branch="main",
            expected_old_base="a" * 40,
            candidate_sha="b" * 40,
            candidate_branch="mojilex/batch/0123456789ab",
            phase="checks_passed",
        )
