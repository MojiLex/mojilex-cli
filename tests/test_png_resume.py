from pathlib import Path

from mojilex_cli.analysis import decoder_backend_fingerprint
from mojilex_cli.cache import CacheStore
from mojilex_cli.media import MediaProcessor, TemporaryMediaRun
from mojilex_cli.pipeline.runner import (
    _cache_deterministic_analysis,
    _deterministic_key,
    _restore_deterministic_cache_entry,
    _source_descriptor_sha256,
)
from mojilex_cli.runs import ElementCheckpoint
from test_dataset_helpers import write_fixture
from test_pipeline_resume_cache import _item, _processed


def test_original_png_cache_restores_under_telegram_static_expectation(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    source = _item("png", unique_id="original-png", file_id="png-file")
    assert source.media_format == "webp"
    base = _processed(snapshot)
    value = base.model_copy(
        update={
            "metadata": base.metadata.model_copy(
                update={"format": "png", "mime_type": "image/png"}
            ),
            "analysis": base.analysis.model_copy(
                update={"decoder_backend_fingerprint": decoder_backend_fingerprint("png")}
            ),
        }
    )
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=snapshot.root)
    try:
        _cache_deterministic_analysis(cache, source, value)
        checkpoint = ElementCheckpoint(
            stage="fingerprint_ready",
            source_descriptor_sha256=_source_descriptor_sha256(source),
            media_sha256=(value.metadata.sha256,),
            deterministic_cache_key=_deterministic_key(value),
            palette_complete=True,
            fingerprint_complete=True,
        )
        with TemporaryMediaRun(root=tmp_path) as temporary:
            restored = _restore_deterministic_cache_entry(
                cache, source, MediaProcessor(temporary), checkpoint
            )
        assert restored is not None
        assert restored.metadata == value.metadata
        assert restored.analysis == value.analysis
    finally:
        cache.close()
