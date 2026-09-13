"""High-level stream-to-safe-frames media orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from mojilex_cli.concurrency import current_batch_limits

from .models import MediaError, ProcessedMedia
from .sandbox import SafeMediaWorker
from .temporary import TemporaryMediaRun


class SourceChangedDuringRunError(MediaError):
    """Downloaded media no longer matches the hash captured by the run."""

    code = "SOURCE_CHANGED_DURING_RUN"
    retryable = False


class MediaProcessor:
    def __init__(
        self,
        run: TemporaryMediaRun,
        worker: SafeMediaWorker | None = None,
        *,
        render_concurrency: int = 2,
    ) -> None:
        if type(render_concurrency) is not int or not 1 <= render_concurrency <= 8:
            raise ValueError("render concurrency must be between 1 and 8")
        self.run = run
        self.worker = worker or SafeMediaWorker(run.limits)
        # Downloads can overlap freely within the pipeline's network limit. Only
        # isolated decoder processes occupy these slots, so queued work does not
        # consume its worker wall-time budget or spawn extra Python processes.
        batch = current_batch_limits()
        self._render_slots = (
            batch.render_slots if batch is not None else asyncio.Semaphore(render_concurrency)
        )

    async def verify_stream(
        self,
        chunks: AsyncIterator[bytes],
        *,
        declared_size: int | None = None,
        expected_sha256: str,
    ) -> tuple[str, int]:
        """Boundedly download and hash media without invoking any decoder."""

        source, digest, size = await self.run.write_stream(chunks, expected_size=declared_size)
        try:
            if digest != expected_sha256:
                raise SourceChangedDuringRunError(
                    "downloaded media SHA-256 no longer matches the staging run"
                )
            return digest, size
        finally:
            # The raw media was needed only for the streaming digest. Accounting
            # remains conservative for the lifetime of this private temp run.
            source.unlink(missing_ok=True)

    async def process_stream(
        self,
        chunks: AsyncIterator[bytes],
        *,
        expected_format: str,
        declared_size: int | None = None,
        expected_sha256: str | None = None,
        needs_repainting: bool = False,
    ) -> ProcessedMedia:
        source, digest, _ = await self.run.write_stream(chunks, expected_size=declared_size)
        if expected_sha256 is not None and digest != expected_sha256:
            source.unlink(missing_ok=True)
            raise SourceChangedDuringRunError(
                "downloaded media SHA-256 no longer matches the staging run"
            )
        return await self._process_source(
            source,
            expected_format=expected_format,
            needs_repainting=needs_repainting,
        )

    async def process_stream_reusing_analysis(
        self,
        chunks: AsyncIterator[bytes],
        *,
        expected_format: str,
        declared_size: int | None,
        cached: ProcessedMedia,
        needs_repainting: bool = False,
    ) -> ProcessedMedia:
        """Recreate transient frames while retaining exact verified deterministic analysis."""

        if cached.analysis is None:
            raise ValueError("cached media has no deterministic analysis")
        source, digest, size = await self.run.write_stream(chunks, expected_size=declared_size)
        if digest != cached.metadata.sha256 or size != cached.metadata.byte_size:
            source.unlink(missing_ok=True)
            raise SourceChangedDuringRunError(
                "downloaded media no longer matches the deterministic cache"
            )
        processed = await self._process_source(
            source,
            expected_format=expected_format,
            needs_repainting=needs_repainting,
            cached=cached,
        )
        if (
            processed.metadata != cached.metadata
            or processed.analysis != cached.analysis
            or processed.semantic_frame_count != cached.semantic_frame_count
            or processed.semantic_has_dark_render is not cached.semantic_has_dark_render
        ):
            raise SourceChangedDuringRunError(
                "re-rendered media differs from the deterministic cache context"
            )
        return processed

    async def _process_source(
        self,
        source: Path,
        *,
        expected_format: str,
        needs_repainting: bool,
        cached: ProcessedMedia | None = None,
    ) -> ProcessedMedia:
        async with self._render_slots:
            return await self._render_source(
                source,
                expected_format=expected_format,
                needs_repainting=needs_repainting,
                cached=cached,
            )

    async def _render_source(
        self,
        source: Path,
        *,
        expected_format: str,
        needs_repainting: bool,
        cached: ProcessedMedia | None,
    ) -> ProcessedMedia:
        worker_task = asyncio.create_task(
            asyncio.to_thread(
                self.worker.process,
                source,
                self.run.output_dir(),
                expected_format=expected_format,
                needs_repainting=needs_repainting,
                cached_analysis=cached.analysis if cached is not None else None,
                expected_dark_render=(
                    cached.semantic_has_dark_render if cached is not None else None
                ),
            )
        )
        try:
            # Shield prevents cancellation from abandoning a decoder thread that still
            # owns files below the run directory. On cancellation we reap it below.
            processed = await asyncio.shield(worker_task)
        except BaseException:
            # Keep the slot and private files until the actual worker is reaped,
            # even when the caller receives another cancellation while waiting.
            while not worker_task.done():
                try:
                    await asyncio.shield(worker_task)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if worker_task.done() and not worker_task.cancelled():
                worker_task.exception()
            raise
        tile_paths = (
            (processed.composition_tile_path,)
            if processed.composition_tile_path is not None
            else ()
        )
        self.run.account_outputs((*processed.frame_paths, *processed.dark_frame_paths, *tile_paths))
        return processed
