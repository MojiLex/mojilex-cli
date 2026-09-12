"""Bounded human progress on stderr, including heartbeats during slow requests."""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from contextlib import suppress

from mojilex_cli.i18n import current_ui_language

from .runtime import report_progress


class BatchProgress:
    """Track actual completions; never present an elapsed timer as completed work."""

    def __init__(
        self, label: str, total: int, *, interval: float = 5.0, batch_total: int | None = None
    ) -> None:
        self.label = label
        self.total = total
        self.interval = interval
        self.completed = 0
        self.failed = 0
        self.active: dict[str, str] = {}
        self.active_counts: dict[str, int] = {}
        self.batch_total = batch_total
        self.completed_batches = 0
        self.queue_stopped = False
        self.started = time.monotonic()
        self.last_completion = self.started
        self.last_report = self.started
        self._heartbeat: asyncio.Task[None] | None = None

    async def __aenter__(self) -> BatchProgress:
        self._report()
        self._heartbeat = asyncio.create_task(self._tick())
        return self

    async def __aexit__(self, exc_type: object, *_: object) -> None:
        if self._heartbeat is not None:
            self._heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await self._heartbeat
        self._report(interrupted=exc_type is not None)

    def phase(self, key: str, phase: str, *, count: int | None = None) -> None:
        self.active[key] = phase
        self.active_counts[key] = count if count is not None else self.active_counts.get(key, 1)

    def stop_queue(self) -> None:
        self.queue_stopped = True

    def finish(self, key: str, *, count: int = 1, failed: bool = False) -> None:
        self.active.pop(key, None)
        self.active_counts.pop(key, None)
        if failed:
            self.failed += count
        else:
            self.completed += count
            self.completed_batches += 1
        self.last_completion = time.monotonic()
        if failed or self.completed == count or self.last_completion - self.last_report >= 1:
            self._report()

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(self.interval)
            self._report()

    def _report(self, *, interrupted: bool = False) -> None:
        now = time.monotonic()
        elapsed = int(now - self.started)
        idle = int(now - self.last_completion)
        phases = Counter(self.active.values())
        ru = current_ui_language() == "ru"
        labels = {
            "download": "скачивание" if ru else "downloading",
            "render": "обработка" if ru else "processing",
            "save": "сохранение" if ru else "saving",
            "ai": "ожидание AI" if ru else "waiting for AI",
            "approval": "проверка бюджета / подтверждение" if ru else "budget check / approval",
            "request": "ожидание ответа AI" if ru else "waiting for AI response",
            "retry": "повтор AI-запроса" if ru else "retrying AI request",
            "recovery": "повтор по одному эмодзи" if ru else "retrying individual emojis",
            "verify": "проверка" if ru else "verifying",
        }
        detail = ", ".join(f"{labels.get(key, key)}: {value}" for key, value in phases.items())
        percent = (100 * self.completed // self.total) if self.total else 100
        unit = (" эмодзи" if ru else " emojis") if self.batch_total is not None else ""
        text = (
            f"{self.label}: {self.completed}/{self.total}{unit} ({percent}%) | "
            f"{'ошибок' if ru else 'errors'}: {self.failed} | "
            f"{'прошло' if ru else 'elapsed'} {elapsed // 60:02d}:{elapsed % 60:02d}"
        )
        if self.batch_total is not None:
            pending = max(
                0, self.total - self.completed - self.failed - sum(self.active_counts.values())
            )
            waiting_label = (
                ("не начато" if ru else "not started")
                if self.queue_stopped
                else ("в очереди" if ru else "queued")
            )
            text += (
                f" | {'пачки' if ru else 'batches'}: {self.completed_batches}/{self.batch_total}"
                f" | {'пачек в работе' if ru else 'active batches'}: {len(self.active)}"
                f" | {waiting_label}: {pending}{unit}"
            )
        if detail:
            text += f" | {detail}"
        if idle >= self.interval and self.active:
            text += (
                f" | {'без завершений' if ru else 'no completions for'} "
                f"{idle}{' сек.' if ru else 's'}"
            )
        if interrupted:
            text += " | остановлено" if ru else " | stopped"
        report_progress(text)
        self.last_report = now
