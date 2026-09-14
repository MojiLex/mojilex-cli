"""Pack-level queue state shared across consecutive saved-run commands."""

# ruff: noqa: RUF001

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from rich.console import Console, ConsoleOptions, RenderResult
from rich.text import Text

from mojilex_cli.i18n import current_ui_language

PACK: ContextVar[str | None] = ContextVar("mojilex_progress_pack", default=None)
SHARED: ContextVar[PackQueue | None] = ContextVar("mojilex_pack_queue", default=None)


@dataclass
class PackQueue:
    stages: dict[str, str] = field(default_factory=dict)
    batches: dict[object, str] = field(default_factory=dict)

    def register(self, sources: Sequence[str]) -> None:
        for source in sources:
            self.stages.setdefault(source, "waiting")

    def groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {
            key: []
            for key in (
                "download",
                "render",
                "ai",
                "waiting",
                "ai_wait",
                "finalize",
                "ready",
                "failed",
            )
        }
        for source, stage in self.stages.items():
            categories = set()
            for batch, owner in self.batches.items():
                if owner != source:
                    continue
                for phase in getattr(batch, "active", {}).values():
                    categories.add(
                        "download"
                        if phase == "download"
                        else "render"
                        if phase in {"render", "save", "verify", "media_retry"}
                        else "ai"
                    )
            if not categories:
                categories.add(stage)
            for category in categories:
                groups[category].append(source)
        return groups

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        ru = current_ui_language() == "ru"
        labels = dict(
            zip(
                ("download", "render", "ai", "waiting", "ai_wait", "finalize", "ready", "failed"),
                (
                    "Скачивание",
                    "Обработка на ПК",
                    "ИИ",
                    "Ожидают очереди",
                    "Готовы к ИИ, ждут очереди",
                    "Финальная проверка и сохранение",
                    "Готовы к GitHub",
                    "Остановлены / ошибки",
                )
                if ru
                else (
                    "Downloading",
                    "Processing on PC",
                    "AI",
                    "Waiting",
                    "Prepared, waiting for AI",
                    "Final validation and saving",
                    "Ready to send to GitHub",
                    "Stopped / errors",
                ),
                strict=True,
            )
        )
        groups = self.groups()
        yield Text(
            ("Всего паков: " if ru else "Total packs: ") + str(len(self.stages)), style="bold"
        )
        # Reserve room for summaries and a prompt; distribute detail rows fairly.
        detail_slots = max(0, console.size.height - 15)
        active = [
            key
            for key in ("download", "render", "ai", "ai_wait", "finalize", "failed")
            if groups[key]
        ]
        per_group = detail_slots // max(1, len(active))
        for key, values in groups.items():
            yield Text(
                f"{labels[key]}: {len(values)}",
                style="green" if key == "ready" else "cyan",
                no_wrap=True,
                overflow="ellipsis",
            )
            if key not in active:
                continue
            shown = values[:per_group]
            for source in shown:
                name = source.rstrip("/").rsplit("/", 1)[-1]
                batches = [batch for batch, owner in self.batches.items() if owner == source]
                suffix = ""
                media = [batch for batch in batches if getattr(batch, "unit", "") == "media"]
                if key in {"download", "render"} and media:
                    batch = max(media, key=lambda item: getattr(item, "total", 0))
                    total = getattr(batch, "total", 0)
                    cached = getattr(batch, "cached", 0)
                    if key == "download":
                        received = len(getattr(batch, "downloaded", set()))
                        suffix = f" · {received}/{total - cached}"
                        if cached:
                            suffix += f" · {'из кэша' if ru else 'cached'}: {cached}"
                    else:
                        suffix = f" · {getattr(batch, 'completed', 0)}/{total}"
                elif batches:
                    batch = max(batches, key=lambda item: getattr(item, "total", 0))
                    if getattr(batch, "unit", "") == "backends":
                        suffix = " · проверка кэша" if ru else " · checking cache"
                    else:
                        suffix = f" · {getattr(batch, 'completed', 0)}/{getattr(batch, 'total', 0)}"
                    if "approval" in getattr(batch, "active", {}).values():
                        suffix += " · требуется ответ" if ru else " · awaiting confirmation"
                elif key == "download":
                    suffix = " · получение списка файлов" if ru else " · fetching file list"
                if source == shown[-1] and len(values) > len(shown):
                    suffix += f" (+{len(values) - len(shown)})"
                yield Text(
                    f"  {name} — {labels[key].lower()}{suffix}", no_wrap=True, overflow="ellipsis"
                )
        yield Text(
            "Один пак может одновременно скачиваться и обрабатываться."
            if ru
            else "A pack may download and decode simultaneously.",
            style="dim",
            no_wrap=True,
            overflow="ellipsis",
        )


@contextmanager
def pack_queue_scope() -> Iterator[None]:
    token = SHARED.set(SHARED.get() or PackQueue())
    try:
        yield
    finally:
        SHARED.reset(token)
