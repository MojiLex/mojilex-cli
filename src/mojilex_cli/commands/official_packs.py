"""One batch-wide decision before importing links already published officially."""

# ruff: noqa: RUF001 -- intentional Russian interface strings.

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from mojilex_cli.config import load_config
from mojilex_cli.dataset import load_dataset, validate_dataset
from mojilex_cli.i18n import current_ui_language
from mojilex_cli.output import RunStatus
from mojilex_cli.pipeline.runner import repository_workspace
from mojilex_cli.sources.telegram import parse_telegram_source

from .runtime import CommandError, CommandResult, report_progress


@dataclass(frozen=True)
class SourceSelection:
    selected: tuple[str, ...]
    skipped: tuple[str, ...] = ()
    approved_sources: tuple[str, ...] = ()

    def annotate(self, result: CommandResult) -> CommandResult:
        if self.skipped:
            result.result["official_packs_skipped"] = list(self.skipped)
        return result

    def empty_result(self) -> CommandResult:
        return CommandResult(
            status=RunStatus.NOOP,
            result={
                "official_packs_skipped": list(self.skipped),
                "message": "All selected packs are already in the official repository; skipped.",
            },
        )


def official_pack_names() -> frozenset[str]:
    # The official main branch, never a user's local draft, fork, or pending PR.
    # One bounded checkout per input batch, not one GitHub request per URL.
    with repository_workspace("MojiLex/mojilex", "main") as workspace:
        validate_dataset(workspace.root, strict=True).raise_for_errors()
        snapshot = load_dataset(workspace.root)
        return frozenset(
            collection.native_id.casefold()
            for collection in snapshot.collections.values()
            if collection.platform == "telegram"
            and collection.native_namespace == "sticker_set.name"
            and collection.scope_id == "global"
        )


def select_sources(
    sources: Sequence[str],
    *,
    platform: str,
    policy: str | None = None,
    confirmation: Callable[[str], bool] | None = None,
    approved_sources: Sequence[str] = (),
) -> SourceSelection:
    selected_policy = (
        policy if policy is not None else load_config().processing.official_pack_policy
    )
    if selected_policy not in {"ask", "skip", "allow"}:
        raise CommandError(
            "CONFIG_INVALID",
            "Unknown official pack policy.",
            hint="Choose ask, skip, or allow with --official-packs or settings.",
        )
    references = tuple(
        parse_telegram_source(source, allow_bare=platform == "telegram") for source in sources
    )
    if selected_policy == "allow":
        return SourceSelection(tuple(sources), approved_sources=tuple(sources))
    approved = set(approved_sources) if selected_policy == "ask" else set()
    receipts = tuple(source for source in sources if source in approved)
    if len(receipts) == len(sources):
        return SourceSelection(tuple(sources), approved_sources=receipts)
    known = official_pack_names()
    existing = tuple(
        source
        for source, ref in zip(sources, references, strict=True)
        if ref.native_id.casefold() in known and source not in approved
    )
    if not existing:
        return SourceSelection(tuple(sources), approved_sources=receipts)
    if selected_policy == "ask" and confirmation is not None:
        names = [
            ref.native_id
            for source, ref in zip(sources, references, strict=True)
            if source in existing
        ]
        listing = ", ".join(names)
        prompt = (
            f"В официальном репозитории уже есть {len(existing)} паков: {listing}. "
            "Разрешить их повторную обработку? Enter — Нет"
            if current_ui_language() == "ru"
            else f"The official repository already contains {len(existing)} packs: {listing}. "
            "Allow processing them again? Enter means No"
        )
        if confirmation(prompt):
            return SourceSelection(
                tuple(sources),
                approved_sources=tuple(
                    source for source in sources if source in approved or source in existing
                ),
            )
    remaining = tuple(source for source in sources if source not in existing)
    report_progress(
        f"Пропущено паков из официального репозитория: {len(existing)}. "
        f"Осталось для обработки: {len(remaining)}."
        if current_ui_language() == "ru"
        else f"Skipped official packs: {len(existing)}. Remaining: {len(remaining)}."
    )
    return SourceSelection(remaining, existing, receipts)
