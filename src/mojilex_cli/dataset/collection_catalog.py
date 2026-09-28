"""Deterministic, human-readable index of Telegram collections."""

from __future__ import annotations

import hashlib
import html
from collections.abc import Iterable, Mapping

from mojilex_cli.domain.models import Collection, Emoji, Membership

_MARKDOWN_ESCAPES = frozenset(r"\`*_[]!|~$")


def _cell(value: object) -> str:
    plain = html.escape(" ".join(str(value).split()), quote=False)
    return "".join(f"\\{char}" if char in _MARKDOWN_ESCAPES else char for char in plain)


def _legacy_cell(value: object) -> str:
    return html.escape(str(value).replace("\r", " ").replace("\n", " "), quote=False).replace(
        "|", "\\|"
    )


def render_catalog(collections: Iterable[Collection]) -> bytes:
    rows = sorted(
        collections,
        key=lambda row: (row.title.casefold(), row.native_id.casefold(), row.id),
    )
    lines = [
        "# Каталог паков Telegram / Telegram pack catalog",
        "",
        "Названия берутся из карточек паков. Каталог не является источником данных.",
        "Titles come from `collection.json`; the records remain the source of truth.",
        "",
        f"Паков / Packs: {len(rows)}<br>",
        f"Эмодзи в паках / Emojis across packs: {sum(row.item_count for row in rows)}",
        "",
        "| Название / Title | Имя Telegram / Telegram name | Эмодзи / Emojis |",
        "|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {_cell(row.title)} | "
            f"[{_cell(row.native_id)}]({row.id}/README.md) | {row.item_count} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_legacy_catalog(collections: Iterable[Collection]) -> bytes:
    """The pre-navigation index, accepted only for unchanged saved datasets."""
    rows = sorted(
        collections,
        key=lambda row: (row.title.casefold(), row.native_id.casefold(), row.id),
    )
    lines = [
        "# Каталог паков Telegram / Telegram pack catalog",
        "",
        "Названия берутся из карточек паков. Каталог не является источником данных.",
        "Titles come from `collection.json`; the records remain the source of truth.",
        "",
        f"Паков / Packs: {len(rows)}<br>",
        f"Эмодзи в паках / Emojis across packs: {sum(row.item_count for row in rows)}",
        "",
        "| Название / Title | Имя Telegram / Telegram name | Эмодзи / Emojis |",
        "|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {_legacy_cell(row.title)} | "
            f"[{_legacy_cell(row.native_id)}]({row.id}/) | {row.item_count} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def render_pack_page(
    collection: Collection,
    memberships: Iterable[Membership],
    emojis_by_id: Mapping[str, Emoji],
) -> bytes:
    """Render the same navigable pack index as the data repository generator."""
    rows = sorted(
        memberships,
        key=lambda row: (
            row.status != "active",
            row.position,
            row.status,
            row.emoji_id,
            row.id,
        ),
    )
    lines = [
        f"# {_cell(collection.title)}",
        "",
        "[Каталог паков / Pack catalog](../README.md)",
        "",
        f"Пак Telegram / Telegram pack: {_cell(collection.native_id)}",
        "",
        "Описания — навигационный указатель; канонические данные находятся в "
        "[collection.json](collection.json), [memberships.jsonl](memberships.jsonl) "
        "и записях эмодзи.",
        "Descriptions are a navigation aid; canonical records are the linked JSON/JSONL files.",
        "",
        "| Позиция / Position | Статус / Status | Telegram ID | Emoji ID | "
        "Описание (RU) | Description (EN) |",
        "|---:|---|---|---|---|---|",
    ]
    for row in rows:
        emoji_id = row.emoji_id
        emoji = emojis_by_id[emoji_id]
        digest = hashlib.sha256(emoji_id.encode("utf-8")).hexdigest()
        lines.append(
            f"| {row.position} | {_cell(row.status)} | {_cell(emoji.native_id)} | "
            f"[{_cell(emoji_id)}](../../emojis/{digest}.jsonl) | "
            f"{_cell(emoji.descriptions['ru'].text)} | "
            f"{_cell(emoji.descriptions['en'].text)} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")
