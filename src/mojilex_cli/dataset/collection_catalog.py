"""Deterministic, human-readable index of Telegram collections."""

from __future__ import annotations

import html
from collections.abc import Iterable

from mojilex_cli.domain.models import Collection


def _cell(value: object) -> str:
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
        f"Паков / Packs: {len(rows)}",
        "",
        "| Название / Title | Имя Telegram / Telegram name | Эмодзи / Emojis |",
        "|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {_cell(row.title)} | [{_cell(row.native_id)}]({row.id}/) | {row.item_count} |"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")
