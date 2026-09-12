"""Bounded local snapshot discovery; discovery never establishes release trust."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from mojilex_cli.commands.runtime import CommandError, machine_output_requested
from mojilex_cli.config import load_config
from mojilex_cli.dataset.layout import assert_no_link_or_reparse
from mojilex_cli.i18n import current_ui_language
from mojilex_cli.read.snapshot import MAX_MANIFEST_BYTES, parse_bounded_json

_SNAPSHOT_ID = re.compile(r"data-[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]+\Z")


def local_text(english: str, russian: str) -> str:
    return russian if current_ui_language() == "ru" and not machine_output_requested() else english


@dataclass(frozen=True)
class LocalSnapshot:
    path: Path
    snapshot_id: str
    manifest_sha256: str


def discovery_roots() -> tuple[Path, ...]:
    roots = [Path.cwd()]
    target = Path(load_config().repository.target).expanduser()
    if target.is_dir():
        roots.append(target)
    return tuple(dict.fromkeys(path.resolve() for path in roots))


def discover_local_snapshots() -> tuple[LocalSnapshot, ...]:
    explicit = os.environ.get("MOJILEX_SNAPSHOT")
    candidates: list[Path] = [Path(explicit).expanduser()] if explicit else []
    for root in () if explicit else discovery_roots():
        candidates.append(root)
        for name in ("dist", "snapshots", "releases"):
            folder = root / name
            candidates.append(folder)
            try:
                assert_no_link_or_reparse(folder)
                if folder.is_dir():
                    children = list(islice(folder.iterdir(), 257))
                    if len(children) > 256:
                        raise CommandError(
                            "RESOURCE_LIMIT_EXCEEDED",
                            local_text(
                                "Too many local snapshot candidates.",
                                "Слишком много локальных снимков.",
                            ),
                            hint=local_text(
                                "Pass --snapshot PATH explicitly.",
                                "Укажите нужный путь через --snapshot ПУТЬ.",
                            ),
                        )
                    candidates.extend(path for path in children if path.is_dir())
            except (OSError, ValueError):
                continue
    found: dict[Path, LocalSnapshot] = {}
    for path in candidates:
        manifest = path if path.name == "manifest.json" else path / "manifest.json"
        try:
            assert_no_link_or_reparse(manifest)
            if not manifest.is_file() or manifest.stat().st_size > MAX_MANIFEST_BYTES:
                continue
            raw = manifest.read_bytes()
            document = parse_bounded_json(raw, source="local snapshot manifest")
            identifier = document.get("snapshot_id")
            if (
                document.get("dataset") != "mojilex"
                or not isinstance(identifier, str)
                or not _SNAPSHOT_ID.fullmatch(identifier)
            ):
                continue
            root = manifest.parent.resolve()
            found[root] = LocalSnapshot(root, identifier, hashlib.sha256(raw).hexdigest())
        except (OSError, ValueError, CommandError):
            continue
    return tuple(sorted(found.values(), key=lambda value: (value.snapshot_id, str(value.path))))


def select_local_snapshot(selected: Path | None) -> Path:
    if selected is not None:
        return selected
    explicit = os.environ.get("MOJILEX_SNAPSHOT")
    if explicit:
        return Path(explicit).expanduser()
    candidates = discover_local_snapshots()
    if len(candidates) == 1:
        return candidates[0].path
    if not candidates:
        raise CommandError(
            "SNAPSHOT_NOT_FOUND",
            local_text(
                "No local release snapshot was found.", "Локальный снимок релиза не найден."
            ),
            hint=local_text(
                "Run mojilex snapshots; pass an existing release directory with --snapshot PATH. "
                "Import and AI drafts are not release snapshots; no remote catalog is configured.",
                "Выполните mojilex snapshots и укажите готовый снимок через --snapshot ПУТЬ. "
                "Импорт и AI-черновик ещё не являются снимком релиза; "
                "удалённый каталог не настроен.",
            ),
        )
    raise CommandError(
        "OPTION_CONFLICT",
        local_text(
            "Several local snapshots were found; select one.",
            "Найдено несколько локальных снимков; выберите один.",
        ),
        hint=local_text(
            "Run mojilex snapshots, then pass --snapshot PATH or set MOJILEX_SNAPSHOT.",
            "Посмотрите пути командой mojilex snapshots и укажите --snapshot ПУТЬ "
            "или переменную MOJILEX_SNAPSHOT.",
        ),
    )
