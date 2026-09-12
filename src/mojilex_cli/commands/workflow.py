"""Public workflow command adapters kept thin for testability."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import TextIO, cast

from mojilex_cli.cache import CacheStore
from mojilex_cli.config import load_config
from mojilex_cli.dataset import load_dataset
from mojilex_cli.pipeline.runner import (
    PipelineOptions,
    repository_workspace,
    run_add,
    run_describe,
    run_import,
    run_resume_sync,
    run_submit,
)

from .runtime import CommandError, CommandResult

_MAX_SOURCE_FILE_BYTES = 1024 * 1024


def collect_sources(
    positional: Sequence[str],
    *,
    from_file: Path | None,
    use_stdin: bool,
    stream: TextIO,
) -> tuple[str, ...]:
    values = list(positional)
    if from_file is not None:
        path = from_file.expanduser().resolve()
        if not path.is_file() or path.is_symlink():
            raise CommandError(
                "CONFIG_INVALID",
                "--from-file must be a regular non-symlink UTF-8 file.",
                hint="Pass a safe text file with one source per line.",
            )
        if path.stat().st_size > _MAX_SOURCE_FILE_BYTES:
            raise CommandError(
                "CONFIG_INVALID",
                "Source list exceeds the 1 MiB safety limit.",
                hint="Split the import into smaller batches.",
            )
        try:
            values.extend(_source_lines(path.read_text(encoding="utf-8")))
        except UnicodeError as exc:
            raise CommandError(
                "CONFIG_INVALID",
                "Source list is not valid UTF-8.",
                hint="Save the file as UTF-8 and retry.",
            ) from exc
    if use_stdin:
        payload = stream.read(_MAX_SOURCE_FILE_BYTES + 1)
        if len(payload.encode("utf-8")) > _MAX_SOURCE_FILE_BYTES:
            raise CommandError(
                "CONFIG_INVALID",
                "Standard-input source list exceeds 1 MiB.",
                hint="Split the import into smaller batches.",
            )
        values.extend(_source_lines(payload))
    normalized: list[str] = []
    for value in values:
        source = value.strip()
        if not source or source.startswith("#"):
            continue
        if source != value and value in positional:
            raise CommandError(
                "SOURCE_UNSUPPORTED",
                "A positional source contains leading or trailing whitespace.",
                hint="Pass the exact canonical source URL.",
            )
        if any(ord(character) < 32 or ord(character) == 127 for character in source):
            raise CommandError(
                "SOURCE_UNSUPPORTED",
                "A source contains control characters.",
                hint="Pass one clean source URL per line.",
            )
        if source not in normalized:
            normalized.append(source)
    if not normalized:
        raise CommandError(
            "CONFIG_MISSING",
            "No sources were provided.",
            hint="Pass URLs positionally, with --from-file, or with --stdin.",
        )
    return tuple(normalized)


def add_command(
    sources: Sequence[str],
    *,
    repo: str | None,
    platform: str,
    provider: str | None,
    model: str | None,
    languages: tuple[str, ...],
    publish: str | None,
    direct_push: bool,
    base: str | None,
    redescribe: str,
    overwrite_reviewed: bool,
    new_identity: bool,
    same_identity: bool,
    max_items: int | None,
    max_ai_requests: int | None,
    max_cost_usd: Decimal | None,
    allow_unknown_cost: bool,
    ai_concurrency: int | None,
    download_concurrency: int | None,
    dedupe: str | None,
    max_dedupe_candidates: int | None,
    dedupe_profile: str | None,
    model_routing: str | None,
    escalation_model: str | None,
    dry_run: bool,
    check_media: bool,
    fail_fast: bool,
    confirmation: Callable[[str], bool] | None = None,
    unknown_cost_confirmation: Callable[[int], bool] | None = None,
) -> CommandResult:
    return run_add(
        sources,
        PipelineOptions(
            repository=repo,
            platform=platform,
            provider=provider,
            model=model,
            languages=languages,
            publish=publish,
            direct_push=direct_push,
            base=base,
            redescribe=redescribe,
            overwrite_reviewed=overwrite_reviewed,
            new_identity=new_identity,
            same_identity=same_identity,
            max_items=max_items,
            max_ai_requests=max_ai_requests,
            max_cost_usd=max_cost_usd,
            allow_unknown_cost=allow_unknown_cost,
            ai_concurrency=ai_concurrency,
            download_concurrency=download_concurrency,
            dedupe=dedupe,
            max_dedupe_candidates=max_dedupe_candidates,
            dedupe_profile=dedupe_profile,
            model_routing=model_routing,
            escalation_model=escalation_model,
            dry_run=dry_run,
            check_media=check_media,
            fail_fast=fail_fast,
            confirmation=confirmation,
            unknown_cost_confirmation=unknown_cost_confirmation,
        ),
    )


def import_command(
    sources: Sequence[str],
    *,
    repo: str | None,
    platform: str,
    max_items: int | None,
    download_concurrency: int | None,
    check_media: bool,
    fail_fast: bool,
) -> CommandResult:
    del check_media  # import always verifies bytes; the option remains explicit in the CLI contract
    return run_import(
        sources,
        PipelineOptions(
            repository=repo,
            platform=platform,
            max_items=max_items,
            download_concurrency=download_concurrency,
            check_media=True,
            fail_fast=fail_fast,
            publish="local",
        ),
    )


def describe_command(
    selectors: Sequence[str],
    *,
    provider: str | None,
    model: str | None,
    max_ai_requests: int | None,
    max_cost_usd: Decimal | None,
    allow_unknown_cost: bool,
    ai_concurrency: int | None = None,
    unknown_cost_confirmation: Callable[[int], bool] | None = None,
) -> CommandResult:
    if len(selectors) == 1 and selectors[0].startswith("mlxrun_"):
        return run_describe(
            selectors[0],
            PipelineOptions(
                provider=provider,
                model=model,
                ai_concurrency=ai_concurrency,
                max_ai_requests=max_ai_requests,
                max_cost_usd=max_cost_usd,
                allow_unknown_cost=allow_unknown_cost,
                unknown_cost_confirmation=unknown_cost_confirmation,
            ),
        )
    config = load_config(
        cli={
            "ai": {
                "provider": provider,
                "model": model,
                "ai_concurrency": ai_concurrency,
                "max_ai_requests": max_ai_requests,
                "max_cost_usd": max_cost_usd,
                "allow_unknown_cost": allow_unknown_cost or None,
            }
        }
    )
    sources = _sources_for_selectors(
        selectors,
        config.repository.target,
        base_branch=config.repository.base_branch,
        cache_dir=config.cache_dir,
    )
    return run_add(
        sources,
        PipelineOptions(
            repository=config.repository.target,
            provider=config.ai.provider,
            model=config.ai.model,
            ai_concurrency=config.ai.ai_concurrency,
            max_ai_requests=config.ai.max_ai_requests,
            max_cost_usd=config.ai.max_cost_usd,
            allow_unknown_cost=config.ai.allow_unknown_cost,
            unknown_cost_confirmation=unknown_cost_confirmation,
            redescribe="all",
            publish="local",
        ),
    )


def update_command(
    selector: str | None,
    *,
    all_collections: bool,
    repo: str | None,
    dry_run: bool,
) -> CommandResult:
    if (selector is None) == (not all_collections):
        raise CommandError(
            "CONFIG_INVALID",
            "Pass exactly one source/collection selector or --all.",
            hint="Use `mojilex update COLLECTION_ID` or `mojilex update --all`.",
        )
    config = load_config(cli={"repository": {"target": repo}})
    if all_collections:
        sources = _sources_for_selectors(
            (),
            config.repository.target,
            base_branch=config.repository.base_branch,
            select_all=True,
            cache_dir=config.cache_dir,
            read_only=dry_run,
        )
    elif selector is not None and "://" in selector:
        sources = (selector,)
    else:
        sources = _sources_for_selectors(
            (cast(str, selector),),
            config.repository.target,
            base_branch=config.repository.base_branch,
            cache_dir=config.cache_dir,
            read_only=dry_run,
        )
    return run_add(
        sources,
        PipelineOptions(
            repository=config.repository.target,
            dry_run=dry_run,
            check_media=dry_run,
            explicit_verification=True,
        ),
    )


def submit_command(
    target: str | None,
    *,
    repo: str | None,
    publish: str | None = None,
    direct_push: bool,
    base: str | None,
    confirmation: Callable[[str], bool] | None = None,
) -> CommandResult:
    return run_submit(
        target,
        repository=repo,
        publish=publish,
        direct_push=direct_push,
        base=base,
        confirmation=confirmation,
    )


def resume_command(
    run_id: str,
    *,
    ai_concurrency: int | None = None,
    confirmation: Callable[[str], bool] | None = None,
    unknown_cost_confirmation: Callable[[int], bool] | None = None,
) -> CommandResult:
    return run_resume_sync(
        run_id,
        confirmation=confirmation,
        unknown_cost_confirmation=unknown_cost_confirmation,
        **({"ai_concurrency": ai_concurrency} if ai_concurrency is not None else {}),
    )


def cache_info_command() -> CommandResult:
    config = load_config()
    path = cast(Path, config.cache_dir) / "cache-v1.sqlite3"
    if not path.exists():
        return CommandResult(
            result={
                "path": str(path),
                "metadata_entries": 0,
                "ai_entries": 0,
                "database_bytes": 0,
            }
        )
    with CacheStore(path, read_only=True) as cache:
        return CommandResult(result={"path": str(path), **cache.info()})


def cache_prune_command(older_than_days: int) -> CommandResult:
    config = load_config()
    path = cast(Path, config.cache_dir) / "cache-v1.sqlite3"
    if not path.exists():
        return CommandResult(status="noop", result={"path": str(path)})  # type: ignore[arg-type]
    cutoff = int(time.time()) - older_than_days * 86_400
    with CacheStore(path) as cache:
        removed = cache.prune(older_than_epoch=cutoff)
    return CommandResult(result={"path": str(path), **removed})


def _sources_for_selectors(
    selectors: Sequence[str],
    repository: str,
    *,
    base_branch: str,
    select_all: bool = False,
    cache_dir: Path | None = None,
    read_only: bool = False,
) -> tuple[str, ...]:
    with repository_workspace(repository, base_branch) as workspace:
        snapshot = None
        if cache_dir is not None and not select_all:
            from mojilex_cli.cache.lookup import lookup_authoring_snapshot

            snapshot = lookup_authoring_snapshot(
                workspace.root,
                cache_dir / "lookup-v1.sqlite3",
                selectors,
                read_only=read_only,
            )
        if snapshot is None:
            snapshot = load_dataset(workspace.root)
        if select_all:
            return tuple(
                sorted(
                    collection.canonical_url
                    for collection in snapshot.collections.values()
                    if collection.canonical_url
                )
            )
        collection_ids: set[str] = set()
        for selector in selectors:
            if selector in snapshot.collections:
                collection_ids.add(selector)
                continue
            if selector in snapshot.emojis:
                collection_ids.update(
                    membership.collection_id
                    for membership in snapshot.memberships.values()
                    if membership.emoji_id == selector
                )
                continue
            matches = [
                collection.id
                for collection in snapshot.collections.values()
                if collection.native_id == selector or collection.canonical_url == selector
            ]
            if len(matches) != 1:
                raise CommandError(
                    "SOURCE_NOT_FOUND",
                    f"Selector did not resolve uniquely: {selector}",
                    hint="Pass a collection ID, emoji ID, native short name, or source URL.",
                )
            collection_ids.add(matches[0])
        urls = [snapshot.collections[value].canonical_url for value in collection_ids]
        if not urls or any(value is None for value in urls):
            raise CommandError(
                "SOURCE_NOT_FOUND",
                "No resolvable source collection was selected.",
                hint="Check the selector and dataset provenance.",
            )
        return tuple(sorted(cast(list[str], urls)))


def _source_lines(value: str) -> list[str]:
    return [
        line.strip()
        for line in value.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
