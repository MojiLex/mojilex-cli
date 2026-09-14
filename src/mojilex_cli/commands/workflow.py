"""Public workflow command adapters kept thin for testability."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Literal, TextIO, TypedDict, cast

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


class _ResumeOverrides(TypedDict, total=False):
    selected_sources: tuple[str, ...]
    ai_concurrency: int
    download_concurrency: int
    max_ai_requests: int | Literal["unlimited"]
    official_pack_policy: str
    official_confirmation: Callable[[str], bool]


def collect_sources(
    positional: Sequence[str],
    *,
    from_file: Path | None,
    use_stdin: bool,
    stream: TextIO,
) -> tuple[str, ...]:
    values = list(positional)
    if from_file is None and len(values) == 1 and "://" not in values[0]:
        candidate = Path(values[0].strip().strip('"')).expanduser()
        if candidate.is_file() or candidate.suffix.lower() == ".txt" or "\\" in str(candidate):
            from_file = candidate
            values = []
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
            values.extend(_source_lines(path.read_text(encoding="utf-8-sig")))
        except UnicodeError as exc:
            raise CommandError(
                "CONFIG_INVALID",
                "Source list is not valid UTF-8.",
                hint="Save the file as UTF-8 and retry.",
            ) from exc
        except OSError as exc:
            raise CommandError(
                "CONFIG_INVALID",
                "Cannot read the source list file.",
                hint="Check the path and file permissions.",
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
    max_ai_requests: int | Literal["unlimited"] | None,
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
    unknown_cost_confirmation: Callable[[int | None], bool] | None = None,
    official_pack_policy: str | None = None,
    official_confirmation: Callable[[str], bool] | None = None,
) -> CommandResult:
    from .official_packs import select_sources

    selection = select_sources(
        sources,
        platform=platform,
        policy=official_pack_policy,
        confirmation=official_confirmation,
    )
    if not selection.selected:
        return selection.empty_result()
    result = run_add(
        selection.selected,
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
            official_approved_sources=selection.approved_sources,
        ),
    )
    return selection.annotate(result)


def import_command(
    sources: Sequence[str],
    *,
    repo: str | None,
    platform: str,
    max_items: int | None,
    download_concurrency: int | None,
    check_media: bool,
    fail_fast: bool,
    preparation: Literal["full", "metadata", "download_all"] = "full",
    official_pack_policy: str | None = None,
    official_confirmation: Callable[[str], bool] | None = None,
    refresh: bool = False,
) -> CommandResult:
    from itertools import groupby

    from mojilex_cli.i18n import current_ui_language
    from mojilex_cli.runs.pack_scope import source_state

    from .import_reuse import import_complete, reusable_imports
    from .official_packs import select_sources
    from .packs import _source_name
    from .runtime import report_progress

    selection = select_sources(
        sources,
        platform=platform,
        policy=official_pack_policy,
        confirmation=official_confirmation,
    )
    if not selection.selected:
        return selection.empty_result()
    del check_media  # import always verifies bytes; the option remains explicit in the CLI contract
    config = load_config(cli={"repository": {"target": repo}})
    existing = {} if refresh else reusable_imports(selection.selected, config, max_items=max_items)
    if existing:
        selectors: list[str] = []
        resumed: set[tuple[str, str]] = set()
        result = CommandResult(status="noop")  # type: ignore[arg-type]
        groups = groupby(
            selection.selected,
            key=lambda source: existing[source][0].run_id if source in existing else None,
        )
        for parent_id, entries in groups:
            group_sources = tuple(entries)
            if parent_id is None:
                result = run_import(
                    group_sources,
                    PipelineOptions(
                        repository=repo,
                        platform=platform,
                        max_items=max_items,
                        download_concurrency=download_concurrency,
                        check_media=True,
                        fail_fast=fail_fast,
                        import_strategy=preparation,
                        publish="local",
                        official_approved_sources=selection.approved_sources,
                    ),
                )
                if result.status not in {"succeeded", "noop"}:
                    return selection.annotate(result)
                if result.run_id:
                    selectors.append(result.run_id)
                continue
            selected = []
            for source in group_sources:
                checkpoint, saved_source = existing[source]
                if (
                    not import_complete(checkpoint, saved_source)
                    and (parent_id, saved_source) not in resumed
                ):
                    if source_state(checkpoint, saved_source)["phase"] != "import":
                        name = _source_name(saved_source)
                        raise CommandError(
                            "CONFIG_INVALID",
                            "A saved analysis needs to be resumed.",
                            hint=f"Use mojilex resume {parent_id}:{name} to continue safely.",
                        )
                    selected.append(saved_source)
            if selected:
                result = run_resume_sync(
                    parent_id,
                    selected_sources=tuple(dict.fromkeys(selected)),
                    download_concurrency=download_concurrency,
                    official_pack_policy="allow",
                )
                if result.status not in {"succeeded", "noop"}:
                    return selection.annotate(result)
                resumed.update((parent_id, saved) for saved in selected)
            for source in group_sources:
                checkpoint, saved_source = existing[source]
                name = _source_name(saved_source)
                state = source_state(checkpoint, saved_source)
                if state["phase"] != "describe" or state["status"] not in {"succeeded", "noop"}:
                    selectors.append(f"{parent_id}:{name}")
            result.run_id = parent_id
        result.result["analysis_selectors"] = list(dict.fromkeys(selectors))
        result.result["reused_packs"] = len(existing)
        new_count = sum(source not in existing for source in selection.selected)
        report_progress(
            f"Использованы сохранённые паки: {len(existing)}; новых паков: {new_count}."
            if current_ui_language() == "ru"
            else f"Reused saved packs: {len(existing)}; new packs: {new_count}."
        )
        return selection.annotate(result)
    result = run_import(
        selection.selected,
        PipelineOptions(
            repository=repo,
            platform=platform,
            max_items=max_items,
            download_concurrency=download_concurrency,
            check_media=True,
            fail_fast=fail_fast,
            import_strategy=preparation,
            publish="local",
            official_approved_sources=selection.approved_sources,
        ),
    )
    return selection.annotate(result)


def describe_command(
    selectors: Sequence[str],
    *,
    provider: str | None,
    model: str | None,
    max_ai_requests: int | Literal["unlimited"] | None,
    max_cost_usd: Decimal | None,
    allow_unknown_cost: bool,
    ai_concurrency: int | None = None,
    unknown_cost_confirmation: Callable[[int | None], bool] | None = None,
    official_pack_policy: str | None = None,
    official_confirmation: Callable[[str], bool] | None = None,
) -> CommandResult:
    if len(selectors) > 1 and all(value.startswith("mlxrun_") for value in selectors):
        from .packs import _sources, resolve_pack_run, selected_pack_sources

        groups: list[tuple[str, list[str]]] = []
        for selector in selectors:
            checkpoint = resolve_pack_run(selector, purpose="describe")
            selected = selected_pack_sources(checkpoint, selector)
            if not groups or groups[-1][0] != checkpoint.run_id:
                groups.append((checkpoint.run_id, []))
            values = groups[-1][1]
            values.extend(selected or _sources(checkpoint))
        approved: bool | None = None

        def approve_once(limit: int | None) -> bool:
            nonlocal approved
            if approved is None:
                approved = bool(unknown_cost_confirmation and unknown_cost_confirmation(limit))
            return approved

        result = CommandResult()
        for run_id, values in groups:
            result = run_describe(
                run_id,
                PipelineOptions(
                    selected_sources=tuple(dict.fromkeys(values)),
                    provider=provider,
                    model=model,
                    ai_concurrency=ai_concurrency,
                    max_ai_requests=max_ai_requests,
                    max_cost_usd=max_cost_usd,
                    allow_unknown_cost=allow_unknown_cost,
                    unknown_cost_confirmation=approve_once,
                    official_pack_policy=official_pack_policy,
                    official_confirmation=official_confirmation,
                ),
            )
            if result.status not in {"succeeded", "noop"}:
                break
        return result
    selected_sources: tuple[str, ...] = ()
    saved_run = selectors[0] if len(selectors) == 1 and selectors[0].startswith("mlxrun_") else None
    if saved_run is not None and ":" in saved_run:
        from .packs import resolve_pack_run, selected_pack_sources

        checkpoint = resolve_pack_run(saved_run, purpose="describe")
        selected_sources = selected_pack_sources(checkpoint, saved_run)
        saved_run = checkpoint.run_id
    if len(selectors) == 1 and saved_run is None and not selectors[0].startswith(("mxe_", "mxc_")):
        from .packs import resolve_pack_run, selected_pack_sources

        try:
            checkpoint = resolve_pack_run(selectors[0], purpose="describe")
            saved_run = checkpoint.run_id
            selected_sources = selected_pack_sources(checkpoint, selectors[0])
        except CommandError as exc:
            if exc.error.code != "CONFIG_MISSING":
                raise
    if saved_run is not None:
        return run_describe(
            saved_run,
            PipelineOptions(
                provider=provider,
                selected_sources=selected_sources,
                model=model,
                ai_concurrency=ai_concurrency,
                max_ai_requests=max_ai_requests,
                max_cost_usd=max_cost_usd,
                allow_unknown_cost=allow_unknown_cost,
                unknown_cost_confirmation=unknown_cost_confirmation,
                official_pack_policy=official_pack_policy,
                official_confirmation=official_confirmation,
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
    from .official_packs import select_sources

    selection = select_sources(
        sources,
        platform="auto",
        policy=official_pack_policy,
        confirmation=official_confirmation,
    )
    if not selection.selected:
        return selection.empty_result()
    result = run_add(
        selection.selected,
        PipelineOptions(
            repository=config.repository.target,
            provider=config.ai.provider,
            model=config.ai.model,
            ai_concurrency=config.ai.ai_concurrency,
            max_ai_requests=(
                config.ai.max_ai_requests if config.ai.max_ai_requests is not None else "unlimited"
            ),
            max_cost_usd=config.ai.max_cost_usd,
            allow_unknown_cost=config.ai.allow_unknown_cost,
            unknown_cost_confirmation=unknown_cost_confirmation,
            redescribe="all",
            publish="local",
            official_approved_sources=selection.approved_sources,
        ),
    )
    return selection.annotate(result)


def update_command(
    selector: str | None,
    *,
    all_collections: bool,
    repo: str | None,
    dry_run: bool,
    official_pack_policy: str | None = None,
    official_confirmation: Callable[[str], bool] | None = None,
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
    from .official_packs import SourceSelection, select_sources

    selection = (
        SourceSelection(tuple(sources))
        if dry_run
        else select_sources(
            sources,
            platform="auto",
            policy=official_pack_policy,
            confirmation=official_confirmation,
        )
    )
    if not selection.selected:
        return selection.empty_result()
    result = run_add(
        selection.selected,
        PipelineOptions(
            repository=config.repository.target,
            dry_run=dry_run,
            check_media=dry_run,
            explicit_verification=True,
            official_approved_sources=selection.approved_sources,
        ),
    )
    return selection.annotate(result)


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
    download_concurrency: int | None = None,
    max_ai_requests: int | Literal["unlimited"] | None = None,
    confirmation: Callable[[str], bool] | None = None,
    unknown_cost_confirmation: Callable[[int | None], bool] | None = None,
    official_pack_policy: str | None = None,
    official_confirmation: Callable[[str], bool] | None = None,
) -> CommandResult:
    from .packs import _pack_phase_status, _selector_name, resolve_pack_run, selected_pack_sources

    selector = run_id
    checkpoint = resolve_pack_run(selector, purpose="resume")
    run_id = checkpoint.run_id
    selected_sources = selected_pack_sources(checkpoint, selector)
    phase, status = _pack_phase_status(checkpoint, _selector_name(selector))
    if status in {"succeeded", "noop"}:
        next_command = "describe" if phase == "import" else "show"
        return CommandResult(
            run_id=run_id,
            status="noop",  # type: ignore[arg-type]
            result={
                "message": "This run is already complete.",
                "next": f"mojilex {next_command} {selector}",
            },
        )
    overrides: _ResumeOverrides = {}
    if selected_sources:
        overrides["selected_sources"] = selected_sources
    if max_ai_requests is not None:
        overrides["max_ai_requests"] = max_ai_requests
    if ai_concurrency is not None:
        overrides["ai_concurrency"] = ai_concurrency
    if download_concurrency is not None:
        overrides["download_concurrency"] = download_concurrency
    if official_pack_policy is not None:
        overrides["official_pack_policy"] = official_pack_policy
    if official_confirmation is not None:
        overrides["official_confirmation"] = official_confirmation
    return run_resume_sync(
        run_id,
        confirmation=confirmation,
        unknown_cost_confirmation=unknown_cost_confirmation,
        **overrides,
    )


def publish_pack_command(
    selector: str,
    *,
    local: bool = False,
    direct_push: bool = False,
    confirmation: Callable[[str], bool] | None = None,
) -> CommandResult:
    from .packs import _selector_name, resolve_pack_run, selected_pack_sources

    if local and direct_push:
        raise CommandError(
            "CONFIG_INVALID",
            "Choose either --local or --direct-push.",
            hint="Use mojilex publish PACK for a GitHub pull request.",
        )
    checkpoint = resolve_pack_run(selector, purpose="publish")
    if selected_pack_sources(checkpoint, selector):
        from mojilex_cli.pipeline.pack_publication import prepare_pack_publication

        checkpoint = prepare_pack_publication(
            checkpoint, cast(str, _selector_name(selector)), load_config()
        )
    if checkpoint.command != "describe" or checkpoint.status not in {"succeeded", "noop"}:
        raise CommandError(
            "CONFIG_INVALID",
            "The pack analysis is not complete.",
            hint=f"Complete mojilex describe {selector} before publishing.",
        )
    return submit_command(
        checkpoint.run_id,
        repo=None,
        publish="local" if local else "pr",
        direct_push=direct_push,
        base=None,
        confirmation=confirmation,
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
