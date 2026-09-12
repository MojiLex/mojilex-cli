"""End-to-end collection processing with explicit persistence boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import re
import sqlite3
import subprocess
import tempfile
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import rfc8785

from mojilex_cli import __version__
from mojilex_cli.ai import (
    AIError,
    AIOutputError,
    BilingualDescriptions,
    ContentClassification,
    DescriptionBatch,
    DescriptionItem,
    DescriptionRequest,
    DescriptionResult,
    RequestBudget,
    SemanticFacets,
    VisionContext,
    VisionImage,
    default_registry,
    describe_with_recovery,
)
from mojilex_cli.ai import (
    LocalizedDescription as AILocalizedDescription,
)
from mojilex_cli.ai.concepts import ConceptContext, load_concept_context
from mojilex_cli.ai.prompts import (
    gemini_request_parameters_sha256,
    prompt_sha256,
)
from mojilex_cli.analysis import (
    AnalysisError,
    DeterministicMediaAnalysis,
    decoder_backend_fingerprint,
    load_analysis_profile,
)
from mojilex_cli.cache import (
    AICacheWrite,
    CachedAIResult,
    CacheError,
    CacheStore,
    ai_cache_key,
    canonical_context_hash,
    deterministic_analysis_key,
)
from mojilex_cli.cache import (
    media_digest as cache_media_digest,
)
from mojilex_cli.commands.runtime import (
    CommandError,
    CommandResult,
    report_progress,
    report_run_id,
    structured_exception,
)
from mojilex_cli.config import MojiLexConfig, load_config, load_credentials
from mojilex_cli.dataset import (
    DatasetSnapshot,
    apply_snapshot,
    load_dataset,
    validate_dataset,
    validate_snapshot,
)
from mojilex_cli.dedupe import scan_snapshot
from mojilex_cli.domain import SCHEMA_VERSION, DeterministicEmojiAnalysis, Emoji, RoutingReason
from mojilex_cli.domain import media_digest as domain_media_digest
from mojilex_cli.git import (
    GitIdentity,
    GitPublisher,
    GitRunner,
    PreparedCommit,
    git_subprocess_environment,
    make_import_branch,
)
from mojilex_cli.github import (
    DirectPushAuthorization,
    GitHubCLI,
    GitHubPublisher,
    PublicationPhase,
    RepositoryRef,
)
from mojilex_cli.media import (
    PIPELINE_VERSION,
    ContactSheet,
    ContactSheetInput,
    MediaLimits,
    MediaMetadata,
    MediaProcessor,
    ProcessedMedia,
    SourceChangedDuringRunError,
    TemporaryMediaRun,
    build_contact_sheets,
    expected_labels,
)
from mojilex_cli.output.models import RunStatus, StructuredError
from mojilex_cli.policy import (
    ROUTING_POLICY_VERSION,
    ModelQualificationRegistry,
    QualificationQuery,
    ReviewRoutingReport,
    RoutingReasonRegistry,
    compute_review_routing,
    deterministic_routing_reasons,
    load_review_policy,
    match_qualification,
    official_submission_report,
    semantic_routing_reasons,
    should_escalate,
)
from mojilex_cli.policy.model_routing import build_model_routing_binding
from mojilex_cli.policy.qualification import CONCEPT_BINDING_FIELDS
from mojilex_cli.runs import (
    AIRequestCheckpoint,
    DedupeScanCheckpoint,
    ElementCheckpoint,
    PublicationCheckpoint,
    RunCheckpoint,
    RunIssue,
    RunStore,
    new_checkpoint,
    new_run_id,
)
from mojilex_cli.sources import (
    SourceCollection,
    SourceEmoji,
    SourceNotFoundError,
    TelegramBotAPI,
    telegram_set_fingerprint,
)

from .reapply import reapply_candidate
from .transform import (
    PROMPT_VERSION,
    IdentityConflictError,
    SemanticGenerationMetadata,
    plan_collection_merge,
)
from .workspaces import prepare_staging_workspace, snapshot_at_revision

_OWNER_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9_.-]{1,100}\Z")
_TERMINAL_ERROR_CODES = frozenset(
    {
        "AI_BUDGET_EXCEEDED",
        "AUTH_FAILED",
        "BUDGET_EXCEEDED",
        "CONFIG_INVALID",
        "CONFIG_MISSING",
        "CREDENTIAL_MISSING",
        "IDENTITY_CONFLICT",
        "POLICY_INVALID",
        "POLICY_REVIEW_BLOCKING",
        "SOURCE_CHANGED_DURING_RUN",
        "UNKNOWN_COST",
    }
)


def _configured_git_identity(config: MojiLexConfig) -> GitIdentity | None:
    name = config.git_identity.name
    email = config.git_identity.email
    return GitIdentity(name, email) if name is not None and email is not None else None


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    repository: str | None = None
    platform: str = "auto"
    provider: str | None = None
    model: str | None = None
    languages: tuple[str, ...] = ()
    publish: str | None = None
    direct_push: bool = False
    base: str | None = None
    redescribe: str = "changed"
    overwrite_reviewed: bool = False
    new_identity: bool = False
    same_identity: bool = False
    max_items: int | None = None
    max_ai_requests: int | None = None
    max_cost_usd: Decimal | None = None
    allow_unknown_cost: bool = False
    ai_concurrency: int | None = None
    download_concurrency: int | None = None
    dedupe: str | None = None
    max_dedupe_candidates: int | None = None
    dedupe_profile: str | None = None
    model_routing: str | None = None
    escalation_model: str | None = None
    dry_run: bool = False
    check_media: bool = False
    fail_fast: bool = False
    explicit_verification: bool = False
    confirmation: Callable[[str], bool] | None = field(default=None, repr=False, compare=False)
    unknown_cost_confirmation: Callable[[int], bool] | None = field(
        default=None, repr=False, compare=False
    )


@dataclass(frozen=True, slots=True)
class RepositoryWorkspace:
    root: Path
    target: RepositoryRef
    temporary: bool


@dataclass(slots=True)
class _AIState:
    providers: dict[str, Any] = field(default_factory=dict)
    credentials_validated: set[str] = field(default_factory=set)
    initialization_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cache_hits: int = 0


@dataclass(frozen=True, slots=True)
class _GenerationInputs:
    concepts: ConceptContext
    routing_fields: dict[str, str]
    routing_body: dict[str, Any]

    @property
    def fields(self) -> dict[str, str]:
        return {**self.concepts.provenance_fields, **self.routing_fields}


_GENERATION_INPUTS: ContextVar[_GenerationInputs | None] = ContextVar(
    "mojilex_generation_inputs", default=None
)


def _generation_binding_fields() -> dict[str, str]:
    inputs = _GENERATION_INPUTS.get()
    return inputs.fields if inputs is not None else {}


def _load_generation_inputs(
    snapshot: DatasetSnapshot, config: MojiLexConfig
) -> _GenerationInputs | None:
    registry = snapshot.root / "taxonomy" / "v1" / "concepts.json"
    profile = snapshot.root / "analysis-profiles" / "concept-candidates-v1.json"
    if not registry.exists() and not profile.exists():
        # Legacy/local fixtures have no Stage-B concept rollout. They remain
        # pending and cannot gain trust merely by using a newer importer.
        return None
    concepts = load_concept_context(snapshot.root)
    if not config.ai.model:
        return None  # Metadata-only dry runs do not require an AI model.
    primary = {
        "provider": config.ai.provider,
        "model": config.ai.model,
        "model_revision": None,
        "description_profile": "standard-v1",
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": str(snapshot.manifest["taxonomy_version"]),
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "request_parameters_sha256": gemini_request_parameters_sha256(),
        "languages": sorted(config.ai.languages),
        **concepts.provenance_fields,
    }
    escalation = (
        {**primary, "model": config.ai.escalation_model}
        if config.ai.model_routing == "rules"
        else None
    )
    routing = build_model_routing_binding(primary, escalation, mode=config.ai.model_routing)
    return _GenerationInputs(concepts, routing.provenance_fields, routing.body)


@dataclass(frozen=True, slots=True)
class _AIRequestIdentity:
    plan_sha256: str
    request_sha256: str
    shown_media_sha256: tuple[str, ...]
    labels_by_native: tuple[tuple[str, str], ...]

    def label_for(self, native_id: str) -> str:
        labels = dict(self.labels_by_native)
        try:
            return labels[native_id]
        except KeyError as exc:
            raise AIOutputError("AI request identity does not cover the source item") from exc

    @property
    def cache_identity_sha256(self) -> str:
        """Bind exact model-visible bytes to the ordered native request plan."""

        return hashlib.sha256(
            rfc8785.dumps(
                {
                    "format_version": 1,
                    "plan_sha256": self.plan_sha256,
                    "request_sha256": self.request_sha256,
                }
            )
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class _PreparedAIRequest:
    request: DescriptionRequest
    identity: _AIRequestIdentity


@dataclass(frozen=True, slots=True)
class _AICacheTrace:
    stage: Literal["primary", "escalated"]
    model: str
    model_revision: str | None
    cache_key: str
    request_identity: _AIRequestIdentity


@dataclass(frozen=True, slots=True)
class _SemanticOutcome:
    description: DescriptionItem
    generation: SemanticGenerationMetadata
    request_trace: tuple[_AICacheTrace, ...] = ()


@dataclass(frozen=True, slots=True)
class _AIResultWrite:
    item: SourceEmoji
    lookup_key: str
    storage_key: str
    result: DescriptionResult
    generated_at: str


_MediaCompletion = Callable[[SourceEmoji, ProcessedMedia], Awaitable[None]]
_AIChunkCompletion = Callable[
    [Sequence[SourceEmoji], Mapping[str, _SemanticOutcome]], Awaitable[None]
]


def run_add(sources: Sequence[str], options: PipelineOptions) -> CommandResult:
    return asyncio.run(_run_add(sources, options))


def run_import(sources: Sequence[str], options: PipelineOptions) -> CommandResult:
    return asyncio.run(_run_import(sources, options))


def run_describe(run_id: str, overrides: PipelineOptions | None = None) -> CommandResult:
    config = load_config()
    store = RunStore(cast(Path, config.runs_dir))
    with store.execution_lock(run_id):
        return asyncio.run(_run_describe(run_id, overrides))


def run_resume_sync(
    run_id: str,
    *,
    confirmation: Callable[[str], bool] | None = None,
    unknown_cost_confirmation: Callable[[int], bool] | None = None,
) -> CommandResult:
    config = load_config()
    store = RunStore(cast(Path, config.runs_dir))
    with store.execution_lock(run_id):
        return asyncio.run(
            run_resume(
                run_id,
                confirmation=confirmation,
                unknown_cost_confirmation=unknown_cost_confirmation,
            )
        )


def run_submit(
    target: str | None,
    *,
    repository: str | None,
    direct_push: bool,
    base: str | None,
    confirmation: Callable[[str], bool] | None = None,
) -> CommandResult:
    if target is not None and target.startswith("mlxrun_"):
        config = load_config()
        store = RunStore(cast(Path, config.runs_dir))
        with store.execution_lock(target):
            return asyncio.run(
                _run_submit(
                    target,
                    repository=repository,
                    direct_push=direct_push,
                    base=base,
                    confirmation=confirmation,
                )
            )
    return asyncio.run(
        _run_submit(
            target,
            repository=repository,
            direct_push=direct_push,
            base=base,
            confirmation=confirmation,
        )
    )


async def _run_add(
    sources: Sequence[str],
    options: PipelineOptions,
    *,
    resume_id: str | None = None,
    expected_hashes: Mapping[str, tuple[str, ...]] | None = None,
    expected_memberships: Mapping[str, tuple[str, ...]] | None = None,
    resume_checkpoint: RunCheckpoint | None = None,
    stage_only: bool = False,
) -> CommandResult:
    if not sources:
        raise CommandError(
            "CONFIG_MISSING",
            "At least one source is required.",
            hint="Pass an addemoji URL, --from-file, or --stdin.",
        )
    all_sources = tuple(sources)
    config = _resolved_config(options)
    _validate_options(options, config)
    _validate_sources_for_platform(all_sources, options.platform)
    credentials = load_credentials()
    if not credentials.telegram_bot_token and (
        resume_checkpoint is None or resume_checkpoint.publication is None
    ):
        raise CommandError(
            "CREDENTIAL_MISSING",
            "TELEGRAM_BOT_TOKEN is not available.",
            hint="Set it in the current process environment; do not put it in a config file.",
        )
    isolated_publication = (
        not options.dry_run
        and not stage_only
        and (config.repository.publish == "pr" or options.direct_push)
    )
    with repository_workspace(
        config.repository.target,
        config.repository.base_branch,
        isolated=isolated_publication,
    ) as workspace:
        initial_report = validate_dataset(workspace.root, strict=True)
        if not initial_report.valid and not (
            stage_only and _only_staging_review_issues(initial_report)
        ):
            initial_report.raise_for_errors()
        initial = load_dataset(workspace.root)
        git = GitRunner(workspace.root, github_token=credentials.github_token)
        base_sha = git.current_sha()
        run_identifier = resume_id or new_run_id()
        report_run_id(run_identifier)
        run_store = RunStore(
            cast(Path, config.runs_dir),
            repository_root=workspace.root,
            write_enabled=not options.dry_run,
        )
        completed_source_indexes: set[int] = set()
        if not options.dry_run and (config.repository.publish == "pr" or options.direct_push):
            git.fetch("origin", config.repository.base_branch)
            remote_base = git.remote_sha("origin", config.repository.base_branch)
            if remote_base != base_sha:
                raise CommandError(
                    "GIT_CONFLICT",
                    "The local checkout is not at the current remote base commit.",
                    hint="Update the clean data checkout to origin/main and retry.",
                )
            base_sha = remote_base
            if resume_checkpoint is not None and resume_checkpoint.publication is not None:
                resume_checkpoint, completed_source_indexes = _reconcile_publication_for_resume(
                    resume_checkpoint,
                    git=git,
                    remote_base=remote_base,
                    base_branch=config.repository.base_branch,
                    expected_mode="direct" if options.direct_push else "pr",
                    source_count=len(all_sources),
                )
                run_store.save(resume_checkpoint)
        elif resume_checkpoint is not None and resume_checkpoint.publication is not None:
            raise CommandError(
                "CONFIG_INVALID",
                "A remote publication checkpoint cannot resume in a local-only workflow.",
                hint="Resume with the publication mode recorded by the original run.",
            )
        source_entries = _remaining_source_entries(all_sources, completed_source_indexes)
        sources = tuple(source for _, source in source_entries)
        checkpoint = None
        if not options.dry_run:
            if resume_checkpoint is not None:
                if resume_checkpoint.run_id != run_identifier:
                    raise CommandError(
                        "CONFIG_INVALID",
                        "The resume run ID does not match.",
                        hint="Resume the checkpoint using its original run ID.",
                    )
                if base_sha != resume_checkpoint.base_revision:
                    raise CommandError(
                        "SOURCE_CHANGED_DURING_RUN",
                        "The staging checkout no longer matches the imported base revision.",
                        hint="Keep the staging checkout unchanged or start a new import.",
                    )
                checkpoint = resume_checkpoint.model_copy(
                    update={
                        "command": "describe" if stage_only else resume_checkpoint.command,
                        "status": "running",
                        "safe_parameters": {
                            **resume_checkpoint.safe_parameters,
                            **_safe_parameters(
                                all_sources,
                                _materialized_options(options, config),
                            ),
                        },
                        "updated_at": datetime.now(UTC).replace(microsecond=0),
                    }
                )
            else:
                checkpoint = new_checkpoint(
                    command="add",
                    safe_parameters=_safe_parameters(
                        all_sources, _materialized_options(options, config)
                    ),
                    cli_version=__version__,
                    schema_version=SCHEMA_VERSION,
                    target_repository=str(workspace.target),
                    base_revision=base_sha,
                    run_id=run_identifier,
                )
            run_store.save(checkpoint)
            if not source_entries:
                checkpoint = _finish_checkpoint(
                    checkpoint,
                    "succeeded",
                    checkpoint.ai_requests_used,
                    checkpoint.ai_cost_reserved_usd,
                )
                run_store.save(checkpoint)
                publication_checkpoint = checkpoint.publication
                assert publication_checkpoint is not None
                return CommandResult(
                    run_id=run_identifier,
                    status=RunStatus.SUCCEEDED,
                    result={"sources_processed": 0, "changed_paths": []},
                    publication={
                        "mode": publication_checkpoint.mode,
                        "commit": publication_checkpoint.candidate_sha,
                        "candidate_branch": publication_checkpoint.candidate_branch,
                        "reconciled": True,
                    },
                )
        if not credentials.telegram_bot_token:
            raise CommandError(
                "CREDENTIAL_MISSING",
                "TELEGRAM_BOT_TOKEN is not available.",
                hint="Set it in the current process environment; do not put it in a config file.",
            )

        cache: CacheStore | None = None
        cache_path = cast(Path, config.cache_dir) / "cache-v1.sqlite3"
        if options.dry_run and cache_path.is_file():
            try:
                cache = CacheStore(cache_path, repository_root=workspace.root, read_only=True)
            except CacheError:
                # A broken optional cache cannot make a read-only source preview fail.
                report_progress(
                    "The existing AI cache could not be inspected; plan assumes misses."
                )
        elif not options.dry_run:
            cache = CacheStore(
                cache_path,
                repository_root=workspace.root,
            )

        def record_budget_reservation(requests_used: int, cost_reserved: Decimal) -> None:
            nonlocal checkpoint
            if checkpoint is None:
                return
            checkpoint = _persist_budget_reservation(
                checkpoint,
                run_store,
                requests_used=requests_used,
                cost_reserved=cost_reserved,
            )

        budget = RequestBudget(
            max_requests=config.ai.max_ai_requests,
            max_cost_usd=config.ai.max_cost_usd,
            allow_unknown_cost=config.ai.allow_unknown_cost,
            unknown_cost_authorizer=options.unknown_cost_confirmation,
            reservation_recorder=(None if options.dry_run else record_budget_reservation),
            requests_used=resume_checkpoint.ai_requests_used if resume_checkpoint else 0,
            cost_reserved=(
                resume_checkpoint.ai_cost_reserved_usd if resume_checkpoint else Decimal("0")
            ),
        )
        ai_state = _AIState()
        current = initial
        warnings: list[dict[str, Any] | str] = []
        errors: list[StructuredError] = []
        totals = {
            "collections_created": 0,
            "collections_updated": 0,
            "items_added": 0,
            "items_updated": 0,
            "items_unchanged": 0,
            "items_disappeared": 0,
            "memberships_removed": 0,
        }
        source_collections: list[SourceCollection] = []
        preview_ai = {
            "ai_items_planned": 0,
            "ai_batches_planned": 0,
            "ai_cache_hits_estimated": 0,
            "ai_cache_hits_unknown": 0,
            "ai_requests_estimated_upper_bound": 0,
        }
        successful_source_indexes: set[int] = set()
        dedupe_emoji_ids: set[str] = set()
        dedupe_report: dict[str, Any] | None = None
        progress_lock = asyncio.Lock()

        async def record_media_completion(item: SourceEmoji, value: ProcessedMedia) -> None:
            nonlocal checkpoint
            if checkpoint is None or cache is None:
                return
            async with progress_lock:
                _cache_deterministic_analysis(cache, item, value)
                updated = _checkpoint_media_item(checkpoint, item, value)
                run_store.save(updated)
                checkpoint = updated
                report_progress(f"Media verified: {item.native_id}", verbose=True)

        async def record_ai_chunk_completion(
            items: Sequence[SourceEmoji],
            outcomes: Mapping[str, _SemanticOutcome],
        ) -> None:
            nonlocal checkpoint
            if checkpoint is None:
                return
            traces = {
                native_id: outcome.request_trace
                for native_id, outcome in outcomes.items()
                if outcome.request_trace
            }
            generation = {native_id: outcome.generation for native_id, outcome in outcomes.items()}
            partial_source = source.model_copy(
                update={"item_count": len(items), "items": tuple(items)}
            )
            async with progress_lock:
                updated = _checkpoint_ai_keys(
                    checkpoint,
                    partial_source,
                    {},
                    config,
                    generation,
                    taxonomy_version=str(current.manifest["taxonomy_version"]),
                    request_traces=traces,
                )
                updated = _checkpoint_stage(
                    updated,
                    tuple(item.native_id for item in items),
                    "ai_cached",
                )
                updated = _checkpoint_budget(updated, budget)
                run_store.save(updated)
                checkpoint = updated
                report_progress(f"Semantic results cached: {len(items)} item(s)", verbose=True)

        generation_token = _GENERATION_INPUTS.set(None)
        try:
            generation_inputs = _load_generation_inputs(current, config)
            _GENERATION_INPUTS.set(generation_inputs)
            if checkpoint is not None and generation_inputs is not None:
                checkpoint = checkpoint.model_copy(
                    update={
                        "safe_parameters": {
                            **checkpoint.safe_parameters,
                            "concept_generation_binding": generation_inputs.fields,
                            "model_routing_policy": generation_inputs.routing_body,
                        }
                    }
                )
                run_store.save(checkpoint)
            async with TelegramBotAPI(
                credentials.telegram_bot_token,
                timeout_seconds=config.telegram.timeout_seconds,
                max_attempts=config.telegram.max_attempts,
                max_download_bytes=config.processing.max_download_bytes,
            ) as adapter:
                await adapter.validate_credentials()
                for source_index, source_text in source_entries:
                    collection_lock: Any | None = None
                    lock_entered = False
                    try:
                        reference = adapter.canonicalize(source_text)
                        report_progress(
                            f"Checking source {source_index + 1}/{len(all_sources)}: "
                            f"{reference.native_id}"
                        )
                        if not options.dry_run:
                            collection_lock = run_store.collection_lock(
                                reference.platform, reference.native_id
                            )
                            collection_lock.__enter__()
                            lock_entered = True
                        try:
                            source = await adapter.fetch_collection(reference)
                        except SourceNotFoundError:
                            if not options.explicit_verification:
                                raise
                            current, availability_updates = await _mark_missing_collection(
                                current,
                                adapter,
                                platform=reference.platform,
                                native_id=reference.native_id,
                            )
                            totals["collections_updated"] += int(availability_updates > 0)
                            totals["items_updated"] += max(0, availability_updates - 1)
                            successful_source_indexes.add(source_index)
                            continue
                        imported_members = (
                            expected_memberships.get(source.native_id)
                            if expected_memberships is not None
                            else None
                        )
                        current_members = tuple(item.native_id for item in source.items)
                        if imported_members is not None and current_members != imported_members:
                            raise CommandError(
                                "SOURCE_CHANGED_DURING_RUN",
                                f"Collection {source.native_id} changed after import.",
                                hint="Start a new import so media and membership metadata agree.",
                                source=source.canonical_url,
                                details={
                                    "imported_count": len(imported_members),
                                    "current_count": len(current_members),
                                },
                            )
                        if options.max_items is not None and source.item_count > options.max_items:
                            raise CommandError(
                                "CONFIG_INVALID",
                                f"Collection {source.native_id} has {source.item_count} items, "
                                f"above --max-items={options.max_items}.",
                                hint=(
                                    "Raise --max-items deliberately or choose a smaller collection."
                                ),
                                source=source.canonical_url,
                            )
                        if options.dry_run and not options.check_media:
                            _accumulate_preview(totals, current, source)
                            preview_plan = await _preview_ai_plan(
                                current, source, config=config, options=options, cache=cache
                            )
                            for key, value in preview_plan.items():
                                preview_ai[key] += value
                            source_collections.append(source)
                            successful_source_indexes.add(source_index)
                            continue
                        with TemporaryMediaRun(limits=_media_limits(config)) as temporary:
                            processor = MediaProcessor(temporary)
                            verified_resume_outcomes: dict[str, _SemanticOutcome] = {}
                            source, processed = await _prepare_collection_media(
                                current,
                                adapter,
                                source,
                                processor,
                                concurrency=config.telegram.download_concurrency,
                                expected_hashes=expected_hashes,
                                cache=None if options.dry_run else cache,
                                resume_elements=(
                                    checkpoint.elements if checkpoint is not None else None
                                ),
                                config=config,
                                taxonomy_version=str(current.manifest["taxonomy_version"]),
                                cache_alias_scope=run_identifier,
                                redescribe=options.redescribe,
                                overwrite_reviewed=options.overwrite_reviewed,
                                verified_semantic_outcomes=verified_resume_outcomes,
                                on_item_completed=(
                                    None if options.dry_run else record_media_completion
                                ),
                            )
                            if checkpoint is not None:
                                checkpoint = _checkpoint_media(checkpoint, source, processed)
                                run_store.save(checkpoint)
                            if options.dry_run:
                                _accumulate_preview(totals, current, source, processed=processed)
                                preview_plan = await _preview_ai_plan(
                                    current,
                                    source,
                                    config=config,
                                    options=options,
                                    cache=cache,
                                    processed=processed,
                                    temporary=temporary,
                                )
                                for key, value in preview_plan.items():
                                    preview_ai[key] += value
                                source_collections.append(source)
                                successful_source_indexes.add(source_index)
                                continue
                            assert cache is not None
                            _cache_deterministic_analyses(cache, source, processed)
                            checkpoint_request_traces = _request_traces_from_checkpoint(
                                source.items,
                                checkpoint.elements if checkpoint is not None else {},
                            )
                            request_traces: dict[str, tuple[_AICacheTrace, ...]] = {}
                            descriptions, generation_metadata = await _descriptions_for_collection(
                                current,
                                source,
                                processed,
                                config=config,
                                cache=cache,
                                budget=budget,
                                ai_state=ai_state,
                                api_key=credentials.gemini_api_key,
                                redescribe=options.redescribe,
                                overwrite_reviewed=options.overwrite_reviewed,
                                temporary=temporary,
                                resume_ai_cache_keys=(
                                    {
                                        native_id: element.ai_cache_key
                                        for native_id, element in checkpoint.elements.items()
                                        if element.ai_cache_key is not None
                                    }
                                    if checkpoint is not None
                                    else None
                                ),
                                cache_alias_scope=run_identifier,
                                verified_resume_outcomes=verified_resume_outcomes,
                                request_traces_out=request_traces,
                                resume_request_traces=checkpoint_request_traces,
                                on_chunk_completed=record_ai_chunk_completion,
                            )
                            analyses = _bind_deterministic_analyses(processed)
                            if checkpoint is not None:
                                checkpoint = _checkpoint_ai_keys(
                                    checkpoint,
                                    source,
                                    processed,
                                    config,
                                    generation_metadata,
                                    taxonomy_version=str(current.manifest["taxonomy_version"]),
                                    request_traces=request_traces,
                                )
                                checkpoint = _checkpoint_stage(
                                    checkpoint,
                                    tuple(item.native_id for item in source.items),
                                    "ai_cached",
                                )
                                checkpoint = _checkpoint_budget(checkpoint, budget)
                                run_store.save(checkpoint)
                            plan = plan_collection_merge(
                                current,
                                source,
                                processed,
                                descriptions,
                                analyses,
                                generation_metadata,
                                overwrite_reviewed=options.overwrite_reviewed,
                                new_identity=options.new_identity,
                                same_identity=options.same_identity,
                                explicit_verification=options.explicit_verification,
                            )
                            planned_snapshot = plan.snapshot
                            if options.explicit_verification:
                                (
                                    planned_snapshot,
                                    availability_updates,
                                ) = await _refresh_emoji_availability(
                                    planned_snapshot,
                                    current,
                                    source,
                                    adapter,
                                )
                                plan = replace(
                                    plan,
                                    snapshot=planned_snapshot,
                                    updated=plan.updated + availability_updates,
                                )
                            report = validate_snapshot(
                                plan.snapshot,
                                schemas=True,
                                repository_files=True,
                            )
                            if not report.valid:
                                if stage_only and _only_staging_review_issues(report):
                                    warnings.extend(_validation_warnings(report))
                                else:
                                    report.raise_for_errors()
                            sensitive_flags = tuple(
                                flag
                                for enabled, flag in (
                                    (options.overwrite_reviewed, "--overwrite-reviewed"),
                                    (options.new_identity, "--new-identity"),
                                    (options.same_identity, "--same-identity"),
                                )
                                if enabled
                            )
                            if sensitive_flags:
                                affected_ids = _changed_entity_ids(current, plan.snapshot)
                                changed_plan_paths = _changed_paths(current, plan.snapshot)
                                affected_text = ", ".join(affected_ids) or "(none)"
                                path_text = ", ".join(map(str, changed_plan_paths)) or "(none)"
                                _require_confirmation(
                                    options.confirmation,
                                    (
                                        f"Apply sensitive plan for source {source.canonical_url} "
                                        f"with flags {', '.join(sensitive_flags)}? "
                                        f"Exact affected IDs: {affected_text}. "
                                        "Exact changed paths: "
                                        f"{path_text}."
                                    ),
                                    hint=(
                                        "Review every affected ID, then confirm interactively "
                                        "or rerun with --yes."
                                    ),
                                )
                            totals["collections_created"] += int(
                                plan.collection_id not in current.collections
                            )
                            totals["collections_updated"] += int(
                                plan.collection_id in current.collections
                            )
                            totals["items_added"] += plan.created
                            totals["items_updated"] += plan.updated
                            totals["items_unchanged"] += max(
                                0, source.item_count - plan.created - plan.updated
                            )
                            totals["items_disappeared"] += plan.removed_memberships
                            totals["memberships_removed"] += plan.removed_memberships
                            for changed_id in plan.changed_entity_ids:
                                if not changed_id.startswith("mxe_"):
                                    continue
                                before_emoji = current.emojis.get(changed_id)
                                after_emoji = plan.snapshot.emojis.get(changed_id)
                                if (
                                    before_emoji is None
                                    or after_emoji is None
                                    or before_emoji.as_dict() != after_emoji.as_dict()
                                ):
                                    dedupe_emoji_ids.add(changed_id)
                            current = plan.snapshot
                            source_collections.append(source)
                            if checkpoint is not None:
                                checkpoint = _checkpoint_stage(
                                    checkpoint,
                                    tuple(item.native_id for item in source.items),
                                    "validated",
                                )
                                run_store.save(checkpoint)
                            successful_source_indexes.add(source_index)
                    except Exception as exc:
                        error = structured_exception(exc)
                        errors.append(error)
                        terminal_error = error.code in _TERMINAL_ERROR_CODES
                        if checkpoint is not None:
                            checkpoint = _checkpoint_budget(checkpoint, budget)
                            checkpoint = _checkpoint_issue(
                                checkpoint,
                                error,
                                terminal=terminal_error,
                            )
                            run_store.save(checkpoint)
                        if options.fail_fast or terminal_error:
                            raise
                    finally:
                        if collection_lock is not None and lock_entered:
                            collection_lock.__exit__(None, None, None)
            if options.dry_run:
                status = RunStatus.DRY_RUN
                if errors:
                    status = RunStatus.PARTIAL if len(errors) < len(sources) else RunStatus.FAILED
                return CommandResult(
                    run_id=run_identifier,
                    status=status,
                    result={
                        **totals,
                        "sources_checked": len(source_collections),
                        **preview_ai,
                        "ai_requests_estimated_upper_bound": min(
                            config.ai.max_ai_requests,
                            preview_ai["ai_requests_estimated_upper_bound"],
                        ),
                        "ai_plan_basis": "verified-media"
                        if options.check_media
                        else "source-metadata",
                        "persistent_writes": 0,
                    },
                    warnings=warnings,
                    errors=errors,
                )

            if isolated_publication:
                git.fetch("origin", config.repository.base_branch)
                publication_base = git.current_sha(
                    f"refs/remotes/origin/{config.repository.base_branch}"
                )
                if publication_base != base_sha:
                    if checkpoint is not None and checkpoint.publication is not None:
                        saved_publication = checkpoint.publication
                        if publication_base != saved_publication.candidate_sha:
                            raise CommandError(
                                "GIT_CONFLICT",
                                "The remote base changed to an unexpected commit during resume.",
                                hint=(
                                    "Inspect the remote base and candidate refs; automatic resume "
                                    "will not replace either ref."
                                ),
                            )
                        completed_source_indexes.update(saved_publication.completed_source_indexes)
                        checkpoint = checkpoint.model_copy(
                            update={
                                "publication": saved_publication.model_copy(
                                    update={"phase": "completed"}
                                )
                            }
                        )
                    git.run("switch", "--detach", publication_base)
                    latest = load_dataset(workspace.root)
                    current = reapply_candidate(initial, current, latest)
                    rebased_report = validate_snapshot(
                        current,
                        schemas=True,
                        repository_files=True,
                    )
                    rebased_report.raise_for_errors()
                    initial = latest
                    base_sha = publication_base
                    dedupe_emoji_ids = {
                        entity_id
                        for entity_id in _changed_entity_ids(initial, current)
                        if entity_id.startswith("mxe_")
                    }
                    if checkpoint is not None:
                        checkpoint = checkpoint.model_copy(
                            update={
                                "base_revision": publication_base,
                                "updated_at": datetime.now(UTC).replace(microsecond=0),
                            }
                        )
                        run_store.save(checkpoint)
            if config.dedupe.mode != "off":
                dedupe_emoji_ids = _resume_dedupe_selected_ids(
                    checkpoint,
                    current,
                    dedupe_emoji_ids,
                )
                dedupe_report = _cached_dedupe_report(
                    checkpoint,
                    current,
                    dedupe_emoji_ids,
                    config,
                )
                if dedupe_report is None:
                    dedupe_report = scan_snapshot(
                        current,
                        selected_emoji_ids=dedupe_emoji_ids,
                        max_candidates=config.dedupe.max_candidates,
                        mode=config.dedupe.mode,
                    ).as_dict()
                if checkpoint is not None:
                    # Re-materialize element completion flags even on a global
                    # checkpoint hit: media verification deliberately cleared
                    # them earlier in this resumed run.
                    checkpoint = _checkpoint_dedupe_report(
                        checkpoint,
                        current,
                        dedupe_emoji_ids,
                        config,
                        dedupe_report,
                    )
                    run_store.save(checkpoint)
            local_only = config.repository.publish == "local" and not options.direct_push
            if stage_only or local_only:
                qualifications = ModelQualificationRegistry.load(current.root)
                _, review_policy = load_review_policy(current.root)
                review_routing = compute_review_routing(
                    current,
                    qualifications,
                    review_policy,
                )
            else:
                review_routing = official_submission_report(current)
            changed_paths = _changed_paths(initial, current)
            publication: dict[str, Any] = {
                "mode": "staging" if stage_only else config.repository.publish
            }

            def record_publication(value: PublicationCheckpoint) -> None:
                nonlocal checkpoint
                if checkpoint is None:
                    raise RuntimeError("remote publication requires a durable run checkpoint")
                checkpoint = _checkpoint_publication_progress(checkpoint, value)
                run_store.save(checkpoint)

            if changed_paths:
                publisher = GitPublisher(git)
                guard = publisher.guard_targets(tuple(str(path) for path in changed_paths))
                _apply_with_rollback(
                    initial,
                    current,
                    allow_policy_review=stage_only or local_only,
                )
                if stage_only:
                    publication = {"mode": "staging", "path": str(workspace.root)}
                else:
                    publication = await _publish(
                        workspace,
                        config,
                        options,
                        git,
                        publisher,
                        guard,
                        changed_paths,
                        source_collections,
                        run_identifier,
                        totals,
                        credentials.github_token,
                        review_routing,
                        completed_source_indexes=tuple(
                            sorted(completed_source_indexes | successful_source_indexes)
                        ),
                        previous_publication=(
                            checkpoint.publication if checkpoint is not None else None
                        ),
                        record_publication=record_publication,
                    )
            status = RunStatus.SUCCEEDED if changed_paths else RunStatus.NOOP
            if errors:
                status = RunStatus.PARTIAL if changed_paths else RunStatus.FAILED
            if checkpoint is not None:
                checkpoint = _finish_checkpoint(
                    checkpoint,
                    status.value,
                    budget.requests_used,
                    budget.cost_reserved,
                )
                run_store.save(checkpoint)
            return CommandResult(
                run_id=run_identifier,
                status=status,
                result={
                    **totals,
                    "sources_processed": len(source_collections),
                    "changed_paths": [str(path) for path in changed_paths],
                    "ai_requests": budget.requests_used,
                    "ai_cache_hits": ai_state.cache_hits,
                    "ai_cost_reserved_usd": str(budget.cost_reserved),
                    "dedupe": dedupe_report,
                    "review_routing": review_routing.as_dict(),
                    **({"staging_repository": str(workspace.root)} if stage_only else {}),
                },
                publication=publication,
                warnings=warnings,
                errors=errors,
            )
        except BaseException as exc:
            if checkpoint is not None:
                error = structured_exception(exc)
                checkpoint = _checkpoint_issue(checkpoint, error, terminal=True)
                run_store.save(checkpoint)
            raise
        finally:
            _GENERATION_INPUTS.reset(generation_token)
            if cache is not None:
                cache.close()


async def _run_import(
    sources: Sequence[str],
    options: PipelineOptions,
    *,
    resume_id: str | None = None,
    expected_hashes: Mapping[str, tuple[str, ...]] | None = None,
    expected_memberships: Mapping[str, tuple[str, ...]] | None = None,
    resume_checkpoint: RunCheckpoint | None = None,
) -> CommandResult:
    if not sources:
        raise CommandError(
            "CONFIG_MISSING",
            "At least one source is required.",
            hint="Pass one or more public Telegram addemoji URLs.",
        )
    config = _resolved_config(options)
    _validate_options(options, config)
    _validate_sources_for_platform(sources, options.platform)
    credentials = load_credentials()
    if not credentials.telegram_bot_token:
        raise CommandError(
            "CREDENTIAL_MISSING",
            "TELEGRAM_BOT_TOKEN is not available.",
            hint="Set it in the current process environment.",
        )
    with repository_workspace(config.repository.target, config.repository.base_branch) as workspace:
        validate_dataset(workspace.root, strict=True).raise_for_errors()
        initial = load_dataset(workspace.root)
        base_sha = GitRunner(workspace.root).current_sha()
        runs_dir = cast(Path, config.runs_dir)
        store = RunStore(runs_dir, repository_root=workspace.root)
        if resume_checkpoint is None:
            checkpoint = new_checkpoint(
                command="import",
                safe_parameters=_safe_parameters(sources, _materialized_options(options, config)),
                cli_version=__version__,
                schema_version=SCHEMA_VERSION,
                target_repository=str(workspace.target),
                base_revision=base_sha,
                run_id=resume_id,
            )
        else:
            checkpoint = resume_checkpoint.model_copy(
                update={
                    "status": "running",
                    "updated_at": datetime.now(UTC).replace(microsecond=0),
                }
            )
            if checkpoint.base_revision != base_sha:
                raise CommandError(
                    "SOURCE_CHANGED_DURING_RUN",
                    "The import base no longer matches the target checkout.",
                    hint="Start a new import against the current base revision.",
                )
        staging = prepare_staging_workspace(
            workspace.root,
            target=workspace.target,
            runs_dir=runs_dir,
            run_id=checkpoint.run_id,
            base_branch=config.repository.base_branch,
            base_revision=checkpoint.base_revision,
        )
        safe_parameters = dict(checkpoint.safe_parameters)
        safe_parameters["staging_repository"] = str(staging)
        safe_parameters.setdefault("source_memberships", {})
        checkpoint = checkpoint.model_copy(update={"safe_parameters": safe_parameters})
        store.save(checkpoint)
        report_run_id(checkpoint.run_id)
        imported = 0
        failures: list[StructuredError] = []
        cache = CacheStore(
            cast(Path, config.cache_dir) / "cache-v1.sqlite3",
            repository_root=workspace.root,
        )
        try:
            async with TelegramBotAPI(
                credentials.telegram_bot_token,
                timeout_seconds=config.telegram.timeout_seconds,
                max_attempts=config.telegram.max_attempts,
                max_download_bytes=config.processing.max_download_bytes,
            ) as adapter:
                await adapter.validate_credentials()
                for source_text in sources:
                    try:
                        source = await adapter.fetch_collection(adapter.canonicalize(source_text))
                        current_members = tuple(item.native_id for item in source.items)
                        imported_members = (
                            expected_memberships.get(source.native_id)
                            if expected_memberships is not None
                            else None
                        )
                        if imported_members is not None and current_members != imported_members:
                            raise CommandError(
                                "SOURCE_CHANGED_DURING_RUN",
                                f"Collection {source.native_id} changed while resuming import.",
                                hint="Start a new import from the current Telegram collection.",
                                source=source.canonical_url,
                            )
                        if options.max_items is not None and source.item_count > options.max_items:
                            raise CommandError(
                                "CONFIG_INVALID",
                                f"Collection exceeds --max-items={options.max_items}.",
                                hint="Raise the explicit item budget and retry.",
                                source=source.canonical_url,
                            )
                        with TemporaryMediaRun(limits=_media_limits(config)) as temporary:
                            source, processed = await _prepare_collection_media(
                                initial,
                                adapter,
                                source,
                                MediaProcessor(temporary),
                                concurrency=config.telegram.download_concurrency,
                                expected_hashes=expected_hashes,
                            )
                            _cache_deterministic_analyses(cache, source, processed)
                        checkpoint = _checkpoint_media(checkpoint, source, processed)
                        memberships = _membership_map(
                            checkpoint.safe_parameters.get("source_memberships")
                        )
                        memberships[source.native_id] = current_members
                        safe_parameters = dict(checkpoint.safe_parameters)
                        safe_parameters["source_memberships"] = {
                            key: list(value) for key, value in sorted(memberships.items())
                        }
                        checkpoint = checkpoint.model_copy(
                            update={"safe_parameters": safe_parameters}
                        )
                        store.save(checkpoint)
                        imported += 1
                    except Exception as exc:
                        error = structured_exception(exc)
                        failures.append(error)
                        terminal_error = error.code in _TERMINAL_ERROR_CODES
                        checkpoint = _checkpoint_issue(
                            checkpoint,
                            error,
                            terminal=terminal_error,
                        )
                        store.save(checkpoint)
                        if options.fail_fast or terminal_error:
                            raise
            status = "succeeded" if not failures else "partial" if imported else "failed"
            checkpoint = _finish_checkpoint(checkpoint, status, 0, Decimal("0"))
            store.save(checkpoint)
            return CommandResult(
                run_id=checkpoint.run_id,
                status=RunStatus(status),
                result={"collections_imported": imported, "canonical_writes": 0},
                publication={"mode": "staging", "path": str(staging)},
                errors=failures,
            )
        except BaseException as exc:
            checkpoint = _checkpoint_issue(checkpoint, structured_exception(exc), terminal=True)
            store.save(checkpoint)
            raise
        finally:
            cache.close()


async def _run_describe(
    run_id: str,
    overrides: PipelineOptions | None = None,
) -> CommandResult:
    config = load_config()
    store = RunStore(cast(Path, config.runs_dir))
    checkpoint = store.load_for_resume(run_id, schema_version=SCHEMA_VERSION)
    report_run_id(checkpoint.run_id)
    if checkpoint.command not in {"import", "describe"}:
        raise CommandError(
            "CONFIG_INVALID",
            "The selected run cannot be described.",
            hint="Pass a run ID created by `mojilex import`.",
        )
    sources = _string_sequence(checkpoint.safe_parameters.get("sources"))
    staging = _staging_path_from_checkpoint(checkpoint)
    options = _options_from_safe(checkpoint.safe_parameters)
    if overrides is not None:
        options = replace(
            options,
            provider=overrides.provider if overrides.provider is not None else options.provider,
            model=overrides.model if overrides.model is not None else options.model,
            max_ai_requests=(
                overrides.max_ai_requests
                if overrides.max_ai_requests is not None
                else options.max_ai_requests
            ),
            max_cost_usd=(
                overrides.max_cost_usd
                if overrides.max_cost_usd is not None
                else options.max_cost_usd
            ),
            allow_unknown_cost=overrides.allow_unknown_cost or options.allow_unknown_cost,
            unknown_cost_confirmation=overrides.unknown_cost_confirmation,
        )
    options = replace(
        options,
        repository=str(staging),
        publish="local",
        direct_push=False,
        dry_run=False,
    )
    expected = {
        native_id: element.media_sha256
        for native_id, element in checkpoint.elements.items()
        if element.media_sha256
    }
    memberships = _membership_map(checkpoint.safe_parameters.get("source_memberships"))
    return await _run_add(
        sources,
        options,
        resume_id=run_id,
        expected_hashes=expected,
        expected_memberships=memberships,
        resume_checkpoint=checkpoint,
        stage_only=True,
    )


async def run_resume(
    run_id: str,
    *,
    confirmation: Callable[[str], bool] | None = None,
    unknown_cost_confirmation: Callable[[int], bool] | None = None,
) -> CommandResult:
    config = load_config()
    store = RunStore(cast(Path, config.runs_dir))
    checkpoint = store.load_for_resume(run_id, schema_version=SCHEMA_VERSION)
    report_run_id(checkpoint.run_id)
    parameters = checkpoint.safe_parameters
    sources = _string_sequence(parameters.get("sources"))
    options = _options_from_safe(parameters)
    staging_value = _optional_string(parameters.get("staging_repository"))
    use_staging = checkpoint.command in {"import", "describe"} and staging_value is not None
    options = replace(
        options,
        repository=staging_value if use_staging else checkpoint.target_repository,
        dry_run=False,
        confirmation=confirmation,
        unknown_cost_confirmation=unknown_cost_confirmation,
    )
    expected = {
        native_id: element.media_sha256
        for native_id, element in checkpoint.elements.items()
        if element.media_sha256
    }
    memberships = _membership_map(parameters.get("source_memberships"))
    if checkpoint.command == "import":
        return await _run_import(
            sources,
            options,
            resume_id=run_id,
            expected_hashes=expected,
            expected_memberships=memberships,
            resume_checkpoint=checkpoint,
        )
    return await _run_add(
        sources,
        options,
        resume_id=run_id,
        expected_hashes=expected,
        expected_memberships=memberships if use_staging else None,
        resume_checkpoint=checkpoint,
        stage_only=checkpoint.command == "describe",
    )


async def _run_submit(
    target: str | None,
    *,
    repository: str | None,
    direct_push: bool,
    base: str | None,
    confirmation: Callable[[str], bool] | None,
) -> CommandResult:
    configured = load_config()
    credentials = load_credentials()
    run_identifier = new_run_id()
    report_run_id(run_identifier)
    checkpoint: RunCheckpoint | None = None
    staged_repository: Path | None = None
    review_routing: ReviewRoutingReport | None = None
    if target and target.startswith("mlxrun_"):
        checkpoint = RunStore(cast(Path, configured.runs_dir)).load_for_resume(
            target, schema_version=SCHEMA_VERSION
        )
        if checkpoint.command != "describe" or checkpoint.status not in {"succeeded", "noop"}:
            raise CommandError(
                "CONFIG_INVALID",
                "The run does not contain a completed described change set.",
                hint="Complete `mojilex describe RUN_ID`, resolve review items, then submit.",
            )
        run_identifier = checkpoint.run_id
        report_run_id(run_identifier)
        repository = repository or checkpoint.target_repository
        staged_repository = _staging_path_from_checkpoint(checkpoint)
    elif target:
        candidate = await asyncio.to_thread(_resolve_directory, target)
        if candidate is not None:
            repository = repository or str(candidate)
        else:
            raise CommandError(
                "CONFIG_INVALID",
                "Submit target must be an existing dataset path or a run ID.",
                hint="Pass an mlxrun_ ID or a local canonical dataset checkout.",
            )
    config = load_config(
        cli={
            "repository": {
                "target": repository,
                "base_branch": base,
                "publish": "pr" if direct_push else None,
            }
        }
    )
    if staged_repository is not None:
        staged_report = validate_dataset(staged_repository, strict=True)
        staged_report.raise_for_errors()
        staged_snapshot = load_dataset(staged_repository)
        review_routing = official_submission_report(staged_snapshot)
        if config.repository.publish == "local" and not direct_push:
            with snapshot_at_revision(
                staged_repository, cast(RunCheckpoint, checkpoint).base_revision
            ) as base_root:
                changed = _changed_paths(load_dataset(base_root), load_dataset(staged_repository))
            return CommandResult(
                run_id=run_identifier,
                status=RunStatus.NOOP if not changed else RunStatus.SUCCEEDED,
                result={
                    "changed_paths": [str(path) for path in changed],
                    "validated": True,
                    "review_routing": review_routing.as_dict(),
                },
                publication={"mode": "staging", "path": str(staged_repository)},
            )

    with repository_workspace(
        config.repository.target,
        config.repository.base_branch,
        isolated=staged_repository is not None,
    ) as workspace:
        git = GitRunner(workspace.root, github_token=credentials.github_token)
        git_publisher = GitPublisher(git)
        guard: Any | None = None
        if staged_repository is not None:
            latest_report = validate_dataset(workspace.root, strict=True)
            latest_report.raise_for_errors()
            latest = load_dataset(workspace.root)
            candidate_snapshot = load_dataset(staged_repository)
            assert checkpoint is not None
            with snapshot_at_revision(staged_repository, checkpoint.base_revision) as base_root:
                imported_base = load_dataset(base_root)
                merged = reapply_candidate(imported_base, candidate_snapshot, latest)
            merged_report = validate_snapshot(merged, schemas=True, repository_files=True)
            merged_report.raise_for_errors()
            review_routing = official_submission_report(merged)
            calculated_paths = _changed_paths(latest, merged)
            if any(not _submit_path_allowed(str(path)) for path in calculated_paths):
                raise CommandError(
                    "DIRTY_WORKTREE",
                    "The staged run contains changes outside canonical data paths.",
                    hint=(
                        "Start a clean import; run workspaces must only alter data/ or tombstones/."
                    ),
                    details={"paths": [str(path) for path in calculated_paths]},
                )
            if calculated_paths:
                guard = git_publisher.guard_targets(tuple(str(path) for path in calculated_paths))
                _apply_with_rollback(latest, merged)
            paths = tuple(str(path) for path in calculated_paths)
        else:
            report = validate_dataset(workspace.root, strict=True)
            report.raise_for_errors()
            review_routing = official_submission_report(load_dataset(workspace.root))
            paths = tuple(dict.fromkeys(git.status_paths()))
        assert review_routing is not None
        if not paths:
            return CommandResult(
                run_id=run_identifier,
                status=RunStatus.NOOP,
                result={
                    "changed_paths": [],
                    "review_routing": review_routing.as_dict(),
                },
                publication={"mode": config.repository.publish},
            )
        if any(not _submit_path_allowed(path) for path in paths):
            raise CommandError(
                "DIRTY_WORKTREE",
                "Submit found changes outside canonical generated data paths.",
                hint="Commit or move unrelated changes, then submit only data/ and tombstones/.",
                details={"paths": list(paths)},
            )
        if config.repository.publish == "local" and not direct_push:
            return CommandResult(
                run_id=run_identifier,
                result={
                    "changed_paths": list(paths),
                    "validated": True,
                    "review_routing": review_routing.as_dict(),
                },
                publication={"mode": "local", "path": str(workspace.root)},
            )
        base_sha = git.current_sha()
        branch = make_import_branch(pack_name=None, run_id=run_identifier, batch=True)
        if guard is not None:
            prepared = git_publisher.prepare_commit(
                paths=paths,
                message="data: submit validated MojiLex change set",
                branch=branch,
                base_revision="HEAD",
                identity=_configured_git_identity(config),
                guard=guard,
            )
            if prepared is None:
                return CommandResult(run_id=run_identifier, status=RunStatus.NOOP)
            staged = prepared.paths
            commit_sha = prepared.commit_sha
        else:
            git.create_branch(branch, base_sha)
            staged = git.stage_paths(paths)
            if not git.has_staged_changes():
                return CommandResult(run_id=run_identifier, status=RunStatus.NOOP)
            commit_message = "data: submit validated MojiLex change set"
            commit_identity = git.resolve_identity(_configured_git_identity(config))
            commit_sha = git.commit(
                commit_message,
                identity=commit_identity,
            )
            prepared = PreparedCommit(
                branch=branch,
                base_sha=base_sha,
                commit_sha=commit_sha,
                paths=staged,
                message=commit_message,
                identity=commit_identity,
            )
        if not staged:
            return CommandResult(run_id=run_identifier, status=RunStatus.NOOP)
        github = GitHubCLI(token=credentials.github_token)
        github.auth_status()
        info = github.repository_info(workspace.target)
        github_publisher = GitHubPublisher(git, github)
        publication: dict[str, Any]
        fork = workspace.target
        remote = "origin"
        if not direct_push:
            fork = workspace.target if info.can_write else github.ensure_fork(workspace.target)
            if fork != workspace.target:
                remote = "mojilex-fork"
                remotes = git.run("remote", check=False).stdout.splitlines()
                if remote not in remotes:
                    git.run("remote", "add", remote, f"https://github.com/{fork}.git")
        prepared = git_publisher.reconcile_remote_branch(
            prepared,
            remote=remote,
            path_is_allowed=_submit_path_allowed,
        )
        commit_sha = prepared.commit_sha
        if direct_push:
            required = github.required_checks(workspace.target, config.repository.base_branch)
            bypass = github.has_direct_push_bypass(workspace.target, config.repository.base_branch)
            user_confirmed = _confirm_direct_push(
                confirmation,
                git,
                prepared,
                target=workspace.target,
                base_branch=config.repository.base_branch,
            )
            published_sha = await github_publisher.publish_direct(
                prepared,
                target=workspace.target,
                repository_info=info,
                remote="origin",
                base_branch=config.repository.base_branch,
                candidate_branch=branch,
                required_checks=required,
                authorization=DirectPushAuthorization(
                    explicit_flag=True,
                    user_confirmed=user_confirmed,
                    validation_succeeded=True,
                    bypass_verified=bypass,
                ),
            )
            publication = {
                "mode": "direct",
                "commit": published_sha,
                "candidate_branch": branch,
            }
        else:
            pull_request = github_publisher.publish_pr(
                prepared,
                target=workspace.target,
                fork=fork,
                fork_remote=remote,
                base_branch=config.repository.base_branch,
                title="data: submit validated MojiLex change set",
                body=(
                    "Validated MojiLex metadata change set. No source media, download URLs, "
                    f"or secrets are included.\n\nRun: `{run_identifier}`\n"
                ),
            )
            publication = {
                "mode": "pr",
                "pull_request_url": pull_request.url,
                "pull_request_number": pull_request.number,
                "reused": pull_request.reused,
                "completed": pull_request.completed,
                "reopened": pull_request.reopened,
                "commit": commit_sha,
            }
        return CommandResult(
            run_id=run_identifier,
            result={
                "changed_paths": list(staged),
                "validated": True,
                "review_routing": review_routing.as_dict(),
            },
            publication=publication,
        )


def _submit_path_allowed(path: str) -> bool:
    normalized = PurePosixPath(path.replace("\\", "/"))
    return bool(normalized.parts) and normalized.parts[0] in {"data", "tombstones"}


def _resolve_directory(value: str) -> Path | None:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_dir() else None


def _resolved_config(options: PipelineOptions) -> MojiLexConfig:
    layer: dict[str, Any] = {
        "repository": {
            "target": options.repository,
            "base_branch": options.base,
            "publish": options.publish,
        },
        "telegram": {"download_concurrency": options.download_concurrency},
        "ai": {
            "provider": options.provider,
            "model": options.model,
            "languages": options.languages or None,
            "max_ai_requests": options.max_ai_requests,
            "max_cost_usd": options.max_cost_usd,
            "allow_unknown_cost": options.allow_unknown_cost or None,
            "ai_concurrency": options.ai_concurrency,
            "model_routing": options.model_routing,
            "escalation_model": options.escalation_model,
        },
        "dedupe": {
            "mode": options.dedupe,
            "max_candidates": options.max_dedupe_candidates,
            "profile": options.dedupe_profile,
        },
    }
    return load_config(cli=layer)


def _validate_options(options: PipelineOptions, config: MojiLexConfig) -> None:
    if options.platform not in {"auto", "telegram"}:
        raise ValueError("--platform must be auto or telegram")
    if options.redescribe not in {"missing", "changed", "all"}:
        raise ValueError("--redescribe must be missing, changed, or all")
    if options.new_identity and options.same_identity:
        raise ValueError("--new-identity and --same-identity are mutually exclusive")
    if config.repository.publish not in {"local", "pr"}:
        raise ValueError("--publish must be local or pr")
    if options.direct_push and config.repository.publish == "local":
        raise ValueError("--direct-push cannot be combined with --publish local")
    if set(config.ai.languages) != {"ru", "en"}:
        raise ValueError("the MVP requires exactly ru and en")
    if config.dedupe.profile != "dedupe-v1":
        raise ValueError("only the immutable dedupe-v1 profile is supported")


def _validate_sources_for_platform(sources: Sequence[str], platform: str) -> None:
    if platform == "auto" and any("://" not in source for source in sources):
        raise CommandError(
            "SOURCE_UNSUPPORTED",
            "A bare source name is ambiguous when --platform is auto.",
            hint="Pass a canonical addemoji URL or set --platform telegram explicitly.",
        )


def _media_limits(config: MojiLexConfig) -> MediaLimits:
    return MediaLimits(
        max_file_bytes=config.processing.max_download_bytes,
        frames=config.processing.keyframes,
        worker_timeout_seconds=config.processing.render_timeout_seconds,
        max_run_temp_bytes=config.processing.max_temp_bytes,
    )


async def _prepare_collection_media(
    snapshot: DatasetSnapshot,
    adapter: TelegramBotAPI,
    collection: SourceCollection,
    processor: MediaProcessor,
    *,
    concurrency: int,
    expected_hashes: Mapping[str, tuple[str, ...]] | None = None,
    cache: CacheStore | None = None,
    resume_elements: Mapping[str, ElementCheckpoint] | None = None,
    config: MojiLexConfig | None = None,
    taxonomy_version: str | None = None,
    cache_alias_scope: str | None = None,
    redescribe: str = "changed",
    overwrite_reviewed: bool = False,
    verified_semantic_outcomes: dict[str, _SemanticOutcome] | None = None,
    on_item_completed: _MediaCompletion | None = None,
) -> tuple[SourceCollection, dict[str, ProcessedMedia]]:
    """Download media and reconcile global IDs already seen in another collection."""

    guarded = _cross_collection_existing_emojis(snapshot, collection)
    force_direct_ids = {
        native_id
        for native_id, existing in guarded.items()
        if resume_elements is not None
        and (element := resume_elements.get(native_id)) is not None
        and element.media_sha256
        and element.media_sha256 != _emoji_media_hashes(existing)
    }
    cached, cached_outcomes, cached_analysis = await _resume_cached_processed_media(
        snapshot,
        collection,
        processor,
        cache=cache,
        resume_elements=resume_elements,
        config=config,
        taxonomy_version=taxonomy_version,
        cache_alias_scope=cache_alias_scope,
        # Existing global identities require the full reported/direct/canonical
        # reconciliation path. A hash-only resume shortcut must never intercept
        # a changed reported file before that three-way check can run.
        forbidden_native_ids=set(guarded),
        redescribe=redescribe,
        overwrite_reviewed=overwrite_reviewed,
    )
    if verified_semantic_outcomes is not None:
        verified_semantic_outcomes.clear()
        verified_semantic_outcomes.update(cached_outcomes)

    async def record_unguarded(item: SourceEmoji, value: ProcessedMedia) -> None:
        if item.native_id in guarded or on_item_completed is None:
            return
        _verify_expected_media_hash(item.native_id, value, expected_hashes)
        await on_item_completed(item, value)

    processed = await _process_media(
        adapter,
        collection,
        processor,
        concurrency=concurrency,
        expected_hashes=expected_hashes,
        defer_expected_ids=set(guarded),
        resume_cached=cached,
        resume_analysis=cached_analysis,
        on_item_completed=record_unguarded,
    )
    collection, processed = await _reconcile_cross_collection_media(
        adapter,
        collection,
        processor,
        processed,
        guarded,
        concurrency=concurrency,
        force_direct_ids=force_direct_ids,
        expected_hashes=expected_hashes,
        on_item_completed=on_item_completed,
    )
    collection = _canonicalize_guarded_source_items(collection, guarded, processed)
    _verify_expected_media_hashes(processed, expected_hashes)
    return collection, processed


async def _resume_cached_processed_media(
    snapshot: DatasetSnapshot,
    collection: SourceCollection,
    processor: MediaProcessor,
    *,
    cache: CacheStore | None,
    resume_elements: Mapping[str, ElementCheckpoint] | None,
    config: MojiLexConfig | None,
    taxonomy_version: str | None,
    cache_alias_scope: str | None,
    forbidden_native_ids: set[str],
    redescribe: str,
    overwrite_reviewed: bool,
) -> tuple[
    dict[str, ProcessedMedia],
    dict[str, _SemanticOutcome],
    dict[str, ProcessedMedia],
]:
    if cache is None or resume_elements is None or config is None or taxonomy_version is None:
        return {}, {}, {}
    qualifications = ModelQualificationRegistry.load(snapshot.root)
    routing_registry = RoutingReasonRegistry.load(snapshot.root)
    restored: dict[str, ProcessedMedia] = {}
    semantic_outcomes: dict[str, _SemanticOutcome] = {}
    candidates: dict[str, ProcessedMedia] = {}
    traces = _request_traces_from_checkpoint(collection.items, resume_elements)
    for item in collection.items:
        element = resume_elements.get(item.native_id)
        if (
            element is None
            or item.native_id in forbidden_native_ids
            or element.source_descriptor_sha256 != _source_descriptor_sha256(item)
            or len(element.media_sha256) != 1
            or element.deterministic_cache_key is None
            or not element.palette_complete
            or not element.fingerprint_complete
        ):
            continue
        candidate = _restore_deterministic_cache_entry(
            cache,
            item,
            processor,
            element,
        )
        if candidate is None:
            continue
        candidates[item.native_id] = candidate
        if not _needs_generated_description(
            snapshot,
            collection.platform,
            item,
            candidate,
            redescribe=redescribe,
            overwrite_reviewed=overwrite_reviewed,
        ):
            restored[item.native_id] = candidate
            continue
        if (
            element.ai_cache_key is None
            or not element.ai_facets_complete
            or not element.ai_requests
        ):
            continue
    # Batch membership is decided before cache lookup. If even one item's
    # deterministic context is unavailable, none of the AI chunks can be
    # reconstructed without risking a different neighbour set.
    if len(candidates) != len(collection.items):
        return (
            restored,
            semantic_outcomes,
            {
                native_id: candidate
                for native_id, candidate in candidates.items()
                if native_id not in restored
            },
        )
    ai_items = [
        item
        for item in collection.items
        if _needs_generated_description(
            snapshot,
            collection.platform,
            item,
            candidates[item.native_id],
            redescribe=redescribe,
            overwrite_reviewed=overwrite_reviewed,
        )
    ]
    for chunk in _description_chunks(ai_items, candidates, config):
        chunk_traces = _recover_ai_request_traces(
            cache,
            chunk,
            candidates,
            config=config,
            taxonomy_version=taxonomy_version,
            cache_alias_scope=cache_alias_scope,
            checkpoint_traces=traces,
        )
        if any(item.native_id not in chunk_traces for item in chunk):
            continue
        try:
            outcomes = await _describe_batch(
                chunk,
                candidates,
                config=config,
                cache=cache,
                budget=RequestBudget(max_requests=0),
                ai_state=_AIState(),
                api_key=None,
                temporary=TemporaryMediaRun(),
                taxonomy_version=taxonomy_version,
                qualifications=qualifications,
                routing_registry=routing_registry,
                cache_alias_scope=cache_alias_scope,
                resume_request_traces=chunk_traces,
                require_exact_resume=True,
            )
        except (AIError, CacheError, CommandError, ValueError):
            continue
        if set(outcomes) != {item.native_id for item in chunk}:
            continue
        for item in chunk:
            restored[item.native_id] = candidates[item.native_id]
            semantic_outcomes[item.native_id] = outcomes[item.native_id]
    return (
        restored,
        semantic_outcomes,
        {
            native_id: candidate
            for native_id, candidate in candidates.items()
            if native_id not in restored
        },
    )


def _request_traces_from_checkpoint(
    items: Sequence[SourceEmoji],
    elements: Mapping[str, ElementCheckpoint],
) -> dict[str, tuple[_AICacheTrace, ...]]:
    traces: dict[str, tuple[_AICacheTrace, ...]] = {}
    for item in items:
        element = elements.get(item.native_id)
        if element is None or not element.ai_requests:
            continue
        traces[item.native_id] = tuple(
            _AICacheTrace(
                stage=request.stage,
                model=request.model,
                model_revision=request.model_revision,
                cache_key=request.cache_key,
                request_identity=_AIRequestIdentity(
                    plan_sha256=request.plan_sha256,
                    request_sha256=request.request_sha256,
                    shown_media_sha256=request.shown_media_sha256,
                    labels_by_native=((item.native_id, request.item_label),),
                ),
            )
            for request in element.ai_requests
        )
    return traces


def _load_ai_request_envelope(
    cache: CacheStore,
    items: Sequence[SourceEmoji],
    processed: Mapping[str, ProcessedMedia],
    *,
    config: MojiLexConfig,
    taxonomy_version: str,
    cache_alias_scope: str | None,
    model: str,
    stage: Literal["primary", "escalated"],
) -> dict[str, tuple[_AICacheTrace, ...]] | None:
    if not items:
        return None
    plan_sha256 = _ai_request_plan_sha256(items, processed, model=model)
    envelope_key = _ai_request_envelope_key(
        cache_alias_scope,
        stage=stage,
        model=model,
        plan_sha256=plan_sha256,
    )
    if envelope_key is None:
        return None
    try:
        payload = cache.get_metadata(envelope_key)
        if payload is None or set(payload) != {"format_version", "items"}:
            return None
        raw_items = payload["items"]
        if payload["format_version"] != 1 or not isinstance(raw_items, list):
            return None
        if len(raw_items) != len(items) or len(raw_items) > 16:
            return None
        traces: dict[str, tuple[_AICacheTrace, ...]] = {}
        requests: dict[str, AIRequestCheckpoint] = {}
        for item, raw in zip(items, raw_items, strict=True):
            if not isinstance(raw, Mapping) or set(raw) != {"native_id", "request"}:
                return None
            if raw["native_id"] != item.native_id:
                return None
            request = AIRequestCheckpoint.model_validate(raw["request"])
            if (
                request.stage != stage
                or request.model != model
                or request.plan_sha256 != plan_sha256
            ):
                return None
            requests[item.native_id] = request
            traces[item.native_id] = (
                _AICacheTrace(
                    stage=request.stage,
                    model=request.model,
                    model_revision=request.model_revision,
                    cache_key=request.cache_key,
                    request_identity=_AIRequestIdentity(
                        plan_sha256=request.plan_sha256,
                        request_sha256=request.request_sha256,
                        shown_media_sha256=request.shown_media_sha256,
                        labels_by_native=((item.native_id, request.item_label),),
                    ),
                ),
            )
        identity = _resume_request_identity(
            items,
            processed,
            traces,
            model=model,
            stage=stage,
        )
        if identity is None:
            return None
        for item in items:
            request = requests[item.native_id]
            cached = _load_cached_description(
                cache,
                request.cache_key,
                source=item,
                processed=processed[item.native_id],
                context=_vision_context(item, processed[item.native_id]),
                config=config,
                model=model,
                taxonomy_version=taxonomy_version,
                resume_cache_key=None,
                cache_alias_scope=None,
                request_identity=identity,
                item_label=request.item_label,
            )
            if (
                cached is None
                or cached.result.model != request.model
                or cached.result.model_revision != request.model_revision
            ):
                return None
        return traces
    except (AIError, CacheError, TypeError, ValueError):
        return None


def _recover_ai_request_traces(
    cache: CacheStore,
    items: Sequence[SourceEmoji],
    processed: Mapping[str, ProcessedMedia],
    *,
    config: MojiLexConfig,
    taxonomy_version: str,
    cache_alias_scope: str | None,
    checkpoint_traces: Mapping[str, Sequence[_AICacheTrace]] | None,
) -> dict[str, tuple[_AICacheTrace, ...]]:
    recovered = {
        native_id: tuple(traces) for native_id, traces in (checkpoint_traces or {}).items()
    }

    def merge(candidate: Mapping[str, Sequence[_AICacheTrace]] | None) -> None:
        if candidate is None:
            return
        for native_id, additions in candidate.items():
            current = list(recovered.get(native_id, ()))
            for addition in additions:
                # Envelope traces have just been checked against the exact
                # current plan and every immutable result row. They therefore
                # supersede an untrusted/stale checkpoint trace for the same
                # request stage instead of being shadowed by it.
                current = [
                    trace
                    for trace in current
                    if not (trace.stage == addition.stage and trace.model == addition.model)
                ]
                current.append(addition)
            current.sort(key=lambda trace: 0 if trace.stage == "primary" else 1)
            recovered[native_id] = tuple(current)

    primary_items: list[SourceEmoji] = []
    for item in items:
        reasons = deterministic_routing_reasons(processed[item.native_id])
        if should_escalate(
            config.ai.model_routing,
            reasons,
            escalation_model=config.ai.escalation_model,
        ):
            merge(
                _load_ai_request_envelope(
                    cache,
                    (item,),
                    processed,
                    config=config,
                    taxonomy_version=taxonomy_version,
                    cache_alias_scope=cache_alias_scope,
                    model=config.ai.escalation_model,
                    stage="escalated",
                )
            )
        else:
            primary_items.append(item)
    if primary_items:
        primary = _load_ai_request_envelope(
            cache,
            primary_items,
            processed,
            config=config,
            taxonomy_version=taxonomy_version,
            cache_alias_scope=cache_alias_scope,
            model=config.ai.model,
            stage="primary",
        )
        if primary is None:
            for item in primary_items:
                merge(
                    _load_ai_request_envelope(
                        cache,
                        (item,),
                        processed,
                        config=config,
                        taxonomy_version=taxonomy_version,
                        cache_alias_scope=cache_alias_scope,
                        model=config.ai.model,
                        stage="primary",
                    )
                )
        else:
            merge(primary)
        for item in primary_items:
            merge(
                _load_ai_request_envelope(
                    cache,
                    (item,),
                    processed,
                    config=config,
                    taxonomy_version=taxonomy_version,
                    cache_alias_scope=cache_alias_scope,
                    model=config.ai.escalation_model,
                    stage="escalated",
                )
            )
    return recovered


def _restore_deterministic_cache_entry(
    cache: CacheStore,
    source: SourceEmoji,
    processor: MediaProcessor,
    checkpoint: ElementCheckpoint,
) -> ProcessedMedia | None:
    key = checkpoint.deterministic_cache_key
    descriptor_sha256 = checkpoint.source_descriptor_sha256
    if key is None or descriptor_sha256 is None or len(checkpoint.media_sha256) != 1:
        return None
    try:
        render_context_sha256 = _deterministic_render_context_sha256(source)
        payload = cache.get_metadata(_deterministic_cache_storage_key(key, render_context_sha256))
        if payload is None or set(payload) != {
            "analysis",
            "deterministic_cache_key",
            "format_version",
            "metadata",
            "pipeline_version",
            "render_context",
            "render_context_sha256",
        }:
            return None
        if (
            payload["format_version"] != 3
            or payload["pipeline_version"] != PIPELINE_VERSION
            or payload["deterministic_cache_key"] != key
            or payload["render_context_sha256"] != render_context_sha256
            or descriptor_sha256 != _source_descriptor_sha256(source)
        ):
            return None
        metadata = MediaMetadata.model_validate(payload["metadata"])
        analysis = DeterministicMediaAnalysis.model_validate(payload["analysis"])
        render_context = payload["render_context"]
        if not isinstance(render_context, Mapping):
            return None
        if set(render_context) != {"background_variants", "frame_count"}:
            return None
        frame_count = render_context["frame_count"]
        backgrounds = render_context["background_variants"]
        if (
            not isinstance(frame_count, int)
            or isinstance(frame_count, bool)
            or not isinstance(backgrounds, list)
            or backgrounds not in (["light"], ["light", "dark"])
        ):
            return None
        expected_frames = 1 if source.media_format == "webp" else processor.worker.limits.frames
        expected_animated = source.media_format != "webp"
        if (
            frame_count != expected_frames
            or metadata.sha256 != checkpoint.media_sha256[0]
            or metadata.format != source.media_format
            or metadata.width != source.width
            or metadata.height != source.height
            or metadata.animated is not expected_animated
            or (
                source.declared_file_size is not None
                and metadata.byte_size != source.declared_file_size
            )
            or (source.needs_repainting and backgrounds != ["light", "dark"])
        ):
            return None
        color_profile = load_analysis_profile("color-v1")
        dedupe_profile = load_analysis_profile("dedupe-v1")
        if (
            analysis.color_profile != color_profile.profile_id
            or analysis.color_profile_sha256 != color_profile.sha256
            or analysis.dedupe_profile != dedupe_profile.profile_id
            or analysis.dedupe_profile_sha256 != dedupe_profile.sha256
            or not _decoder_backend_is_current(source.media_format, analysis, processor)
        ):
            return None
        restored = ProcessedMedia(
            metadata=metadata,
            analysis=analysis,
            frame_paths=(),
            dark_frame_paths=(),
            rendered_frame_count=frame_count,
            has_dark_render=backgrounds == ["light", "dark"],
        )
        if _deterministic_key(restored) != key:
            return None
        return restored
    except (AnalysisError, CacheError, TypeError, ValueError):
        return None


def _decoder_backend_is_current(
    media_format: str,
    analysis: DeterministicMediaAnalysis,
    processor: MediaProcessor,
) -> bool:
    worker = processor.worker
    candidates: tuple[str, ...]
    if media_format == "webp":
        candidates = (decoder_backend_fingerprint("webp"),)
    elif media_format == "tgs":
        candidates = (decoder_backend_fingerprint("tgs", rlottie_renderer=worker.rlottie_renderer),)
    elif media_format == "webm":
        candidates = tuple(
            decoder_backend_fingerprint(
                "webm",
                ffmpeg=worker.ffmpeg,
                ffprobe=worker.ffprobe,
                webm_codec=codec,
                webm_preserve_alpha=preserve_alpha,
            )
            for codec in ("av1", "vp8", "vp9")
            for preserve_alpha in (False, True)
        )
    else:
        return False
    return analysis.decoder_backend_fingerprint in candidates


async def _process_media(
    adapter: TelegramBotAPI,
    collection: SourceCollection,
    processor: MediaProcessor,
    *,
    concurrency: int,
    expected_hashes: Mapping[str, tuple[str, ...]] | None = None,
    defer_expected_ids: set[str] | None = None,
    resume_cached: Mapping[str, ProcessedMedia] | None = None,
    resume_analysis: Mapping[str, ProcessedMedia] | None = None,
    on_item_completed: _MediaCompletion | None = None,
) -> dict[str, ProcessedMedia]:
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(item: SourceEmoji) -> tuple[str, ProcessedMedia]:
        expected = expected_hashes.get(item.native_id) if expected_hashes else None
        if defer_expected_ids is not None and item.native_id in defer_expected_ids:
            expected = None
        async with semaphore:
            cached = resume_cached.get(item.native_id) if resume_cached is not None else None
            analysis = resume_analysis.get(item.native_id) if resume_analysis is not None else None
            if cached is not None:
                digest, size = await processor.verify_stream(
                    adapter.fetch_media(item),
                    declared_size=item.declared_file_size,
                    expected_sha256=cached.metadata.sha256,
                )
                if digest != cached.metadata.sha256 or size != cached.metadata.byte_size:
                    raise SourceChangedDuringRunError(
                        "downloaded media no longer matches the deterministic cache"
                    )
                value = cached
            elif analysis is not None:
                value = await processor.process_stream_reusing_analysis(
                    adapter.fetch_media(item),
                    expected_format=item.media_format,
                    declared_size=item.declared_file_size,
                    cached=analysis,
                    needs_repainting=item.needs_repainting,
                )
            else:
                value = await processor.process_stream(
                    adapter.fetch_media(item),
                    expected_format=item.media_format,
                    declared_size=item.declared_file_size,
                    expected_sha256=expected[0] if expected and len(expected) == 1 else None,
                    needs_repainting=item.needs_repainting,
                )
            if on_item_completed is not None:
                await on_item_completed(item, value)
        return item.native_id, value

    outcomes = await asyncio.gather(
        *(process_one(item) for item in collection.items), return_exceptions=True
    )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    return dict(cast(list[tuple[str, ProcessedMedia]], outcomes))


def _cross_collection_existing_emojis(
    snapshot: DatasetSnapshot,
    source: SourceCollection,
) -> dict[str, Emoji]:
    source_key = (
        source.platform,
        source.native_namespace,
        source.scope_id,
        source.native_id,
    )
    memberships_by_emoji: dict[str, set[str]] = {}
    for membership in snapshot.memberships.values():
        memberships_by_emoji.setdefault(membership.emoji_id, set()).add(membership.collection_id)

    result: dict[str, Emoji] = {}
    for item in source.items:
        existing = _existing_emoji(snapshot, source.platform, item.native_id)
        if existing is None:
            continue
        for collection_id in memberships_by_emoji.get(existing.id, ()):
            known_collection = snapshot.collections.get(collection_id)
            if known_collection is None:
                continue
            known_key = (
                known_collection.platform,
                known_collection.native_namespace,
                known_collection.scope_id,
                known_collection.native_id,
            )
            if known_key != source_key:
                result[item.native_id] = existing
                break
    return result


async def _reconcile_cross_collection_media(
    adapter: TelegramBotAPI,
    collection: SourceCollection,
    processor: MediaProcessor,
    processed: Mapping[str, ProcessedMedia],
    guarded: Mapping[str, Emoji],
    *,
    concurrency: int,
    force_direct_ids: set[str] | None = None,
    expected_hashes: Mapping[str, tuple[str, ...]] | None = None,
    on_item_completed: _MediaCompletion | None = None,
) -> tuple[SourceCollection, dict[str, ProcessedMedia]]:
    originals = {item.native_id: item for item in collection.items}
    conflicts = tuple(
        sorted(
            {
                native_id
                for native_id, existing in guarded.items()
                if _processed_media_hashes(processed[native_id]) != _emoji_media_hashes(existing)
            }
            | (force_direct_ids or set())
        )
    )

    async def record_guarded(item: SourceEmoji, value: ProcessedMedia) -> None:
        if on_item_completed is None:
            return
        _verify_expected_media_hash(item.native_id, value, expected_hashes)
        canonical = _canonicalize_guarded_source_item(
            collection.platform,
            item,
            guarded[item.native_id],
            value,
        )
        await on_item_completed(canonical, value)

    for native_id in guarded.keys() - set(conflicts):
        await record_guarded(originals[native_id], processed[native_id])
    if not conflicts:
        return collection, dict(processed)

    direct = await adapter.fetch_emojis(conflicts)
    if set(direct) != set(conflicts):
        raise IdentityConflictError(
            "Telegram could not reconcile an existing custom emoji identity directly."
        )
    direct_items: list[SourceEmoji] = []
    for native_id in conflicts:
        item = direct[native_id]
        original = originals[native_id]
        if (
            item.native_id,
            item.native_namespace,
            item.scope_id,
        ) != (
            original.native_id,
            original.native_namespace,
            original.scope_id,
        ):
            raise IdentityConflictError(
                "Telegram returned inconsistent identity metadata for a direct emoji lookup."
            )
        direct_items.append(item.model_copy(update={"position": original.position}))

    direct_collection = collection.model_copy(
        update={"items": tuple(direct_items), "item_count": len(direct_items)}
    )

    async def record_direct(item: SourceEmoji, value: ProcessedMedia) -> None:
        direct_hashes = _processed_media_hashes(value)
        reported_hashes = _processed_media_hashes(processed[item.native_id])
        existing_hashes = _emoji_media_hashes(guarded[item.native_id])
        if direct_hashes not in {reported_hashes, existing_hashes}:
            raise IdentityConflictError(
                "Telegram direct lookup did not resolve an existing custom emoji media mismatch."
            )
        await record_guarded(item, value)

    direct_processed = await _process_media(
        adapter,
        direct_collection,
        processor,
        concurrency=concurrency,
        on_item_completed=record_direct,
    )
    for native_id in conflicts:
        direct_hashes = _processed_media_hashes(direct_processed[native_id])
        reported_hashes = _processed_media_hashes(processed[native_id])
        existing_hashes = _emoji_media_hashes(guarded[native_id])
        if direct_hashes not in {reported_hashes, existing_hashes}:
            raise IdentityConflictError(
                "Telegram direct lookup did not resolve an existing custom emoji media mismatch."
            )

    reconciled_items = tuple(
        direct[item.native_id].model_copy(update={"position": item.position})
        if item.native_id in direct
        else item
        for item in collection.items
    )
    extension = dict(collection.extension)
    extension["set_fingerprint_sha256"] = telegram_set_fingerprint(reconciled_items)
    reconciled_collection = collection.model_copy(
        update={"items": reconciled_items, "extension": extension}
    )
    reconciled_processed = dict(processed)
    reconciled_processed.update(direct_processed)
    return reconciled_collection, reconciled_processed


def _verify_expected_media_hashes(
    processed: Mapping[str, ProcessedMedia],
    expected_hashes: Mapping[str, tuple[str, ...]] | None,
) -> None:
    if expected_hashes is None:
        return
    for native_id, value in processed.items():
        _verify_expected_media_hash(native_id, value, expected_hashes)


def _verify_expected_media_hash(
    native_id: str,
    value: ProcessedMedia,
    expected_hashes: Mapping[str, tuple[str, ...]] | None,
) -> None:
    expected = expected_hashes.get(native_id) if expected_hashes is not None else None
    if expected is not None and _processed_media_hashes(value) != expected:
        raise SourceChangedDuringRunError(
            "downloaded media SHA-256 no longer matches the staging run"
        )


def _canonicalize_guarded_source_items(
    collection: SourceCollection,
    guarded: Mapping[str, Emoji],
    processed: Mapping[str, ProcessedMedia],
) -> SourceCollection:
    """Keep a known global emoji record stable when only a membership is new."""

    changed = False
    items: list[SourceEmoji] = []
    for item in collection.items:
        existing = guarded.get(item.native_id)
        if existing is None:
            items.append(item)
            continue
        canonical = _canonicalize_guarded_source_item(
            collection.platform,
            item,
            existing,
            processed[item.native_id],
        )
        items.append(canonical)
        changed = changed or canonical != item
    if not changed:
        return collection
    normalized_items = tuple(items)
    extension = dict(collection.extension)
    extension["set_fingerprint_sha256"] = telegram_set_fingerprint(normalized_items)
    return collection.model_copy(update={"items": normalized_items, "extension": extension})


def _canonicalize_guarded_source_item(
    platform: str,
    item: SourceEmoji,
    existing: Emoji,
    processed: ProcessedMedia,
) -> SourceEmoji:
    if _processed_media_hashes(processed) != _emoji_media_hashes(existing):
        return item
    media = existing.media[0]
    extension = existing.extensions.get(platform, {})
    return item.model_copy(
        update={
            "file_unique_id": extension.get("file_unique_id", item.file_unique_id),
            "width": media.width,
            "height": media.height,
            "animated": media.format.value == "tgs",
            "video": media.format.value == "webm",
            "needs_repainting": extension.get("needs_repainting") is True,
            "fallback_emoji": extension.get("fallback_emoji"),
            "declared_file_size": media.byte_size,
            "media_format": media.format.value,
        }
    )


def _processed_media_hashes(processed: ProcessedMedia) -> tuple[str, ...]:
    return (processed.metadata.sha256,)


def _emoji_media_hashes(emoji: Emoji) -> tuple[str, ...]:
    return tuple(sorted(media.sha256 for media in emoji.media))


async def _descriptions_for_collection(
    snapshot: DatasetSnapshot,
    source: SourceCollection,
    processed: Mapping[str, ProcessedMedia],
    *,
    config: MojiLexConfig,
    cache: CacheStore,
    budget: RequestBudget,
    ai_state: _AIState,
    api_key: str | None,
    redescribe: str,
    overwrite_reviewed: bool,
    temporary: TemporaryMediaRun,
    resume_ai_cache_keys: Mapping[str, str] | None = None,
    cache_alias_scope: str | None = None,
    verified_resume_outcomes: Mapping[str, _SemanticOutcome] | None = None,
    request_traces_out: dict[str, tuple[_AICacheTrace, ...]] | None = None,
    resume_request_traces: Mapping[str, Sequence[_AICacheTrace]] | None = None,
    on_chunk_completed: _AIChunkCompletion | None = None,
) -> tuple[dict[str, DescriptionItem], dict[str, SemanticGenerationMetadata]]:
    outcomes_by_native: dict[str, _SemanticOutcome] = {}
    candidates: list[SourceEmoji] = []
    qualifications = ModelQualificationRegistry.load(snapshot.root)
    routing_registry = RoutingReasonRegistry.load(snapshot.root)
    for item in source.items:
        existing = _existing_emoji(snapshot, source.platform, item.native_id)
        should_generate = _needs_generated_description(
            snapshot,
            source.platform,
            item,
            processed[item.native_id],
            redescribe=redescribe,
            overwrite_reviewed=overwrite_reviewed,
        )
        if not should_generate and existing is not None:
            outcomes_by_native[item.native_id] = _SemanticOutcome(
                description=_description_from_existing(existing),
                generation=_generation_from_existing(existing, config),
            )
        elif verified_resume_outcomes is not None and item.native_id in verified_resume_outcomes:
            outcomes_by_native[item.native_id] = verified_resume_outcomes[item.native_id]
            ai_state.cache_hits += 1
        else:
            candidates.append(item)
    chunks = _description_chunks(candidates, processed, config)
    report_progress(
        f"AI plan: {len(candidates)} item(s), {len(chunks)} candidate batch(es), "
        f"provider={config.ai.provider}, model={config.ai.model}. Exact cache hits can reduce "
        f"requests; retries and escalation share the {budget.max_requests}-request limit."
    )
    taxonomy_version = str(snapshot.manifest["taxonomy_version"])

    semaphore = asyncio.Semaphore(config.ai.ai_concurrency)

    async def describe_chunk(chunk: Sequence[SourceEmoji]) -> dict[str, _SemanticOutcome]:
        async with semaphore:
            effective_traces = _recover_ai_request_traces(
                cache,
                chunk,
                processed,
                config=config,
                taxonomy_version=taxonomy_version,
                cache_alias_scope=cache_alias_scope,
                checkpoint_traces=resume_request_traces,
            )
            generated = await _describe_batch(
                chunk,
                processed,
                config=config,
                cache=cache,
                budget=budget,
                ai_state=ai_state,
                api_key=api_key,
                temporary=temporary,
                taxonomy_version=taxonomy_version,
                qualifications=qualifications,
                routing_registry=routing_registry,
                resume_ai_cache_keys=resume_ai_cache_keys,
                cache_alias_scope=cache_alias_scope,
                resume_request_traces=effective_traces or None,
            )
            expected_native_ids = {item.native_id for item in chunk}
            if set(generated) != expected_native_ids or (
                on_chunk_completed is not None
                and any(not outcome.request_trace for outcome in generated.values())
            ):
                raise AIOutputError(
                    "AI chunk did not produce one exact traced outcome per source item"
                )
            if on_chunk_completed is not None:
                await on_chunk_completed(chunk, generated)
            return generated

    outcomes = await asyncio.gather(
        *(describe_chunk(chunk) for chunk in chunks), return_exceptions=True
    )
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            raise outcome
    for generated in cast(list[dict[str, _SemanticOutcome]], outcomes):
        outcomes_by_native.update(generated)
    if request_traces_out is not None:
        request_traces_out.clear()
        request_traces_out.update(
            {
                native_id: outcome.request_trace
                for native_id, outcome in outcomes_by_native.items()
                if outcome.request_trace
            }
        )
    return (
        {native_id: outcome.description for native_id, outcome in outcomes_by_native.items()},
        {native_id: outcome.generation for native_id, outcome in outcomes_by_native.items()},
    )


def _description_chunks(
    candidates: Sequence[SourceEmoji],
    processed: Mapping[str, ProcessedMedia],
    config: MojiLexConfig,
) -> list[list[SourceEmoji]]:
    chunks: list[list[SourceEmoji]] = []
    for animated, batch_size in (
        (False, config.processing.static_batch_size),
        (True, config.processing.animated_batch_size),
    ):
        for has_dark_render in (False, True):
            selected = [
                item
                for item in candidates
                if processed[item.native_id].metadata.animated is animated
                and processed[item.native_id].semantic_has_dark_render is has_dark_render
            ]
            for offset in range(0, len(selected), batch_size):
                chunks.append(selected[offset : offset + batch_size])
    return chunks


def _needs_generated_description(
    snapshot: DatasetSnapshot,
    platform: str,
    item: SourceEmoji,
    processed: ProcessedMedia,
    *,
    redescribe: str,
    overwrite_reviewed: bool,
) -> bool:
    """Mirror the top-level semantic reuse decision without touching rendered frames."""

    existing = _existing_emoji(snapshot, platform, item.native_id)
    current_media = [processed.dataset_metadata()]
    same_media = existing is not None and domain_media_digest(existing.media) == cache_media_digest(
        cast(Sequence[Mapping[str, object]], current_media)
    )
    existing_extension = existing.extensions.get("telegram", {}) if existing else {}
    same_context = existing is not None and (
        existing_extension.get("fallback_emoji") == item.fallback_emoji
        and existing_extension.get("needs_repainting") is item.needs_repainting
    )
    protected = existing is not None and (
        existing.review.status.value == "approved"
        or existing.provenance.origin.value in {"human", "mixed"}
    )
    should_generate = existing is None or not same_media or not same_context
    if redescribe == "all" and not (protected and not overwrite_reviewed):
        should_generate = True
    if redescribe == "missing" and existing is not None and same_media and same_context:
        should_generate = not {"ru", "en"}.issubset(existing.descriptions)
    return should_generate


def _trace_cache_key(
    traces: Mapping[str, Sequence[_AICacheTrace]] | None,
    native_id: str,
    *,
    model: str,
    stage: Literal["primary", "escalated"],
) -> str | None:
    if traces is None:
        return None
    matches = [
        trace
        for trace in traces.get(native_id, ())
        if trace.stage == stage and trace.model == model
    ]
    return matches[0].cache_key if len(matches) == 1 else None


async def _preview_cached_batch(
    items: Sequence[SourceEmoji],
    processed: Mapping[str, ProcessedMedia],
    *,
    model: str,
    config: MojiLexConfig,
    cache: CacheStore | None,
    temporary: TemporaryMediaRun,
    taxonomy_version: str,
) -> list[CachedAIResult] | None:
    """Inspect exact would-be request keys without repairing aliases or touching timestamps."""

    if not items or cache is None:
        return None
    prepared = await _prepare_ai_request(items, processed, model=model, temporary=temporary)
    hits: list[CachedAIResult] = []
    for item in items:
        context = _vision_context(item, processed[item.native_id])
        label = prepared.identity.label_for(item.native_id)
        key = _cache_key(
            item,
            processed[item.native_id],
            context,
            config,
            model=model,
            model_revision=None,
            taxonomy_version=taxonomy_version,
            request_identity=prepared.identity,
            item_label=label,
        )
        try:
            hit = _load_cached_description(
                cache,
                key,
                source=item,
                processed=processed[item.native_id],
                context=context,
                config=config,
                model=model,
                taxonomy_version=taxonomy_version,
                resume_cache_key=None,
                cache_alias_scope=None,
                request_identity=prepared.identity,
                item_label=label,
            )
        except (AIError, CacheError, ValueError, sqlite3.Error):
            return None
        if hit is None:
            return None
        hits.append(hit)
    if len({hit.result.model_revision for hit in hits}) != 1:
        return None
    return hits


async def _preview_ai_plan(
    snapshot: DatasetSnapshot,
    source: SourceCollection,
    *,
    config: MojiLexConfig,
    options: PipelineOptions,
    cache: CacheStore | None,
    processed: Mapping[str, ProcessedMedia] | None = None,
    temporary: TemporaryMediaRun | None = None,
) -> dict[str, int]:
    candidates: list[SourceEmoji] = []
    for item in source.items:
        if processed is not None:
            needed = _needs_generated_description(
                snapshot,
                source.platform,
                item,
                processed[item.native_id],
                redescribe=options.redescribe,
                overwrite_reviewed=options.overwrite_reviewed,
            )
        else:
            existing = _existing_emoji(snapshot, source.platform, item.native_id)
            extension = existing.extensions.get("telegram", {}) if existing else {}
            needed = existing is None or extension.get("file_unique_id") != item.file_unique_id
            if existing is not None and not needed:
                # Metadata-only previews may estimate reuse, but cannot verify bytes or
                # derive the exact contact-sheet request/cache key without --check-media.
                approximate = ProcessedMedia(
                    frame_paths=(),
                    metadata=MediaMetadata.model_validate(
                        existing.media[0].model_dump(
                            exclude={"role", "variant_id"}, exclude_none=True
                        )
                    ),
                )
                needed = _needs_generated_description(
                    snapshot,
                    source.platform,
                    item,
                    approximate,
                    redescribe=options.redescribe,
                    overwrite_reviewed=options.overwrite_reviewed,
                )
        if needed:
            candidates.append(item)
    plan = {
        "ai_items_planned": len(candidates),
        "ai_batches_planned": 0,
        "ai_cache_hits_estimated": 0,
        "ai_cache_hits_unknown": 0,
        "ai_requests_estimated_upper_bound": 0,
    }
    if not candidates:
        return plan
    if processed is None:
        for animated, size in (
            (False, config.processing.static_batch_size),
            (True, config.processing.animated_batch_size),
        ):
            for adaptive in (False, True):
                count = sum(
                    (item.media_format != "webp") is animated and item.needs_repainting is adaptive
                    for item in candidates
                )
                plan["ai_batches_planned"] += (count + size - 1) // size
        plan["ai_cache_hits_unknown"] = len(candidates) if cache is not None else 0
        # Unknown backgrounds/routing may split batches; include bounded retries.
        plan["ai_requests_estimated_upper_bound"] = len(candidates) * (
            6 if config.ai.model_routing == "rules" else 4
        )
        return plan
    assert temporary is not None
    taxonomy_version = str(snapshot.manifest["taxonomy_version"])
    qualifications = ModelQualificationRegistry.load(snapshot.root)
    routing = RoutingReasonRegistry.load(snapshot.root)

    async def inspect_single(item: SourceEmoji) -> None:
        hit = await _preview_cached_batch(
            (item,),
            processed,
            model=config.ai.escalation_model,
            config=config,
            cache=cache,
            temporary=temporary,
            taxonomy_version=taxonomy_version,
        )
        if hit is not None:
            plan["ai_cache_hits_estimated"] += 1
        else:
            plan["ai_batches_planned"] += 1
            plan["ai_requests_estimated_upper_bound"] += 2

    for chunk in _description_chunks(candidates, processed, config):
        primary: list[SourceEmoji] = []
        for item in chunk:
            reasons = routing.canonicalize(deterministic_routing_reasons(processed[item.native_id]))
            if should_escalate(
                config.ai.model_routing, reasons, escalation_model=config.ai.escalation_model
            ):
                await inspect_single(item)
            else:
                primary.append(item)
        if not primary:
            continue
        hits = await _preview_cached_batch(
            primary,
            processed,
            model=config.ai.model,
            config=config,
            cache=cache,
            temporary=temporary,
            taxonomy_version=taxonomy_version,
        )
        if hits is None:
            plan["ai_batches_planned"] += 1
            plan["ai_requests_estimated_upper_bound"] += 2 + len(primary) * (
                4 if config.ai.model_routing == "rules" else 2
            )
            continue
        for item, hit in zip(primary, hits, strict=True):
            outcome = _semantic_outcome(
                hit,
                generation_stage="primary",
                routing_reasons=(),
                config=config,
                taxonomy_version=taxonomy_version,
                qualifications=qualifications,
                routing_registry=routing,
            )
            semantic_reasons = set(semantic_routing_reasons(outcome.description))
            if outcome.generation.qualification_id is None:
                semantic_reasons.add(RoutingReason.UNQUALIFIED_MODEL)
            if should_escalate(
                config.ai.model_routing,
                semantic_reasons,
                escalation_model=config.ai.escalation_model,
            ):
                await inspect_single(item)
            else:
                plan["ai_cache_hits_estimated"] += 1
    return plan


async def _load_or_describe_primary_batch(
    items: Sequence[SourceEmoji],
    processed: Mapping[str, ProcessedMedia],
    *,
    config: MojiLexConfig,
    cache: CacheStore,
    budget: RequestBudget,
    ai_state: _AIState,
    api_key: str | None,
    temporary: TemporaryMediaRun,
    taxonomy_version: str,
    request_identity: _AIRequestIdentity | None,
    resume_request_traces: Mapping[str, Sequence[_AICacheTrace]] | None,
    cache_alias_scope: str | None,
) -> dict[str, tuple[CachedAIResult, _AICacheTrace]]:
    prepared: _PreparedAIRequest | None = None
    if request_identity is None:
        prepared = await _prepare_ai_request(
            items,
            processed,
            model=config.ai.model,
            temporary=temporary,
        )
        request_identity = prepared.identity
    contexts = {item.native_id: _vision_context(item, processed[item.native_id]) for item in items}
    hits: dict[str, tuple[CachedAIResult, _AICacheTrace]] = {}
    lookup_keys: dict[str, str] = {}
    for item in items:
        label = request_identity.label_for(item.native_id)
        lookup_key = _cache_key(
            item,
            processed[item.native_id],
            contexts[item.native_id],
            config,
            model=config.ai.model,
            model_revision=None,
            taxonomy_version=taxonomy_version,
            request_identity=request_identity,
            item_label=label,
        )
        lookup_keys[item.native_id] = lookup_key
        try:
            cached = _load_cached_description(
                cache,
                lookup_key,
                source=item,
                processed=processed[item.native_id],
                context=contexts[item.native_id],
                config=config,
                model=config.ai.model,
                taxonomy_version=taxonomy_version,
                resume_cache_key=_trace_cache_key(
                    resume_request_traces,
                    item.native_id,
                    model=config.ai.model,
                    stage="primary",
                ),
                cache_alias_scope=cache_alias_scope,
                request_identity=request_identity,
                item_label=label,
            )
        except CacheError:
            cached = None
        if cached is None:
            continue
        actual_key = _cache_key(
            item,
            processed[item.native_id],
            contexts[item.native_id],
            config,
            model=cached.result.model,
            model_revision=cached.result.model_revision,
            taxonomy_version=taxonomy_version,
            request_identity=request_identity,
            item_label=label,
        )
        hits[item.native_id] = (
            cached,
            _AICacheTrace(
                stage="primary",
                model=cached.result.model,
                model_revision=cached.result.model_revision,
                cache_key=actual_key,
                request_identity=request_identity,
            ),
        )
    if len(hits) == len(items):
        if (
            len(items) == 1
            or len({cached.result.model_revision for cached, _trace in hits.values()}) == 1
        ):
            ai_state.cache_hits += len(items)
            return _cache_ai_request_results(
                cache,
                tuple(
                    _AIResultWrite(
                        item=item,
                        lookup_key=lookup_keys[item.native_id],
                        storage_key=hits[item.native_id][1].cache_key,
                        result=hits[item.native_id][0].result,
                        generated_at=hits[item.native_id][0].generated_at,
                    )
                    for item in items
                ),
                stage="primary",
                request_identity=request_identity,
                cache_alias_scope=cache_alias_scope,
                repair_request_envelope=True,
            )
        hits.clear()
    if resume_request_traces is not None:
        raise AIOutputError("resume AI batch cache is missing, malformed, or partial")
    if prepared is None:
        raise AIOutputError("AI batch request cannot be reconstructed")
    provider = await _provider_for_model(
        ai_state,
        config=config,
        api_key=api_key,
        model=config.ai.model,
    )
    response = await describe_with_recovery(provider, prepared.request, budget)
    _validate_actual_result(
        response,
        config.ai.provider,
        config.ai.model,
        require_single=False,
    )
    by_label = {described.label: described for described in response.batch.items}
    generated_at = _utc_text()
    writes: list[_AIResultWrite] = []
    for item in items:
        label = request_identity.label_for(item.native_id)
        described = by_label[label]
        lookup_key = _cache_key(
            item,
            processed[item.native_id],
            contexts[item.native_id],
            config,
            model=response.model,
            model_revision=None,
            taxonomy_version=taxonomy_version,
            request_identity=request_identity,
            item_label=label,
        )
        key = _cache_key(
            item,
            processed[item.native_id],
            contexts[item.native_id],
            config,
            model=response.model,
            model_revision=response.model_revision,
            taxonomy_version=taxonomy_version,
            request_identity=request_identity,
            item_label=label,
        )
        writes.append(
            _AIResultWrite(
                item=item,
                lookup_key=lookup_key,
                storage_key=key,
                result=response.model_copy(update={"batch": DescriptionBatch(items=(described,))}),
                generated_at=generated_at,
            )
        )
    return _cache_ai_request_results(
        cache,
        writes,
        stage="primary",
        request_identity=request_identity,
        cache_alias_scope=cache_alias_scope,
        repair_request_envelope=True,
    )


async def _load_primary_from_traces(
    items: Sequence[SourceEmoji],
    processed: Mapping[str, ProcessedMedia],
    traces: Mapping[str, Sequence[_AICacheTrace]],
    *,
    config: MojiLexConfig,
    cache: CacheStore,
    budget: RequestBudget,
    ai_state: _AIState,
    api_key: str | None,
    temporary: TemporaryMediaRun,
    taxonomy_version: str,
    cache_alias_scope: str | None,
) -> dict[str, tuple[CachedAIResult, _AICacheTrace]]:
    batch_identity = _resume_request_identity(
        items,
        processed,
        traces,
        model=config.ai.model,
        stage="primary",
    )
    if batch_identity is not None:
        return await _load_or_describe_primary_batch(
            items,
            processed,
            config=config,
            cache=cache,
            budget=budget,
            ai_state=ai_state,
            api_key=api_key,
            temporary=temporary,
            taxonomy_version=taxonomy_version,
            request_identity=batch_identity,
            resume_request_traces=traces,
            cache_alias_scope=cache_alias_scope,
        )
    results: dict[str, tuple[CachedAIResult, _AICacheTrace]] = {}
    for item in items:
        identity = _resume_request_identity(
            (item,),
            processed,
            traces,
            model=config.ai.model,
            stage="primary",
        )
        if identity is None:
            raise AIOutputError("resume AI request trace is incomplete")
        trace: list[_AICacheTrace] = []
        cached = await _load_or_describe_single(
            item,
            processed[item.native_id],
            model=config.ai.model,
            config=config,
            cache=cache,
            budget=budget,
            ai_state=ai_state,
            api_key=api_key,
            context=_vision_context(item, processed[item.native_id]),
            temporary=temporary,
            taxonomy_version=taxonomy_version,
            resume_cache_key=_trace_cache_key(
                traces,
                item.native_id,
                model=config.ai.model,
                stage="primary",
            ),
            cache_alias_scope=cache_alias_scope,
            request_identity=identity,
            generation_stage="primary",
            trace_out=trace,
        )
        results[item.native_id] = (cached, trace[0])
    return results


async def _describe_batch(
    items: Sequence[SourceEmoji],
    processed: Mapping[str, ProcessedMedia],
    *,
    config: MojiLexConfig,
    cache: CacheStore,
    budget: RequestBudget,
    ai_state: _AIState,
    api_key: str | None,
    temporary: TemporaryMediaRun,
    taxonomy_version: str,
    qualifications: ModelQualificationRegistry,
    routing_registry: RoutingReasonRegistry,
    resume_ai_cache_keys: Mapping[str, str] | None = None,
    cache_alias_scope: str | None = None,
    resume_request_traces: Mapping[str, Sequence[_AICacheTrace]] | None = None,
    require_exact_resume: bool = False,
) -> dict[str, _SemanticOutcome]:
    del resume_ai_cache_keys
    if not items:
        return {}
    contexts = {item.native_id: _vision_context(item, processed[item.native_id]) for item in items}

    async def routed_single(
        item: SourceEmoji,
        *,
        model: str,
        generation_stage: Literal["primary", "escalated"],
        routing_reasons: Sequence[RoutingReason],
        request_identity: _AIRequestIdentity | None,
        prior_trace: tuple[_AICacheTrace, ...] = (),
    ) -> _SemanticOutcome:
        resume_cache_key = (
            _trace_cache_key(
                resume_request_traces,
                item.native_id,
                model=model,
                stage=generation_stage,
            )
            if request_identity is not None
            else None
        )
        try:
            return await _describe_routed_single(
                item,
                processed[item.native_id],
                model=model,
                generation_stage=generation_stage,
                routing_reasons=routing_reasons,
                config=config,
                cache=cache,
                budget=budget,
                ai_state=ai_state,
                api_key=api_key,
                context=contexts[item.native_id],
                temporary=temporary,
                taxonomy_version=taxonomy_version,
                qualifications=qualifications,
                routing_registry=routing_registry,
                resume_cache_key=resume_cache_key,
                cache_alias_scope=cache_alias_scope,
                request_identity=request_identity,
                prior_trace=prior_trace,
            )
        except (AIError, CacheError, ValueError):
            if request_identity is None or require_exact_resume:
                raise
            # The exact checkpoint trace is only an optimization after a full
            # decode. If its row was pruned or corrupted, build a fresh exact
            # single-item request from the verified frames.
            return await _describe_routed_single(
                item,
                processed[item.native_id],
                model=model,
                generation_stage=generation_stage,
                routing_reasons=routing_reasons,
                config=config,
                cache=cache,
                budget=budget,
                ai_state=ai_state,
                api_key=api_key,
                context=contexts[item.native_id],
                temporary=temporary,
                taxonomy_version=taxonomy_version,
                qualifications=qualifications,
                routing_registry=routing_registry,
                cache_alias_scope=cache_alias_scope,
                prior_trace=prior_trace,
            )

    result: dict[str, _SemanticOutcome] = {}
    primary_items: list[SourceEmoji] = []
    for item in items:
        pre_reasons = routing_registry.canonicalize(
            deterministic_routing_reasons(processed[item.native_id])
        )
        if should_escalate(
            config.ai.model_routing,
            pre_reasons,
            escalation_model=config.ai.escalation_model,
        ):
            identity = (
                _resume_request_identity(
                    (item,),
                    processed,
                    resume_request_traces,
                    model=config.ai.escalation_model,
                    stage="escalated",
                )
                if resume_request_traces is not None
                else None
            )
            if resume_request_traces is not None and identity is None and require_exact_resume:
                raise AIOutputError("resume escalation request trace is incomplete")
            result[item.native_id] = await routed_single(
                item,
                model=config.ai.escalation_model,
                generation_stage="escalated",
                routing_reasons=pre_reasons,
                request_identity=identity,
            )
        else:
            primary_items.append(item)
    if not primary_items:
        return result

    primary_results: dict[str, tuple[CachedAIResult, _AICacheTrace]] = {}
    loaded_from_resume = False
    if resume_request_traces is not None:
        try:
            primary_results = await _load_primary_from_traces(
                primary_items,
                processed,
                resume_request_traces,
                config=config,
                cache=cache,
                budget=budget,
                ai_state=ai_state,
                api_key=api_key,
                temporary=temporary,
                taxonomy_version=taxonomy_version,
                cache_alias_scope=cache_alias_scope,
            )
        except (AIError, CacheError, ValueError):
            if require_exact_resume:
                raise
        else:
            loaded_from_resume = True
    if not loaded_from_resume:
        try:
            primary_results = await _load_or_describe_primary_batch(
                primary_items,
                processed,
                config=config,
                cache=cache,
                budget=budget,
                ai_state=ai_state,
                api_key=api_key,
                temporary=temporary,
                taxonomy_version=taxonomy_version,
                request_identity=None,
                resume_request_traces=None,
                cache_alias_scope=cache_alias_scope,
            )
        except AIOutputError:
            for item in primary_items:
                trace: list[_AICacheTrace] = []
                try:
                    cached = await _load_or_describe_single(
                        item,
                        processed[item.native_id],
                        model=config.ai.model,
                        config=config,
                        cache=cache,
                        budget=budget,
                        ai_state=ai_state,
                        api_key=api_key,
                        context=contexts[item.native_id],
                        temporary=temporary,
                        taxonomy_version=taxonomy_version,
                        cache_alias_scope=cache_alias_scope,
                        generation_stage="primary",
                        trace_out=trace,
                    )
                except AIOutputError:
                    if config.ai.model_routing != "rules":
                        raise
                    result[item.native_id] = await _describe_routed_single(
                        item,
                        processed[item.native_id],
                        model=config.ai.escalation_model,
                        generation_stage="escalated",
                        routing_reasons=(RoutingReason.SCHEMA_RETRY_EXHAUSTED,),
                        config=config,
                        cache=cache,
                        budget=budget,
                        ai_state=ai_state,
                        api_key=api_key,
                        context=contexts[item.native_id],
                        temporary=temporary,
                        taxonomy_version=taxonomy_version,
                        qualifications=qualifications,
                        routing_registry=routing_registry,
                        cache_alias_scope=cache_alias_scope,
                    )
                    continue
                primary_results[item.native_id] = (cached, trace[0])

    for item in primary_items:
        if item.native_id in result:
            continue
        primary, primary_trace = primary_results[item.native_id]
        primary_outcome = _semantic_outcome(
            primary,
            generation_stage="primary",
            routing_reasons=(),
            config=config,
            taxonomy_version=taxonomy_version,
            qualifications=qualifications,
            routing_registry=routing_registry,
            request_trace=(primary_trace,),
        )
        reasons = set(semantic_routing_reasons(primary_outcome.description))
        if primary_outcome.generation.qualification_id is None:
            reasons.add(RoutingReason.UNQUALIFIED_MODEL)
        canonical_reasons = routing_registry.canonicalize(reasons)
        if should_escalate(
            config.ai.model_routing,
            canonical_reasons,
            escalation_model=config.ai.escalation_model,
        ):
            identity = (
                _resume_request_identity(
                    (item,),
                    processed,
                    resume_request_traces,
                    model=config.ai.escalation_model,
                    stage="escalated",
                )
                if resume_request_traces is not None
                else None
            )
            result[item.native_id] = await routed_single(
                item,
                model=config.ai.escalation_model,
                generation_stage="escalated",
                routing_reasons=canonical_reasons,
                request_identity=identity,
                prior_trace=(primary_trace,),
            )
        else:
            result[item.native_id] = primary_outcome
    return result


def _cache_description(
    cache: CacheStore,
    key: str,
    result: DescriptionResult,
    *,
    generated_at: str,
    aliases: Sequence[str] = (),
) -> CachedAIResult:
    if len(result.batch.items) != 1:
        raise ValueError("per-item AI cache entries must contain exactly one item")
    normalized = result.batch.items[0].model_copy(update={"label": "E001"})
    return cache.put_ai(
        key,
        result.model_copy(update={"batch": DescriptionBatch(items=(normalized,))}),
        generated_at=generated_at,
        aliases=aliases,
    )


def _ai_request_envelope_key(
    scope: str | None,
    *,
    stage: Literal["primary", "escalated"],
    model: str,
    plan_sha256: str,
) -> str | None:
    if scope is None:
        return None
    encoded = scope.encode("utf-8")
    if not 1 <= len(encoded) <= 256 or any(ord(character) < 32 for character in scope):
        raise CacheError("AI request envelope scope is invalid")
    digest = hashlib.sha256(
        rfc8785.dumps(
            {
                "format_version": 1,
                "scope": scope,
                "stage": stage,
                "model": model,
                "plan_sha256": plan_sha256,
            }
        )
    ).hexdigest()
    return f"ai-request-envelope-v1:{digest}"


def _cache_ai_request_results(
    cache: CacheStore,
    writes: Sequence[_AIResultWrite],
    *,
    stage: Literal["primary", "escalated"],
    request_identity: _AIRequestIdentity,
    cache_alias_scope: str | None,
    repair_request_envelope: bool = False,
) -> dict[str, tuple[CachedAIResult, _AICacheTrace]]:
    if not writes:
        raise ValueError("AI request cache write requires at least one item")
    if len({write.item.native_id for write in writes}) != len(writes):
        raise CacheError("AI request cache write contains duplicate source items")
    models = {write.result.model for write in writes}
    revisions = {write.result.model_revision for write in writes}
    if len(models) != 1 or (len(writes) > 1 and len(revisions) != 1):
        raise AIOutputError("one AI batch returned inconsistent model provenance")

    traces: dict[str, _AICacheTrace] = {}
    cache_writes: list[AICacheWrite] = []
    for write in writes:
        if len(write.result.batch.items) != 1:
            raise ValueError("per-item AI cache entries must contain exactly one item")
        normalized = write.result.batch.items[0].model_copy(update={"label": "E001"})
        normalized_result = write.result.model_copy(
            update={"batch": DescriptionBatch(items=(normalized,))}
        )
        cache_writes.append(
            AICacheWrite(
                key=write.storage_key,
                result=normalized_result,
                generated_at=write.generated_at,
                aliases=_cache_aliases(cache_alias_scope, write.lookup_key),
            )
        )
        traces[write.item.native_id] = _AICacheTrace(
            stage=stage,
            model=write.result.model,
            model_revision=write.result.model_revision,
            cache_key=write.storage_key,
            request_identity=request_identity,
        )

    envelope_key = _ai_request_envelope_key(
        cache_alias_scope,
        stage=stage,
        model=next(iter(models)),
        plan_sha256=request_identity.plan_sha256,
    )
    envelope: dict[str, object] | None = None
    if envelope_key is not None:
        envelope = {
            "format_version": 1,
            "items": [
                {
                    "native_id": write.item.native_id,
                    "request": AIRequestCheckpoint(
                        stage=stage,
                        model=trace.model,
                        model_revision=trace.model_revision,
                        cache_key=trace.cache_key,
                        plan_sha256=request_identity.plan_sha256,
                        request_sha256=request_identity.request_sha256,
                        shown_media_sha256=request_identity.shown_media_sha256,
                        item_label=request_identity.label_for(write.item.native_id),
                    ).model_dump(mode="json"),
                }
                for write in writes
                for trace in (traces[write.item.native_id],)
            ],
        }
    stored = cache.put_ai_batch(
        cache_writes,
        envelope_key=envelope_key,
        envelope=envelope,
        repair_request_envelope=(repair_request_envelope and envelope_key is not None),
    )
    result: dict[str, tuple[CachedAIResult, _AICacheTrace]] = {}
    for write in writes:
        cached = stored[write.storage_key]
        trace = traces[write.item.native_id]
        if (
            cached.result.model != trace.model
            or cached.result.model_revision != trace.model_revision
        ):
            raise CacheError("immutable AI cache row differs from its request envelope")
        result[write.item.native_id] = (cached, trace)
    return result


async def _describe_single(
    source: SourceEmoji,
    processed: ProcessedMedia,
    *,
    model: str,
    budget: RequestBudget,
    provider: Any,
    context: VisionContext,
    temporary: TemporaryMediaRun,
    prepared_request: _PreparedAIRequest | None = None,
) -> DescriptionResult:
    del context
    prepared = prepared_request
    if prepared is None:
        prepared = await _prepare_ai_request(
            (source,),
            {source.native_id: processed},
            model=model,
            temporary=temporary,
        )
    return await describe_with_recovery(provider, prepared.request, budget)


async def _describe_routed_single(
    source: SourceEmoji,
    processed: ProcessedMedia,
    *,
    model: str,
    generation_stage: Literal["primary", "escalated"],
    routing_reasons: Sequence[RoutingReason],
    config: MojiLexConfig,
    cache: CacheStore,
    budget: RequestBudget,
    ai_state: _AIState,
    api_key: str | None,
    context: VisionContext,
    temporary: TemporaryMediaRun,
    taxonomy_version: str,
    qualifications: ModelQualificationRegistry,
    routing_registry: RoutingReasonRegistry,
    resume_cache_key: str | None = None,
    cache_alias_scope: str | None = None,
    request_identity: _AIRequestIdentity | None = None,
    prior_trace: tuple[_AICacheTrace, ...] = (),
) -> _SemanticOutcome:
    trace: list[_AICacheTrace] = []
    result = await _load_or_describe_single(
        source,
        processed,
        model=model,
        config=config,
        cache=cache,
        budget=budget,
        ai_state=ai_state,
        api_key=api_key,
        context=context,
        temporary=temporary,
        taxonomy_version=taxonomy_version,
        resume_cache_key=resume_cache_key,
        cache_alias_scope=cache_alias_scope,
        request_identity=request_identity,
        generation_stage=generation_stage,
        trace_out=trace,
    )
    return _semantic_outcome(
        result,
        generation_stage=generation_stage,
        routing_reasons=routing_reasons,
        config=config,
        taxonomy_version=taxonomy_version,
        qualifications=qualifications,
        routing_registry=routing_registry,
        request_trace=(*prior_trace, *trace),
    )


async def _load_or_describe_single(
    source: SourceEmoji,
    processed: ProcessedMedia,
    *,
    model: str,
    config: MojiLexConfig,
    cache: CacheStore,
    budget: RequestBudget,
    ai_state: _AIState,
    api_key: str | None,
    context: VisionContext,
    temporary: TemporaryMediaRun,
    taxonomy_version: str,
    resume_cache_key: str | None = None,
    cache_alias_scope: str | None = None,
    request_identity: _AIRequestIdentity | None = None,
    generation_stage: Literal["primary", "escalated"] = "primary",
    trace_out: list[_AICacheTrace] | None = None,
) -> CachedAIResult:
    prepared: _PreparedAIRequest | None = None
    if request_identity is None:
        prepared = await _prepare_ai_request(
            (source,),
            {source.native_id: processed},
            model=model,
            temporary=temporary,
        )
        request_identity = prepared.identity
    elif (
        request_identity.plan_sha256
        != _ai_request_plan_sha256((source,), {source.native_id: processed}, model=model)
        or request_identity.label_for(source.native_id) != "E001"
    ):
        raise AIOutputError("resume AI request identity does not match the single-item plan")
    item_label = request_identity.label_for(source.native_id)
    lookup_key = _cache_key(
        source,
        processed,
        context,
        config,
        model=model,
        model_revision=None,
        taxonomy_version=taxonomy_version,
        request_identity=request_identity,
        item_label=item_label,
    )
    try:
        cached = _load_cached_description(
            cache,
            lookup_key,
            source=source,
            processed=processed,
            context=context,
            config=config,
            model=model,
            taxonomy_version=taxonomy_version,
            resume_cache_key=resume_cache_key,
            cache_alias_scope=cache_alias_scope,
            request_identity=request_identity,
            item_label=item_label,
        )
    except CacheError:
        if prepared is None:
            raise
        cached = None
    if cached is not None:
        ai_state.cache_hits += 1
        storage_key = _cache_key(
            source,
            processed,
            context,
            config,
            model=cached.result.model,
            model_revision=cached.result.model_revision,
            taxonomy_version=taxonomy_version,
            request_identity=request_identity,
            item_label=item_label,
        )
        cached, trace = _cache_ai_request_results(
            cache,
            (
                _AIResultWrite(
                    item=source,
                    lookup_key=lookup_key,
                    storage_key=storage_key,
                    result=cached.result,
                    generated_at=cached.generated_at,
                ),
            ),
            stage=generation_stage,
            request_identity=request_identity,
            cache_alias_scope=cache_alias_scope,
            repair_request_envelope=True,
        )[source.native_id]
        if trace_out is not None:
            trace_out.append(trace)
        return cached
    if prepared is None:
        raise AIOutputError("resume AI cache entry is missing or invalid")
    provider = await _provider_for_model(
        ai_state,
        config=config,
        api_key=api_key,
        model=model,
    )
    result = await _describe_single(
        source,
        processed,
        model=model,
        budget=budget,
        provider=provider,
        context=context,
        temporary=temporary,
        prepared_request=prepared,
    )
    _validate_actual_result(result, config.ai.provider, model)
    storage_key = _cache_key(
        source,
        processed,
        context,
        config,
        model=result.model,
        model_revision=result.model_revision,
        taxonomy_version=taxonomy_version,
        request_identity=request_identity,
        item_label=item_label,
    )
    stored, trace = _cache_ai_request_results(
        cache,
        (
            _AIResultWrite(
                item=source,
                lookup_key=lookup_key,
                storage_key=storage_key,
                result=result,
                generated_at=_utc_text(),
            ),
        ),
        stage=generation_stage,
        request_identity=request_identity,
        cache_alias_scope=cache_alias_scope,
        repair_request_envelope=True,
    )[source.native_id]
    if trace_out is not None:
        trace_out.append(trace)
    return stored


def _cache_aliases(scope: str | None, lookup_key: str) -> tuple[str, ...]:
    if scope is None:
        return ()
    digest = hashlib.sha256(f"{scope}\0{lookup_key}".encode()).hexdigest()
    return (f"run-v1:{digest}",)


def _load_cached_description(
    cache: CacheStore,
    lookup_key: str,
    *,
    source: SourceEmoji,
    processed: ProcessedMedia,
    context: VisionContext,
    config: MojiLexConfig,
    model: str,
    taxonomy_version: str,
    resume_cache_key: str | None,
    cache_alias_scope: str | None,
    request_identity: _AIRequestIdentity | None = None,
    item_label: str = "E001",
) -> CachedAIResult | None:
    """Load an exact/checkpoint/run-scoped hit after recomputing its actual key."""

    candidate_keys: list[tuple[str, bool, bool]] = []
    if resume_cache_key is not None and resume_cache_key != lookup_key:
        candidate_keys.append((resume_cache_key, False, True))
    candidate_keys.append((lookup_key, False, False))
    aliases = _cache_aliases(cache_alias_scope, lookup_key)
    if aliases:
        candidate_keys.append((aliases[0], True, False))
    for candidate_key, follow_aliases, allow_other_model in candidate_keys:
        hit = cache.get_ai_entry(candidate_key, follow_aliases=follow_aliases)
        if hit is None:
            continue
        resolved_key, entry = hit
        cached = entry.result
        if cached.provider != config.ai.provider:
            raise AIOutputError("AI cache entry belongs to a different provider")
        if cached.model != model:
            # A completed routed run records the final escalation key. While
            # reconstructing its primary routing decision that key is expected
            # to name another model, so fall through to the request-key alias.
            if allow_other_model:
                continue
            raise AIOutputError("AI cache entry belongs to a different model")
        _validate_actual_result(cached, config.ai.provider, model)
        expected_actual_key = _cache_key(
            source,
            processed,
            context,
            config,
            model=cached.model,
            model_revision=cached.model_revision,
            taxonomy_version=taxonomy_version,
            request_identity=request_identity,
            item_label=item_label,
        )
        if resolved_key != expected_actual_key:
            raise AIOutputError(
                "AI cache entry does not match the current media, prompt, or policy context"
            )
        return entry
    return None


async def _provider_for_model(
    ai_state: _AIState,
    *,
    config: MojiLexConfig,
    api_key: str | None,
    model: str,
) -> Any:
    key = f"{config.ai.provider}\0{model}"
    async with ai_state.initialization_lock:
        provider = ai_state.providers.get(key)
        if provider is None:
            if config.ai.provider == "gemini" and not api_key:
                raise CommandError(
                    "CREDENTIAL_MISSING",
                    "GEMINI_API_KEY is required for uncached descriptions.",
                    hint="Set it in the process environment or use an already populated cache.",
                )
            report_progress(
                f"Derived contact-sheet PNG images will be sent to provider={config.ai.provider}, "
                f"model={model}. Provider processing terms: "
                "https://ai.google.dev/gemini-api/terms . "
                "MojiLex does not guarantee zero retention by the provider."
            )
            provider = default_registry().create(
                config.ai.provider,
                model=model,
                api_key=api_key,
            )
            ai_state.providers[key] = provider
        if key not in ai_state.credentials_validated:
            await provider.validate_credentials()
            ai_state.credentials_validated.add(key)
        return provider


def _validate_actual_result(
    result: DescriptionResult,
    expected_provider: str,
    expected_model: str,
    *,
    require_single: bool = True,
) -> None:
    if result.provider != expected_provider or result.model != expected_model:
        raise AIOutputError("AI provider returned a result for a different provider or model")
    if require_single and len(result.batch.items) != 1:
        raise AIOutputError("per-item semantic result must contain exactly one full object")
    inputs = _GENERATION_INPUTS.get()
    if inputs is not None:
        for item in result.batch.items:
            try:
                inputs.concepts.validate_selection(item.concept_ids)
            except ValueError as exc:
                raise AIOutputError("AI concepts do not match the exact candidate set") from exc


def _semantic_outcome(
    cached_result: CachedAIResult,
    *,
    generation_stage: Literal["primary", "escalated"],
    routing_reasons: Sequence[RoutingReason],
    config: MojiLexConfig,
    taxonomy_version: str,
    qualifications: ModelQualificationRegistry,
    routing_registry: RoutingReasonRegistry,
    request_trace: tuple[_AICacheTrace, ...] = (),
) -> _SemanticOutcome:
    result = cached_result.result
    _validate_actual_result(result, config.ai.provider, result.model)
    generated_at = cached_result.generated_at
    match = match_qualification(
        qualifications,
        QualificationQuery(
            provider=result.provider,
            model=result.model,
            model_revision=result.model_revision,
            description_profile="standard-v1",
            prompt_sha256=prompt_sha256(),
            request_parameters_sha256=gemini_request_parameters_sha256(),
            schema_version=SCHEMA_VERSION,
            taxonomy_version=taxonomy_version,
            pipeline_version=PIPELINE_VERSION,
            routing_policy_version=ROUTING_POLICY_VERSION,
            languages=config.ai.languages,
            generated_at=generated_at,
            **_generation_binding_fields(),
        ),
    )
    reasons = set(routing_reasons)
    if not match.qualified:
        reasons.add(RoutingReason.UNQUALIFIED_MODEL)
    canonical_reasons = routing_registry.canonicalize(reasons)
    return _SemanticOutcome(
        description=result.batch.items[0],
        generation=SemanticGenerationMetadata(
            provider=result.provider,
            model=result.model,
            model_revision=result.model_revision,
            description_profile="standard-v1",
            prompt_sha256=prompt_sha256(),
            request_parameters_sha256=gemini_request_parameters_sha256(),
            qualification_id=match.qualification_id if match.qualified else None,
            generation_stage=generation_stage,
            routing_policy_version=ROUTING_POLICY_VERSION,
            routing_reason_codes=tuple(reason.value for reason in canonical_reasons),
            generated_at=generated_at,
            **_generation_binding_fields(),
        ),
        request_trace=request_trace,
    )


def _vision_context(source: SourceEmoji, processed: ProcessedMedia) -> VisionContext:
    variants: tuple[Literal["light", "dark"], ...] = (
        ("light", "dark") if processed.semantic_has_dark_render else ("light",)
    )
    return VisionContext(
        fallback_emoji=source.fallback_emoji,
        needs_repainting=source.needs_repainting,
        requested_languages=("ru", "en"),
        frame_count=processed.semantic_frame_count,
        canvas_size=256,
        background_variants=variants,
    )


def _bind_deterministic_analyses(
    processed: Mapping[str, ProcessedMedia],
) -> dict[str, DeterministicEmojiAnalysis]:
    """Bind worker-owned analysis to the canonical role/variant identity."""

    result: dict[str, DeterministicEmojiAnalysis] = {}
    for native_id, value in processed.items():
        analysis = value.analysis
        if analysis is None:
            raise CommandError(
                "MEDIA_RENDER_FAILED",
                "The media worker returned no deterministic analysis.",
                hint="Run `mojilex doctor` and verify the decoder prerequisites.",
                entity_id=native_id,
            )
        rendering_item = {
            "role": value.metadata.role,
            **analysis.rendering.model_dump(mode="json", exclude_none=True),
        }
        fingerprint_item = {
            "role": value.metadata.role,
            **analysis.fingerprint.model_dump(mode="json"),
        }
        result[native_id] = DeterministicEmojiAnalysis.model_validate(
            {
                "color_profile_sha256": analysis.color_profile_sha256,
                "dedupe_profile_sha256": analysis.dedupe_profile_sha256,
                "rendering": {
                    "profile": analysis.color_profile,
                    "items": [rendering_item],
                },
                "fingerprints": {
                    "status": "complete",
                    "profile": analysis.dedupe_profile,
                    "input_media_digest": domain_media_digest([value.dataset_metadata()]),
                    "items": [fingerprint_item],
                },
            }
        )
    return result


def _source_descriptor_sha256(source: SourceEmoji) -> str:
    return hashlib.sha256(rfc8785.dumps(cast(Any, source.safe_dump()))).hexdigest()


def _deterministic_render_context_sha256(source: SourceEmoji) -> str:
    """Hash source-controlled rendering inputs without changing the normative key."""

    return hashlib.sha256(
        rfc8785.dumps(
            {
                "format_version": 1,
                "needs_repainting": source.needs_repainting,
            }
        )
    ).hexdigest()


def _deterministic_cache_storage_key(deterministic_key: str, render_context_sha256: str) -> str:
    return f"analysis:{deterministic_key}:render-v1:{render_context_sha256}"


def _deterministic_key(value: ProcessedMedia) -> str:
    analysis = value.analysis
    if analysis is None:
        raise ValueError("processed media has no deterministic analysis")
    return deterministic_analysis_key(
        media_digest_value=domain_media_digest([value.dataset_metadata()]),
        pipeline_version=PIPELINE_VERSION,
        color_profile=analysis.color_profile,
        color_profile_sha256=analysis.color_profile_sha256,
        dedupe_profile=analysis.dedupe_profile,
        dedupe_profile_sha256=analysis.dedupe_profile_sha256,
        decoder_backend_fingerprint=analysis.decoder_backend_fingerprint,
    )


def _cache_deterministic_analyses(
    cache: CacheStore,
    source: SourceCollection,
    processed: Mapping[str, ProcessedMedia],
) -> None:
    for item in source.items:
        _cache_deterministic_analysis(cache, item, processed[item.native_id])


def _cache_deterministic_analysis(
    cache: CacheStore,
    item: SourceEmoji,
    value: ProcessedMedia,
) -> None:
    analysis = value.analysis
    if analysis is None:
        raise ValueError("processed media has no deterministic analysis")
    key = _deterministic_key(value)
    render_context_sha256 = _deterministic_render_context_sha256(item)
    frame_count = value.semantic_frame_count
    if frame_count < 1:
        raise ValueError("processed media has no rendered frames")
    if item.needs_repainting and not value.semantic_has_dark_render:
        raise ValueError("processed media render context does not match the source")
    cache.put_metadata(
        _deterministic_cache_storage_key(key, render_context_sha256),
        {
            "format_version": 3,
            "pipeline_version": PIPELINE_VERSION,
            "deterministic_cache_key": key,
            "render_context_sha256": render_context_sha256,
            "metadata": value.dataset_metadata(),
            "analysis": analysis.model_dump(mode="json"),
            "render_context": {
                "frame_count": frame_count,
                "background_variants": (
                    ["light", "dark"] if value.semantic_has_dark_render else ["light"]
                ),
            },
        },
    )


def _default_generation_metadata(config: MojiLexConfig) -> SemanticGenerationMetadata:
    if config.ai.provider != "gemini":
        raise CommandError(
            "CONFIG_INVALID",
            f"No provenance parameter profile exists for provider {config.ai.provider}.",
            hint="Use the supported gemini provider or add a versioned provider profile.",
        )
    return SemanticGenerationMetadata(
        provider=config.ai.provider,
        model=config.ai.model,
        prompt_sha256=prompt_sha256(),
        request_parameters_sha256=gemini_request_parameters_sha256(),
        generated_at=_utc_text(),
    )


def _generation_from_existing(
    emoji: Emoji,
    config: MojiLexConfig,
) -> SemanticGenerationMetadata:
    provenance = emoji.provenance
    if provenance.origin.value in {"ai", "mixed"}:
        return SemanticGenerationMetadata(
            provider=cast(str, provenance.provider),
            model=cast(str, provenance.model),
            model_revision=provenance.model_revision,
            description_profile=cast(str, provenance.description_profile),
            prompt_sha256=cast(str, provenance.prompt_sha256),
            request_parameters_sha256=cast(str, provenance.request_parameters_sha256),
            qualification_id=provenance.qualification_id,
            generation_stage=cast(Any, provenance.generation_stage).value,
            routing_policy_version=cast(str, provenance.routing_policy_version),
            routing_reason_codes=tuple(
                reason.value for reason in (provenance.routing_reason_codes or [])
            ),
            generated_at=provenance.generated_at,
            **{name: getattr(provenance, name) for name in CONCEPT_BINDING_FIELDS},
        )
    # The merge preserves human provenance whenever these existing semantics are
    # protected.  A complete placeholder is still required by the transform API.
    return _default_generation_metadata(config)


def _build_sheets_in_directory(
    inputs: Sequence[ContactSheetInput], output_dir: Path
) -> tuple[ContactSheet, ...]:
    output_dir.mkdir(parents=True, exist_ok=False)
    return build_contact_sheets(inputs, output_dir)


def _ai_request_plan_sha256(
    items: Sequence[SourceEmoji],
    processed: Mapping[str, ProcessedMedia],
    *,
    model: str,
) -> str:
    payload = {
        "format_version": 1,
        "model": model,
        "items": [
            {
                "label": f"E{index:03d}",
                "native_id": item.native_id,
                "source_descriptor_sha256": _source_descriptor_sha256(item),
                "deterministic_cache_key": _deterministic_key(processed[item.native_id]),
                "context": _vision_context(item, processed[item.native_id]).model_dump(mode="json"),
            }
            for index, item in enumerate(items, start=1)
        ],
    }
    if _generation_binding_fields():
        payload["concept_generation_binding"] = _generation_binding_fields()
    return hashlib.sha256(rfc8785.dumps(cast(Any, payload))).hexdigest()


async def _prepare_ai_request(
    items: Sequence[SourceEmoji],
    processed: Mapping[str, ProcessedMedia],
    *,
    model: str,
    temporary: TemporaryMediaRun,
) -> _PreparedAIRequest:
    if not items:
        raise ValueError("AI request requires at least one item")
    inputs = [
        ContactSheetInput(identifier=item.native_id, media=processed[item.native_id])
        for item in items
    ]
    sheets = await asyncio.to_thread(
        _build_sheets_in_directory,
        inputs,
        temporary.output_dir(),
    )
    temporary.account_outputs(sheet.path for sheet in sheets)
    labels = expected_labels(sheets)
    label_to_native: dict[str, str] = {}
    for sheet in sheets:
        for label, native_id in sheet.mapping.items():
            previous = label_to_native.setdefault(label, native_id)
            if previous != native_id:
                raise AIOutputError("contact-sheet label maps to multiple source items")
    expected_native_ids = {item.native_id for item in items}
    if set(label_to_native.values()) != expected_native_ids:
        raise AIOutputError("contact-sheet mapping does not cover the full request batch")
    native_to_label = {native_id: label for label, native_id in label_to_native.items()}
    if len(native_to_label) != len(items):
        raise AIOutputError("contact-sheet request contains duplicate native IDs")
    image_data = await asyncio.gather(
        *(asyncio.to_thread(sheet.path.read_bytes) for sheet in sheets)
    )
    contexts = {
        label: _vision_context(
            next(item for item in items if item.native_id == native_id),
            processed[native_id],
        )
        for label, native_id in label_to_native.items()
    }
    images = tuple(
        VisionImage(
            data=data,
            labels=tuple(sheet.mapping),
            variant=cast(Literal["light", "dark"], sheet.variant),
        )
        for sheet, data in zip(sheets, image_data, strict=True)
    )
    generation_inputs = _GENERATION_INPUTS.get()
    request = DescriptionRequest(
        model=model,
        images=images,
        expected_labels=labels,
        context=contexts,
        concept_context=(
            generation_inputs.concepts.prompt_context if generation_inputs is not None else None
        ),
    )
    shown_media_sha256 = tuple(hashlib.sha256(image.data).hexdigest() for image in images)
    request_payload = {
        "format_version": 1,
        "model": model,
        "expected_labels": list(labels),
        "images": [
            {
                "sha256": digest,
                "labels": list(image.labels),
                "variant": image.variant,
            }
            for image, digest in zip(images, shown_media_sha256, strict=True)
        ],
        "contexts": {label: contexts[label].model_dump(mode="json") for label in labels},
    }
    if _generation_binding_fields():
        request_payload["concept_generation_binding"] = _generation_binding_fields()
    identity = _AIRequestIdentity(
        plan_sha256=_ai_request_plan_sha256(items, processed, model=model),
        request_sha256=hashlib.sha256(rfc8785.dumps(cast(Any, request_payload))).hexdigest(),
        shown_media_sha256=shown_media_sha256,
        labels_by_native=tuple((item.native_id, native_to_label[item.native_id]) for item in items),
    )
    return _PreparedAIRequest(request=request, identity=identity)


def _resume_request_identity(
    items: Sequence[SourceEmoji],
    processed: Mapping[str, ProcessedMedia],
    traces: Mapping[str, Sequence[_AICacheTrace]],
    *,
    model: str,
    stage: Literal["primary", "escalated"],
) -> _AIRequestIdentity | None:
    selected: list[_AICacheTrace] = []
    for item in items:
        matches = [
            trace
            for trace in traces.get(item.native_id, ())
            if trace.stage == stage and trace.model == model
        ]
        if len(matches) != 1:
            return None
        selected.append(matches[0])
    first = selected[0].request_identity
    if len(items) > 1 and len({trace.model_revision for trace in selected}) != 1:
        return None
    if any(
        trace.request_identity.plan_sha256 != first.plan_sha256
        or trace.request_identity.request_sha256 != first.request_sha256
        or trace.request_identity.shown_media_sha256 != first.shown_media_sha256
        for trace in selected[1:]
    ):
        return None
    labels = tuple(
        (item.native_id, selected[index].request_identity.label_for(item.native_id))
        for index, item in enumerate(items)
    )
    if tuple(label for _, label in labels) != tuple(
        f"E{index:03d}" for index in range(1, len(items) + 1)
    ):
        return None
    identity = _AIRequestIdentity(
        plan_sha256=first.plan_sha256,
        request_sha256=first.request_sha256,
        shown_media_sha256=first.shown_media_sha256,
        labels_by_native=labels,
    )
    if identity.plan_sha256 != _ai_request_plan_sha256(items, processed, model=model):
        return None
    return identity


def _cache_key(
    source: SourceEmoji,
    processed: ProcessedMedia,
    context: VisionContext,
    config: MojiLexConfig,
    *,
    model: str,
    model_revision: str | None,
    taxonomy_version: str,
    request_identity: _AIRequestIdentity | None = None,
    item_label: str = "E001",
) -> str:
    digest = cache_media_digest(
        cast(Sequence[Mapping[str, object]], [processed.dataset_metadata()])
    )
    shown_media_sha256: tuple[str, ...]
    if request_identity is None:
        request_identity_sha256 = hashlib.sha256(
            rfc8785.dumps(
                {
                    "format_version": 0,
                    "media_digest": digest,
                    "context": context.model_dump(mode="json"),
                    "item_label": item_label,
                }
            )
        ).hexdigest()
        shown_media_sha256 = (processed.metadata.sha256,)
    else:
        request_identity_sha256 = request_identity.cache_identity_sha256
        shown_media_sha256 = request_identity.shown_media_sha256
        if request_identity.label_for(source.native_id) != item_label:
            raise CacheError("AI request item label does not match its request identity")
    return ai_cache_key(
        media_digest_value=digest,
        provider=config.ai.provider,
        model=model,
        model_revision=model_revision,
        prompt_version=PROMPT_VERSION,
        prompt_sha256=prompt_sha256(),
        schema_version=SCHEMA_VERSION,
        pipeline_version=PIPELINE_VERSION,
        languages=("ru", "en"),
        canonical_context_hash_value=canonical_context_hash(context),
        description_profile="standard-v1",
        taxonomy_version=taxonomy_version,
        routing_policy_version="1.0.0",
        shown_media_sha256=shown_media_sha256,
        request_parameters_sha256=gemini_request_parameters_sha256(),
        **_generation_binding_fields(),
        request_identity_sha256=request_identity_sha256,
        item_label=item_label,
    )


def _existing_emoji(snapshot: DatasetSnapshot, platform: str, native_id: str) -> Emoji | None:
    matches = [
        value
        for value in snapshot.emojis.values()
        if value.platform == platform and value.native_id == native_id
    ]
    return max(matches, key=lambda value: value.identity_epoch) if matches else None


def _description_from_existing(emoji: Emoji) -> DescriptionItem:
    def localized(language: str) -> AILocalizedDescription:
        value = emoji.descriptions[language]
        return AILocalizedDescription(
            text=value.text,
            motion_status=cast(Any, value.motion_status.value),
            motion=value.motion,
            usage=tuple(value.usage),
        )

    return DescriptionItem(
        label="E001",
        concept_ids=tuple(emoji.concept_ids),
        descriptions=BilingualDescriptions(ru=localized("ru"), en=localized("en")),
        facets=SemanticFacets.model_validate(
            {
                "text_content": emoji.facets.text_content.as_dict(),
                "content_types": [value.value for value in emoji.facets.content_types],
                "styles": [value.value for value in emoji.facets.styles],
                "suggested_uses": [value.value for value in emoji.facets.suggested_uses],
                "uncertainties": [value.value for value in emoji.facets.uncertainties],
            }
        ),
        semantic_tags=tuple(emoji.semantic_tags),
        content=ContentClassification(
            rating=cast(Any, emoji.content.rating.value),
            warnings=tuple(cast(Any, item.value) for item in emoji.content.warnings),
        ),
    )


def _accumulate_preview(
    totals: dict[str, int],
    snapshot: DatasetSnapshot,
    source: SourceCollection,
    *,
    processed: Mapping[str, ProcessedMedia] | None = None,
) -> None:
    existing_by_native = {
        value.native_id: value
        for value in snapshot.collections.values()
        if value.platform == source.platform
    }
    existing_collections = set(existing_by_native)
    totals["collections_created"] += int(source.native_id not in existing_collections)
    totals["collections_updated"] += int(source.native_id in existing_collections)
    for item in source.items:
        existing = _existing_emoji(snapshot, source.platform, item.native_id)
        if existing is None:
            totals["items_added"] += 1
        elif processed is not None:
            digest = cache_media_digest(
                cast(Sequence[Mapping[str, object]], [processed[item.native_id].dataset_metadata()])
            )
            changed = domain_media_digest(existing.media) != digest
            totals["items_updated"] += int(changed)
            totals["items_unchanged"] += int(not changed)
        else:
            extension = existing.extensions.get("telegram", {})
            changed = extension.get("file_unique_id") != item.file_unique_id
            totals["items_updated"] += int(changed)
            totals["items_unchanged"] += int(not changed)
    existing_collection = existing_by_native.get(source.native_id)
    if existing_collection is not None:
        old_native_ids = {
            snapshot.emojis[membership.emoji_id].native_id
            for membership in snapshot.memberships.values()
            if membership.collection_id == existing_collection.id
            and membership.status.value == "active"
            and membership.emoji_id in snapshot.emojis
        }
        disappeared = len(old_native_ids - {item.native_id for item in source.items})
        totals["items_disappeared"] += disappeared
        totals["memberships_removed"] += disappeared


async def _mark_missing_collection(
    snapshot: DatasetSnapshot,
    adapter: Any,
    *,
    platform: str,
    native_id: str,
) -> tuple[DatasetSnapshot, int]:
    matches = [
        value
        for value in snapshot.collections.values()
        if value.platform == platform and value.native_id == native_id
    ]
    if not matches:
        raise SourceNotFoundError("the selected collection does not exist in source or dataset")
    collection = max(matches, key=lambda value: value.identity_epoch)
    after = snapshot.clone()
    now = _utc_text()
    changed = 0
    target_collection = after.collections[collection.id]
    previous = target_collection.availability.model_dump(mode="python")
    target_collection.availability = _verified_availability(
        target_collection.availability, active=False, now=now
    )
    changed += int(target_collection.availability.model_dump(mode="python") != previous)

    emoji_ids = {
        membership.emoji_id
        for membership in snapshot.memberships.values()
        if membership.collection_id == collection.id and membership.emoji_id in snapshot.emojis
    }
    by_native = {snapshot.emojis[value].native_id: value for value in emoji_ids}
    availability = await adapter.check_emoji_availability(tuple(sorted(by_native)))
    for emoji_native_id, available in availability.items():
        emoji_id = by_native.get(emoji_native_id)
        if emoji_id is None:
            continue
        emoji = after.emojis[emoji_id]
        previous = emoji.availability.model_dump(mode="python")
        emoji.availability = _verified_availability(emoji.availability, active=available, now=now)
        changed += int(emoji.availability.model_dump(mode="python") != previous)
    return after, changed


async def _refresh_emoji_availability(
    snapshot: DatasetSnapshot,
    previous_snapshot: DatasetSnapshot,
    source: SourceCollection,
    adapter: Any,
) -> tuple[DatasetSnapshot, int]:
    after = snapshot.clone()
    source_native_ids = {item.native_id for item in source.items}
    collection_ids = {
        collection.id
        for collection in previous_snapshot.collections.values()
        if collection.platform == source.platform and collection.native_id == source.native_id
    }
    for membership in previous_snapshot.memberships.values():
        if membership.collection_id in collection_ids:
            emoji = previous_snapshot.emojis.get(membership.emoji_id)
            if emoji is not None:
                source_native_ids.add(emoji.native_id)
    result = await adapter.check_emoji_availability(tuple(sorted(source_native_ids)))
    now = _utc_text()
    changed = 0
    for emoji in after.emojis.values():
        if emoji.platform != source.platform or emoji.native_id not in result:
            continue
        previous = emoji.availability.model_dump(mode="python")
        emoji.availability = _verified_availability(
            emoji.availability, active=result[emoji.native_id], now=now
        )
        changed += int(emoji.availability.model_dump(mode="python") != previous)
    return after, changed


def _verified_availability(value: Any, *, active: bool, now: str) -> Any:
    payload = value.model_dump(mode="python")
    new_status = "active" if active else "unavailable"
    payload.update(
        {
            "status": new_status,
            "last_changed_at": value.last_changed_at if value.status.value == new_status else now,
            "last_verified_at": now,
            "reason_code": None if active else "source_not_found",
            "set_by": None,
        }
    )
    return type(value).model_validate(payload)


def _utc_text() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _changed_paths(before: DatasetSnapshot, after: DatasetSnapshot) -> tuple[PurePosixPath, ...]:
    old, new = before.to_files(), after.to_files()
    return tuple(
        sorted(
            (path for path in set(old) | set(new) if old.get(path) != new.get(path)),
            key=str,
        )
    )


def _changed_entity_ids(before: DatasetSnapshot, after: DatasetSnapshot) -> tuple[str, ...]:
    changed: set[str] = set()
    for attribute in ("collections", "emojis", "memberships", "relations", "tombstones"):
        old_values = cast(Mapping[str, object], getattr(before, attribute))
        new_values = cast(Mapping[str, object], getattr(after, attribute))
        changed.update(
            entity_id
            for entity_id in set(old_values) | set(new_values)
            if old_values.get(entity_id) != new_values.get(entity_id)
        )
    return tuple(sorted(changed))


def _require_confirmation(
    callback: Callable[[str], bool] | None,
    message: str,
    *,
    hint: str,
) -> bool:
    if callback is None:
        raise CommandError(
            "CONFIG_INVALID",
            "The exact sensitive-operation plan requires explicit confirmation.",
            hint=hint,
        )
    confirmed = callback(message)
    if confirmed is not True:
        raise CommandError(
            "CONFIG_INVALID",
            "The exact sensitive-operation plan was not confirmed.",
            hint=hint,
        )
    return confirmed


def _confirm_direct_push(
    callback: Callable[[str], bool] | None,
    git: GitRunner,
    prepared: PreparedCommit,
    *,
    target: RepositoryRef,
    base_branch: str,
) -> bool:
    exact_paths = git.changed_paths_between(prepared.base_sha, prepared.commit_sha)
    if not exact_paths or any(not _submit_path_allowed(path) for path in exact_paths):
        raise CommandError(
            "GIT_CONFLICT",
            "The exact direct-push commit diff is empty or contains a forbidden path.",
            hint="Inspect the validated commit and submit only data/ or tombstones/ paths.",
            details={"commit": prepared.commit_sha, "paths": list(exact_paths)},
        )
    return _require_confirmation(
        callback,
        (
            f"Publish exact validated commit {prepared.commit_sha} to {target}:{base_branch}? "
            f"Candidate branch: {prepared.branch}. "
            f"Exact changed paths: {', '.join(exact_paths)}."
        ),
        hint=(
            "Review the commit ID and every changed path, then confirm interactively or rerun "
            "with --yes."
        ),
    )


def _apply_with_rollback(
    before: DatasetSnapshot,
    after: DatasetSnapshot,
    *,
    allow_policy_review: bool = False,
) -> None:
    staged_report = validate_snapshot(after, schemas=True, repository_files=True)
    if not staged_report.valid and not (
        allow_policy_review and _only_staging_review_issues(staged_report)
    ):
        staged_report.raise_for_errors()
    apply_snapshot(before, after)
    report = validate_dataset(before.root, strict=True)
    if report.valid or (allow_policy_review and _only_staging_review_issues(report)):
        return
    apply_snapshot(after, before, validator=validate_snapshot)
    report.raise_for_errors()


def _only_staging_review_issues(report: Any) -> bool:
    """Return whether every failure is an expected, human-resolvable staging gate.

    Structural qualification-registry failures remain fatal.  Only an emoji-level
    missing/mismatched qualification can be held in staging for human approval.
    """

    issues = tuple(getattr(report, "issues", ()))
    allowed = {
        "POLICY_REVIEW",
        "POLICY_REVIEW_BLOCKING",
        "QUALIFICATION",
        "STYLE_CONFLICT",
    }
    return bool(issues) and all(getattr(issue, "code", None) in allowed for issue in issues)


def _validation_warnings(report: Any) -> list[dict[str, str]]:
    return [
        {
            "code": str(getattr(issue, "code", "VALIDATION_WARNING")),
            "path": str(getattr(issue, "path", "")),
            "message": str(getattr(issue, "message", "Manual review is required.")),
        }
        for issue in getattr(report, "issues", ())
    ]


def _remaining_source_entries(
    sources: Sequence[str], completed_source_indexes: set[int]
) -> tuple[tuple[int, str], ...]:
    return tuple(
        (index, source)
        for index, source in enumerate(sources)
        if index not in completed_source_indexes
    )


def _checkpoint_publication_progress(
    checkpoint: RunCheckpoint,
    publication: PublicationCheckpoint,
) -> RunCheckpoint:
    if checkpoint.base_revision not in {
        publication.expected_old_base,
        publication.candidate_sha,
    }:
        raise CommandError(
            "CONFIG_INVALID",
            "Publication progress does not match the run base revision.",
            hint="Do not edit run checkpoints; restart from a verified remote base.",
        )
    updates: dict[str, object] = {
        "publication": publication,
        "updated_at": datetime.now(UTC).replace(microsecond=0),
    }
    if publication.mode == "direct" and publication.phase == "completed":
        updates["base_revision"] = publication.candidate_sha
    return checkpoint.model_copy(update=updates)


def _reconcile_publication_for_resume(
    checkpoint: RunCheckpoint,
    *,
    git: GitRunner,
    remote_base: str,
    base_branch: str,
    expected_mode: Literal["pr", "direct"],
    source_count: int,
) -> tuple[RunCheckpoint, set[int]]:
    publication = checkpoint.publication
    if publication is None:
        return checkpoint, set()
    if publication.mode != expected_mode or publication.base_branch != base_branch:
        raise CommandError(
            "CONFIG_INVALID",
            "The publication checkpoint does not match the resumed publication mode.",
            hint="Resume with the repository mode and base branch recorded by the original run.",
        )
    if expected_mode == "direct" and publication.remote != "origin":
        raise CommandError(
            "CONFIG_INVALID",
            "A direct publication checkpoint must target the origin remote.",
            hint="Do not edit run checkpoints; start a new run if the remote changed.",
        )
    if any(index >= source_count for index in publication.completed_source_indexes):
        raise CommandError(
            "CONFIG_INVALID",
            "The publication checkpoint refers to a source outside the saved source plan.",
            hint="Do not edit run checkpoints; start a new run from the original source list.",
        )
    if checkpoint.base_revision not in {
        publication.expected_old_base,
        publication.candidate_sha,
    }:
        raise CommandError(
            "CONFIG_INVALID",
            "The publication checkpoint does not match the run base revision.",
            hint="Inspect the checkpoint and remote refs before retrying.",
        )
    if remote_base == publication.candidate_sha:
        reconciled = checkpoint.model_copy(
            update={
                "base_revision": publication.candidate_sha,
                "publication": publication.model_copy(update={"phase": "completed"}),
                "updated_at": datetime.now(UTC).replace(microsecond=0),
            }
        )
        return reconciled, set(publication.completed_source_indexes)
    if remote_base != publication.expected_old_base:
        raise CommandError(
            "GIT_CONFLICT",
            "The remote base is neither the expected old base nor the published candidate.",
            hint="Inspect the remote base and candidate refs; do not overwrite either ref.",
            details={
                "expected_old_base": publication.expected_old_base,
                "candidate_sha": publication.candidate_sha,
                "actual_base": remote_base,
            },
        )
    if checkpoint.base_revision != publication.expected_old_base:
        raise CommandError(
            "GIT_CONFLICT",
            "The saved run base was advanced but the remote base no longer contains it.",
            hint="Inspect the remote history manually; automatic resume is unsafe.",
        )
    remotes = {line.strip() for line in git.run("remote").stdout.splitlines() if line.strip()}
    if publication.remote in remotes:
        candidate = git.optional_remote_sha(
            publication.remote,
            publication.candidate_branch,
        )
        if candidate not in {None, publication.candidate_sha}:
            raise CommandError(
                "GIT_CONFLICT",
                "The remote candidate branch no longer matches the saved publication intent.",
                hint="Inspect the candidate branch manually; automatic resume will not replace it.",
            )
    return checkpoint, set()


def _same_publication_attempt(
    previous: PublicationCheckpoint | None,
    prepared: PreparedCommit,
    *,
    mode: Literal["pr", "direct"],
    remote: str,
    base_branch: str,
) -> bool:
    return bool(
        previous is not None
        and previous.mode == mode
        and previous.remote == remote
        and previous.base_branch == base_branch
        and previous.expected_old_base == prepared.base_sha
        and previous.candidate_branch == prepared.branch
    )


def _validate_previous_candidate_ref(
    previous: PublicationCheckpoint | None,
    prepared: PreparedCommit,
    *,
    git: GitRunner,
    mode: Literal["pr", "direct"],
    remote: str,
    base_branch: str,
) -> None:
    if not _same_publication_attempt(
        previous,
        prepared,
        mode=mode,
        remote=remote,
        base_branch=base_branch,
    ):
        return
    assert previous is not None
    remote_candidate = git.optional_remote_sha(remote, prepared.branch)
    if remote_candidate not in {None, previous.candidate_sha}:
        raise CommandError(
            "GIT_CONFLICT",
            "The remote candidate branch differs from the saved publication checkpoint.",
            hint="Inspect the candidate branch manually; automatic resume will not replace it.",
        )


async def _publish(
    workspace: RepositoryWorkspace,
    config: MojiLexConfig,
    options: PipelineOptions,
    git: GitRunner,
    publisher: GitPublisher,
    guard: Any,
    paths: tuple[PurePosixPath, ...],
    sources: Sequence[SourceCollection],
    run_id: str,
    totals: Mapping[str, int],
    github_token: str | None,
    review_routing: ReviewRoutingReport,
    *,
    completed_source_indexes: tuple[int, ...],
    previous_publication: PublicationCheckpoint | None,
    record_publication: Callable[[PublicationCheckpoint], None],
) -> dict[str, Any]:
    if config.repository.publish == "local" and not options.direct_push:
        return {"mode": "local", "path": str(workspace.root)}
    github = GitHubCLI(token=github_token)
    github.auth_status()
    target = workspace.target
    info = github.repository_info(target)
    branch = make_import_branch(
        pack_name=sources[0].native_id if len(sources) == 1 else None,
        run_id=run_id,
        batch=len(sources) != 1,
    )
    if len(sources) == 1:
        message = f"data(telegram): add {sources[0].native_id} custom emoji pack"
    elif sources:
        message = f"data(telegram): update {len(sources)} custom emoji packs"
    else:
        message = "data(telegram): update verified source availability"
    prepared = publisher.prepare_commit(
        paths=tuple(str(path) for path in paths),
        message=message,
        branch=branch,
        base_revision="HEAD",
        identity=_configured_git_identity(config),
        guard=guard,
    )
    if prepared is None:
        return {"mode": "local", "status": "noop"}
    service = GitHubPublisher(git, github)
    mode: Literal["pr", "direct"] = "direct" if options.direct_push else "pr"
    intent: PublicationCheckpoint | None = None

    def persist_intent(
        candidate: PreparedCommit,
        *,
        remote: str,
        phase: Literal["prepared", "candidate_pushed", "checks_passed", "completed"] = "prepared",
    ) -> None:
        nonlocal intent
        intent = PublicationCheckpoint(
            mode=mode,
            remote=remote,
            base_branch=config.repository.base_branch,
            expected_old_base=candidate.base_sha,
            candidate_sha=candidate.commit_sha,
            candidate_branch=candidate.branch,
            phase=phase,
            completed_source_indexes=completed_source_indexes,
        )
        record_publication(intent)

    def record_phase(phase: PublicationPhase) -> None:
        nonlocal intent
        if intent is None:
            raise RuntimeError("publication phase cannot precede its durable intent")
        intent = intent.model_copy(update={"phase": phase})
        record_publication(intent)

    if options.direct_push:
        _validate_previous_candidate_ref(
            previous_publication,
            prepared,
            git=git,
            mode="direct",
            remote="origin",
            base_branch=config.repository.base_branch,
        )
        prepared = publisher.reconcile_remote_branch(
            prepared,
            remote="origin",
            path_is_allowed=_submit_path_allowed,
        )
        persist_intent(prepared, remote="origin")
        required = github.required_checks(target, config.repository.base_branch)
        bypass = github.has_direct_push_bypass(target, config.repository.base_branch)
        user_confirmed = _confirm_direct_push(
            options.confirmation,
            git,
            prepared,
            target=target,
            base_branch=config.repository.base_branch,
        )
        commit = await service.publish_direct(
            prepared,
            target=target,
            repository_info=info,
            remote="origin",
            base_branch=config.repository.base_branch,
            candidate_branch=prepared.branch,
            required_checks=required,
            authorization=DirectPushAuthorization(
                explicit_flag=True,
                user_confirmed=user_confirmed,
                validation_succeeded=True,
                bypass_verified=bypass,
            ),
            progress=record_phase,
        )
        return {"mode": "direct", "commit": commit, "candidate_branch": prepared.branch}
    fork = target
    fork_remote = "origin"
    if not info.can_write:
        login, _ = github.current_user()
        expected_fork = RepositoryRef(owner=login, name=target.name)
        fork_remote = "mojilex-fork"
        if not _same_publication_attempt(
            previous_publication,
            prepared,
            mode="pr",
            remote=fork_remote,
            base_branch=config.repository.base_branch,
        ):
            persist_intent(prepared, remote=fork_remote)
        fork = github.ensure_fork(target)
        if fork != expected_fork:
            raise CommandError(
                "GIT_CONFLICT",
                "GitHub created or selected a different fork than the publication intent.",
                hint="Inspect the authenticated GitHub account and retry the same run.",
            )
        existing = git.run("remote", check=False).stdout.splitlines()
        if fork_remote not in existing:
            git.run("remote", "add", fork_remote, f"https://github.com/{fork}.git")
    _validate_previous_candidate_ref(
        previous_publication,
        prepared,
        git=git,
        mode="pr",
        remote=fork_remote,
        base_branch=config.repository.base_branch,
    )
    prepared = publisher.reconcile_remote_branch(
        prepared,
        remote=fork_remote,
        path_is_allowed=_submit_path_allowed,
    )
    persist_intent(prepared, remote=fork_remote)
    title = message
    body = _pull_request_body(sources, totals, config, run_id, review_routing)
    pull_request = service.publish_pr(
        prepared,
        target=target,
        fork=fork,
        fork_remote=fork_remote,
        base_branch=config.repository.base_branch,
        title=title,
        body=body,
        progress=record_phase,
    )
    return {
        "mode": "pr",
        "pull_request_url": pull_request.url,
        "pull_request_number": pull_request.number,
        "reused": pull_request.reused,
        "completed": pull_request.completed,
        "reopened": pull_request.reopened,
        "commit": prepared.commit_sha,
    }


def _pull_request_body(
    sources: Sequence[SourceCollection],
    totals: Mapping[str, int],
    config: MojiLexConfig,
    run_id: str,
    review_routing: ReviewRoutingReport,
) -> str:
    source_lines = "\n".join(f"- {item.canonical_url}" for item in sources)
    collection_lines = "\n".join(
        f"- `{item.native_id}` ({item.item_count} items)" for item in sources
    )
    review_counts = review_routing.counts()
    review_line = (
        f"blocking={review_counts['blocking']}, high={review_counts['high']}, "
        f"normal={review_counts['normal']}, low={review_counts['low']}"
    )
    return f"""## MojiLex import

Sources:
{source_lines}

Collections:
{collection_lines}

- Added emoji: {totals["items_added"]}
- Updated emoji: {totals["items_updated"]}
- Removed memberships: {totals["memberships_removed"]}
- Languages: ru, en
- CLI/schema: {__version__} / {SCHEMA_VERSION}
- AI: {config.ai.provider} / {config.ai.model}
- Prompt: {PROMPT_VERSION}
- Run: `{run_id}`
- Review routing: {review_line}

Validation passed. The change contains metadata only: no source media, download URLs, or secrets.
New AI descriptions are intentionally `unreviewed` unless policy requires approval before publish.
"""


@contextmanager
def repository_workspace(
    target: str,
    base_branch: str,
    *,
    isolated: bool = False,
    github_token: str | None = None,
) -> Iterator[RepositoryWorkspace]:
    if github_token is None:
        github_token = load_credentials().github_token
    path = Path(target).expanduser()
    if path.is_dir() and not isolated:
        root = path.resolve()
        git = GitRunner(root)
        reference = _reference_from_remote(git.remote_url())
        yield RepositoryWorkspace(root=root, target=reference, temporary=False)
        return
    reference = (
        _reference_from_remote(GitRunner(path.resolve()).remote_url())
        if path.is_dir()
        else RepositoryRef.parse(target)
    )
    with tempfile.TemporaryDirectory(prefix="mojilex-repository-") as raw:
        root = Path(raw) / reference.name
        with git_subprocess_environment(github_token) as environment:
            completed = subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-tags",
                    "--single-branch",
                    "--branch",
                    base_branch,
                    f"https://github.com/{reference}.git",
                    str(root),
                ],
                capture_output=True,
                check=False,
                timeout=120,
                shell=False,
                env=environment,
            )
        if completed.returncode != 0:
            raise CommandError(
                "GIT_CONFLICT",
                "Could not create an isolated checkout of the target dataset.",
                hint="Check Git credentials, repository access, and the configured base branch.",
            )
        yield RepositoryWorkspace(root=root, target=reference, temporary=True)


def _reference_from_remote(value: str) -> RepositoryRef:
    if value.startswith("git@github.com:"):
        value = "https://github.com/" + value.removeprefix("git@github.com:")
    return RepositoryRef.parse(value)


def _safe_parameters(sources: Sequence[str], options: PipelineOptions) -> dict[str, object]:
    return {
        "sources": list(sources),
        "repository": options.repository or "",
        "platform": options.platform,
        "provider": options.provider or "",
        "model": options.model or "",
        "languages": list(options.languages),
        "publish": options.publish or "",
        "direct_push": options.direct_push,
        "base": options.base or "",
        "redescribe": options.redescribe,
        "overwrite_reviewed": options.overwrite_reviewed,
        "new_identity": options.new_identity,
        "same_identity": options.same_identity,
        "max_items": options.max_items,
        "max_ai_requests": options.max_ai_requests,
        "max_cost_usd": str(options.max_cost_usd) if options.max_cost_usd is not None else "",
        "allow_unknown_cost": options.allow_unknown_cost,
        "ai_concurrency": options.ai_concurrency,
        "download_concurrency": options.download_concurrency,
        "dedupe": options.dedupe or "",
        "max_dedupe_candidates": options.max_dedupe_candidates,
        "dedupe_profile": options.dedupe_profile or "",
        "model_routing": options.model_routing or "",
        "escalation_model": options.escalation_model or "",
        "check_media": options.check_media,
        "fail_fast": options.fail_fast,
        "explicit_verification": options.explicit_verification,
    }


def _materialized_options(
    options: PipelineOptions,
    config: MojiLexConfig,
) -> PipelineOptions:
    return replace(
        options,
        repository=config.repository.target,
        provider=config.ai.provider,
        model=config.ai.model,
        languages=tuple(config.ai.languages),
        publish=config.repository.publish,
        base=config.repository.base_branch,
        max_ai_requests=config.ai.max_ai_requests,
        max_cost_usd=config.ai.max_cost_usd,
        allow_unknown_cost=config.ai.allow_unknown_cost,
        ai_concurrency=config.ai.ai_concurrency,
        download_concurrency=config.telegram.download_concurrency,
        dedupe=config.dedupe.mode,
        max_dedupe_candidates=config.dedupe.max_candidates,
        dedupe_profile=config.dedupe.profile,
        model_routing=config.ai.model_routing,
        escalation_model=config.ai.escalation_model,
    )


def _options_from_safe(value: Mapping[str, object]) -> PipelineOptions:
    amount = value.get("max_cost_usd")
    return PipelineOptions(
        repository=_optional_string(value.get("repository")),
        platform=str(value.get("platform") or "auto"),
        provider=_optional_string(value.get("provider")),
        model=_optional_string(value.get("model")),
        languages=tuple(_string_sequence(value.get("languages"))),
        publish=_optional_string(value.get("publish")),
        direct_push=bool(value.get("direct_push", False)),
        base=_optional_string(value.get("base")),
        redescribe=str(value.get("redescribe") or "changed"),
        overwrite_reviewed=bool(value.get("overwrite_reviewed", False)),
        new_identity=bool(value.get("new_identity", False)),
        same_identity=bool(value.get("same_identity", False)),
        max_items=_optional_int(value.get("max_items")),
        max_ai_requests=_optional_int(value.get("max_ai_requests")),
        max_cost_usd=Decimal(str(amount)) if amount else None,
        allow_unknown_cost=bool(value.get("allow_unknown_cost", False)),
        ai_concurrency=_optional_int(value.get("ai_concurrency")),
        download_concurrency=_optional_int(value.get("download_concurrency")),
        dedupe=_optional_string(value.get("dedupe")),
        max_dedupe_candidates=_optional_int(value.get("max_dedupe_candidates")),
        dedupe_profile=_optional_string(value.get("dedupe_profile")),
        model_routing=_optional_string(value.get("model_routing")),
        escalation_model=_optional_string(value.get("escalation_model")),
        check_media=bool(value.get("check_media", False)),
        fail_fast=bool(value.get("fail_fast", False)),
        explicit_verification=bool(value.get("explicit_verification", False)),
    )


def _checkpoint_media(
    checkpoint: RunCheckpoint,
    source: SourceCollection,
    processed: Mapping[str, ProcessedMedia],
) -> RunCheckpoint:
    source_by_native = {item.native_id: item for item in source.items}
    for native_id, value in processed.items():
        checkpoint = _checkpoint_media_item(checkpoint, source_by_native[native_id], value)
    return checkpoint


def _checkpoint_media_item(
    checkpoint: RunCheckpoint,
    item: SourceEmoji,
    value: ProcessedMedia,
) -> RunCheckpoint:
    elements = dict(checkpoint.elements)
    descriptor_sha256 = _source_descriptor_sha256(item)
    media_sha256 = (value.metadata.sha256,)
    deterministic_key = _deterministic_key(value)
    previous = elements.get(item.native_id, ElementCheckpoint(stage="discovered"))
    changed = (
        previous.source_descriptor_sha256 != descriptor_sha256
        or previous.media_sha256 != media_sha256
        or previous.deterministic_cache_key != deterministic_key
    )
    updates: dict[str, object] = {
        "stage": "fingerprint_ready",
        "source_descriptor_sha256": descriptor_sha256,
        "media_sha256": media_sha256,
        "deterministic_cache_key": deterministic_key,
        "palette_complete": True,
        "fingerprint_complete": True,
        "candidate_scan_complete": False,
        "review_complete": False,
    }
    if changed:
        updates.update(
            {
                "ai_cache_key": None,
                "ai_requests": (),
                "ai_facets_complete": False,
            }
        )
    elements[item.native_id] = previous.model_copy(update=updates)
    return checkpoint.model_copy(
        update={"elements": elements, "updated_at": datetime.now(UTC).replace(microsecond=0)}
    )


def _canonical_snapshot_sha256(snapshot: DatasetSnapshot) -> str:
    files = snapshot.to_files()
    inventory = [
        {
            "path": path.as_posix(),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for path, content in sorted(files.items(), key=lambda item: item[0].as_posix())
    ]
    return hashlib.sha256(rfc8785.dumps(inventory)).hexdigest()


def _dedupe_scan_identity(
    snapshot: DatasetSnapshot,
    selected_emoji_ids: set[str],
    config: MojiLexConfig,
) -> tuple[str, str, str, tuple[str, ...]]:
    profile = load_analysis_profile(config.dedupe.profile)
    snapshot_sha256 = _canonical_snapshot_sha256(snapshot)
    selected = tuple(sorted(selected_emoji_ids))
    input_sha256 = hashlib.sha256(
        rfc8785.dumps(
            {
                "snapshot_sha256": snapshot_sha256,
                "selected_emoji_ids": selected,
                "mode": config.dedupe.mode,
                "max_candidates": config.dedupe.max_candidates,
                "profile": profile.profile_id,
                "profile_sha256": profile.sha256,
            }
        )
    ).hexdigest()
    return input_sha256, snapshot_sha256, profile.sha256, selected


def _resume_dedupe_selected_ids(
    checkpoint: RunCheckpoint | None,
    snapshot: DatasetSnapshot,
    selected_emoji_ids: set[str],
) -> set[str]:
    """Recover the immutable scan selection after a crash following dataset apply."""

    if selected_emoji_ids or checkpoint is None or checkpoint.dedupe_scan is None:
        return selected_emoji_ids
    saved = checkpoint.dedupe_scan
    if saved.snapshot_sha256 != _canonical_snapshot_sha256(snapshot):
        return selected_emoji_ids
    recovered = set(saved.selected_emoji_ids)
    if not recovered or any(emoji_id not in snapshot.emojis for emoji_id in recovered):
        return selected_emoji_ids
    return recovered


def _cached_dedupe_report(
    checkpoint: RunCheckpoint | None,
    snapshot: DatasetSnapshot,
    selected_emoji_ids: set[str],
    config: MojiLexConfig,
) -> dict[str, Any] | None:
    if checkpoint is None or checkpoint.dedupe_scan is None:
        return None
    saved = checkpoint.dedupe_scan
    input_sha256, snapshot_sha256, profile_sha256, selected = _dedupe_scan_identity(
        snapshot, selected_emoji_ids, config
    )
    if any(
        (
            saved.input_sha256 != input_sha256,
            saved.snapshot_sha256 != snapshot_sha256,
            saved.profile != config.dedupe.profile,
            saved.profile_sha256 != profile_sha256,
            saved.mode != config.dedupe.mode,
            saved.max_candidates != config.dedupe.max_candidates,
            saved.selected_emoji_ids != selected,
            saved.report_sha256
            != hashlib.sha256(rfc8785.dumps(cast(Any, saved.report))).hexdigest(),
            saved.report.get("profile") != config.dedupe.profile,
            saved.report.get("scan_mode") != config.dedupe.mode,
        )
    ):
        return None
    return dict(saved.report)


def _checkpoint_dedupe_report(
    checkpoint: RunCheckpoint,
    snapshot: DatasetSnapshot,
    selected_emoji_ids: set[str],
    config: MojiLexConfig,
    report: dict[str, Any],
) -> RunCheckpoint:
    input_sha256, snapshot_sha256, profile_sha256, selected = _dedupe_scan_identity(
        snapshot, selected_emoji_ids, config
    )
    persisted = DedupeScanCheckpoint(
        input_sha256=input_sha256,
        report_sha256=hashlib.sha256(rfc8785.dumps(report)).hexdigest(),
        snapshot_sha256=snapshot_sha256,
        profile=config.dedupe.profile,
        profile_sha256=profile_sha256,
        mode=cast(Literal["exact", "near"], config.dedupe.mode),
        max_candidates=config.dedupe.max_candidates,
        selected_emoji_ids=selected,
        report=cast(dict[str, object], report),
    )
    elements = dict(checkpoint.elements)
    for native_id, previous in elements.items():
        elements[native_id] = previous.model_copy(update={"candidate_scan_complete": False})
    for emoji_id in selected:
        emoji = snapshot.emojis.get(emoji_id)
        if emoji is None:
            continue
        native_id = emoji.native_id
        selected_checkpoint = elements.get(native_id)
        if selected_checkpoint is not None:
            elements[native_id] = selected_checkpoint.model_copy(
                update={"stage": "candidate_scanned", "candidate_scan_complete": True}
            )
    return checkpoint.model_copy(
        update={
            "dedupe_scan": persisted,
            "elements": elements,
            "updated_at": datetime.now(UTC).replace(microsecond=0),
        }
    )


def _checkpoint_stage(
    checkpoint: RunCheckpoint, native_ids: Sequence[str], stage: str
) -> RunCheckpoint:
    elements = dict(checkpoint.elements)
    for native_id in native_ids:
        previous = elements.get(native_id, ElementCheckpoint(stage="discovered"))
        updates: dict[str, object] = {"stage": stage}
        if stage in {
            "palette_ready",
            "fingerprint_ready",
            "ai_cached",
            "ai_facets_ready",
            "candidate_scanned",
            "reviewed",
            "mapped",
            "validated",
        }:
            updates["palette_complete"] = True
        if stage in {
            "fingerprint_ready",
            "ai_cached",
            "ai_facets_ready",
            "candidate_scanned",
            "reviewed",
            "mapped",
            "validated",
        }:
            updates["fingerprint_complete"] = True
        if stage in {
            "ai_cached",
            "ai_facets_ready",
            "candidate_scanned",
            "reviewed",
            "mapped",
            "validated",
        }:
            updates["ai_facets_complete"] = True
        # Candidate/review completion is intentionally not inferred from a
        # display stage. Those forward-compatible flags may become true only
        # when a separately persisted, validated result can actually be reused.
        elements[native_id] = previous.model_copy(update=updates)
    return checkpoint.model_copy(
        update={"elements": elements, "updated_at": datetime.now(UTC).replace(microsecond=0)}
    )


def _checkpoint_ai_keys(
    checkpoint: RunCheckpoint,
    source: SourceCollection,
    processed: Mapping[str, ProcessedMedia],
    config: MojiLexConfig,
    generation_metadata: Mapping[str, SemanticGenerationMetadata],
    *,
    taxonomy_version: str,
    request_traces: Mapping[str, tuple[_AICacheTrace, ...]],
) -> RunCheckpoint:
    del processed, config, taxonomy_version
    elements = dict(checkpoint.elements)
    for item in source.items:
        previous = elements.get(item.native_id, ElementCheckpoint(stage="media_verified"))
        trace = request_traces.get(item.native_id, ())
        if not trace:
            elements[item.native_id] = previous.model_copy(
                update={"ai_cache_key": None, "ai_requests": ()}
            )
            continue
        generation = generation_metadata[item.native_id]
        if (
            trace[-1].model != generation.model
            or trace[-1].model_revision != generation.model_revision
        ):
            raise ValueError("accepted AI request trace differs from public provenance")
        persisted = tuple(
            AIRequestCheckpoint(
                stage=request.stage,
                model=request.model,
                model_revision=request.model_revision,
                cache_key=request.cache_key,
                plan_sha256=request.request_identity.plan_sha256,
                request_sha256=request.request_identity.request_sha256,
                shown_media_sha256=request.request_identity.shown_media_sha256,
                item_label=request.request_identity.label_for(item.native_id),
            )
            for request in trace
        )
        elements[item.native_id] = previous.model_copy(
            update={"ai_cache_key": trace[-1].cache_key, "ai_requests": persisted}
        )
    return checkpoint.model_copy(
        update={"elements": elements, "updated_at": datetime.now(UTC).replace(microsecond=0)}
    )


def _checkpoint_budget(checkpoint: RunCheckpoint, budget: RequestBudget) -> RunCheckpoint:
    return _checkpoint_budget_values(
        checkpoint,
        requests_used=budget.requests_used,
        cost_reserved=budget.cost_reserved,
    )


def _checkpoint_budget_values(
    checkpoint: RunCheckpoint,
    *,
    requests_used: int,
    cost_reserved: Decimal,
) -> RunCheckpoint:
    return checkpoint.model_copy(
        update={
            "ai_requests_used": requests_used,
            "ai_cost_reserved_usd": cost_reserved,
            "updated_at": datetime.now(UTC).replace(microsecond=0),
        }
    )


def _persist_budget_reservation(
    checkpoint: RunCheckpoint,
    store: RunStore,
    *,
    requests_used: int,
    cost_reserved: Decimal,
) -> RunCheckpoint:
    updated = _checkpoint_budget_values(
        checkpoint,
        requests_used=requests_used,
        cost_reserved=cost_reserved,
    )
    store.save(updated)
    return updated


def _checkpoint_issue(
    checkpoint: RunCheckpoint, error: StructuredError, *, terminal: bool = False
) -> RunCheckpoint:
    issue = RunIssue.model_validate(error.as_dict())
    issues = checkpoint.issues
    if not issues or issues[-1] != issue:
        issues = (*issues, issue)
    terminal_status = checkpoint.status
    if terminal:
        if error.code in {"BUDGET_EXCEEDED", "UNKNOWN_COST", "AI_BUDGET_EXCEEDED"}:
            terminal_status = "budget_exceeded"
        elif error.code == "SOURCE_CHANGED_DURING_RUN":
            terminal_status = "stale"
        elif error.code == "INTERRUPTED":
            terminal_status = "interrupted"
        else:
            terminal_status = "failed"
    return checkpoint.model_copy(
        update={
            "issues": issues,
            "status": terminal_status,
            "updated_at": datetime.now(UTC).replace(microsecond=0),
        }
    )


def _finish_checkpoint(
    checkpoint: RunCheckpoint, status: str, requests: int, cost: Decimal
) -> RunCheckpoint:
    return checkpoint.model_copy(
        update={
            "status": status,
            "ai_requests_used": requests,
            "ai_cost_reserved_usd": cost,
            "updated_at": datetime.now(UTC).replace(microsecond=0),
        }
    )


def _string_sequence(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError("checkpoint source list is invalid")
    return tuple(cast(Sequence[str], value))


def _membership_map(value: object) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise CommandError(
            "CONFIG_INVALID",
            "Checkpoint collection membership data is invalid.",
            hint="Start a new import instead of editing checkpoint files.",
        )
    result: dict[str, tuple[str, ...]] = {}
    for raw_key, raw_items in value.items():
        if not isinstance(raw_key, str):
            raise CommandError(
                "CONFIG_INVALID",
                "Checkpoint collection membership key is invalid.",
                hint="Start a new import instead of editing checkpoint files.",
            )
        result[raw_key] = _string_sequence(raw_items)
    return result


def _staging_path_from_checkpoint(checkpoint: RunCheckpoint) -> Path:
    value = _optional_string(checkpoint.safe_parameters.get("staging_repository"))
    if value is None:
        raise CommandError(
            "CONFIG_INVALID",
            "The run has no persistent staging workspace.",
            hint="Start a new import with this CLI version.",
        )
    path = Path(value).expanduser().resolve()
    if not path.is_dir() or path.is_symlink():
        raise CommandError(
            "DIRTY_WORKTREE",
            "The run staging workspace is missing or unsafe.",
            hint="Restore the run workspace or start a new import.",
        )
    return path


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
