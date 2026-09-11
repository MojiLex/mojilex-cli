"""Local exact/near duplicate scanning, explanation, and reviewed relations."""

from __future__ import annotations

import asyncio
import itertools
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast
from uuid import UUID

from mojilex_cli.commands.runtime import CommandError, CommandResult
from mojilex_cli.config import MojiLexConfig, load_config, load_credentials
from mojilex_cli.dataset import (
    DatasetSnapshot,
    apply_snapshot,
    load_dataset,
    validate_dataset,
    validate_snapshot,
)
from mojilex_cli.dedupe import (
    DedupeIndex,
    best_cyclic_alignment,
    explain_pair,
    scan_snapshot,
)
from mojilex_cli.dedupe.preview import build_side_by_side_preview
from mojilex_cli.domain import (
    Media,
    RelationEvidence,
    RelationMediaPair,
    RelationReview,
    VisualRelation,
    reviewed_relation_sha256,
    visual_relation_id,
)
from mojilex_cli.domain.operations import utc_now
from mojilex_cli.media import MediaLimits, MediaProcessor, ProcessedMedia, TemporaryMediaRun
from mojilex_cli.pipeline.runner import repository_workspace
from mojilex_cli.sources import SourceCollection, SourceEmoji, TelegramBotAPI

ReviewChooser = Callable[[Path, Mapping[str, Any]], str]
_DECISIONS = frozenset({"same-artwork", "variant-of", "related-series", "not-duplicate", "skip"})
_PERSISTED_RELATION_SIGNALS = frozenset(
    {
        "alpha-iou-match",
        "canonical-render-match",
        "cyclic-phash-match",
        "decoded-exact",
        "high-temporal-similarity",
        "phash-match",
        "possible-recolor",
        "possible-variant",
        "shape-match",
        "speed-change",
        "ssim-match",
    }
)
_SIGNAL_ALIASES: dict[str, str | None] = {
    "binary-exact": None,
    "human-visual-review": None,
    "literal-text-mismatch": "possible-variant",
    "static-phash-match": "phash-match",
}
MediaKey = tuple[str, str]


def dedupe_scan_command(
    selector: str | None,
    *,
    all_items: bool,
    repo: str | None,
    max_candidates: int | None,
    profile: str | None,
) -> CommandResult:
    if (selector is None) == (not all_items):
        raise CommandError(
            "CONFIG_INVALID",
            "Pass exactly one selector or --all.",
            hint="Use `mojilex dedupe scan EMOJI_OR_COLLECTION` or `--all`.",
        )
    config = load_config(
        cli={
            "repository": {"target": repo},
            "dedupe": {"profile": profile, "max_candidates": max_candidates},
        }
    )
    _require_profile(config)
    with repository_workspace(config.repository.target, config.repository.base_branch) as workspace:
        validate_dataset(workspace.root, strict=False).raise_for_errors()
        snapshot = load_dataset(workspace.root)
        selected = None if all_items else _resolve_selector(snapshot, cast(str, selector))
        index_path = cast(Path, config.cache_dir) / "dedupe-index-v1.sqlite3"
        with DedupeIndex(index_path, repository_root=workspace.root) as index:
            report = index.update(
                snapshot,
                rebuild=all_items,
                selected_emoji_ids=selected,
                max_candidates=config.dedupe.max_candidates,
            )
        return CommandResult(
            result={
                **report.as_dict(),
                "index_path": str(index_path),
                "selected_emoji_ids": sorted(selected or snapshot.emojis),
                "canonical_writes": 0,
            }
        )


def dedupe_explain_command(
    emoji_id: str,
    against_emoji_id: str,
    *,
    repo: str | None,
    max_candidates: int | None,
) -> CommandResult:
    config = load_config(
        cli={
            "repository": {"target": repo},
            "dedupe": {"max_candidates": max_candidates},
        }
    )
    _require_profile(config)
    with repository_workspace(config.repository.target, config.repository.base_branch) as workspace:
        validate_dataset(workspace.root, strict=False).raise_for_errors()
        snapshot = load_dataset(workspace.root)
        left = _resolve_one_emoji(snapshot, emoji_id)
        right = _resolve_one_emoji(snapshot, against_emoji_id)
        explanation = explain_pair(snapshot, left, right)
        index_path = cast(Path, config.cache_dir) / "dedupe-index-v1.sqlite3"
        with DedupeIndex(index_path, repository_root=workspace.root) as index:
            report = index.update(
                snapshot,
                selected_emoji_ids={left, right},
                max_candidates=config.dedupe.max_candidates,
            )
        explanation["candidate_overflow"] = {
            left: report.candidate_overflow.get(left, False),
            right: report.candidate_overflow.get(right, False),
        }
        explanation["candidate_count_before_limit"] = {
            left: report.candidate_count_before_limit.get(left, 0),
            right: report.candidate_count_before_limit.get(right, 0),
        }
        return CommandResult(result=explanation)


def dedupe_review_command(
    emoji_id: str,
    against_emoji_id: str | None,
    *,
    repo: Path,
    reviewer: str,
    decision: str | None,
    chooser: ReviewChooser | None = None,
) -> CommandResult:
    root = repo.expanduser().resolve(strict=True)
    if not (root / ".git").exists():
        raise CommandError(
            "CONFIG_INVALID",
            "Dedupe review requires a persistent local Git checkout.",
            hint="Pass the local mojilex data repository with --repo.",
        )
    return asyncio.run(
        _dedupe_review(
            emoji_id,
            against_emoji_id,
            repo=root,
            reviewer=reviewer,
            decision=decision,
            chooser=chooser,
        )
    )


async def _dedupe_review(
    emoji_selector: str,
    against_selector: str | None,
    *,
    repo: Path,
    reviewer: str,
    decision: str | None,
    chooser: ReviewChooser | None,
) -> CommandResult:
    root = repo
    config = load_config(cli={"repository": {"target": str(root)}})
    _require_profile(config)
    credentials = load_credentials()
    if not credentials.telegram_bot_token:
        raise CommandError(
            "CREDENTIAL_MISSING",
            "TELEGRAM_BOT_TOKEN is required to verify both review media files.",
            hint="Set it only in the current process environment and retry.",
        )
    validate_dataset(root, strict=False).raise_for_errors()
    before = load_dataset(root)
    left_id = _resolve_one_emoji(before, emoji_selector)
    if against_selector is None:
        scan_report = scan_snapshot(
            before,
            selected_emoji_ids={left_id},
            max_candidates=config.dedupe.max_candidates,
        )
        candidates = scan_report.candidates.get(left_id, ())
        if not candidates:
            raise CommandError(
                "SOURCE_NOT_FOUND",
                "No near-duplicate candidate is available for this emoji.",
                hint="Pass --against explicitly or run `mojilex dedupe scan --all`.",
                entity_id=left_id,
            )
        right_id = candidates[0].against_emoji_id
    else:
        right_id = _resolve_one_emoji(before, against_selector)
    if left_id == right_id:
        raise CommandError(
            "CONFIG_INVALID",
            "A visual relation requires two distinct emoji IDs.",
            hint="Choose a different --against endpoint.",
        )
    explanation = explain_pair(before, left_id, right_id)
    limits = _media_limits(config)
    async with TelegramBotAPI(
        credentials.telegram_bot_token,
        timeout_seconds=config.telegram.timeout_seconds,
        max_attempts=config.telegram.max_attempts,
        max_download_bytes=config.processing.max_download_bytes,
    ) as adapter:
        await adapter.validate_credentials()
        source_items = await _resolve_review_sources(before, (left_id, right_id), adapter)
        with TemporaryMediaRun(limits=limits) as temporary:
            processor = MediaProcessor(temporary)

            async def process(entity_id: str) -> tuple[str, MediaKey, ProcessedMedia]:
                item = source_items[entity_id]
                media_key, expected_media = _sole_downloadable_media(before, entity_id)
                value = await processor.process_stream(
                    adapter.fetch_media(item),
                    expected_format=item.media_format,
                    declared_size=item.declared_file_size,
                    expected_sha256=expected_media.sha256,
                    needs_repainting=item.needs_repainting,
                )
                if value.analysis is None:
                    raise CommandError(
                        "MEDIA_RENDER_FAILED",
                        "Deterministic review analysis is unavailable.",
                        hint="Verify media prerequisites with `mojilex doctor`.",
                        entity_id=entity_id,
                    )
                return entity_id, media_key, value

            outcomes = await asyncio.gather(
                process(left_id), process(right_id), return_exceptions=True
            )
            for outcome in outcomes:
                if isinstance(outcome, BaseException):
                    raise outcome
            processed = {
                entity_id: {media_key: value}
                for entity_id, media_key, value in cast(
                    list[tuple[str, MediaKey, ProcessedMedia]], outcomes
                )
            }
            _verify_current_media(before, processed)
            shift, sample_count = best_cyclic_alignment(before, left_id, right_id)
            left_preview = _sole_processed_media(processed[left_id])
            right_preview = _sole_processed_media(processed[right_id])
            preview_shift = round(shift * len(right_preview.frame_paths) / sample_count)
            preview = build_side_by_side_preview(
                left_preview,
                right_preview,
                temporary.output_dir() / "dedupe-review.png",
                left_label=left_id,
                right_label=right_id,
                right_shift=preview_shift,
            )
            temporary.account_outputs((preview,))
            selected = decision
            if selected is None:
                if chooser is None:
                    raise CommandError(
                        "CONFIG_MISSING",
                        "A review decision is required in non-interactive mode.",
                        hint="Pass --decision with a supported relation type or skip.",
                    )
                selected = chooser(preview, explanation)
            selected = selected.strip().lower()
            if selected not in _DECISIONS:
                raise CommandError(
                    "CONFIG_INVALID",
                    f"Unsupported dedupe review decision: {selected}",
                    hint="Choose same-artwork, variant-of, related-series, not-duplicate, or skip.",
                )
            if selected == "skip":
                return CommandResult(status="noop", result={"decision": "skip"})  # type: ignore[arg-type]
            _guard_literal_text_conflict(before, left_id, right_id, selected)
            after = before.clone()
            relation = _approved_relation(
                after,
                left_id,
                right_id,
                relation_type=selected,
                reviewer=reviewer,
                explanation=explanation,
            )
            after.relations[relation.id] = relation
            changed = _apply_review_relation(before, after)
            return CommandResult(
                result={
                    "relation_id": relation.id,
                    "decision": selected,
                    "changed_paths": [path.as_posix() for path in changed],
                    "preview_deleted": True,
                }
            )


def _approved_relation(
    snapshot: DatasetSnapshot,
    left_id: str,
    right_id: str,
    *,
    relation_type: str,
    reviewer: str,
    explanation: Mapping[str, Any],
) -> VisualRelation:
    symmetric = relation_type in {"same-artwork", "related-series", "not-duplicate"}
    subject_id, object_id = tuple(sorted((left_id, right_id))) if symmetric else (left_id, right_id)
    media_pairs, signals = _relation_evidence(
        snapshot,
        left_id,
        right_id,
        subject_id=subject_id,
        explanation=explanation,
    )
    namespace = UUID(str(snapshot.manifest["visual_relation_namespace"]))
    relation = VisualRelation(
        schema_version="1.0.0",
        entity_type="visual_relation",
        id=visual_relation_id(subject_id, object_id, "entity", namespace=namespace),
        identity_epoch=0,
        subject_id=subject_id,
        object_id=object_id,
        scope="entity",
        relation_type=relation_type,
        evidence=RelationEvidence(
            dedupe_profile=snapshot.emojis[subject_id].fingerprints.profile,
            subject_media_digest=snapshot.emojis[subject_id].fingerprints.input_media_digest,
            object_media_digest=snapshot.emojis[object_id].fingerprints.input_media_digest,
            media_pairs=media_pairs,
            signals=signals,
        ),
        review=RelationReview(status="unreviewed"),
    )
    payload = relation.as_dict()
    payload["review"] = RelationReview(
        status="approved",
        reviewer=reviewer,
        reviewed_at=utc_now(),
        reviewed_relation_sha256=reviewed_relation_sha256(relation),
    ).as_dict()
    return VisualRelation.model_validate(payload)


def _relation_evidence(
    snapshot: DatasetSnapshot,
    left_id: str,
    right_id: str,
    *,
    subject_id: str,
    explanation: Mapping[str, Any],
) -> tuple[list[RelationMediaPair], list[str]]:
    left = snapshot.emojis[left_id]
    right = snapshot.emojis[right_id]
    if left.fingerprints.profile != right.fingerprints.profile:
        raise CommandError(
            "VALIDATION_FAILED",
            "A reviewed relation requires matching dedupe profiles on both endpoints.",
            hint="Migrate both fingerprint sets to the current profile before review.",
        )
    left_keys = tuple(sorted((item.role.value, item.variant_id or "") for item in left.media))
    right_keys = tuple(sorted((item.role.value, item.variant_id or "") for item in right.media))
    if len(left_keys) != len(right_keys):
        raise CommandError(
            "VALIDATION_FAILED",
            "An entity relation requires a full one-to-one mapping of both media sets.",
            hint="Review a single media-pair instead or make both entity media sets complete.",
        )
    raw_comparisons = explanation.get("comparisons")
    if not isinstance(raw_comparisons, list):
        raise CommandError(
            "VALIDATION_FAILED",
            "Dedupe evidence does not contain media-level comparisons.",
            hint="Run a fresh dedupe explain/review after recomputing fingerprints.",
        )
    comparisons: dict[tuple[MediaKey, MediaKey], Mapping[str, Any]] = {}
    for raw in raw_comparisons:
        if not isinstance(raw, Mapping):
            continue
        left_key = _comparison_key(raw.get("subject"))
        right_key = _comparison_key(raw.get("object"))
        if left_key is not None and right_key is not None:
            comparisons[(left_key, right_key)] = raw
    mapping = _best_media_bijection(left_keys, right_keys, comparisons)

    signals: set[str] = set()
    profiles_compatible = explanation.get("profile_compatible") is True
    for left_key, right_key in mapping:
        comparison = comparisons[(left_key, right_key)]
        if comparison.get("decoded_exact") is True and profiles_compatible:
            signals.add("decoded-exact")
        candidate = comparison.get("candidate")
        if isinstance(candidate, Mapping):
            raw_signals = candidate.get("signals", ())
            if isinstance(raw_signals, (list, tuple)):
                for raw_signal in raw_signals:
                    if not isinstance(raw_signal, str):
                        continue
                    normalized = _normalize_relation_signal(raw_signal)
                    if normalized is not None:
                        signals.add(normalized)
    if not signals:
        raise CommandError(
            "VALIDATION_FAILED",
            "No controlled algorithmic signal supports this reviewed relation.",
            hint="Refresh fingerprints and choose a pair produced by dedupe scan.",
        )

    if subject_id == left_id:
        oriented = mapping
    else:
        oriented = tuple((right_key, left_key) for left_key, right_key in mapping)
    pairs = [
        RelationMediaPair(
            subject_role=subject_key[0],
            subject_variant_id=subject_key[1] or None,
            object_role=object_key[0],
            object_variant_id=object_key[1] or None,
        )
        for subject_key, object_key in oriented
    ]
    return sorted(pairs, key=lambda item: item.sort_key), sorted(signals)


def _comparison_key(value: Any) -> MediaKey | None:
    if not isinstance(value, Mapping) or not isinstance(value.get("role"), str):
        return None
    variant = value.get("variant_id")
    if variant is not None and not isinstance(variant, str):
        return None
    return str(value["role"]), variant or ""


def _comparison_cost(value: Mapping[str, Any]) -> int:
    if value.get("binary_exact") is True:
        return 0
    if value.get("decoded_exact") is True:
        return 1
    candidate = value.get("candidate")
    if isinstance(candidate, Mapping):
        return 2
    return 3


def _best_media_bijection(
    left_keys: tuple[MediaKey, ...],
    right_keys: tuple[MediaKey, ...],
    comparisons: Mapping[tuple[MediaKey, MediaKey], Mapping[str, Any]],
) -> tuple[tuple[MediaKey, MediaKey], ...]:
    if not left_keys:
        raise CommandError(
            "VALIDATION_FAILED",
            "A relation cannot map an empty media set.",
            hint="Refresh the entity media before review.",
        )
    if set(left_keys) == set(right_keys) and all((key, key) in comparisons for key in left_keys):
        return tuple((key, key) for key in left_keys)
    if len(left_keys) > 8:
        raise CommandError(
            "VALIDATION_FAILED",
            "A non-identical media-key mapping is too large to verify safely.",
            hint="Use matching role/variant keys or review narrower media-pair relations.",
        )
    best_score: tuple[int, int] | None = None
    best: tuple[tuple[MediaKey, MediaKey], ...] | None = None
    best_count = 0
    for permutation in itertools.permutations(right_keys):
        mapping = tuple(zip(left_keys, permutation, strict=True))
        if any(pair not in comparisons for pair in mapping):
            continue
        score = (
            sum(0 if left == right else 1 if left[0] == right[0] else 2 for left, right in mapping),
            sum(_comparison_cost(comparisons[pair]) for pair in mapping),
        )
        if best_score is None or score < best_score:
            best_score = score
            best = mapping
            best_count = 1
        elif score == best_score:
            best_count += 1
    if best is None or best_count != 1:
        raise CommandError(
            "VALIDATION_FAILED",
            "The full media mapping is missing or ambiguous.",
            hint="Use matching role/variant keys or review explicit media pairs.",
        )
    return best


def _normalize_relation_signal(value: str) -> str | None:
    normalized = _SIGNAL_ALIASES.get(value, value)
    if normalized is None:
        return None
    if normalized not in _PERSISTED_RELATION_SIGNALS:
        raise CommandError(
            "VALIDATION_FAILED",
            f"Unsupported dedupe evidence signal: {value}",
            hint="Rebuild candidates with the current dedupe profile.",
        )
    return normalized


def _media_key(value: Media) -> MediaKey:
    return value.role.value, value.variant_id or ""


def _sole_downloadable_media(snapshot: DatasetSnapshot, emoji_id: str) -> tuple[MediaKey, Media]:
    emoji = snapshot.emojis[emoji_id]
    media = {_media_key(item): item for item in emoji.media}
    fingerprints = {item.key: item for item in emoji.fingerprints.items}
    rendering = {item.key: item for item in emoji.facets.rendering.items}
    if set(media) != set(fingerprints) or set(media) != set(rendering):
        raise CommandError(
            "VALIDATION_FAILED",
            "Review requires complete media, rendering, and fingerprint bindings.",
            hint="Refresh deterministic analysis before reviewing duplicates.",
            entity_id=emoji_id,
        )
    if len(media) != 1 or next(iter(media)) != ("primary", ""):
        # Telegram exposes one source file per custom emoji.  Silently checking
        # media[0] would let an entity-level decision cover unseen variants.
        raise CommandError(
            "VALIDATION_FAILED",
            "This source adapter cannot re-download and verify every entity media variant.",
            hint="Use a future multi-media adapter or review explicit media pairs.",
            entity_id=emoji_id,
        )
    key = next(iter(media))
    return key, media[key]


def _sole_processed_media(values: Mapping[MediaKey, ProcessedMedia]) -> ProcessedMedia:
    if len(values) != 1:
        raise CommandError(
            "VALIDATION_FAILED",
            "A single-image preview cannot represent the complete media mapping.",
            hint="Verify and preview every mapped media item before approval.",
        )
    return next(iter(values.values()))


def _apply_review_relation(
    before: DatasetSnapshot, after: DatasetSnapshot
) -> tuple[PurePosixPath, ...]:
    schemas_available = (before.root / "schemas" / "v1").is_dir()
    staged = validate_snapshot(
        after,
        canonical=False,
        schemas=schemas_available,
        repository_files=True,
    )
    staged.raise_for_errors()
    changed = apply_snapshot(before, after)
    persisted = validate_dataset(before.root, strict=True)
    if persisted.valid:
        return changed
    # Canonical byte/path checks can only be meaningful after the staged files
    # exist.  Restore the exact prior snapshot before surfacing the failure.
    apply_snapshot(
        after,
        before,
        validator=lambda value: validate_snapshot(
            value,
            canonical=False,
            schemas=schemas_available,
            repository_files=False,
        ),
    )
    persisted.raise_for_errors()
    raise AssertionError("unreachable")


async def _resolve_review_sources(
    snapshot: DatasetSnapshot,
    emoji_ids: tuple[str, str],
    adapter: TelegramBotAPI,
) -> dict[str, SourceEmoji]:
    collections: dict[str, SourceCollection] = {}
    result: dict[str, SourceEmoji] = {}
    for emoji_id in emoji_ids:
        urls = sorted(
            {
                collection.canonical_url
                for membership in snapshot.memberships.values()
                if membership.emoji_id == emoji_id and membership.status.value == "active"
                for collection in (snapshot.collections.get(membership.collection_id),)
                if collection is not None and collection.canonical_url
            }
        )
        for url in urls:
            source = collections.get(url)
            if source is None:
                source = await adapter.fetch_collection(adapter.canonicalize(url))
                collections[url] = source
            match = next(
                (
                    item
                    for item in source.items
                    if item.native_id == snapshot.emojis[emoji_id].native_id
                ),
                None,
            )
            if match is not None:
                result[emoji_id] = match
                break
        if emoji_id not in result:
            raise CommandError(
                "SOURCE_NOT_FOUND",
                "The current emoji media could not be resolved from an active collection.",
                hint="Refresh availability/source metadata before reviewing this pair.",
                entity_id=emoji_id,
            )
    return result


def _verify_current_media(
    snapshot: DatasetSnapshot,
    processed: Mapping[str, Mapping[MediaKey, ProcessedMedia]],
) -> None:
    for emoji_id, values in processed.items():
        current = snapshot.emojis[emoji_id]
        media = {_media_key(item): item for item in current.media}
        fingerprints = {item.key: item for item in current.fingerprints.items}
        rendering = {item.key: item for item in current.facets.rendering.items}
        if set(values) != set(media) or set(values) != set(fingerprints):
            raise CommandError(
                "SOURCE_CHANGED_DURING_RUN",
                "Review did not verify the complete current media set.",
                hint="Run update/add to refresh the emoji before reviewing duplicates.",
                entity_id=emoji_id,
            )
        for key, value in values.items():
            expected_media = media[key]
            metadata = value.metadata
            if (
                metadata.sha256 != expected_media.sha256
                or metadata.byte_size != expected_media.byte_size
                or metadata.width != expected_media.width
                or metadata.height != expected_media.height
                or metadata.animated != expected_media.animated
                or metadata.duration_ms != expected_media.duration_ms
                or metadata.kind != expected_media.kind.value
                or metadata.format != expected_media.format.value
                or metadata.mime_type != expected_media.mime_type
            ):
                raise CommandError(
                    "SOURCE_CHANGED_DURING_RUN",
                    "Review media changed after the canonical fingerprint was created.",
                    hint="Run update/add to refresh the emoji before reviewing duplicates.",
                    entity_id=emoji_id,
                )
            analysis = value.analysis
            expected_fingerprint = fingerprints[key].model_dump(
                mode="json", exclude={"role", "variant_id"}
            )
            expected_rendering = rendering[key].model_dump(
                mode="json", exclude={"role", "variant_id"}, exclude_none=True
            )
            if (
                analysis is None
                or analysis.dedupe_profile != current.fingerprints.profile
                or analysis.color_profile != current.facets.rendering.profile
                or analysis.fingerprint.model_dump(mode="json") != expected_fingerprint
                or analysis.rendering.model_dump(mode="json", exclude_none=True)
                != expected_rendering
            ):
                raise CommandError(
                    "SOURCE_CHANGED_DURING_RUN",
                    "Review analysis no longer matches the canonical fingerprint.",
                    hint="Refresh fingerprints before recording a human relation.",
                    entity_id=emoji_id,
                )


def _guard_literal_text_conflict(
    snapshot: DatasetSnapshot, left_id: str, right_id: str, decision: str
) -> None:
    if decision != "same-artwork":
        return
    left = snapshot.emojis[left_id]
    right = snapshot.emojis[right_id]
    left_text = {item.value for item in left.facets.text_content.items}
    right_text = {item.value for item in right.facets.text_content.items}
    if (
        left.review.status.value == "approved"
        and right.review.status.value == "approved"
        and left_text != right_text
    ):
        raise CommandError(
            "VALIDATION_FAILED",
            "Approved literal text differs, so same-artwork is not allowed.",
            hint="Use variant-of or not-duplicate after reviewing the visual difference.",
        )


def _resolve_selector(snapshot: DatasetSnapshot, selector: str) -> set[str]:
    if selector in snapshot.emojis:
        return {selector}
    if selector in snapshot.collections:
        return {
            membership.emoji_id
            for membership in snapshot.memberships.values()
            if membership.collection_id == selector and membership.status.value == "active"
        }
    native = {emoji.id for emoji in snapshot.emojis.values() if emoji.native_id == selector}
    if len(native) == 1:
        return native
    raise CommandError(
        "SOURCE_NOT_FOUND",
        f"Selector did not resolve uniquely: {selector}",
        hint="Pass a canonical emoji ID, collection ID, or unique native emoji ID.",
    )


def _resolve_one_emoji(snapshot: DatasetSnapshot, selector: str) -> str:
    result = _resolve_selector(snapshot, selector)
    if len(result) != 1:
        raise CommandError(
            "CONFIG_INVALID",
            "This operation requires one emoji rather than a collection.",
            hint="Pass an mxe_ ID or one unique native emoji ID.",
        )
    return next(iter(result))


def _require_profile(config: MojiLexConfig) -> None:
    if config.dedupe.profile != "dedupe-v1":
        raise CommandError(
            "CONFIG_INVALID",
            f"Unsupported dedupe profile: {config.dedupe.profile}",
            hint="Use the immutable dedupe-v1 profile included with this CLI release.",
        )


def _media_limits(config: MojiLexConfig) -> MediaLimits:
    return MediaLimits(
        max_file_bytes=config.processing.max_download_bytes,
        frames=config.processing.keyframes,
        worker_timeout_seconds=config.processing.render_timeout_seconds,
        max_run_temp_bytes=config.processing.max_temp_bytes,
    )
