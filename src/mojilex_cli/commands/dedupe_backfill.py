"""Verified, AI-free migration of missing fingerprints, without partial publication."""

from __future__ import annotations

from collections.abc import Iterable

from mojilex_cli.commands.runtime import CommandError
from mojilex_cli.config import MojiLexConfig, load_credentials
from mojilex_cli.dataset import DatasetSnapshot
from mojilex_cli.domain import Emoji, media_digest
from mojilex_cli.media import MediaLimits, MediaProcessor, TemporaryMediaRun
from mojilex_cli.sources import SourceNotFoundError, TelegramBotAPI


async def backfill_fingerprints(
    before: DatasetSnapshot, selected: Iterable[str], config: MojiLexConfig
) -> tuple[DatasetSnapshot, tuple[str, ...]]:
    targets = sorted(
        identifier
        for identifier in set(selected)
        if before.emojis[identifier].fingerprints.status.value != "complete"
    )
    if not targets:
        return before, ()
    for identifier in targets:
        emoji = before.emojis[identifier]
        if emoji.fingerprints.profile != config.dedupe.profile:
            raise CommandError(
                "VALIDATION_FAILED",
                "Backfill cannot silently migrate fingerprint profiles.",
                hint="Use an explicit full profile migration first.",
                entity_id=identifier,
            )
        if (
            emoji.platform != "telegram"
            or emoji.native_namespace != "custom_emoji.id"
            or emoji.scope_id != "global"
            or len(emoji.media) != 1
            or (emoji.media[0].role.value != "primary" or emoji.media[0].variant_id is not None)
        ):
            raise CommandError(
                "SOURCE_UNSUPPORTED",
                "This adapter cannot verify every legacy media variant.",
                hint="Use a source adapter that can retrieve all role/variant media.",
                entity_id=identifier,
            )
    credentials = load_credentials()
    if not credentials.telegram_bot_token:
        raise CommandError(
            "CREDENTIAL_MISSING",
            "Fingerprint backfill needs TELEGRAM_BOT_TOKEN, but no AI key.",
            hint="Set the bot token in the process environment and retry dedupe scan.",
        )
    after = before.clone()
    limits = MediaLimits(
        max_file_bytes=config.processing.max_download_bytes,
        frames=config.processing.keyframes,
        worker_timeout_seconds=config.processing.render_timeout_seconds,
        max_run_temp_bytes=config.processing.max_temp_bytes,
    )
    async with TelegramBotAPI(
        credentials.telegram_bot_token,
        timeout_seconds=config.telegram.timeout_seconds,
        max_attempts=config.telegram.max_attempts,
        max_download_bytes=config.processing.max_download_bytes,
    ) as adapter:
        await adapter.validate_credentials()
        for identifier in targets:
            emoji = before.emojis[identifier]
            expected = emoji.media[0]
            try:
                items = await adapter.fetch_emojis((emoji.native_id,))
                item = items.get(emoji.native_id)
                if item is None:
                    raise SourceNotFoundError("Legacy source emoji is not available")
                with TemporaryMediaRun(limits=limits) as temporary:
                    processed = await MediaProcessor(temporary).process_stream(
                        adapter.fetch_media(item),
                        expected_format=item.media_format,
                        declared_size=item.declared_file_size,
                        expected_sha256=expected.sha256,
                        needs_repainting=item.needs_repainting,
                    )
                    if processed.dataset_metadata() != expected.as_dict():
                        raise CommandError(
                            "SOURCE_CHANGED_DURING_RUN",
                            "Legacy source media metadata changed.",
                            hint="Run update and review the changed media before backfill.",
                            entity_id=identifier,
                        )
                    analysis = processed.analysis
                    if analysis is None:
                        raise CommandError(
                            "MEDIA_RENDER_FAILED",
                            "Backfill did not obtain complete analysis.",
                            hint="Run mojilex doctor and install the required decoder.",
                            entity_id=identifier,
                        )
                    if (
                        analysis.dedupe_profile != before.manifest["dedupe_profile"]
                        or analysis.dedupe_profile_sha256
                        != before.manifest["dedupe_profile_sha256"]
                        or analysis.color_profile != before.manifest["color_profile"]
                        or analysis.color_profile_sha256 != before.manifest["color_profile_sha256"]
                    ):
                        raise CommandError(
                            "VALIDATION_FAILED",
                            "Backfill analysis differs from pinned profiles.",
                            hint="Use the exact dataset analysis profiles; do not mix versions.",
                            entity_id=identifier,
                        )
                    # Fingerprint migration must preserve reviewed semantic/rendering data.
                    rendering = {
                        "role": "primary",
                        **analysis.rendering.model_dump(mode="json", exclude_none=True),
                    }
                    if rendering != emoji.facets.rendering.items[0].as_dict():
                        raise CommandError(
                            "VALIDATION_FAILED",
                            "Legacy rendering metadata needs explicit review.",
                            hint="Refresh deterministic rendering with add/update before backfill.",
                            entity_id=identifier,
                        )
                    payload = emoji.as_dict()
                    payload["fingerprints"] = {
                        "status": "complete",
                        "profile": analysis.dedupe_profile,
                        "input_media_digest": media_digest(emoji.media),
                        "items": [
                            {"role": "primary", **analysis.fingerprint.model_dump(mode="json")}
                        ],
                    }
                    after.emojis[identifier] = Emoji.model_validate(payload)
            except SourceNotFoundError:
                if emoji.availability.status.value == "active":
                    raise CommandError(
                        "SOURCE_NOT_FOUND",
                        "Active legacy emoji media is unavailable.",
                        hint=(
                            "First verify and record non-active availability using update; "
                            "no fingerprints were written."
                        ),
                        entity_id=identifier,
                    ) from None
                if emoji.fingerprints.items:
                    raise CommandError(
                        "SOURCE_NOT_FOUND",
                        "Cannot complete partially fingerprinted legacy media.",
                        hint="Keep existing fingerprints and retry when source media is available.",
                        entity_id=identifier,
                    ) from None
                payload = emoji.as_dict()
                payload["fingerprints"]["status"] = "unavailable"
                after.emojis[identifier] = Emoji.model_validate(payload)
    return after, tuple(targets)
