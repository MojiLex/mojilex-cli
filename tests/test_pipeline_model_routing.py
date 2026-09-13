from __future__ import annotations

import hashlib
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Literal

import pytest
from PIL import Image

from mojilex_cli.ai import (
    AIOutputError,
    CostEstimate,
    DescriptionBatch,
    DescriptionItem,
    DescriptionRequest,
    DescriptionResult,
    RequestBudget,
)
from mojilex_cli.ai.prompts import (
    gemini_request_parameters_sha256,
    prompt_sha256,
)
from mojilex_cli.analysis import DeterministicMediaAnalysis
from mojilex_cli.cache import CacheStore
from mojilex_cli.config import AIConfig, MojiLexConfig
from mojilex_cli.media import MediaMetadata, ProcessedMedia, TemporaryMediaRun
from mojilex_cli.pipeline import runner as pipeline_runner
from mojilex_cli.pipeline.runner import (
    _ai_request_plan_sha256,
    _AICacheTrace,
    _AIRequestIdentity,
    _AIState,
    _cache_aliases,
    _cache_key,
    _describe_batch,
    _load_or_describe_single,
    _vision_context,
)
from mojilex_cli.policy import (
    ModelQualification,
    ModelQualificationRegistry,
    RoutingReasonRegistry,
)
from mojilex_cli.sources import SourceEmoji
from test_dataset_helpers import write_fixture


def _description(text: str, *, sensitive: bool) -> DescriptionItem:
    localized = {
        "text": text,
        "motion_status": "not_applicable",
        "usage": ["reaction"],
    }
    return DescriptionItem.model_validate(
        {
            "label": "E001",
            "descriptions": {"ru": localized, "en": localized},
            "facets": {
                "text_content": {"status": "none", "dynamics": "stable", "items": []},
                "content_types": ["reaction"],
                "styles": ["flat"],
                "suggested_uses": ["message-accent"],
                "uncertainties": [],
            },
            "semantic_tags": ["synthetic"],
            "content": {
                "rating": "sensitive" if sensitive else "general",
                "warnings": ["flashing"] if sensitive else [],
            },
        }
    )


def _source() -> SourceEmoji:
    return SourceEmoji(
        native_namespace="custom_emoji.id",
        scope_id="global",
        native_id="5368324170671202286",
        file_unique_id="AgADExampleUniqueId",
        position=0,
        width=100,
        height=100,
        animated=False,
        video=False,
        media_format="webp",
        file_id="transient",
    )


def _processed() -> ProcessedMedia:
    return ProcessedMedia(
        metadata=MediaMetadata(
            kind="static",
            format="webp",
            mime_type="image/webp",
            sha256="8" * 64,
            byte_size=128,
            width=100,
            height=100,
            animated=False,
        ),
        analysis=DeterministicMediaAnalysis.model_validate(
            {
                "color_profile_sha256": "2" * 64,
                "dedupe_profile_sha256": "3" * 64,
                "decoder_backend_fingerprint": "4" * 64,
                "rendering": {
                    "color_behavior": "fixed",
                    "palette_dynamics": "stable",
                    "alpha_mode": "translucent",
                    "visible_area_bp": 5000,
                    "dominant_colors": [{"hex": "#ff0000", "family": "red", "coverage_bp": 10000}],
                },
                "fingerprint": {
                    "decoded_payload_sha256": "5" * 64,
                    "canonical_render_sha256": "6" * 64,
                    "shape_sha256": "7" * 64,
                    "perceptual": {
                        "sample_count": 1,
                        "layout_phash64": "AAAAAAAAAAA",
                        "content_phash64": "AAAAAAAAAAA",
                        "alpha_phash64": "AAAAAAAAAAA",
                        "edge_phash64": "AAAAAAAAAAA",
                        "temporal_energy_bp": 0,
                        "low_information": False,
                    },
                },
            }
        ),
        frame_paths=(Path("unused-frame.png"),),
    )


def _animated_processed() -> ProcessedMedia:
    value = _processed()
    assert value.analysis is not None
    perceptual = value.analysis.fingerprint.perceptual.model_copy(
        update={
            "sample_count": 16,
            "layout_phash64": "A" * 171,
            "content_phash64": "A" * 171,
            "alpha_phash64": "A" * 171,
            "edge_phash64": "A" * 171,
            "temporal_energy_bp": 100,
        }
    )
    fingerprint = value.analysis.fingerprint.model_copy(update={"perceptual": perceptual})
    analysis = value.analysis.model_copy(update={"fingerprint": fingerprint})
    return ProcessedMedia(
        metadata=MediaMetadata(
            kind="animation",
            format="tgs",
            mime_type="application/x-tgsticker",
            sha256=value.metadata.sha256,
            byte_size=value.metadata.byte_size,
            width=value.metadata.width,
            height=value.metadata.height,
            animated=True,
            duration_ms=1000,
        ),
        analysis=analysis,
        frame_paths=value.frame_paths,
    )


def _qualifications(root: Path) -> ModelQualificationRegistry:
    template = ModelQualificationRegistry.load(root).entries[0].model_dump(mode="json")
    entries = []
    for identifier, model in (
        ("mq_standard-v1_primary", "primary-model"),
        ("mq_standard-v1_strong", "strong-model"),
    ):
        raw = {
            **template,
            "qualification_id": identifier,
            "model": model,
            "model_revision": None,
            "prompt_sha256": prompt_sha256(),
            "request_parameters_sha256": gemini_request_parameters_sha256(),
            "valid_until": "2027-01-01T00:00:00Z",
        }
        entries.append(ModelQualification.model_validate(raw))
    return ModelQualificationRegistry(
        schema_version="1.0.0",
        registry_id="model-qualifications-v1",
        entries=tuple(entries),
    )


def _put_result(
    cache: CacheStore,
    dataset_root: Path,
    config: MojiLexConfig,
    source: SourceEmoji,
    processed: ProcessedMedia,
    *,
    model: str,
    description: DescriptionItem,
    stage: Literal["primary", "escalated"] = "primary",
    identity: _AIRequestIdentity | None = None,
) -> _AICacheTrace:
    del dataset_root
    identity = identity or _request_identity(source, processed, model=model)
    label = identity.label_for(source.native_id)
    key = _cache_key(
        source,
        processed,
        _vision_context(source, processed),
        config,
        model=model,
        model_revision=None,
        taxonomy_version="1.0.0",
        request_identity=identity,
        item_label=label,
    )
    cache.put_ai(
        key,
        DescriptionResult(
            batch=DescriptionBatch(items=(description,)),
            provider="gemini",
            model=model,
        ),
    )
    return _AICacheTrace(
        stage=stage,
        model=model,
        model_revision=None,
        cache_key=key,
        request_identity=identity,
    )


def _request_identity(
    source: SourceEmoji,
    processed: ProcessedMedia,
    *,
    model: str,
    discriminator: str = "single",
) -> _AIRequestIdentity:
    plan_sha256 = _ai_request_plan_sha256(
        (source,),
        {source.native_id: processed},
        model=model,
    )
    return _AIRequestIdentity(
        plan_sha256=plan_sha256,
        request_sha256=hashlib.sha256(
            f"request\0{plan_sha256}\0{discriminator}".encode()
        ).hexdigest(),
        shown_media_sha256=(
            hashlib.sha256(f"shown\0{plan_sha256}\0{discriminator}".encode()).hexdigest(),
        ),
        labels_by_native=((source.native_id, "E001"),),
    )


class _BatchProvider:
    name = "gemini"

    def __init__(self) -> None:
        self.calls: list[DescriptionRequest] = []

    def estimate(self, request: DescriptionRequest) -> CostEstimate:
        del request
        return CostEstimate(upper_bound_usd=Decimal("0"), note="test")

    async def describe(self, request: DescriptionRequest) -> DescriptionResult:
        self.calls.append(request)
        return DescriptionResult(
            batch=DescriptionBatch(
                items=tuple(
                    _description(f"Provider result for {label}.", sensitive=False).model_copy(
                        update={"label": label}
                    )
                    for label in request.expected_labels
                )
            ),
            provider="gemini",
            model=request.model,
        )


def _source_variant(native_id: str, position: int) -> SourceEmoji:
    return _source().model_copy(
        update={
            "native_id": native_id,
            "file_unique_id": f"unique-{native_id}",
            "file_id": f"transient-{native_id}",
            "position": position,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("facet_conflict", [False, True])
async def test_rules_escalate_quality_conflict_but_not_content_labels(
    tmp_path: Path, facet_conflict: bool
) -> None:
    dataset_root = tmp_path / "dataset"
    write_fixture(dataset_root)
    config = MojiLexConfig(
        ai=AIConfig(
            provider="gemini",
            model="primary-model",
            model_routing="rules",
            escalation_model="strong-model",
        )
    )
    source = _source()
    processed = _processed()
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=dataset_root)
    primary = _description("Primary sensitive draft.", sensitive=True)
    if facet_conflict:
        primary_data = primary.model_dump(mode="json")
        primary_data["facets"]["uncertainties"] = ["style"]
        primary = DescriptionItem.model_validate(primary_data)
    # Keep the final result sensitive too: it must be accepted, never escalated twice.
    strong = _description("Strong complete final object.", sensitive=True)
    primary_trace = _put_result(
        cache,
        dataset_root,
        config,
        source,
        processed,
        model="primary-model",
        description=primary,
    )
    strong_trace = _put_result(
        cache,
        dataset_root,
        config,
        source,
        processed,
        model="strong-model",
        description=strong,
        stage="escalated",
    )

    result = await _describe_batch(
        (source,),
        {source.native_id: processed},
        config=config,
        cache=cache,
        budget=RequestBudget(max_requests=0),
        ai_state=_AIState(),
        api_key=None,
        temporary=TemporaryMediaRun(),
        taxonomy_version="1.0.0",
        qualifications=_qualifications(dataset_root),
        routing_registry=RoutingReasonRegistry.load(dataset_root),
        resume_request_traces={source.native_id: (primary_trace, strong_trace)},
        require_exact_resume=True,
    )

    outcome = result[source.native_id]
    assert outcome.description == (strong if facet_conflict else primary)
    assert outcome.generation.model == ("strong-model" if facet_conflict else "primary-model")
    assert outcome.generation.generation_stage == ("escalated" if facet_conflict else "primary")
    assert outcome.generation.qualification_id == (
        "mq_standard-v1_strong" if facet_conflict else "mq_standard-v1_primary"
    )
    assert outcome.generation.routing_reason_codes == (
        ("facet-conflict",) if facet_conflict else ()
    )
    cache.close()


@pytest.mark.asyncio
async def test_routing_off_keeps_primary_even_when_validated_signal_exists(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    write_fixture(dataset_root)
    config = MojiLexConfig(
        ai=AIConfig(
            provider="gemini",
            model="primary-model",
            model_routing="off",
            escalation_model="strong-model",
        )
    )
    source = _source()
    processed = _processed()
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=dataset_root)
    primary = _description("Primary sensitive draft.", sensitive=True)
    primary_trace = _put_result(
        cache,
        dataset_root,
        config,
        source,
        processed,
        model="primary-model",
        description=primary,
    )

    result = await _describe_batch(
        (source,),
        {source.native_id: processed},
        config=config,
        cache=cache,
        budget=RequestBudget(max_requests=0),
        ai_state=_AIState(),
        api_key=None,
        temporary=TemporaryMediaRun(),
        taxonomy_version="1.0.0",
        qualifications=_qualifications(dataset_root),
        routing_registry=RoutingReasonRegistry.load(dataset_root),
        resume_request_traces={source.native_id: (primary_trace,)},
        require_exact_resume=True,
    )

    outcome = result[source.native_id]
    assert outcome.description == primary
    assert outcome.generation.model == "primary-model"
    assert outcome.generation.generation_stage == "primary"
    assert outcome.generation.routing_reason_codes == ()
    cache.close()


@pytest.mark.asyncio
async def test_deterministic_pre_route_skips_primary_and_marks_escalated(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    write_fixture(dataset_root)
    config = MojiLexConfig(
        ai=AIConfig(
            provider="gemini",
            model="primary-model",
            model_routing="rules",
            escalation_model="strong-model",
        )
    )
    source = _source()
    processed = _animated_processed()
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=dataset_root)
    strong = _description("Direct pre-routed complete object.", sensitive=False)
    strong_trace = _put_result(
        cache,
        dataset_root,
        config,
        source,
        processed,
        model="strong-model",
        description=strong,
        stage="escalated",
    )

    result = await _describe_batch(
        (source,),
        {source.native_id: processed},
        config=config,
        cache=cache,
        budget=RequestBudget(max_requests=0),
        ai_state=_AIState(),
        api_key=None,
        temporary=TemporaryMediaRun(),
        taxonomy_version="1.0.0",
        qualifications=_qualifications(dataset_root),
        routing_registry=RoutingReasonRegistry.load(dataset_root),
        resume_request_traces={source.native_id: (strong_trace,)},
        require_exact_resume=True,
    )

    outcome = result[source.native_id]
    assert outcome.description == strong
    assert outcome.generation.model == "strong-model"
    assert outcome.generation.generation_stage == "escalated"
    assert outcome.generation.routing_reason_codes == ("complex-motion",)
    cache.close()


@pytest.mark.asyncio
async def test_actual_revision_cache_alias_and_checkpoint_prevent_paid_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    write_fixture(dataset_root)
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    source = _source()
    processed = _processed()
    context = _vision_context(source, processed)
    provider_initializations = 0
    descriptions = 0
    run_scope = "mlxrun_" + "a" * 32

    async def fake_provider(*args: object, **kwargs: object) -> object:
        nonlocal provider_initializations
        provider_initializations += 1
        return object()

    async def fake_describe(*args: object, **kwargs: object) -> DescriptionResult:
        nonlocal descriptions
        descriptions += 1
        revision = "revision-paid-once" if descriptions == 1 else "revision-next-run"
        return DescriptionResult(
            batch=DescriptionBatch(items=(_description("Paid result.", sensitive=False),)),
            provider="gemini",
            model="primary-model",
            model_revision=revision,
        )

    async def fake_prepare(
        items: tuple[SourceEmoji, ...],
        processed_values: dict[str, ProcessedMedia],
        *,
        model: str,
        temporary: TemporaryMediaRun,
    ) -> pipeline_runner._PreparedAIRequest:
        del temporary
        item = items[0]
        identity = _request_identity(item, processed_values[item.native_id], model=model)
        context_value = _vision_context(item, processed_values[item.native_id])
        request = pipeline_runner.DescriptionRequest(
            model=model,
            images=(
                pipeline_runner.VisionImage(
                    data=b"\x89PNG\r\n\x1a\nsynthetic",
                    labels=("E001",),
                ),
            ),
            expected_labels=("E001",),
            context={"E001": context_value},
        )
        return pipeline_runner._PreparedAIRequest(request=request, identity=identity)

    monkeypatch.setattr(pipeline_runner, "_provider_for_model", fake_provider)
    monkeypatch.setattr(pipeline_runner, "_describe_single", fake_describe)
    monkeypatch.setattr(pipeline_runner, "_prepare_ai_request", fake_prepare)
    monkeypatch.setattr(pipeline_runner, "_utc_text", lambda: "2026-01-02T03:04:05Z")

    cache_path = tmp_path / "cache.sqlite3"
    cache = CacheStore(cache_path, repository_root=dataset_root)
    first_trace: list[_AICacheTrace] = []
    first = await _load_or_describe_single(
        source,
        processed,
        model="primary-model",
        config=config,
        cache=cache,
        budget=RequestBudget(max_requests=1),
        ai_state=_AIState(),
        api_key="unused-test-key",
        context=context,
        temporary=TemporaryMediaRun(),
        taxonomy_version="1.0.0",
        cache_alias_scope=run_scope,
        trace_out=first_trace,
    )
    assert first.result.model_revision == "revision-paid-once"
    assert first.generated_at == "2026-01-02T03:04:05Z"

    # The request key intentionally contains a null revision, but a run-scoped
    # alias resolves it during an interruption/resume of this same run.
    second = await _load_or_describe_single(
        source,
        processed,
        model="primary-model",
        config=config,
        cache=cache,
        budget=RequestBudget(max_requests=0),
        ai_state=_AIState(),
        api_key=None,
        context=context,
        temporary=TemporaryMediaRun(),
        taxonomy_version="1.0.0",
        cache_alias_scope=run_scope,
    )
    assert second == first
    assert descriptions == 1
    assert provider_initializations == 1

    # A new independent run must not inherit the old null-revision alias. It
    # calls the provider and can observe a newly deployed revision.
    next_run = await _load_or_describe_single(
        source,
        processed,
        model="primary-model",
        config=config,
        cache=cache,
        budget=RequestBudget(max_requests=1),
        ai_state=_AIState(),
        api_key="unused-test-key",
        context=context,
        temporary=TemporaryMediaRun(),
        taxonomy_version="1.0.0",
        cache_alias_scope="mlxrun_" + "b" * 32,
    )
    assert next_run.result.model_revision == "revision-next-run"
    assert descriptions == 2
    assert provider_initializations == 2

    actual_key = _cache_key(
        source,
        processed,
        context,
        config,
        model="primary-model",
        model_revision="revision-paid-once",
        taxonomy_version="1.0.0",
        request_identity=first_trace[0].request_identity,
        item_label="E001",
    )
    cache.close()
    connection = sqlite3.connect(cache_path)
    connection.execute("DELETE FROM ai_cache_alias")
    connection.commit()
    connection.close()

    # Even if the alias is unavailable, the exact ai_cache_key persisted in a
    # run checkpoint is sufficient to resume without another provider call.
    with CacheStore(cache_path, repository_root=dataset_root) as reopened:
        resumed = await _load_or_describe_single(
            source,
            processed,
            model="primary-model",
            config=config,
            cache=reopened,
            budget=RequestBudget(max_requests=0),
            ai_state=_AIState(),
            api_key=None,
            context=context,
            temporary=TemporaryMediaRun(),
            taxonomy_version="1.0.0",
            resume_cache_key=actual_key,
            request_identity=first_trace[0].request_identity,
        )
    assert resumed == first
    assert descriptions == 2
    assert provider_initializations == 2


@pytest.mark.asyncio
async def test_cache_hit_uses_original_generation_time_for_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    write_fixture(dataset_root)
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    source = _source()
    processed = _processed()
    context = _vision_context(source, processed)
    revision = "revision-before-qualification"
    description = _description("Generated too early.", sensitive=False)
    result = DescriptionResult(
        batch=DescriptionBatch(items=(description,)),
        provider="gemini",
        model="primary-model",
        model_revision=revision,
    )
    template = ModelQualificationRegistry.load(dataset_root).entries[0].model_dump(mode="json")
    qualification = ModelQualification.model_validate(
        {
            **template,
            "qualification_id": "mq_standard-v1_future",
            "provider": "gemini",
            "model": "primary-model",
            "model_revision": revision,
            "prompt_sha256": prompt_sha256(),
            "request_parameters_sha256": gemini_request_parameters_sha256(),
            "valid_from": "2026-06-01T00:00:00Z",
            "valid_until": None,
        }
    )
    qualifications = ModelQualificationRegistry(
        schema_version="1.0.0",
        registry_id="model-qualifications-v1",
        entries=(qualification,),
    )
    request_identity = _request_identity(source, processed, model="primary-model")
    actual_key = _cache_key(
        source,
        processed,
        context,
        config,
        model="primary-model",
        model_revision=revision,
        taxonomy_version="1.0.0",
        request_identity=request_identity,
        item_label="E001",
    )
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=dataset_root)
    cache.put_ai(
        actual_key,
        result,
        generated_at="2026-01-01T00:00:00Z",
    )
    # The old buggy behavior used this later wall-clock instant and would have
    # incorrectly granted the qualification.
    monkeypatch.setattr(pipeline_runner, "_utc_text", lambda: "2026-09-01T00:00:00Z")

    outcome = (
        await _describe_batch(
            (source,),
            {source.native_id: processed},
            config=config,
            cache=cache,
            budget=RequestBudget(max_requests=0),
            ai_state=_AIState(),
            api_key=None,
            temporary=TemporaryMediaRun(),
            taxonomy_version="1.0.0",
            qualifications=qualifications,
            routing_registry=RoutingReasonRegistry.load(dataset_root),
            resume_request_traces={
                source.native_id: (
                    _AICacheTrace(
                        stage="primary",
                        model="primary-model",
                        model_revision=revision,
                        cache_key=actual_key,
                        request_identity=request_identity,
                    ),
                )
            },
            require_exact_resume=True,
        )
    )[source.native_id]

    assert outcome.generation.generated_at == "2026-01-01T00:00:00Z"
    assert outcome.generation.qualification_id is None
    assert outcome.generation.routing_reason_codes == ("unqualified-model",)
    cache.close()


@pytest.mark.asyncio
async def test_cache_alias_with_different_request_context_fails_closed(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    write_fixture(dataset_root)
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    original = _source()
    changed = original.model_copy(update={"fallback_emoji": "🙂"})
    processed = _processed()
    original_context = _vision_context(original, processed)
    changed_context = _vision_context(changed, processed)
    original_identity = _request_identity(original, processed, model="primary-model")
    changed_identity = _request_identity(changed, processed, model="primary-model")
    actual_key = _cache_key(
        original,
        processed,
        original_context,
        config,
        model="primary-model",
        model_revision="revision-1",
        taxonomy_version="1.0.0",
        request_identity=original_identity,
        item_label="E001",
    )
    poisoned_alias = _cache_key(
        changed,
        processed,
        changed_context,
        config,
        model="primary-model",
        model_revision=None,
        taxonomy_version="1.0.0",
        request_identity=changed_identity,
        item_label="E001",
    )
    run_scope = "mlxrun_" + "c" * 32
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=dataset_root)
    cache.put_ai(
        actual_key,
        DescriptionResult(
            batch=DescriptionBatch(items=(_description("Original context.", sensitive=False),)),
            provider="gemini",
            model="primary-model",
            model_revision="revision-1",
        ),
        generated_at="2026-01-01T00:00:00Z",
        aliases=_cache_aliases(run_scope, poisoned_alias),
    )

    with pytest.raises(AIOutputError, match="current media, prompt, or policy context"):
        await _load_or_describe_single(
            changed,
            processed,
            model="primary-model",
            config=config,
            cache=cache,
            budget=RequestBudget(max_requests=0),
            ai_state=_AIState(),
            api_key=None,
            context=changed_context,
            temporary=TemporaryMediaRun(),
            taxonomy_version="1.0.0",
            cache_alias_scope=run_scope,
            request_identity=changed_identity,
        )
    cache.close()


@pytest.mark.asyncio
async def test_batch_cache_identity_binds_full_ordered_plan_and_item_label(
    tmp_path: Path,
) -> None:
    frame = tmp_path / "frame.png"
    Image.new("RGBA", (256, 256), (255, 0, 0, 255)).save(frame, format="PNG")
    processed = _processed().model_copy(update={"frame_paths": (frame,)})
    first = _source_variant("native-a", 0)
    second = _source_variant("native-b", 1)
    third = _source_variant("native-c", 1)
    values = {
        first.native_id: processed,
        second.native_id: processed,
        third.native_id: processed,
    }

    with TemporaryMediaRun(root=tmp_path) as temporary:
        batch_ab = await pipeline_runner._prepare_ai_request(
            (first, second), values, model="primary-model", temporary=temporary
        )
        batch_ac = await pipeline_runner._prepare_ai_request(
            (first, third), values, model="primary-model", temporary=temporary
        )
        single_a = await pipeline_runner._prepare_ai_request(
            (first,), values, model="primary-model", temporary=temporary
        )

    context = _vision_context(first, processed)
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    key_a_ab = _cache_key(
        first,
        processed,
        context,
        config,
        model="primary-model",
        model_revision=None,
        taxonomy_version="1.0.0",
        request_identity=batch_ab.identity,
        item_label="E001",
    )
    key_b_ab = _cache_key(
        second,
        processed,
        _vision_context(second, processed),
        config,
        model="primary-model",
        model_revision=None,
        taxonomy_version="1.0.0",
        request_identity=batch_ab.identity,
        item_label="E002",
    )
    key_a_ac = _cache_key(
        first,
        processed,
        context,
        config,
        model="primary-model",
        model_revision=None,
        taxonomy_version="1.0.0",
        request_identity=batch_ac.identity,
        item_label="E001",
    )
    key_a_single = _cache_key(
        first,
        processed,
        context,
        config,
        model="primary-model",
        model_revision=None,
        taxonomy_version="1.0.0",
        request_identity=single_a.identity,
        item_label="E001",
    )

    assert batch_ab.identity.request_sha256 == batch_ac.identity.request_sha256
    assert batch_ab.identity.plan_sha256 != batch_ac.identity.plan_sha256
    assert len({key_a_ab, key_b_ab, key_a_ac, key_a_single}) == 4


@pytest.mark.asyncio
async def test_partial_exact_batch_cache_still_sends_full_original_chunk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    write_fixture(dataset_root)
    frame = tmp_path / "frame.png"
    Image.new("RGBA", (256, 256), (0, 128, 255, 255)).save(frame, format="PNG")
    processed = _processed().model_copy(update={"frame_paths": (frame,)})
    first = _source_variant("native-a", 0)
    second = _source_variant("native-b", 1)
    items = (first, second)
    values = {item.native_id: processed for item in items}
    config = MojiLexConfig(
        ai=AIConfig(provider="gemini", model="primary-model", model_routing="off")
    )
    cache = CacheStore(tmp_path / "cache.sqlite3", repository_root=dataset_root)
    provider = _BatchProvider()

    with TemporaryMediaRun(root=tmp_path) as temporary:
        prepared = await pipeline_runner._prepare_ai_request(
            items, values, model="primary-model", temporary=temporary
        )
        _put_result(
            cache,
            dataset_root,
            config,
            first,
            processed,
            model="primary-model",
            description=_description("Only A was cached.", sensitive=False),
            identity=prepared.identity,
        )

        async def fixed_prepare(*args: object, **kwargs: object) -> object:
            return prepared

        async def fake_provider(*args: object, **kwargs: object) -> _BatchProvider:
            return provider

        monkeypatch.setattr(pipeline_runner, "_prepare_ai_request", fixed_prepare)
        monkeypatch.setattr(pipeline_runner, "_provider_for_model", fake_provider)
        outcomes = await _describe_batch(
            items,
            values,
            config=config,
            cache=cache,
            budget=RequestBudget(max_requests=1),
            ai_state=_AIState(),
            api_key="unused-test-key",
            temporary=temporary,
            taxonomy_version="1.0.0",
            qualifications=_qualifications(dataset_root),
            routing_registry=RoutingReasonRegistry.load(dataset_root),
        )

    assert set(outcomes) == {first.native_id, second.native_id}
    assert len(provider.calls) == 1
    assert provider.calls[0].expected_labels == ("E001", "E002")
    cache.close()
