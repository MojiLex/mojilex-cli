"""Gemini adapter using the official ``google-genai`` async SDK."""

from __future__ import annotations

import asyncio
import base64
import json
import math
from datetime import date
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .base import (
    AIError,
    AIOutputError,
    AIUsage,
    CostEstimate,
    DescriptionBatch,
    DescriptionRequest,
    DescriptionResult,
    ProviderCapabilities,
    validate_result_labels,
)
from .prompts import build_context_prompt, build_prompt, gemini_request_parameters
from .runtime_parameters import MAX_OUTPUT_TOKENS


class ModelPricing(BaseModel):
    """User/updater supplied pricing; no stale permanent table is embedded in code."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str
    input_usd_per_million_tokens: Decimal = Field(ge=0)
    output_usd_per_million_tokens: Decimal = Field(ge=0)
    conservative_tokens_per_image: int = Field(gt=0)
    conservative_output_tokens: int = Field(gt=0)
    updated_at: date


class GeminiVisionProvider:
    name = "gemini"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
        pricing: ModelPricing | None = None,
        timeout_seconds: float = 30,
    ) -> None:
        if not model:
            raise ValueError("Gemini model must be selected explicitly")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Gemini timeout must be finite and positive")
        if client is None:
            if not api_key:
                raise AIError("Gemini credential is missing")
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(
                        timeout=int(timeout_seconds * 1000),
                        # The outer budget counts every logical call. Disable hidden SDK retries.
                        # The SDK bridge also needs the per-instance guard below:
                        # google-genai 2.23 mutates zero to one during initialization.
                        retry_options=types.HttpRetryOptions(attempts=0),
                    ),
                )
            except Exception as exc:
                raise AIError(f"cannot initialize Gemini SDK: {type(exc).__name__}") from None
        self.model = model
        self._client = client
        self._timeout_seconds = timeout_seconds
        self._pricing = pricing if pricing is not None and pricing.model == model else None

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            structured_json=True,
            image_mime_types=("image/png",),
            max_images=32,
            supports_cost_estimate=self._pricing is not None,
        )

    async def validate_credentials(self) -> None:
        try:
            await asyncio.wait_for(
                self._client.aio.models.get(model=self.model), timeout=self._timeout_seconds
            )
        except TimeoutError:
            raise AIError(
                f"Gemini model check timed out after {self._timeout_seconds:g} seconds."
            ) from None
        except Exception as exc:
            raise AIError(
                f"Gemini credential/model validation failed: "
                f"{type(exc).__name__}{_safe_provider_status(exc)}"
            ) from None

    def estimate(self, request: DescriptionRequest) -> CostEstimate:
        if request.model != self.model:
            raise AIError("request model differs from the configured Gemini model")
        if self._pricing is None:
            return CostEstimate(
                upper_bound_usd=None,
                note="No current externally supplied price record exists for this model.",
            )
        # Count UTF-8 bytes conservatively, including the output schema, instead
        # of the non-conservative characters/3 heuristic (not safe for Cyrillic).
        prompt_tokens = max(1, len(_request_prompt(request).encode("utf-8")))
        prompt_tokens += (
            len(json.dumps(DescriptionBatch.model_json_schema()).encode("utf-8")) + 2048
        )
        input_tokens = (
            prompt_tokens + len(request.images) * self._pricing.conservative_tokens_per_image
        )
        input_cost = (
            Decimal(input_tokens) * self._pricing.input_usd_per_million_tokens / Decimal(1_000_000)
        )
        output_cost = (
            Decimal(max(MAX_OUTPUT_TOKENS, self._pricing.conservative_output_tokens))
            * self._pricing.output_usd_per_million_tokens
            / Decimal(1_000_000)
        )
        return CostEstimate(
            upper_bound_usd=input_cost + output_cost,
            pricing_updated_at=self._pricing.updated_at,
            note="Conservative estimate from an externally supplied dated price record.",
        )

    async def describe(self, request: DescriptionRequest) -> DescriptionResult:
        if request.model != self.model:
            raise AIError("request model differs from the configured Gemini model")
        try:
            parameters = gemini_request_parameters()
            content: list[dict[str, Any]] = [{"type": "text", "text": _request_prompt(request)}]
            content.extend(
                {
                    "type": "image",
                    "data": base64.b64encode(image.data).decode("ascii"),
                    "mime_type": image.mime_type,
                }
                for image in request.images
            )
            interactions = self._client.aio.interactions
            _disable_interaction_retries(interactions)
            response = await asyncio.wait_for(
                interactions.create(
                    model=self.model,
                    api_version=parameters["api_version"],
                    input=[{"type": "user_input", "content": content}],
                    store=parameters["store"],
                    background=parameters["background"],
                    stream=parameters["stream"],
                    response_format=parameters["response_format"],
                    generation_config=parameters["generation_config"],
                ),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            raise AIError(
                f"Gemini request timed out after {self._timeout_seconds:g} seconds."
            ) from None
        except AIError:
            raise
        except Exception as exc:
            # Deliberately omit exception text: SDK errors can include request details.
            raise AIError(
                f"Gemini request failed: {type(exc).__name__}{_safe_provider_status(exc)}"
            ) from None
        try:
            if getattr(response, "status", None) != "completed":
                raise ValueError("interaction is incomplete")
            if getattr(response, "model", None) not in {self.model, f"models/{self.model}"}:
                raise ValueError("interaction returned a different model")
            text = getattr(response, "output_text", None)
            if not isinstance(text, str) or not text:
                raise ValueError("missing structured response")
            batch = DescriptionBatch.model_validate_json(text)
        except (ValidationError, ValueError, TypeError):
            # Validation errors may quote generated content; never preserve their chain.
            raise AIOutputError("Gemini returned invalid or incomplete structured JSON") from None
        usage_metadata = getattr(response, "usage", None)
        usage = AIUsage(
            input_tokens=_int_or_none(getattr(usage_metadata, "total_input_tokens", None)),
            output_tokens=_output_token_count(usage_metadata),
        )
        result = DescriptionResult(
            batch=batch,
            provider=self.name,
            model=self.model,
            # Interactions returns a model alias, not a documented immutable revision.
            # Never upgrade that alias to revision evidence.
            model_revision=None,
            usage=usage,
        )
        validate_result_labels(result, request.expected_labels)
        return result


def _disable_interaction_retries(interactions: Any) -> None:
    """One budget reservation must mean one HTTP attempt, including SDK errors.

    google-genai 2.23's bridge turns legacy attempts=0 into one *extra* retry.
    Configure only this resource instance, never mutate the installed SDK.
    """

    configuration = getattr(interactions, "sdk_configuration", None)
    if configuration is None:
        return  # Injected provider doubles have no SDK-owned retry machinery.
    retry = getattr(configuration, "retry_config", None)
    if retry is None or not hasattr(retry, "strategy"):
        raise AIError("Gemini SDK cannot enforce the one-attempt request budget")
    retry.strategy = "none"
    retry.max_retries = 0
    retry.retry_connection_errors = False


def _safe_provider_status(exc: Exception) -> str:
    """Keep only bounded status values from documented closed vocabularies."""

    parts: list[str] = []
    code = getattr(exc, "code", getattr(exc, "status_code", None))
    if type(code) is int and 100 <= code <= 599:
        parts.append(f"http={code}")
    status = getattr(exc, "status", None)
    if isinstance(status, str) and status in {
        "CANCELLED",
        "UNKNOWN",
        "INVALID_ARGUMENT",
        "DEADLINE_EXCEEDED",
        "NOT_FOUND",
        "ALREADY_EXISTS",
        "PERMISSION_DENIED",
        "RESOURCE_EXHAUSTED",
        "FAILED_PRECONDITION",
        "ABORTED",
        "OUT_OF_RANGE",
        "UNIMPLEMENTED",
        "INTERNAL",
        "UNAVAILABLE",
        "DATA_LOSS",
        "UNAUTHENTICATED",
    }:
        parts.append(f"status={status}")
    body = getattr(exc, "body", None)
    provider_error = body.get("error") if isinstance(body, dict) else None
    provider_code = provider_error.get("code") if isinstance(provider_error, dict) else None
    if isinstance(provider_code, str) and provider_code in {
        "invalid_request",
        "failed_precondition",
        "out_of_range",
        "parameter_unknown",
        "authentication",
        "permission_denied",
        "not_found",
        "model_not_found",
        "already_exists",
        "aborted",
        "rate_limit_exceeded",
        "quota_exceeded",
        "too_many_requests",
        "cancelled",
        "api_error",
        "unimplemented",
        "service_unavailable",
        "deadline_exceeded",
        "safety",
        "recitation",
        "language",
        "prohibited_content",
        "spii",
        "blocklist",
        "image_safety",
        "image_prohibited_content",
        "image_recitation",
        "image_other",
        "content_blocked",
        "malformed_function_call",
        "malformed_tool_call",
        "unexpected_tool_call",
        "no_image",
        "too_many_tool_calls",
        "missing_thought_signature",
    }:
        parts.append(f"provider_code={provider_code}")
    return " (" + ", ".join(parts) + ")" if parts else ""


def _animated(request: DescriptionRequest) -> bool:
    return any(context.frame_count > 1 for context in request.context.values())


def _request_prompt(request: DescriptionRequest) -> str:
    allowed_context = {
        label: request.context[label].model_dump(mode="json") for label in request.expected_labels
    }
    context_payload = (
        {"items": allowed_context, "concept_context": request.concept_context}
        if request.concept_context is not None
        else allowed_context
    )
    context_json = json.dumps(
        context_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (
        build_prompt(request.expected_labels, animated=_animated(request))
        + "\n"
        + build_context_prompt(context_json)
    )


def _output_token_count(metadata: Any) -> int | None:
    # Interactions reports response and thought tokens separately (not double-counted).
    candidates = _int_or_none(getattr(metadata, "total_output_tokens", None))
    thoughts = _int_or_none(getattr(metadata, "total_thought_tokens", None))
    if candidates is None:
        return None
    return candidates + (thoughts or 0)


def _int_or_none(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None
