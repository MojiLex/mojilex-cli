"""Gemini adapter using the official ``google-genai`` async SDK."""

from __future__ import annotations

import json
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
from .prompts import build_context_prompt, build_prompt


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
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                )
            except Exception as exc:
                raise AIError(f"cannot initialize Gemini SDK: {type(exc).__name__}") from None
        self.model = model
        self._client = client
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
            await self._client.aio.models.get(model=self.model)
        except Exception as exc:
            raise AIError(
                f"Gemini credential/model validation failed: {type(exc).__name__}"
            ) from None

    def estimate(self, request: DescriptionRequest) -> CostEstimate:
        if request.model != self.model:
            raise AIError("request model differs from the configured Gemini model")
        if self._pricing is None:
            return CostEstimate(
                upper_bound_usd=None,
                note="No current externally supplied price record exists for this model.",
            )
        prompt_tokens = max(1, len(_request_prompt(request)) // 3)
        input_tokens = (
            prompt_tokens + len(request.images) * self._pricing.conservative_tokens_per_image
        )
        input_cost = (
            Decimal(input_tokens) * self._pricing.input_usd_per_million_tokens / Decimal(1_000_000)
        )
        output_cost = (
            Decimal(self._pricing.conservative_output_tokens)
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
            from google.genai import types

            contents: list[Any] = [_request_prompt(request)]
            contents.extend(
                types.Part.from_bytes(data=image.data, mime_type=image.mime_type)
                for image in request.images
            )
            response = await self._client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DescriptionBatch,
                    temperature=0,
                ),
            )
        except AIError:
            raise
        except Exception as exc:
            # Deliberately omit exception text: SDK errors can include request details.
            raise AIError(f"Gemini request failed: {type(exc).__name__}") from None
        try:
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, DescriptionBatch):
                batch = parsed
            elif parsed is not None:
                batch = DescriptionBatch.model_validate(parsed)
            else:
                text = getattr(response, "text", None)
                if not isinstance(text, str):
                    raise ValueError("missing structured response")
                batch = DescriptionBatch.model_validate_json(text)
        except (ValidationError, ValueError, TypeError) as exc:
            raise AIOutputError("Gemini returned invalid structured JSON") from exc
        usage_metadata = getattr(response, "usage_metadata", None)
        usage = AIUsage(
            input_tokens=_int_or_none(getattr(usage_metadata, "prompt_token_count", None)),
            output_tokens=_int_or_none(getattr(usage_metadata, "candidates_token_count", None)),
        )
        result = DescriptionResult(
            batch=batch,
            provider=self.name,
            model=self.model,
            model_revision=_string_or_none(getattr(response, "model_version", None)),
            usage=usage,
        )
        validate_result_labels(result, request.expected_labels)
        return result


def _animated(request: DescriptionRequest) -> bool:
    return any(context.frame_count > 1 for context in request.context.values())


def _request_prompt(request: DescriptionRequest) -> str:
    allowed_context = {
        label: request.context[label].model_dump(mode="json") for label in request.expected_labels
    }
    context_json = json.dumps(
        allowed_context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return (
        build_prompt(request.expected_labels, animated=_animated(request))
        + "\n"
        + build_context_prompt(context_json)
    )


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and value >= 0 else None


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) and value else None
