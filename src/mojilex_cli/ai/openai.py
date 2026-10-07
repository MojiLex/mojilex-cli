"""OpenAI Responses vision adapter with exactly one HTTP attempt per reservation."""

from __future__ import annotations

import asyncio
import base64
import math
import re
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from .base import (
    AIError,
    AIOutputError,
    AIPaymentRequiredError,
    AITransientError,
    AIUsage,
    CostEstimate,
    DescriptionBatch,
    DescriptionRequest,
    DescriptionResult,
    ProviderCapabilities,
    validate_result_labels,
)
from .gemini import _request_prompt, _retry_after_seconds, _safe_validation_summary
from .openai_schema import openai_transport_schema
from .prompts import openai_request_parameters

_API = "https://api.openai.com/v1/"


class OpenAIVisionProvider:
    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 90,
    ) -> None:
        if not model or any(ord(char) < 33 or ord(char) == 127 for char in model):
            raise ValueError("OpenAI model must be selected explicitly")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("OpenAI timeout must be finite and positive")
        if client is None and not api_key:
            raise AIError("OpenAI credential is missing")
        self.model = model
        self._api_key = api_key
        self._client = client
        self._timeout_seconds = timeout_seconds

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            structured_json=True,
            image_mime_types=("image/png",),
            max_images=32,
            supports_cost_estimate=False,
        )

    def estimate(self, request: DescriptionRequest) -> CostEstimate:
        if request.model != self.model:
            raise AIError("request model differs from the configured OpenAI model")
        return CostEstimate(
            upper_bound_usd=None,
            note="No current externally supplied price record exists for this model.",
        )

    async def _send(self, method: str, endpoint: str, body: dict[str, Any] | None = None) -> Any:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        try:
            if self._client is None:
                # No implicit retries or redirects. Only the budgeted outer recovery retries.
                async with httpx.AsyncClient(timeout=self._timeout_seconds) as client:
                    response = await asyncio.wait_for(
                        client.request(method, _API + endpoint, headers=headers, json=body),
                        timeout=self._timeout_seconds,
                    )
            else:
                response = await asyncio.wait_for(
                    self._client.request(
                        method,
                        _API + endpoint,
                        headers=headers,
                        json=body,
                        timeout=self._timeout_seconds,
                        follow_redirects=False,
                    ),
                    timeout=self._timeout_seconds,
                )
        except (TimeoutError, httpx.TimeoutException):
            raise AITransientError(
                f"OpenAI request timed out after {self._timeout_seconds:g} seconds."
            ) from None
        except httpx.TransportError as exc:
            raise AITransientError(f"OpenAI connection failed: {type(exc).__name__}") from None
        if len(response.content) > 2 * 1024 * 1024:
            raise AIOutputError("OpenAI structured response: response_too_large")
        try:
            payload = response.json()
        except (ValueError, UnicodeError):
            payload = None
        if response.status_code != 200:
            error = payload.get("error") if isinstance(payload, dict) else None
            quota_exhausted = isinstance(error, dict) and (
                error.get("code") in ("insufficient_quota", "billing_hard_limit_reached")
                or error.get("type") == "insufficient_quota"
            )
            if response.status_code == 402 or quota_exhausted:
                raise AIPaymentRequiredError(
                    "OpenAI requires API billing or credits; "
                    "check provider billing before resuming."
                ) from None
            message = f"OpenAI request failed: HTTP {response.status_code}"
            if response.status_code in {408, 409, 429, 500, 502, 503, 504}:
                error_for_retry = httpx.HTTPStatusError(
                    message, request=response.request, response=response
                )
                raise AITransientError(
                    message, retry_after_seconds=_retry_after_seconds(error_for_retry)
                ) from None
            if response.status_code == 401:
                message += "; check the OpenAI API key"
            elif response.status_code in {403, 404}:
                message += "; check model access and the exact model ID"
            raise AIError(message) from None
        if not isinstance(payload, dict):
            raise AIOutputError("OpenAI structured response: invalid_response")
        return payload

    async def validate_credentials(self) -> None:
        # This read-only model lookup does not generate or charge inference tokens.
        await self._send("GET", "models/" + quote(self.model, safe=""))

    async def structured_response(
        self,
        *,
        prompt: str,
        images: tuple[bytes, ...],
        schema: dict[str, Any],
        max_output_tokens: int,
        schema_name: str,
    ) -> dict[str, Any]:
        """The caller reserves its shared budget before this single paid call."""
        parameters = openai_request_parameters()
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        content.extend(
            {
                "type": "input_image",
                "detail": parameters["image_detail"],
                "image_url": "data:image/png;base64," + base64.b64encode(data).decode("ascii"),
            }
            for data in images
        )
        body = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "store": parameters["store"],
            "background": parameters["background"],
            "stream": parameters["stream"],
            "reasoning": parameters["reasoning"],
            "max_output_tokens": max_output_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": openai_transport_schema(schema),
                }
            },
        }
        payload = await self._send("POST", "responses", body)
        _completed_text(payload, self.model)
        return payload  # type: ignore[no-any-return]

    async def describe(self, request: DescriptionRequest) -> DescriptionResult:
        if request.model != self.model:
            raise AIError("request model differs from the configured OpenAI model")
        parameters = openai_request_parameters()
        payload = await self.structured_response(
            prompt=_request_prompt(request),
            images=tuple(image.data for image in request.images),
            schema=DescriptionBatch.model_json_schema(),
            max_output_tokens=parameters["max_output_tokens"],
            schema_name=parameters["text"]["format"]["name"],
        )
        try:
            batch = DescriptionBatch.model_validate_json(_completed_text(payload, self.model))
        except ValidationError as exc:
            raise AIOutputError(_safe_validation_summary(exc, provider="OpenAI")) from None
        usage = payload.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        # output_tokens already includes reasoning tokens; never add them again.
        result = DescriptionResult(
            batch=batch,
            provider=self.name,
            model=self.model,
            model_revision=None,
            usage=AIUsage(
                input_tokens=_token_count(usage.get("input_tokens")),
                output_tokens=_token_count(usage.get("output_tokens")),
            ),
        )
        validate_result_labels(result, request.expected_labels)
        return result


def _completed_text(payload: dict[str, Any], model: str) -> str:
    if payload.get("status") != "completed":
        raise AIOutputError("OpenAI structured response: response_incomplete")
    actual_model = payload.get("model")
    if actual_model != model and not (
        isinstance(actual_model, str)
        and re.fullmatch(re.escape(model) + r"-\d{4}-\d{2}-\d{2}", actual_model)
    ):
        raise AIOutputError("OpenAI structured response: model_mismatch")
    output = payload.get("output")
    if not isinstance(output, list):
        raise AIOutputError("OpenAI structured response: output_missing")
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            raise AIOutputError("OpenAI structured response: invalid_response")
        if item.get("type") == "reasoning":
            continue
        if item.get("type") != "message" or item.get("role") != "assistant":
            raise AIOutputError("OpenAI structured response: unexpected_output")
        if item.get("status") != "completed" or not isinstance(item.get("content"), list):
            raise AIOutputError("OpenAI structured response: response_incomplete")
        for part in item["content"]:
            if not isinstance(part, dict):
                raise AIOutputError("OpenAI structured response: invalid_response")
            if part.get("type") == "refusal":
                raise AIOutputError("OpenAI structured response: refusal")
            if part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                raise AIOutputError("OpenAI structured response: unexpected_output")
            texts.append(part["text"])
    if len(texts) != 1 or not texts[0].strip():
        raise AIOutputError("OpenAI structured response: output_missing")
    return texts[0]


def _token_count(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None
