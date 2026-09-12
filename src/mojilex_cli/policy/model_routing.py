"""Hash-bound local Stage-B routing configuration; never a trust attestation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import rfc8785
from pydantic import ConfigDict, Field, field_validator, model_validator

from .qualification import ConceptQualificationBinding
from .routing import ROUTING_POLICY_VERSION, PolicyError

_SHA256 = r"^[0-9a-f]{64}$"
LOCAL_MODEL_ROUTING_POLICY_ID = "model-routing-local-v1"
_TRIGGER_ORDER = (
    "schema-invalid-after-retry",
    "critical-uncertainty",
    "literal-text-low-confidence",
    "safety-filter",
)


class LocalGenerationConfiguration(ConceptQualificationBinding):
    """Actual current CLI configuration, not the later attested core projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=200)
    model_revision: str | None = Field(default=None, min_length=1, max_length=256)
    description_profile: Literal["standard-v1"]
    schema_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    taxonomy_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    pipeline_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    prompt_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    prompt_sha256: str = Field(pattern=_SHA256)
    request_parameters_sha256: str = Field(pattern=_SHA256)
    languages: tuple[Literal["en", "ru"], ...]

    @field_validator("provider", "model", "model_revision")
    @classmethod
    def safe_identifier(cls, value: str | None) -> str | None:
        if value is not None and (
            value != value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
            or "://" in value
            or "\\" in value
        ):
            raise ValueError("model selector must be a safe ID, never credentials or a URL")
        return value

    @field_validator("languages")
    @classmethod
    def canonical_languages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if value != ("en", "ru"):
            raise ValueError("local generation languages must be exactly en, ru in canonical order")
        return value

    @model_validator(mode="after")
    def input_has_only_concept_group(self) -> LocalGenerationConfiguration:
        # These fields would make the configuration selector self-referential.
        if self.model_routing_policy_id is not None or self.model_routing_policy_sha256 is not None:
            raise ValueError("routing selector input must omit its own policy fields")
        required = (
            self.concept_registry_id,
            self.concept_registry_sha256,
            self.concept_candidate_set_sha256,
            self.concept_candidate_profile_id,
            self.concept_candidate_profile_sha256,
        )
        if any(value is None for value in required):
            raise ValueError("local generation configuration requires all five concept fields")
        return self

    @model_validator(mode="after")
    def complete_concept_binding(self) -> LocalGenerationConfiguration:
        # Override the public seven-field atomic group: a selector deliberately
        # contains only the five concept fields before the routing hash exists.
        return self

    def selector_bytes(self) -> bytes:
        return rfc8785.dumps(
            self.model_dump(
                mode="json",
                exclude={"model_routing_policy_id", "model_routing_policy_sha256"},
            )
        )


@dataclass(frozen=True, slots=True)
class ModelRoutingBinding:
    raw_bytes: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw_bytes).hexdigest()

    @property
    def provenance_fields(self) -> dict[str, str]:
        return {
            "model_routing_policy_id": LOCAL_MODEL_ROUTING_POLICY_ID,
            "model_routing_policy_sha256": self.sha256,
        }

    @property
    def body(self) -> dict[str, Any]:
        import json

        value: dict[str, Any] = json.loads(self.raw_bytes)
        return value


def build_model_routing_binding(
    primary_configuration: Mapping[str, Any],
    escalated_configuration: Mapping[str, Any] | None = None,
    *,
    mode: Literal["off", "rules"] = "off",
) -> ModelRoutingBinding:
    """Bind exact model choices, inputs and routing mode without storing secrets.

    The local ID explicitly distinguishes this Stage-B selector projection from
    SPEC-003's future signed generation-configuration-v1 qualification contract.
    The body belongs in safe run state/audit evidence; it grants no qualification.
    """

    try:
        primary = LocalGenerationConfiguration.model_validate(dict(primary_configuration))
        escalation = (
            LocalGenerationConfiguration.model_validate(dict(escalated_configuration))
            if escalated_configuration is not None
            else None
        )
        if mode not in {"off", "rules"} or (mode == "rules") != (escalation is not None):
            raise ValueError("rules mode requires exactly one explicit escalation configuration")
        if escalation is not None:
            excluded = {"provider", "model", "model_revision"}
            if primary.model_dump(exclude=excluded) != escalation.model_dump(exclude=excluded):
                raise ValueError(
                    "escalation must use the same original input and semantic contract"
                )
        selectors = [
            {
                "stage": "primary",
                "configuration_selector_sha256": hashlib.sha256(
                    primary.selector_bytes()
                ).hexdigest(),
            }
        ]
        if escalation is not None:
            selectors.append(
                {
                    "stage": "escalated",
                    "configuration_selector_sha256": hashlib.sha256(
                        escalation.selector_bytes()
                    ).hexdigest(),
                }
            )
        body: dict[str, Any] = {
            "schema_version": "1.0.0",
            "policy_id": LOCAL_MODEL_ROUTING_POLICY_ID,
            "policy_version": ROUTING_POLICY_VERSION,
            "configuration_selectors": selectors,
            "trigger_order": list(_TRIGGER_ORDER) if escalation is not None else [],
            "max_escalations_per_item": 1,
            "merge_policy": "whole-result-only-v1",
        }
        return ModelRoutingBinding(rfc8785.dumps(body))
    except (ValueError, TypeError) as exc:
        raise PolicyError("local model-routing configuration is invalid or incomplete") from exc
