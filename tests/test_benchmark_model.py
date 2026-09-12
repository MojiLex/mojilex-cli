from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path

import pytest

from mojilex_cli.ai import (
    AIUsage,
    CostEstimate,
    DescriptionBatch,
    DescriptionItem,
    DescriptionRequest,
    DescriptionResult,
    ProviderCapabilities,
)
from mojilex_cli.ai.prompts import (
    PROMPT_VERSION,
    gemini_request_parameters_sha256,
    prompt_sha256,
)
from mojilex_cli.benchmark import (
    MANDATORY_MODEL_STRATA,
    AllowedFacetSets,
    DeclaredFile,
    HumanAdjudication,
    ManualScores,
    ModelBenchmarkCase,
    ModelBenchmarkManifest,
    ModelObservation,
    RequiredFact,
    RightsRecord,
    description_item_sha256,
    evaluate_model_benchmark,
    model_benchmark_assets_sha256,
    model_benchmark_split_sha256,
    run_model_benchmark,
)
from mojilex_cli.benchmark.common import BenchmarkError
from mojilex_cli.benchmark.model import _macro_f1_bp, _micro_f1_bp, _safety_pass
from mojilex_cli.commands import benchmark as benchmark_commands
from mojilex_cli.commands.runtime import CommandError
from test_ai_concepts import PROFILE, _registry


def _description(*, animated: bool, ambiguous: bool = False) -> DescriptionItem:
    ru = {
        "text": "Синтетическая улыбающаяся морда; надпись — OK.",
        "motion_status": "described" if animated else "not_applicable",
        "usage": ["реакция"],
    }
    en = {
        "text": "A synthetic smiling face with the text OK.",
        "motion_status": "described" if animated else "not_applicable",
        "usage": ["reaction"],
    }
    if animated:
        ru["motion"] = "Морда плавно улыбается."
        en["motion"] = "The face slowly smiles."
    return DescriptionItem.model_validate(
        {
            "label": "E001",
            "concept_ids": ["animal.cat"],
            "descriptions": {"ru": ru, "en": en},
            "facets": {
                "text_content": {
                    "status": "recognized",
                    "dynamics": "stable",
                    "items": [
                        {
                            "value": "OK",
                            "kind": "word",
                            "script": "Latn",
                            "language": "en",
                            "temporal_scope": "persistent",
                            "media_refs": [{"role": "primary"}],
                        }
                    ],
                },
                "content_types": ["reaction", "text"],
                "styles": ["flat"],
                "suggested_uses": ["message-accent"],
                "uncertainties": ["content-type"] if ambiguous else [],
            },
            "semantic_tags": ["face", "smile"],
            "content": {"rating": "general", "warnings": []},
        }
    )


def _case(
    index: int,
    stratum: str,
    *,
    image_sha256: str | None = None,
) -> ModelBenchmarkCase:
    animated = stratum in {"tgs-animation", "webm-animation"}
    expected = _description(
        animated=animated,
        ambiguous=stratum == "ambiguous-emotion-gesture",
    )
    digest = image_sha256 or hashlib.sha256(f"image-{index}".encode()).hexdigest()
    response_hash = description_item_sha256(expected)
    return ModelBenchmarkCase(
        case_id=f"case-{index:04d}",
        split="development" if index % 30 < 15 else "holdout",
        rights_id="synthetic",
        strata=(stratum,),
        isolation_groups=(f"artwork-{index:04d}",),
        source_media_sha256=hashlib.sha256(f"media-{index}".encode()).hexdigest(),
        image=DeclaredFile(path=f"images/case-{index:04d}.png", sha256=digest),
        needs_repainting=stratum == "adaptive",
        animated=animated,
        background_variants=("light", "dark") if stratum == "adaptive" else ("light",),
        expected=expected,
        allowed_facets=AllowedFacetSets(
            content_types=(expected.facets.content_types,),
            styles=(expected.facets.styles,),
            suggested_uses=(expected.facets.suggested_uses,),
            uncertainties=(expected.facets.uncertainties,),
        ),
        required_facts=(
            RequiredFact(
                fact_id="smiling-face",
                ru_phrases=("улыбающаяся",),
                en_phrases=("smiling",),
            ),
        ),
        forbidden_claims=("official brand",),
        text_present=True,
        text_readable=True,
        ambiguous=stratum == "ambiguous-emotion-gesture",
        brand_identity_present=False,
        injection_test=index % 40 == 0,
        adjudication=HumanAdjudication(
            response_sha256=response_hash,
            reviewer_count=1,
            hallucinated_observable_fact=False,
            brand_identity_hallucination=False,
            critical_error=False,
            ru_en_consistent=True,
            uncertainty_appropriate=True,
            full_pass=True,
            substantive_error_count=0,
            scores=ManualScores(
                factuality=5,
                main_content_completeness=5,
                no_extra_assumption=5,
                motion_accuracy=5 if animated else "not_applicable",
                text_accuracy=5,
                russian_naturalness=5,
                english_naturalness=5,
                facet_consistency=5,
            ),
        ),
    )


def _manifest(
    cases: tuple[ModelBenchmarkCase, ...],
    *,
    provider: str = "fake",
    model: str = "model-test",
    revision: str | None = "revision-1",
) -> ModelBenchmarkManifest:
    return ModelBenchmarkManifest(
        manifest_type="mojilex-model-benchmark-v1",
        schema_version="1.0.0",
        benchmark_id="golden-v1",
        benchmark_version="1.0.0",
        benchmark_assets_sha256=model_benchmark_assets_sha256(cases),
        split_sha256=model_benchmark_split_sha256(cases),
        comparison_kind="immutable-revision" if revision else "dated",
        run_started_at_utc="2026-09-11T00:00:00Z",
        target_provider=provider,
        target_model=model,
        target_model_revision=revision,
        description_profile="standard-v1",
        prompt_version=PROMPT_VERSION,
        prompt_sha256=prompt_sha256(),
        request_parameters_sha256=(
            gemini_request_parameters_sha256() if provider == "gemini" else "a" * 64
        ),
        output_schema_version="1.0.0",
        taxonomy_version="1.0.0",
        media_pipeline_version="1.0.0",
        routing_rules_version="1.0.0",
        languages=("ru", "en"),
        cli_commit="b" * 40,
        lockfile_sha256="c" * 64,
        rights=(
            RightsRecord(
                rights_id="synthetic",
                basis="synthetic",
                license_spdx="CC0-1.0",
                attribution="Generated by the MojiLex benchmark test suite.",
                redistribution_allowed=True,
            ),
        ),
        cases=cases,
        styles_macro_f1_min_bp=9_000,
        suggested_uses_macro_f1_min_bp=9_000,
        max_requests=max(4, len(cases) * 4),
        max_cost_usd=Decimal("10"),
        holdout_run_count=3,
        single_review_limitation=True,
        concept_registry=_registry(),
        concept_candidate_profile=PROFILE,
    )


def _observation(case: ModelBenchmarkCase) -> ModelObservation:
    response = case.expected
    return ModelObservation(
        case_id=case.case_id,
        schema_success=True,
        response=response,
        response_sha256=description_item_sha256(response),
        provider="fake",
        model="model-test",
        model_revision="revision-1",
        input_tokens=10,
        output_tokens=20,
        estimated_cost_usd=Decimal("0.001"),
        latency_ms=10,
        requests_used=1,
        safety_pass=True,
    )


def test_model_golden_report_passes_all_hard_gates_deterministically() -> None:
    cases = tuple(_case(index, MANDATORY_MODEL_STRATA[index // 30]) for index in range(240))
    manifest = _manifest(cases)
    observations = tuple(
        _observation(case).model_copy(update={"run_index": run_index})
        for case in cases
        for run_index in range(1, 4 if case.split == "holdout" else 2)
    )

    first = evaluate_model_benchmark(manifest, observations, manifest_sha256="d" * 64)
    second = evaluate_model_benchmark(manifest, observations, manifest_sha256="d" * 64)

    assert first == second
    assert first["passed"] is True
    assert all(first["gates"].values())
    assert first["metrics"]["literal_precision_bp"] == 10_000
    assert first["strata"]["animated"]["case_count"] == 60
    assert "text_presence_f1_bp" in first["strata"]["text"]["metric_ci95_bp"]
    assert first["case_count"] == 240
    assert first["observation_count"] == 480
    assert first["holdout_run_count"] == 3


def test_three_run_declaration_without_three_observations_fails_closed() -> None:
    case = _case(15, "static-full-color")
    report = evaluate_model_benchmark(
        _manifest((case,)), (_observation(case),), manifest_sha256="d" * 64
    )
    assert report["requested_holdout_run_count"] == 3
    assert report["holdout_run_count"] == 1
    assert report["gates"]["complex_holdout_has_at_least_3_runs"] is False
    assert report["gates"]["declared_runs_complete"] is False
    assert report["passed"] is False


@pytest.mark.parametrize("run_indices", [(1, 1, 2), (1, 2, 4), (2, 3)])
def test_duplicate_foreign_or_missing_initial_runs_are_rejected(run_indices) -> None:
    case = _case(15, "static-full-color")
    observations = tuple(
        _observation(case).model_copy(update={"run_index": index}) for index in run_indices
    )
    with pytest.raises(BenchmarkError, match="unique declared runs"):
        evaluate_model_benchmark(_manifest((case,)), observations, manifest_sha256="d" * 64)


def test_repeats_do_not_inflate_fixture_coverage_and_unreviewed_variation_blocks() -> None:
    case = _case(15, "static-full-color")
    altered_payload = case.expected.model_dump(mode="json")
    altered_payload["descriptions"]["en"]["text"] = "A different observation."
    altered = DescriptionItem.model_validate(altered_payload)
    observations = (
        *(_observation(case).model_copy(update={"run_index": index}) for index in (1, 2)),
        ModelObservation(
            **{
                **_observation(case).model_dump(mode="python"),
                "run_index": 3,
                "response": altered,
                "response_sha256": description_item_sha256(altered),
            }
        ),
    )
    report = evaluate_model_benchmark(_manifest((case,)), observations, manifest_sha256="d" * 64)
    assert report["gates"]["complex_holdout_has_at_least_3_runs"] is True
    assert report["gates"]["every_response_human_bound"] is False
    assert report["holdout_critical_error_count"] == 1
    assert report["case_count"] == 1
    assert report["observation_count"] == 3
    assert report["mandatory_strata_counts"]["static-full-color"] == 1
    assert report["strata"]["static"]["case_count"] == 1
    assert report["strata"]["static"]["observation_count"] == 3
    assert report["passed"] is False


class _FakeProvider:
    name = "fake"

    def __init__(self, response: DescriptionItem) -> None:
        self.response = response
        self.calls = 0

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            structured_json=True,
            image_mime_types=("image/png",),
            max_images=1,
            supports_cost_estimate=True,
        )

    async def validate_credentials(self) -> None:
        return None

    def estimate(self, request: DescriptionRequest) -> CostEstimate:
        assert request.expected_labels == ("E001",)
        return CostEstimate(upper_bound_usd=Decimal("0.001"), note="test")

    async def describe(self, request: DescriptionRequest) -> DescriptionResult:
        assert request.concept_context is not None
        assert request.concept_context["candidate_ids"] == ["animal.cat"]
        self.calls += 1
        return DescriptionResult(
            batch=DescriptionBatch(items=(self.response,)),
            provider=self.name,
            model=request.model,
            model_revision="revision-1",
            usage=AIUsage(
                input_tokens=1,
                output_tokens=2,
                estimated_cost_usd=Decimal("0.001"),
            ),
        )


@pytest.mark.asyncio
async def test_live_runner_uses_only_declared_hashed_png_and_fake_provider(
    tmp_path: Path,
) -> None:
    image = b"\x89PNG\r\n\x1a\nsynthetic"
    digest = hashlib.sha256(image).hexdigest()
    case = _case(0, "static-full-color", image_sha256=digest)
    image_path = tmp_path / case.image.path
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(image)
    manifest = _manifest((case,))
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    provider = _FakeProvider(case.expected)
    ticks = iter((0, 1_000_000))

    report = await run_model_benchmark(
        manifest_path,
        provider,
        clock_ns=lambda: next(ticks),
    )

    assert provider.calls == 1
    assert report["cases"][0]["schema_success"] is True
    assert report["passed"] is False  # a one-item development fixture is not releasable


@pytest.mark.asyncio
async def test_live_runner_rejects_tampered_input_before_provider_call(tmp_path: Path) -> None:
    image = b"\x89PNG\r\n\x1a\nchanged"
    case = _case(0, "static-full-color", image_sha256="e" * 64)
    image_path = tmp_path / case.image.path
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(image)
    manifest = _manifest((case,))
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    provider = _FakeProvider(case.expected)

    with pytest.raises(BenchmarkError, match="hash mismatch"):
        await run_model_benchmark(manifest_path, provider)
    assert provider.calls == 0


@pytest.mark.asyncio
async def test_live_runner_executes_three_independent_holdout_requests(tmp_path: Path) -> None:
    image = b"\x89PNG\r\n\x1a\nsynthetic"
    case = _case(15, "static-full-color", image_sha256=hashlib.sha256(image).hexdigest())
    image_path = tmp_path / case.image.path
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(image)
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(_manifest((case,)).model_dump_json(), encoding="utf-8")
    provider = _FakeProvider(case.expected)
    report = await run_model_benchmark(manifest_path, provider)
    assert provider.calls == 3
    binding = report["concept_generation_binding"]
    assert binding["concept_registry_id"] == _registry()["registry_id"]
    assert binding["model_routing_policy_id"] == "model-routing-local-v1"
    assert binding["model_routing_policy"]["trigger_order"] == []
    assert report["gates"]["concept_generation_context_bound"] is True
    assert report["holdout_run_count"] == 3
    assert report["gates"]["declared_runs_complete"] is True
    assert [item["run_index"] for item in report["cases"]] == [1, 2, 3]
    assert sum(item["requests_used"] for item in report["cases"]) == 3


@pytest.mark.asyncio
async def test_repeated_holdout_requests_share_one_budget(tmp_path: Path) -> None:
    image = b"\x89PNG\r\n\x1a\nsynthetic"
    case = _case(15, "static-full-color", image_sha256=hashlib.sha256(image).hexdigest())
    image_path = tmp_path / case.image.path
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(image)
    manifest = _manifest((case,)).model_copy(update={"max_requests": 1})
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    provider = _FakeProvider(case.expected)
    report = await run_model_benchmark(manifest_path, provider)
    assert provider.calls == 1
    assert [item["schema_success"] for item in report["cases"]] == [True, False, False]
    assert report["holdout_run_count"] == 1
    assert report["gates"]["complex_holdout_has_at_least_3_runs"] is False
    assert report["gates"]["declared_runs_complete"] is False
    assert report["gates"]["schema_success_100_percent"] is False
    assert report["passed"] is False


def test_unbound_adjudication_and_security_failure_are_fail_closed() -> None:
    case = _case(0, "static-full-color")
    manifest = _manifest((case,))
    observation = _observation(case).model_copy(
        update={"response_sha256": description_item_sha256(case.expected), "safety_pass": False}
    )
    report = evaluate_model_benchmark(
        manifest,
        (observation,),
        manifest_sha256="f" * 64,
    )
    assert report["gates"]["security_pass"] is False
    assert report["passed"] is False


def test_security_scan_rejects_invented_local_path_even_inside_literal_field() -> None:
    case = _case(0, "static-full-color")
    payload = case.expected.model_dump(mode="json")
    payload["facets"]["text_content"]["items"][0]["value"] = "C:\\Users\\example\\secret.txt"
    malicious = DescriptionItem.model_validate(payload)

    assert _safety_pass(malicious, case, runtime_secrets=()) is False
    assert _safety_pass(case.expected, case, runtime_secrets=("OK",)) is False


def test_provider_identity_mismatch_cannot_qualify() -> None:
    case = _case(0, "static-full-color")
    report = evaluate_model_benchmark(
        _manifest((case,)),
        (_observation(case).model_copy(update={"provider": "other"}),),
        manifest_sha256="a" * 64,
    )
    assert report["gates"]["provider_model_identity_exact"] is False


def test_model_command_requires_explicit_secret_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _case(0, "static-full-color")
    manifest = _manifest((case,), provider="gemini", model="gemini-test", revision=None)
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        benchmark_commands,
        "load_credentials",
        lambda: type("Credentials", (), {"gemini_api_key": None})(),
    )

    with pytest.raises(CommandError) as captured:
        benchmark_commands.benchmark_model_command(
            provider_name="gemini",
            model_id="gemini-test",
            benchmark_manifest=manifest_path,
        )
    assert captured.value.error.code == "CREDENTIAL_MISSING"


def test_repeat_hallucination_rate_counts_observations_without_dilution() -> None:
    case = _case(15, "static-full-color")
    assert case.adjudication is not None
    adjudication = case.adjudication.model_copy(
        update={"hallucinated_observable_fact": True, "full_pass": False}
    )
    case = case.model_copy(update={"adjudication": adjudication})
    report = evaluate_model_benchmark(
        _manifest((case,)),
        tuple(_observation(case).model_copy(update={"run_index": index}) for index in (1, 2, 3)),
        manifest_sha256="d" * 64,
    )
    assert report["metrics"]["hallucination_count"] == 3
    assert report["metrics"]["hallucination_bp"] == 10_000
    assert report["gates"]["hallucinations_at_most_2_percent"] is False


def test_additional_human_adjudication_binds_varied_repeat_response() -> None:
    case = _case(15, "static-full-color")
    assert case.adjudication is not None
    altered = case.expected.model_copy(update={"semantic_tags": ("face", "happy", "smile")})
    altered_hash = description_item_sha256(altered)
    case = case.model_copy(
        update={
            "additional_adjudications": (
                case.adjudication.model_copy(update={"response_sha256": altered_hash}),
            )
        }
    )
    report = evaluate_model_benchmark(
        _manifest((case,)),
        (
            _observation(case),
            _observation(case).model_copy(update={"run_index": 2}),
            _observation(case).model_copy(
                update={"run_index": 3, "response": altered, "response_sha256": altered_hash}
            ),
        ),
        manifest_sha256="d" * 64,
    )
    assert report["gates"]["every_response_human_bound"] is True
    assert report["metrics"]["human_bound_count"] == 3
    assert report["holdout_critical_error_count"] == 0


def test_missing_generation_documents_cannot_pass_concept_gate() -> None:
    case = _case(0, "static-full-color")
    manifest = _manifest((case,)).model_copy(
        update={"concept_registry": None, "concept_candidate_profile": None}
    )
    report = evaluate_model_benchmark(manifest, (_observation(case),), manifest_sha256="d" * 64)
    assert report["gates"]["concept_generation_context_bound"] is False
    assert report["concept_generation_binding"] is None


def test_macro_f1_weights_labels_equally_instead_of_hiding_rare_label_errors() -> None:
    pairs = [({"common"}, {"common"}, 9), ({"rare"}, set(), 1)]
    assert _macro_f1_bp(pairs) == 5_000
    assert _micro_f1_bp(pairs) == 9_474
    assert _macro_f1_bp([(set(), set(), 1)]) == 10_000
    assert _macro_f1_bp([]) == 0
