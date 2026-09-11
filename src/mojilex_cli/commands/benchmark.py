"""Public benchmark command adapters."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mojilex_cli.ai import default_registry
from mojilex_cli.benchmark import (
    BenchmarkError,
    load_model_benchmark_manifest,
    run_dedupe_benchmark,
    run_model_benchmark,
    validate_model_benchmark_runtime,
)
from mojilex_cli.config import load_credentials
from mojilex_cli.output.models import RunStatus, StructuredError

from .runtime import CommandError, CommandResult


def benchmark_dedupe_command(manifest: Path) -> CommandResult:
    report = run_dedupe_benchmark(manifest)
    return _qualification_result(report)


def benchmark_model_command(
    *,
    provider_name: str,
    model_id: str,
    benchmark_manifest: Path,
) -> CommandResult:
    manifest, _, _ = load_model_benchmark_manifest(benchmark_manifest)
    if provider_name != manifest.target_provider or model_id != manifest.target_model:
        raise CommandError(
            "CONFIG_INVALID",
            "Provider and model must exactly match the signed benchmark manifest.",
            hint="Pass the manifest target through --provider and --model without aliases.",
        )
    validate_model_benchmark_runtime(manifest, provider_name)
    credentials = load_credentials()
    if provider_name != "gemini":
        raise CommandError(
            "CONFIG_INVALID",
            "This CLI has no credential adapter for the requested benchmark provider.",
            hint="Use an explicitly registered provider with a dedicated secret source.",
        )
    api_key = credentials.gemini_api_key
    if not api_key:
        raise CommandError(
            "CREDENTIAL_MISSING",
            "GEMINI_API_KEY is required for this explicit live model benchmark.",
            hint="Set it only in the current process environment, then retry.",
        )
    try:
        provider = default_registry().create(provider_name, model=model_id, api_key=api_key)
    except ValueError as exc:
        raise CommandError(
            "CONFIG_INVALID",
            "The requested benchmark provider is not registered.",
            hint="Select one of the providers included in this CLI build.",
        ) from exc

    async def execute_live() -> dict[str, object]:
        await provider.validate_credentials()
        return await run_model_benchmark(
            benchmark_manifest,
            provider,
            runtime_secrets=(api_key,),
        )

    report = asyncio.run(execute_live())
    return _qualification_result(report)


def _qualification_result(report: dict[str, object]) -> CommandResult:
    if report.get("passed") is True:
        return CommandResult(result=report)
    return CommandResult(
        result=report,
        status=RunStatus.FAILED,
        errors=[
            StructuredError(
                code="VALIDATION_FAILED",
                message="Benchmark qualification gates did not all pass.",
                retryable=False,
                hint="Inspect the deterministic report gates and correct the benchmark inputs.",
            )
        ],
    )


__all__ = ["BenchmarkError", "benchmark_dedupe_command", "benchmark_model_command"]
