"""Operation-scoped contracts preserve exact provenance and caller isolation."""

import asyncio
from collections import Counter

import pytest

from mojilex_cli.ai import prompts
from mojilex_cli.domain.hashes import jcs_sha256


@pytest.mark.parametrize("version", ["1.1.0", "1.2.0", "1.2.1"])
def test_cached_contracts_match_immutable_version_bytes_and_hashes(version):
    module = prompts._VERSIONS[version]
    functions = (
        "prompt_templates",
        "prompt_manifest",
        "prompt_template_bytes",
        "prompt_sha256",
        "prompt_manifest_sha256",
        "gemini_request_parameters",
        "gemini_request_parameters_sha256",
    )
    expected = {name: getattr(module, name)() for name in functions}
    with prompts.prompt_contract_scope(), prompts.use_prompt_version(version):
        for _ in range(2):
            for name in functions:
                assert getattr(prompts, name)() == expected[name]
        parameters = prompts.gemini_request_parameters()
        parameters["response_format"]["schema"].clear()
        parameters["generation_config"]["max_output_tokens"] = 1
        prompts.prompt_templates()["user"].clear()
        prompts.prompt_manifest().clear()
        assert prompts.gemini_request_parameters() == expected["gemini_request_parameters"]
        assert prompts.prompt_templates() == expected["prompt_templates"]
        assert prompts.prompt_manifest() == expected["prompt_manifest"]
        assert (
            prompts.gemini_request_parameters_sha256()
            == expected["gemini_request_parameters_sha256"]
        )


def test_scope_builds_parameters_once_and_outside_scope_keeps_original_behavior(monkeypatch):
    module = prompts.v1_2_1
    original = module.gemini_request_parameters
    calls = 0

    def parameters():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(module, "gemini_request_parameters", parameters)
    with prompts.use_prompt_version("1.2.1"):
        prompts.gemini_request_parameters()
        prompts.gemini_request_parameters()
        assert calls == 2
        with prompts.prompt_contract_scope():
            for _ in range(20):
                prompts.gemini_request_parameters_sha256()
                assert (
                    jcs_sha256(prompts.gemini_request_parameters())
                    == prompts.gemini_request_parameters_sha256()
                )
            with prompts.prompt_contract_scope():
                prompts.gemini_request_parameters_sha256()
            assert calls == 3
        with prompts.prompt_contract_scope():
            prompts.gemini_request_parameters_sha256()
        assert calls == 4
        prompts.gemini_request_parameters_sha256()
        assert calls == 5


def test_versions_and_replaced_functions_never_share_cached_contract(monkeypatch):
    calls = Counter()
    for version, module in prompts._VERSIONS.items():
        original = module.gemini_request_parameters

        def parameters(version=version, original=original):
            calls[version] += 1
            return original()

        monkeypatch.setattr(module, "gemini_request_parameters", parameters)
    with prompts.prompt_contract_scope():
        for _ in range(2):
            for version in prompts._VERSIONS:
                with prompts.use_prompt_version(version):
                    assert prompts.prompt_sha256() == prompts._VERSIONS[version].prompt_sha256()
                    assert prompts.gemini_request_parameters_sha256() == jcs_sha256(
                        prompts.gemini_request_parameters()
                    )
        assert calls == {version: 1 for version in prompts._VERSIONS}
        with prompts.use_prompt_version("1.2.1"):
            old_hash = prompts.gemini_request_parameters_sha256()
            monkeypatch.setattr(
                prompts.v1_2_1, "gemini_request_parameters", lambda: {"changed": True}
            )
            assert prompts.gemini_request_parameters() == {"changed": True}
            assert (
                prompts.gemini_request_parameters_sha256()
                == jcs_sha256({"changed": True})
                != old_hash
            )


@pytest.mark.asyncio
async def test_child_threads_share_operation_cache_without_duplicate_compilation(monkeypatch):
    original = prompts.v1_2_1.gemini_request_parameters
    calls = 0

    def parameters():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(prompts.v1_2_1, "gemini_request_parameters", parameters)
    with prompts.use_prompt_version("1.2.1"), prompts.prompt_contract_scope():
        values = await asyncio.gather(
            *(asyncio.to_thread(prompts.gemini_request_parameters_sha256) for _ in range(8))
        )
    assert calls == 1
    assert len(set(values)) == 1
