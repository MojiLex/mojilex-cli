from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from PIL import Image

from mojilex_cli.ai import DescriptionBatch, DescriptionResult
from mojilex_cli.ai.base import BudgetExceededError
from mojilex_cli.cache import CacheStore
from mojilex_cli.cache.store import CachedAIResult
from mojilex_cli.commands.runtime import CommandResult, execute
from mojilex_cli.config import AIConfig, MojiLexConfig
from mojilex_cli.config.models import Credentials, RepositoryConfig
from mojilex_cli.dataset import DatasetSnapshot
from mojilex_cli.github import RepositoryRef
from mojilex_cli.media import TemporaryMediaRun
from mojilex_cli.pipeline import runner
from mojilex_cli.pipeline.transform import _emoji
from mojilex_cli.policy import ModelQualificationRegistry, RoutingReasonRegistry
from mojilex_cli.runs import RunStore
from test_ai_concepts import PROFILE, _registry
from test_dataset_helpers import MEDIA_HASH, write_fixture
from test_pipeline_identity_media import _processed, _source
from test_pipeline_resume_cache import _description, _prepare_single_request


@pytest.mark.parametrize(
    ("failure", "status", "exit_code"),
    [
        (KeyboardInterrupt(), "interrupted", 130),
        (BudgetExceededError("limit reached"), "budget_exceeded", 9),
    ],
)
def test_pipeline_failure_reports_the_durable_checkpoint_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: BaseException,
    status: str,
    exit_code: int,
) -> None:
    repository = tmp_path / "dataset"
    repository.mkdir()
    config = MojiLexConfig(
        repository=RepositoryConfig(target=str(repository), publish="local"),
        cache_dir=tmp_path / "cache",
        runs_dir=tmp_path / "runs",
    )

    @contextmanager
    def workspace(*_args, **_kwargs):
        yield runner.RepositoryWorkspace(repository, RepositoryRef.parse("MojiLex/mojilex"), False)

    class Adapter:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def validate_credentials(self):
            raise failure

    monkeypatch.setattr(runner, "_resolved_config", lambda _options: config)
    monkeypatch.setattr(runner, "repository_workspace", workspace)
    monkeypatch.setattr(
        runner,
        "load_validated_dataset",
        lambda *_a, **_kw: (DatasetSnapshot(repository, {}), SimpleNamespace(valid=True)),
    )
    monkeypatch.setattr(
        runner, "GitRunner", lambda *_a, **_kw: SimpleNamespace(current_sha=lambda: "a" * 40)
    )
    monkeypatch.setattr(
        runner, "load_credentials", lambda: Credentials(telegram_bot_token="present")
    )
    monkeypatch.setattr(runner, "TelegramBotAPI", Adapter)

    with pytest.raises(typer.Exit) as stopped:
        execute(
            "add",
            lambda: runner.run_add(("https://t.me/addemoji/Pack",), runner.PipelineOptions()),
            json_output=True,
        )

    payload = json.loads(capsys.readouterr().out)
    assert stopped.value.exit_code == exit_code
    assert payload["status"] == status
    checkpoint = RunStore(config.runs_dir).load(payload["run_id"])
    assert checkpoint.status == status
    assert payload["run_id"].startswith("mlxrun_")

    def unrelated_failure():
        raise ValueError("another command")

    with pytest.raises(typer.Exit):
        execute("config show", unrelated_failure, json_output=True)
    assert json.loads(capsys.readouterr().out)["run_id"] != checkpoint.run_id


async def test_check_media_preview_reads_exact_cache_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    source = _source(snapshot)
    item = source.items[0]
    media = _processed(snapshot, MEDIA_HASH)
    values = {item.native_id: media}
    config = MojiLexConfig(ai=AIConfig(model="configured-model"))
    cache_path = tmp_path / "cache.sqlite3"
    monkeypatch.setattr(runner, "_prepare_ai_request", _prepare_single_request)
    with TemporaryMediaRun() as temporary:
        prepared = await _prepare_single_request(
            (item,), values, model=config.ai.model, temporary=temporary
        )
        kwargs = dict(
            model=config.ai.model,
            taxonomy_version=str(snapshot.manifest["taxonomy_version"]),
            request_identity=prepared.identity,
        )
        lookup_key = runner._cache_key(
            item, media, runner._vision_context(item, media), config, model_revision=None, **kwargs
        )
        actual_key = runner._cache_key(
            item,
            media,
            runner._vision_context(item, media),
            config,
            model_revision=None,
            **kwargs,
        )
        with CacheStore(cache_path) as cache:
            cache.put_ai(
                actual_key,
                DescriptionResult(
                    batch=DescriptionBatch(items=(_description(),)),
                    provider="gemini",
                    model=config.ai.model,
                    model_revision=None,
                ),
                generated_at="2026-09-10T18:00:00Z",
                aliases=(lookup_key,),
            )
        before = hashlib.sha256(cache_path.read_bytes()).hexdigest()
        before_names = await asyncio.to_thread(lambda: {path.name for path in tmp_path.iterdir()})
        with CacheStore(cache_path, read_only=True) as cache:
            plan = await runner._preview_ai_plan(
                snapshot,
                source,
                config=config,
                options=runner.PipelineOptions(
                    dry_run=True, check_media=True, redescribe="all", overwrite_reviewed=True
                ),
                cache=cache,
                processed=values,
                temporary=temporary,
            )
    assert plan["ai_cache_hits_estimated"] == 1
    assert plan["ai_cache_hits_unknown"] == 0
    assert plan["ai_batches_planned"] == 0
    assert plan["ai_requests_estimated_upper_bound"] == 0
    assert hashlib.sha256(cache_path.read_bytes()).hexdigest() == before
    assert (
        await asyncio.to_thread(lambda: {path.name for path in tmp_path.iterdir()}) == before_names
    )


async def test_metadata_preview_marks_unverifiable_cache_and_counts_retries(tmp_path: Path) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    source = _source(snapshot)
    config = MojiLexConfig(ai=AIConfig(model="configured-model"))
    with CacheStore(tmp_path / "cache.sqlite3") as cache:
        plan = await runner._preview_ai_plan(
            snapshot,
            source,
            config=config,
            options=runner.PipelineOptions(dry_run=True, redescribe="all"),
            cache=cache,
        )
    assert plan["ai_cache_hits_estimated"] == 0
    assert plan["ai_cache_hits_unknown"] == 1
    assert plan["ai_items_planned"] == plan["ai_batches_planned"] == 1
    assert plan["ai_requests_estimated_upper_bound"] == 4


def test_provider_notice_precedes_client_and_preserves_json_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    events: list[str] = []

    class Provider:
        async def validate_credentials(self):
            events.append("validated")

    def create(*_args, **_kwargs):
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "Derived contact-sheet PNG" in captured.err
        assert "model=chosen-model" in captured.err
        assert "https://ai.google.dev/gemini-api/terms" in captured.err
        events.append("created")
        return Provider()

    monkeypatch.setattr(runner, "default_registry", lambda: SimpleNamespace(create=create))

    def action():
        asyncio.run(
            runner._provider_for_model(
                runner._AIState(),
                config=MojiLexConfig(ai=AIConfig(model="chosen-model")),
                api_key="not-a-real-credential",
                model="chosen-model",
            )
        )
        return CommandResult()

    execute("add", action, json_output=True)
    assert events == ["created", "validated"]
    assert json.loads(capsys.readouterr().out)["ok"] is True


async def test_stage_b_inputs_bind_request_cache_provenance_and_concept_output(
    tmp_path: Path,
) -> None:
    snapshot = write_fixture(tmp_path / "dataset")
    source = _source(snapshot)
    item = source.items[0]
    config = MojiLexConfig(ai=AIConfig(model="configured-model"))
    registry = _registry()
    registry["registry_id"] = "concepts-v1-fixture"
    registry_path = snapshot.root / "taxonomy" / "v1" / "concepts.json"
    profile_path = snapshot.root / "analysis-profiles" / "concept-candidates-v1.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    profile_path.parent.mkdir(exist_ok=True)
    profile_path.write_bytes(runner.rfc8785.dumps(PROFILE))
    inputs = runner._load_generation_inputs(snapshot, config)
    assert inputs is not None
    assert len(inputs.fields) == 7
    token = runner._GENERATION_INPUTS.set(inputs)
    try:
        with TemporaryMediaRun() as temporary:
            assert temporary.path is not None
            frame = temporary.path / "synthetic.png"
            Image.new("RGBA", (100, 100), (20, 30, 40, 255)).save(frame)
            media = _processed(snapshot, MEDIA_HASH).model_copy(update={"frame_paths": (frame,)})
            values = {item.native_id: media}
            prepared = await runner._prepare_ai_request(
                (item,), values, model=config.ai.model, temporary=temporary
            )
            assert prepared.request.concept_context == inputs.concepts.prompt_context
            context = runner._vision_context(item, media)
            original_key = runner._cache_key(
                item,
                media,
                context,
                config,
                model=config.ai.model,
                model_revision=None,
                taxonomy_version="1.0.0",
                request_identity=prepared.identity,
            )
            description = _description().model_copy(update={"concept_ids": ("animal.cat",)})
            accepted = DescriptionResult(
                batch=DescriptionBatch(items=(description,)),
                provider="gemini",
                model=config.ai.model,
            )
            outcome = runner._semantic_outcome(
                CachedAIResult(generated_at="2026-09-10T18:00:00Z", result=accepted),
                generation_stage="primary",
                routing_reasons=(),
                config=config,
                taxonomy_version="1.0.0",
                qualifications=ModelQualificationRegistry.load(snapshot.root),
                routing_registry=RoutingReasonRegistry.load(snapshot.root),
            )
            canonical = _emoji(
                "telegram",
                item,
                media,
                description,
                runner._bind_deterministic_analyses(values)[item.native_id],
                outcome.generation,
                manifest=snapshot.manifest,
                epoch=0,
                now="2026-09-10T18:00:00Z",
            )
            assert canonical.concept_ids == ["animal.cat"]
            assert canonical.concept_mapping_status.value == "complete"
            assert {
                name: getattr(canonical.provenance, name) for name in inputs.fields
            } == inputs.fields
            assert runner._description_from_existing(canonical).concept_ids == ("animal.cat",)
            assert canonical.provenance.qualification_id is None
            wrong = accepted.model_copy(
                update={
                    "batch": DescriptionBatch(
                        items=(description.model_copy(update={"concept_ids": ("animal.unknown",)}),)
                    )
                }
            )
            with pytest.raises(runner.AIOutputError, match="candidate set"):
                runner._validate_actual_result(wrong, "gemini", config.ai.model)

            registry["concepts"][0]["definitions"]["en"] = "A changed exact concept definition."
            registry_path.write_text(json.dumps(registry), encoding="utf-8")
            changed = runner._load_generation_inputs(snapshot, config)
            assert changed is not None
            runner._GENERATION_INPUTS.set(changed)
            changed_key = runner._cache_key(
                item,
                media,
                context,
                config,
                model=config.ai.model,
                model_revision=None,
                taxonomy_version="1.0.0",
                request_identity=prepared.identity,
            )
            assert changed_key != original_key
            assert (
                runner._ai_request_plan_sha256((item,), values, model=config.ai.model)
                != prepared.identity.plan_sha256
            )
    finally:
        runner._GENERATION_INPUTS.reset(token)
