from __future__ import annotations

import asyncio

import pytest

from mojilex_cli.ai import AIPaymentRequiredError
from mojilex_cli.commands import runtime
from mojilex_cli.i18n import text, use_ui_language
from mojilex_cli.pipeline import runner
from mojilex_cli.runs.pack_scope import source_state
from test_add_run_staging import saved_add  # noqa: F401
from test_incremental_pack_readiness import _assert_durable_pack, _real_packs
from test_pack_describe_pipeline import pipeline  # noqa: F401


async def test_payment_failure_stops_batch_after_preserving_completed_pack(request, monkeypatch):
    state = request.getfixturevalue("pipeline")
    _real_packs(state, monkeypatch)
    alpha, beta = state.sources
    ready = asyncio.Event()
    original_describe = runner._descriptions_for_collection

    def stage(source, phase):
        if source == alpha.canonical_url and phase == "ready":
            ready.set()

    async def describe(snapshot, source, processed, **kwargs):
        if source.native_id == beta.native_id:
            await ready.wait()
            raise AIPaymentRequiredError("synthetic provider payment failure")
        return await original_describe(snapshot, source, processed, **kwargs)

    monkeypatch.setattr(runner, "report_pack_stage", stage)
    monkeypatch.setattr(runner, "_descriptions_for_collection", describe)
    with pytest.raises(AIPaymentRequiredError):
        await asyncio.wait_for(state.run(), 30)
    checkpoint = _assert_durable_pack(state, alpha)
    assert checkpoint.status == "failed"
    assert source_state(checkpoint, beta.canonical_url)["status"] == "failed"
    assert checkpoint.issues[-1].code == "AI_PAYMENT_REQUIRED"


def test_payment_error_has_stable_exit_code_and_translated_recovery_hint():
    message = "Gemini returned HTTP 402 Payment Required; check provider billing before resuming."
    error = runtime.structured_exception(AIPaymentRequiredError(message))
    assert error.code == "AI_PAYMENT_REQUIRED"
    assert int(error.exit_code) == 8
    assert not error.retryable
    assert error.hint is not None
    with use_ui_language("ru"):
        assert "требуется оплата" in text(error.message)
        assert "продолжите существующий запуск" in text(error.hint)
    with use_ui_language("en"):
        assert text(error.message) == message
