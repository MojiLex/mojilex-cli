from __future__ import annotations

import asyncio

import httpx
import pytest

from mojilex_cli.config import TelegramConfig
from mojilex_cli.sources import telegram
from mojilex_cli.sources.base import SourceError, SourceNetworkError

TOKEN = "12345:abcdefghijklmnopqrstuvwxyz"


class InterruptedStream(httpx.AsyncByteStream):
    def __init__(self, error: type[Exception]) -> None:
        self.error = error
        self.closed = False

    async def __aiter__(self):
        yield b"discard-me" * (128 * 1024)
        raise self.error(f"Server disconnected with {TOKEN}")

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("error", [httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout])
@pytest.mark.parametrize("recover", [False, True])
async def test_media_retries_disconnected_body_without_exposing_partial_bytes(
    monkeypatch, error, recover
):
    streams = []
    attempts = 0
    sleeps = []
    buffers = []
    real_spool = telegram.SpooledTemporaryFile

    def spool(*args, **kwargs):
        buffer = real_spool(*args, **kwargs)
        buffers.append(buffer)
        return buffer

    monkeypatch.setattr(telegram, "SpooledTemporaryFile", spool)

    def handler(request):
        nonlocal attempts
        attempts += 1
        if recover and attempts == 3:
            return httpx.Response(200, content=b"complete-file", request=request)
        stream = InterruptedStream(error)
        streams.append(stream)
        return httpx.Response(200, stream=stream, request=request)

    async def sleep(delay):
        sleeps.append(delay)

    received = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = telegram.TelegramBotAPI(
            TOKEN, client=client, max_attempts=3, sleep=sleep, jitter=lambda _: 0
        )

        async def download():
            async for chunk in adapter._download("stickers/file.webp"):
                received.append(chunk)

        if recover:
            await download()
            assert b"".join(received) == b"complete-file"
        else:
            with pytest.raises(SourceNetworkError) as caught:
                await download()
            assert "3 attempts" in str(caught.value)
            assert TOKEN not in str(caught.value)
            assert not received

    assert attempts == 3
    assert sleeps == [1, 2]
    assert len(buffers) == 3 and all(buffer.closed for buffer in buffers)
    assert all(stream.closed for stream in streams)


@pytest.mark.parametrize("recover", [False, True])
async def test_api_remote_disconnect_uses_six_default_attempts(recover):
    attempts = 0
    sleeps = []

    def handler(request):
        nonlocal attempts
        attempts += 1
        if recover and attempts == 6:
            return httpx.Response(200, json={"ok": True, "result": {}}, request=request)
        raise httpx.RemoteProtocolError(f"Server disconnected with {TOKEN}", request=request)

    async def sleep(delay):
        sleeps.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = telegram.TelegramBotAPI(TOKEN, client=client, sleep=sleep, jitter=lambda _: 0)
        if recover:
            await adapter.validate_credentials()
        else:
            with pytest.raises(SourceNetworkError) as caught:
                await adapter.validate_credentials()
            assert "6 attempts" in str(caught.value)
            assert TOKEN not in str(caught.value)
    assert TelegramConfig().max_attempts == attempts == 6
    assert TelegramConfig(max_attempts=2).max_attempts == 2
    assert sleeps == [1, 2, 4, 8, 16]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
@pytest.mark.parametrize("media", [False, True])
async def test_permanent_http_failure_is_not_retried(status, media):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(status, json={"ok": False, "error_code": status}, request=request)

    async def no_sleep(delay):
        pytest.fail("permanent HTTP failure must not schedule a retry")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = telegram.TelegramBotAPI(TOKEN, client=client, sleep=no_sleep)
        with pytest.raises(SourceError):
            if media:
                _ = [chunk async for chunk in adapter._download("stickers/file.webp")]
            else:
                await adapter.validate_credentials()
    assert attempts == 1


async def test_download_cancellation_is_not_retried():
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        raise asyncio.CancelledError

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = telegram.TelegramBotAPI(TOKEN, client=client)
        with pytest.raises(asyncio.CancelledError):
            _ = [chunk async for chunk in adapter._download("stickers/file.webp")]
    assert attempts == 1


async def test_streamed_size_limit_stops_without_retry_or_partial_output():
    class OversizedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * (64 * 1024)
            yield b"y"

    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(200, stream=OversizedStream(), request=request)

    received = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = telegram.TelegramBotAPI(TOKEN, client=client, max_download_bytes=64 * 1024)
        with pytest.raises(telegram.TelegramMediaLimitError):
            async for chunk in adapter._download("stickers/file.webp"):
                received.append(chunk)
    assert attempts == 1
    assert not received
