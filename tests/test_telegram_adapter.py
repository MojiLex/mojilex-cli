import json

import httpx
import pytest

from mojilex_cli.sources import (
    SourceAuthError,
    SourceNetworkError,
    SourceNotFoundError,
    SourceRateLimitError,
    TelegramBotAPI,
    TelegramMediaLimitError,
    UnsupportedSourceError,
    parse_telegram_source,
)
from mojilex_cli.sources.telegram import TelegramProtocolError


@pytest.mark.parametrize(
    ("source", "allow_bare"),
    [
        ("https://t.me/addemoji/Pack_Name", False),
        ("https://telegram.me/addemoji/Pack_Name", False),
        ("tg://addemoji?set=Pack_Name", False),
        ("Pack_Name", True),
    ],
)
def test_strict_parser_accepts_only_canonicalizable_forms(source: str, allow_bare: bool) -> None:
    reference = parse_telegram_source(source, allow_bare=allow_bare)
    assert reference.native_id == "Pack_Name"
    assert reference.canonical_url == "https://t.me/addemoji/Pack_Name"


@pytest.mark.parametrize(
    "source",
    [
        "http://t.me/addemoji/Pack",
        "https://evil.example/addemoji/Pack",
        "https://user@t.me/addemoji/Pack",
        "https://t.me:443/addemoji/Pack",
        "https://t.me/addemoji/Pack/extra",
        "https://t.me/addemoji/Pack?x=1",
        "https://t.me/addemoji/Pack#fragment",
        "https://t.me/addstickers/Pack",
        "https://t.me/addemoji/%252e%252e",
        "tg://addemoji?set=Pack&x=1",
    ],
)
def test_strict_parser_rejects_ambiguous_or_dangerous_forms(source: str) -> None:
    with pytest.raises(UnsupportedSourceError):
        parse_telegram_source(source)


def _sticker_set() -> dict[str, object]:
    return {
        "name": "Pack",
        "title": "Synthetic pack",
        "sticker_type": "custom_emoji",
        "stickers": [
            {
                "file_id": "transient-file-id",
                "file_unique_id": "stable-file-id",
                "custom_emoji_id": "1234567890123456789",
                "width": 100,
                "height": 100,
                "is_animated": False,
                "is_video": False,
                "emoji": "🙂",
                "file_size": 42,
            }
        ],
    }


def test_multiline_telegram_title_is_normalized_without_accepting_controls() -> None:
    payload = _sticker_set()
    payload["title"] = "Lake\nmermaid\r\npack\tname"
    assert TelegramBotAPI._parse_sticker_set(payload).title == "Lake mermaid pack name"
    payload["title"] = "  Lake \n  cafe\u0301\u00a0 pack  "
    assert TelegramBotAPI._parse_sticker_set(payload).title == "Lake caf\u00e9 pack"
    payload["title"] = "Lake\x01mermaid"
    with pytest.raises(TelegramProtocolError):
        TelegramBotAPI._parse_sticker_set(payload)


@pytest.mark.asyncio
async def test_adapter_normalizes_collection_without_persisting_file_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "result": _sticker_set()}, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = TelegramBotAPI("12345:abcdefghijklmnopqrstuvwxyz", client=client)
    collection = await adapter.fetch_collection(adapter.canonicalize("Pack"))
    await client.aclose()

    assert collection.items[0].native_id == "1234567890123456789"
    assert collection.items[0].media_format == "webp"
    assert "file_id" not in collection.safe_dump()["items"][0]
    assert len(collection.extension["set_fingerprint_sha256"]) == 64


@pytest.mark.asyncio
async def test_direct_emoji_lookup_is_batched_at_bot_api_limit() -> None:
    requested_batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requested = payload["custom_emoji_ids"]
        requested_batches.append(requested)
        stickers = []
        for native_id in reversed(requested):
            sticker = dict(_sticker_set()["stickers"][0])
            sticker.update(
                {
                    "custom_emoji_id": native_id,
                    "file_id": f"file-{native_id}",
                    "file_unique_id": f"unique-{native_id}",
                }
            )
            stickers.append(sticker)
        return httpx.Response(200, json={"ok": True, "result": stickers}, request=request)

    native_ids = tuple(str(10**18 + index) for index in range(401))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = TelegramBotAPI("12345:abcdefghijklmnopqrstuvwxyz", client=client)
    found = await adapter.fetch_emojis(native_ids)
    await client.aclose()

    assert [len(batch) for batch in requested_batches] == [200, 200, 1]
    assert requested_batches == [
        list(native_ids[:200]),
        list(native_ids[200:400]),
        list(native_ids[400:]),
    ]
    assert set(found) == set(native_ids)
    assert found[native_ids[-1]].position == len(native_ids) - 1
    assert "file_id" not in found[native_ids[-1]].safe_dump()


@pytest.mark.asyncio
async def test_bounded_retry_and_secret_redaction() -> None:
    token = "12345:abcdefghijklmnopqrstuvwxyz"
    attempts = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError(f"cannot connect with {token}", request=request)

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = TelegramBotAPI(
        token, client=client, max_attempts=4, sleep=fake_sleep, jitter=lambda _: 0
    )
    with pytest.raises(SourceNetworkError) as caught:
        await adapter.validate_credentials()
    await client.aclose()

    assert attempts == 4
    assert sleeps == [1, 2, 4]
    assert token not in str(caught.value)


@pytest.mark.asyncio
async def test_retry_after_over_120_requests_checkpoint_instead_of_sleep() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={"ok": False, "error_code": 429, "parameters": {"retry_after": 121}},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = TelegramBotAPI("12345:abcdefghijklmnopqrstuvwxyz", client=client)
    with pytest.raises(SourceRateLimitError) as caught:
        await adapter.validate_credentials()
    await client.aclose()
    assert caught.value.retry_after == 121


@pytest.mark.asyncio
async def test_download_is_streamed_and_enforces_limit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/getFile"):
            return httpx.Response(
                200,
                json={"ok": True, "result": {"file_path": "stickers/file.webp"}},
                request=request,
            )
        return httpx.Response(200, content=b"1234", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = TelegramBotAPI(
        "12345:abcdefghijklmnopqrstuvwxyz", client=client, max_download_bytes=3
    )
    item = TelegramBotAPI._parse_sticker(_sticker_set()["stickers"][0], 0)
    with pytest.raises(TelegramMediaLimitError):
        _ = b"".join([chunk async for chunk in adapter.fetch_media(item)])
    await client.aclose()


@pytest.mark.parametrize("http_status", [200, 400])
async def test_missing_set_requires_valid_auth_and_bounded_confirmation(http_status: int) -> None:
    methods: list[str] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        methods.append(method)
        if method == "getMe":
            return httpx.Response(200, json={"ok": True, "result": {}}, request=request)
        return httpx.Response(
            http_status,
            json={"ok": False, "error_code": 400, "description": "Bad Request: STICKERSET_INVALID"},
            request=request,
        )

    async def sleep(delay: float) -> None:
        sleeps.append(delay)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = TelegramBotAPI(
            "12345:abcdefghijklmnopqrstuvwxyz",
            client=client,
            max_attempts=4,
            sleep=sleep,
            jitter=lambda _: 0,
        )
        with pytest.raises(SourceNotFoundError):
            await adapter.fetch_collection(adapter.canonicalize("Pack"))

    assert methods == ["getStickerSet", "getMe", "getStickerSet", "getStickerSet", "getStickerSet"]
    assert sleeps == [1, 2, 4]


async def test_transient_missing_set_recovers_without_revalidating_known_credentials() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        methods.append(method)
        if method == "getMe":
            return httpx.Response(200, json={"ok": True, "result": {}}, request=request)
        if methods.count("getStickerSet") == 1:
            return httpx.Response(
                400,
                json={
                    "ok": False,
                    "error_code": 400,
                    "description": "Bad Request: STICKERSET_INVALID",
                },
                request=request,
            )
        return httpx.Response(200, json={"ok": True, "result": _sticker_set()}, request=request)

    async def sleep(_: float) -> None:
        pass

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = TelegramBotAPI("12345:abcdefghijklmnopqrstuvwxyz", client=client, sleep=sleep)
        await adapter.validate_credentials()
        assert await adapter.check_collection_availability(adapter.canonicalize("Pack")) is True

    assert methods == ["getMe", "getStickerSet", "getStickerSet"]


@pytest.mark.parametrize(
    "description", ["Bad Request: invalid request", "file not found", "STICKERSET_INVALID suffix"]
)
async def test_arbitrary_bad_request_is_not_set_unavailability(description: str) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.url.path.rsplit("/", 1)[-1])
        return httpx.Response(
            400, json={"ok": False, "error_code": 400, "description": description}, request=request
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = TelegramBotAPI("12345:abcdefghijklmnopqrstuvwxyz", client=client)
        with pytest.raises(TelegramProtocolError):
            await adapter.check_collection_availability(adapter.canonicalize("Pack"))

    assert methods == ["getStickerSet"]


async def test_missing_set_with_invalid_credentials_never_reports_unavailable() -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        method = request.url.path.rsplit("/", 1)[-1]
        methods.append(method)
        if method == "getMe":
            return httpx.Response(401, json={"ok": False, "error_code": 401}, request=request)
        return httpx.Response(
            400,
            json={"ok": False, "error_code": 400, "description": "Bad Request: STICKERSET_INVALID"},
            request=request,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = TelegramBotAPI("12345:abcdefghijklmnopqrstuvwxyz", client=client)
        with pytest.raises(SourceAuthError):
            await adapter.check_collection_availability(adapter.canonicalize("Pack"))

    assert methods == ["getStickerSet", "getMe"]
