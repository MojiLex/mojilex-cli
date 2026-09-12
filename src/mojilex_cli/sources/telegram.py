"""Strict Telegram source parsing and an asynchronous Bot API adapter."""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import parse_qsl, unquote, urlsplit

import httpx
import rfc8785

from mojilex_cli.config.secrets import redact_text

from .base import (
    SourceAdapter,
    SourceAuthError,
    SourceCapabilities,
    SourceCollection,
    SourceEmoji,
    SourceError,
    SourceMembership,
    SourceNetworkError,
    SourceNotFoundError,
    SourceRateLimitError,
    SourceReference,
    UnsupportedSourceError,
)

MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
MAX_API_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_RETRY_AFTER_SECONDS = 120.0
_SHORT_NAME = re.compile(r"[A-Za-z0-9_]{1,64}\Z")
_BOT_TOKEN = re.compile(r"[0-9]{5,20}:[A-Za-z0-9_-]{20,}\Z")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_RETRIABLE_STATUS = frozenset({408, 425, 500, 502, 503, 504})
_MAX_CUSTOM_EMOJI_LOOKUP = 200


class TelegramProtocolError(SourceError):
    code = "SOURCE_INVALID_RESPONSE"


class TelegramMediaLimitError(SourceError):
    code = "MEDIA_LIMIT_EXCEEDED"


def _decode_short_name(raw: str) -> str:
    try:
        decoded = unquote(raw, encoding="utf-8", errors="strict")
    except UnicodeError as exc:
        raise UnsupportedSourceError("Telegram source contains invalid UTF-8 encoding") from exc
    if _CONTROL.search(decoded) or not _SHORT_NAME.fullmatch(decoded):
        raise UnsupportedSourceError("invalid Telegram custom emoji set short name")
    return decoded


def parse_telegram_source(source: str, *, allow_bare: bool = False) -> SourceReference:
    """Parse only documented custom-emoji forms, decoding the name exactly once."""

    if not isinstance(source, str) or not source or source != source.strip():
        raise UnsupportedSourceError("invalid Telegram source")
    if _CONTROL.search(source):
        raise UnsupportedSourceError("Telegram source contains control characters")

    if "://" not in source:
        if not allow_bare:
            raise UnsupportedSourceError("bare short names require --platform telegram")
        short_name = _decode_short_name(source)
        return _reference(short_name)

    try:
        parsed = urlsplit(source)
        port = parsed.port
    except ValueError as exc:
        raise UnsupportedSourceError("malformed Telegram source URL") from exc
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or port is not None
    ):
        raise UnsupportedSourceError("credentials, ports, and fragments are forbidden")

    if parsed.scheme.lower() == "https":
        if parsed.hostname is None or parsed.hostname.lower() not in {"t.me", "telegram.me"}:
            raise UnsupportedSourceError("unsupported Telegram host")
        if parsed.query:
            raise UnsupportedSourceError("unexpected query parameters")
        # Splitting rather than path normalization deliberately rejects //, . and .. forms.
        parts = parsed.path.split("/")
        if len(parts) != 3 or parts[0] != "" or parts[1] != "addemoji" or not parts[2]:
            if "addstickers" in parsed.path:
                raise UnsupportedSourceError("ordinary Telegram sticker packs are not supported")
            raise UnsupportedSourceError("expected /addemoji/{short_name}")
        short_name = _decode_short_name(parts[2])
    elif parsed.scheme.lower() == "tg":
        if parsed.netloc.lower() != "addemoji" or parsed.path not in {"", "/"}:
            raise UnsupportedSourceError("expected tg://addemoji?set={short_name}")
        if parsed.fragment:
            raise UnsupportedSourceError("fragments are forbidden")
        try:
            pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True)
        except ValueError as exc:
            raise UnsupportedSourceError("malformed Telegram deep-link query") from exc
        if len(pairs) != 1 or pairs[0][0] != "set":
            raise UnsupportedSourceError("unexpected Telegram deep-link parameters")
        # parse_qsl has performed the one and only percent decoding.
        short_name = pairs[0][1]
        if _CONTROL.search(short_name) or not _SHORT_NAME.fullmatch(short_name):
            raise UnsupportedSourceError("invalid Telegram custom emoji set short name")
    else:
        raise UnsupportedSourceError("unsupported Telegram source scheme")
    return _reference(short_name)


def _reference(short_name: str) -> SourceReference:
    return SourceReference(
        platform="telegram",
        native_namespace="sticker_set.name",
        scope_id="global",
        native_id=short_name,
        canonical_url=f"https://t.me/addemoji/{short_name}",
    )


def telegram_set_fingerprint(items: tuple[SourceEmoji, ...]) -> str:
    value = [
        {"custom_emoji_id": item.native_id, "file_unique_id": item.file_unique_id}
        for item in sorted(items, key=lambda item: (item.native_id, item.file_unique_id))
    ]
    return hashlib.sha256(rfc8785.dumps(value)).hexdigest()


class TelegramBotAPI(SourceAdapter):
    """Read-only Telegram Bot API adapter with bounded retries and streaming."""

    def __init__(
        self,
        token: str,
        *,
        client: httpx.AsyncClient | None = None,
        timeout_seconds: float = 30.0,
        max_attempts: int = 4,
        max_download_bytes: int = MAX_DOWNLOAD_BYTES,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float], float] | None = None,
    ) -> None:
        if not token or not _BOT_TOKEN.fullmatch(token):
            raise SourceAuthError("Telegram credential is missing or malformed")
        if not 1 <= max_attempts <= 8:
            raise ValueError("max_attempts must be between 1 and 8")
        if not 1 <= max_download_bytes <= MAX_DOWNLOAD_BYTES:
            raise ValueError("download byte limit exceeds the 20 MiB hard limit")
        self._token = token
        self._secrets = (token,)
        self._base_url = f"https://api.telegram.org/bot{token}"
        self._file_base_url = f"https://api.telegram.org/file/bot{token}"
        if client is not None and client.follow_redirects:
            raise ValueError("Telegram client must disable redirects to protect credentials")
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds), follow_redirects=False
        )
        self._owns_client = client is None
        self._max_attempts = max_attempts
        self._max_download_bytes = max_download_bytes
        self._sleep = sleep
        self._jitter = jitter or (lambda delay: random.uniform(0.0, min(0.25, delay / 4)))
        self._collections: dict[str, SourceCollection] = {}
        self._credentials_validated = False

    async def __aenter__(self) -> TelegramBotAPI:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def recognize(self, source: str) -> bool:
        try:
            parse_telegram_source(source)
        except UnsupportedSourceError:
            return False
        return True

    def canonicalize(self, source: str) -> SourceReference:
        # Choosing this adapter is equivalent to explicit --platform telegram.
        return parse_telegram_source(source, allow_bare=True)

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            platform="telegram",
            media_formats=("webp", "tgs", "webm"),
            stable_collection_ids=True,
            stable_emoji_ids=True,
            supports_collection_availability=True,
            supports_emoji_availability=True,
            max_direct_emoji_lookup=200,
        )

    async def validate_credentials(self) -> None:
        await self._request("getMe")
        self._credentials_validated = True

    async def fetch_collection(self, native_ref: SourceReference) -> SourceCollection:
        if native_ref.platform != "telegram":
            raise UnsupportedSourceError("reference is not a Telegram source")
        payload = await self._request("getStickerSet", {"name": native_ref.native_id})
        collection = self._parse_sticker_set(payload)
        self._collections[collection.native_id] = collection
        return collection

    async def list_memberships(
        self, collection_ref: SourceReference
    ) -> tuple[SourceMembership, ...]:
        collection = self._collections.get(collection_ref.native_id)
        if collection is None:
            collection = await self.fetch_collection(collection_ref)
        return tuple(
            SourceMembership(
                collection_native_id=collection.native_id,
                emoji_native_id=item.native_id,
                position=item.position,
            )
            for item in collection.items
        )

    async def check_collection_availability(self, native_ref: SourceReference) -> bool:
        try:
            await self.fetch_collection(native_ref)
        except SourceNotFoundError:
            return False
        return True

    async def check_emoji_availability(self, native_ids: tuple[str, ...]) -> dict[str, bool]:
        found = await self.fetch_emojis(native_ids)
        return {native_id: native_id in found for native_id in native_ids}

    async def fetch_emojis(self, native_ids: tuple[str, ...]) -> dict[str, SourceEmoji]:
        """Resolve custom emoji IDs directly in bounded Bot API batches."""

        if len(native_ids) != len(set(native_ids)):
            raise ValueError("emoji IDs must be unique")
        if any(not isinstance(native_id, str) or not native_id for native_id in native_ids):
            raise ValueError("emoji IDs must be non-empty strings")
        result: dict[str, SourceEmoji] = {}
        positions = {native_id: position for position, native_id in enumerate(native_ids)}
        for offset in range(0, len(native_ids), _MAX_CUSTOM_EMOJI_LOOKUP):
            batch = native_ids[offset : offset + _MAX_CUSTOM_EMOJI_LOOKUP]
            response = await self._request(
                "getCustomEmojiStickers", {"custom_emoji_ids": list(batch)}
            )
            if not isinstance(response, list):
                raise TelegramProtocolError("Telegram returned an invalid emoji lookup result")
            for raw in response:
                item = self._parse_sticker(raw, 0)
                if item.native_id not in positions or item.native_id not in batch:
                    raise TelegramProtocolError(
                        "Telegram returned an unrequested custom emoji identifier"
                    )
                if item.native_id in result:
                    raise TelegramProtocolError(
                        "Telegram returned a duplicate custom emoji identifier"
                    )
                result[item.native_id] = item.model_copy(
                    update={"position": positions[item.native_id]}
                )
        return result

    async def fetch_media(self, emoji_ref: SourceEmoji) -> AsyncIterator[bytes]:
        file_response = await self._request("getFile", {"file_id": emoji_ref.file_id})
        if not isinstance(file_response, Mapping) or not isinstance(
            file_response.get("file_path"), str
        ):
            raise TelegramProtocolError("Telegram getFile response has no safe file path")
        file_path = self._validate_file_path(file_response["file_path"])
        async for chunk in self._download(file_path):
            yield chunk

    async def _request(self, method: str, params: Mapping[str, object] | None = None) -> Any:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]{0,63}", method):
            raise ValueError("invalid Bot API method")
        url = f"{self._base_url}/{method}"
        last_error: Exception | None = None
        for attempt in range(self._max_attempts):
            try:
                response = await self._client.post(url, json=dict(params or {}))
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = exc
                if attempt + 1 == self._max_attempts:
                    break
                await self._backoff(attempt)
                continue
            if (
                response.history
                or response.url.scheme != "https"
                or response.url.host != "api.telegram.org"
            ):
                raise SourceNetworkError("Telegram Bot API redirect was rejected")
            if len(response.content) > MAX_API_RESPONSE_BYTES:
                raise TelegramProtocolError("Telegram response exceeded the safe size limit")
            parsed = self._parse_response_json(response)
            error_code = self._error_code(response.status_code, parsed)
            if response.status_code in {401, 403} or error_code in {401, 403}:
                raise SourceAuthError("Telegram rejected the configured credential")
            if response.status_code == 429 or error_code == 429:
                retry_after = self._retry_after(response, parsed)
                if retry_after is not None and retry_after > MAX_RETRY_AFTER_SECONDS:
                    raise SourceRateLimitError(
                        "Telegram requested a retry later; checkpoint the run",
                        retry_after=retry_after,
                    )
                if attempt + 1 == self._max_attempts:
                    raise SourceRateLimitError(
                        "Telegram rate limit persisted after bounded retries",
                        retry_after=retry_after,
                    )
                await self._sleep(retry_after if retry_after is not None else self._delay(attempt))
                continue
            if response.status_code in _RETRIABLE_STATUS or error_code >= 500:
                if attempt + 1 == self._max_attempts:
                    raise SourceNetworkError("Telegram service remained temporarily unavailable")
                await self._backoff(attempt)
                continue
            if response.status_code in {200, 400} and self._is_missing_sticker_set(
                method, error_code, parsed
            ):
                # A missing set must not turn an invalid credential or one
                # transient/ambiguous response into an availability change.
                if not self._credentials_validated:
                    await self.validate_credentials()
                if attempt + 1 < self._max_attempts:
                    await self._backoff(attempt)
                    continue
                raise SourceNotFoundError("Telegram custom emoji set was not found")
            if not 200 <= response.status_code < 300:
                if response.status_code == 404:
                    raise SourceAuthError("Telegram rejected the configured credential")
                if error_code == 404:
                    raise SourceNotFoundError("Telegram custom emoji set was not found")
                raise TelegramProtocolError(f"Telegram returned HTTP status {response.status_code}")
            if not isinstance(parsed, Mapping):
                raise TelegramProtocolError("Telegram returned invalid JSON")
            if parsed.get("ok") is not True:
                raise TelegramProtocolError(
                    redact_text("Telegram rejected the read request", self._secrets)
                )
            return parsed.get("result")
        raise SourceNetworkError(
            redact_text(
                f"Telegram request failed after {self._max_attempts} attempts: "
                f"{type(last_error).__name__ if last_error else 'network error'}",
                self._secrets,
            )
        ) from None

    @staticmethod
    def _is_missing_sticker_set(method: str, error_code: int, payload: Any) -> bool:
        if (
            method != "getStickerSet"
            or error_code != 400
            or not isinstance(payload, Mapping)
            or payload.get("ok") is not False
        ):
            return False
        description = payload.get("description")
        return isinstance(description, str) and description.lower() in {
            "bad request: stickerset_invalid",
            "bad request: sticker set not found",
            "bad request: invalid sticker set",
            "stickerset_invalid",
            "sticker set not found",
            "invalid sticker set",
        }

    @staticmethod
    def _parse_response_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            if response.status_code >= 500:
                return {}
            raise TelegramProtocolError("Telegram returned malformed JSON") from exc

    @staticmethod
    def _error_code(status: int, payload: Any) -> int:
        if isinstance(payload, Mapping):
            value = payload.get("error_code")
            if isinstance(value, int):
                return value
        return status

    @staticmethod
    def _retry_after(response: httpx.Response, payload: Any) -> float | None:
        value: object | None = response.headers.get("Retry-After")
        if isinstance(payload, Mapping):
            parameters = payload.get("parameters")
            if isinstance(parameters, Mapping) and "retry_after" in parameters:
                value = parameters["retry_after"]
        try:
            delay = float(value) if isinstance(value, (str, int, float)) else None
        except ValueError:
            return None
        return max(0.0, delay) if delay is not None else None

    def _delay(self, attempt: int) -> float:
        base = min(30.0, float(2**attempt))
        return min(30.0, base + max(0.0, self._jitter(base)))

    async def _backoff(self, attempt: int) -> None:
        await self._sleep(self._delay(attempt))

    @staticmethod
    def _validate_file_path(value: str) -> str:
        if (
            not value
            or _CONTROL.search(value)
            or "\\" in value
            or "%" in value
            or "//" in value
            or "?" in value
            or "#" in value
        ):
            raise TelegramProtocolError("Telegram returned an unsafe file path")
        path = PurePosixPath(value)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise TelegramProtocolError("Telegram returned an unsafe file path")
        return path.as_posix()

    async def _download(self, file_path: str) -> AsyncIterator[bytes]:
        url = f"{self._file_base_url}/{file_path}"
        for attempt in range(self._max_attempts):
            yielded = False
            try:
                async with self._client.stream("GET", url) as response:
                    if (
                        response.history
                        or response.url.scheme != "https"
                        or response.url.host != "api.telegram.org"
                    ):
                        raise SourceNetworkError("Telegram media redirect was rejected")
                    if response.status_code in {401, 403}:
                        raise SourceAuthError("Telegram rejected the configured credential")
                    if response.status_code == 404:
                        raise SourceNotFoundError("Telegram media is no longer available")
                    if response.status_code == 429:
                        retry_after = self._retry_after(response, None)
                        if retry_after is not None and retry_after > MAX_RETRY_AFTER_SECONDS:
                            raise SourceRateLimitError(
                                "Telegram media retry requires checkpointing",
                                retry_after=retry_after,
                            )
                        if attempt + 1 == self._max_attempts:
                            raise SourceRateLimitError(
                                "Telegram media rate limit persisted after bounded retries",
                                retry_after=retry_after,
                            )
                        await self._sleep(
                            retry_after if retry_after is not None else self._delay(attempt)
                        )
                        continue
                    if response.status_code in _RETRIABLE_STATUS:
                        if attempt + 1 == self._max_attempts:
                            raise SourceNetworkError(
                                "Telegram media service remained temporarily unavailable"
                            )
                        await self._backoff(attempt)
                        continue
                    if not 200 <= response.status_code < 300:
                        raise SourceNetworkError(
                            f"Telegram media download returned HTTP {response.status_code}"
                        )
                    declared = response.headers.get("Content-Length")
                    if declared is not None:
                        try:
                            declared_size = int(declared)
                        except ValueError as exc:
                            raise TelegramProtocolError("invalid media Content-Length") from exc
                        if declared_size < 0:
                            raise TelegramProtocolError("invalid media Content-Length")
                        if declared_size > self._max_download_bytes:
                            raise TelegramMediaLimitError("Telegram media exceeds 20 MiB")
                    total = 0
                    async for chunk in response.aiter_bytes(64 * 1024):
                        total += len(chunk)
                        if total > self._max_download_bytes:
                            raise TelegramMediaLimitError("Telegram media exceeds 20 MiB")
                        if chunk:
                            yielded = True
                            yield chunk
                    return
            except (SourceError, TelegramMediaLimitError):
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if yielded or attempt + 1 == self._max_attempts:
                    raise SourceNetworkError(
                        redact_text(
                            f"Telegram media download failed: {type(exc).__name__}",
                            self._secrets,
                        )
                    ) from None
                await self._backoff(attempt)

    @staticmethod
    def _parse_sticker_set(value: Any) -> SourceCollection:
        if not isinstance(value, Mapping):
            raise TelegramProtocolError("Telegram returned an invalid StickerSet")
        name = value.get("name")
        title = value.get("title")
        sticker_type = value.get("sticker_type")
        stickers = value.get("stickers")
        if (
            not isinstance(name, str)
            or not _SHORT_NAME.fullmatch(name)
            or not isinstance(title, str)
            or _CONTROL.search(title)
            or sticker_type != "custom_emoji"
            or not isinstance(stickers, list)
        ):
            if sticker_type not in {None, "custom_emoji"}:
                raise UnsupportedSourceError("Telegram set is not a custom emoji set")
            raise TelegramProtocolError("Telegram StickerSet fields are invalid")
        items = tuple(
            TelegramBotAPI._parse_sticker(item, index) for index, item in enumerate(stickers)
        )
        if len({item.native_id for item in items}) != len(items):
            raise TelegramProtocolError("Telegram StickerSet contains duplicate custom emoji IDs")
        extension: dict[str, object] = {
            "schema_version": "1.0.0",
            "retrieved_via": "bot_api",
            "short_name": name,
            "sticker_type": "custom_emoji",
            "set_fingerprint_sha256": telegram_set_fingerprint(items),
        }
        return SourceCollection(
            platform="telegram",
            kind="custom_emoji_set",
            native_namespace="sticker_set.name",
            scope_id="global",
            native_id=name,
            title=title,
            canonical_url=f"https://t.me/addemoji/{name}",
            item_count=len(items),
            items=items,
            extension=extension,
        )

    @staticmethod
    def _parse_sticker(value: Any, position: int) -> SourceEmoji:
        if not isinstance(value, Mapping):
            raise TelegramProtocolError("Telegram returned an invalid Sticker")
        required_strings = ("file_id", "file_unique_id", "custom_emoji_id")
        if any(not isinstance(value.get(key), str) or not value[key] for key in required_strings):
            raise TelegramProtocolError("Telegram custom emoji is missing a stable identifier")
        width, height = value.get("width"), value.get("height")
        if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
            raise TelegramProtocolError("Telegram custom emoji dimensions are invalid")
        animated = value.get("is_animated") is True
        video = value.get("is_video") is True
        if animated and video:
            raise TelegramProtocolError("Telegram custom emoji has conflicting media flags")
        return SourceEmoji(
            native_namespace="custom_emoji.id",
            scope_id="global",
            native_id=value["custom_emoji_id"],
            file_unique_id=value["file_unique_id"],
            position=position,
            width=width,
            height=height,
            animated=animated,
            video=video,
            needs_repainting=value.get("needs_repainting") is True,
            fallback_emoji=value.get("emoji") if isinstance(value.get("emoji"), str) else None,
            declared_file_size=value.get("file_size")
            if isinstance(value.get("file_size"), int)
            else None,
            media_format="tgs" if animated else "webm" if video else "webp",
            file_id=value["file_id"],
        )
