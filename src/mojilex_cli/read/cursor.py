"""Deterministic, tamper-evident (but deliberately unsigned) CLI cursors."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import Any, cast

import rfc8785

from mojilex_cli.commands.runtime import CommandError

MAX_CURSOR_BYTES = 4096


def _jcs(value: object) -> bytes:
    try:
        return rfc8785.dumps(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise CommandError(
            "CURSOR_INVALID",
            "The cursor contains a value that cannot be canonically serialized.",
            hint="Restart pagination without --cursor.",
        ) from exc


def _sha(value: object) -> str:
    return hashlib.sha256(_jcs(value)).hexdigest()


def pagination_domain(
    *, command: str, pinned_state: dict[str, Any], semantic_arguments: dict[str, Any]
) -> str:
    """Bind pagination to one command, exact snapshot, and normalized request."""

    return _sha(
        {
            "domain": "cli-pagination-v1",
            "command": command,
            "pinned_state": pinned_state,
            "semantic_arguments_without_cursor": semantic_arguments,
        }
    )


def encode_cursor(domain_sha256: str, last_tuple: list[object]) -> str:
    payload = {
        "version": "1",
        "pagination_domain_sha256": domain_sha256,
        "last_tuple": last_tuple,
    }
    payload_bytes = _jcs(payload)
    encoded = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode("ascii")
    check = _sha({"domain": "cli-cursor-v1", "payload": payload})
    cursor = f"{encoded}.{check}"
    if len(cursor.encode("ascii")) > MAX_CURSOR_BYTES:
        raise CommandError(
            "RESOURCE_LIMIT_EXCEEDED",
            "The generated cursor exceeds the 4096-byte safety limit.",
            hint="Use a narrower query or report an incompatible index tuple.",
        )
    return cursor


def decode_cursor(cursor: str, expected_domain_sha256: str) -> list[object]:
    try:
        raw = cursor.encode("ascii")
    except UnicodeEncodeError as exc:
        raise _invalid_cursor() from exc
    if len(raw) > MAX_CURSOR_BYTES:
        raise CommandError(
            "RESOURCE_LIMIT_EXCEEDED",
            "The pagination cursor exceeds the 4096-byte safety limit.",
            hint="Restart pagination without --cursor.",
        )
    if not raw or raw.count(b".") != 1:
        raise _invalid_cursor()
    encoded, supplied_check = raw.split(b".", 1)
    if len(supplied_check) != 64 or b"=" in encoded:
        raise _invalid_cursor()
    try:
        padding = b"=" * ((4 - len(encoded) % 4) % 4)
        payload_bytes = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
        payload = json.loads(
            payload_bytes.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid number: {token}")
            ),
        )
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid_cursor() from exc
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "pagination_domain_sha256",
        "last_tuple",
    }:
        raise _invalid_cursor()
    last_tuple = payload.get("last_tuple")
    if (
        payload.get("version") != "1"
        or not isinstance(last_tuple, list)
        or len(last_tuple) > 16
        or any(
            isinstance(item, (dict, list, float))
            or not isinstance(item, (str, int, bool, type(None)))
            for item in last_tuple
        )
    ):
        raise _invalid_cursor()
    try:
        canonical = rfc8785.dumps(payload)
    except (TypeError, ValueError) as exc:
        raise _invalid_cursor() from exc
    reencoded = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=")
    if canonical != payload_bytes or reencoded != encoded:
        raise _invalid_cursor()
    if payload.get("pagination_domain_sha256") != expected_domain_sha256:
        raise CommandError(
            "CURSOR_INVALID",
            "The cursor does not belong to this command, snapshot, or request.",
            hint="Restart pagination without --cursor.",
        )
    expected_check = _sha({"domain": "cli-cursor-v1", "payload": payload})
    if not hmac.compare_digest(supplied_check.decode("ascii"), expected_check):
        raise _invalid_cursor()
    return list(last_tuple)


def _invalid_cursor() -> CommandError:
    return CommandError(
        "CURSOR_INVALID",
        "The pagination cursor is malformed or its integrity check failed.",
        hint="Restart pagination without --cursor.",
    )
