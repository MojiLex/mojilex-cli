"""Bounded checkpoint receipts for completed private previews, never AI reuse."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from mojilex_cli.concurrency import current_batch_limits
from mojilex_cli.media.models import HARD_MAX_FRAMES, ProcessedMedia
from mojilex_cli.media.resume import RetainedMediaStore
from mojilex_cli.runs import RunCheckpoint

COMPLETED_MEDIA_PARAMETER = "retained_completed_media"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_MAX_RECEIPTS = 100_000
_RECEIPT_BYTES = 16 * 1024


def mark_completed_retained(root: Path, completed: Mapping[str, ProcessedMedia]) -> None:
    """Mark already-admitted previews after durable save and temporary cleanup."""
    batch = current_batch_limits()
    if batch is None or len(completed) > _MAX_RECEIPTS:
        return
    key = os.path.normcase(os.path.abspath(root))
    with batch.retained_lock:
        store = batch.retained_stores.get(key)
    if isinstance(store, RetainedMediaStore):
        for descriptor, media in completed.items():
            store.mark_completed(descriptor, media)


def _bounded_json(value: object) -> str:
    """Reject large/deep input before allocating its serialized representation."""
    nodes = 0
    text_bytes = 0

    def check(item: object, depth: int = 0) -> None:
        nonlocal nodes, text_bytes
        nodes += 1
        if nodes > 2048 or depth > 16:
            raise ValueError("completed preview receipt exceeds structural bounds")
        if isinstance(item, str):
            if len(item) > _RECEIPT_BYTES:
                raise ValueError("completed preview receipt string is too long")
            text_bytes += len(item.encode("utf-8"))
            if text_bytes > _RECEIPT_BYTES:
                raise ValueError("completed preview receipt text exceeds its bound")
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("completed preview receipt keys must be strings")
                check(key, depth + 1)
                check(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                check(child, depth + 1)
        elif item is not None and type(item) not in {int, bool, float}:
            raise ValueError("completed preview receipt is not JSON")

    check(value)
    encoded = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    if len(encoded.encode("utf-8")) > _RECEIPT_BYTES:
        raise ValueError("completed preview receipt exceeds its byte bound")
    return encoded


def completed_media_receipt(key: str, media: ProcessedMedia) -> dict[str, Any]:
    """Serialize the exact old rendering identity without temporary paths."""
    if not _SHA256.fullmatch(key):
        raise ValueError("completed preview descriptor must be a canonical SHA-256")
    count = media.semantic_frame_count
    if not 1 <= count <= HARD_MAX_FRAMES or (not media.metadata.animated and count != 1):
        raise ValueError("completed preview frame count is invalid")
    result = {
        "format_version": 1,
        "source_descriptor_sha256": key,
        "metadata": media.metadata.model_dump(mode="json"),
        "analysis": media.analysis.model_dump(mode="json") if media.analysis is not None else None,
        "frame_count": count,
        "has_dark_render": media.semantic_has_dark_render,
    }
    _bounded_json(result)
    return result


def restore_completed_media_receipts(checkpoint: RunCheckpoint) -> dict[str, ProcessedMedia]:
    """Restore reclamation permission only for exact validated checkpoint items.

    This grants no cache hit and does not bypass current decoder/profile checks.
    RetainedMediaStore verifies the old manifest fingerprint and all pixel bytes
    before deleting anything; missing/mismatched entries remain untouched.
    """
    receipts = checkpoint.safe_parameters.get(COMPLETED_MEDIA_PARAMETER)
    if not isinstance(receipts, dict) or len(receipts) > _MAX_RECEIPTS:
        return {}
    restored = {}
    for native_id, receipt in receipts.items():
        element = checkpoint.elements.get(native_id)
        complete_stage = element is not None and (
            element.stage == "validated"
            or (element.stage == "candidate_scanned" and checkpoint.status in {"succeeded", "noop"})
        )
        if not complete_stage or element is None or len(element.media_sha256) != 1:
            continue
        if not isinstance(receipt, dict) or set(receipt) != {
            "format_version",
            "source_descriptor_sha256",
            "metadata",
            "analysis",
            "frame_count",
            "has_dark_render",
        }:
            continue
        key = receipt["source_descriptor_sha256"]
        count = receipt["frame_count"]
        dark = receipt["has_dark_render"]
        if (
            type(receipt["format_version"]) is not int
            or receipt["format_version"] != 1
            or not isinstance(key, str)
            or not _SHA256.fullmatch(key)
            or key != element.source_descriptor_sha256
            or type(count) is not int
            or not 1 <= count <= HARD_MAX_FRAMES
            or type(dark) is not bool
        ):
            continue
        try:
            _bounded_json(receipt)
            payload = {
                "metadata": receipt["metadata"],
                "analysis": receipt["analysis"],
                "frame_paths": [],
                "rendered_frame_count": count,
                "has_dark_render": dark,
            }
            media = ProcessedMedia.model_validate_json(_bounded_json(payload), strict=True)
            if media.metadata.sha256 != element.media_sha256[0]:
                continue
            if not media.metadata.animated and count != 1:
                continue
            restored[key] = media
        except (ValueError, TypeError, RecursionError, OverflowError):
            continue
    return restored
