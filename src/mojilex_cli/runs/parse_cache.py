"""Bounded exact-byte checkpoint parsing memo with independent mutable results."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import RunCheckpoint

_LOCK = RLock()
_ENTRIES: OrderedDict[bytes, tuple[tuple[object, ...], RunCheckpoint]] = OrderedDict()
_MAX_BYTES = 16 * 1024 * 1024
_size = 0


def lookup(payload: bytes, rules: tuple[object, ...]) -> RunCheckpoint | None:
    with _LOCK:
        entry = _ENTRIES.get(payload)
        if entry is None or entry[0] != rules:
            return None
        _ENTRIES.move_to_end(payload)
        checkpoint = entry[1]
    return checkpoint.model_copy(deep=True)


def remember(payload: bytes, rules: tuple[object, ...], checkpoint: RunCheckpoint) -> None:
    global _size
    if len(payload) > _MAX_BYTES:
        return
    independent = checkpoint.model_copy(deep=True)
    with _LOCK:
        if payload in _ENTRIES:
            _size -= len(payload)
        _ENTRIES[payload] = rules, independent
        _ENTRIES.move_to_end(payload)
        _size += len(payload)
        while _size > _MAX_BYTES or len(_ENTRIES) > 8:
            old, _ = _ENTRIES.popitem(last=False)
            _size -= len(old)
