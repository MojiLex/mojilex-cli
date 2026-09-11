"""Offline, read-only access to pinned MojiLex release snapshots."""

from __future__ import annotations

from mojilex_cli.read.snapshot import LoadedSnapshot, load_snapshot

__all__ = ["LoadedSnapshot", "load_snapshot"]
