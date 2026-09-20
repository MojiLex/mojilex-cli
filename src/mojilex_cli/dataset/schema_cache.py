"""Authenticated local reuse of successful schema checks; never authoritative data.

The key is local to the application user. Signatures detect damaged or copied
cache rows, not an attacker who already controls that user's files and code.
All cache failures fall back to validation; no record contents are stored here.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .layout import assert_no_link_or_reparse

_LIMIT = 65_536
_MAX_DATABASE_BYTES = 32 * 1024 * 1024


def _rules_digest() -> bytes:
    digest = hashlib.sha256(b"mojilex-schema-success-v1\0")
    for name in ("validation.py", "schema_cache.py"):
        digest.update(Path(__file__).with_name(name).read_bytes().replace(b"\r\n", b"\n"))
    for package in ("jsonschema", "referencing"):
        digest.update(package.encode() + b"\0" + version(package).encode() + b"\0")
    return digest.digest()


class SchemaSuccessCache:
    """A bounded, best-effort cache for one operation; close persists new successes."""

    def __init__(self, root: Path) -> None:
        self.connection: sqlite3.Connection | None = None
        self.secret = b""
        self.rules = b""
        self.loaded: set[bytes] = set()
        try:
            assert_no_link_or_reparse(root)
            root.mkdir(parents=True, exist_ok=True)
            key_path = root / "authentication.key"
            database = root / "successes.sqlite3"
            assert_no_link_or_reparse(key_path)
            assert_no_link_or_reparse(database)
            try:
                fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                pass
            else:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(os.urandom(32))
                    stream.flush()
                    os.fsync(stream.fileno())
            if key_path.stat().st_size != 32:
                return
            self.secret = key_path.read_bytes()
            self.rules = _rules_digest()
            if database.exists() and database.stat().st_size > _MAX_DATABASE_BYTES:
                return
            connection = sqlite3.connect(database, timeout=0.1, check_same_thread=False)
            self.connection = connection
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS successes "
                "(cache_key BLOB PRIMARY KEY, signature BLOB NOT NULL)"
            )
            rows = connection.execute(
                "SELECT cache_key, signature FROM successes "
                "WHERE length(cache_key)=32 AND length(signature)=32 "
                "ORDER BY rowid DESC LIMIT ?",
                (_LIMIT,),
            )
            for key, signature in rows:
                if isinstance(key, bytes) and isinstance(signature, bytes):
                    if hmac.compare_digest(self._signature(key), signature):
                        self.loaded.add(key)
        except (OSError, ValueError, sqlite3.Error, PackageNotFoundError):
            self.close(())

    def _signature(self, key: bytes) -> bytes:
        return hmac.digest(self.secret, self.rules + key, "sha256")

    def close(self, successes: object) -> None:
        connection, self.connection = self.connection, None
        if connection is None:
            return
        try:
            if isinstance(successes, (dict, set, tuple, list)):
                rows = [
                    (key, self._signature(key))
                    for key in successes
                    if isinstance(key, bytes) and len(key) == 32 and key not in self.loaded
                ][-_LIMIT:]
                with connection:
                    connection.executemany(
                        "INSERT OR REPLACE INTO successes(cache_key,signature) VALUES (?,?)", rows
                    )
                    connection.execute(
                        "DELETE FROM successes WHERE rowid NOT IN "
                        "(SELECT rowid FROM successes ORDER BY rowid DESC LIMIT ?)",
                        (_LIMIT,),
                    )
        except (OSError, ValueError, sqlite3.Error, PackageNotFoundError):
            pass  # Cache persistence must never fail the user's operation.
        finally:
            connection.close()
