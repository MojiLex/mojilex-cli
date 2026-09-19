"""Atomic, secret-free checkpoints and per-collection execution locks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from collections import OrderedDict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Literal

from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mojilex_cli.config import secrets as secret_rules
from mojilex_cli.config.secrets import contains_secret_text, is_secret_key, url_has_credentials

_RUN_ID = re.compile(r"mlxrun_[0-9a-f]{32}\Z")
_SHA = re.compile(r"[0-9a-f]{40,64}\Z")
_FULL_GIT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_AI_ITEM_LABEL = re.compile(r"E[0-9]{3}\Z")
_SAFE_REMOTE = re.compile(r"[A-Za-z0-9._-]{1,100}\Z")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_TELEGRAM_TOKEN = re.compile(r"\b[0-9]{5,12}:[A-Za-z0-9_-]{20,}\b")
# A multi-pack run retains exact request identities for every emoji. Keep the
# reader and writer bound identical so every saved checkpoint remains resumable.
_MAX_CHECKPOINT_BYTES = 128 * 1024 * 1024
_FORBIDDEN_FIELDS = frozenset(
    {
        "file_id",
        "download_url",
        "raw_download_url",
        "local_path",
        "media_bytes",
        "frame_paths",
        "contact_sheet",
    }
)


def _safety_rule_identity() -> tuple[object, ...]:
    # Current detectors are pure functions over immutable rules; they do not read
    # environment credentials. Disable memoization if a detector/rule is replaced.
    return (
        contains_secret_text,
        is_secret_key,
        url_has_credentials,
        _TELEGRAM_TOKEN,
        _SHA256,
        _FORBIDDEN_FIELDS,
        secret_rules.contains_secret_text,
        secret_rules.is_secret_key,
        secret_rules.url_has_credentials,
        secret_rules.redact_text,
        secret_rules.SECRET_ENV_NAMES,
        secret_rules._SECRET_KEY,
        secret_rules._KNOWN_CREDENTIALS,
        secret_rules._TELEGRAM_BOT_PATH,
        secret_rules._AUTHORITY_CREDENTIALS,
    )


_ORIGINAL_SAFETY_RULES = _safety_rule_identity()


class _SafeTextMemo:
    """Bounded successful checks of exact immutable text, never mutable subtrees."""

    MAX_ENTRIES = 8192
    MAX_BYTES = 1024 * 1024
    MAX_TEXT_BYTES = 1024

    def __init__(self) -> None:
        self.entries: OrderedDict[tuple[bool, str], int] = OrderedDict()
        self.size_bytes = 0

    def clear(self) -> None:
        self.entries.clear()
        self.size_bytes = 0

    def contains(self, text: str, *, key: bool) -> bool:
        identity = (key, text)
        if identity not in self.entries:
            return False
        self.entries.move_to_end(identity)
        return True

    def remember(self, text: str, *, key: bool) -> None:
        if len(text) > self.MAX_TEXT_BYTES:
            return
        size = len(text.encode("utf-8"))
        if size > self.MAX_TEXT_BYTES:
            return
        identity = (key, text)
        if identity in self.entries:
            self.entries.move_to_end(identity)
            return
        self.entries[identity] = size
        self.size_bytes += size
        while len(self.entries) > self.MAX_ENTRIES or self.size_bytes > self.MAX_BYTES:
            _, removed_size = self.entries.popitem(last=False)
            self.size_bytes -= removed_size


class _SafePayloadMemo:
    """Bounded exact JSON payloads which passed the complete safety traversal.

    Model instances and their nested dictionaries can be mutated through copies.
    Only serialized content is a reusable identity, never an object ID or a hash.
    """

    MAX_ENTRIES = 16384
    MAX_BYTES = 16 * 1024 * 1024

    def __init__(self) -> None:
        self.entries: OrderedDict[bytes, None] = OrderedDict()
        self.size_bytes = 0

    def clear(self) -> None:
        self.entries.clear()
        self.size_bytes = 0

    def contains(self, payload: bytes) -> bool:
        if payload not in self.entries:
            return False
        self.entries.move_to_end(payload)
        return True

    def remember(self, payload: bytes) -> None:
        if len(payload) > self.MAX_BYTES:
            return
        if payload in self.entries:
            self.entries.move_to_end(payload)
            return
        self.entries[payload] = None
        self.size_bytes += len(payload)
        while len(self.entries) > self.MAX_ENTRIES or self.size_bytes > self.MAX_BYTES:
            removed, _ = self.entries.popitem(last=False)
            self.size_bytes -= len(removed)


class RunStoreError(RuntimeError):
    """Invalid or unavailable persisted run state."""

    code = "CONFIG_INVALID"
    retryable = False


class RunLockedError(RunStoreError):
    """A run or collection is already being mutated by another process."""

    code = "GIT_CONFLICT"
    retryable = True


class ResumeIncompatibleError(RunStoreError):
    """The recorded run can no longer be applied to the selected target."""

    code = "SOURCE_CHANGED_DURING_RUN"


class RunIssue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    message: str
    retryable: bool
    hint: str
    source: str | None = None
    entity_id: str | None = None
    details: dict[str, object] | None = None


class AIRequestCheckpoint(BaseModel):
    """Exact, bounded identity of one paid-or-cached model request step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal["primary", "escalated"]
    model: str = Field(min_length=1, max_length=256)
    model_revision: str | None = Field(default=None, min_length=1, max_length=256)
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shown_media_sha256: tuple[str, ...] = Field(min_length=1, max_length=32)
    item_label: str = Field(pattern=r"^E[0-9]{3}$")

    @field_validator("cache_key", "plan_sha256", "request_sha256")
    @classmethod
    def request_hashes_are_canonical(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("AI request hashes must be canonical lowercase SHA-256")
        return value

    @field_validator("shown_media_sha256")
    @classmethod
    def shown_media_hashes_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _SHA256.fullmatch(item) for item in value):
            raise ValueError("AI shown-media values must be canonical SHA-256")
        return value

    @field_validator("item_label")
    @classmethod
    def item_label_is_canonical(cls, value: str) -> str:
        if not _AI_ITEM_LABEL.fullmatch(value):
            raise ValueError("AI request item label must use E followed by three digits")
        return value

    @field_validator("model", "model_revision")
    @classmethod
    def model_identifiers_are_bounded_text(cls, value: str | None) -> str | None:
        if value is not None and (
            value != value.strip()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError("AI model identifiers must be trimmed printable text")
        return value


class ElementCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: Literal[
        "discovered",
        "metadata_ready",
        "media_verified",
        "palette_ready",
        "fingerprint_ready",
        "ai_cached",
        "ai_facets_ready",
        "candidate_scanned",
        "reviewed",
        "mapped",
        "validated",
        "failed",
    ]
    source_descriptor_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    media_sha256: tuple[str, ...] = ()
    deterministic_cache_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ai_cache_key: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    ai_requests: tuple[AIRequestCheckpoint, ...] = Field(default=(), max_length=2)
    palette_complete: bool = False
    fingerprint_complete: bool = False
    ai_facets_complete: bool = False
    candidate_scan_complete: bool = False
    review_complete: bool = False
    error_code: str | None = None

    @field_validator("source_descriptor_sha256", "deterministic_cache_key", "ai_cache_key")
    @classmethod
    def optional_hashes_are_canonical(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("checkpoint hashes must be canonical lowercase SHA-256")
        return value

    @field_validator("media_sha256")
    @classmethod
    def media_hashes_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _SHA256.fullmatch(item) for item in value):
            raise ValueError("checkpoint media hashes must be canonical lowercase SHA-256")
        return value

    @model_validator(mode="after")
    def exact_request_context_is_complete(self) -> ElementCheckpoint:
        if self.ai_requests:
            if self.ai_cache_key != self.ai_requests[-1].cache_key:
                raise ValueError("final AI cache key must match the request trace")
            stages = tuple(request.stage for request in self.ai_requests)
            if stages not in {("primary",), ("escalated",), ("primary", "escalated")}:
                raise ValueError("AI request trace stages are not canonical")
        return self


class DedupeScanCheckpoint(BaseModel):
    """Bounded, content-addressed result of one exact dedupe scan input."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: Literal[1] = 1
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    profile: str = Field(min_length=1, max_length=64)
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["exact", "near"]
    max_candidates: int = Field(ge=1, le=10_000)
    selected_emoji_ids: tuple[str, ...]
    report: dict[str, object]

    @field_validator("selected_emoji_ids")
    @classmethod
    def selected_ids_are_canonical(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(sorted(set(value))) != value:
            raise ValueError("dedupe selected IDs must be unique and sorted")
        return value

    @model_validator(mode="after")
    def report_is_bounded(self) -> DedupeScanCheckpoint:
        encoded = json.dumps(
            self.report, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(encoded) > 4 * 1024 * 1024:
            raise ValueError("dedupe checkpoint report exceeds 4 MiB")
        return self


class PublicationCheckpoint(BaseModel):
    """Bounded intent and observed progress for one remote publication attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    format_version: Literal[1] = 1
    mode: Literal["pr", "direct"]
    remote: str = Field(min_length=1, max_length=100)
    base_branch: str = Field(min_length=1, max_length=200)
    expected_old_base: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    candidate_sha: str = Field(pattern=r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
    candidate_branch: str = Field(min_length=1, max_length=200)
    phase: Literal["prepared", "candidate_pushed", "checks_passed", "completed"]
    completed_source_indexes: tuple[int, ...] = Field(default=(), max_length=10_000)

    @field_validator("expected_old_base", "candidate_sha")
    @classmethod
    def object_ids_are_canonical(cls, value: str) -> str:
        if not _FULL_GIT_SHA.fullmatch(value):
            raise ValueError("publication object IDs must be canonical lowercase Git hashes")
        return value

    @field_validator("remote")
    @classmethod
    def remote_is_safe(cls, value: str) -> str:
        if not _SAFE_REMOTE.fullmatch(value):
            raise ValueError("publication remote name is unsafe")
        return value

    @field_validator("base_branch", "candidate_branch")
    @classmethod
    def branches_are_safe(cls, value: str) -> str:
        if (
            _CONTROL.search(value)
            or value.startswith(("-", "/"))
            or value.endswith(("/", ".", ".lock"))
            or ".." in value
            or "//" in value
            or "@{" in value
            or any(character in value for character in " ~^:?*[\\")
        ):
            raise ValueError("publication branch name is unsafe")
        return value

    @field_validator("completed_source_indexes")
    @classmethod
    def source_indexes_are_canonical(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(index < 0 or index > 1_000_000 for index in value):
            raise ValueError("publication source index is outside the safe range")
        if value != tuple(sorted(set(value))):
            raise ValueError("publication source indexes must be unique and sorted")
        return value

    @model_validator(mode="after")
    def state_is_consistent(self) -> PublicationCheckpoint:
        if self.expected_old_base == self.candidate_sha:
            raise ValueError("publication candidate must differ from the expected old base")
        if len(self.expected_old_base) != len(self.candidate_sha):
            raise ValueError("publication object IDs must use one Git object format")
        if self.base_branch == self.candidate_branch:
            raise ValueError("publication candidate branch must differ from the base branch")
        if not self.candidate_branch.startswith("mojilex/"):
            raise ValueError("publication candidate branch must use the mojilex namespace")
        if self.mode == "pr" and self.phase == "checks_passed":
            raise ValueError("Pull Request publication has no direct-checks phase")
        return self


class RunCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(pattern=r"^mlxrun_[0-9a-f]{32}$")
    command: str
    safe_parameters: dict[str, object]
    cli_version: str
    schema_version: str
    target_repository: str
    base_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    created_at: datetime
    updated_at: datetime
    status: Literal[
        "running",
        "succeeded",
        "noop",
        "stale",
        "partial",
        "failed",
        "interrupted",
        "budget_exceeded",
    ] = "running"
    elements: dict[str, ElementCheckpoint] = {}
    dedupe_scan: DedupeScanCheckpoint | None = None
    publication: PublicationCheckpoint | None = None
    ai_requests_used: int = Field(default=0, ge=0)
    ai_cost_reserved_usd: Decimal = Field(default=Decimal("0"), ge=0)
    issues: tuple[RunIssue, ...] = ()

    @field_validator("created_at", "updated_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise ValueError("checkpoint timestamps must be UTC-aware")
        return value


def new_run_id() -> str:
    return f"mlxrun_{uuid.uuid4().hex}"


def new_checkpoint(
    *,
    command: str,
    safe_parameters: Mapping[str, object],
    cli_version: str,
    schema_version: str,
    target_repository: str,
    base_revision: str,
    run_id: str | None = None,
) -> RunCheckpoint:
    _assert_safe(safe_parameters)
    if not _SHA.fullmatch(base_revision):
        raise RunStoreError("base revision is not a full Git object ID")
    now = datetime.now(UTC).replace(microsecond=0)
    return RunCheckpoint(
        run_id=run_id or new_run_id(),
        command=command,
        safe_parameters=dict(safe_parameters),
        cli_version=cli_version,
        schema_version=schema_version,
        target_repository=target_repository,
        base_revision=base_revision,
        created_at=now,
        updated_at=now,
    )


class RunStore:
    def __init__(
        self,
        root: Path,
        *,
        repository_root: Path | None = None,
        write_enabled: bool = True,
    ) -> None:
        self.root = root.resolve()
        if repository_root is not None and self.root.is_relative_to(repository_root.resolve()):
            raise RunStoreError("run checkpoints must be outside the target repository")
        self.write_enabled = write_enabled
        self._safe_text = _SafeTextMemo()
        self._safe_payloads = _SafePayloadMemo()
        if write_enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "locks").mkdir(exist_ok=True)
            try:
                os.chmod(self.root, 0o700)
            except OSError:
                pass

    def save(self, checkpoint: RunCheckpoint) -> Path:
        if not self.write_enabled:
            raise RunStoreError("checkpoints are disabled in dry-run mode")
        self._assert_checkpoint_safe(checkpoint)
        destination = self._checkpoint_path(checkpoint.run_id)
        serialized = checkpoint.model_dump_json().encode("utf-8") + b"\n"
        if len(serialized) > _MAX_CHECKPOINT_BYTES:
            raise RunStoreError("checkpoint exceeds the safe size limit")
        with self.run_lock(checkpoint.run_id, timeout=5):
            _atomic_write(destination, serialized)
        return destination

    def load(self, run_id: str) -> RunCheckpoint:
        path = self._checkpoint_path(run_id)
        try:
            if path.is_symlink():
                raise RunStoreError("checkpoint symlinks are forbidden")
            with path.open("rb") as stream:
                payload = stream.read(_MAX_CHECKPOINT_BYTES + 1)
        except OSError as exc:
            raise RunStoreError(f"cannot read checkpoint {run_id}") from exc
        if len(payload) > _MAX_CHECKPOINT_BYTES:
            raise RunStoreError("checkpoint exceeds the safe size limit")
        try:
            checkpoint = RunCheckpoint.model_validate_json(payload)
        except ValueError as exc:
            raise RunStoreError("checkpoint is malformed") from exc
        self._assert_checkpoint_safe(checkpoint)
        return checkpoint

    def _assert_checkpoint_safe(self, checkpoint: RunCheckpoint) -> None:
        memo: _SafeTextMemo | None = self._safe_text
        if _safety_rule_identity() != _ORIGINAL_SAFETY_RULES:
            self._safe_text.clear()
            self._safe_payloads.clear()
            _assert_safe(checkpoint.model_dump(mode="json"))
            return
        payload = checkpoint.model_dump(mode="json")
        elements = payload.pop("elements")
        _assert_safe(dict.fromkeys(payload), memo=memo)
        for field, value in payload.items():
            self._assert_payload_safe(value, path=f"checkpoint.{field}")
        if not isinstance(elements, dict):
            _assert_safe({"elements": elements}, memo=memo)
            return
        # Check container and element keys even when the corresponding value was
        # seen under a different key. A safe value never legitimizes an unsafe key.
        self._assert_payload_safe({"elements": dict.fromkeys(elements)}, path="checkpoint")
        for key, element in elements.items():
            self._assert_payload_safe(element, path=f"checkpoint.elements.{key}")

    def _assert_payload_safe(self, value: object, *, path: str) -> None:
        serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if not self._safe_payloads.contains(serialized):
            _assert_safe(value, path=path, memo=self._safe_text)
            self._safe_payloads.remember(serialized)

    def load_for_resume(
        self,
        run_id: str,
        *,
        schema_version: str,
        target_repository: str | None = None,
    ) -> RunCheckpoint:
        checkpoint = self.load(run_id)
        if checkpoint.schema_version != schema_version:
            raise ResumeIncompatibleError(
                "checkpoint schema version differs from the target schema"
            )
        if target_repository is not None and checkpoint.target_repository != target_repository:
            raise ResumeIncompatibleError("checkpoint targets a different repository")
        return checkpoint

    @contextmanager
    def run_lock(self, run_id: str, *, timeout: float = 0) -> Iterator[None]:
        self._validate_run_id(run_id)
        if not self.write_enabled:
            raise RunStoreError("locks are disabled in dry-run mode")
        lock = FileLock(str(self.root / "locks" / f"run-{run_id}.lock"))
        try:
            with lock.acquire(timeout=timeout):
                yield
        except Timeout as exc:
            raise RunLockedError(f"run is already locked: {run_id}") from exc

    @contextmanager
    def execution_lock(self, run_id: str, *, timeout: float = 0) -> Iterator[None]:
        """Serialize one complete resume/describe/submit operation."""

        self._validate_run_id(run_id)
        if not self.write_enabled:
            raise RunStoreError("locks are disabled in dry-run mode")
        lock = FileLock(str(self.root / "locks" / f"execution-{run_id}.lock"))
        try:
            with lock.acquire(timeout=timeout):
                yield
        except Timeout as exc:
            raise RunLockedError(f"run is already executing: {run_id}") from exc

    @contextmanager
    def collection_lock(
        self, platform: str, native_id: str, *, timeout: float = 0
    ) -> Iterator[None]:
        if not platform or not native_id or "\x00" in platform + native_id:
            raise ValueError("invalid collection lock identity")
        digest = hashlib.sha256(f"{platform}\0{native_id}".encode()).hexdigest()
        lock = FileLock(str(self.root / "locks" / f"collection-{digest}.lock"))
        try:
            with lock.acquire(timeout=timeout):
                yield
        except Timeout as exc:
            raise RunLockedError("another run is already processing this collection") from exc

    def _checkpoint_path(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self.root / f"{run_id}.json"

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not _RUN_ID.fullmatch(run_id):
            raise RunStoreError("invalid run ID")


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        os.replace(temp, path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temp.unlink(missing_ok=True)


def _assert_safe(
    value: object, *, path: str = "checkpoint", memo: _SafeTextMemo | None = None
) -> None:
    if isinstance(value, bytes):
        raise RunStoreError(f"binary content is forbidden in {path}")
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if memo is None or not memo.contains(key, key=True):
                if key.lower() in _FORBIDDEN_FIELDS or is_secret_key(key):
                    raise RunStoreError(f"unsafe field is forbidden in {path}: {key}")
                if memo is not None:
                    memo.remember(key, key=True)
            _assert_safe(child, path=f"{path}.{key}", memo=memo)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_safe(child, path=path, memo=memo)
    elif isinstance(value, str):
        # With the original rules, exactly 64 lowercase hex characters cannot
        # contain any credential prefix, URL separator, or userinfo delimiter.
        # RunStore disables this path along with memoization if any rule changes.
        if memo is not None and len(value) == 64 and _SHA256.fullmatch(value):
            return
        if memo is not None and memo.contains(value, key=False):
            return
        if _TELEGRAM_TOKEN.search(value) or contains_secret_text(value):
            raise RunStoreError(f"credential is forbidden in {path}")
        if "://" in value and url_has_credentials(value):
            raise RunStoreError(f"credential URL is forbidden in {path}")
        if memo is not None:
            memo.remember(value, key=False)
