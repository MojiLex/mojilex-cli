"""Shared command execution, output, and stable error handling."""

from __future__ import annotations

import os
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mojilex_cli.config import ConfigError, redact_text
from mojilex_cli.dataset import DatasetLoadError, DatasetValidationError
from mojilex_cli.i18n import text as ui_text
from mojilex_cli.output.models import (
    ERROR_EXIT_CODES,
    OutputEnvelope,
    RunStatus,
    StructuredError,
    redact,
)

_T = TypeVar("_T")
_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_MACHINE_JSON_MODE: ContextVar[bool] = ContextVar("mojilex_machine_json_mode", default=False)
_ENVELOPE_EMITTED: ContextVar[bool] = ContextVar("mojilex_envelope_emitted", default=False)


@dataclass(slots=True)
class _CommandContext:
    run_id: str
    quiet: bool = False
    verbose: bool = False
    no_color: bool = False


_COMMAND_CONTEXT: ContextVar[_CommandContext | None] = ContextVar(
    "mojilex_command_context", default=None
)


def report_run_id(run_id: str) -> None:
    """Bind the durable checkpoint ID to the enclosing synchronous CLI invocation."""

    context = _COMMAND_CONTEXT.get()
    if context is not None:
        # The shared holder survives asyncio.run's copied ContextVar context.
        context.run_id = run_id


def report_progress(message: str, *, verbose: bool = False) -> None:
    """Human diagnostics always go to stderr, leaving machine stdout untouched."""

    context = _COMMAND_CONTEXT.get()
    if context is None or context.quiet or (verbose and not context.verbose):
        return
    console = Console(stderr=True, no_color=context.no_color)
    console.print(f"[{context.run_id}] {redact(message)}", markup=False)


class CommandError(RuntimeError):
    """Expected public failure carrying a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        hint: str,
        retryable: bool = False,
        source: str | None = None,
        entity_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if code not in ERROR_EXIT_CODES:
            raise ValueError(f"unregistered command error code: {code}")
        self.error = StructuredError(
            code=code,
            message=message,
            retryable=retryable,
            hint=hint,
            source=source,
            entity_id=entity_id,
            details=details,
        )


@dataclass(slots=True)
class CommandResult:
    result: dict[str, Any] = field(default_factory=dict)
    publication: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, Any] | str] = field(default_factory=list)
    errors: list[StructuredError] = field(default_factory=list)
    status: RunStatus = RunStatus.SUCCEEDED
    run_id: str | None = None


def new_run_id() -> str:
    """Return a dependency-free, time-sortable 26-character ULID."""

    value = (int(time.time_ns() // 1_000_000) << 80) | int.from_bytes(os.urandom(10), "big")
    encoded: list[str] = []
    for _ in range(26):
        encoded.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(encoded))


def require_confirmation(
    message: str, *, yes: bool, non_interactive: bool, json_output: bool
) -> None:
    if yes:
        return
    if non_interactive or json_output or _MACHINE_JSON_MODE.get():
        raise CommandError(
            "CONFIG_INVALID",
            "This sensitive operation requires explicit confirmation.",
            hint="Rerun with --yes after reviewing the exact target.",
        )
    if not typer.confirm(message, default=False):
        raise CommandError(
            "CONFIG_INVALID",
            "Operation was not confirmed.",
            hint="Review the target and rerun when ready.",
        )


def require_local_repository(value: str | Path) -> Path:
    text = str(value)
    if "://" in text or (
        "/" in text and not Path(text).exists() and not text.startswith(("./", "../"))
    ):
        raise CommandError(
            "CONFIG_INVALID",
            "This command requires a local dataset checkout.",
            hint="Clone MojiLex/mojilex and pass its path with --repo.",
        )
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise CommandError(
            "CONFIG_MISSING",
            f"Dataset directory does not exist: {path}",
            hint="Pass an existing MojiLex dataset checkout.",
        )
    return path


def structured_exception(exc: BaseException, *, debug: bool = False) -> StructuredError:
    if isinstance(exc, CommandError):
        return exc.error
    if isinstance(exc, (KeyboardInterrupt, typer.Abort)):
        return StructuredError(
            code="INTERRUPTED",
            message="Operation interrupted by the user.",
            retryable=True,
            hint="Use mojilex resume with the reported run ID when a checkpoint exists.",
        )
    raw_code = getattr(exc, "code", None)
    code = (
        {
            "SOURCE_ERROR": "SOURCE_NOT_FOUND",
            "SOURCE_INVALID_RESPONSE": "SOURCE_NOT_FOUND",
        }.get(raw_code, raw_code)
        if isinstance(raw_code, str)
        else None
    )
    if not isinstance(code, str) or code not in ERROR_EXIT_CODES:
        if isinstance(exc, (DatasetValidationError, DatasetLoadError, KeyError)):
            code = "VALIDATION_FAILED"
        elif isinstance(exc, (ConfigError, ValueError)) or is_usage_error(exc):
            code = "CONFIG_INVALID"
        else:
            code = "INTERNAL_ERROR"
    retryable = bool(getattr(exc, "retryable", False))
    hints = {
        "CONFIG_INVALID": "Check the command options and non-secret configuration.",
        "VALIDATION_FAILED": "Run mojilex validate --strict and fix every reported issue.",
        "INTERNAL_ERROR": "Rerun with --debug and report the sanitized traceback.",
    }
    details = {"exception_type": type(exc).__name__} if debug else None
    return StructuredError(
        code=code,
        message=redact_text(str(exc) or type(exc).__name__, ()),
        retryable=retryable,
        hint=hints.get(code, "Correct the reported condition and retry."),
        details=details,
    )


@contextmanager
def machine_output_mode(enabled: bool = True) -> Iterator[None]:
    """Force nested commands into the one-envelope JSON output contract."""

    mode_token = _MACHINE_JSON_MODE.set(enabled)
    emitted_token = _ENVELOPE_EMITTED.set(False)
    try:
        yield
    finally:
        _ENVELOPE_EMITTED.reset(emitted_token)
        _MACHINE_JSON_MODE.reset(mode_token)


def machine_envelope_emitted() -> bool:
    return _ENVELOPE_EMITTED.get()


def machine_output_requested() -> bool:
    """Return whether the outer CLI boundary requested machine JSON output."""

    return _MACHINE_JSON_MODE.get()


def mark_machine_envelope_emitted() -> None:
    """Mark output emitted by a specialized envelope implementation."""

    _ENVELOPE_EMITTED.set(True)


def execute(
    command: str,
    action: Callable[[], CommandResult],
    *,
    json_output: bool,
    quiet: bool = False,
    debug: bool = False,
    verbose: bool = False,
    no_color: bool = False,
) -> None:
    """Run one command and emit either one JSON object or concise human output."""

    effective_json = json_output or _MACHINE_JSON_MODE.get()
    run_id = new_run_id()
    context = _CommandContext(run_id, quiet=quiet, verbose=verbose, no_color=no_color)
    context_token = _COMMAND_CONTEXT.set(context)
    try:
        command_result = action()
        envelope = OutputEnvelope(
            ok=not command_result.errors,
            command=command,
            status=command_result.status,
            run_id=command_result.run_id or context.run_id,
            result=command_result.result,
            publication=command_result.publication,
            warnings=command_result.warnings,
            errors=command_result.errors,
        )
    except (KeyboardInterrupt, typer.Abort) as exc:
        error = structured_exception(exc)
        envelope = OutputEnvelope(
            ok=False,
            command=command,
            status=RunStatus.INTERRUPTED,
            run_id=context.run_id,
            errors=[error],
        )
    except BaseException as exc:  # commands must always preserve the output contract
        if debug and not effective_json:
            formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            sanitized = str(redact(formatted))
            typer.echo(sanitized, err=True, nl=not sanitized.endswith("\n"))
        error = structured_exception(exc, debug=debug)
        status = (
            RunStatus.BUDGET_EXCEEDED
            if error.code in {"BUDGET_EXCEEDED", "UNKNOWN_COST"}
            else RunStatus.FAILED
        )
        if error.code in {"SOURCE_CHANGED_DURING_RUN", "IDENTITY_CONFLICT"}:
            status = RunStatus.STALE
        envelope = OutputEnvelope(
            ok=False,
            command=command,
            status=status,
            run_id=context.run_id,
            errors=[error],
        )
    finally:
        _COMMAND_CONTEXT.reset(context_token)

    if effective_json:
        typer.echo(envelope.to_json())
    elif not quiet or not envelope.ok:
        _render_human(envelope, no_color=no_color)
    _ENVELOPE_EMITTED.set(True)
    if envelope.exit_code:
        raise typer.Exit(code=int(envelope.exit_code))


def _render_human(envelope: OutputEnvelope, *, no_color: bool = False) -> None:
    console = Console(stderr=not envelope.ok, no_color=no_color)
    if envelope.ok:
        raw_label = str(envelope.status).replace("RunStatus.", "").lower()
        label = ui_text(raw_label)
        console.print(f"[green]MojiLex {envelope.command}: {label}[/green]")
        safe_result = redact(envelope.result)
        if isinstance(safe_result, Mapping) and safe_result:
            table = Table(show_header=False, box=None, pad_edge=False)
            for key, value in safe_result.items():
                table.add_row(
                    Text(str(key).replace("_", " ")),
                    Text(str(value)),
                )
            console.print(table)
        for warning in envelope.warnings:
            console.print(
                f"{ui_text('Warning')}: {redact(warning)}", style="yellow", markup=False
            )
        return
    for error in envelope.errors:
        safe_error = error.as_dict()
        message = ui_text(str(safe_error["message"]))
        hint = ui_text(str(safe_error["hint"]))
        console.print(f"{error.code}: {message}", style="red", markup=False)
        console.print(f"{ui_text('Hint')}: {hint}", markup=False)
    console.print(f"{ui_text('Run ID')}: {envelope.run_id}", markup=False)


def is_usage_error(exc: BaseException) -> bool:
    """Recognize Click/Typer parse errors without depending on a private Click package."""

    return getattr(exc, "exit_code", None) == 2 and callable(getattr(exc, "format_message", None))
