"""Shared command execution, output, and stable error handling."""

# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
import os
import time
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar, get_args

import typer
from pydantic import BaseModel, ValidationError
from pydantic_core.core_schema import ErrorType
from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.live import Live
from rich.segment import Segment
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from mojilex_cli.config import ConfigError, redact_text
from mojilex_cli.dataset import DatasetLoadError, DatasetValidationError
from mojilex_cli.i18n import confirm as ui_confirm
from mojilex_cli.i18n import current_ui_language
from mojilex_cli.i18n import text as ui_text
from mojilex_cli.output.models import (
    ERROR_EXIT_CODES,
    OutputEnvelope,
    RunStatus,
    StructuredError,
    redact,
)

from .queue_progress import SHARED, PackQueue

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
    json_output: bool = False
    live: Live | None = None
    progress_view: RenderableType | None = None
    progress_views: dict[object, RenderableType] = field(default_factory=dict)
    progress_pause_keys: set[object] = field(default_factory=set)
    progress_note: str = ""
    progress_paused: bool = False
    operations: list[tuple[str, float]] = field(default_factory=list)
    operation_live: Live | None = None
    operation_spinner: Spinner = field(default_factory=lambda: Spinner("dots"))
    prompt_depth: int = 0
    pending_progress: list[RenderableType] = field(default_factory=list)
    pack_queue: PackQueue | None = None
    last_pack_refresh: float = 0.0
    command: str = ""


class _ProgressDisplay:
    """Fit concurrent progress into the current terminal, including after resize."""

    def __init__(self, context: _CommandContext) -> None:
        self.context = context

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        views = self.context.progress_views
        height = max(1, console.size.height - 5)
        if len(views) > 1:
            rows: list[RenderableType] = []
            for key, view in views.items():
                compact = getattr(key, "compact_view", None)
                rows.append(compact() if callable(compact) else view)
            view = Group(*rows)
        else:
            view = Group(*views.values())
        lines = console.render_lines(view, options.update(height=None), pad=False)
        if len(lines) > height:
            for line in lines[: height - 1]:
                yield from line
                yield Segment.line()
            hidden = (
                f"Активных этапов: {len(views)}; остальные скрыты по высоте окна"
                if current_ui_language() == "ru"
                else f"Active stages: {len(views)}; remaining details hidden to fit the terminal"
            )
            yield Text(hidden, overflow="ellipsis", no_wrap=True, style="dim")
        else:
            for line in lines:
                yield from line
                yield Segment.line()


_COMMAND_CONTEXT: ContextVar[_CommandContext | None] = ContextVar(
    "mojilex_command_context", default=None
)


def _operation_view(context: _CommandContext) -> RenderableType:
    table = Table.grid(padding=(0, 1))
    operations = tuple(context.operations)
    if operations:
        label, started = operations[-1]
        elapsed = int(time.monotonic() - started)
        table.add_row(
            context.operation_spinner,
            Text(label),
            Text(f"{elapsed // 60:02d}:{elapsed % 60:02d}", style="dim"),
        )
    return table


def _stop_operation_live(context: _CommandContext) -> None:
    if context.operation_live is not None:
        context.operation_live.stop()
        context.operation_live = None


def _resume_operation_live(context: _CommandContext) -> None:
    if (
        not context.operations
        or context.quiet
        or context.json_output
        or context.progress_paused
        or context.prompt_depth
        or context.live is not None
    ):
        return
    console = Console(stderr=True, no_color=context.no_color)
    if not console.is_terminal or console.is_dumb_terminal:
        return
    if context.operation_live is None:
        context.operation_live = Live(
            get_renderable=lambda: _operation_view(context),
            console=console,
            auto_refresh=True,
            refresh_per_second=4,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        context.operation_live.start(refresh=True)
    else:
        context.operation_live.refresh()


@contextmanager
def operation_progress(label: str) -> Iterator[None]:
    """Show activity and elapsed time even while a synchronous subprocess blocks."""
    context = _COMMAND_CONTEXT.get()
    if context is None or context.quiet or context.json_output:
        yield
        return
    safe_label = ui_text(str(redact(label)))
    context.operations.append((safe_label, time.monotonic()))
    try:
        _resume_operation_live(context)
        if context.operation_live is None and context.live is None:
            report_progress(safe_label)
        yield
    finally:
        context.operations.pop()
        if context.operations:
            _resume_operation_live(context)
        else:
            _stop_operation_live(context)


@contextmanager
def suspend_progress() -> Iterator[None]:
    """Do not redraw or animate over a credential or confirmation prompt."""
    context = _COMMAND_CONTEXT.get()
    if context is None:
        yield
        return
    previous = None in context.progress_pause_keys
    context.prompt_depth += 1
    pause_live_progress(True)
    try:
        yield
    finally:
        context.prompt_depth -= 1
        pause_live_progress(previous)


def _command_activity(command: str) -> str:
    ru = current_ui_language() == "ru"
    labels = {
        "publish": ("Подготовка публикации", "Preparing publication"),
        "submit": ("Подготовка и отправка данных", "Preparing and submitting data"),
        "import": ("Импорт и обработка пака", "Importing and processing pack"),
        "describe": ("Подготовка анализа", "Preparing analysis"),
        "add": ("Обработка пака", "Processing pack"),
        "resume": ("Восстановление сохранённой обработки", "Resuming saved processing"),
        "gallery": ("Подготовка галереи", "Preparing gallery"),
        "validate": ("Проверка данных", "Validating data"),
        "doctor": ("Проверка подключений и инструментов", "Checking connections and tools"),
        "build-index": ("Сборка поискового индекса", "Building search index"),
        "update": ("Проверка обновлений пака", "Checking pack updates"),
        "init": ("Подготовка настроек и инструментов", "Preparing settings and tools"),
    }
    return labels.get(command, ("Выполнение " + command, "Running " + command))[0 if ru else 1]


def report_run_id(run_id: str) -> None:
    """Bind the durable checkpoint ID to the enclosing synchronous CLI invocation."""

    context = _COMMAND_CONTEXT.get()
    if context is not None:
        # The shared holder survives asyncio.run's copied ContextVar context.
        context.run_id = run_id


def begin_pack_queue(sources: list[str]) -> None:
    context = _COMMAND_CONTEXT.get()
    if context is None or context.quiet or context.json_output:
        return
    console = Console(stderr=True, no_color=context.no_color)
    if not console.is_terminal or console.is_dumb_terminal:
        return
    if context.pack_queue is None:
        context.pack_queue = SHARED.get() or PackQueue()
    context.pack_queue.register(sources)
    update_live_progress(context.pack_queue, key="packs")


def report_pack_stage(source: str, stage: str) -> None:
    context = _COMMAND_CONTEXT.get()
    if (
        context is not None
        and context.pack_queue is not None
        and source in context.pack_queue.stages
    ):
        context.pack_queue.stages[source] = stage
        update_live_progress(context.pack_queue, key="packs")


def report_progress(message: str, *, verbose: bool = False) -> None:
    """Human diagnostics always go to stderr, leaving machine stdout untouched."""

    context = _COMMAND_CONTEXT.get()
    if context is None or context.quiet or (verbose and not context.verbose):
        return
    if (
        context.pack_queue is not None
        and not context.verbose
        and not message.startswith(("Derived contact-sheet PNG", "Подготовленные PNG"))
    ):
        context.progress_note = ui_text(str(redact(message)))
        if not context.progress_paused:
            _refresh_live(context)
        return
    if context.progress_paused:
        safe_message = ui_text(str(redact(message)))
        if message.startswith(("AI batch ", "AI-пачка ")):
            context.progress_note = safe_message
        else:
            context.pending_progress.append(Text(safe_message))
        return
    if context.operation_live is not None and not context.progress_paused:
        context.operation_live.console.print(ui_text(str(redact(message))), markup=False)
        return
    if context.live is not None and not context.progress_paused:
        safe_message = ui_text(str(redact(message)))
        if message.startswith(("AI batch ", "AI-пачка ")):
            context.progress_note = safe_message
            _refresh_live(context)
        else:
            # Keep disclosures and exceptional diagnostics in scrollback.
            context.live.console.print(safe_message, markup=False)
        return
    console = Console(stderr=True, no_color=context.no_color)
    prefix = f"[{context.run_id}] " if context.verbose or not console.is_terminal else ""
    console.print(f"{prefix}{ui_text(str(redact(message)))}", markup=False)


def _refresh_live(context: _CommandContext) -> None:
    if context.pack_queue is not None:
        now = time.monotonic()
        if now - context.last_pack_refresh < 0.15:
            return
        context.last_pack_refresh = now
    if context.live is not None and context.progress_view is not None:
        context.live.update(
            Group(
                context.progress_view,
                Text(context.progress_note, no_wrap=True, overflow="ellipsis"),
            ),
            refresh=True,
        )


def update_pack_progress(batch: object) -> bool:
    """Update counters without constructing a hidden per-item Rich table."""
    context = _COMMAND_CONTEXT.get()
    if context is None or context.pack_queue is None:
        return False
    owner = getattr(batch, "pack", None)
    if owner is not None:
        context.pack_queue.batches[batch] = owner
    if context.progress_paused:
        return True
    if context.live is None:
        return update_live_progress(context.pack_queue, key="packs")
    _refresh_live(context)
    return True


def update_live_progress(view: RenderableType, *, key: object = None) -> bool:
    """Use one terminal panel; retain ordinary logs for pipes and JSON callers."""
    context = _COMMAND_CONTEXT.get()
    if context is None or context.quiet or context.json_output:
        return False
    console = Console(stderr=True, no_color=context.no_color)
    if not console.is_terminal or console.is_dumb_terminal:
        return False
    if context.pack_queue is not None:
        owner = getattr(key, "pack", None)
        if owner is not None:
            context.pack_queue.batches[key] = owner
        context.progress_views = {"packs": context.pack_queue}
    else:
        context.progress_views[key] = view
    context.progress_view = _ProgressDisplay(context)
    if context.progress_paused:
        return True
    _stop_operation_live(context)
    if context.live is None:
        context.live = Live(
            context.progress_view,
            console=console,
            auto_refresh=False,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        context.live.start(refresh=True)
    _refresh_live(context)
    return True


def pause_live_progress(paused: bool, *, key: object = None) -> None:
    """Keep confirmation prompts visible and free of terminal redraws."""
    context = _COMMAND_CONTEXT.get()
    if context is None:
        return
    if paused:
        context.progress_pause_keys.add(key)
    else:
        context.progress_pause_keys.discard(key)
    context.progress_paused = bool(context.progress_pause_keys) or context.prompt_depth > 0
    if context.progress_paused:
        _stop_operation_live(context)
    if context.progress_paused and context.live is not None:
        context.live.stop()
        context.live = None
    if not context.progress_paused:
        if context.pending_progress:
            console = Console(stderr=True, no_color=context.no_color)
            for message in context.pending_progress:
                console.print(message)
            context.pending_progress.clear()
        _resume_operation_live(context)


def finish_live_progress(*, key: object = None) -> None:
    """Finish one active batch, or flush every panel at command shutdown."""
    context = _COMMAND_CONTEXT.get()
    if context is None:
        return
    if key is not None and context.pack_queue is not None:
        context.pack_queue.batches.pop(key, None)
        context.progress_pause_keys.discard(key)
        context.progress_paused = bool(context.progress_pause_keys) or context.prompt_depth > 0
        _refresh_live(context)
        return
    if key is not None:
        finished = context.progress_views.pop(key, None)
        compact = getattr(key, "compact_view", None)
        if finished is not None and callable(compact):
            finished = compact()
        context.progress_pause_keys.discard(key)
        context.progress_paused = bool(context.progress_pause_keys) or context.prompt_depth > 0
        if context.progress_views:
            context.progress_view = _ProgressDisplay(context)
            if finished is not None:
                console = (
                    context.live.console
                    if context.live is not None
                    else Console(stderr=True, no_color=context.no_color)
                )
                if context.progress_paused:
                    context.pending_progress.append(finished)
                else:
                    console.print(finished)
            if context.live is not None:
                _refresh_live(context)
            return
        context.progress_view = finished
    if context.live is not None:
        context.live.stop()
        context.live = None
    if context.progress_view is not None and not (
        context.command == "import" and SHARED.get() is not None
    ):
        if context.progress_paused:
            context.pending_progress.append(context.progress_view)
        else:
            Console(stderr=True, no_color=context.no_color).print(context.progress_view)
    context.progress_views.clear()
    context.progress_view = None
    context.progress_note = ""
    if key is None:
        context.progress_pause_keys.clear()
    context.progress_paused = bool(context.progress_pause_keys) or context.prompt_depth > 0
    _resume_operation_live(context)


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


_RESULT_COLLECTOR: ContextVar[list[CommandResult] | None] = ContextVar(
    "mojilex_result_collector", default=None
)


@contextmanager
def capture_command_results() -> Iterator[list[CommandResult]]:
    """Pass exact completed command identities to an enclosing interactive flow."""
    results: list[CommandResult] = []
    token = _RESULT_COLLECTOR.set(results)
    try:
        yield results
    finally:
        _RESULT_COLLECTOR.reset(token)


def new_run_id() -> str:
    """Return a dependency-free, time-sortable 26-character ULID."""

    value = (int(time.time_ns() // 1_000_000) << 80) | int.from_bytes(os.urandom(10), "big")
    encoded: list[str] = []
    for _ in range(26):
        encoded.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(encoded))


def require_confirmation(
    message: str, *, yes: bool, non_interactive: bool, json_output: bool, default: bool = False
) -> None:
    if yes:
        return
    if non_interactive or json_output or _MACHINE_JSON_MODE.get():
        raise CommandError(
            "CONFIG_INVALID",
            "This sensitive operation requires explicit confirmation.",
            hint="Rerun with --yes after reviewing the exact target.",
        )
    with suspend_progress():
        confirmed = ui_confirm(message, default=default)
    if not confirmed:
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
    if isinstance(exc, ValidationError):
        return _safe_model_validation_error(exc, debug=debug)
    if isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError, typer.Abort)):
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
        "SYSTEM_DEPENDENCY_MISSING": (
            "Run mojilex doctor and install the backend it reports. TGS setup: "
            "https://github.com/MojiLex/mojilex-cli/blob/main/docs/media-prerequisites.md"
        ),
        "VALIDATION_FAILED": "Run mojilex validate --strict and fix every reported issue.",
        "AI_OUTPUT_INVALID": (
            "The model response failed validation. Saved results are retained; "
            "resume with the same run ID. If it repeats, report the validation code and field path."
        ),
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


def _safe_model_validation_error(exc: ValidationError, *, debug: bool) -> StructuredError:
    """Pydantic messages, inputs, mapping keys, and custom error codes are untrusted."""
    from mojilex_cli.config import models as config_models
    from mojilex_cli.domain import models as domain_models

    schema: dict[str, Any] = {}
    code = "VALIDATION_FAILED"
    for module in (domain_models, config_models):
        model = getattr(module, exc.title, None)
        if (
            isinstance(model, type)
            and issubclass(model, BaseModel)
            and model.__module__ == module.__name__
        ):
            schema = model.model_json_schema()
            if module is config_models:
                code = "CONFIG_INVALID"
            break
    known_codes = frozenset(get_args(ErrorType))
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    summaries = []
    for error in errors[:4]:
        node = schema
        path = "$"
        for part in error["loc"]:
            if "$ref" in node:
                node = schema.get("$defs", {}).get(node["$ref"].rsplit("/", 1)[-1], {})
            alternatives = [entry for entry in node.get("anyOf", []) if entry.get("type") != "null"]
            if len(alternatives) == 1:
                node = alternatives[0]
                if "$ref" in node:
                    node = schema.get("$defs", {}).get(node["$ref"].rsplit("/", 1)[-1], {})
            if type(part) is int and "items" in node:
                node = node["items"]
                path += "[]"
            elif isinstance(part, str) and part in node.get("properties", {}):
                node = node["properties"][part]
                path += "." + part
            else:
                path += ".<unknown-field>"
                break
        error_code = error["type"] if error["type"] in known_codes else "schema_error"
        summary = f"{path} ({error_code})"
        if summary not in summaries:
            summaries.append(summary)
    if len(errors) > 4:
        summaries.append("additional errors omitted")
    return StructuredError(
        code=code,
        message="Model validation failed: " + "; ".join(summaries),
        retryable=False,
        hint=(
            "Check the command options and non-secret configuration."
            if code == "CONFIG_INVALID"
            else (
                "Report the validation code and field path with the run ID; "
                "saved results are retained."
            )
        ),
        details={"exception_type": "ValidationError"} if debug else None,
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
    context = _CommandContext(
        run_id,
        quiet=quiet,
        verbose=verbose,
        no_color=no_color,
        json_output=effective_json,
        command=command,
    )
    context_token = _COMMAND_CONTEXT.set(context)
    try:
        if command in {
            "add",
            "import",
            "describe",
            "publish",
            "submit",
            "resume",
            "update",
            "validate",
            "build-index",
            "gallery",
            "doctor",
            "dedupe scan",
            "benchmark-model",
            "benchmark-dedupe",
            "cache prune",
            "init",
        }:
            with operation_progress(_command_activity(command)):
                command_result = action()
        else:
            command_result = action()
        collector = _RESULT_COLLECTOR.get()
        if collector is not None:
            collector.append(command_result)
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
            if isinstance(exc, ValidationError):
                # Even format_exception's final line contains the complete input.
                # Keep frame locations without source snippets, locals, or causes.
                locations = (
                    f'  File "{frame.filename}", line {frame.lineno}, in {frame.name}\n'
                    for frame in traceback.extract_tb(exc.__traceback__)
                )
                formatted = "Traceback (validation input omitted):\n" + "".join(locations)
            else:
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
        finish_live_progress()
        _stop_operation_live(context)
        _COMMAND_CONTEXT.reset(context_token)

    if effective_json:
        typer.echo(envelope.to_json())
    elif not quiet or not envelope.ok:
        _render_human(envelope, no_color=no_color, detailed=debug or verbose)
    _ENVELOPE_EMITTED.set(True)
    if envelope.exit_code:
        raise typer.Exit(code=int(envelope.exit_code))


def _render_human(
    envelope: OutputEnvelope, *, no_color: bool = False, detailed: bool = False
) -> None:
    if (
        envelope.ok
        and envelope.command == "import"
        and SHARED.get() is not None
        and envelope.run_id
        and not envelope.warnings
    ):
        return
    console = Console(stderr=not envelope.ok, no_color=no_color)
    ru = current_ui_language() == "ru"
    if envelope.ok:
        raw_label = str(envelope.status).replace("RunStatus.", "").lower()
        label = ui_text(raw_label)
        console.print(f"[green]MojiLex {envelope.command}: {label}[/green]")
        safe_result = redact(envelope.result)
        rendered_pack = isinstance(safe_result, Mapping) and _render_pack_result(
            console, envelope.command, safe_result
        )
        if isinstance(safe_result, Mapping) and safe_result and not rendered_pack:
            table = Table(show_header=False, box=None, pad_edge=False)
            for key, value in safe_result.items():
                table.add_row(
                    Text(ui_text(str(key).replace("_", " "))),
                    Text(ui_text(str(value))),
                )
            console.print(table)
        publication = redact(envelope.publication)
        if isinstance(publication, Mapping):
            if publication.get("mode") in {"local", "staging"}:
                console.print(
                    Text(
                        "Результат сохранён локально. Эта команда не отправляла его на GitHub."
                        if ru
                        else "Results saved locally. This command did not send them to GitHub."
                    )
                )
            elif publication.get("mode") == "pr" and any(
                publication.get(key) for key in ("pr_url", "pull_request_url", "url")
            ):
                console.print(
                    Text(
                        "Изменения отправлены в pull request. Слияние на GitHub — отдельный шаг."
                        if ru
                        else "Changes sent as a pull request. Merging on GitHub is a separate step."
                    )
                )
            for key in ("pr_url", "pull_request_url", "url"):
                if isinstance(publication.get(key), str):
                    console.print(str(publication[key]), markup=False)
        for warning in envelope.warnings:
            safe_warning = redact(warning)
            if isinstance(safe_warning, Mapping) and isinstance(safe_warning.get("message"), str):
                warning_text = ui_text(str(safe_warning["message"]))
                if safe_warning.get("code"):
                    warning_text = f"{safe_warning['code']}: {warning_text}"
            else:
                warning_text = ui_text(str(safe_warning))
            console.print(
                f"{ui_text('Warning')}: {warning_text}",
                style="yellow",
                markup=False,
            )
        return
    for error in envelope.errors:
        safe_error = error.as_dict()
        message = ui_text(str(safe_error["message"]))
        hint = ui_text(str(safe_error["hint"]))
        if error.code == "AI_OUTPUT_INVALID" and not detailed:
            message = (
                "Не удалось получить корректное описание от модели."
                if ru
                else "The model could not produce a valid description."
            )
            hint = (
                "Откройте список паков: mojilex list. Продолжите нужный пак: mojilex resume ИМЯ."
                if ru
                else "Open your packs: mojilex list. Continue a pack: mojilex resume NAME."
            )
        console.print(f"{error.code}: {message}", style="red", markup=False)
        console.print(f"{ui_text('Hint')}: {hint}", markup=False)
    if detailed or not console.is_terminal:
        console.print(f"{ui_text('Run ID')}: {envelope.run_id}", markup=False)
    else:
        console.print(
            Text(
                "Подробности ошибки: повторите команду с --debug."
                if ru
                else "Error details: rerun the command with --debug."
            )
        )


def _render_pack_result(console: Console, command: str, result: Mapping[str, Any]) -> bool:
    """Render pack data as readable text, never Python dicts or Rich markup."""
    ru = current_ui_language() == "ru"
    if result.get("view") == "settings":
        table = Table()
        for label in (
            ("Настройка", "Значение", "Что означает") if ru else ("Setting", "Value", "Meaning")
        ):
            table.add_column(label)
        for setting in result.get("settings", []):
            value = setting["value"]
            display = str(value)
            if isinstance(value, bool):
                display = (
                    ("Включено" if value else "Выключено")
                    if ru
                    else ("Enabled" if value else "Disabled")
                )
            elif value is None:
                if setting.get("key") == "max_ai_requests":
                    display = "Без лимита" if ru else "Unlimited"
                else:
                    display = "не задан" if ru else "not set"
            elif value == "":
                display = "не настроено" if ru else "not configured"
            table.add_row(
                Text(str(setting["label"])),
                Text(display),
                Text(str(setting["description"])),
            )
        console.print(table)
        for note in result.get("notes", []):
            console.print(Text(str(note)))
        return True
    if result.get("view") == "setting_updated":
        value = result["value"]
        if result.get("key") == "max_ai_requests" and value is None:
            value = "Без лимита" if ru else "Unlimited"
        console.print(Text(f"{result['label']}: {value}"))
        console.print(Text("Настройка сохранена." if ru else "Setting saved."))
        return True
    if "gallery_path" in result:
        opened = bool(result.get("browser_opened"))
        console.print(
            Text(
                ("Галерея открыта в браузере." if ru else "Gallery opened in your browser.")
                if opened
                else (
                    "Галерея сохранена. Откройте файл:" if ru else "Gallery saved. Open this file:"
                )
            )
        )
        counts = result.get("counts", {})
        if "ready" in counts:
            console.print(Text(f"{'Описаний' if ru else 'Descriptions'}: {counts['ready']}"))
        if not opened:
            console.print(Text(str(result["gallery_path"])))
        return True
    if command in {"add", "import", "describe", "resume"} and any(
        key in result
        for key in ("sources_processed", "collections_imported", "message", "analysis_selectors")
    ):
        if result.get("message"):
            console.print(Text(ui_text(str(result["message"]))))
        if "analysis_selectors" in result:
            count = len(result["analysis_selectors"])
            console.print(
                Text(f"{'Паков для анализа' if ru else 'Packs queued for analysis'}: {count}")
            )
        for key, label in (
            ("collections_imported", "Загружено паков" if ru else "Packs downloaded"),
            ("sources_processed", "Обработано паков" if ru else "Packs processed"),
            ("items_added", "Добавлено эмодзи" if ru else "Emojis added"),
            ("items_updated", "Обновлено эмодзи" if ru else "Emojis updated"),
            ("items_unchanged", "Уже готовы" if ru else "Already up to date"),
            ("ai_requests", "Использовано запросов к ИИ" if ru else "AI requests used"),
            ("ai_cache_hits", "Взято из кеша" if ru else "Reused cached results"),
        ):
            if key in result:
                console.print(Text(f"{label}: {result[key]}"))
        if result.get("next"):
            console.print(Text(str(result["next"])))
        elif isinstance(result.get("pack_name"), str) and result["pack_name"]:
            name = result["pack_name"]
            selector = (
                name
                if all(char.isalnum() or char == "_" for char in name)
                else ("'" + name.replace("'", "''") + "'")
            )
            if command == "import":
                console.print(
                    Text(
                        f"{'Создать описания' if ru else 'Create descriptions'}: "
                        f"mojilex describe {selector}"
                    )
                )
            else:
                console.print(
                    Text(f"{'Посмотреть' if ru else 'View results'}: mojilex show {selector}")
                )
                console.print(
                    Text(
                        f"{'Отправить на GitHub' if ru else 'Send to GitHub'}: "
                        f"mojilex publish {selector}"
                    )
                )
        else:
            console.print(
                Text(
                    "Паки и следующие действия: mojilex list"
                    if ru
                    else "Packs and next steps: mojilex list"
                )
            )
        return True
    if command == "publish":
        if result.get("validated"):
            console.print(ui_text("Validation passed."))
        if isinstance(result.get("changed_paths"), list):
            console.print(Text(f"{ui_text('Changed data files')}: {len(result['changed_paths'])}"))
        return True
    if command == "list" and isinstance(result.get("packs"), list):
        packs = result["packs"]
        if not packs:
            console.print(ui_text("No saved packs. Start with mojilex import PACK_URL."))
            return True
        table = Table()
        for heading in ("Pack", "Status", "Descriptions", "Updated"):
            table.add_column(ui_text(heading))
        names = [name.casefold() for pack in packs for name in pack["names"]]
        ambiguous = len(names) != len(set(names))
        unfinished = any(pack.get("latest_unfinished") for pack in packs)
        if unfinished:
            table.add_column(ui_text("Unfinished run"))
        if ambiguous:
            table.add_column("Run ID")
        for pack in packs:
            cells = [
                Text(", ".join(pack["names"])),
                Text(ui_text(str(pack["status"]))),
                Text(f"{pack['ai_ready']}/{pack['items']}"),
                Text(str(pack["updated_at"]).replace("T", " ")[:19]),
            ]
            if unfinished:
                active = pack.get("latest_unfinished")
                cells.append(Text(f"{active['ai_ready']}/{active['items']}" if active else "—"))
            if ambiguous:
                cells.append(Text(pack["run_id"]))
            table.add_row(*cells)
        console.print(table)
        console.print(ui_text("Open a pack: mojilex show NAME"), markup=False)
        if unfinished:
            console.print(ui_text("Continue unfinished work: mojilex resume NAME"), markup=False)
        return True
    if command not in {"show", "review"} or "pack" not in result:
        return False
    pack = result["pack"]
    counts = result["counts"]
    console.print(
        Text(
            f"{', '.join(pack['names'])} — {counts['ready']}/{pack['items']} "
            f"{ui_text('descriptions')}"
        ),
    )
    if command == "review":
        console.print(ui_text("Optional viewing. Content warnings do not require approval."))
    for index, item in enumerate(result["items"], 1):
        console.print()
        console.print(Text(f"{index}. {item['native_id']}", style="bold"))
        for language, description in item["descriptions"].items():
            console.print(Text(f"{language.upper()}: {description['text']}"))
            if description.get("motion"):
                console.print(Text(f"  {ui_text('Motion')}: {description['motion']}"))
            if description.get("usage"):
                console.print(
                    Text(f"  {ui_text('Usage examples')}: " + "; ".join(description["usage"]))
                )
        if item.get("semantic_tags"):
            console.print(Text(f"{ui_text('Tags')}: " + ", ".join(item["semantic_tags"])))
        content = item["content"]
        console.print(Text(f"{ui_text('Content rating')}: {content['rating']}"))
        if content["warnings"]:
            console.print(Text(f"{ui_text('Content warnings')}: " + ", ".join(content["warnings"])))
        facets = item.get("facets", {})
        for facet_name in ("content_types", "styles", "suggested_uses", "uncertainties"):
            if facets.get(facet_name):
                console.print(Text(f"{ui_text(facet_name)}: " + ", ".join(facets[facet_name])))
        for fragment in facets.get("text_content", {}).get("items", []):
            console.print(Text(f"{ui_text('Text in emoji')}: {fragment.get('value', '')}"))
    if any(counts.get(key) for key in ("pending", "missing", "invalid")):
        console.print(
            Text(
                f"{ui_text('Pending')}: {counts.get('pending', 0)}; "
                f"{ui_text('Unavailable')}: {counts.get('missing', 0)}; "
                f"{ui_text('Invalid')}: {counts.get('invalid', 0)}"
            )
        )
    return True


def is_usage_error(exc: BaseException) -> bool:
    """Recognize Click/Typer parse errors without depending on a private Click package."""

    return getattr(exc, "exit_code", None) == 2 and callable(getattr(exc, "format_message", None))
