"""Typer adapters and closed output envelopes for SPEC-003 offline reads."""

from __future__ import annotations

import hashlib
import json
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypeVar, cast

import rfc8785
import typer

from mojilex_cli.commands.runtime import (
    CommandError,
    machine_output_requested,
    mark_machine_envelope_emitted,
    new_run_id,
    structured_exception,
)
from mojilex_cli.i18n import text as ui_text
from mojilex_cli.output.models import RunStatus, StructuredError, redact
from mojilex_cli.read.service import SnapshotReader
from mojilex_cli.read.snapshot import (
    LoadedSnapshot,
    load_snapshot,
    parse_bounded_json,
    validate_embedded_schema_instance,
)

_T = TypeVar("_T")
_MAX_REQUEST_BYTES = 1024 * 1024
_VIEWS = {"canonical", "search", "agent"}
_SORTS = {"relevance", "emoji-id"}
_RATINGS = {"general", "sensitive", "adult"}
_RIGHTS = {"allowed", "restricted", "unknown", "withdrawn"}
_REVIEWS = {"approved", "unreviewed", "changes_requested", "rejected", "qualified-ai"}
_AVAILABILITY = {"active", "unknown", "unavailable", "private", "deleted"}
_YES_NO_ANY = {"yes", "no", "any"}
_COLOR_FAMILIES = {
    "black",
    "white",
    "gray",
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "pink",
    "brown",
    "beige",
}
_FILTER_KEYS = {
    "platform",
    "collection",
    "availability",
    "review",
    "rating",
    "exclude_warning",
    "animated",
    "media_kind",
    "color_behavior",
    "color_family",
    "contains_text",
    "content_type",
    "style",
    "suggested_use",
    "uncertainty",
    "concept",
    "rights",
    "require_no_warnings",
}

snapshot_app = typer.Typer(
    help="Verify or manage immutable release snapshots.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)


def get_cli(
    emoji_id: Annotated[str, typer.Argument()],
    snapshot_path: Annotated[Path, typer.Option("--snapshot")],
    manifest_sha256: Annotated[str | None, typer.Option("--manifest-sha256")] = None,
    view: Annotated[str, typer.Option("--view")] = "canonical",
    language: Annotated[str | None, typer.Option("--language")] = None,
    include_sensitive: Annotated[bool, typer.Option("--include-sensitive")] = False,
    allow_unverified: Annotated[bool, typer.Option("--allow-unverified")] = False,
    offline: Annotated[bool, typer.Option("--offline/--no-offline")] = True,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    state: dict[str, LoadedSnapshot] = {}

    def action() -> ReadCommandResult:
        selected_view = _enum(view, _VIEWS, "--view")
        if selected_view == "canonical" and language is not None:
            raise CommandError(
                "OPTION_CONFLICT",
                "--language is forbidden with canonical view.",
                hint="Remove --language or select search/agent view.",
            )
        snapshot, reader, warnings = _open(
            state, snapshot_path, manifest_sha256, allow_unverified, offline
        )
        result = reader.get(
            emoji_id, view=selected_view, language=language, include_sensitive=include_sensitive
        )
        if include_sensitive:
            warnings.append(
                {
                    "code": "SENSITIVE_CONTENT_INCLUDED",
                    "message": "Explicit diagnostic content override was used.",
                }
            )
        return ReadCommandResult(
            result=result,
            dataset=snapshot.dataset_context(),
            arguments={
                "emoji_id": emoji_id,
                "view": selected_view,
                "language": language,
                "include_sensitive": include_sensitive,
            },
            view=selected_view,
            warnings=warnings,
        )

    execute_read(
        "get",
        action,
        json_output=json_output,
        quiet=quiet,
        debug=debug,
        dataset_provider=_dataset_provider(state),
    )


def get_collection_cli(
    collection_id: Annotated[str, typer.Argument()],
    snapshot_path: Annotated[Path, typer.Option("--snapshot")],
    manifest_sha256: Annotated[str | None, typer.Option("--manifest-sha256")] = None,
    include_history: Annotated[bool, typer.Option("--include-history")] = False,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    allow_unverified: Annotated[bool, typer.Option("--allow-unverified")] = False,
    offline: Annotated[bool, typer.Option("--offline/--no-offline")] = True,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    jsonl_output: Annotated[bool, typer.Option("--jsonl")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    state: dict[str, LoadedSnapshot] = {}
    request_context: dict[str, Any] = {}

    def action() -> ReadCommandResult:
        selected_limit = _limit(limit)
        request_context.update(
            {
                "collection_id": collection_id,
                "include_history": include_history,
                "limit": selected_limit,
                "cursor": cursor,
            }
        )
        snapshot, reader, warnings = _open(
            state, snapshot_path, manifest_sha256, allow_unverified, offline
        )
        result = reader.get_collection(
            collection_id, include_history=include_history, limit=selected_limit, cursor=cursor
        )
        if include_history:
            warnings.append(
                {
                    "code": "HISTORY_INCLUDED",
                    "message": (
                        "Permitted diagnostic history was requested; safe_eligible remains false."
                    ),
                }
            )
        return ReadCommandResult(
            result=result,
            dataset=snapshot.dataset_context(),
            arguments=dict(request_context),
            view="canonical",
            warnings=warnings,
            stream_items=cast(list[Any], result["memberships"]),
        )

    execute_read(
        "get-collection",
        action,
        json_output=json_output,
        jsonl_output=jsonl_output,
        quiet=quiet,
        debug=debug,
        dataset_provider=_dataset_provider(state),
        request_context_provider=lambda: (dict(request_context), "canonical"),
    )


def resolve_cli(
    platform: Annotated[str, typer.Option("--platform")],
    namespace: Annotated[str, typer.Option("--namespace")],
    scope: Annotated[str, typer.Option("--scope")],
    native_id: Annotated[str, typer.Option("--native-id")],
    snapshot_path: Annotated[Path, typer.Option("--snapshot")],
    manifest_sha256: Annotated[str | None, typer.Option("--manifest-sha256")] = None,
    identity_epoch: Annotated[int | None, typer.Option("--identity-epoch")] = None,
    as_of: Annotated[str | None, typer.Option("--as-of")] = None,
    include_history: Annotated[bool, typer.Option("--include-history")] = False,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    allow_unverified: Annotated[bool, typer.Option("--allow-unverified")] = False,
    offline: Annotated[bool, typer.Option("--offline/--no-offline")] = True,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    state: dict[str, LoadedSnapshot] = {}

    def action() -> ReadCommandResult:
        if identity_epoch is not None and as_of is not None:
            raise CommandError(
                "OPTION_CONFLICT",
                "--identity-epoch and --as-of are mutually exclusive.",
                hint="Select one identity-time constraint.",
            )
        if identity_epoch is not None and identity_epoch < 0:
            raise CommandError(
                "QUERY_INVALID",
                "Identity epoch must be non-negative.",
                hint="Pass a decimal epoch starting at 0.",
            )
        for name, value in {
            "platform": platform,
            "namespace": namespace,
            "scope": scope,
            "native-id": native_id,
        }.items():
            if not value or len(value.encode("utf-8")) > 256 or "\x00" in value:
                raise CommandError(
                    "QUERY_INVALID",
                    f"--{name} is empty or exceeds the locator limit.",
                    hint="Pass the exact bounded native locator component.",
                )
        selected_limit = _limit(limit)
        snapshot, reader, warnings = _open(
            state, snapshot_path, manifest_sha256, allow_unverified, offline
        )
        result = reader.resolve(
            platform=platform,
            namespace=namespace,
            scope=scope,
            native_id=native_id,
            identity_epoch=identity_epoch,
            as_of=as_of,
            include_history=include_history,
            limit=selected_limit,
            cursor=cursor,
        )
        if include_history:
            warnings.append(
                {
                    "code": "HISTORY_INCLUDED",
                    "message": (
                        "Permitted identity history was requested; no target body was disclosed."
                    ),
                }
            )
        return ReadCommandResult(
            result=result,
            dataset=snapshot.dataset_context(),
            arguments={
                "platform": platform,
                "namespace": namespace,
                "scope": scope,
                "native_id": native_id,
                "identity_epoch": identity_epoch,
                "as_of": as_of,
                "include_history": include_history,
                "limit": selected_limit,
                "cursor": cursor,
            },
            view="canonical",
            warnings=warnings,
        )

    execute_read(
        "resolve",
        action,
        json_output=json_output,
        quiet=quiet,
        debug=debug,
        dataset_provider=_dataset_provider(state),
    )


def similar_cli(
    emoji_id: Annotated[str, typer.Argument()],
    snapshot_path: Annotated[Path, typer.Option("--snapshot")],
    manifest_sha256: Annotated[str | None, typer.Option("--manifest-sha256")] = None,
    view: Annotated[str, typer.Option("--view")] = "agent",
    language: Annotated[str | None, typer.Option("--language")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    allow_unverified: Annotated[bool, typer.Option("--allow-unverified")] = False,
    offline: Annotated[bool, typer.Option("--offline/--no-offline")] = True,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    jsonl_output: Annotated[bool, typer.Option("--jsonl")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    state: dict[str, LoadedSnapshot] = {}
    request_context: dict[str, Any] = {}

    def action() -> ReadCommandResult:
        selected_view = _enum(view, _VIEWS, "--view")
        if selected_view == "canonical" and language is not None:
            raise CommandError(
                "OPTION_CONFLICT",
                "--language is forbidden with canonical view.",
                hint="Remove --language or select search/agent view.",
            )
        selected_limit = _limit(limit)
        request_context.update(
            {
                "emoji_id": emoji_id,
                "view": selected_view,
                "language": language,
                "limit": selected_limit,
                "cursor": cursor,
                "policy": "confirmed-only-v1",
            }
        )
        snapshot, reader, warnings = _open(
            state, snapshot_path, manifest_sha256, allow_unverified, offline
        )
        result = reader.similar(
            emoji_id, view=selected_view, language=language, limit=selected_limit, cursor=cursor
        )
        return ReadCommandResult(
            result=result,
            dataset=snapshot.dataset_context(),
            arguments=dict(request_context),
            view=selected_view,
            warnings=warnings,
            stream_items=cast(list[Any], result["items"]),
        )

    execute_read(
        "similar",
        action,
        json_output=json_output,
        jsonl_output=jsonl_output,
        quiet=quiet,
        debug=debug,
        dataset_provider=_dataset_provider(state),
        request_context_provider=lambda: (dict(request_context), str(request_context["view"])),
    )


def register_read_commands(app: typer.Typer) -> None:
    """Register read-only commands without coupling them to authoring modules."""

    app.add_typer(snapshot_app, name="snapshot")
    app.command("snapshots", help="List release snapshots from the configured catalog.")(
        snapshots_cli
    )
    app.command("search", help="Search a pinned local snapshot by text and filters.")(
        search_cli
    )
    app.command("get", help="Read one emoji record from a pinned local snapshot.")(get_cli)
    app.command(
        "get-collection",
        help="Read a collection and its members from a pinned snapshot.",
    )(get_collection_cli)
    app.command(
        "resolve",
        help="Resolve a platform-native reference to a canonical MojiLex identity.",
    )(resolve_cli)
    app.command("similar", help="Find related or duplicate candidates for an emoji.")(
        similar_cli
    )


@dataclass(slots=True)
class ReadCommandResult:
    result: dict[str, Any]
    dataset: dict[str, Any] | None
    arguments: dict[str, Any]
    view: str
    warnings: list[dict[str, Any]] = field(default_factory=list)
    stream_items: list[Any] | None = None
    status: RunStatus = RunStatus.SUCCEEDED


def _request_sha256(command: str, dataset: dict[str, Any] | None, arguments: dict[str, Any]) -> str:
    dataset_component: dict[str, Any] = (
        {"present": True, "value": dataset} if dataset is not None else {"present": False}
    )
    body = {
        "request_profile_id": "cli-request-v1",
        "command": command,
        "dataset": dataset_component,
        "arguments": arguments,
    }
    return hashlib.sha256(rfc8785.dumps(cast(Any, body))).hexdigest()


def execute_read(
    command: str,
    action: Callable[[], ReadCommandResult],
    *,
    json_output: bool,
    jsonl_output: bool = False,
    quiet: bool = False,
    debug: bool = False,
    dataset_provider: Callable[[], dict[str, Any] | None] | None = None,
    request_context_provider: Callable[[], tuple[dict[str, Any], str]] | None = None,
) -> None:
    """Emit one closed read envelope, or a complete framed JSONL stream."""

    effective_json = json_output or machine_output_requested()
    run_id = new_run_id()
    result: ReadCommandResult | None = None
    error: StructuredError | None = None
    if effective_json and jsonl_output:
        error = CommandError(
            "OPTION_CONFLICT",
            "--json and --jsonl are mutually exclusive.",
            hint="Select exactly one machine-output framing mode.",
        ).error
    else:
        try:
            candidate = action()
            _require_unmodified_result(candidate)
            result = candidate
        except BaseException as exc:
            if debug and not effective_json and not jsonl_output:
                typer.echo(str(redact("".join(traceback.format_exception(exc)))), err=True)
            error = structured_exception(exc, debug=debug)
    dataset = (
        result.dataset
        if result is not None
        else (dataset_provider() if dataset_provider is not None else None)
    )
    status: RunStatus = result.status if result is not None else RunStatus.FAILED
    warnings = result.warnings if result is not None else []
    errors = [] if error is None else [error]
    ok = error is None
    if jsonl_output:
        # A command-scoped JSONL export begins only after its exact snapshot
        # has been selected. Before that point there is no schema-valid
        # dataset component for metadata, so fail without emitting a partial
        # stream. Snapshot discovery is the sole dataset-absent JSONL branch.
        if dataset is None and command != "snapshots":
            if error is None:
                error = CommandError(
                    "INTERNAL_ERROR",
                    "A JSONL export completed without selecting its snapshot.",
                    hint="Report this output-contract failure.",
                ).error
            typer.echo(f"{error.code}: {error.message}", err=True)
            typer.echo(f"Hint: {error.hint}", err=True)
            raise typer.Exit(code=int(error.exit_code))
        if result is not None:
            arguments, view = result.arguments, result.view
        elif request_context_provider is not None:
            arguments, view = request_context_provider()
        else:
            arguments = {}
            view = "snapshot" if command == "snapshots" else "canonical"
        metadata: dict[str, Any] = {
            "schema_version": "1",
            "record_type": "metadata",
            "command": command,
            "view": view,
            "dataset": {"present": True, "value": dataset}
            if dataset is not None
            else {"present": False},
            "request_sha256": _request_sha256(command, dataset, arguments),
        }
        typer.echo(_compact(metadata, redact_values=not ok))
        stream_items = result.stream_items if result is not None and result.stream_items else []
        for ordinal, item in enumerate(stream_items):
            typer.echo(
                _compact(
                    {
                        "schema_version": "1",
                        "record_type": "item",
                        "ordinal": ordinal,
                        "item": item,
                    },
                    redact_values=False,
                )
            )
        next_cursor = (
            result.result.get("next_cursor", {"present": False})
            if result is not None
            else {"present": False}
        )
        typer.echo(
            _compact(
                {
                    "schema_version": "1",
                    "record_type": "summary",
                    "ok": ok,
                    "status": status.value,
                    "item_count": len(stream_items),
                    "next_cursor": next_cursor,
                    "warnings": warnings,
                    "errors": [item.as_dict() for item in errors],
                },
                redact_values=not ok,
            )
        )
    else:
        envelope: dict[str, Any] = {
            "schema_version": "1",
            "ok": ok,
            "command": command,
            "status": status.value,
            "run_id": run_id,
        }
        if dataset is not None:
            envelope["dataset"] = dataset
        if result is not None:
            envelope["result"] = result.result
        envelope["warnings"] = warnings
        envelope["errors"] = [item.as_dict() for item in errors]
        if effective_json:
            typer.echo(_compact(envelope, redact_values=not ok))
        elif not quiet or not ok:
            if ok:
                assert result is not None
                typer.echo(f"MojiLex {command}: {ui_text(status.value)}")
                typer.echo(json.dumps(result.result, ensure_ascii=True, indent=2))
            else:
                assert error is not None
                typer.echo(f"{error.code}: {ui_text(error.message)}", err=True)
                typer.echo(f"{ui_text('Hint')}: {ui_text(error.hint)}", err=True)
    mark_machine_envelope_emitted()
    if error is not None:
        raise typer.Exit(code=int(error.exit_code))


def _require_unmodified_result(result: ReadCommandResult) -> None:
    visible = {
        "dataset": result.dataset,
        "result": result.result,
        "warnings": result.warnings,
        "stream_items": result.stream_items,
    }
    if redact(visible) != visible:
        raise CommandError(
            "CONTENT_POLICY_BLOCKED",
            "The response cannot be emitted exactly without content-policy redaction.",
            hint="Narrow the request or inspect the trusted source locally.",
        )


def _compact(value: object, *, redact_values: bool = True) -> str:
    # Machine output must survive legacy Windows console code pages. JSON
    # escapes preserve the exact Unicode scalar values without relying on the
    # host stdout encoding.
    visible = redact(value) if redact_values else value
    return json.dumps(visible, ensure_ascii=True, separators=(",", ":"))


def _limit(value: int | None) -> int:
    result = 20 if value is None else value
    if isinstance(result, bool) or not 1 <= result <= 200:
        raise CommandError(
            "LIMIT_OUT_OF_RANGE",
            "Pagination limit must be between 1 and 200.",
            hint="Pass --limit with an integer from 1 through 200.",
        )
    return result


def _enum(value: str, allowed: set[str], option: str, code: str = "FILTER_INVALID") -> str:
    if value not in allowed:
        raise CommandError(
            code,
            f"Invalid {option} value: {value!r}.",
            hint=f"Choose one of: {', '.join(sorted(allowed))}.",
        )
    return value


def _list(value: object, field_name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            f"Structured request field filters.{field_name} must be an array of strings.",
            hint="Use the cli-search-request-v1 field types.",
        )
    return sorted(set(cast(list[str], value)), key=lambda item: item.encode("utf-8"))


def _validate_filters(filters: dict[str, Any]) -> dict[str, Any]:
    unknown = set(filters).difference(_FILTER_KEYS)
    if unknown:
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            f"Unknown structured search filter(s): {', '.join(sorted(unknown))}.",
            hint="Remove fields not defined by cli-search-request-v1.",
        )
    result: dict[str, Any] = {
        "platform": [],
        "collection": [],
        "availability": ["active"],
        "review": ["approved", "qualified-ai"],
        "rating": ["general"],
        "exclude_warning": [],
        "animated": "any",
        "media_kind": [],
        "color_behavior": [],
        "color_family": [],
        "contains_text": "any",
        "content_type": [],
        "style": [],
        "suggested_use": [],
        "uncertainty": [],
        "concept": [],
        "rights": ["allowed"],
        "require_no_warnings": True,
    }
    list_fields = {
        "platform",
        "collection",
        "availability",
        "review",
        "rating",
        "exclude_warning",
        "media_kind",
        "color_behavior",
        "color_family",
        "content_type",
        "style",
        "suggested_use",
        "uncertainty",
        "concept",
        "rights",
    }
    for key in list_fields:
        if key in filters:
            result[key] = _list(filters[key], key)
    for key, allowed in {
        "availability": _AVAILABILITY,
        "review": _REVIEWS,
        "rating": _RATINGS,
        "rights": _RIGHTS,
        "color_family": _COLOR_FAMILIES,
    }.items():
        for value in result.get(key, []):
            _enum(value, allowed, f"filters.{key}", "REQUEST_SCHEMA_INVALID")
    for key in ("animated", "contains_text"):
        value = filters.get(key, "any")
        if not isinstance(value, str):
            raise CommandError(
                "REQUEST_SCHEMA_INVALID",
                f"filters.{key} must be a string.",
                hint="Use yes, no, or any.",
            )
        result[key] = _enum(value, _YES_NO_ANY, f"filters.{key}", "REQUEST_SCHEMA_INVALID")
    no_warnings = filters.get("require_no_warnings", True)
    if not isinstance(no_warnings, bool):
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            "filters.require_no_warnings must be boolean.",
            hint="Use true or false.",
        )
    result["require_no_warnings"] = no_warnings
    return result


def _default_filters(
    *,
    platform: list[str] | None,
    collection: list[str] | None,
    availability: list[str] | None,
    review: list[str] | None,
    rating: list[str] | None,
    exclude_warning: list[str] | None,
    animated: str | None,
    media_kind: list[str] | None,
    color_behavior: list[str] | None,
    color_family: list[str] | None,
    contains_text: str | None,
    content_type: list[str] | None,
    style: list[str] | None,
    suggested_use: list[str] | None,
    uncertainty: list[str] | None,
    concept: list[str] | None,
    rights: list[str] | None,
    include_sensitive: bool,
    include_unreviewed: bool,
) -> dict[str, Any]:
    selected_review = list(review or ["approved", "qualified-ai"])
    if include_unreviewed and "unreviewed" not in selected_review:
        selected_review.append("unreviewed")
    raw: dict[str, Any] = {
        "platform": list(platform or []),
        "collection": list(collection or []),
        "availability": list(availability or ["active"]),
        "review": selected_review,
        "rating": list(
            rating or (["general", "sensitive", "adult"] if include_sensitive else ["general"])
        ),
        "exclude_warning": list(exclude_warning or []),
        "animated": animated or "any",
        "media_kind": list(media_kind or []),
        "color_behavior": list(color_behavior or []),
        "color_family": list(color_family or []),
        "contains_text": contains_text or "any",
        "content_type": list(content_type or []),
        "style": list(style or []),
        "suggested_use": list(suggested_use or []),
        "uncertainty": list(uncertainty or []),
        "concept": list(concept or []),
        "rights": list(rights or ["allowed"]),
        "require_no_warnings": not include_sensitive,
    }
    result = _validate_filters(raw)
    # CLI-option failures have FILTER_INVALID rather than request-schema semantics.
    for key, allowed in {
        "availability": _AVAILABILITY,
        "review": _REVIEWS,
        "rating": _RATINGS,
        "rights": _RIGHTS,
        "color_family": _COLOR_FAMILIES,
    }.items():
        for value in result.get(key, []):
            _enum(value, allowed, f"--{key.replace('_', '-')}")
    _enum(str(result["animated"]), _YES_NO_ANY, "--animated")
    _enum(str(result["contains_text"]), _YES_NO_ANY, "--contains-text")
    return result


def _read_request(path: Path) -> dict[str, Any]:
    try:
        if str(path) == "-":
            text = sys.stdin.read(_MAX_REQUEST_BYTES + 1)
            data = text.encode("utf-8")
        else:
            if path.stat().st_size > _MAX_REQUEST_BYTES:
                raise CommandError(
                    "REQUEST_SCHEMA_INVALID",
                    "Structured request exceeds the 1 MiB limit.",
                    hint="Use a bounded cli-search-request-v1 object.",
                )
            data = path.read_bytes()
    except CommandError:
        raise
    except OSError as exc:
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            f"Structured request cannot be read: {path}",
            hint="Pass a readable local JSON file or - for stdin.",
        ) from exc
    if len(data) > _MAX_REQUEST_BYTES:
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            "Structured request exceeds the 1 MiB limit.",
            hint="Use a bounded cli-search-request-v1 object.",
        )
    try:
        return parse_bounded_json(data, source="cli search request")
    except CommandError as exc:
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            str(exc),
            hint="Use a valid closed cli-search-request-v1 JSON object.",
        ) from exc


def _structured_search(
    path: Path,
) -> tuple[str, str | None, dict[str, Any], int, str | None, str, str]:
    body = _read_request(path)
    validate_embedded_schema_instance(
        body,
        "mlx://schemas/distribution/v1/search-request.schema.json",
        location="cli search request",
        invalid_code="REQUEST_SCHEMA_INVALID",
    )
    required = {"schema_version", "query", "language", "filters", "pagination", "sort", "view"}
    if set(body) != required:
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            (
                "Structured request must contain exactly schema_version, query, language, "
                "filters, pagination, sort, and view."
            ),
            hint="Use cli-search-request-v1 without unknown or omitted fields.",
        )
    if body.get("schema_version") != "1" or not isinstance(body.get("query"), str):
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            "Structured request schema_version or query is invalid.",
            hint="Use schema_version 1 and a string query.",
        )
    language = body.get("language")
    if language is not None and not isinstance(language, str):
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            "language must be a string or null.",
            hint="Use a BCP 47 tag or null for the en default.",
        )
    if not isinstance(body.get("filters"), dict) or not isinstance(body.get("pagination"), dict):
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            "filters and pagination must be objects.",
            hint="Use cli-search-request-v1 field types.",
        )
    pagination = cast(dict[str, Any], body["pagination"])
    if set(pagination) != {"limit", "cursor"}:
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            "pagination must contain exactly limit and cursor.",
            hint="Materialize both pagination defaults.",
        )
    cursor = pagination.get("cursor")
    if cursor is not None and not isinstance(cursor, str):
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            "pagination.cursor must be a string or null.",
            hint="Use the opaque prior cursor or null.",
        )
    limit_value = pagination.get("limit")
    if not isinstance(limit_value, int) or isinstance(limit_value, bool):
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            "pagination.limit must be an integer.",
            hint="Use an integer from 1 through 200.",
        )
    sort = body.get("sort")
    view = body.get("view")
    if not isinstance(sort, str) or not isinstance(view, str):
        raise CommandError(
            "REQUEST_SCHEMA_INVALID",
            "sort and view must be strings.",
            hint="Use documented enum values.",
        )
    return (
        str(body["query"]),
        language,
        _validate_filters(cast(dict[str, Any], body["filters"])),
        _limit(limit_value),
        cursor,
        _enum(sort, _SORTS, "sort", "REQUEST_SCHEMA_INVALID"),
        _enum(view, _VIEWS, "view", "REQUEST_SCHEMA_INVALID"),
    )


def _open(
    state: dict[str, LoadedSnapshot],
    selected: Path,
    manifest_sha256: str | None,
    allow_unverified: bool,
    offline: bool,
) -> tuple[LoadedSnapshot, SnapshotReader, list[dict[str, Any]]]:
    if not offline:
        raise CommandError(
            "OPTION_CONFLICT",
            (
                "This MVP reader accepts only explicit local snapshots and cannot enable "
                "network access."
            ),
            hint="Use --offline with --snapshot PATH.",
        )
    snapshot = load_snapshot(selected, expected_manifest_sha256=manifest_sha256)
    state["snapshot"] = snapshot
    warnings = snapshot.require_diagnostic_opt_in(allow_unverified)
    return snapshot, SnapshotReader(snapshot), warnings


def _dataset_provider(state: dict[str, LoadedSnapshot]) -> Callable[[], dict[str, Any] | None]:
    return lambda: state["snapshot"].dataset_context() if "snapshot" in state else None


def snapshots_cli(
    channel: Annotated[str, typer.Option("--channel")] = "stable",
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    jsonl_output: Annotated[bool, typer.Option("--jsonl")] = False,
) -> None:
    request_context = {
        "channel": channel,
        "limit": 20 if limit is None else limit,
        "cursor": cursor,
    }

    def action() -> ReadCommandResult:
        _limit(limit)
        raise CommandError(
            "MIRROR_UNAVAILABLE",
            (
                f"Snapshot discovery for channel {channel!r} is not configured in the "
                "strictly offline MVP."
            ),
            hint="Pass an explicit local --snapshot path to a read command.",
            details={"cursor_present": cursor is not None},
        )

    execute_read(
        "snapshots",
        action,
        json_output=json_output,
        jsonl_output=jsonl_output,
        request_context_provider=lambda: (request_context, "snapshot"),
    )


@snapshot_app.command(
    "pull", help="Fetch a named immutable snapshot from the configured mirror."
)
def snapshot_pull_cli(
    snapshot_id: Annotated[str, typer.Argument()] = "latest",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def action() -> ReadCommandResult:
        raise CommandError(
            "MIRROR_UNAVAILABLE",
            f"No signed release catalog or mirror is configured for {snapshot_id!r}.",
            hint="Download an immutable snapshot separately and verify its local path.",
        )

    execute_read("snapshot-pull", action, json_output=json_output)


@snapshot_app.command("update", help="Update to a named or latest immutable snapshot.")
def snapshot_update_cli(
    to: Annotated[str, typer.Option("--to")] = "latest",
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    def action() -> ReadCommandResult:
        raise CommandError(
            "MIRROR_UNAVAILABLE",
            f"Snapshot update to {to!r} is not configured without a signed catalog and mirror.",
            hint="Continue using the explicitly pinned local snapshot.",
        )

    execute_read("snapshot-update", action, json_output=json_output)


@snapshot_app.command(
    "verify", help="Verify a local snapshot, hashes, schemas, and trust metadata."
)
def snapshot_verify_cli(
    path: Annotated[Path, typer.Argument()],
    manifest_sha256: Annotated[str | None, typer.Option("--manifest-sha256")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    state: dict[str, LoadedSnapshot] = {}

    def action() -> ReadCommandResult:
        snapshot = load_snapshot(path, expected_manifest_sha256=manifest_sha256)
        state["snapshot"] = snapshot
        checks = [
            {"check_id": "manifest-jcs-and-schema", "status": "passed"},
            {"check_id": "artifact-resource-size-hash-schema", "status": "passed"},
            {"check_id": "semantic-roots-artifact-set-build-inputs", "status": "passed"},
            {"check_id": "sha256sums-and-physical-file-set", "status": "passed"},
        ]
        return ReadCommandResult(
            result={
                "overall_status": snapshot.release_verification_status,
                "manifest_sha256": snapshot.manifest_sha256,
                "checks": checks,
            },
            dataset=snapshot.dataset_context(),
            arguments={"path": str(path), "manifest_sha256": manifest_sha256},
            view="snapshot",
        )

    execute_read(
        "snapshot-verify",
        action,
        json_output=json_output,
        quiet=quiet,
        debug=debug,
        dataset_provider=_dataset_provider(state),
    )


def search_cli(
    snapshot_path: Annotated[Path, typer.Option("--snapshot")],
    query: Annotated[str | None, typer.Argument()] = None,
    manifest_sha256: Annotated[str | None, typer.Option("--manifest-sha256")] = None,
    request_json: Annotated[Path | None, typer.Option("--request-json")] = None,
    language: Annotated[str | None, typer.Option("--language")] = None,
    platform: Annotated[list[str] | None, typer.Option("--platform")] = None,
    collection: Annotated[list[str] | None, typer.Option("--collection")] = None,
    availability: Annotated[list[str] | None, typer.Option("--availability")] = None,
    review: Annotated[list[str] | None, typer.Option("--review")] = None,
    rating: Annotated[list[str] | None, typer.Option("--rating")] = None,
    exclude_warning: Annotated[list[str] | None, typer.Option("--exclude-warning")] = None,
    animated: Annotated[str | None, typer.Option("--animated")] = None,
    media_kind: Annotated[list[str] | None, typer.Option("--media-kind")] = None,
    color_behavior: Annotated[list[str] | None, typer.Option("--color-behavior")] = None,
    color_family: Annotated[list[str] | None, typer.Option("--color-family")] = None,
    contains_text: Annotated[str | None, typer.Option("--contains-text")] = None,
    content_type: Annotated[list[str] | None, typer.Option("--content-type")] = None,
    style: Annotated[list[str] | None, typer.Option("--style")] = None,
    suggested_use: Annotated[list[str] | None, typer.Option("--suggested-use")] = None,
    uncertainty: Annotated[list[str] | None, typer.Option("--uncertainty")] = None,
    concept: Annotated[list[str] | None, typer.Option("--concept")] = None,
    rights: Annotated[list[str] | None, typer.Option("--rights")] = None,
    include_sensitive: Annotated[bool, typer.Option("--include-sensitive")] = False,
    include_unreviewed: Annotated[bool, typer.Option("--include-unreviewed")] = False,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    cursor: Annotated[str | None, typer.Option("--cursor")] = None,
    sort: Annotated[str | None, typer.Option("--sort")] = None,
    view: Annotated[str | None, typer.Option("--view")] = None,
    allow_unverified: Annotated[bool, typer.Option("--allow-unverified")] = False,
    offline: Annotated[bool, typer.Option("--offline/--no-offline")] = True,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    jsonl_output: Annotated[bool, typer.Option("--jsonl")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    state: dict[str, LoadedSnapshot] = {}
    request_context: dict[str, Any] = {}

    def action() -> ReadCommandResult:
        individual_values = [
            query,
            language,
            platform,
            collection,
            availability,
            review,
            rating,
            exclude_warning,
            animated,
            media_kind,
            color_behavior,
            color_family,
            contains_text,
            content_type,
            style,
            suggested_use,
            uncertainty,
            concept,
            rights,
            limit,
            cursor,
            sort,
            view,
        ]
        if request_json is not None and (
            any(value is not None for value in individual_values)
            or include_sensitive
            or include_unreviewed
        ):
            raise CommandError(
                "OPTION_CONFLICT",
                (
                    "--request-json cannot be combined with positional query or individual "
                    "search flags."
                ),
                hint="Put all semantic search arguments in the structured request.",
            )
        if request_json is not None:
            (
                selected_query,
                selected_language,
                filters,
                selected_limit,
                selected_cursor,
                selected_sort,
                selected_view,
            ) = _structured_search(request_json)
        else:
            selected_query = query or ""
            selected_language = language
            filters = _default_filters(
                platform=platform,
                collection=collection,
                availability=availability,
                review=review,
                rating=rating,
                exclude_warning=exclude_warning,
                animated=animated,
                media_kind=media_kind,
                color_behavior=color_behavior,
                color_family=color_family,
                contains_text=contains_text,
                content_type=content_type,
                style=style,
                suggested_use=suggested_use,
                uncertainty=uncertainty,
                concept=concept,
                rights=rights,
                include_sensitive=include_sensitive,
                include_unreviewed=include_unreviewed,
            )
            selected_limit = _limit(limit)
            selected_cursor = cursor
            selected_sort = _enum(sort or "relevance", _SORTS, "--sort")
            selected_view = _enum(view or "agent", _VIEWS, "--view")
        request_context.update(
            {
                "query": selected_query,
                "language": selected_language or "en",
                "filters": filters,
                "sort": selected_sort,
                "view": selected_view,
                "limit": selected_limit,
                "cursor": selected_cursor,
            }
        )
        snapshot, reader, warnings = _open(
            state, snapshot_path, manifest_sha256, allow_unverified, offline
        )
        result = reader.search(
            query=selected_query,
            language=selected_language,
            filters=filters,
            sort=selected_sort,
            view=selected_view,
            limit=selected_limit,
            cursor=selected_cursor,
        )
        return ReadCommandResult(
            result=result,
            dataset=snapshot.dataset_context(),
            arguments=dict(request_context),
            view=selected_view,
            warnings=warnings,
            stream_items=cast(list[Any], result["items"]),
        )

    execute_read(
        "search",
        action,
        json_output=json_output,
        jsonl_output=jsonl_output,
        quiet=quiet,
        debug=debug,
        dataset_provider=_dataset_provider(state),
        request_context_provider=lambda: (dict(request_context), str(request_context["view"])),
    )
