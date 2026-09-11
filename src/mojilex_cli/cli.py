"""Public ``mojilex`` command-line interface."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, NoReturn, cast

import typer

from mojilex_cli import __version__
from mojilex_cli.commands.dataset import (
    build_index_command,
    review_command,
    set_status_command,
    takedown_command,
    takedown_preview_command,
    validate_command,
)
from mojilex_cli.commands.runtime import (
    CommandError,
    execute,
    is_usage_error,
    machine_envelope_emitted,
    machine_output_mode,
    require_confirmation,
)
from mojilex_cli.commands.system import config_show_command, doctor_command, init_command

app = typer.Typer(
    name="mojilex",
    help="Build, validate, and publish the media-free MojiLex emoji dataset.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
config_app = typer.Typer(help="Inspect non-secret configuration.")
cache_app = typer.Typer(help="Inspect or prune the content-addressed AI cache.")
dedupe_app = typer.Typer(help="Scan and review exact or visual duplicate candidates.")
app.add_typer(config_app, name="config")
app.add_typer(cache_app, name="cache")
app.add_typer(dedupe_app, name="dedupe")


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"mojilex {__version__}")
        raise typer.Exit()


def _confirmation_callback(
    *, yes: bool, non_interactive: bool, json_output: bool
) -> Callable[[str], bool]:
    def confirm(message: str) -> bool:
        require_confirmation(
            message,
            yes=yes,
            non_interactive=non_interactive,
            json_output=json_output,
        )
        return True

    return confirm


def _unknown_cost_callback(
    *, yes: bool, non_interactive: bool, json_output: bool
) -> Callable[[int], bool]:
    def confirm(requests: int) -> bool:
        message = f"Authorize {requests} new AI request(s)? The provider's USD cost is unknown."
        try:
            require_confirmation(
                message,
                yes=yes,
                non_interactive=non_interactive,
                json_output=json_output,
            )
        except CommandError as exc:
            raise CommandError(
                "UNKNOWN_COST",
                "The provider's USD cost is unknown and was not authorized.",
                hint=(
                    "Rerun with --allow-unknown-cost after reviewing the planned AI requests, "
                    "or use --yes in an interactive workflow."
                ),
                details={"new_ai_requests": requests, "estimated_cost_usd": None},
            ) from exc
        return True

    return confirm


@app.callback()
def root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """MojiLex dataset authoring utility."""


@app.command("init")
def initialize(
    repo: Annotated[
        str, typer.Option("--repo", help="Dataset path or OWNER/REPO.")
    ] = "MojiLex/mojilex",
    provider: Annotated[str, typer.Option("--provider")] = "gemini",
    model: Annotated[str, typer.Option("--model", help="Explicit provider model ID.")] = "",
    publish: Annotated[str, typer.Option("--publish", help="local or pr")] = "pr",
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Config file to create.")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing config.")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    execute(
        "init",
        lambda: init_command(
            repo=repo,
            provider=provider,
            model=model,
            publish=publish,
            config_path=config_path,
            force=force,
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("add")
def add(
    sources: Annotated[list[str] | None, typer.Argument(help="Public source URLs.")] = None,
    repo: Annotated[str | None, typer.Option("--repo", help="Dataset path or OWNER/REPO.")] = None,
    from_file: Annotated[Path | None, typer.Option("--from-file")] = None,
    stdin: Annotated[bool, typer.Option("--stdin", help="Read one source per stdin line.")] = False,
    platform: Annotated[str, typer.Option("--platform")] = "auto",
    provider: Annotated[str | None, typer.Option("--provider")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    lang: Annotated[list[str] | None, typer.Option("--lang")] = None,
    publish: Annotated[str | None, typer.Option("--publish")] = None,
    direct_push: Annotated[bool, typer.Option("--direct-push")] = False,
    base: Annotated[str | None, typer.Option("--base")] = None,
    redescribe: Annotated[str, typer.Option("--redescribe")] = "changed",
    overwrite_reviewed: Annotated[bool, typer.Option("--overwrite-reviewed")] = False,
    new_identity: Annotated[bool, typer.Option("--new-identity")] = False,
    same_identity: Annotated[bool, typer.Option("--same-identity")] = False,
    max_items: Annotated[int | None, typer.Option("--max-items", min=1)] = None,
    max_ai_requests: Annotated[int | None, typer.Option("--max-ai-requests", min=0)] = None,
    max_cost_usd: Annotated[str | None, typer.Option("--max-cost-usd")] = None,
    allow_unknown_cost: Annotated[bool, typer.Option("--allow-unknown-cost")] = False,
    ai_concurrency: Annotated[int | None, typer.Option("--ai-concurrency", min=1)] = None,
    download_concurrency: Annotated[
        int | None, typer.Option("--download-concurrency", min=1)
    ] = None,
    dedupe: Annotated[str | None, typer.Option("--dedupe", help="off, exact, or near")] = None,
    max_dedupe_candidates: Annotated[
        int | None, typer.Option("--max-dedupe-candidates", min=1, max=200)
    ] = None,
    dedupe_profile: Annotated[str | None, typer.Option("--dedupe-profile")] = None,
    model_routing: Annotated[
        str | None, typer.Option("--model-routing", help="off or rules")
    ] = None,
    escalation_model: Annotated[str | None, typer.Option("--escalation-model")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    check_media: Annotated[bool, typer.Option("--check-media")] = False,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    no_color: Annotated[bool, typer.Option("--no-color")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
) -> None:
    del verbose, no_color
    from mojilex_cli.commands.workflow import add_command, collect_sources

    def action():  # type: ignore[no-untyped-def]
        selected = collect_sources(
            sources or [], from_file=from_file, use_stdin=stdin, stream=sys.stdin
        )
        return add_command(
            selected,
            repo=repo,
            platform=platform,
            provider=provider,
            model=model,
            languages=tuple(lang or ()),
            publish=publish,
            direct_push=direct_push,
            base=base,
            redescribe=redescribe,
            overwrite_reviewed=overwrite_reviewed,
            new_identity=new_identity,
            same_identity=same_identity,
            max_items=max_items,
            max_ai_requests=max_ai_requests,
            max_cost_usd=_decimal(max_cost_usd),
            allow_unknown_cost=allow_unknown_cost,
            ai_concurrency=ai_concurrency,
            download_concurrency=download_concurrency,
            dedupe=dedupe,
            max_dedupe_candidates=max_dedupe_candidates,
            dedupe_profile=dedupe_profile,
            model_routing=model_routing,
            escalation_model=escalation_model,
            dry_run=dry_run,
            check_media=check_media,
            fail_fast=fail_fast,
            confirmation=_confirmation_callback(
                yes=yes,
                non_interactive=non_interactive,
                json_output=json_output,
            ),
            unknown_cost_confirmation=_unknown_cost_callback(
                yes=yes,
                non_interactive=non_interactive,
                json_output=json_output,
            ),
        )

    execute("add", action, json_output=json_output, quiet=quiet, debug=debug)


@app.command("import")
def import_sources(
    sources: Annotated[list[str], typer.Argument(help="Public source URLs.")],
    repo: Annotated[str | None, typer.Option("--repo")] = None,
    platform: Annotated[str, typer.Option("--platform")] = "auto",
    max_items: Annotated[int | None, typer.Option("--max-items", min=1)] = None,
    check_media: Annotated[bool, typer.Option("--check-media")] = True,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.workflow import import_command

    execute(
        "import",
        lambda: import_command(
            sources,
            repo=repo,
            platform=platform,
            max_items=max_items,
            check_media=check_media,
            fail_fast=fail_fast,
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("describe")
def describe(
    selectors: Annotated[list[str], typer.Argument(help="Run ID or entity selectors.")],
    provider: Annotated[str | None, typer.Option("--provider")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    max_ai_requests: Annotated[int | None, typer.Option("--max-ai-requests", min=0)] = None,
    max_cost_usd: Annotated[str | None, typer.Option("--max-cost-usd")] = None,
    allow_unknown_cost: Annotated[bool, typer.Option("--allow-unknown-cost")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.workflow import describe_command

    execute(
        "describe",
        lambda: describe_command(
            selectors,
            provider=provider,
            model=model,
            max_ai_requests=max_ai_requests,
            max_cost_usd=_decimal(max_cost_usd),
            allow_unknown_cost=allow_unknown_cost,
            unknown_cost_confirmation=_unknown_cost_callback(
                yes=yes,
                non_interactive=non_interactive,
                json_output=json_output,
            ),
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("validate")
def validate(
    path: Annotated[Path, typer.Argument(help="Local dataset root.")] = Path("."),
    strict: Annotated[bool, typer.Option("--strict/--no-strict")] = True,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    execute(
        "validate",
        lambda: validate_command(path, strict=strict),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("update")
def update(
    selector: Annotated[str | None, typer.Argument(help="Source URL or collection ID.")] = None,
    all_collections: Annotated[bool, typer.Option("--all")] = False,
    repo: Annotated[str | None, typer.Option("--repo")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.workflow import update_command

    execute(
        "update",
        lambda: update_command(
            selector, all_collections=all_collections, repo=repo, dry_run=dry_run
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("submit")
def submit(
    target: Annotated[str | None, typer.Argument(help="Path or run ID.")] = None,
    repo: Annotated[str | None, typer.Option("--repo")] = None,
    direct_push: Annotated[bool, typer.Option("--direct-push")] = False,
    base: Annotated[str | None, typer.Option("--base")] = None,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.workflow import submit_command

    def action():  # type: ignore[no-untyped-def]
        return submit_command(
            target,
            repo=repo,
            direct_push=direct_push,
            base=base,
            confirmation=_confirmation_callback(
                yes=yes,
                non_interactive=non_interactive,
                json_output=json_output,
            ),
        )

    execute("submit", action, json_output=json_output, quiet=quiet, debug=debug)


@app.command("build-index")
def build_index_cli(
    path: Annotated[Path, typer.Argument(help="Local dataset root.")] = Path("."),
    output: Annotated[Path | None, typer.Option("--output")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    execute(
        "build-index",
        lambda: build_index_command(path, output),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("benchmark-dedupe")
def benchmark_dedupe_cli(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Versioned dedupe benchmark manifest.",
        ),
    ],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.benchmark import benchmark_dedupe_command

    execute(
        "benchmark-dedupe",
        lambda: benchmark_dedupe_command(manifest),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("benchmark-model")
def benchmark_model_cli(
    provider: Annotated[str, typer.Option("--provider", help="Exact manifest provider ID.")],
    model: Annotated[str, typer.Option("--model", help="Exact manifest model ID.")],
    benchmark_manifest: Annotated[
        Path,
        typer.Option(
            "--benchmark-manifest",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help="Versioned model benchmark manifest with human adjudication.",
        ),
    ],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.benchmark import benchmark_model_command

    execute(
        "benchmark-model",
        lambda: benchmark_model_command(
            provider_name=provider,
            model_id=model,
            benchmark_manifest=benchmark_manifest,
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@dedupe_app.command("scan")
def dedupe_scan(
    selector: Annotated[str | None, typer.Argument(help="Emoji or collection selector.")] = None,
    all_items: Annotated[bool, typer.Option("--all", help="Rebuild the complete index.")] = False,
    repo: Annotated[str | None, typer.Option("--repo")] = None,
    max_candidates: Annotated[
        int | None, typer.Option("--max-dedupe-candidates", min=1, max=200)
    ] = None,
    profile: Annotated[str | None, typer.Option("--dedupe-profile")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.dedupe import dedupe_scan_command

    execute(
        "dedupe scan",
        lambda: dedupe_scan_command(
            selector,
            all_items=all_items,
            repo=repo,
            max_candidates=max_candidates,
            profile=profile,
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@dedupe_app.command("explain")
def dedupe_explain(
    emoji_id: Annotated[str, typer.Argument()],
    against_emoji_id: Annotated[str, typer.Argument()],
    repo: Annotated[str | None, typer.Option("--repo")] = None,
    max_candidates: Annotated[
        int | None, typer.Option("--max-dedupe-candidates", min=1, max=200)
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.dedupe import dedupe_explain_command

    execute(
        "dedupe explain",
        lambda: dedupe_explain_command(
            emoji_id,
            against_emoji_id,
            repo=repo,
            max_candidates=max_candidates,
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@dedupe_app.command("review")
def dedupe_review(
    emoji_id: Annotated[str, typer.Argument()],
    reviewer: Annotated[str, typer.Option("--reviewer")],
    against: Annotated[str | None, typer.Option("--against")] = None,
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    decision: Annotated[
        str | None,
        typer.Option(
            "--decision",
            help="same-artwork, variant-of, related-series, not-duplicate, or skip",
        ),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.dedupe import dedupe_review_command

    def choose(preview: Path, explanation: object) -> str:
        del explanation
        typer.echo(f"Temporary comparison preview: {preview}")
        return str(
            typer.prompt("Decision [same-artwork/variant-of/related-series/not-duplicate/skip]")
        )

    execute(
        "dedupe review",
        lambda: dedupe_review_command(
            emoji_id,
            against,
            repo=repo,
            reviewer=reviewer,
            decision=decision,
            chooser=None if json_output or quiet else choose,
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("resume")
def resume(
    run_id: Annotated[str, typer.Argument()],
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.workflow import resume_command

    execute(
        "resume",
        lambda: resume_command(
            run_id,
            confirmation=_confirmation_callback(
                yes=yes,
                non_interactive=non_interactive,
                json_output=json_output,
            ),
            unknown_cost_confirmation=_unknown_cost_callback(
                yes=yes,
                non_interactive=non_interactive,
                json_output=json_output,
            ),
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("review")
def review(
    emoji_id: Annotated[str, typer.Argument()],
    action: Annotated[str, typer.Argument(help="approve, request-changes, or reject")],
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    reviewer: Annotated[str | None, typer.Option("--reviewer")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    execute(
        "review",
        lambda: review_command(repo, emoji_id, action, reviewer),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("set-status")
def set_status(
    entity_id: Annotated[str, typer.Argument()],
    availability: Annotated[str, typer.Option("--availability")],
    reason: Annotated[str | None, typer.Option("--reason")] = None,
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    reviewer: Annotated[str | None, typer.Option("--reviewer")] = None,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    def action():  # type: ignore[no-untyped-def]
        if availability in {"private", "deleted"}:
            require_confirmation(
                f"Set {entity_id} availability to {availability}?",
                yes=yes,
                non_interactive=non_interactive,
                json_output=json_output,
            )
        return set_status_command(repo, entity_id, availability, reason, reviewer)

    execute("set-status", action, json_output=json_output, quiet=quiet, debug=debug)


@app.command("takedown")
def takedown_cli(
    entity_id: Annotated[str, typer.Argument()],
    reason: Annotated[str, typer.Option("--reason")],
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    def action():  # type: ignore[no-untyped-def]
        impact = takedown_preview_command(repo, entity_id, reason)
        affected_ids = ", ".join(
            str(value) for value in cast(Sequence[object], impact["affected_ids"])
        )
        changed_paths = ", ".join(
            str(value) for value in cast(Sequence[object], impact["changed_paths"])
        )
        require_confirmation(
            (
                f"Permanently remove public fields for {entity_id} and create a tombstone? "
                f"Affected IDs: {affected_ids or '(none)'}. "
                f"Changed paths: {changed_paths or '(none)'}."
            ),
            yes=yes,
            non_interactive=non_interactive,
            json_output=json_output,
        )
        return takedown_command(
            repo,
            entity_id,
            reason,
            expected_source_sha256=str(impact["source_sha256"]),
        )

    execute("takedown", action, json_output=json_output, quiet=quiet, debug=debug)


@app.command("doctor")
def doctor(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    execute("doctor", doctor_command, json_output=json_output, quiet=quiet, debug=debug)


@config_app.command("show")
def config_show(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    execute(
        "config show",
        config_show_command,
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@cache_app.command("info")
def cache_info(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.workflow import cache_info_command

    execute("cache info", cache_info_command, json_output=json_output, quiet=quiet, debug=debug)


@cache_app.command("prune")
def cache_prune(
    older_than_days: Annotated[int, typer.Option("--older-than-days", min=0)] = 30,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.workflow import cache_prune_command

    def action():  # type: ignore[no-untyped-def]
        require_confirmation(
            f"Prune cache entries older than {older_than_days} day(s)?",
            yes=yes,
            non_interactive=non_interactive,
            json_output=json_output,
        )
        return cache_prune_command(older_than_days)

    execute("cache prune", action, json_output=json_output, quiet=quiet, debug=debug)


def _decimal(value: str | None) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise CommandError(
            "CONFIG_INVALID",
            "--max-cost-usd must be a decimal amount.",
            hint="Use a value such as 1.50.",
        ) from exc
    if not result.is_finite() or result < 0:
        raise CommandError(
            "CONFIG_INVALID",
            "--max-cost-usd must be finite and non-negative.",
            hint="Use a value such as 1.50.",
        )
    return result


def _extract_json_flag(argv: Sequence[str]) -> tuple[bool, list[str]]:
    """Remove machine-output flags before Typer parses their original position."""

    enabled = False
    passthrough = False
    cleaned: list[str] = []
    for argument in argv:
        if argument == "--":
            passthrough = True
        if not passthrough and argument == "--json":
            enabled = True
            continue
        cleaned.append(argument)
    return enabled, cleaned


def _command_label(argv: Sequence[str]) -> str:
    commands = {
        "add",
        "benchmark-dedupe",
        "benchmark-model",
        "build-index",
        "cache",
        "config",
        "dedupe",
        "describe",
        "doctor",
        "import",
        "init",
        "resume",
        "review",
        "set-status",
        "submit",
        "takedown",
        "update",
        "validate",
    }
    for index, argument in enumerate(argv):
        if argument not in commands:
            continue
        if argument in {"cache", "config", "dedupe"}:
            for child in argv[index + 1 :]:
                if not child.startswith("-"):
                    return f"{argument} {child}"
        return argument
    return "mojilex"


def _raise_boundary_error(exc: BaseException) -> NoReturn:
    raise exc


def _emit_boundary_error(command: str, exc: BaseException) -> NoReturn:
    try:
        execute(command, lambda: _raise_boundary_error(exc), json_output=True)
    except typer.Exit as exit_error:
        raise SystemExit(exit_error.exit_code) from exc
    raise AssertionError("an error envelope must terminate with a non-zero exit code")


def main() -> None:
    json_requested, argv = _extract_json_flag(sys.argv[1:])
    if not json_requested:
        app()
        return

    label = _command_label(argv)
    with machine_output_mode():
        if any(argument in {"-h", "--help", "--version"} for argument in argv):
            _emit_boundary_error(
                label,
                CommandError(
                    "CONFIG_INVALID",
                    "Human help and version output are unavailable in JSON mode.",
                    hint="Remove --json to view help or version information.",
                ),
            )
        command = typer.main.get_command(app)
        try:
            result = command.main(
                args=argv,
                prog_name="mojilex",
                standalone_mode=False,
                windows_expand_args=False,
            )
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, typer.Abort)) or is_usage_error(exc):
                _emit_boundary_error(label, exc)
            raise
        if isinstance(result, int) and result != 0:
            if not machine_envelope_emitted():
                boundary_error: BaseException
                if result == 130:
                    boundary_error = typer.Abort()
                else:
                    boundary_error = CommandError(
                        "INTERNAL_ERROR",
                        "The command terminated without a machine-readable result.",
                        hint="Rerun with --debug and report this output-contract failure.",
                    )
                _emit_boundary_error(label, boundary_error)
            raise SystemExit(result)


if __name__ == "__main__":
    main()
