"""Public ``mojilex`` command-line interface."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, cast

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
from mojilex_cli.commands.read import execute_read, register_read_commands
from mojilex_cli.commands.runtime import (
    CommandError,
    CommandResult,
    execute,
    is_usage_error,
    machine_envelope_emitted,
    machine_output_mode,
    require_confirmation,
    suspend_progress,
)
from mojilex_cli.i18n import (
    confirm as ui_confirm,
)
from mojilex_cli.i18n import (
    current_ui_language,
    extract_ui_language,
    localize_command_tree,
    use_ui_language,
)
from mojilex_cli.i18n import text as ui_text

app = typer.Typer(
    name="mojilex",
    help="Build, validate, and publish the media-free MojiLex emoji dataset.",
    no_args_is_help=False,
    invoke_without_command=True,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
config_app = typer.Typer(help="Inspect configuration and manage stored API credentials.")
cache_app = typer.Typer(help="Inspect or prune the content-addressed AI cache.")
dedupe_app = typer.Typer(help="Scan and review exact or visual duplicate candidates.")
app.add_typer(config_app, name="config")
app.add_typer(cache_app, name="cache")
app.add_typer(dedupe_app, name="dedupe")
register_read_commands(app)


def init_command(
    *,
    repo: str | None,
    provider: str,
    model: str,
    publish: str,
    config_path: Path | None,
    force: bool,
    languages: Sequence[str] = ("ru", "en"),
    prompt: Callable[[str, str], str] | None = None,
    ui_language: str = "en",
) -> CommandResult:
    """Load authoring diagnostics only when the init command is invoked."""

    from mojilex_cli.commands.system import init_command as implementation

    return implementation(
        repo=repo,
        provider=provider,
        model=model,
        publish=publish,
        config_path=config_path,
        force=force,
        languages=languages,
        prompt=prompt,
        ui_language=ui_language,
    )


def doctor_command() -> CommandResult:
    """Load media probes only when the doctor command is invoked."""

    from mojilex_cli.commands.system import doctor_command as implementation

    return implementation()


def config_show_command() -> CommandResult:
    """Load credential/config adapters only when config show is invoked."""

    from mojilex_cli.commands.system import config_show_command as implementation

    return implementation()


def config_set_credentials_command(values: dict[str, str]) -> CommandResult:
    """Load the system-keyring adapter only when credentials are being saved."""

    from mojilex_cli.commands.system import config_set_credentials_command as implementation

    return implementation(values)


def config_clear_credentials_command() -> CommandResult:
    """Load the system-keyring adapter only when credentials are being deleted."""

    from mojilex_cli.commands.system import config_clear_credentials_command as implementation

    return implementation()


def install_media_dependencies_command(checks: dict[str, Any]) -> CommandResult:
    """Run the bundled installer only after explicit interactive authorization."""

    from mojilex_cli.commands.system import install_media_dependencies_command as implementation

    return implementation(checks)


def uninstall_preview_command(*, keep_data: bool) -> dict[str, object]:
    from mojilex_cli.commands.system import uninstall_preview_command as implementation

    return implementation(keep_data=keep_data)


def uninstall_command(*, keep_data: bool) -> CommandResult:
    from mojilex_cli.commands.system import uninstall_command as implementation

    return implementation(keep_data=keep_data)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"mojilex {__version__}")
        raise typer.Exit()


def _help_all_callback(ctx: typer.Context, value: bool) -> None:
    if value:
        for command in getattr(ctx.command, "commands", {}).values():
            command.hidden = False
        typer.echo(ctx.get_help())
        raise typer.Exit()


def _official_confirmation_callback(
    *, non_interactive: bool, json_output: bool, quiet: bool
) -> Callable[[str], bool]:
    def confirm_official(message: str) -> bool:
        # General --yes never bypasses this separate, explicitly negative default.
        if non_interactive or json_output or quiet or not sys.stdin.isatty():
            return False
        with suspend_progress():
            return ui_confirm(message, default=False)

    return confirm_official


def _confirmation_callback(
    *, yes: bool, non_interactive: bool, json_output: bool, default: bool = False
) -> Callable[[str], bool]:
    def confirm(message: str) -> bool:
        require_confirmation(
            message,
            yes=yes,
            non_interactive=non_interactive,
            json_output=json_output,
            default=default,
        )
        return True

    return confirm


def _unknown_cost_callback(
    *, yes: bool, non_interactive: bool, json_output: bool
) -> Callable[[int | None], bool]:
    def confirm(requests: int | None) -> bool:
        message = (
            "Authorize AI requests without a request-count limit for this run, including retries? "
            if requests is None
            else f"Authorize up to {requests} additional AI requests for this run, "
            "including retries? "
        ) + "The USD cost may be unknown. This is a one-time approval for this invocation."
        try:
            require_confirmation(
                message,
                yes=yes,
                non_interactive=non_interactive,
                json_output=json_output,
                default=True,
            )
        except CommandError as exc:
            raise CommandError(
                "UNKNOWN_COST",
                "AI requests were not authorized.",
                hint=(
                    "Review the planned AI requests and confirm when restarting, "
                    "or use --allow-unknown-cost (or --yes in an interactive workflow)."
                ),
                details={"new_ai_requests": requests, "estimated_cost_usd": None},
            ) from exc
        return True

    return confirm


def _with_runtime_secrets(
    action: Callable[[], CommandResult],
    *,
    names: Sequence[str],
    prompt: Callable[[str], str] | None,
) -> CommandResult:
    """Prompt for missing credentials without persisting them after the command."""

    previous = {name: os.environ.get(name) for name in names}
    labels = {
        "TELEGRAM_BOT_TOKEN": "Telegram Bot API token",
        "GEMINI_API_KEY": "Gemini API key",
        "OPENAI_API_KEY": "OpenAI API key",
    }
    try:
        from mojilex_cli.config import load_credentials

        credentials = load_credentials()
        stored_or_environment = {
            "TELEGRAM_BOT_TOKEN": credentials.telegram_bot_token,
            "GEMINI_API_KEY": credentials.gemini_api_key,
            "OPENAI_API_KEY": credentials.openai_api_key,
        }
        for name in names:
            if not os.environ.get(name) and stored_or_environment.get(name):
                os.environ[name] = cast(str, stored_or_environment[name])
        if prompt is not None:
            for name in names:
                if os.environ.get(name):
                    continue
                prompted_value = prompt(ui_text(labels.get(name, name))).strip()
                if not prompted_value:
                    raise CommandError(
                        "CREDENTIAL_MISSING",
                        f"{name} was not provided.",
                        hint="Enter the credential or set it in the process environment.",
                    )
                os.environ[name] = prompted_value
        return action()
    finally:
        for name, previous_value in previous.items():
            if previous_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous_value


def _secret_prompt(
    *, non_interactive: bool, json_output: bool, quiet: bool
) -> Callable[[str], str] | None:
    if non_interactive or json_output or quiet or not sys.stdin.isatty():
        return None
    return lambda label: _progress_prompt(label, hide_input=True, err=True)


def _progress_prompt(label: str, **options: Any) -> str:
    with suspend_progress():
        return str(typer.prompt(label, **options))


def _pack_action(action: Callable[[], CommandResult], selectors: Sequence[str]) -> CommandResult:
    """Attach a usable next-step name without changing execution or saved data."""
    result = action()
    from mojilex_cli.commands.packs import _names, _selector_name, resolve_pack_run

    names = [name for selector in selectors if (name := _selector_name(selector))]
    if not names and result.run_id:
        try:
            names = list(_names(resolve_pack_run(result.run_id, purpose="view")))
        except (OSError, ValueError, CommandError):
            pass  # A display hint must never turn a successful operation into failure.
    if len(set(names)) == 1:
        result.result.setdefault("pack_name", names[0])
    return result


@app.callback()
def root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
    ui_language: Annotated[
        str | None,
        typer.Option(
            "--ui-language",
            help="Human interface language: en or ru. Commands and JSON fields stay unchanged.",
        ),
    ] = None,
    help_all: Annotated[
        bool,
        typer.Option(
            "--help-all",
            is_eager=True,
            callback=_help_all_callback,
            help="Show all advanced commands.",
        ),
    ] = False,
) -> None:
    """MojiLex dataset authoring utility."""
    if ctx.invoked_subcommand is None:
        from mojilex_cli.commands.interactive import dispatch_command, is_interactive, run_menu

        if machine_output_mode_requested():
            raise typer.BadParameter("Choose a command, such as list.")
        if is_interactive():
            run_menu(dispatch_command)
        else:
            typer.echo(ctx.get_help())
            raise typer.Exit(2)


def machine_output_mode_requested() -> bool:
    from mojilex_cli.commands.runtime import machine_output_requested

    return machine_output_requested()


@app.command("init")
def initialize(
    repo: Annotated[str | None, typer.Option("--repo", help="Dataset path or OWNER/REPO.")] = None,
    provider: Annotated[str, typer.Option("--provider")] = "gemini",
    model: Annotated[str, typer.Option("--model", help="Explicit provider model ID.")] = "",
    publish: Annotated[str, typer.Option("--publish", help="local or pr")] = "pr",
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Config file to create.")
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing config.")] = False,
    lang: Annotated[
        list[str] | None, typer.Option("--lang", help="Repeat for each language.")
    ] = None,
    non_interactive: Annotated[
        bool,
        typer.Option(
            "--non-interactive",
            help="Skip the non-secret setup wizard; init never requests credentials.",
        ),
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Create non-secret settings; authoring commands request missing credentials."""

    execute(
        "init",
        lambda: init_command(
            repo=repo,
            provider=provider,
            model=model,
            publish=publish,
            config_path=config_path,
            force=force,
            languages=tuple(lang or ("ru", "en")),
            prompt=(lambda label, default: _progress_prompt(label, default=default, err=True))
            if sys.stdin.isatty() and not (non_interactive or json_output or quiet)
            else None,
            ui_language=current_ui_language(),
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
    official_packs: Annotated[
        str | None, typer.Option("--official-packs", help="ask, skip, or allow official packs.")
    ] = None,
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
    max_ai_requests: Annotated[
        str | None, typer.Option("--max-ai-requests", help="Whole-run request limit or unlimited.")
    ] = None,
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
    """Analyze public emoji packs and publish validated metadata."""

    from mojilex_cli.commands.workflow import add_command, collect_sources

    request_limit = _request_limit(max_ai_requests)

    def action():  # type: ignore[no-untyped-def]
        selected = collect_sources(
            sources or [], from_file=from_file, use_stdin=stdin, stream=sys.stdin
        )
        return add_command(
            selected,
            official_pack_policy=official_packs,
            official_confirmation=_official_confirmation_callback(
                non_interactive=non_interactive, json_output=json_output, quiet=quiet
            ),
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
            max_ai_requests=request_limit,
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

    required_secrets = (
        ("TELEGRAM_BOT_TOKEN",) if dry_run else ("TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY")
    )
    execute(
        "add",
        lambda: _with_runtime_secrets(
            action,
            names=required_secrets,
            prompt=_secret_prompt(
                non_interactive=non_interactive,
                json_output=json_output,
                quiet=quiet,
            ),
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
        verbose=verbose,
        no_color=no_color,
    )


@app.command("import")
def import_sources(
    sources: Annotated[
        list[str] | None, typer.Argument(help="Public source URLs or a text file path.")
    ] = None,
    from_file: Annotated[Path | None, typer.Option("--from-file")] = None,
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh", help="Check sources for updates instead of reusing saved imports."
        ),
    ] = False,
    official_packs: Annotated[
        str | None, typer.Option("--official-packs", help="ask, skip, or allow official packs.")
    ] = None,
    repo: Annotated[str | None, typer.Option("--repo")] = None,
    platform: Annotated[str, typer.Option("--platform")] = "auto",
    max_items: Annotated[int | None, typer.Option("--max-items", min=1)] = None,
    download_concurrency: Annotated[
        int | None, typer.Option("--download-concurrency", min=1)
    ] = None,
    check_media: Annotated[bool, typer.Option("--check-media")] = True,
    fail_fast: Annotated[bool, typer.Option("--fail-fast")] = False,
    preparation: Annotated[
        str,
        typer.Option("--preparation", hidden=True),
    ] = "full",
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Download and verify media in a staging run without AI or publication."""

    from mojilex_cli.commands.workflow import collect_sources, import_command

    def action():  # type: ignore[no-untyped-def]
        selected = collect_sources(
            sources or [], from_file=from_file, use_stdin=False, stream=sys.stdin
        )
        return _pack_action(
            lambda: import_command(
                selected,
                refresh=refresh,
                official_pack_policy=official_packs,
                official_confirmation=_official_confirmation_callback(
                    non_interactive=False, json_output=json_output, quiet=quiet
                ),
                repo=repo,
                platform=platform,
                max_items=max_items,
                download_concurrency=download_concurrency,
                check_media=check_media,
                fail_fast=fail_fast,
                preparation=cast(Literal["full", "metadata", "download_all"], preparation),
            ),
            selected,
        )

    execute(
        "import",
        lambda: _with_runtime_secrets(
            action,
            names=("TELEGRAM_BOT_TOKEN",),
            prompt=_secret_prompt(
                non_interactive=False,
                json_output=json_output,
                quiet=quiet,
            ),
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("describe")
def describe(
    selectors: Annotated[list[str], typer.Argument(help="Run ID or entity selectors.")],
    official_packs: Annotated[
        str | None, typer.Option("--official-packs", help="ask, skip, or allow official packs.")
    ] = None,
    provider: Annotated[str | None, typer.Option("--provider")] = None,
    model: Annotated[str | None, typer.Option("--model")] = None,
    max_ai_requests: Annotated[
        str | None, typer.Option("--max-ai-requests", help="Whole-run request limit or unlimited.")
    ] = None,
    max_cost_usd: Annotated[str | None, typer.Option("--max-cost-usd")] = None,
    allow_unknown_cost: Annotated[bool, typer.Option("--allow-unknown-cost")] = False,
    ai_concurrency: Annotated[int | None, typer.Option("--ai-concurrency", min=1)] = None,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Generate AI metadata for a staged run without publishing it."""

    from mojilex_cli.commands.workflow import describe_command

    request_limit = _request_limit(max_ai_requests)

    execute(
        "describe",
        lambda: _with_runtime_secrets(
            lambda: _pack_action(
                lambda: describe_command(
                    selectors,
                    official_pack_policy=official_packs,
                    official_confirmation=_official_confirmation_callback(
                        non_interactive=non_interactive, json_output=json_output, quiet=quiet
                    ),
                    provider=provider,
                    model=model,
                    ai_concurrency=ai_concurrency,
                    max_ai_requests=request_limit,
                    max_cost_usd=_decimal(max_cost_usd),
                    allow_unknown_cost=allow_unknown_cost,
                    unknown_cost_confirmation=_unknown_cost_callback(
                        yes=yes,
                        non_interactive=non_interactive,
                        json_output=json_output,
                    ),
                ),
                selectors,
            ),
            names=("TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY"),
            prompt=_secret_prompt(
                non_interactive=non_interactive,
                json_output=json_output,
                quiet=quiet,
            ),
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("list")
def list_packs(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """List saved packs and analysis progress."""
    from mojilex_cli.commands.packs import list_packs_command

    execute("list", list_packs_command, json_output=json_output, quiet=quiet, debug=debug)


@app.command("show")
def show_pack(
    pack: Annotated[str, typer.Argument(help="Pack name or run ID.")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
    all_fields: Annotated[bool, typer.Option("--all", help="Print all saved fields.")] = False,
    browser: Annotated[
        bool, typer.Option("--browser", help="Open a local browser gallery.")
    ] = False,
) -> None:
    """Read saved descriptions without AI requests."""
    from mojilex_cli.commands.interactive import browse_descriptions, is_interactive
    from mojilex_cli.commands.packs import show_pack_command

    if browser:
        gallery(pack, json_output=json_output, quiet=quiet, debug=debug)
        return
    interactive = is_interactive() and not (
        all_fields or json_output or quiet or machine_output_mode_requested()
    )

    execute(
        "show",
        lambda: browse_descriptions(pack) if interactive else show_pack_command(pack),
        json_output=json_output,
        quiet=quiet or interactive,
        debug=debug,
    )


@app.command("gallery")
def gallery(
    pack: Annotated[str, typer.Argument(help="Pack name or run ID.")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Open saved descriptions and local previews in a browser."""
    from mojilex_cli.commands.gallery import gallery_command

    execute(
        "gallery",
        lambda: gallery_command(
            pack, open_browser=not (json_output or quiet or machine_output_mode_requested())
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("settings")
def settings(
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """View and edit ordinary analysis settings."""
    from mojilex_cli.commands.interactive import _settings, dispatch_command, is_interactive
    from mojilex_cli.commands.settings import settings_command

    interactive = is_interactive() and not (json_output or quiet or machine_output_mode_requested())

    def action() -> CommandResult:
        if interactive:
            _settings(dispatch_command)
        return settings_command()

    execute("settings", action, json_output=json_output, quiet=quiet or interactive, debug=debug)


@app.command("publish")
def publish_pack(
    pack: Annotated[str, typer.Argument(help="Pack name or run ID.")],
    local: Annotated[bool, typer.Option("--local", help="Validate without uploading.")] = False,
    direct_push: Annotated[bool, typer.Option("--direct-push")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Publish a completed pack through a GitHub pull request."""
    from mojilex_cli.commands.workflow import publish_pack_command

    execute(
        "publish",
        lambda: publish_pack_command(
            pack,
            local=local,
            direct_push=direct_push,
            confirmation=_confirmation_callback(
                yes=yes,
                non_interactive=non_interactive,
                json_output=json_output,
                default=True,
            ),
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("sync")
def sync_packs(
    local: Annotated[bool, typer.Option("--local", help="Validate without uploading.")] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Send all completed new packs to GitHub in one pull request."""
    from mojilex_cli.pipeline.batch import sync_packs_command

    execute(
        "sync",
        lambda: sync_packs_command(
            local=local,
            confirmation=_confirmation_callback(
                yes=yes,
                non_interactive=non_interactive,
                json_output=json_output,
                default=True,
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
    official_packs: Annotated[
        str | None, typer.Option("--official-packs", help="Official packs: ask, skip, or allow.")
    ] = None,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.workflow import update_command

    execute(
        "update",
        lambda: update_command(
            selector,
            all_collections=all_collections,
            repo=repo,
            dry_run=dry_run,
            official_pack_policy=official_packs,
            official_confirmation=_official_confirmation_callback(
                non_interactive=non_interactive, json_output=json_output, quiet=quiet
            ),
        ),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("submit")
def submit(
    target: Annotated[str | None, typer.Argument(help="Path or run ID.")] = None,
    repo: Annotated[str | None, typer.Option("--repo")] = None,
    publish: Annotated[
        str | None,
        typer.Option("--publish", help="Publication mode: local (no upload) or pr."),
    ] = None,
    direct_push: Annotated[bool, typer.Option("--direct-push")] = False,
    base: Annotated[str | None, typer.Option("--base")] = None,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Validate and publish a staged run or local data change."""

    from mojilex_cli.commands.workflow import submit_command

    def action():  # type: ignore[no-untyped-def]
        return submit_command(
            target,
            repo=repo,
            publish=publish,
            direct_push=direct_push,
            base=base,
            confirmation=_confirmation_callback(
                yes=yes,
                non_interactive=non_interactive,
                json_output=json_output,
                default=True,
            ),
        )

    execute("submit", action, json_output=json_output, quiet=quiet, debug=debug)


@app.command("build-index")
def build_index_cli(
    snapshot_id: Annotated[
        str,
        typer.Option("--snapshot-id", help="Immutable snapshot ID: data-YYYY.MM.DD.N."),
    ],
    source_date_epoch: Annotated[
        int,
        typer.Option(
            "--source-date-epoch",
            min=0,
            help="Immutable release source time as a Unix epoch.",
        ),
    ],
    path: Annotated[Path, typer.Argument(help="Local dataset root.")] = Path("."),
    output: Annotated[Path | None, typer.Option("--output")] = None,
    git_commit: Annotated[str | None, typer.Option("--git-commit")] = None,
    tool_commit: Annotated[str | None, typer.Option("--tool-commit")] = None,
    dependency_lock_sha256: Annotated[str | None, typer.Option("--dependency-lock-sha256")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    execute(
        "build-index",
        lambda: build_index_command(
            path,
            output,
            snapshot_id=snapshot_id,
            source_date_epoch=source_date_epoch,
            git_commit=git_commit,
            tool_commit=tool_commit,
            dependency_lock_sha256=dependency_lock_sha256,
        ),
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
        typer.echo(ui_text(f"Temporary comparison preview: {preview}"))
        return str(
            typer.prompt(
                ui_text("Decision [same-artwork/variant-of/related-series/not-duplicate/skip]")
            )
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
    max_ai_requests: Annotated[
        str | None, typer.Option("--max-ai-requests", help="Whole-run request limit or unlimited.")
    ] = None,
    official_packs: Annotated[
        str | None, typer.Option("--official-packs", help="ask, skip, or allow official packs.")
    ] = None,
    ai_concurrency: Annotated[int | None, typer.Option("--ai-concurrency", min=1)] = None,
    download_concurrency: Annotated[
        int | None, typer.Option("--download-concurrency", min=1)
    ] = None,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.workflow import resume_command

    request_limit = _request_limit(max_ai_requests)

    def action():  # type: ignore[no-untyped-def]
        from mojilex_cli.commands.packs import _pack_phase_status, _selector_name, resolve_pack_run

        checkpoint = resolve_pack_run(run_id, purpose="resume")
        selected_run_id = checkpoint.run_id
        phase, status = _pack_phase_status(checkpoint, _selector_name(run_id))
        if status in {"succeeded", "noop"}:
            next_command = "describe" if phase == "import" else "show"
            return CommandResult(
                run_id=selected_run_id,
                status="noop",  # type: ignore[arg-type]
                result={
                    "message": "This run is already complete.",
                    "next": f"mojilex {next_command} {run_id}",
                },
            )
        return _with_runtime_secrets(
            lambda: resume_command(
                run_id,
                max_ai_requests=request_limit,
                official_pack_policy=official_packs,
                official_confirmation=_official_confirmation_callback(
                    non_interactive=non_interactive, json_output=json_output, quiet=quiet
                ),
                ai_concurrency=ai_concurrency,
                download_concurrency=download_concurrency,
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
            names=(
                ("TELEGRAM_BOT_TOKEN",)
                if phase == "import"
                else ("TELEGRAM_BOT_TOKEN", "GEMINI_API_KEY")
            ),
            prompt=_secret_prompt(
                non_interactive=non_interactive,
                json_output=json_output,
                quiet=quiet,
            ),
        )

    execute(
        "resume",
        lambda: _pack_action(action, [run_id]),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("review")
def review(
    emoji_id: Annotated[str, typer.Argument()],
    action: Annotated[
        str | None, typer.Argument(help="Optional: approve, request-changes, or reject")
    ] = None,
    repo: Annotated[Path, typer.Option("--repo")] = Path("."),
    reviewer: Annotated[str | None, typer.Option("--reviewer")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    """Browse a pack; optionally record an explicit review of one emoji."""
    from mojilex_cli.commands.interactive import is_interactive
    from mojilex_cli.commands.packs import show_pack_command

    if (
        action is None
        and is_interactive()
        and not (json_output or quiet or machine_output_mode_requested())
    ):
        show_pack(emoji_id, json_output=json_output, quiet=quiet, debug=debug)
        return

    execute(
        "review",
        lambda: (
            show_pack_command(emoji_id, review=True)
            if action is None
            else review_command(repo, emoji_id, action, reviewer)
        ),
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
    install: Annotated[
        bool,
        typer.Option(
            "--install",
            help="Install missing Windows media dependencies, then rerun all checks.",
        ),
    ] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    def action() -> CommandResult:
        result = doctor_command()
        install_commands = cast(list[object], result.result.get("install_commands", []))
        if result.result.get("ready") or not install_commands:
            return result
        if install and (json_output or quiet):
            raise CommandError(
                "CONFIG_INVALID",
                "Interactive installer output is unavailable with --json or --quiet.",
                hint="Rerun `mojilex doctor --install` in a normal terminal.",
            )
        authorized = install
        if (
            not authorized
            and not non_interactive
            and not json_output
            and not quiet
            and sys.stdin.isatty()
        ):
            with suspend_progress():
                authorized = ui_confirm(
                    ui_text(
                        "Install the missing Windows media components now? "
                        "This may install FFmpeg or Visual Studio Build Tools."
                    ),
                    default=False,
                    err=True,
                )
        if not authorized:
            return result
        with suspend_progress():
            return install_media_dependencies_command(cast(dict[str, Any], result.result["checks"]))

    execute("doctor", action, json_output=json_output, quiet=quiet, debug=debug)


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


@config_app.command("set-credentials")
def config_set_credentials(
    telegram: Annotated[
        bool, typer.Option("--telegram/--no-telegram", help="Save a Telegram Bot API token.")
    ] = True,
    gemini: Annotated[
        bool, typer.Option("--gemini/--no-gemini", help="Save a Gemini API key.")
    ] = True,
    openai: Annotated[bool, typer.Option("--openai", help="Also save an OpenAI API key.")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    selected = [
        name
        for name, enabled in (
            ("TELEGRAM_BOT_TOKEN", telegram),
            ("GEMINI_API_KEY", gemini),
            ("OPENAI_API_KEY", openai),
        )
        if enabled
    ]

    def action() -> CommandResult:
        prompt = _secret_prompt(
            non_interactive=non_interactive,
            json_output=json_output,
            quiet=quiet,
        )
        if not selected:
            raise CommandError(
                "CONFIG_INVALID",
                "Select at least one credential to save.",
                hint="Enable --telegram, --gemini, or --openai.",
            )
        if prompt is None:
            raise CommandError(
                "CONFIG_INVALID",
                "Saving credentials requires an interactive terminal with hidden input.",
                hint="Rerun without --non-interactive, --json, or --quiet.",
            )
        labels = {
            "TELEGRAM_BOT_TOKEN": "Telegram Bot API token",
            "GEMINI_API_KEY": "Gemini API key",
            "OPENAI_API_KEY": "OpenAI API key",
        }
        values = {name: prompt(ui_text(labels[name])) for name in selected}
        return config_set_credentials_command(values)

    execute(
        "config set-credentials",
        action,
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@config_app.command("set-ui-language")
def config_set_ui_language(
    language: Annotated[str, typer.Argument(help="Human interface language: en or ru.")],
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    from mojilex_cli.commands.system import config_set_ui_language_command

    execute(
        "config set-ui-language",
        lambda: config_set_ui_language_command(language),
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@config_app.command("clear-credentials")
def config_clear_credentials(
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    def action() -> CommandResult:
        require_confirmation(
            ui_text("Delete every Telegram, Gemini, and OpenAI credential saved by MojiLex?"),
            yes=yes,
            non_interactive=non_interactive,
            json_output=json_output,
        )
        return config_clear_credentials_command()

    execute(
        "config clear-credentials",
        action,
        json_output=json_output,
        quiet=quiet,
        debug=debug,
    )


@app.command("uninstall")
def uninstall(
    keep_data: Annotated[
        bool,
        typer.Option("--keep-data", help="Keep configuration, run data, cache, and credentials."),
    ] = False,
    yes: Annotated[bool, typer.Option("--yes")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    quiet: Annotated[bool, typer.Option("--quiet")] = False,
    debug: Annotated[bool, typer.Option("--debug")] = False,
) -> None:
    def action() -> CommandResult:
        preview = uninstall_preview_command(keep_data=keep_data)
        require_confirmation(
            ui_text("Completely uninstall MojiLex with this exact plan: ") + str(preview),
            yes=yes,
            non_interactive=non_interactive,
            json_output=json_output,
        )
        return uninstall_command(keep_data=keep_data)

    execute("uninstall", action, json_output=json_output, quiet=quiet, debug=debug)


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


def _request_limit(value: str | None) -> int | Literal["unlimited"] | None:
    if value is None:
        return None
    if value.strip().lower() == "unlimited":
        return "unlimited"
    try:
        parsed = int(value)
        if parsed < 0:
            raise ValueError("negative request budget")
        return parsed
    except ValueError as exc:
        raise typer.BadParameter(
            "Use a non-negative integer or unlimited.", param_hint="--max-ai-requests"
        ) from exc


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
        "list",
        "show",
        "publish",
        "gallery",
        "settings",
        "add",
        "benchmark-dedupe",
        "benchmark-model",
        "build-index",
        "cache",
        "config",
        "dedupe",
        "describe",
        "doctor",
        "get",
        "get-collection",
        "import",
        "init",
        "resume",
        "review",
        "resolve",
        "search",
        "set-status",
        "similar",
        "snapshot",
        "snapshots",
        "submit",
        "takedown",
        "update",
        "uninstall",
        "validate",
    }
    for index, argument in enumerate(argv):
        if argument not in commands:
            continue
        if argument in {"cache", "config", "dedupe", "snapshot"}:
            for child in argv[index + 1 :]:
                if not child.startswith("-"):
                    return (
                        f"{argument}-{child}" if argument == "snapshot" else f"{argument} {child}"
                    )
        return argument
    return "mojilex"


def _raise_boundary_error(exc: BaseException) -> NoReturn:
    raise exc


def _emit_boundary_error(command: str, exc: BaseException) -> NoReturn:
    try:
        if command in {
            "snapshots",
            "snapshot-pull",
            "snapshot-verify",
            "snapshot-update",
            "search",
            "get",
            "get-collection",
            "resolve",
            "similar",
        }:
            execute_read(
                command,
                lambda: _raise_boundary_error(exc),
                json_output=True,
            )
        execute(command, lambda: _raise_boundary_error(exc), json_output=True)
    except typer.Exit as exit_error:
        raise SystemExit(exit_error.exit_code) from exc
    raise AssertionError("an error envelope must terminate with a non-zero exit code")


def main() -> None:
    json_requested, argv = _extract_json_flag(sys.argv[1:])
    try:
        ui_language, argv = extract_ui_language(argv)
    except ValueError as exc:
        if json_requested:
            _emit_boundary_error(
                _command_label(argv),
                CommandError(
                    "CONFIG_INVALID",
                    str(exc),
                    hint="Pass --ui-language en or --ui-language ru.",
                ),
            )
        typer.echo(f"Error: {exc}", err=True)
        raise SystemExit(2) from exc
    label = _command_label(argv)
    with use_ui_language(ui_language):
        command = typer.main.get_command(app)
        localize_command_tree(command, ui_language)
        if not json_requested:
            command.main(
                args=argv,
                prog_name="mojilex",
                windows_expand_args=False,
            )
            return

    with use_ui_language(ui_language), machine_output_mode():
        if any(argument in {"-h", "--help", "--help-all", "--version"} for argument in argv):
            _emit_boundary_error(
                label,
                CommandError(
                    "CONFIG_INVALID",
                    "Human help and version output are unavailable in JSON mode.",
                    hint="Remove --json to view help or version information.",
                ),
            )
        command = typer.main.get_command(app)
        localize_command_tree(command, ui_language)
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
