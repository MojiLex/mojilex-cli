"""Offline public-description lookup, independent of authoring credentials."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from mojilex_cli.commands.runtime import CommandError, CommandResult, execute
from mojilex_cli.config.paths import default_workspace_dir
from mojilex_cli.dataset.layout import assert_no_link_or_reparse
from mojilex_cli.read.lookup import (
    PUBLIC_REPOSITORY_URL,
    LookupError,
    LookupIndex,
    validate_public_repository,
)
from mojilex_cli.read.project import read_preferences, save_preferences

lookup_app = typer.Typer(help="Find published emoji descriptions without paid API calls.")
project_app = typer.Typer(help="Save an emoji pack choice for this project and both agents.")
lookup_app.add_typer(project_app, name="project")


def default_lookup_repository() -> Path:
    """Keep the public read cache separate from authoring runs and partial clones."""

    return default_workspace_dir() / "public-lookup-repository"


def _index(repo: Path | None) -> LookupIndex:
    repository = repo.expanduser().absolute() if repo is not None else default_lookup_repository()
    if not repository.is_dir():
        raise CommandError(
            "CONFIG_MISSING",
            "The cached public MojiLex source is not available.",
            hint="Run mojilex lookup refresh once, then search offline without API keys.",
        )
    return LookupIndex(repository, default_workspace_dir() / "lookup" / "index")


def _result(action: Callable[[], dict[str, Any]]) -> CommandResult:
    try:
        value = action()
    except LookupError as exc:
        code = {
            "LOOKUP_QUERY": "QUERY_INVALID",
            "LOOKUP_NOT_FOUND": "ENTITY_NOT_FOUND",
            "LOOKUP_AMBIGUOUS": "AMBIGUOUS_REFERENCE",
            "LOOKUP_CACHE": "INDEX_CORRUPT",
            "LOOKUP_LIMIT": "RESOURCE_LIMIT_EXCEEDED",
            "LOOKUP_BUSY": "GIT_CONFLICT",
        }.get(exc.code, "CONFIG_INVALID")
        raise CommandError(
            code,
            str(exc),
            hint=exc.hint or "Check the query and public source with mojilex lookup info.",
        ) from exc
    metadata = value.get("index", value)
    return CommandResult(result=value, warnings=list(metadata.get("warnings", [])))


def _public_network_git(*arguments: str) -> None:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith(("GIT_", "MOJILEX_GIT_")) or name in {
            "TELEGRAM_BOT_TOKEN",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "GH_TOKEN",
            "GITHUB_TOKEN",
        }:
            environment.pop(name, None)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    prefix = [
        "git",
        "--no-replace-objects",
        "-c",
        "credential.helper=",
        "-c",
        "core.askPass=",
        "-c",
        f"core.hooksPath={os.devnull}",
    ]
    try:
        if arguments[:1] == ("-C",):
            location = ["-C", arguments[1]]
            target = PUBLIC_REPOSITORY_URL + ".git"
            effective = subprocess.run(
                [*prefix, *location, "ls-remote", "--get-url", target],
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=30,
                shell=False,
                check=False,
            )
            headers = subprocess.run(
                [*prefix, *location, "config", "--get-regexp", r"^http\..*extraheader$"],
                env=environment,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=30,
                shell=False,
                check=False,
            )
            if (
                effective.returncode
                or effective.stdout.decode("utf-8").strip() != target
                or headers.returncode != 1
            ):
                raise CommandError(
                    "CONFIG_INVALID",
                    "Public lookup cannot use redirected or authenticated Git URLs.",
                    hint=(
                        "Use the managed lookup source without --repo, or remove the clone's "
                        "URL rewrites and extra headers."
                    ),
                )
        completed = subprocess.run(
            [*prefix, *arguments],
            env=environment,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=120,
            shell=False,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CommandError(
            "NETWORK_ERROR",
            "Could not refresh the public MojiLex source.",
            hint=(
                "Check Git and internet access; "
                "an existing cached source remains available offline."
            ),
        ) from exc
    if completed.returncode:
        # Diagnostics from Git can contain credentials/configured URLs. Keep them out of output.
        raise CommandError(
            "NETWORK_ERROR",
            "Git could not download the public MojiLex main revision.",
            hint="Check internet access; use lookup refresh --offline with an existing source.",
        )


def _refresh(repo: Path | None, *, offline: bool) -> dict[str, Any]:
    repository = repo.expanduser().absolute() if repo is not None else default_lookup_repository()
    assert_no_link_or_reparse(repository)
    if not offline:
        if repository.exists():
            validate_public_repository(repository)
            _public_network_git(
                "-C",
                str(repository),
                "fetch",
                "--no-tags",
                PUBLIC_REPOSITORY_URL + ".git",
                "refs/heads/main:refs/remotes/origin/main",
            )
        else:
            if repo is not None:
                raise CommandError(
                    "CONFIG_MISSING",
                    "The --repo override must be an existing public clone.",
                    hint="Omit --repo to create the managed public lookup source.",
                )
            repository.parent.mkdir(parents=True, exist_ok=True)
            # Only this task-owned temporary clone is cleaned up on failure.
            with tempfile.TemporaryDirectory(
                prefix="mojilex-public-lookup-", dir=repository.parent
            ) as raw:
                source = Path(raw) / "source"
                _public_network_git(
                    "clone",
                    "--no-checkout",
                    "--single-branch",
                    "--branch",
                    "main",
                    "--no-tags",
                    PUBLIC_REPOSITORY_URL + ".git",
                    str(source),
                )
                validate_public_repository(source)
                if repository.exists():
                    raise CommandError(
                        "GIT_CONFLICT",
                        "The public lookup source was created by another process.",
                        hint="Rerun lookup refresh to use the existing public source.",
                    )
                os.replace(source, repository)
    value = _index(repo).refresh(force=True)
    value["public_source_fetched"] = not offline
    return value


@lookup_app.command("search")
def search_cli(
    query: Annotated[str, typer.Argument(help="Words, pack name, or emoji ID to find.")],
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 10,
    offset: Annotated[int, typer.Option("--offset", min=0, max=100000)] = 0,
    pack: Annotated[
        list[str] | None, typer.Option("--pack", help="Exact pack name, ID or URL; repeatable.")
    ] = None,
    project: Annotated[
        Path | None,
        typer.Option("--project", help="Project directory with saved pack preferences."),
    ] = None,
    repo: Annotated[
        Path | None, typer.Option("--repo", help="Full clone of public MojiLex/mojilex.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Search published descriptions, respecting the project's selected emoji packs."""

    def search() -> dict[str, Any]:
        preferences = read_preferences(project) if not pack else None
        chosen = pack or (preferences["packs"] if preferences else [])
        result = _index(repo).search(query, limit=limit, offset=offset, packs=chosen)
        result["selection"] = {
            "source": "explicit" if pack else "project" if chosen else "all",
            "path": preferences["path"] if preferences else None,
        }
        return result

    execute(
        "lookup search",
        lambda: _result(search),
        json_output=json_output,
    )


@lookup_app.command("packs")
def packs_cli(
    query: Annotated[
        str, typer.Argument(help="Optional words for pack names or emoji context.")
    ] = "",
    limit: Annotated[int, typer.Option("--limit", min=1, max=100)] = 20,
    offset: Annotated[int, typer.Option("--offset", min=0, max=100000)] = 0,
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Discover published packs with Telegram links and example descriptions."""
    execute(
        "lookup packs",
        lambda: _result(lambda: _index(repo).packs(query, limit=limit, offset=offset)),
        json_output=json_output,
    )


@project_app.command("show")
def project_show_cli(
    project: Annotated[Path | None, typer.Option("--project")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Read the saved choice without API keys or a public index."""
    execute(
        "lookup project show",
        lambda: _result(lambda: read_preferences(project)),
        json_output=json_output,
    )


@project_app.command("select")
def project_select_cli(
    pack: Annotated[list[str], typer.Option("--pack", help="Chosen pack name or Telegram URL.")],
    project: Annotated[Path | None, typer.Option("--project")] = None,
    repo: Annotated[Path | None, typer.Option("--repo")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Save only the packs explicitly chosen for the project."""

    def select() -> dict[str, Any]:
        packs = _index(repo).resolve_packs(pack)
        return {
            **save_preferences([item["name"] for item in packs], project),
            "selected_packs": packs,
        }

    execute("lookup project select", lambda: _result(select), json_output=json_output)


@lookup_app.command("get")
def get_cli(
    identifier: Annotated[str, typer.Argument(help="Canonical or Telegram custom emoji ID.")],
    repo: Annotated[
        Path | None, typer.Option("--repo", help="Full clone of public MojiLex/mojilex.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Read one saved public description, identity, pack, and provenance."""

    execute(
        "lookup get", lambda: _result(lambda: _index(repo).get(identifier)), json_output=json_output
    )


@lookup_app.command("info")
def info_cli(
    repo: Annotated[
        Path | None, typer.Option("--repo", help="Full clone of public MojiLex/mojilex.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Report public source version, coverage, and review status offline."""

    execute("lookup info", lambda: _result(lambda: _index(repo).info()), json_output=json_output)


@lookup_app.command("refresh")
def refresh_cli(
    offline: Annotated[
        bool, typer.Option("--offline", help="Rebuild without contacting GitHub.")
    ] = False,
    repo: Annotated[
        Path | None, typer.Option("--repo", help="Full clone of public MojiLex/mojilex.")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Fetch public main and rebuild the lookup cache; never generate descriptions."""

    execute(
        "lookup refresh",
        lambda: _result(lambda: _refresh(repo, offline=offline)),
        json_output=json_output,
    )


def register_lookup_commands(app: typer.Typer) -> None:
    app.add_typer(lookup_app, name="lookup")
