"""Terminal navigation over the existing commands; browsing never starts analysis."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from typing import Any

import typer
from rich.console import Console
from rich.text import Text

from mojilex_cli.config import load_config
from mojilex_cli.i18n import confirm, current_ui_language, use_ui_language

from .packs import (
    _group,
    _names,
    _pack_phase_status,
    _resolve,
    _runs,
    _selector_name,
    _source_name,
    _sources,
    _summary,
    list_packs_command,
    show_pack_command,
)
from .runtime import CommandError, CommandResult, is_usage_error

Dispatch = Callable[[list[str]], bool]


def label(ru: str, en: str) -> str:
    return ru if current_ui_language() == "ru" else en


def is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def select(title: str, options: Sequence[str], *, detail: str = "") -> int | None:
    """Arrow navigation, with a bounded screen for large packs."""
    if not options:
        return None
    console = Console()
    position = 0
    while True:
        console.clear()
        console.print(Text(title, style="bold cyan"))
        if detail:
            console.print(Text(detail))
        console.print()
        detail_rows = len(console.render_lines(Text(detail), console.options)) if detail else 0
        page_size = max(1, min(10, console.size.height - detail_rows - 6))
        start = (position // page_size) * page_size
        for index in range(start, min(start + page_size, len(options))):
            console.print(
                Text(
                    ("> " if index == position else "  ") + options[index],
                    style="bold cyan" if index == position else "",
                ),
                overflow="ellipsis",
                no_wrap=True,
            )
        console.print(
            Text(
                label(
                    f"\n↑↓ Выбрать · Enter Открыть · Esc Назад · {position + 1}/{len(options)}",
                    f"\n↑↓ Select · Enter Open · Esc Back · {position + 1}/{len(options)}",
                ),
                style="dim",
            )
        )
        try:
            key = typer.getchar()
        except (KeyboardInterrupt, EOFError):
            return None
        if key in {"\x1b", "q", "Q"}:
            return None
        if key in {"\r", "\n"}:
            return position
        if key in {"\x1b[A", "\xe0H", "\x00H", "k"}:
            position = (position - 1) % len(options)
        if key in {"\x1b[B", "\xe0P", "\x00P", "j"}:
            position = (position + 1) % len(options)


def pause() -> None:
    typer.echo(label("\nEnter — вернуться", "\nEnter — return"))
    try:
        typer.getchar()
    except (KeyboardInterrupt, EOFError):
        pass


def _notice(message: str) -> None:
    Console().print(Text(message))
    pause()


def _invoke(dispatch: Dispatch, arguments: list[str]) -> bool:
    try:
        return dispatch(arguments)
    finally:
        pause()


def _description(item: dict[str, Any], language: str) -> str:
    descriptions = item.get("descriptions", {})
    value: dict[str, Any] = descriptions.get(language) or next(iter(descriptions.values()), {})
    return str(value.get("text", ""))


def browse_descriptions(selector: str) -> CommandResult:
    with Console().status(label("Читаю сохранённые описания…", "Reading saved descriptions…")):
        saved = show_pack_command(selector)
    items = saved.result["items"]
    language = "ru" if current_ui_language() == "ru" else "en"
    query = ""
    title = ", ".join(saved.result["pack"]["names"])
    counts = saved.result["counts"]
    while True:
        filtered = [
            item
            for item in items
            if query.casefold() in _description(item, language).casefold()
            or query.casefold() in str(item["native_id"]).casefold()
        ]
        options = [
            label("Поиск по описанию", "Search descriptions"),
            label(
                f"Язык: {language.upper()} — переключить", f"Language: {language.upper()} — switch"
            ),
        ]
        options.extend(
            f"{index + 1}. {_description(item, language)[:110]}"
            for index, item in enumerate(filtered)
        )
        detail = label(
            f"Сохранено {counts['ready']}/{saved.result['pack']['items']}. "
            f"Найдено: {len(filtered)}. Поиск: {query or '—'}",
            f"Saved {counts['ready']}/{saved.result['pack']['items']}. "
            f"Matches: {len(filtered)}. Search: {query or '—'}",
        )
        if saved.warnings:
            detail += label(
                "\nНекоторые описания недоступны. Прогресс сохранён.",
                "\nSome descriptions are unavailable. Progress is preserved.",
            )
        selected = select(title, options, detail=detail)
        if selected is None:
            return saved
        if selected == 0:
            query = str(
                typer.prompt(
                    label("Поиск (пусто — все)", "Search (empty — all)"),
                    default="",
                    show_default=False,
                )
            )
        elif selected == 1:
            language = "en" if language == "ru" else "ru"
        else:
            _browse_item(filtered[selected - 2], language)


def _read_text(title: str, text: str) -> None:
    """Keep details visible without depending on a system pager or its exit policy."""
    console = Console()
    offset = 0
    while True:
        lines = Text(text).wrap(console, max(1, console.size.width))
        page_size = max(1, console.size.height - 4)
        last_offset = max(0, len(lines) - page_size)
        offset = min(offset, last_offset)
        console.clear()
        console.print(Text(title, style="bold cyan"), no_wrap=True, overflow="ellipsis")
        for line in lines[offset : offset + page_size]:
            console.print(line, no_wrap=True, overflow="crop")
        console.print(
            Text(
                label(
                    "\n↑↓ Прокрутка · PgUp/PgDn Страница · Enter/Esc Назад",
                    "\n↑↓ Scroll · PgUp/PgDn Page · Enter/Esc Back",
                )
                + f" · {offset + 1}–{min(offset + page_size, len(lines))}/{len(lines)}",
                style="dim",
            ),
            no_wrap=True,
            overflow="ellipsis",
        )
        try:
            key = typer.getchar()
        except (KeyboardInterrupt, EOFError):
            return
        if key in {"\r", "\n", "\x1b", "q", "Q"}:
            return
        if key in {"\x1b[A", "\xe0H", "\x00H", "k"}:
            offset = max(0, offset - 1)
        elif key in {"\x1b[B", "\xe0P", "\x00P", "j"}:
            offset = min(last_offset, offset + 1)
        elif key in {"\x1b[5~", "\xe0I", "\x00I"}:
            offset = max(0, offset - page_size)
        elif key in {"\x1b[6~", "\xe0Q", "\x00Q", " "}:
            offset = min(last_offset, offset + page_size)
        elif key in {"\x1b[H", "\x1b[1~", "\xe0G", "\x00G"}:
            offset = 0
        elif key in {"\x1b[F", "\x1b[4~", "\xe0O", "\x00O"}:
            offset = last_offset


def _browse_item(item: dict[str, Any], language: str) -> None:
    detailed = False
    while True:
        description = item.get("descriptions", {}).get(language, {})
        lines = [_description(item, language)]
        if description.get("motion"):
            lines.append(label("Движение: ", "Motion: ") + description["motion"])
        warnings = item.get("content", {}).get("warnings", [])
        if warnings:
            lines.append(label("Предупреждения: ", "Warnings: ") + ", ".join(warnings))
        if detailed:
            for lang, value in item.get("descriptions", {}).items():
                lines.append(f"\n{lang.upper()}: {value['text']}")
                if value.get("usage"):
                    lines.append(label("Примеры: ", "Examples: ") + "; ".join(value["usage"]))
            lines.append(label("Теги: ", "Tags: ") + ", ".join(item.get("semantic_tags", [])))
            lines.append(
                label("Категория контента: ", "Content rating: ")
                + str(item.get("content", {}).get("rating", ""))
            )
            for key, value in item.get("facets", {}).items():
                if value:
                    lines.append(f"{key}: {value}")
            _read_text(
                label("Все поля и английский текст", "All fields and English text"),
                "\n\n".join(lines),
            )
            detailed = False
            continue
        choice = select(
            label("Описание эмодзи", "Emoji description"),
            [
                label(
                    "Скрыть подробности" if detailed else "Все поля и английский текст",
                    "Hide details" if detailed else "All fields and English text",
                ),
                label("Назад к списку", "Back to list"),
            ],
            detail="\n\n".join(lines),
        )
        if choice in {None, 1}:
            return
        detailed = not detailed


def _pack_state(selector: str) -> dict[str, Any]:
    config = load_config()
    checkpoint = _resolve(selector, config, purpose="view")
    runs, _ = _runs(config)
    selected_name = _selector_name(selector)
    history = [
        run
        for run in runs
        if run.target_repository == checkpoint.target_repository
        and (
            selected_name.casefold() in {name.casefold() for name in _names(run)}
            if selected_name
            else _group(run) == _group(checkpoint)
        )
    ]
    unfinished = next(
        (
            run
            for run in history
            if _pack_phase_status(run, selected_name)[1] not in {"succeeded", "noop"}
        ),
        None,
    )
    phase, status = _pack_phase_status(checkpoint, selected_name)
    saved = show_pack_command(selector)
    publication = checkpoint.publication
    # Absence of a receipt is not evidence that the pack was never published.
    github = label(
        "Не проверен — откройте GitHub для актуального статуса",
        "Not checked — open GitHub for current status",
    )
    if publication is not None:
        github = label("Есть сохранённая попытка отправки: ", "Saved publication attempt: ")
        github += publication.phase
    return {
        "saved": saved,
        "history": [
            (_summary(run, config, selected_name) if selected_name else _summary(run, config))
            | {"command": _pack_phase_status(run, selected_name)[0]}
            for run in history
        ],
        "unfinished": (
            _summary(unfinished, config, selected_name)
            if selected_name
            else _summary(unfinished, config)
        )
        if unfinished
        else None,
        "github": github,
        "target": checkpoint.target_repository,
        "publishable": phase == "describe" and status in {"succeeded", "noop"},
        "analyzable": phase == "import"
        and status in {"succeeded", "noop"}
        and bool(saved.result["pack"]["items"] if selected_name else checkpoint.elements),
        "run_id": checkpoint.run_id,
        "selector": selector,
        "sources": [
            source
            for source in _sources(checkpoint)
            if selected_name is None
            or (_source_name(source) or "").casefold() == selected_name.casefold()
        ],
    }


def _pack_page(selector: str, dispatch: Dispatch) -> None:
    while True:
        with Console().status(label("Открываю сохранённый пак…", "Opening saved pack…")):
            state = _pack_state(selector)
        saved = state["saved"].result
        pack = saved["pack"]
        name = ", ".join(pack["names"])
        warnings = sum(bool(item.get("content", {}).get("warnings")) for item in saved["items"])
        detail = (
            label(
                f"Готовые описания: {saved['counts']['ready']}/{pack['items']}\n"
                f"Эмодзи с предупреждениями: {warnings} (одобрение не требуется)\n",
                f"Saved descriptions: {saved['counts']['ready']}/{pack['items']}\n"
                f"Emojis with warnings: {warnings} (no approval needed)\n",
            )
            + f"GitHub: {state['github']}\n"
            + label(
                (
                    "Запросы ИИ всей массовой операции: "
                    if pack.get("budget_scope") == "batch"
                    else "Запросы ИИ этого результата: "
                )
                + f"{pack['requests_used']}/"
                + _setting_value(pack["max_ai_requests"], key="max_ai_requests"),
                (
                    "AI requests for the whole batch: "
                    if pack.get("budget_scope") == "batch"
                    else "AI requests for this result: "
                )
                + f"{pack['requests_used']}/"
                + _setting_value(pack["max_ai_requests"], key="max_ai_requests"),
            )
        )
        if saved.get("compositions"):
            count = len(saved["compositions"])
            detail += label(
                f"\nСвязанные группы фрагментов: {count}. Сборки — в галерее.",
                f"\nRelated fragment groups: {count}. Open gallery to view assemblies.",
            )
        if state["analyzable"]:
            detail += label(
                "\nМедиа сохранены. Можно запустить анализ ИИ из этого импорта.",
                "\nMedia are saved. AI analysis can start from this import.",
            )
        active = state["unfinished"]
        if active:
            detail += label(
                f"\nОтдельная незавершённая обработка: {active['ai_ready']}/{active['items']}. "
                "Продолжение относится к ней.",
                f"\nSeparate unfinished processing: {active['ai_ready']}/{active['items']}. "
                "Continue resumes that run.",
            )
        actions = ["show", "gallery", "history"]
        titles = [
            label("Посмотреть описания", "Browse descriptions"),
            label("Открыть галерею в браузере", "Open browser gallery"),
            label("История обработки", "Processing history"),
        ]
        if active:
            actions.append("resume")
            titles.append(
                label("Продолжить незавершённую обработку", "Continue unfinished processing")
            )
        if state["analyzable"]:
            actions.append("describe")
            titles.append(
                label("Проанализировать сохранённый импорт с ИИ", "Analyze saved import with AI")
            )
        if state["publishable"]:
            actions.extend(["check", "publish"])
            titles.extend(
                [
                    label("Проверить готовность к отправке", "Check publication readiness"),
                    label("Отправить на GitHub через PR", "Send to GitHub via PR"),
                ]
            )
        actions.append("refresh")
        titles.append(label("Обновить пак из Telegram", "Refresh pack from Telegram"))
        selected = select(name, titles, detail=detail)
        if selected is None:
            return
        action = actions[selected]
        if action == "show":
            browse_descriptions(selector)
        elif action == "gallery":
            _invoke(dispatch, ["gallery", selector])
        elif action == "history":
            _history(state)
        elif action == "check":
            _invoke(dispatch, ["publish", state.get("selector", state["run_id"]), "--local"])
        elif action == "publish":
            if confirm(
                label(
                    f"Отправить {name} в {state['target']} через Pull Request?",
                    f"Send {name} to {state['target']} via a pull request?",
                ),
                default=True,
            ):
                _invoke(dispatch, ["publish", state.get("selector", state["run_id"]), "--yes"])
        elif action == "resume":
            selector = active.get("selector", active["run_id"])
            _invoke(dispatch, ["resume", selector])
        elif action == "describe":
            # The describe command owns the single batch-wide AI approval.
            _invoke(dispatch, ["describe", state.get("selector", state["run_id"])])
            return
        elif action == "refresh":
            sources = state["sources"]
            if len(sources) != 1:
                _notice(
                    label(
                        "Обновляйте паки по одной ссылке через главное меню.",
                        "Refresh one pack URL at a time from the main menu.",
                    )
                )
            elif confirm(
                label(
                    "Загрузить обновления и проанализировать недостающее?",
                    "Download updates and analyze missing descriptions?",
                ),
                default=True,
            ):
                _analyze(str(sources[0]), dispatch, repository=state["target"], refresh=True)
                return


def _history(state: dict[str, Any]) -> None:
    rows = [
        f"{run['updated_at'][:19]} · {run['command']} · "
        f"{run['ai_ready']}/{run['items']} · {run['status']}"
        for run in state["history"]
    ]
    index = select(label("История обработки", "Processing history"), rows)
    if index is not None:
        run = state["history"][index]
        _notice(
            label("Сведения о запуске\n", "Run details\n")
            + f"{run['run_id']}\n{run['requests_used']}/"
            + _setting_value(run["max_ai_requests"], key="max_ai_requests")
            + " "
            + label("запросов ИИ", "AI requests")
        )


def _analyze(
    source: str, dispatch: Dispatch, *, repository: str | None = None, refresh: bool = False
) -> None:
    from .queue_progress import pack_queue_scope

    with pack_queue_scope():
        _analyze_impl(source, dispatch, repository=repository, refresh=refresh)


def _analyze_impl(
    source: str, dispatch: Dispatch, *, repository: str | None = None, refresh: bool = False
) -> None:
    from .runtime import capture_command_results

    arguments = ["import", source]
    mode = load_config().processing.file_analysis_mode
    preparation = {
        "fast": "metadata",
        "sequential": "metadata",
        "download_all": "download_all",
        "prepare_all": "full",
    }[mode]
    arguments.extend(["--preparation", preparation])
    if refresh:
        arguments.append("--refresh")
    if repository is not None:
        arguments.extend(["--repo", repository])
    with capture_command_results() as results:
        success = dispatch(arguments)
    if not success:
        pause()
        return
    if results and results[-1].run_id is None and results[-1].status == "noop":
        # A policy may skip every requested official pack before creating a run.
        # Its command output already explains the skip; there is nothing to analyze.
        pause()
        return
    if not results or results[-1].run_id is None:
        _notice(
            label(
                "Не удалось определить сохранённый импорт. Откройте «Мои паки».",
                "Cannot identify the saved import. Open My packs.",
            )
        )
        return
    selectors = results[-1].result.get("analysis_selectors", [results[-1].run_id])
    if selectors:
        _invoke(dispatch, ["describe", *selectors])
    else:
        _notice(label("Все выбранные паки уже обработаны.", "All selected packs are complete."))


def _settings(dispatch: Dispatch) -> None:
    from .settings import settings_command, update_setting_command

    while True:
        settings = settings_command().result
        entries = settings["settings"]
        options = [
            f"{entry['label']}: {_setting_value(entry['value'], key=entry['key'])}"
            for entry in entries
        ]
        options.extend(
            [
                label("Первоначальная настройка", "Initial setup"),
                label("Настроить ключи доступа", "Configure credentials"),
            ]
        )
        choice = select(
            label("Настройки", "Settings"), options, detail="\n".join(settings["notes"])
        )
        if choice is None:
            return
        if choice == len(entries):
            _invoke(dispatch, ["init"])
        elif choice == len(entries) + 1:
            _invoke(dispatch, ["config", "set-credentials"])
        else:
            entry = entries[choice]
            explanation = entry["description"]
            if entry.get("choices"):
                explanation += label("\nВарианты: ", "\nChoices: ") + ", ".join(entry["choices"])
            if entry.get("minimum") is not None:
                explanation += label("\nМинимум: ", "\nMinimum: ") + str(entry["minimum"])
            if entry.get("maximum") is not None:
                explanation += label(" · Максимум: ", " · Maximum: ") + str(entry["maximum"])
            source_labels = {
                "environment": label("окружение", "environment"),
                "project": label("настройки проекта", "project settings"),
                "user": label("общие настройки", "user settings"),
                "default": label("по умолчанию", "default"),
            }
            explanation += label("\nИсточник: ", "\nSource: ") + source_labels[entry["source"]]
            Console().print(Text(explanation))
            if not entry["editable"]:
                _notice(
                    label(
                        "Значение задано окружением. Уберите переопределение перед изменением.",
                        "An environment override controls this setting. Remove it before editing.",
                    )
                    + "\n"
                    + str(entry.get("environment_variable", ""))
                )
                continue
            while True:
                try:
                    value = str(
                        typer.prompt(
                            label("Новое значение", "New value"),
                            default=(
                                "unlimited"
                                if entry["key"] == "max_ai_requests" and entry["value"] is None
                                else str(entry["value"])
                            ),
                        )
                    )
                    changed = update_setting_command(entry["key"], value).result
                except CommandError:
                    Console().print(
                        Text(
                            label(
                                "Не удалось сохранить. Проверьте значение "
                                "и доступ к файлу настроек.",
                                "Could not save. Check the value and access to the settings file.",
                            ),
                            style="yellow",
                        )
                    )
                    if confirm(
                        label("Попробовать другое значение?", "Try another value?"), default=True
                    ):
                        continue
                    break
                except (KeyboardInterrupt, EOFError, typer.Abort):
                    break
                message = (
                    f"{changed['label']}: "
                    + _setting_value(changed["value"], key=entry["key"])
                    + "\n"
                    + changed.get("note", "")
                )
                if entry["key"] == "ui_language":
                    message += label(
                        "\nНовый язык включится при следующем запуске mojilex.",
                        "\nThe new language applies when you next start mojilex.",
                    )
                _notice(message)
                break


def _setting_value(value: object, *, key: str | None = None) -> str:
    if key == "max_ai_requests" and value in {None, "unlimited"}:
        return label("Без лимита", "Unlimited")
    if value is None:
        return label("не задан", "not set")
    if value == "":
        return label("не настроено", "not configured")
    return str(value)


def run_menu(dispatch: Dispatch) -> None:
    while True:
        try:
            choice = select(
                "MojiLex",
                [
                    label("Мои паки", "My packs"),
                    label("Добавить паки по ссылке или из файла", "Add packs from a URL or file"),
                    label("Настройки", "Settings"),
                    label("Проверить подключение", "Check connections"),
                    label(
                        "Отправить все новые готовые паки на GitHub",
                        "Sync all new completed packs to GitHub",
                    ),
                    label("Выход", "Exit"),
                ],
                detail=label(
                    "Эмодзи → понятные текстовые описания", "Emoji → readable descriptions"
                ),
            )
            if choice in {None, 5}:
                return
            if choice == 0:
                rows = list_packs_command().result["packs"]
                if not rows:
                    _notice(
                        label(
                            "Паков пока нет. Выберите «Добавить пак по ссылке».",
                            "No packs yet. Choose Add a pack URL.",
                        )
                    )
                    continue
                index = select(
                    label("Мои паки", "My packs"),
                    [
                        f"{', '.join(row['names'])} · {row['ai_ready']}/{row['items']} "
                        + label("описаний", "descriptions")
                        for row in rows
                    ],
                )
                if index is not None:
                    _pack_page(rows[index].get("selector", rows[index]["run_id"]), dispatch)
            elif choice == 1:
                source = str(
                    typer.prompt(
                        label(
                            "Ссылка на пак Telegram или путь к файлу",
                            "Telegram pack URL or file path",
                        )
                    )
                ).strip()
                _analyze(source, dispatch)
            elif choice == 2:
                _settings(dispatch)
            elif choice == 3:
                _invoke(dispatch, ["doctor"])
            elif choice == 4:
                _invoke(dispatch, ["sync"])
        except (KeyboardInterrupt, EOFError, typer.Abort):
            continue
        except CommandError as exc:
            _notice(str(exc) + "\n" + exc.error.hint)
        except ValueError as exc:
            from mojilex_cli.config.secrets import redact_text

            _notice(redact_text(exc))


def dispatch_command(arguments: list[str]) -> bool:
    """Use the same command validation, credentials and budget checks as direct CLI."""
    from mojilex_cli.cli import app
    from mojilex_cli.i18n import localize_command_tree

    command = typer.main.get_command(app)
    language = current_ui_language()
    localize_command_tree(command, language)
    try:
        with use_ui_language(language):
            result = command.main(
                args=arguments,
                prog_name="mojilex",
                standalone_mode=False,
                windows_expand_args=False,
            )
        return result in {None, 0}
    except (KeyboardInterrupt, EOFError, typer.Abort):
        return False
    except Exception as exc:
        if is_usage_error(exc):
            typer.echo(str(exc), err=True)
            return False
        raise
