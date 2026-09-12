"""Human CLI localization without changing machine-readable contracts."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal

UiLanguage = Literal["en", "ru"]
SUPPORTED_UI_LANGUAGES: tuple[UiLanguage, ...] = ("en", "ru")
UI_LANGUAGE_ENV = "MOJILEX_UI_LANGUAGE"

_CURRENT_UI_LANGUAGE: ContextVar[UiLanguage] = ContextVar(
    "mojilex_ui_language", default="en"
)

_ROOT_COMMAND_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pack_workflow", ("add", "import", "describe", "submit", "resume", "update")),
    (
        "read",
        ("search", "get", "get-collection", "resolve", "similar", "snapshots"),
    ),
    ("quality", ("validate", "dedupe", "review", "set-status", "takedown")),
    ("releases", ("snapshot", "build-index", "benchmark-dedupe", "benchmark-model")),
    ("setup", ("init", "doctor", "config", "cache")),
)

_PANEL_TITLES: dict[str, tuple[str, str]] = {
    "pack_workflow": ("Pack analysis and publication", "Анализ и публикация паков"),
    "read": ("Search and read", "Поиск и чтение"),
    "quality": ("Validation and moderation", "Проверка и модерация"),
    "releases": ("Snapshots and benchmarks", "Снимки и тесты"),
    "setup": ("Setup and maintenance", "Настройка и обслуживание"),
}


_COMMAND_HELP: dict[str, tuple[str, str]] = {
    "mojilex": (
        "Build, validate, and publish the media-free MojiLex emoji dataset.",
        "Собирайте, проверяйте и публикуйте набор данных эмодзи MojiLex без исходных медиа.",
    ),
    "snapshots": (
        "List release snapshots from the configured catalog.",
        "Показать снимки релизов из настроенного каталога.",
    ),
    "search": (
        "Search a pinned local snapshot by text and filters.",
        "Искать в закреплённом локальном снимке по тексту и фильтрам.",
    ),
    "get": (
        "Read one emoji record from a pinned local snapshot.",
        "Показать одну запись эмодзи из закреплённого локального снимка.",
    ),
    "get-collection": (
        "Read a collection and its members from a pinned snapshot.",
        "Показать коллекцию и её элементы из закреплённого снимка.",
    ),
    "resolve": (
        "Resolve a platform-native reference to a canonical MojiLex identity.",
        "Найти канонический идентификатор MojiLex по ссылке платформы.",
    ),
    "similar": (
        "Find related or duplicate candidates for an emoji.",
        "Найти похожие эмодзи и кандидатов в дубликаты.",
    ),
    "init": (
        "Create non-secret settings; working commands request missing credentials.",
        "Создать несекретные настройки; рабочие команды сами запросят недостающие ключи.",
    ),
    "add": (
        "Import, analyze, validate, and optionally publish packs in one command.",
        "Импортировать, проанализировать, проверить и при необходимости "
        "опубликовать паки одной командой.",
    ),
    "import": (
        "Download and verify media in a persistent staging run without AI or publication.",
        "Скачать и проверить медиа в сохранённом черновике без AI и публикации.",
    ),
    "describe": (
        "Generate AI metadata for a staged run without publishing it.",
        "Создать AI-описания для черновика без публикации.",
    ),
    "validate": (
        "Validate a local dataset and report integrity errors.",
        "Проверить локальный набор данных и показать ошибки целостности.",
    ),
    "update": (
        "Refresh one or all existing collections.",
        "Обновить одну или все существующие коллекции.",
    ),
    "submit": (
        "Validate a staged run, preview locally, open a PR, or push directly.",
        "Проверить черновик, просмотреть локально, открыть PR или отправить напрямую.",
    ),
    "build-index": (
        "Build deterministic immutable snapshot artifacts.",
        "Собрать детерминированные неизменяемые файлы снимка.",
    ),
    "benchmark-dedupe": (
        "Run a versioned duplicate-detection benchmark.",
        "Запустить версионированный тест поиска дубликатов.",
    ),
    "benchmark-model": (
        "Run a live AI model benchmark against an adjudicated manifest.",
        "Проверить AI-модель по проверенному эталонному манифесту.",
    ),
    "resume": (
        "Resume a staged run by its run ID.",
        "Продолжить сохранённую операцию по её Run ID.",
    ),
    "review": (
        "Record a human review decision for an emoji.",
        "Записать решение ручной проверки эмодзи.",
    ),
    "set-status": (
        "Change an entity availability status.",
        "Изменить статус доступности сущности.",
    ),
    "takedown": (
        "Remove public fields and create a tombstone after confirmation.",
        "Удалить публичные поля и создать tombstone после подтверждения.",
    ),
    "doctor": (
        "Check configuration, credentials, GitHub access, and media tools.",
        "Проверить конфигурацию, ключи, доступ к GitHub и медиаинструменты.",
    ),
    "config": (
        "Inspect non-secret configuration.",
        "Просмотреть несекретную конфигурацию.",
    ),
    "config show": (
        "Show the resolved non-secret configuration.",
        "Показать итоговую несекретную конфигурацию.",
    ),
    "cache": (
        "Inspect or prune the content-addressed AI cache.",
        "Просмотреть или очистить адресуемый по содержимому AI-кеш.",
    ),
    "cache info": (
        "Show the AI cache path, size, and entry counts.",
        "Показать путь, размер и число записей AI-кеша.",
    ),
    "cache prune": (
        "Remove AI cache entries older than the selected age.",
        "Удалить записи AI-кеша старше указанного срока.",
    ),
    "dedupe": (
        "Scan and review exact or visual duplicate candidates.",
        "Найти и проверить точные или визуальные дубликаты.",
    ),
    "dedupe scan": (
        "Build or update exact and near-duplicate candidates.",
        "Построить или обновить список точных и похожих дубликатов.",
    ),
    "dedupe explain": (
        "Explain duplicate evidence for an emoji pair.",
        "Объяснить признаки дубликата для пары эмодзи.",
    ),
    "dedupe review": (
        "Review and save a duplicate relationship.",
        "Проверить и сохранить связь между дубликатами.",
    ),
    "snapshot": (
        "Verify or manage immutable release snapshots.",
        "Проверить или управлять неизменяемыми снимками релизов.",
    ),
    "snapshot pull": (
        "Fetch a named immutable snapshot from the configured mirror.",
        "Скачать указанный неизменяемый снимок с настроенного зеркала.",
    ),
    "snapshot update": (
        "Update to a named or latest immutable snapshot.",
        "Обновиться до указанного или последнего неизменяемого снимка.",
    ),
    "snapshot verify": (
        "Verify a local snapshot, hashes, schemas, and trust metadata.",
        "Проверить локальный снимок, хеши, схемы и данные доверия.",
    ),
}


_PARAMETER_HELP: dict[str, tuple[str, str]] = {
    "version": ("Show the installed version.", "Показать установленную версию."),
    "install_completion": (
        "Install completion for the current shell.",
        "Установить автодополнение для текущей оболочки.",
    ),
    "show_completion": (
        "Show the completion script for the current shell.",
        "Показать скрипт автодополнения для текущей оболочки.",
    ),
    "help": ("Show this message and exit.", "Показать эту справку и выйти."),
    "ui_language": (
        "Human interface language: en or ru. Commands and JSON fields stay unchanged.",
        "Язык интерфейса: en или ru. Команды и поля JSON не переводятся.",
    ),
    "repo": (
        "Existing local dataset path or GitHub OWNER/REPO.",
        "Существующий локальный путь к данным или GitHub OWNER/REPO.",
    ),
    "publish": (
        "Publication mode: local (no upload) or pr.",
        "Режим публикации: local (без загрузки) или pr.",
    ),
    "direct_push": (
        "Push directly to the base branch after validation and confirmation.",
        "После проверки и подтверждения отправить изменения прямо в базовую ветку.",
    ),
    "dry_run": (
        "Inspect without AI requests, persistent writes, or publication.",
        "Проверить без AI-запросов, постоянной записи и публикации.",
    ),
    "check_media": (
        "Download and validate source media during inspection.",
        "Скачать и проверить исходные медиа во время анализа.",
    ),
    "run_id": (
        "Run ID returned by import or another staged command.",
        "Run ID, который вернула команда import или другая команда черновика.",
    ),
    "target": (
        "Existing local dataset path or staged run ID.",
        "Существующий локальный путь к данным или Run ID черновика.",
    ),
    "yes": (
        "Confirm the exact planned operation without an interactive question.",
        "Подтвердить точно рассчитанную операцию без интерактивного вопроса.",
    ),
    "non_interactive": (
        "Disable interactive questions; missing required input becomes an error.",
        "Отключить вопросы; отсутствие обязательных данных станет ошибкой.",
    ),
    "json_output": (
        "Emit the stable machine-readable JSON envelope.",
        "Вывести стабильный машиночитаемый JSON-конверт.",
    ),
    "quiet": (
        "Suppress successful human-readable output.",
        "Не выводить человекочитаемый результат успешной команды.",
    ),
    "debug": (
        "Show sanitized debugging details on failures.",
        "Показать очищенные от секретов отладочные сведения при ошибке.",
    ),
}


_TEXT: dict[str, tuple[str, str]] = {
    "Warning": ("Warning", "Предупреждение"),
    "Hint": ("Hint", "Подсказка"),
    "Run ID": ("Run ID", "ID запуска"),
    "succeeded": ("succeeded", "выполнено"),
    "noop": ("no changes", "без изменений"),
    "partial": ("partially completed", "выполнено частично"),
    "failed": ("failed", "ошибка"),
    "interrupted": ("interrupted", "прервано"),
    "Telegram Bot API token": ("Telegram Bot API token", "Токен Telegram Bot API"),
    "Gemini API key": ("Gemini API key", "API-ключ Gemini"),
    "RuntimeError: the MojiLex rlottie RGBA renderer is required for TGS": (
        "The MojiLex rlottie RGBA renderer is required for TGS.",
        "Для обработки TGS требуется RGBA-рендерер MojiLex на базе rlottie.",
    ),
    (
        "Run mojilex doctor and install the backend it reports. TGS setup: "
        "https://github.com/MojiLex/mojilex-cli/blob/main/docs/media-prerequisites.md"
    ): (
        "Run mojilex doctor and install the backend it reports. TGS setup: "
        "https://github.com/MojiLex/mojilex-cli/blob/main/docs/media-prerequisites.md",
        "Запустите mojilex doctor и установите указанный им компонент. Настройка TGS: "
        "https://github.com/MojiLex/mojilex-cli/blob/main/docs/media-prerequisites.md",
    ),
    "Configured repository target looks like a local path, but it does not exist.": (
        "Configured repository target looks like a local path, but it does not exist.",
        "Настроенный репозиторий похож на локальный путь, но такого пути не существует.",
    ),
    (
        "Pass --repo OWNER/REPO or an existing absolute local path. "
        "Relative paths are resolved from the current working directory."
    ): (
        "Pass --repo OWNER/REPO or an existing absolute local path. "
        "Relative paths are resolved from the current working directory.",
        "Передайте --repo OWNER/REPO или существующий абсолютный локальный путь. "
        "Относительные пути считаются от текущей рабочей папки.",
    ),
}


def normalize_ui_language(value: str) -> UiLanguage:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_UI_LANGUAGES:
        raise ValueError("--ui-language must be en or ru")
    return normalized


def extract_ui_language(
    argv: Sequence[str], *, environment: Mapping[str, str] | None = None
) -> tuple[UiLanguage, list[str]]:
    """Extract a global UI language flag from any pre-command position."""

    selected: str | None = None
    cleaned: list[str] = []
    index = 0
    passthrough = False
    while index < len(argv):
        argument = argv[index]
        if argument == "--":
            passthrough = True
            cleaned.append(argument)
            index += 1
            continue
        if not passthrough and argument == "--ui-language":
            if index + 1 >= len(argv):
                raise ValueError("--ui-language requires en or ru")
            selected = argv[index + 1]
            index += 2
            continue
        if not passthrough and argument.startswith("--ui-language="):
            selected = argument.partition("=")[2]
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    source = selected
    if source is None:
        source = (environment or os.environ).get(UI_LANGUAGE_ENV, "en")
    return normalize_ui_language(source), cleaned


@contextmanager
def use_ui_language(language: UiLanguage) -> Iterator[None]:
    token = _CURRENT_UI_LANGUAGE.set(language)
    try:
        yield
    finally:
        _CURRENT_UI_LANGUAGE.reset(token)


def current_ui_language() -> UiLanguage:
    return _CURRENT_UI_LANGUAGE.get()


def text(value: str, *, language: UiLanguage | None = None) -> str:
    selected = language or current_ui_language()
    translated = _TEXT.get(value)
    return value if translated is None else translated[1 if selected == "ru" else 0]


def localize_command_tree(command: Any, language: UiLanguage, path: tuple[str, ...] = ()) -> None:
    """Apply localized help to a generated Click tree without changing command names."""

    if not path:
        _localize_rich_framework(language)
        _group_root_commands(command, language)
    key = " ".join(path) if path else "mojilex"
    if key in _COMMAND_HELP:
        command.help = _COMMAND_HELP[key][1 if language == "ru" else 0]
        command.short_help = command.help
    for parameter in command.params:
        help_text = _PARAMETER_HELP.get(getattr(parameter, "name", ""))
        if help_text is not None:
            parameter.help = help_text[1 if language == "ru" else 0]
    for name, child in getattr(command, "commands", {}).items():
        localize_command_tree(child, language, (*path, name))


def _group_root_commands(command: Any, language: UiLanguage) -> None:
    commands = getattr(command, "commands", None)
    if not isinstance(commands, dict):
        return
    ordered: dict[str, Any] = {}
    for section, names in _ROOT_COMMAND_SECTIONS:
        title = _PANEL_TITLES[section][1 if language == "ru" else 0]
        for name in names:
            child = commands.get(name)
            if child is None:
                continue
            child.rich_help_panel = title
            ordered[name] = child
    for name, child in commands.items():
        if name not in ordered:
            ordered[name] = child
    command.commands = ordered


def _localize_rich_framework(language: UiLanguage) -> None:
    """Translate stable Typer Rich headings for the pinned CLI dependency."""

    import typer.rich_utils as rich_utils

    if language == "ru":
        rich_utils.ARGUMENTS_PANEL_TITLE = "Аргументы"
        rich_utils.OPTIONS_PANEL_TITLE = "Параметры"
        rich_utils.COMMANDS_PANEL_TITLE = "Команды"
        rich_utils.ERRORS_PANEL_TITLE = "Ошибка"
        rich_utils.ABORTED_TEXT = "Прервано."
        rich_utils.RICH_HELP = "Справка: [blue]'{command_path} {help_option}'[/]."
        rich_utils.DEFAULT_STRING = "[по умолчанию: {}]"
        rich_utils.REQUIRED_LONG_STRING = "[обязательно]"
        return
    rich_utils.ARGUMENTS_PANEL_TITLE = "Arguments"
    rich_utils.OPTIONS_PANEL_TITLE = "Options"
    rich_utils.COMMANDS_PANEL_TITLE = "Commands"
    rich_utils.ERRORS_PANEL_TITLE = "Error"
    rich_utils.ABORTED_TEXT = "Aborted."
    rich_utils.RICH_HELP = "Try [blue]'{command_path} {help_option}'[/] for help."
    rich_utils.DEFAULT_STRING = "[default: {}]"
    rich_utils.REQUIRED_LONG_STRING = "[required]"
