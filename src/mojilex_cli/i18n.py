"""Human CLI localization without changing machine-readable contracts."""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Literal

UiLanguage = Literal["en", "ru"]
SUPPORTED_UI_LANGUAGES: tuple[UiLanguage, ...] = ("en", "ru")
UI_LANGUAGE_ENV = "MOJILEX_UI_LANGUAGE"

_CURRENT_UI_LANGUAGE: ContextVar[UiLanguage] = ContextVar("mojilex_ui_language", default="en")

_ROOT_COMMAND_SECTIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "pack_workflow",
        (
            "list",
            "show",
            "gallery",
            "import",
            "describe",
            "publish",
            "sync",
            "resume",
            "add",
            "submit",
            "update",
        ),
    ),
    (
        "read",
        ("search", "get", "get-collection", "resolve", "similar", "snapshots"),
    ),
    ("quality", ("validate", "dedupe", "review", "set-status", "takedown")),
    ("releases", ("snapshot", "build-index", "benchmark-dedupe", "benchmark-model")),
    ("setup", ("settings", "init", "doctor", "config", "cache", "uninstall")),
)

_PANEL_TITLES: dict[str, tuple[str, str]] = {
    "pack_workflow": ("Pack analysis and publication", "Анализ и публикация паков"),
    "read": ("Search and read", "Поиск и чтение"),
    "quality": ("Validation and moderation", "Проверка и модерация"),
    "releases": ("Snapshots and benchmarks", "Снимки и тесты"),
    "setup": ("Setup and maintenance", "Настройка и обслуживание"),
}


_COMMAND_HELP: dict[str, tuple[str, str]] = {
    "sync": (
        "Send all completed new packs in one GitHub pull request.",
        "Отправить все новые готовые паки одним Pull Request на GitHub.",
    ),
    "gallery": (
        "Open a local gallery of saved descriptions and previews.",
        "Открыть локальную галерею эмодзи с описаниями.",
    ),
    "settings": (
        "View and edit analysis settings and limits.",
        "Посмотреть и изменить настройки анализа и лимиты.",
    ),
    "list": ("List saved packs and progress.", "Показать сохранённые паки и прогресс анализа."),
    "show": ("Read saved pack descriptions.", "Показать готовые описания пака без AI-запросов."),
    "publish": (
        "Publish a pack through a GitHub pull request.",
        "Отправить готовый пак на GitHub через Pull Request.",
    ),
    "mojilex": (
        "Build, validate, and publish the media-free MojiLex emoji dataset.",
        "Превращайте эмодзи в текстовые описания. Запустите mojilex без команды, "
        "чтобы открыть меню. Все команды: mojilex --help-all.",
    ),
    "snapshots": (
        "List local release snapshots and their paths.",
        "Показать локальные снимки релизов и пути к ним.",
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
        "Resume a saved pack by name or run ID.",
        "Продолжить сохранённый пак по имени или Run ID.",
    ),
    "review": (
        "Browse a pack or optionally record an emoji review.",
        "Просмотреть пак; при желании записать проверку отдельного эмодзи.",
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
        "Inspect configuration and manage stored API credentials.",
        "Просмотреть конфигурацию и управлять сохранёнными API-ключами.",
    ),
    "config show": (
        "Show the resolved non-secret configuration.",
        "Показать итоговую несекретную конфигурацию.",
    ),
    "config set-ui-language": (
        "Persist the human interface language without changing other settings.",
        "Сохранить язык интерфейса, не меняя остальные настройки.",
    ),
    "config set-credentials": (
        "Save API credentials in the operating-system keyring using hidden input.",
        "Сохранить API-ключи в системном хранилище через скрытый ввод.",
    ),
    "config clear-credentials": (
        "Delete every API credential saved by MojiLex.",
        "Удалить все API-ключи, сохранённые MojiLex.",
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
    "uninstall": (
        "Completely remove MojiLex, its stored credentials, and owned local data.",
        "Полностью удалить MojiLex, сохранённые ключи и собственные локальные данные.",
    ),
}


_PARAMETER_HELP: dict[str, tuple[str, str]] = {
    "help_all": ("Show all advanced commands.", "Показать все дополнительные команды."),
    "all_fields": ("Print every saved field.", "Вывести все сохранённые поля сразу."),
    "browser": ("Open a local browser gallery.", "Открыть локальную галерею в браузере."),
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
    "install": (
        "Install missing Windows media dependencies, then rerun all checks.",
        "Установить недостающие медиакомпоненты Windows и повторить все проверки.",
    ),
    "keep_data": (
        "Keep configuration, run data, cache, and credentials.",
        "Сохранить конфигурацию, данные запусков, кеш и ключи.",
    ),
    "telegram": (
        "Save a Telegram Bot API token.",
        "Сохранить токен Telegram Bot API.",
    ),
    "gemini": ("Save a Gemini API key.", "Сохранить API-ключ Gemini."),
    "openai": ("Also save an OpenAI API key.", "Также сохранить API-ключ OpenAI."),
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
    "OpenAI API key": ("OpenAI API key", "API-ключ OpenAI"),
    "Delete every Telegram, Gemini, and OpenAI credential saved by MojiLex?": (
        "Delete every Telegram, Gemini, and OpenAI credential saved by MojiLex?",
        "Удалить все ключи Telegram, Gemini и OpenAI, сохранённые MojiLex?",
    ),
    "Completely uninstall MojiLex with this exact plan: ": (
        "Completely uninstall MojiLex with this exact plan: ",
        "Полностью удалить MojiLex по этому точному плану: ",
    ),
    (
        "Install the missing Windows media components now? "
        "This may install FFmpeg or Visual Studio Build Tools."
    ): (
        "Install the missing Windows media components now? "
        "This may install FFmpeg or Visual Studio Build Tools.",
        "Установить недостающие медиакомпоненты Windows сейчас? "
        "Могут быть установлены FFmpeg или Visual Studio Build Tools.",
    ),
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

_RUSSIAN_MESSAGES = {
    "Whole-run request limit or unlimited.": "Лимит запросов на всю операцию или unlimited.",
    "Use a non-negative integer or unlimited.": "Введите целое число от 0 или unlimited.",
    "ask, skip, or allow official packs.": (
        "Паки из официальной базы: ask — спросить, skip — пропустить, allow — не проверять."
    ),
    "All selected packs are already in the official repository; skipped.": (
        "Все выбранные паки уже есть в официальном репозитории и пропущены."
    ),
    "Unknown official pack policy.": "Неизвестный режим проверки официальных паков.",
    "Choose ask, skip, or allow with --official-packs or settings.": (
        "Выберите ask, skip или allow через --official-packs или настройки."
    ),
    "The default repository folder is occupied by unrelated or incomplete data.": (
        "В стандартной папке репозитория находятся посторонние или неполные данные."
    ),
    "Choose an existing dataset with --repo. Files in this folder were preserved.": (
        "Укажите существующую базу через --repo. Файлы в этой папке сохранены."
    ),
    "The default repository must not be a symlink.": (
        "Стандартная папка репозитория не должна быть символической ссылкой."
    ),
    "Choose an existing dataset with --repo.": "Укажите существующую базу через --repo.",
    "Could not initialize the default dataset repository.": (
        "Не удалось подготовить стандартный репозиторий базы."
    ),
    "Check GitHub access and the configured base branch, then retry.": (
        "Проверьте доступ к GitHub и указанную основную ветку, затем повторите попытку."
    ),
    "The repository folder became occupied.": "Папка репозитория уже занята.",
    "Check other running MojiLex processes. Existing files were preserved.": (
        "Проверьте другие запущенные процессы MojiLex. Существующие файлы сохранены."
    ),
    "Could not prepare the application's dataset folder.": (
        "Не удалось подготовить папку базы в данных программы."
    ),
    "Check folder permissions, GitHub access, and other running MojiLex processes.": (
        "Проверьте права доступа к папке, доступ к GitHub и другие запущенные процессы MojiLex."
    ),
    "Use a pack name or an exact RunID.": "Укажите имя пака или Run ID.",
    "Run mojilex list to see saved packs.": "Посмотрите сохранённые паки командой mojilex list.",
    "This pack name matches different source groups or repositories.": (
        "Имя пака неоднозначно: найдены разные группы паков или репозитории."
    ),
    "Choose an exact RunID from mojilex list.": "Выберите Run ID из mojilex list.",
    "No completed description run is ready to publish for this pack.": (
        "Для этого пака пока нет завершённого черновика для публикации."
    ),
    "Complete mojilex describe for this pack first, or choose an exact RunID.": (
        "Сначала выполните mojilex describe ИМЯ_ПАКА или выберите нужный Run ID."
    ),
    "Some saved run files could not be read.": (
        "Некоторые сохранённые запуски не удалось прочитать."
    ),
    "This saved run contains multiple packs.": "В этом запуске несколько паков.",
    "The saved staging dataset could not be read; using exact cached results.": (
        "Черновик сейчас недоступен; показаны сохранённые ответы из кэша."
    ),
    "The cache cannot be read without changing it; retry after the running command ends.": (
        "Кэш занят записью. Повторите просмотр после завершения текущей команды."
    ),
    "Some exact cached descriptions are unavailable; saved progress was not changed.": (
        "Часть сохранённых описаний недоступна. Прогресс запуска не изменён."
    ),
    "Unfinished run": "Незавершённый запуск",
    "Continue unfinished work: mojilex resume NAME": "Продолжить незавершённое: mojilex resume ИМЯ",
    "Validation passed.": "Проверка пройдена.",
    "Changed data files": "Изменено файлов данных",
    "Pack": "Пак",
    "Descriptions": "Описания",
    "descriptions": "описаний",
    "Updated": "Обновлено",
    "Content rating": "Категория контента",
    "Content warnings": "Предупреждения о содержимом",
    "Motion": "Движение",
    "Usage examples": "Примеры использования",
    "Tags": "Теги",
    "Text in emoji": "Текст на эмодзи",
    "Pending": "Ещё не готовы",
    "Unavailable": "Недоступны",
    "Invalid": "Повреждены",
    "content_types": "Типы содержимого",
    "styles": "Стили",
    "suggested_uses": "Назначение",
    "uncertainties": "Неопределённости",
    "No saved packs. Start with mojilex import PACK_URL.": (
        "Сохранённых паков пока нет. Начните с mojilex import ССЫЛКА_НА_ПАК."
    ),
    "Open a pack: mojilex show NAME": "Открыть пак: mojilex show ИМЯ",
    "Optional viewing. Content warnings do not require approval.": (
        "Добровольный просмотр. Предупреждения о содержимом не требуют одобрения."
    ),
    "This run is already complete.": "Этот запуск уже завершён.",
    "Pack name or run ID.": "Имя пака или Run ID.",
    "Validate without uploading.": "Проверить без загрузки на GitHub.",
    "Optional: approve, request-changes, or reject": (
        "Необязательно: approve, request-changes или reject"
    ),
    "The pack analysis is not complete.": "Анализ пака ещё не завершён.",
    "No saved run matches this pack name.": "Нет сохранённого запуска с таким именем пака.",
    "Run mojilex list to see saved packs, or pass an exact RunID.": (
        "Посмотрите сохранённые паки командой mojilex list."
    ),
    "Could not inspect the configured dataset repository.": (
        "Не удалось проверить настроенный репозиторий данных."
    ),
    "Check Git and access to the configured local repository.": (
        "Проверьте Git и доступ к настроенному локальному репозиторию."
    ),
    "Could not create an isolated dataset checkout.": (
        "Не удалось создать изолированную рабочую копию репозитория данных."
    ),
    "Check Git, repository access, and the configured base branch.": (
        "Проверьте Git, доступ к репозиторию и настроенную базовую ветку."
    ),
    "Error": "Ошибка",
    "Operation interrupted by the user.": "Операция прервана пользователем.",
    "Use mojilex resume with the reported run ID when a checkpoint exists.": (
        "Продолжите операцию командой mojilex resume с указанным ID запуска, "
        "если есть контрольная точка."
    ),
    "provider cost is unknown; explicit approval is required": (
        "Стоимость запросов к провайдеру неизвестна; требуется явное подтверждение."
    ),
    "AI requests require explicit approval": "Для запросов к ИИ требуется подтверждение.",
    "AI requests were not authorized.": "Разрешение на запросы к ИИ не получено.",
    (
        "Review the planned AI requests and confirm when restarting, "
        "or use --yes in an interactive workflow."
    ): (
        "Проверьте запланированные запросы к ИИ и подтвердите их при повторном запуске "
        "либо используйте --yes в интерактивном режиме."
    ),
    "The provider's USD cost is unknown and was not authorized.": (
        "Стоимость запросов к провайдеру в USD неизвестна; разрешение не получено."
    ),
    (
        "Rerun with --allow-unknown-cost after reviewing the planned AI requests, "
        "or use --yes in an interactive workflow."
    ): (
        "Проверьте запланированные AI-запросы и повторите с --allow-unknown-cost "
        "либо используйте --yes в интерактивном режиме."
    ),
    "Correct the reported condition and retry.": "Устраните указанную причину и повторите команду.",
    "Check the command options and non-secret configuration.": (
        "Проверьте параметры команды и несекретные настройки."
    ),
    "Run mojilex validate --strict and fix every reported issue.": (
        "Запустите mojilex validate --strict и исправьте все найденные ошибки."
    ),
    "Rerun with --debug and report the sanitized traceback.": (
        "Повторите команду с --debug и сообщите очищенную от секретов трассировку ошибки."
    ),
    "This sensitive operation requires explicit confirmation.": (
        "Для этой операции требуется явное подтверждение."
    ),
    "Rerun with --yes after reviewing the exact target.": (
        "Проверьте точную цель операции и повторите команду с --yes."
    ),
    "Operation was not confirmed.": "Операция не подтверждена.",
    "Review the target and rerun when ready.": (
        "Проверьте цель операции и повторите команду, когда будете готовы."
    ),
    "Pass an explicit local --snapshot path to a read command.": (
        "Передайте команде чтения локальный путь к снимку через --snapshot."
    ),
    "Download an immutable snapshot separately and verify its local path.": (
        "Скачайте неизменяемый снимок отдельно и проверьте его локальный путь."
    ),
    "Continue using the explicitly pinned local snapshot.": (
        "Продолжайте использовать явно выбранный локальный снимок."
    ),
    "estimated AI cost limit would be exceeded": "Будет превышен лимит расчётной стоимости AI.",
    "AI request limit would be exceeded": "Будет превышен лимит AI-запросов.",
    "The requested budget is below the usage already saved for this run.": (
        "Новый лимит меньше расхода, уже сохранённого для этого запуска."
    ),
    "Choose limits at least as high as recorded usage. "
    "unlimited disables only the request count.": (
        "Укажите лимиты не ниже уже учтённого расхода. "
        "unlimited отключает только ограничение числа запросов."
    ),
    "The model response failed validation. Saved results are retained; "
    "resume with the same run ID. If it repeats, report the validation code and field path.": (
        "Ответ модели не прошёл проверку. Сохранённые результаты остаются; "
        "продолжите через resume с тем же ID запуска. При повторении сообщите код и поле ошибки."
    ),
    "Gemini structured response: interaction_incomplete": "Gemini не завершил ответ.",
    "Gemini structured response: model_mismatch": "Gemini вернул ответ другой модели.",
    "Gemini structured response: output_missing": "Gemini вернул ответ без текста JSON.",
    "Gemini structured response: invalid_json": "Gemini вернул некорректный JSON.",
    "The existing AI cache could not be inspected; plan assumes misses.": (
        "Не удалось проверить существующий AI-кеш; план рассчитан без его использования."
    ),
    "GEMINI_API_KEY is required for uncached descriptions.": (
        "Для описаний, которых нет в кеше, нужен GEMINI_API_KEY."
    ),
    "Set it in the process environment or use an already populated cache.": (
        "Задайте ключ в окружении процесса или используйте уже заполненный кеш."
    ),
    "This command requires a local dataset checkout.": (
        "Для этой команды нужна локальная копия репозитория данных."
    ),
    "Clone MojiLex/mojilex and pass its path with --repo.": (
        "Склонируйте MojiLex/mojilex и передайте путь через --repo."
    ),
    "Pass an existing MojiLex dataset checkout.": (
        "Укажите существующую локальную копию репозитория данных MojiLex."
    ),
    "Decision [same-artwork/variant-of/related-series/not-duplicate/skip]": (
        "Решение [same-artwork/variant-of/related-series/not-duplicate/skip]"
    ),
    "True": "Да",
    "False": "Нет",
    "None": "нет данных",
    (
        "Diagnostic read from an integrity-checked unsigned snapshot; "
        "safe_eligible is always false."
    ): (
        "Диагностическое чтение снимка: целостность проверена, подписи нет; "
        "safe_eligible всегда false."
    ),
    "The local snapshot has integrity checks but no enforceable release signature.": (
        "Целостность локального снимка проверена, но подтверждённой подписи релиза нет."
    ),
    ("Rerun with --allow-unverified only for diagnostic use of this exact local snapshot."): (
        "Повторите с --allow-unverified только для диагностического чтения "
        "этого конкретного локального снимка."
    ),
    "Config file to create.": "Создаваемый файл настроек.",
    "Replace an existing config.": "Заменить существующий файл настроек.",
    "Rebuild the complete index.": "Перестроить полный индекс.",
    "Repeat for each language.": "Повторите параметр для каждого языка.",
    "Human interface language: en or ru.": "Язык интерфейса: en или ru.",
    "Versioned model benchmark manifest with human adjudication.": (
        "Версионированный манифест теста модели с результатами ручной оценки."
    ),
    "Versioned dedupe benchmark manifest.": "Версионированный манифест теста дубликатов.",
    "Exact manifest model ID.": "Точный ID модели из манифеста.",
    "Explicit provider model ID.": "Явно указанный ID модели провайдера.",
    "Local dataset root.": "Корневая папка локального набора данных.",
    "Exact manifest provider ID.": "Точный ID провайдера из манифеста.",
    "Emoji or collection selector.": "Селектор эмодзи или коллекции.",
    "Source URL or collection ID.": "URL источника или ID коллекции.",
    "Run ID or entity selectors.": "ID запуска или селекторы записей.",
    "Immutable snapshot ID: data-YYYY.MM.DD.N.": "ID неизменяемого снимка: data-YYYY.MM.DD.N.",
    "Immutable release source time as a Unix epoch.": (
        "Фиксированное время исходных данных релиза в формате Unix epoch."
    ),
    "Public source URLs.": "Публичные URL источников.",
    "Read one source per stdin line.": "Читать по одному источнику в строке stdin.",
}

_RUSSIAN_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"Authorize AI requests without a request-count limit for this run, including retries\? "
        r"The USD cost may be unknown\. This is a one-time approval for this invocation\.",
        "Разрешить AI-запросы без ограничения количества для этого запуска, включая повторы? "
        "Стоимость в USD может быть неизвестна. "
        "Разрешение действует только для текущего вызова команды.",
    ),
    (
        r"eligible emoji (?P<emoji_id>\S+) has incomplete concept mapping; "
        r"complete concept mapping before building a release snapshot",
        "У эмодзи {emoji_id} не завершена привязка понятий. "
        "Завершите её перед сборкой снимка релиза.",
    ),
    (
        r"Authorize up to (?P<count>\d+) additional AI requests for this run, including retries\? "
        r"The USD cost may be unknown\. This is a one-time approval for this invocation\.",
        "Разрешить до {count} дополнительных AI-запросов для этого запуска, включая повторы? "
        "Стоимость в USD может быть неизвестна. "
        "Разрешение действует только для текущего вызова команды.",
    ),
    (
        r"Snapshot discovery for channel (?P<channel>.+) is not configured "
        r"in the strictly offline MVP\.",
        "В текущей версии доступна только локальная работа со снимками; "
        "получение списка снимков канала {channel} не настроено.",
    ),
    (
        r"No signed release catalog or mirror is configured for (?P<snapshot>.+)\.",
        "Для снимка {snapshot} не настроены подписанный каталог релизов или зеркало.",
    ),
    (
        r"Snapshot update to (?P<snapshot>.+) is not configured "
        r"without a signed catalog and mirror\.",
        "Обновление до снимка {snapshot} недоступно без подписанного каталога и зеркала.",
    ),
    (r"Checking source (?P<index>\d+/\d+): (?P<source>.+)", "Проверка источника {index}: {source}"),
    (
        r"Packs are processed one at a time\. Within the current pack: up to "
        r"(?P<downloads>\d+) simultaneous downloads, (?P<decoders>\d+) media decoders, "
        r"and (?P<ai>\d+) simultaneous AI requests\. The AI value is concurrency, "
        r"not the total request limit\.",
        "Паки обрабатываются по очереди, по одному. Внутри текущего пака одновременно "
        "выполняются: до {downloads} скачиваний, до {decoders} обработок медиа и до {ai} "
        "запросов к ИИ. Число для ИИ — параллельность, а не общий лимит запросов.",
    ),
    (
        r"Packs are processed one at a time\. Within the current pack: up to "
        r"(?P<downloads>\d+) simultaneous downloads and (?P<decoders>\d+) media decoders\. "
        r"AI is not used during import; AI request limits apply later, during analysis\.",
        "Паки обрабатываются по очереди, по одному. Внутри текущего пака одновременно "
        "выполняются: до {downloads} скачиваний и до {decoders} обработок медиа. "
        "При импорте ИИ не используется; лимит запросов к ИИ применяется позже, "
        "на этапе анализа.",
    ),
    (
        r"Source (?P<source>.+): (?P<count>\d+) media item\(s\); "
        r"download concurrency=(?P<concurrency>\d+); "
        r"decoder concurrency=(?P<decoders>\d+)\.",
        "Источник {source}: медиафайлов — {count}; "
        "параллельных скачиваний — {concurrency}; декодеров — {decoders}.",
    ),
    (r"Media verified: (?P<item>.+)", "Медиафайл проверен: {item}"),
    (
        r"Semantic results cached: (?P<count>\d+) item\(s\)",
        "Результаты анализа сохранены в кеше: {count}",
    ),
    (
        r"AI plan: (?P<count>\d+) item\(s\), (?P<batches>\d+) candidate batch\(es\), "
        r"provider=(?P<provider>[^,]+), model=(?P<model>.+)\. Exact cache hits can reduce "
        r"requests; retries, escalation and puzzle checks share the run budget\. "
        r"Request limit: (?P<limit>\d+)\.",
        "План AI: эмодзи — {count}, возможных пачек — {batches}; провайдер — {provider}, "
        "модель — {model}. Совпадения в кеше могут уменьшить число запросов; "
        "повторы, переход на другую модель и проверки пазлов входят в общий бюджет запуска. "
        "Лимит запросов: {limit}.",
    ),
    (
        r"AI plan: (?P<count>\d+) item\(s\), (?P<batches>\d+) candidate batch\(es\), "
        r"provider=(?P<provider>[^,]+), model=(?P<model>.+)\. Exact cache hits can reduce "
        r"requests; retries, escalation and puzzle checks share the run budget\. "
        r"Request count is unlimited\.",
        "План AI: эмодзи — {count}, возможных пачек — {batches}; провайдер — {provider}, "
        "модель — {model}. Совпадения в кеше могут уменьшить число запросов; "
        "повторы, переход на другую модель и проверки пазлов входят в общий бюджет запуска. "
        "Количество запросов: без лимита.",
    ),
    (
        r"Derived contact-sheet PNG images will be sent to provider=(?P<provider>[^,]+), "
        r"model=(?P<model>.+)\. Provider processing terms: (?P<url>\S+) \. "
        r"MojiLex does not guarantee zero retention by the provider\.",
        "Подготовленные PNG-листы с кадрами будут отправлены провайдеру {provider}, "
        "модель — {model}. Условия обработки: {url} . "
        "MojiLex не гарантирует, что провайдер не сохраняет данные.",
    ),
    (r"Dataset directory does not exist: (?P<path>.+)", "Каталог данных не существует: {path}"),
    (r"Temporary comparison preview: (?P<path>.+)", "Временное изображение для сравнения: {path}"),
    (
        r"Gemini request timed out after (?P<seconds>[\d.]+) seconds\.",
        "Запрос Gemini не завершился за {seconds} с.",
    ),
    (
        r"Gemini model check timed out after (?P<seconds>[\d.]+) seconds\.",
        "Проверка модели Gemini не завершилась за {seconds} с.",
    ),
    (r"Missing argument(?P<detail>.*)", "Не указан обязательный аргумент{detail}"),
    (r"Missing option(?P<detail>.*)", "Не указан обязательный параметр{detail}"),
    (r"Missing parameter(?P<detail>.*)", "Не указан обязательный параметр{detail}"),
    (r"No such option: (?P<option>.+)", "Неизвестный параметр: {option}"),
    (r"No such command (?P<command>.+)\.", "Неизвестная команда {command}."),
    (r"Option (?P<option>.+) requires an argument\.", "Для параметра {option} нужно значение."),
    (
        r"Got unexpected extra arguments? (?P<arguments>.+)",
        "Неожиданные лишние аргументы {arguments}",
    ),
    (r"(?P<value>.+) is not a valid integer\.", "{value} — не целое число."),
    (r"(?P<value>.+) is not a valid int range\.", "{value} — не целое число."),
    (r"(?P<value>.+) is not a valid float(?: range)?\.", "{value} — не число."),
    (r"(?P<value>.+) is not a valid UUID\.", "{value} — некорректный UUID."),
    (
        r"(?P<value>.+) is not in the range (?P<range>.+)\.",
        "{value} вне допустимого диапазона {range}.",
    ),
    (
        r"(?P<value>.+) is not one of (?P<choices>.+)\.",
        "{value} не входит в допустимые значения: {choices}.",
    ),
    (
        r"(?P<value>.+) does not match the formats (?P<formats>.+)\.",
        "{value} не соответствует форматам {formats}.",
    ),
)


def normalize_ui_language(value: str) -> UiLanguage:
    normalized = value.strip().lower()
    if normalized not in SUPPORTED_UI_LANGUAGES:
        raise ValueError("--ui-language must be en or ru")
    return normalized


def extract_ui_language(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    user_path: Path | None = None,
    project_path: Path | None = None,
) -> tuple[UiLanguage, list[str]]:
    """Resolve CLI > environment > project > user > English before loading commands."""

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
    selected_environment = os.environ if environment is None else environment
    source = selected if selected is not None else selected_environment.get(UI_LANGUAGE_ENV)
    if source is None:
        source = _saved_ui_language(user_path=user_path, project_path=project_path)
    if source is None:
        source = "en"
    return normalize_ui_language(source), cleaned


def _saved_ui_language(*, user_path: Path | None, project_path: Path | None = None) -> str | None:
    if user_path is None:
        from mojilex_cli.config import default_user_config_path

        user_path = default_user_config_path()
    project_path = Path.cwd() / ".mojilex.toml" if project_path is None else project_path
    for path in (project_path, user_path):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as stream:
                value = tomllib.load(stream).get("ui_language")
        except (OSError, tomllib.TOMLDecodeError):
            # Keep help and diagnostics usable; load_config reports invalid config later.
            continue
        if isinstance(value, str):
            return value
    return None


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
    if translated is not None:
        return translated[1 if selected == "ru" else 0]
    if selected != "ru":
        return value
    if value in _RUSSIAN_MESSAGES:
        return _RUSSIAN_MESSAGES[value]
    if value.startswith("Gemini structured response: schema_validation: "):
        return "Ответ Gemini нарушает схему: " + value.removeprefix(
            "Gemini structured response: schema_validation: "
        )
    invalid = re.fullmatch(r"Invalid value for (.+?): (.+)", value, flags=re.DOTALL)
    if invalid:
        return f"Некорректное значение {invalid[1]}: {text(invalid[2], language=selected)}"
    if value.startswith("Invalid value: "):
        return "Некорректное значение: " + text(
            value.removeprefix("Invalid value: "), language=selected
        )
    suggestion = re.fullmatch(r"(.+) \(Possible options: (.+)\)", value, flags=re.DOTALL)
    if suggestion:
        return f"{text(suggestion[1], language=selected)} (Возможные параметры: {suggestion[2]})"
    coded = re.fullmatch(r"([A-Z][A-Z0-9_]+): (.+)", value, flags=re.DOTALL)
    if coded:
        return f"{coded[1]}: {text(coded[2], language=selected)}"
    for pattern, template in _RUSSIAN_PATTERNS:
        match = re.fullmatch(pattern, value, flags=re.DOTALL)
        if match:
            return template.format(**match.groupdict())
    return value


def confirm(message: str, *, default: bool = False, err: bool = False) -> bool:
    """Accept Russian and English answers while keeping an empty reply negative by default."""

    import typer

    if current_ui_language() != "ru":
        return bool(typer.confirm(message, default=default, err=err))
    suffix = " [Да/нет]: " if default else " [да/Нет]: "
    while True:
        answer = (
            str(
                typer.prompt(
                    text(message),
                    default="",
                    show_default=False,
                    prompt_suffix=suffix,
                    err=err,
                )
            )
            .strip()
            .casefold()
        )
        if not answer:
            return default
        if answer in {"да", "д", "yes", "y"}:
            return True
        if answer in {"нет", "н", "no", "n"}:
            return False
        typer.echo("Введите да/нет или y/n. Пустой ответ выбирает вариант по умолчанию.", err=err)


def localize_command_tree(command: Any, language: UiLanguage, path: tuple[str, ...] = ()) -> None:
    """Apply localized help to a generated Click tree without changing command names."""

    if not path:
        _localize_rich_framework(language)
        _group_root_commands(command, language)
    _localize_usage(command)
    original_help_option = command.get_help_option

    @wraps(original_help_option)
    def help_option(context: Any) -> Any:
        option = original_help_option(context)
        if option is not None:
            option.help = _PARAMETER_HELP["help"][1 if language == "ru" else 0]
        return option

    command.get_help_option = help_option
    key = " ".join(path) if path else "mojilex"
    if key in _COMMAND_HELP:
        command.help = _COMMAND_HELP[key][1 if language == "ru" else 0]
        command.short_help = command.help
    for parameter in command.params:
        help_text = _PARAMETER_HELP.get(getattr(parameter, "name", ""))
        if help_text is not None:
            parameter.help = help_text[1 if language == "ru" else 0]
        elif getattr(parameter, "help", None):
            parameter.help = text(str(parameter.help), language=language)
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
    basic = {
        "list",
        "show",
        "gallery",
        "import",
        "describe",
        "publish",
        "resume",
        "sync",
        "settings",
        "init",
        "doctor",
    }
    for name, child in ordered.items():
        child.hidden = name not in basic
    command.commands = ordered


def _localize_rich_framework(language: UiLanguage) -> None:
    """Translate stable Typer Rich headings for the pinned CLI dependency."""

    import typer.rich_utils as rich_utils

    _localize_rich_errors(rich_utils)

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


def _localize_usage(command: Any) -> None:
    original = command.get_usage
    if getattr(original, "_mojilex_localized", False):
        return

    @wraps(original)
    def usage(*args: Any, **kwargs: Any) -> str:
        rendered = str(original(*args, **kwargs))
        if current_ui_language() == "ru" and rendered.startswith("Usage: "):
            rendered = "Использование: " + rendered.removeprefix("Usage: ")
            return re.sub(
                r"\b(OPTIONS|COMMAND|ARGS)\b",
                lambda match: {"OPTIONS": "ПАРАМЕТРЫ", "COMMAND": "КОМАНДА", "ARGS": "АРГУМЕНТЫ"}[
                    match[0]
                ],
                rendered,
            )
        return rendered

    usage._mojilex_localized = True  # type: ignore[attr-defined]
    command.get_usage = usage


def _localize_rich_errors(rich_utils: Any) -> None:
    """Translate only the Rich display adapter, never stored exceptions or JSON messages."""

    original = rich_utils.rich_format_error
    if getattr(original, "_mojilex_localized", False):
        return

    @wraps(original)
    def render(error: Any) -> None:
        if current_ui_language() != "ru" or type(error).__name__ == "NoArgsIsHelpError":
            original(error)
            return

        class LocalizedError:
            ctx = getattr(error, "ctx", None)

            def format_message(self) -> str:
                return text(str(error.format_message()))

        original(LocalizedError())

    render._mojilex_localized = True  # type: ignore[attr-defined]
    rich_utils.rich_format_error = render
