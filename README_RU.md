# MojiLex CLI — описание и инструкция

[English README](README.md)

`mojilex` — консольная утилита для создания и поддержки открытого каталога
кастомных эмодзи. Она умеет:

- импортировать публичные наборы кастомных эмодзи Telegram;
- локально извлекать технические признаки WebP, TGS и WebM;
- получать через Gemini описания, теги и семантические признаки на русском и
  английском языках;
- находить точные и визуально похожие дубликаты;
- проверять структуру и целостность репозитория данных;
- подготовить локальное изменение или pull request в репозиторий данных;
- читать заранее собранный snapshot полностью офлайн.

Исходные изображения, анимации, кадры, contact sheet, токены и временные ссылки
Telegram не сохраняются в Git. В репозиторий данных попадают только метаданные,
проверяемые хеши и результаты анализа.

## Что где находится

Используются два соседних репозитория:

```text
projects\
├── mojilex-cli\   код утилиты, единственный проект PyCharm
└── mojilex\       данные и JSON Schema
```

- Код: <https://github.com/MojiLex/mojilex-cli>
- Данные: <https://github.com/MojiLex/mojilex>

В PyCharm достаточно открыть папку `mojilex-cli`. Репозиторий `mojilex`
используется утилитой как хранилище данных и не требует отдельного проекта.

## Требования

- Windows, Linux или macOS;
- Python 3.11 или новее;
- Git;
- токен Telegram-бота для импорта;
- API-ключ Gemini для генерации описаний;
- FFmpeg/ffprobe для WebM;
- lossless rlottie RGBA adapter MojiLex для TGS.

Обработка WebP уже входит в Python-зависимости. Доступность всех компонентов
проверяется командой `mojilex doctor`. Подробнее о медиакомпонентах:
[docs/media-prerequisites.md](docs/media-prerequisites.md).

## Установка из исходников в Windows

Откройте PowerShell в корне клонированного репозитория `mojilex-cli`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\mojilex.exe --version
.\.venv\Scripts\mojilex.exe doctor
```

После активации окружения команды можно вводить короче:

```powershell
.\.venv\Scripts\Activate.ps1
mojilex --version
```

Если PowerShell запрещает запуск `Activate.ps1`, активация не обязательна:
используйте полный путь `.\.venv\Scripts\mojilex.exe`.

После официальной публикации пакета его также можно будет установить как
изолированную утилиту:

```powershell
pipx install mojilex-cli
```

## Первоначальная настройка

Находясь в папке `mojilex-cli`, создайте локальный конфигурационный файл:

```powershell
mojilex init `
  --repo ..\mojilex `
  --provider gemini `
  --model gemini-3.8-flash `
  --publish local `
  --lang ru `
  --lang en
```

Без параметров `init` запускает интерактивный мастер. Конфигурация не должна
содержать токены или API-ключи. Точный ID модели указывается явно, чтобы он был
виден в происхождении сгенерированных данных.

Для обычного интерактивного запуска секреты можно не записывать в файлы:
утилита запросит недостающие значения скрытым вводом. Для автоматического или
неинтерактивного запуска используйте переменные окружения:

```text
TELEGRAM_BOT_TOKEN
GEMINI_API_KEY
GH_TOKEN или GITHUB_TOKEN — только для публикации через GitHub
```

Не передавайте секреты параметрами командной строки, не добавляйте их в
`.mojilex.toml` и не коммитьте `.env`.

## Быстрый сценарий работы

### 1. Безопасно посмотреть план импорта

`--dry-run` не вызывает AI и не изменяет постоянные данные:

```powershell
mojilex add https://t.me/addemoji/PackName --repo ..\mojilex --dry-run
```

Чтобы дополнительно скачать и проверить медиа, но всё равно ничего не
публиковать:

```powershell
mojilex add https://t.me/addemoji/PackName --repo ..\mojilex --dry-run --check-media
```

### 2. Импортировать набор локально

```powershell
mojilex add https://t.me/addemoji/PackName `
  --repo ..\mojilex `
  --publish local `
  --max-cost-usd 0.25
```

Перед платным запросом утилита показывает провайдера, модель, количество
запросов и верхнюю оценку стоимости. Не используйте `--yes`, пока не проверили
этот план. Если цена модели неизвестна, команда завершится безопасной ошибкой,
если пользователь явно не разрешил неизвестную стоимость.

### 3. Проверить данные

```powershell
mojilex validate ..\mojilex
```

Для машинной обработки добавьте `--json`. В этом режиме stdout содержит один
JSON-объект, а диагностические сообщения выводятся в stderr.

### 4. Найти дубликаты

Перестроить локальный индекс для всего набора данных:

```powershell
mojilex dedupe scan --all --repo ..\mojilex
```

Проверить отдельный эмодзи или коллекцию:

```powershell
mojilex dedupe scan EMOJI_OR_COLLECTION_ID --repo ..\mojilex
```

Найденные похожие элементы являются кандидатами на ручную проверку и не
объединяются автоматически.

### 5. Подготовить pull request

```powershell
mojilex add https://t.me/addemoji/PackName `
  --repo MojiLex/mojilex `
  --publish pr
```

Для этого нужен `GH_TOKEN` или `GITHUB_TOKEN` с подходящими правами. Утилита не
делает force push. Прямой push отделён от обычной публикации и требует
дополнительного `--direct-push`, пройденных проверок и подтверждения точного
commit SHA.

## Продолжение прерванной операции

При ошибке или остановке утилита выводит `RUN_ID`. Продолжить тот же запуск:

```powershell
mojilex resume RUN_ID
```

Посмотреть подготовленный результат:

```powershell
mojilex describe RUN_ID
```

Повторный запуск использует только полностью совпадающие записи безопасного
кеша. Изменение модели, prompt, схемы или параметров создаёт другой ключ.

## Офлайн-чтение готового snapshot

Команды чтения работают не с исходной папкой `mojilex`, а с уже собранным
каталогом snapshot:

```powershell
mojilex search "радость" --snapshot C:\path\to\snapshot --allow-unverified
mojilex get EMOJI_ID --snapshot C:\path\to\snapshot --allow-unverified
mojilex get-collection COLLECTION_ID --snapshot C:\path\to\snapshot --allow-unverified
mojilex similar EMOJI_ID --snapshot C:\path\to\snapshot --allow-unverified
```

Текущий MVP snapshot проверяется по хешам, но ещё не имеет подписанной цепочки
доверия Stage C. Поэтому диагностическое чтение требует явного
`--allow-unverified`; этот флаг не делает snapshot доверенным. Во время чтения
сеть, Telegram, Gemini и медиадекодеры не используются.

## Полезные команды

```text
mojilex --help                         список команд
mojilex COMMAND --help                 параметры конкретной команды
mojilex doctor                         проверка окружения и декодеров
mojilex config show                    несекретная конфигурация
mojilex cache info                     состояние AI-кеша
mojilex cache prune                    безопасная очистка кеша
mojilex validate PATH                  проверка исходного набора данных
mojilex dedupe scan --all              индекс дубликатов
mojilex snapshot verify PATH           проверка готового snapshot
```

## Частые проблемы

### Команда `mojilex` не найдена

Активируйте `.venv` либо запускайте
`.\.venv\Scripts\mojilex.exe` напрямую.

### Git сообщает `detected dubious ownership`

Не отключайте проверку для всех репозиториев. Добавьте только точные пути:

Если PowerShell открыт в корне `mojilex-cli`, добавьте только два точных
разрешённых пути:

```powershell
git config --global --add safe.directory (Resolve-Path .).Path
git config --global --add safe.directory (Resolve-Path ..\mojilex).Path
```

### Не обрабатывается TGS или WebM

Запустите:

```powershell
mojilex doctor
```

Команда отдельно покажет состояние WebP, rlottie/TGS и FFmpeg/WebM.

### Команда остановилась на проверке AI

Проверьте точный ID модели, наличие `GEMINI_API_KEY`, лимит
`--max-ai-requests` и ограничение `--max-cost-usd`. Ответ модели всё равно
проходит локальную строгую JSON Schema и дополнительные проверки; некорректный
ответ не публикуется частично.

## Разработка и тесты

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m ruff format --check src tests
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m build
```

Архитектура описана в [docs/architecture.md](docs/architecture.md), модель
безопасности — в [docs/security-model.md](docs/security-model.md), публикация —
в [docs/publishing.md](docs/publishing.md).

## Текущие ограничения MVP

- Квалификация модели требует отдельного размеченного benchmark-набора и не
  возникает автоматически после успешного API-вызова.
- Похожие эмодзи требуют человеческого решения.
- Подписанный каталог, аттестации и отзыв доверия относятся к Stage C.
- Разделённые snapshot, delta-обновления и масштабирование относятся к Stage D/E.

Код CLI распространяется по лицензии MIT. Метаданные, передаваемые в репозиторий
данных, публикуются по правилам этого репозитория.
