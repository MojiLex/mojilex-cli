# MojiLex CLI

[English](README.md) | Русский

## Быстрый старт

Для обычного использования не нужно клонировать оба репозитория или создавать
проект в IDE. В Windows сначала один раз установите
[`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```console
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Закройте PowerShell или `cmd` после установки и откройте новое окно. Затем в
любом из этих терминалов выполните:

```console
uv --version
uv tool install --python 3.11 "git+https://github.com/MojiLex/mojilex-cli.git@main"
uv tool update-shell
```

Квадратные скобки и круглые скобки Markdown не являются частью Git URL —
копируйте аргумент внутри кавычек без оформления ссылки. После
`uv tool update-shell` снова откройте новое окно терминала и проверьте установку:

```console
mojilex --version
```

Включить русский интерфейс можно для одной команды или для новых окон PowerShell
и `cmd` постоянно:

```powershell
mojilex --ui-language ru --help
$env:MOJILEX_UI_LANGUAGE = "ru"
setx MOJILEX_UI_LANGUAGE ru
```

Переменная `$env:MOJILEX_UI_LANGUAGE` действует сразу в текущем PowerShell.
После `setx` закройте терминал и откройте новый: эта команда сохраняет язык
только для будущих процессов. Имена команд, флагов и поля JSON при переключении
языка не меняются.

Один раз войдите в GitHub и создайте несекретную конфигурацию:

```console
gh auth login
mojilex init --model gemini-3.8-flash --non-interactive
mojilex config set-credentials
```

`init` создаёт только несекретные настройки и никогда не запрашивает и не
сохраняет API-ключи. Флаг `--non-interactive` дополнительно отключает мастер
настроек. `config set-credentials` скрыто запросит Telegram token и Gemini API
key и сохранит их в системном защищённом хранилище. Переменные окружения имеют
приоритет над сохранёнными значениями. Если постоянное хранение не нужно,
пропустите эту команду: `mojilex add ...` запросит отсутствующие значения только
на время запуска.
Если у текущего сеанса нет доступного системного хранилища, команда завершится
безопасной ошибкой и предложит использовать переменные окружения.

Если в старой конфигурации указан несуществующий относительный путь к
репозиторию, его можно заменить безопасным режимом без автоматической
публикации:

```console
mojilex init --force --repo MojiLex/mojilex --publish local --model gemini-3.8-flash --non-interactive
```

### Поэтапный анализ и публикация

Сначала скачайте и проверьте медиа. Команда ничего не отправляет в GitHub и
возвращает сохранённый `mlxrun_...`:

```console
mojilex import "https://t.me/addemoji/PackName" --repo MojiLex/mojilex
```

Затем создайте AI-описания в том же черновике — по-прежнему без публикации:

```console
mojilex describe mlxrun_ВАШ_ID
```

Проверить готовые изменения локально, не загружая их:

```console
mojilex submit mlxrun_ВАШ_ID --publish local
```

Если результат устраивает, выберите один вариант публикации:

```console
# Создать отдельную ветку и pull request
mojilex submit mlxrun_ВАШ_ID --publish pr

# Отправить прямо в main после проверки и явного подтверждения
mojilex submit mlxrun_ВАШ_ID --direct-push
```

Замените `PackName` именем нужного набора, а `mlxrun_ВАШ_ID` — точным ID из
вывода `import`. Отсутствующие Telegram token и Gemini API key загружаются из
системного хранилища или запрашиваются скрыто только на один запуск.

Для быстрой проверки без AI-запросов и без сохранения черновика остаётся одна
команда:

```console
mojilex add "https://t.me/addemoji/PackName" --dry-run --check-media --repo MojiLex/mojilex
```

Для работы нужны Git, GitHub CLI (`gh`) и поддерживаемые медиакомпоненты. WebP
работает сразу после установки; готовность WebM и TGS проверяет `mojilex doctor`.
В Windows при отсутствии поддерживаемого компонента команда предложит установку
через `[y/N]`. Нажмите `y` либо сразу выполните `mojilex doctor --install` —
MojiLex установит только нужные компоненты и повторит проверку.

MojiLex CLI — консольная утилита для создания, проверки и публикации открытого
каталога метаданных кастомных эмодзи.

Основные возможности:

- импорт публичных наборов кастомных эмодзи Telegram;
- локальный анализ WebP, TGS и WebM;
- создание русских и английских описаний, тегов и семантических признаков с
  помощью поддерживаемого AI-провайдера;
- поиск точных и визуально похожих дубликатов;
- проверка структуры и целостности набора данных;
- подготовка локальных изменений и pull request;
- полностью офлайн-доступ к собранным snapshot.

Исходные изображения, анимации, декодированные кадры, contact sheet, токены и
временные ссылки Telegram не сохраняются в Git. В репозиторий данных включаются
только метаданные, проверяемые хеши и результаты анализа.

## Репозитории

- [mojilex-cli](https://github.com/MojiLex/mojilex-cli) — исходный код CLI;
- [mojilex](https://github.com/MojiLex/mojilex) — схемы и канонические данные.

Для разработки из исходников репозитории размещаются рядом:

```text
workspace\
├── mojilex-cli\
└── mojilex\
```

## Системные требования

- Python 3.11 или новее;
- Git;
- Windows, Linux или macOS;
- Telegram Bot API token для импорта;
- API key поддерживаемого AI-провайдера для генерации метаданных;
- FFmpeg и ffprobe для обработки WebM;
- lossless rlottie RGBA adapter MojiLex для обработки TGS.

Обработка WebP включена в основные Python-зависимости. Состояние окружения и
медиакомпонентов проверяет команда `mojilex doctor`. Подробные требования
приведены в разделе [Media prerequisites](docs/media-prerequisites.md).

## Разработка из исходников

Этот раздел нужен только разработчикам CLI. Для обычной установки используйте
одну команду из раздела «Быстрый старт».

Клонируйте оба репозитория:

```console
git clone https://github.com/MojiLex/mojilex-cli.git
git clone https://github.com/MojiLex/mojilex.git
cd mojilex-cli
```

Установка в Windows через PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\mojilex.exe --version
.\.venv\Scripts\mojilex.exe doctor
```

Если команда `python` недоступна, установите поддерживаемую версию с
[python.org](https://www.python.org/downloads/), включите добавление Python в
PATH и откройте новое окно терминала. Требуется Python 3.11 или новее.

Установка в Linux или macOS:

```console
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
.venv/bin/mojilex --version
.venv/bin/mojilex doctor
```

В дальнейших примерах предполагается, что виртуальное окружение активировано.
В Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

В Linux или macOS:

```console
source .venv/bin/activate
```

Активация не обязательна: команду `mojilex` можно запускать по полному пути из
каталога `.venv`.

## Конфигурация

Обычная конфигурация для публикации в официальный репозиторий:

```console
mojilex init --repo MojiLex/mojilex --provider gemini --model gemini-3.8-flash --publish pr --lang ru --lang en --non-interactive
```

Идентификатор модели задаётся явно и сохраняется в происхождении созданных
описаний. Без `--non-interactive` команда открывает мастер несекретных настроек.

Приоритет параметров:

1. аргументы командной строки;
2. переменные окружения;
3. проектный файл `.mojilex.toml`;
4. пользовательская конфигурация;
5. безопасные значения по умолчанию.

Конфигурационные файлы предназначены только для несекретных параметров.

## Учетные данные

Команда `mojilex config set-credentials` скрыто запрашивает Telegram token и
Gemini API key и сохраняет их в системном хранилище учётных данных. Для OpenAI
есть дополнительный флаг `--openai`. Посмотреть только факт наличия ключей без
их значений можно через `mojilex config show`, а удалить сохранённые записи —
через `mojilex config clear-credentials`.

Переменные окружения имеют приоритет. Если ключа нет ни в окружении, ни в
системном хранилище, интерактивные `add`, `import`, `describe` и `resume`
запрашивают его скрыто только для текущего процесса.

`init` не запрашивает ключи, поскольку эта команда только создаёт несекретную
конфигурацию и завершается до запуска анализа. Она проверяет наличие ключей и
показывает предупреждение, если они пока не заданы.

Для CI рекомендуется задавать учётные данные через переменные окружения; в
обычном локальном запуске с `--json`, `--quiet` или `--non-interactive` также
можно использовать ранее сохранённые ключи:

```text
TELEGRAM_BOT_TOKEN
GEMINI_API_KEY
GH_TOKEN или GITHUB_TOKEN — только для операций GitHub
```

При необходимости заранее задать секреты в Windows PowerShell их можно безопасно
запросить для текущего сеанса без отображения и сохранения в истории команд:

```powershell
function Set-SessionSecret([string]$Name) {
    $secure = Read-Host "Введите $Name" -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
    try {
        $value = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
        [Environment]::SetEnvironmentVariable($Name, $value, "Process")
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

Set-SessionSecret "TELEGRAM_BOT_TOKEN"
Set-SessionSecret "GEMINI_API_KEY"
```

Переменные действуют только в текущем окне PowerShell и передаются запущенному
из него `mojilex`. Для публикации достаточно существующей авторизации
`gh auth login`; `GH_TOKEN` нужен главным образом для автоматизированных запусков.

В Linux или macOS:

```console
read -rsp "TELEGRAM_BOT_TOKEN: " TELEGRAM_BOT_TOKEN; echo
export TELEGRAM_BOT_TOKEN
read -rsp "GEMINI_API_KEY: " GEMINI_API_KEY; echo
export GEMINI_API_KEY
```

Токены и API keys запрещено передавать аргументами командной строки, сохранять
в `.mojilex.toml`, добавлять в `.env` внутри репозитория или коммитить в Git.
Системное хранилище — единственный встроенный постоянный способ хранения.

## Проверка импорта

Параметр `--dry-run` строит план без AI-запросов и постоянных изменений:

```powershell
mojilex add https://t.me/addemoji/PackName `
  --repo ..\mojilex `
  --dry-run
```

Параметр `--check-media` дополнительно загружает и проверяет медиа, не публикуя
результат:

```powershell
mojilex add https://t.me/addemoji/PackName `
  --repo ..\mojilex `
  --dry-run `
  --check-media
```

## Локальный импорт

```powershell
mojilex add https://t.me/addemoji/PackName `
  --repo ..\mojilex `
  --publish local `
  --max-cost-usd 0.25
```

Перед обращением к AI-провайдеру CLI сообщает модель, планируемое количество
запросов и верхнюю оценку стоимости. Если стоимость модели неизвестна,
неинтерактивный запуск завершается ошибкой без отправки запроса. Явное разрешение
неизвестной стоимости задается параметром `--allow-unknown-cost`.

Коллекция является атомарной единицей публикации: ошибка одного элемента не
приводит к публикации неполного набора.

## Проверка данных

```powershell
mojilex validate ..\mojilex
```

Параметр `--json` включает машиночитаемый результат. В этом режиме stdout
содержит один JSON-объект, а диагностические сообщения направляются в stderr.

## Поиск дубликатов

Перестроение локального индекса для всего набора данных:

```powershell
mojilex dedupe scan --all --repo ..\mojilex
```

Проверка отдельного эмодзи или коллекции:

```powershell
mojilex dedupe scan EMOJI_OR_COLLECTION_ID --repo ..\mojilex
```

Найденные совпадения являются кандидатами на ручную проверку. CLI не объединяет
эмодзи автоматически.

## Создание pull request

```powershell
mojilex add https://t.me/addemoji/PackName `
  --repo MojiLex/mojilex `
  --publish pr
```

Операция требует `GH_TOKEN` или `GITHUB_TOKEN` с необходимыми правами. Force
push не используется. Прямой push требует отдельного параметра `--direct-push`,
успешных проверок и подтверждения точного commit SHA.

## Продолжение прерванной операции

При прерывании или восстанавливаемой ошибке CLI возвращает `RUN_ID`:

```powershell
mojilex resume RUN_ID
```

Просмотр результата запуска:

```powershell
mojilex describe RUN_ID
```

Запись AI-кеша используется только при полном совпадении модели, prompt, схемы
и параметров запроса.

## Офлайн-доступ к snapshot

Команды чтения принимают каталог собранного snapshot, а не исходный репозиторий
данных:

```powershell
mojilex search "радость" --snapshot PATH_TO_SNAPSHOT --allow-unverified
mojilex get EMOJI_ID --snapshot PATH_TO_SNAPSHOT --allow-unverified
mojilex get-collection COLLECTION_ID --snapshot PATH_TO_SNAPSHOT --allow-unverified
mojilex similar EMOJI_ID --snapshot PATH_TO_SNAPSHOT --allow-unverified
```

Snapshot версии 0.2.0 проверяется по хешам, но не имеет подписанной цепочки
доверия Stage C. Диагностическое чтение требует явного
`--allow-unverified`. Этот параметр не изменяет статус доверия snapshot.

Команды чтения работают офлайн и не обращаются к Telegram, AI-провайдеру,
медиадекодерам, каталогам или зеркалам.

## Основные команды

```text
mojilex --help                         список команд
mojilex COMMAND --help                 параметры команды
mojilex init                           создание конфигурации
mojilex doctor                         проверка окружения
mojilex add SOURCE                     импорт источника
mojilex validate PATH                  проверка набора данных
mojilex dedupe scan --all              построение индекса дубликатов
mojilex resume RUN_ID                  продолжение операции
mojilex config show                    просмотр несекретной конфигурации
mojilex config set-credentials         сохранение API-ключей в системном хранилище
mojilex config clear-credentials       удаление сохранённых API-ключей
mojilex cache info                     состояние AI-кеша
mojilex cache prune                    очистка AI-кеша
mojilex snapshot verify PATH           проверка snapshot
mojilex uninstall                      полное удаление утилиты
```

Полный перечень команд приведен в [README.md](README.md). Точные параметры
доступны через `mojilex COMMAND --help`.

## Устранение неполадок

### Команда `mojilex` не найдена

После установки через `uv` добавьте каталог инструментов в `PATH` и откройте
новое окно терминала:

```console
uv tool update-shell
```

При разработке из исходников активируйте виртуальное окружение или используйте
исполняемый файл напрямую:

```powershell
.\.venv\Scripts\mojilex.exe --help
```

### Git сообщает `detected dubious ownership`

Добавьте в `safe.directory` только точные пути доверенных репозиториев. Если
PowerShell открыт в корне `mojilex-cli`, выполните:

```powershell
git config --global --add safe.directory (Resolve-Path .).Path
git config --global --add safe.directory (Resolve-Path ..\mojilex).Path
```

Не используйте универсальное значение `safe.directory=*`.

### Не обрабатывается TGS или WebM

```powershell
mojilex doctor
```

Результат содержит отдельное состояние WebP, rlottie/TGS и FFmpeg/WebM. Строка
`MojiLex doctor: succeeded` означает, что диагностика выполнилась; готовность к
работе определяется полем `ready`. Значение `ready False` требует устранить
перечисленные предупреждения. Для импорта TGS необходим доступный
`mojilex-rlottie-rgba`. После установки компонента не обязательно начинать
заново: продолжите сохранённый запуск командой `mojilex resume mlxrun_ВАШ_ID`.
В Windows ответьте `y` на предложение `doctor` или выполните
`mojilex doctor --install`. Установщик может запросить права администратора и
установить Visual Studio Build Tools с C++ workload для сборки TGS-адаптера.

### AI-запрос не выполняется

Проверьте точный ID модели, наличие требуемой переменной окружения,
`--max-ai-requests` и `--max-cost-usd`. Ответ AI-провайдера проходит строгую
локальную JSON Schema и дополнительные проверки. Некорректный ответ не
публикуется.

## Безопасность

- Поддерживаются только разрешенные формы публичных Telegram custom-emoji URL.
- Репозитории GitHub принимаются только через безопасные HTTPS/SSH remotes без
  встроенных учетных данных.
- Для загрузки, распаковки, пикселей, длительности, кадров, памяти, времени и
  временного диска применяются жесткие ограничения.
- Декодирование медиа выполняется в отдельном процессе без shell и API keys.
- Данные с чувствительным содержимым, предупреждениями или недостаточной
  уверенностью требуют человеческой проверки.
- Грязное рабочее дерево не сохраняется автоматически через stash.
- Force push не используется.

Дополнительная информация: [Security model](docs/security-model.md) и
[Publishing](docs/publishing.md).

## Полное удаление

Для установки из раздела «Быстрый старт» выполните:

```console
mojilex uninstall
```

Команда покажет точный план и после подтверждения удалит установленный через
`uv` пакет, стандартные конфигурацию/кеш/данные запусков MojiLex, сохранённые
ключи и собственный TGS-адаптер. Общие `uv`, Git, FFmpeg и Visual Studio
останутся на месте. Флаг `--keep-data` сохранит конфигурацию, кеш, запуски и
ключи.

## Ограничения версии 0.2.0

- Успешный API-запрос не означает автоматическую квалификацию модели.
- Визуально похожие эмодзи требуют ручного решения.
- Подписанный каталог, аттестации и отзыв доверия относятся к Stage C.
- Разделенные snapshot, delta-обновления и масштабирование относятся к Stage
  D/E.

Код CLI распространяется по лицензии [MIT](LICENSE). Условия публикации
метаданных определены в репозитории данных.
