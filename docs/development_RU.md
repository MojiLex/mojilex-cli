# Разработка CLI

[English](development.md) | Русский · [К README](../README_RU.md)

Эта страница нужна для изменения кода CLI. Для обычной работы установите программу
по README и запустите `mojilex`; клонировать репозитории не требуется.

## Установка из исходников

Нужны Python 3.11 или новее и Git. Клонируйте CLI:

```console
git clone https://github.com/MojiLex/mojilex-cli.git
cd mojilex-cli
```

Если установлен `uv`, создайте окружение с редактируемой установкой:

```console
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Или используйте стандартные средства Python в Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Для Linux и macOS:

```console
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

При использовании `python3` сначала проверьте `python3 --version`. Активировать
окружение необязательно: ниже Python из него вызывается явно.

Клонируйте базу рядом с CLI только для работы с реальным локальным набором данных:
`git clone https://github.com/MojiLex/mojilex.git ../mojilex`.
Держите рабочую копию отдельно; у базы собственные схема и команды проверки.

## Запуск проверок

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe tools/verify_no_secrets.py
.\.venv\Scripts\python.exe -m build
```

Linux и macOS:

```console
.venv/bin/python -m pytest
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
.venv/bin/python tools/verify_no_secrets.py
.venv/bin/python -m build
```

Набор `dev` в [pyproject.toml](../pyproject.toml) устанавливает средства тестирования,
проверки типов, оформления, сборки и аудита. `python -m pip_audit` проверяет
зависимости активного окружения и требует сети. CI запускает тесты на Python 3.11
и 3.13 в Windows, Linux и macOS, затем устанавливает собранные wheel-пакеты в
чистые окружения и проверяет обработчики медиа. Локальный успех не подтверждает
работу других ОС и установленного wheel. См. [CI](../.github/workflows/ci.yml).

## Медиа и проверка реального запуска

В Windows запустите `.venv\Scripts\mojilex.exe doctor`, в Linux/macOS —
`.venv/bin/mojilex doctor`. WebP работает через Pillow; WebM требует FFmpeg/ffprobe,
а TGS — адаптер без потерь MojiLex rlottie RGBA. Установка и проверки контрольных
файлов описаны в [Компонентах медиа](media-prerequisites.md).

Успешное завершение `doctor` означает, что диагностика выполнена; для готовности
также нужно значение `ready: true`. На поддерживаемых Windows `doctor --install`
устанавливает недостающее и может запросить права администратора или установить
Visual Studio C++ Build Tools для сборки адаптера.

Тесты используют контролируемые провайдеры и тестовые данные. Реальный импорт,
бенчмарк модели или публикация — отдельная проверка: она может обращаться к внешним
сервисам, расходовать бюджет модели или записывать данные на GitHub. Для таких
команд осознанно выбирайте тестовую базу и режим публикации.

## Устройство проекта и участие

Адаптеры источников, медиа, AI-провайдеров, базы, хранилища запусков, Git и GitHub
разделены; доменные модели не зависят от Telegram, Gemini или GitHub SDK.
Начните с [Архитектуры](architecture.md); по задаче используйте
[Расширенное использование](advanced-usage_RU.md), [Публикацию](publishing.md),
[Бенчмарки](benchmarks.md) и [Модель безопасности](security-model.md).

Код и документация распространяются по [MIT](../LICENSE). Метаданные для базы
передаются в отдельный репозиторий на условиях CC0-1.0.
Перед отправкой изменений прочитайте [Contributing](../CONTRIBUTING.md).
