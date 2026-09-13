# CLI development

English | [Русский](development_RU.md) · [Back to the README](../README.md)

This page is for changing CLI code. For normal use, follow the README installation
and run `mojilex`; cloning repositories is unnecessary.

## Install from source

Use Python 3.11 or newer and Git. Clone the CLI:

```console
git clone https://github.com/MojiLex/mojilex-cli.git
cd mojilex-cli
```

With `uv` installed, create an editable development environment:

```console
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Or use standard Python tools on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

On Linux or macOS:

```console
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[dev]"
```

Check `python3 --version` first if using that interpreter. Activation is optional:
the commands below use the environment's Python explicitly.

Clone the data repository alongside the CLI only when working with a real local
dataset: `git clone https://github.com/MojiLex/mojilex.git ../mojilex`.
Keep its worktree separate; it has its own schema and validation commands.

## Run checks

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe tools/verify_no_secrets.py
.\.venv\Scripts\python.exe -m build
```

Linux or macOS:

```console
.venv/bin/python -m pytest
.venv/bin/python -m ruff format --check .
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
.venv/bin/python tools/verify_no_secrets.py
.venv/bin/python -m build
```

The `dev` extra in [pyproject.toml](../pyproject.toml) supplies test, type-check,
lint, build, and audit tools. `python -m pip_audit` checks the active environment's
dependencies and needs network access. CI tests Python 3.11 and 3.13 on Windows,
Linux, and macOS, then installs built wheels into clean environments and probes
the media backends. A local pass does not establish cross-platform or installed
wheel behavior. See the [CI workflow](../.github/workflows/ci.yml).

## Media and runtime checks

Run `.venv\Scripts\mojilex.exe doctor` on Windows or `.venv/bin/mojilex doctor`
on Linux/macOS. WebP support comes with Pillow; WebM needs FFmpeg/ffprobe and TGS
needs the lossless MojiLex rlottie RGBA adapter. See
[Media prerequisites](media-prerequisites.md) for setup and fixture checks.

`doctor` succeeding means diagnostics completed; readiness also requires its
reported `ready` value to be true. On supported Windows setups, `doctor --install`
can install missing components and may request administrator approval or install
Visual Studio C++ Build Tools for the adapter.

Tests use controlled providers and fixtures. A real import, model benchmark, or
publication is a separate runtime operation: it may access external services,
consume a model budget, or write to GitHub. Use a deliberate test dataset and
explicit publication mode when running those commands.

## Project map and contributions

Source, media, vision-provider, dataset, run-store, Git, and GitHub adapters are
separate; domain models do not depend on Telegram, Gemini, or a GitHub SDK.
Start with [Architecture](architecture.md), then use
[Advanced usage](advanced-usage.md), [Publishing](publishing.md),
[Benchmarks](benchmarks.md), and [Security model](security-model.md) as needed.

Code and documentation use the [MIT license](../LICENSE). Contributions creating
dataset metadata are submitted to the separate data repository under CC0-1.0.
See [Contributing](../CONTRIBUTING.md) before opening a change.
