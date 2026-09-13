# MojiLex CLI

## Everyday commands

After setup, run `mojilex` to open the terminal menu. Use arrows to choose,
Enter to open, and Esc to go back. My packs separates ready descriptions from
another unfinished operation. Adding a URL downloads and analyzes locally;
GitHub publication is a separate action. Unknown GitHub status is never shown
as confirmed publication.

Direct commands remain available:

```console
mojilex list
mojilex show NewsEmoji
mojilex gallery NewsEmoji
mojilex settings
mojilex publish NewsEmoji --local
mojilex publish NewsEmoji
mojilex resume NewsEmoji
```

`show` opens a searchable terminal list with a language switch and item details.
Use `show --all` to print every field. Pipes and JSON retain non-interactive output.
`gallery` (also `show --browser`) opens a standalone local HTML gallery with saved
previews, search, Russian descriptions and expandable details. Missing retained
frames show a placeholder; viewing never downloads media or calls AI. JSON mode
does not open a browser. `review` remains a compatible optional browsing command,
hidden from basic help. `--help-all` lists all advanced commands.

`settings` displays and edits ordinary analysis settings, limits and retry rules
in a terminal. Environment overrides remain authoritative, saved run settings
stay unchanged, and language changes apply on the next invocation.

`publish --local` validates without upload; `publish` creates
a GitHub pull request from the completed draft, without repeating AI analysis.
Content ratings and warnings remain in the data but never require manual approval,
block publication, or trigger model escalation. Missing model qualifications do not
require human approval. Structural validation and verification of declared
qualification metadata still apply.

Start a new pack with `mojilex import PACK_URL`, then `mojilex describe PACK_NAME`.
`show` selects ready results, `publish` selects the latest completed draft, and
`resume` selects the latest unfinished run. Completed runs are not restarted.
`list` shows ready results and separate progress for another unfinished run.
Ambiguous names across repositories/source groups require an explicit Run ID from
`list`. Commands support `--json`; existing Run IDs, `submit`, and explicit
`review ID approve` remain supported.

Long-running `add`, `import`, `describe`, and `resume` operations show one updating
terminal panel with completed items, processing, retries, remaining failures and
elapsed time. AI analysis includes the shared request budget. Prompts pause redraw.
Piped/JSON callers retain line logs. A heartbeat repeats every 5 seconds while
waiting; elapsed time never counts as
completed work. Media failures stop queued jobs while already active jobs finish
safely. Imports checkpoint each completed file, so `mojilex resume RUN_ID` checks
retained frame hashes and source identity, skips completed downloads/renders, and
starts progress at the completed count. Missing or corrupt frames fall back to source
verification. Older runs whose frames were already removed need one recreation pass.
Progress goes to stderr
(including with `--json`) and is hidden by `--quiet`.

English | [Русский](README_RU.md)

## Quick start

Normal use does not require cloning either repository or creating an IDE
project. On Windows, first install
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) once:

```console
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close PowerShell or Command Prompt after installation and open a new window.
Then run the following commands in either shell:

```console
uv --version
uv tool install --python 3.11 "git+https://github.com/MojiLex/mojilex-cli.git@main"
uv tool update-shell
```

Markdown square brackets and parentheses are not part of the Git URL; copy the
quoted argument without link markup. After `uv tool update-shell`, open another
new terminal and verify the installation:

```console
mojilex --version
```

Select Russian for the first setup or one command. `init` stores the selected
interface language in the user configuration, outside the installed package:

```powershell
mojilex --ui-language ru init
mojilex --ui-language ru --help
mojilex config set-ui-language ru
```

`mojilex config set-ui-language` changes only the saved interface language and
preserves the other non-secret settings. Package upgrades do not overwrite this
user configuration. `MOJILEX_UI_LANGUAGE` remains available as a temporary
environment override. Command names, option names, and JSON fields remain stable
in English.

Authenticate with GitHub once and create the non-secret configuration:

```console
gh auth login
mojilex init --model gemini-3.8-flash --non-interactive
mojilex config set-credentials
```

`init` creates only non-secret settings and never requests or stores API keys.
The `--non-interactive` flag also disables the settings wizard.
`config set-credentials` requests the Telegram and Gemini values through hidden
input and stores them in the operating-system keyring. Environment variables
override stored values. To keep credentials only for one run, skip this command:
an interactive `mojilex add ...` requests missing values without persisting them.
If the current desktop session has no usable keyring service, the save command
fails safely and asks you to use environment variables instead.

If an older configuration contains a missing relative repository path, replace
it with a safe no-publication default:

```console
mojilex init --force --repo MojiLex/mojilex --publish local --model gemini-3.8-flash --non-interactive
```

### Staged analysis and publication

AI progress distinguishes validated emojis, batches, requests, retries, and queued
work. Invalid AI output or exhausted transient network retries defer unfinished
emojis while the remaining queue continues. Deferred items are then retried individually
until they succeed or the shared budget runs out; completed descriptions are retained.
Budget/cost limits, declined approval, authentication or persistence errors, and cancellation
stop queued work. Successful in-flight batches still checkpoint their results. Each request has a
30-second timeout. An unchanged emoji count can mean a batch is still awaiting
or validating its response, not that those emojis are complete.
During per-item recovery, each validated result is checkpointed and counted
immediately, even if a later item fails.

Telegram allows up to 6 attempts for transient connection failures by default (explicit
`telegram.max_attempts` values remain unchanged; allowed range 1–8). A broken file
download restarts with a fresh bounded buffer. Gemini allows up to 3 attempts for
transient failures, waiting 1 and 2 seconds, with every attempt charged to the existing
request budget. Authentication and ordinary invalid requests are not retried.

Prompt 1.2.1 explicitly specifies cross-field text, number, style and uncertainty
rules and the exact suggested_uses values, distinct from free-form usage descriptions.
Local semantic validation remains strict and reports specific rule codes.
Validated 1.1.0 and 1.2.0 results keep their exact original provenance and batch grouping
during resume; new descriptions use 1.2.1.

Generated PNG frames are retained in `resume-media` under the configured cache
directory, outside the dataset, and share the run disk budget with transient files.
They remain available after import for describe/resume. Metadata `cache prune` does
not delete frame directories; raw downloaded files are still transient.

To process several AI batches at once, set `--ai-concurrency` (1–16):

```console
mojilex describe RUN_ID --ai-concurrency 4
mojilex resume RUN_ID --ai-concurrency 4
```

To download and process more remaining media concurrently, `resume` also accepts
`--download-concurrency` from 1 to 32:

```console
mojilex resume RUN_ID --download-concurrency 8 --ai-concurrency 4
```

Omitting this flag preserves the saved media concurrency. Higher values use more
CPU and memory; disk and per-file limits still apply.

`add` also accepts `--ai-concurrency`. Without it, saved runs keep their previous
concurrency; new runs use configuration (default: 1). This changes parallelism
without resetting request/cost budgets or increasing their limits. Completed
results are reused on resume. Actual speed depends on model latency and provider
quotas; retrying invalid responses can still dominate the running time.

When current model pricing is unknown, one confirmation covers the remaining
request limit for this invocation, including retries. It never increases
`--max-ai-requests`; a new `resume` invocation asks again. The default is no to
prevent accidental spending. Explicit `--allow-unknown-cost` skips this cost
question but does not authorize publication or other sensitive actions.

First download and verify the media. Nothing is uploaded to GitHub, and the
command returns a persistent `mlxrun_...` ID:

```console
mojilex import "https://t.me/addemoji/PackName" --repo MojiLex/mojilex
```

Generate AI metadata in the same staged run, still without publishing:

```console
mojilex describe mlxrun_YOUR_ID
```

`describe` also accepts an `add` run whose publication has not started. It creates
a persistent local draft at the recorded base, preserving the Run ID, completed
descriptions, and consumed budget. Results requiring human review can therefore
finish analysis before publication. An existing publication checkpoint cannot be
converted to another workflow.

Validate and preview the result without uploading it:

```console
mojilex submit mlxrun_YOUR_ID --publish local
```

When the result is ready, select one publication mode:

```console
# Create a branch and pull request
mojilex submit mlxrun_YOUR_ID --publish pr

# Push directly to main after validation and explicit confirmation
mojilex submit mlxrun_YOUR_ID --direct-push
```

Replace `PackName` and `mlxrun_YOUR_ID` with the actual pack name and the exact
ID returned by `import`. Missing Telegram and Gemini credentials are loaded from
the system keyring or requested through hidden one-time prompts.

For a one-step check without AI requests or a persistent staged run:

```console
mojilex add "https://t.me/addemoji/PackName" --dry-run --check-media --repo MojiLex/mojilex
```

Git, GitHub CLI (`gh`), and the applicable media backends are required. WebP
works after the one-command install. Run `mojilex doctor` to check WebM and TGS.
On Windows it offers an interactive `[y/N]` installation when a supported
component is missing. Choose `y`, or run `mojilex doctor --install`, to install
only the required FFmpeg and/or TGS build components and rerun the checks.

`mojilex` imports public Telegram custom-emoji sets, prepares temporary media for
deterministic analysis, creates Russian and English semantic metadata, finds duplicate
candidates, validates the
[MojiLex dataset](https://github.com/MojiLex/mojilex), and publishes a local change or a
GitHub pull request.

The CLI never stores original emoji media, rendered frames, contact sheets, API
keys, or Telegram download URLs in Git, configuration files, or its persistent
cache. Credential persistence is opt-in and uses the operating-system keyring.
Telegram media remains owned by its respective rights holders.

## Status

This repository contains the `0.2.0` MVP. It adds bounded, offline, read-only access to an
explicitly selected `distribution-v1` snapshot. The data format has its own independently
versioned JSON Schema (`1.0.0`). Python 3.11 or newer is required.

## Development installation from source

This section is for CLI development. Normal users should use the one-command
installation in Quick start.

Clone the CLI and data repositories side by side:

```console
git clone https://github.com/MojiLex/mojilex-cli.git
git clone https://github.com/MojiLex/mojilex.git
cd mojilex-cli
```

Install on Windows with PowerShell:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\mojilex.exe --version
.\.venv\Scripts\mojilex.exe doctor
```

If `python` is not available, install a supported Python release from
[python.org](https://www.python.org/downloads/) with the PATH option enabled,
then open a new terminal. MojiLex requires Python 3.11 or newer.

Install on Linux or macOS:

```console
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install .
.venv/bin/mojilex --version
.venv/bin/mojilex doctor
```

Activation is optional. After activation, the shorter `mojilex` command can be
used in the examples below.

Verify the optional media backends at any time:

```console
mojilex --version
mojilex doctor
```

Pillow-based WebP processing is included. FFmpeg/ffprobe and the MojiLex lossless rlottie RGBA
adapter are external prerequisites for WebM and TGS. The lossy upstream `lottie2gif` utility is
not supported. See
[Media prerequisites](docs/media-prerequisites.md).

`MojiLex doctor: succeeded` means that the diagnostic command completed. The
environment is ready only when the reported `ready` value is `true`. A TGS
import requires an available `mojilex-rlottie-rgba` adapter. After installing
it, continue a saved failed run with `mojilex resume mlxrun_YOUR_ID` instead of
starting over. On Windows, answer `y` when `doctor` offers to install the
missing components, or run `mojilex doctor --install`. The installer may request
administrator approval and install Visual Studio Build Tools with the C++
workload when the TGS adapter must be built.

## Separate analysis and submission

The `add` command above is the normal path and performs analysis and pull-request
publication in one run. To review the generated change before uploading it, use
the staged workflow:

```console
mojilex import https://t.me/addemoji/PackName
mojilex describe RUN_ID
mojilex submit RUN_ID
```

Use the `RUN_ID` printed by `import` in the next two commands. Commands first
use environment variables, then optional system-keyring values saved by
`mojilex config set-credentials`, and finally request missing Telegram and
Gemini credentials through hidden prompts. For JSON, quiet, piped-input, or
non-interactive operation, use saved credentials or set `TELEGRAM_BOT_TOKEN`
and `GEMINI_API_KEY` in the process environment.
`init` does not request secrets because it exits after writing non-secret
configuration. Existing `gh auth login` authorization is sufficient for normal
pull-request publication; `GH_TOKEN` or `GITHUB_TOKEN` is intended primarily for
automation.

Direct pushes remain separate and require `--direct-push`, write access,
successful checks for the exact candidate commit, an unchanged base branch,
ruleset bypass permission, and a late confirmation showing the final commit SHA
and exact paths. Force push is never used.

## Commands

```text
mojilex init
mojilex add SOURCE...
mojilex import SOURCE...
mojilex describe RUN_ID
mojilex validate [PATH]
mojilex update SOURCE|COLLECTION_ID|--all
mojilex submit [PATH|RUN_ID]
mojilex build-index [PATH]
mojilex dedupe scan SELECTOR|--all
mojilex dedupe explain EMOJI_ID EMOJI_ID
mojilex dedupe review EMOJI_ID [--against EMOJI_ID] --reviewer HANDLE
mojilex benchmark-dedupe --manifest PATH
mojilex benchmark-model --provider NAME --model ID --benchmark-manifest PATH
mojilex resume RUN_ID
mojilex review EMOJI_ID approve|request-changes|reject
mojilex set-status ENTITY_ID --availability STATUS --reason CODE
mojilex takedown ENTITY_ID --reason CODE
mojilex doctor
mojilex config show
mojilex config set-credentials
mojilex config clear-credentials
mojilex cache info
mojilex cache prune
mojilex snapshot verify PATH
mojilex search QUERY --snapshot PATH
mojilex get EMOJI_ID --snapshot PATH
mojilex get-collection COLLECTION_ID --snapshot PATH
mojilex resolve --platform NAME --namespace NAME --scope ID --native-id ID --snapshot PATH
mojilex similar EMOJI_ID --snapshot PATH
mojilex uninstall
```

Use `mojilex COMMAND --help` for exact options. With `--json`, stdout contains exactly one
machine-readable envelope; progress and diagnostics go to stderr.

Run `mojilex snapshots` to list existing local release snapshots and their paths. Discovery checks
the current directory and the configured local dataset repository, including their `dist`,
`snapshots`, and `releases` directories and one child-directory level. It does not scan your disk
recursively or download a remote catalog. If exactly one snapshot exists, read commands select it
automatically. Otherwise pass `--snapshot PATH` or set `MOJILEX_SNAPSHOT` to pin an existing snapshot.
An explicit `MOJILEX_SNAPSHOT` also limits listing to that path. Import checkpoints and AI drafts
are not release snapshots; an empty local list does not mean Telegram itself has no such emoji.

`mojilex get TELEGRAM_CUSTOM_EMOJI_ID` accepts the exact decimal Telegram ID as well as a canonical
`mxe_...` ID. It resolves only an unambiguous current identity in the selected snapshot. `search`
also recognizes exact native IDs. Human output shows readable descriptions; `--json` retains the
complete structured records.

The pre-enforcement MVP snapshot is integrity-checked but unsigned, so reads fail closed unless diagnostic use is
explicitly acknowledged with `--allow-unverified`; this never makes the release trusted and
`runtime_trust.safe_eligible` remains false. `search` defaults to the safe `agent` view, while
`--view search --allow-unverified` or `--view canonical --allow-unverified` selects an explicit
diagnostic projection. The default agent view can remain empty for unsigned records even with
`--allow-unverified`. The read path is
offline-only and does not call AI providers, Telegram, media decoders, catalogs, mirrors, or any
other network service.

Signed catalog/revocation/errata enforcement and attestations (Stage C), partitioned releases,
deltas, and scale indexes (Stage D), and the external data plane (Stage E) remain post-MVP.

## Safety defaults

- Sources are limited to exact supported Telegram public custom-emoji URL forms.
- Repository remotes are limited to safe GitHub HTTPS/SSH forms without embedded credentials.
- Download, decompression, pixel, duration, frame, memory, time, and temporary-disk limits are
  enforced before publication.
- Media decoding runs in a separate process without a shell or API credentials.
- Rendering/color/alpha facets and compact fingerprints are computed locally from the complete
  decoded frame stream. Near candidates use a bounded local index rather than an all-pairs scan.
- AI produces descriptions, literal text, semantic tags, and semantic facets as one object. Image
  text is untrusted content, never an instruction. An unqualified result needs content-bound human
  approval before official submission.
- Candidate reports, the local LSH/SQLite index, decoded frames, and review previews never enter
  the data repository. Native emoji IDs are never merged automatically.
- A collection is the atomic publication unit. One broken item cannot publish a partial pack.
- Sensitive, adult, unknown, or critically warned AI output requires human approval before it
  can enter public data.
- A dirty worktree is never stashed automatically and force push is never used.

See [Security model](docs/security-model.md) and [Publishing](docs/publishing.md).

## Configuration

Precedence is command line, environment, project `.mojilex.toml`, user config, then safe
defaults. Configuration files are rejected if they contain known secret fields.

```toml
[repository]
target = "MojiLex/mojilex"
base_branch = "main"
publish = "pr"

[telegram]
timeout_seconds = 30
download_concurrency = 4

[ai]
provider = "gemini"
model = "your-explicit-model-id"
languages = ["ru", "en"]
max_ai_requests = 100
ai_concurrency = 1
model_routing = "off"
# escalation_model = "your-explicit-stronger-model-id"

[dedupe]
mode = "exact"
profile = "dedupe-v1"
max_candidates = 20

[processing]
static_batch_size = 16
animated_batch_size = 8
keyframes = 8
render_timeout_seconds = 15
```

When a provider has no current price record, each real uncached AI request is authorized at its
budget-reservation point. Use `--allow-unknown-cost` for an explicit standing opt-in; JSON or
non-interactive runs otherwise fail closed. Cache-only runs do not ask for cost authorization.

## Uninstallation

For installations created by the Quick start command, run:

```console
mojilex uninstall
```

After showing the exact deletion plan and receiving confirmation, the command
removes the `uv` tool installation, default MojiLex configuration/cache/run
data, credentials saved by MojiLex, and the MojiLex TGS adapter. Shared tools
and components (`uv`, Git, FFmpeg, and Visual Studio) are preserved. Use
`mojilex uninstall --keep-data` to keep configuration, run data, cache, and
saved credentials.

## Development

```console
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check src tests
.venv\Scripts\python -m mypy src
.venv\Scripts\python -m build
```

The code is organized around source, media, vision-provider, dataset, run-store, Git, and GitHub
adapters. Domain models do not depend on Telegram, Gemini, or a GitHub SDK. See
[Architecture](docs/architecture.md).

## License and contributions

CLI code and documentation are available under the [MIT License](LICENSE). Contributions that
create dataset metadata, descriptions, or tags are submitted to the data repository under
CC0-1.0 as described there. See [CONTRIBUTING.md](CONTRIBUTING.md).
