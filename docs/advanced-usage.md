# Advanced usage

English | [Русский](advanced-usage_RU.md) · [Back to the quick start](../README.md)

Use `mojilex` for everyday work. This page covers separate processing stages,
automation, local snapshots, and repository maintenance. Replace `NewsEmoji` with
your pack name; `mojilex list` shows the available names and saved runs.

## Separate download, analysis, and publication

```console
mojilex import "https://t.me/addemoji/NewsEmoji"
mojilex describe NewsEmoji --ai-concurrency 4
mojilex show NewsEmoji
mojilex publish NewsEmoji --local
mojilex publish NewsEmoji
```

`import` downloads and checks media. `describe` creates a persistent local draft.
`publish --local` validates the completed draft without uploading it; `publish`
creates or updates a GitHub pull request without repeating AI analysis.

Previously saved descriptions remain compatible. Dataset writes use filenames
with an 8-character hash prefix; this needs no new AI analysis. When `main` changes
after publication, the repository's enabled **Refresh open data PRs** workflow
updates PRs from branches in the same repository and starts validation. For a
fork PR, repeat `mojilex publish NewsEmoji`. Incompatible edits to the same record
still require resolving that record's conflict.

`show` selects ready descriptions, `publish` selects a completed draft, and
`resume` selects the latest unfinished run. If a name refers to multiple
repositories or source groups, use the exact Run ID shown by `list`.

For scripts, the older Run ID workflow remains available:

```console
mojilex describe mlxrun_YOUR_ID
mojilex submit mlxrun_YOUR_ID --publish local
mojilex submit mlxrun_YOUR_ID --publish pr
```

`describe` can also continue an `add` run before publication starts, preserving
its identity, completed descriptions, and consumed budget. A publication
checkpoint cannot be converted into another workflow.

The direct `add` command combines analysis and the configured publication mode.
Specify `--publish local` when using it without GitHub publication. To check
source media without AI requests or a persistent staged run:

```console
mojilex add "https://t.me/addemoji/NewsEmoji" --dry-run --check-media
```

## Resume, speed, and request limits

```console
mojilex resume NewsEmoji --download-concurrency 8 --ai-concurrency 4
```

AI, download, decoder and pack concurrency accept any positive integer, with no
fixed upper ceiling.
The default `processing.performance_mode = "auto"` sizes concurrency at operation
start from available CPU and RAM; larger configured values remain in effect.
Choose `manual` in settings to use the four configured concurrency values exactly.
The automatic targets are one decoder per available logical CPU, bounded by 75%
of available RAM at 512 MiB per decoder; four times that decoder target for both
preparing packs and AI requests, and four downloads per CPU. This is a startup estimate, not
a guarantee that higher parallelism improves throughput. Automatic temporary storage
can grow to twice available RAM, while remaining at most one quarter of free space
on the temporary volume. If RAM or disk probing fails, its allowance is not increased.
Manual mode preserves the configured temporary storage allowance. Higher concurrency uses more memory and CPU; provider
quotas and response time still limit speed. It never increases request or cost
limits. Change defaults for future runs with `mojilex settings`.

`processing.file_analysis_mode` controls a multi-pack file run: `fast` overlaps
preparation with AI work and saves independent packs as they become ready. Packs
sharing emoji identities keep dependency order, and dataset writes use one writer.
`sequential` completes one
pack at a time; `download_all` persists all originals before decoding; and
`prepare_all` downloads and decodes packs concurrently before any AI request.

The default budget is **100 AI requests for the whole run**, including retries,
individual recovery requests, and optional model escalation. Resume retains the
consumed count. Cache hits avoid requests. A new `add` run can set
`--max-ai-requests` and `--max-cost-usd`; resuming is not a budget reset.

Telegram transient connection failures allow up to 6 attempts by default
(`telegram.max_attempts`: 1–8). A broken download restarts with a fresh bounded
buffer. Gemini transient failures allow up to 3 attempts with 1- and 2-second
waits; each attempt counts toward the budget and has a 90-second timeout.
Authentication and ordinary invalid requests are not retried.

Invalid AI responses and exhausted transient retries defer unfinished emojis
while the remaining AI queue continues. Deferred emojis are retried individually
within the shared budget. Each validated result is saved immediately. Budget or
cost limits, declined authorization, authentication or persistence errors, and
cancellation stop queued work; successful active batches still save results.
Media failures stop queued media jobs after their retry allowance; active jobs
finish safely. Resume checks source identity and retained frame hashes, reuses
completed work, and recreates missing or corrupt frames when necessary.

When pricing is unknown, interactive approval covers the remaining request
allowance for the current invocation, including retries. It never raises the
limit; a later resume asks again. `--allow-unknown-cost`, where supported, provides
explicit opt-in. Automation otherwise stops when that authorization is missing.
Cache-only work does not need cost approval.

Counted work shows completed items; blocking stages show an animated indicator,
the current activity, and elapsed time. Waiting time is not completed work.
Prompts pause redraw. Progress goes to stderr, including with `--json`;
`--quiet` hides progress.

## Local storage and credentials

Original downloads and contact sheets are temporary. Selected PNG frames are
retained locally in `resume-media` under the configured cache directory, outside
the dataset, for resume and gallery previews. They share the run disk budget
with temporary files. Metadata `cache prune` does not remove these frame
directories. Old runs whose frames were already removed may need one rendering
pass to resume; the gallery uses a placeholder when a preview is unavailable.

Original media, retained frames, contact sheets, credentials, and Telegram
download URLs are never committed to the dataset. Viewing saved descriptions or
the gallery makes no AI or media download requests. Telegram media remains owned
by its respective rights holders.

Configuration precedence is **command line → environment → project
`.mojilex.toml` → user configuration → defaults**. Use `mojilex config show` to
inspect resolved non-secret settings. `settings` changes a value in the project
configuration if it is defined there, otherwise in the user configuration. It
preserves saved run settings and cannot override environment variables.

Example project configuration, with an explicit model ID of your choice:

```toml
[repository]
target = "MojiLex/mojilex"
base_branch = "main"
publish = "pr"

[telegram]
timeout_seconds = 30
download_concurrency = 4
max_attempts = 6

[ai]
provider = "gemini"
model = "YOUR_MODEL_ID"
languages = ["ru", "en"]
max_ai_requests = 100
ai_concurrency = 1
model_routing = "off"

[dedupe]
mode = "exact"
profile = "dedupe-v1"
max_candidates = 20

[processing]
file_analysis_mode = "fast"
pack_concurrency = 3
render_concurrency = 2
static_batch_size = 16
animated_batch_size = 8
keyframes = 8
render_timeout_seconds = 15
```

Environment overrides include `MOJILEX_MODEL`, `MOJILEX_AI_CONCURRENCY`,
`MOJILEX_DOWNLOAD_CONCURRENCY`, `MOJILEX_MAX_AI_REQUESTS`, `MOJILEX_MAX_COST_USD`,
`MOJILEX_FILE_ANALYSIS_MODE`, `MOJILEX_REPO`, `MOJILEX_CACHE_DIR`, and
`MOJILEX_RUNS_DIR`.
`mojilex config set-ui-language ru` changes only the saved interface language;
`MOJILEX_UI_LANGUAGE` overrides it temporarily. Language changes apply on the next
invocation. Commands, flags, and JSON field names remain in English.

`init` writes non-secret settings. `config set-credentials` stores secrets in the
OS keyring through hidden input; `config clear-credentials` removes them. Without
saved credentials, interactive analysis asks for missing values for that process
only. Environment variables take precedence over keyring values. If no usable
keyring is available, saving fails without writing secrets to configuration.

For JSON, quiet, piped-input, and non-interactive runs, provide stored credentials
or process environment variables `TELEGRAM_BOT_TOKEN` and `GEMINI_API_KEY`.
Normal GitHub publication uses `gh auth login`; `GH_TOKEN` or `GITHUB_TOKEN` are
also available for automation. Never put these values into a TOML file or Git.

## Publication and optional review

A pull request proposes data for inclusion; creating it is not a merge. The CLI
validates the full dataset before submission. A collection is the atomic unit:
one broken item cannot publish a partial pack. Content warnings and ratings stay
in the records and **never require approval or block publication**. Missing
model qualifications do not require human approval; declared qualification
metadata and structural validity are still checked.

`review NewsEmoji` remains a compatible viewing command. Explicit
`review EMOJI_ID approve|request-changes|reject` records an optional editorial
decision. `dedupe review` is a separate decision about a relation between two
emojis. Similarity never merges native emoji IDs automatically. See
[Duplicate analysis and review](dedupe-and-review.md).

Direct push is an advanced owner operation:

```console
mojilex submit mlxrun_YOUR_ID --direct-push
```

It requires write access and branch-bypass capability, validation, passing
required checks for the exact candidate commit, an unchanged base branch, and
confirmation of the final commit and paths. The CLI never force-pushes or
weakens GitHub rules. Details: [Publishing](publishing.md) and
[Security model](security-model.md).

## Offline release snapshots

Saved pack drafts and release snapshots are different. `mojilex snapshots` lists
existing snapshots near the current directory and configured local data
repository, including `dist`, `snapshots`, and `releases` and one child-directory
level. It does not scan the whole disk or download releases. If exactly one
snapshot exists, read commands select it automatically; otherwise use
`--snapshot PATH`. `MOJILEX_SNAPSHOT` pins one existing snapshot and limits listing
to that path.

```console
mojilex snapshots
mojilex snapshot verify PATH
mojilex search "celebration" --snapshot PATH --view search --allow-unverified
mojilex get TELEGRAM_CUSTOM_EMOJI_ID --snapshot PATH --allow-unverified
```

`get` accepts an exact decimal Telegram custom-emoji ID or a canonical `mxe_...`
ID and requires an unambiguous current identity. `search` also recognizes exact
native IDs. Reads are offline and never call Telegram, AI, decoders, or a catalog.

The MVP `distribution-v1` snapshot is integrity-checked but unsigned. Reading it
requires explicit diagnostic opt-in with `--allow-unverified`; this does not
establish trust and `runtime_trust.safe_eligible` remains false. The default
`agent` view can therefore be empty even with that flag. Use an explicit `search`
or `canonical` view for diagnostic searches. Signed catalog enforcement,
revocations, attestations, partitioned releases, and deltas are future work.

## Command reference

Use `mojilex --help-all`, then `mojilex COMMAND --help` for exact arguments.

| Purpose | Commands |
| --- | --- |
| Daily work | `list`, `show`, `gallery`, `settings`, `publish`, `resume` |
| Import and analysis | `add`, `import`, `describe`, `update SOURCE\|COLLECTION_ID\|--all` |
| Repository maintenance | `validate [PATH]`, `submit [PATH\|RUN_ID]`, `build-index [PATH]` |
| Duplicate relations | `dedupe scan`, `dedupe explain`, `dedupe review` |
| Editorial maintenance | `review`, `set-status`, `takedown` |
| Benchmarks | `benchmark-dedupe`, `benchmark-model` |
| Setup and diagnostics | `init`, `doctor`, `config show`, `config set-ui-language` |
| Stored credentials | `config set-credentials`, `config clear-credentials` |
| Cache | `cache info`, `cache prune` |
| Offline data | `snapshots`, `snapshot verify`, `search`, `get`, `get-collection`, `resolve`, `similar` |
| Removal | `uninstall` |

With `--json`, stdout contains one machine-readable envelope. Diagnostics and
progress use stderr. See [Error and exit codes](error-reference.md) for script
handling, and [Benchmarks](benchmarks.md) for evaluation commands.

## Troubleshooting and removal

For missing media components, run `mojilex doctor`; check the reported `ready`
value, not just whether the diagnostic command succeeded. On supported Windows
setups, `doctor --install` installs the missing FFmpeg or TGS components. TGS
requires the lossless `mojilex-rlottie-rgba` adapter; upstream `lottie2gif` is not
supported. See [Media prerequisites](media-prerequisites.md), then resume the
existing pack instead of importing it again.

For an old configuration pointing to a nonexistent local repository, inspect
`config show` and correct `repository.target` in the relevant configuration file.
Only if you intentionally want to replace the old non-secret user configuration,
back it up and use `mojilex init --force --repo MojiLex/mojilex --publish local`.
This is a configuration replacement, not a general error-recovery command.

To remove the installed tool while retaining configuration, runs, cache, and
saved credentials:

```console
mojilex uninstall --keep-data
```

Without `--keep-data`, the confirmed removal also deletes default MojiLex data,
saved credentials, and its TGS adapter. The command shows exact targets before
confirmation. Shared `uv`, Git, FFmpeg, and Visual Studio installations stay.
