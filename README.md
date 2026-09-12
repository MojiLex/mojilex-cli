# MojiLex CLI

`mojilex` imports public Telegram custom-emoji sets, prepares temporary media for
deterministic analysis, creates Russian and English semantic metadata, finds duplicate
candidates, validates the
[MojiLex dataset](https://github.com/MojiLex/mojilex), and publishes a local change or a
GitHub pull request.

The CLI never stores original emoji media, rendered frames, contact sheets, API keys, or
Telegram download URLs in Git or its persistent cache. Telegram media remains owned by its
respective rights holders.

## Status

This repository contains the `0.2.0` MVP. It adds bounded, offline, read-only access to an
explicitly selected `distribution-v1` snapshot. The data format has its own independently
versioned JSON Schema (`1.0.0`). Python 3.11 or newer is required.

## Install

```console
pipx install mojilex-cli
```

or:

```console
uv tool install mojilex-cli
```

Verify the installation and optional media backends:

```console
mojilex --version
mojilex doctor
```

Pillow-based WebP processing is included. FFmpeg/ffprobe and the MojiLex lossless rlottie RGBA
adapter are external prerequisites for WebM and TGS. The lossy upstream `lottie2gif` utility is
not supported. See
[Media prerequisites](docs/media-prerequisites.md).

## Quick start

1. Clone the data repository and initialize a non-secret local configuration:

   ```console
   git clone https://github.com/MojiLex/mojilex.git
   mojilex init --repo ./mojilex --provider gemini --model "YOUR_EXACT_GEMINI_MODEL_ID"
   ```

   The model ID is deliberately explicit so the selected provider target is visible in the
   configuration and provenance. `init` writes only non-secret settings, runs the same real media
   fixture probes as `doctor`, and reports credential, Git identity, and GitHub access readiness.
   In a terminal, `mojilex init` opens a setup wizard for the target, provider, exact model,
   languages and publication mode. If Git identity is missing, the wizard offers to save it
   only in the MojiLex configuration; it never changes global Git settings. Use
   `--non-interactive` with explicit options for scripts. JSON, quiet and piped-input modes
   never prompt; a missing model is an error. Repeat `--lang` to select languages (ru/en required).

2. Provide credentials through environment variables or hidden interactive input. Never put
   them in `.mojilex.toml` or a command-line argument:

   ```text
   TELEGRAM_BOT_TOKEN
   GEMINI_API_KEY
   GH_TOKEN or GITHUB_TOKEN (only for GitHub publication)
   ```

3. Preview an import without AI calls or persistent changes:

   ```console
   mojilex add https://t.me/addemoji/PackName --repo ./mojilex --dry-run
   ```

4. Import and describe locally, or create a contributor pull request:

   ```console
   mojilex add https://t.me/addemoji/PackName --repo ./mojilex --publish local
   mojilex add https://t.me/addemoji/PackName --repo MojiLex/mojilex --publish pr
   ```

Direct pushes are deliberately separate and require `--direct-push`, write access, successful
checks for the exact candidate commit, an unchanged base branch, ruleset bypass permission, and a
late confirmation that shows the final commit SHA and exact diff paths. Repeated submission of one
run updates its existing PR branch with an ordinary non-force descendant commit.

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
mojilex cache info
mojilex cache prune
mojilex snapshot verify PATH
mojilex search QUERY --snapshot PATH
mojilex get EMOJI_ID --snapshot PATH
mojilex get-collection COLLECTION_ID --snapshot PATH
mojilex resolve --platform NAME --namespace NAME --scope ID --native-id ID --snapshot PATH
mojilex similar EMOJI_ID --snapshot PATH
```

Use `mojilex COMMAND --help` for exact options. With `--json`, stdout contains exactly one
machine-readable envelope; progress and diagnostics go to stderr.

Every data-reading command requires an explicit local `--snapshot PATH`. The pre-enforcement
MVP snapshot is integrity-checked but unsigned, so reads fail closed unless diagnostic use is
explicitly acknowledged with `--allow-unverified`; this never makes the release trusted and
`runtime_trust.safe_eligible` remains false. `search` defaults to the safe `agent` view, while
`--view search --allow-unverified` is the explicit diagnostic projection. The read path is
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
