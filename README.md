# MojiLex CLI

English | [Русский](README_RU.md)

**Turn Telegram custom emoji into readable Russian and English descriptions.**
Add a pack URL, let AI describe it, browse the results, and optionally contribute
them to the [shared MojiLex dataset](https://github.com/MojiLex/mojilex).

Use the terminal menu with arrow keys. No repository cloning, IDE or internal run IDs required.

![MojiLex terminal menu](docs/assets/menu-en.svg)

[Install](#quick-start) · [First pack](#your-first-pack) ·
[Commands](#everyday-commands) · [Help](#questions-and-troubleshooting)

## Quick start

### 1. Install the prerequisites

You need **Git** to install MojiLex and work with the dataset, **uv** to manage
the program, and **GitHub CLI (gh)** to submit results. Installation needs internet access.

<details>
<summary><strong>Windows — PowerShell</strong></summary>

Install [Git](https://git-scm.com/install/windows) and [GitHub CLI](https://cli.github.com/)
using their Windows installers. Then install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close the terminal and open a new PowerShell window.

</details>

<details>
<summary><strong>macOS / Linux</strong></summary>

Install [Git for your OS](https://git-scm.com/install/) and
[GitHub CLI](https://github.com/cli/cli#installation) using their official instructions.
Then install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close the terminal and open a new window.

</details>

### 2. Install MojiLex

These commands work on all supported systems. uv installs Python 3.11 if needed:

```console
uv tool install --python 3.11 "git+https://github.com/MojiLex/mojilex-cli.git@main"
uv tool update-shell
```

Open a new terminal, then run `mojilex --version` to check the installation.
This installs the current main-branch build. MojiLex is an early version;
[report problems here](https://github.com/MojiLex/mojilex-cli/issues).

### 3. Set up once

```console
mojilex --ui-language en init
```

The wizard asks for settings:

- Keep `MojiLex/mojilex` as the target, `gemini` as the provider and `ru,en` as the description languages.
- Enter the exact [Gemini model ID](https://ai.google.dev/gemini-api/docs/models) available to your account. No model is selected automatically.
- You can keep the publication default. The **menu always analyzes locally** and offers sending separately; advanced commands also use this setting.
- If asked for a Git author, enter the name and email to appear on contributions, or skip until publishing.

Already configured? Use `mojilex settings` instead of repeating setup.

### 4. Add access keys and check the tools

| Access | Where to get it | Used for |
|---|---|---|
| Telegram bot token | [Create a bot with BotFather](https://core.telegram.org/bots/tutorial#obtain-your-bot-token) | Download public emoji packs |
| Gemini API key | [Google AI Studio instructions](https://ai.google.dev/gemini-api/docs/api-key) | Generate descriptions |
| GitHub account | `gh auth login` | Submit results; login is optional for local analysis |

To avoid re-entering keys, save them through hidden input, then check the tools:

```console
mojilex config set-credentials
mojilex doctor
```

Keys go into the operating system's credential store, not Git. Saving is optional:
analysis asks for missing keys for that invocation. `init` never asks for keys.

Follow any missing-component instructions from `doctor`. On Windows it offers
WebM/TGS tool installation; TGS can require administrator approval and a sizeable
Visual Studio Build Tools installation. For Linux/macOS, follow [media prerequisites](docs/media-prerequisites.md).

## Your first pack

```console
mojilex
```

1. Choose **Add packs from a URL or file** and paste a public link, such as `https://t.me/addemoji/NewsEmoji`, or a text file path.
2. Read the analysis plan and cost confirmation; allow requests if you agree.
3. After completion, open **My packs → your pack → Browse descriptions**.
4. Search, switch RU/EN, or open **All fields and English text**.
   **↑↓ / PgUp / PgDn** scroll details; **Enter / Esc** return.
5. For another view, choose **Open browser gallery**.

The menu saves locally and does not submit automatically. Viewing descriptions
or the gallery makes no AI requests. Missing local previews appear as placeholders.

Example details, rendered by the program using sample data:

![Example Russian and English descriptions](docs/assets/details-en.svg)

## Multiple packs from a file

Paste a path such as `C:\Users\Me\Desktop\packs.txt` into the menu. Use one URL
per line in a UTF-8 file. Blank lines, `#` comments and duplicate URLs are ignored.
All URLs share one import and analysis run, with one AI cost approval rather than
one per URL. The configured AI request limit still applies to the entire run.
AI cost and publication confirmations default to **Yes**; Enter accepts it.

Choose **Sync all new completed packs to GitHub**, or run:

```console
mojilex sync --local
mojilex sync
```

The first command validates without uploading. The second submits all saved,
completed new packs for the configured repository in one PR after one confirmation.
Packs already on the base branch are skipped and published descriptions are preserved.
A pending PR does not add packs to the base branch until it is merged.

You can also use `mojilex import "C:\path\packs.txt"` or
`mojilex import --from-file "C:\path\packs.txt"`, followed by
`mojilex describe RUN_ID` using the combined import's printed run ID.

## Everyday commands

Replace `NewsEmoji` with your pack's name. These address **your locally saved packs**,
not every pack on Telegram or GitHub.

| What you want to do | Command |
|---|---|
| Open the menu | `mojilex` |
| List your packs and progress | `mojilex list` |
| Read descriptions | `mojilex show NewsEmoji` |
| Open the browser gallery | `mojilex gallery NewsEmoji` |
| Print all fields | `mojilex show NewsEmoji --all` |
| Continue unfinished work | `mojilex resume NewsEmoji` |
| Check a draft without uploading | `mojilex publish NewsEmoji --local` |
| Submit results through a GitHub PR | `mojilex publish NewsEmoji` |
| Change model, limits or parallelism | `mojilex settings` |
| Diagnose tools and access | `mojilex doctor` |

For a new pack through commands, run `mojilex import "PACK_URL"`, then
`mojilex describe PACK_NAME`. Replace the placeholders with your link and pack
name. This keeps analysis separate from publication.

## Send results to GitHub

Sign in once, then submit the completed pack:

```console
gh auth login
mojilex publish NewsEmoji
```

MojiLex validates descriptions, prepares a commit, uploads a branch and creates
or reuses a **pull request (PR)**: a proposal to include your data. The current
stage and elapsed time stay visible. Publishing does not repeat analysis.

Open the returned PR link and check **Checks**. Green checks mean validation passed;
the pack enters the main dataset only when the PR is **merged**. A successful
upload or saved publication attempt does not prove that it has been merged.

Content warnings stay in the records. They do not require manual approval or
block submission; invalid data still fails validation.

## Questions and troubleshooting

| Question | What to do or expect |
|---|---|
| Does analysis cost money? | It may, depending on your Gemini account and model. The default limit is **100 requests per run**, including retries. Resume retains the consumed count. Review the plan before approving unknown pricing. |
| Can it run faster? | Adjust AI parallelism in **Settings**. For an existing run, use `mojilex resume NewsEmoji --ai-concurrency 4`. Speed depends on provider quotas; this does not increase the request budget. |
| The connection dropped or I stopped it | Use `mojilex resume NewsEmoji`. Completed work is reused; missing/corrupt media may need downloading again. Budget or access errors need resolving first. |
| Why is the count not moving? | A batch may be waiting for a response or validation. Watch the stage, retries and elapsed time. Time spent is not completed work. |
| Where are English descriptions? | Switch languages in `show` or open item details. Interface language is separate, under **Settings**. |
| The command is not found | Run `uv tool update-shell`, then open a new terminal. |
| Where is my result? | Use `mojilex list` and `mojilex show NewsEmoji`. Local drafts, GitHub contributions and release snapshots are different results. |
| Another error persists | Run `mojilex doctor`; see the [error reference](docs/error-reference.md). Report the command and error code, without keys or sensitive logs. |

## Update or remove

For an installation made using the commands above:

```console
uv tool upgrade mojilex-cli
```

Restart `mojilex` afterwards. Package updates preserve settings and saved runs.
`mojilex uninstall` shows a removal plan and asks for confirmation.
`mojilex uninstall --keep-data` keeps settings, runs, cache and stored keys.

## More documentation

- [Advanced usage](docs/advanced-usage.md): options, retries, configuration, snapshots and compatibility commands.
- [Development](docs/development.md): source checkout, tests and packaging.
- [Media prerequisites](docs/media-prerequisites.md) · [Publishing rules](docs/publishing.md) · [Security model](docs/security-model.md).
- [Contributing](CONTRIBUTING.md) · [Report an issue](https://github.com/MojiLex/mojilex-cli/issues).

Generated frames may be kept **locally** for resume and previews. Original media,
frames and API keys are never submitted to the dataset repository.
CLI code is [MIT-licensed](LICENSE); data licensing is explained in the
[data repository](https://github.com/MojiLex/mojilex).
