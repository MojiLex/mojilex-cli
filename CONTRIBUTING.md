# Contributing to MojiLex CLI

Thank you for helping make emoji metadata more useful and reproducible.

## Development workflow

1. Create a fork and a focused branch.
2. Install the `dev` extra in a Python 3.11+ virtual environment.
3. Add tests for behavior and security boundaries that you change.
4. Run `pytest`, `ruff check`, `ruff format --check`, `mypy`, and `python -m build`.
5. Open a pull request using the repository template.

Do not commit API tokens, Telegram `file_id` values, raw download URLs, original emoji media,
frames, contact sheets, real user data, or fixtures copied from Telegram. Media fixtures must be
synthetic and carry an explicit compatible license.

Code and documentation contributions are licensed under MIT. Metadata, descriptions, and tags
belong in `MojiLex/mojilex`, whose contribution terms license those contributions under CC0-1.0.
By submitting a contribution, you confirm that you have the right to do so under these terms.

Security vulnerabilities and sensitive takedown material must not be filed as public issues.
Follow [SECURITY.md](SECURITY.md).
