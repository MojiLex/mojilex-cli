# Publishing changes

`--publish local` changes only calculated paths in a local checkout. It creates no commit and makes
no GitHub request. Overlapping uncommitted changes fail closed.

`--publish pr` prepares an isolated branch from a fetched base revision, applies the domain merge,
validates the complete dataset, commits only calculated paths, and opens or updates one pull request
for the run. A repeated `submit RUN_ID` fetches and reuses that run branch by creating an ordinary
fast-forward descendant with the exact validated tree; it never rewrites the branch. Contributors
publish from their fork when they lack upstream write access.

Emoji and visual-relation writes use eight hex characters of SHA-256: a two-character
directory and a six-character filename. Readers still accept the previous two-plus-two
layout. Validated writes remove the old buckets atomically and preserve record contents.

The data repository can enable **Refresh open data PRs** on changes to `main`. It uses
trusted base code to merge data-only same-repository PRs by entity ID, validates the
result, creates a normal merge commit, and explicitly starts CI. It does not merge the
PR into `main`. Fork contributors can repeat publication to refresh their branch; the
repository token cannot write another owner's fork. Actual incompatible edits to the
same record remain conflicts. Deploy the updated data validator before using the new
CLI writer, so new bucket paths are accepted by repository checks.

`--direct-push` is a separate owner operation. The CLI:

1. verifies write and branch-bypass capabilities;
2. shows the final commit SHA and every changed path, then requires confirmation unless `--yes`
   was supplied;
3. validates and pushes the exact commit to a `mojilex/...` candidate branch;
4. waits for every required check to succeed on that commit;
5. verifies that remote `main` still equals the prepared base SHA;
6. performs a non-forced fast-forward push of the already checked commit.

Unavailable, skipped, stale, or unsuccessful checks stop publication. The CLI does not create or
weaken repository rulesets.

Sensitive add flags (`--overwrite-reviewed`, `--new-identity`, and `--same-identity`) are confirmed
only after a valid in-memory plan exists. The confirmation lists the exact affected entity IDs and
changed paths before canonical files are changed. Machine output and non-interactive execution fail
closed without explicit `--yes` authorization.
