# Error and exit-code reference

| Exit | Category | Stable codes |
| ---: | --- | --- |
| 0 | success/noop/dry run | — |
| 1 | internal | `INTERNAL_ERROR` |
| 2 | arguments/configuration | `CONFIG_INVALID`, `CONFIG_MISSING` |
| 3 | system dependency | `SYSTEM_DEPENDENCY_MISSING` |
| 4 | credentials/permissions | `CREDENTIAL_MISSING`, `AUTH_FAILED`, `PERMISSION_DENIED` |
| 5 | source/stale identity | `SOURCE_UNSUPPORTED`, `SOURCE_NOT_FOUND`, `SOURCE_CHANGED_DURING_RUN`, `IDENTITY_CONFLICT` |
| 6 | network/rate limit | `NETWORK_ERROR`, `RATE_LIMITED` |
| 7 | media | `MEDIA_INVALID`, `MEDIA_LIMIT_EXCEEDED`, `MEDIA_RENDER_FAILED` |
| 8 | AI | `AI_REQUEST_FAILED`, `AI_OUTPUT_INVALID` |
| 9 | budget | `BUDGET_EXCEEDED`, `UNKNOWN_COST` |
| 10 | dataset validation | `VALIDATION_FAILED` |
| 11 | Git/conflict | `DIRTY_WORKTREE`, `GIT_CONFLICT` |
| 12 | GitHub publication | `GITHUB_PUBLISH_FAILED`, `REQUIRED_CHECK_FAILED` |
| 13 | partial batch | `PARTIAL_FAILURE` |
| 130 | interruption | `INTERRUPTED` |

With `--json`, stdout is one envelope. Each error contains `code`, `message`, `retryable`, and
`hint`; optional details are sanitized and never contain secrets or raw download URLs.
