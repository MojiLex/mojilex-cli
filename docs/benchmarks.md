# Qualification benchmarks

MojiLex benchmark runs are driven by strict, versioned JSON manifests. A report is successful
only when every mandatory gate passes. The report is canonicalized and includes a SHA-256 over
all report fields except `report_sha256` itself.

## Dedupe

Run:

```console
mojilex benchmark-dedupe --manifest PATH --json
```

The manifest binds the immutable `dedupe-v1` profile hash, exact dataset file hashes, split hash,
CLI commit, decoder fingerprint, runtime component versions, rights/license records, and at least
500 labeled pairs. It must contain development and holdout data and the six required strata.
Exact grouping is checked through group-membership indexes; groups are never expanded into every
possible pair. Near candidates come from the bounded production index with a limit of 20 per
item. The report exposes exact confusion matrices, recall and precision at 20, overflow counts,
pre-limit candidate counts, stratum results, and an explicit all-pairs guard.

## Model

Run manually or in a trusted workflow only:

```console
mojilex benchmark-model --provider NAME --model ID --benchmark-manifest PATH --json
```

The provider and model must exactly equal the manifest target. The CLI currently obtains Gemini
credentials only from `GEMINI_API_KEY`; credentials are neither accepted in a manifest nor
written to a report. The manifest binds every contact-sheet PNG and source-media digest, split,
prompt and request-parameter hashes, schema/taxonomy/media-pipeline/routing versions, runtime
lock/container provenance, and human adjudication tied to the exact structured response hash.
Without an immutable provider revision, the manifest must declare a dated comparison.

Release qualification requires at least 240 rights-cleared or synthetic cases, at least 30 in
each required class, development and holdout splits isolated by declared artwork groups and media
hash, three holdout runs, complete human-bound adjudication, and all quality/security gates from
MLX-SPEC-002. A small development manifest may run, but its report fails closed and exits with the
validation exit code. Unit tests use injected providers and make no network requests.

Never run the live model benchmark for an untrusted fork with provider secrets. Images, URLs,
local paths, secrets, and raw structured descriptions are not copied into reports; only bounded
metrics, counters, safe error types, and response hashes are retained.
