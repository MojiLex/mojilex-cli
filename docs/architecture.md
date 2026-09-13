# Architecture

Content ratings and warnings are retained as metadata and do not require manual
approval. Missing model qualifications are allowed; a declared qualification must
still match its registry entry and generation context. Negative explicit review
decisions and structural validation remain effective.

Canonical records may keep pending concept mappings. Canonical/active exports keep
those records; derived search rows require complete, non-empty concept mappings to
meet the existing search schema. Search text kinds project letter/punctuation to
symbol and code/other to mixed, without rewriting canonical text items.

Text references generated for light/dark PNG views are bound deterministically to
the single primary source media when building a dataset record. Those views are not
separate source files. Only supplied backgrounds with no variant ID can be bound;
alternate media, invented variants, and unavailable views remain invalid. The raw AI
cache and prompt provenance remain unchanged. Duplicate references to the same source
are collapsed without removing text or changing its recognition status.
If an older cached response has an unsupported media reference, a fresh validated
response may replace that exact rejected entry atomically with its request envelope.
The original payload and generation instant are retained in a separate archive row;
valid cache entries remain immutable.

MojiLex is split into two repositories. `mojilex-cli` contains executable Python code;
`MojiLex/mojilex` is the canonical, media-free data repository.

The import flow is:

```text
source reference
  -> SourceAdapter DTOs
  -> isolated temporary media processing
  -> deterministic full-stream rendering facets and fingerprints
  -> VisionProvider descriptions and semantic facets
  -> bounded exact/near duplicate candidate scan
  -> deterministic dataset merge
  -> full validation
  -> local change, pull request, or guarded direct push
```

Core models, identity generation, canonicalization, review hashing, and dataset operations have no
dependency on Telegram, Gemini, or GitHub. Integrations implement narrow adapters. A run store
persists safe metadata and validated AI results. A separate run-scoped directory outside the
dataset retains verified generated PNG frames so interrupted work resumes without downloading
or rendering completed items again. Raw downloaded media remains transient. Retained frames
are checked against source descriptors, checkpoint hashes, decoder/analysis identity and their
own byte hashes, and share the run's disk budget with temporary processing. A missing or
invalid retained entry falls back to downloading and verification.

Completed AI results keep their exact prompt version and routing provenance when a later
prompt version is introduced. Missing descriptions use the new prompt; cache-only compatibility
reads cannot spend the request budget or relabel an old result as a new generation.

Rendering/color/alpha facts and fingerprints come only from the local deterministic analyzer.
Descriptions, literal text, semantic tags, content types, styles, suggested uses, and uncertainties
come as one complete provider response (or from a human). Optional rule-based routing may replace
that entire semantic response once with an escalation model; fields from two AI responses are never
mixed. Exact qualification matching or a content-bound human approval is required before an AI
record can enter an official snapshot.

The canonical dataset stores compact fingerprints and approved visual relations, never candidate
lists. A rebuildable SQLite index outside the repository uses hash maps and bounded LSH buckets for
incremental scans. Exact and reviewed duplicate groups, collection aggregates, and search facets
are derived deterministically by `build-index`; native emoji entities are never merged automatically.

All files for one collection are first produced in a staging tree. They replace calculated target
paths atomically only after collection-level validation succeeds. Publication performs another full
validation against the exact Git tree being committed.
