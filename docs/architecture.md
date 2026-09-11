# Architecture

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
persists only safe metadata and validated AI results so interrupted work can resume without
retaining media or paying for a completed description twice.

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
