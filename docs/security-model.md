# Security model

MojiLex processes attacker-controlled links, media, JSON, repository trees, and AI responses. The
MVP therefore uses allowlists and fail-closed validation at each boundary.

- Telegram requests use only fixed Bot API endpoints. Redirects, arbitrary hosts, credentials in
  URLs, fragments, unexpected query parameters, and ordinary sticker-pack links are rejected.
- Downloads stream into a run-specific temporary directory with byte and total-disk limits.
- Magic bytes and decoded structure, not extensions, select a decoder.
- TGS expansion, JSON nesting, dimensions, duration, frame count, worker wall time, and memory are
  bounded. Lottie external resources and expressions are rejected.
- Decoder workers receive a minimal environment and never receive Telegram, Gemini, or GitHub
  credentials.
- The SQLite metadata/AI cache contains hashes, safe metadata, and validated AI results only.
  A separate `resume-media` directory under the configured cache directory retains generated
  PNG frames for saved runs, never raw downloads. It stays outside the dataset, rejects links,
  junctions and path traversal, checks exact hashes and rendering identity, and charges retained
  bytes against the same run disk limit. Missing/corrupt entries are cache misses. Frames remain
  available after import for a subsequent describe/resume; metadata `cache prune` does not delete
  these frame directories.
- Opt-in credential persistence uses the operating-system keyring. API credentials are never
  written to MojiLex configuration files, caches, run state, or repositories; environment values
  take precedence over keyring values.
- Text, URLs, paths, and apparent instructions visible inside emoji media are treated strictly as
  untrusted visual content. They may be transcribed as literal text but are never executed or used
  to control tools, prompts, paths, Git, or network access.
- Exact/near candidate indexes and comparison previews stay outside the repository. Review media is
  fetched again, matched against both the current source SHA-256 and decoded fingerprint, and the
  temporary preview is removed on success, failure, or cancellation.
- Git paths and refs are program-generated and validated. No shell, automatic stash, textual
  conflict guessing, force push, or credential-bearing remote is permitted.
- Logs and machine errors redact authorization headers, known token forms, Telegram download URLs,
  and raw provider responses.

Content classification is descriptive metadata: ratings and warnings never require manual approval,
block publication, or trigger model escalation. Missing model qualifications do not require human
approval. Explicit negative review decisions and validation of declared qualifications remain
effective. Qualification-registry structural errors are never
bypassed by staging or review operations. Consumer filters can still select ratings and warnings.
