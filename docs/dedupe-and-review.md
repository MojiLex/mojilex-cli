# Duplicate analysis and review

MojiLex keeps every platform-native emoji as a separate entity. Exact hashes and perceptual
similarity produce groups or review candidates; they never merge IDs, descriptions, collection
memberships, or provenance automatically.

`mojilex add` calculates one fingerprint for every media role/variant while the downloaded media
is already inside the bounded temporary run directory. `dedupe-v1` compares only fingerprints from
the same immutable profile. Binary and decoded-exact groups are derived in linear grouping passes.
Near candidates use bounded hash buckets and a local SQLite index outside the dataset repository;
the production path does not perform a complete all-pairs scan.

```console
mojilex dedupe scan --all --repo ./mojilex
mojilex dedupe scan mxe_... --repo ./mojilex
mojilex dedupe explain mxe_... mxe_... --repo ./mojilex
mojilex dedupe review mxe_... --against mxe_... --reviewer HANDLE --repo ./mojilex
```

Review fetches both current Telegram files again. Their source SHA-256 and decoded fingerprint must
still match canonical data before a decision can be recorded. The side-by-side preview is temporary
and is removed for success, failure, and cancellation. Only a human-approved relation is canonical.
An approved `not-duplicate` decision suppresses the same current pair in later scans; a media change
makes the old evidence stale instead of silently applying it to new artwork.

`build-index` derives media/entity exact groups, connected components of approved `same-artwork`
relations, and collection facets. A dedupe scan also computes one-to-one clone/subset candidates
without changing canonical collection membership. `variant-of` is deliberately excluded from the
collection overlap score.
