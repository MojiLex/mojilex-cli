# Security policy

## Reporting

Report vulnerabilities privately through **GitHub Security Advisories → Report a vulnerability**
for `MojiLex/mojilex-cli`. Do not include live tokens, private media, or personal data in a public
issue.

Maintainers aim to acknowledge ordinary security reports within 72 hours. Exposed credentials or
obviously illegal material are handled as quickly as possible. Revoke an exposed credential
immediately; removing it from the current Git tree does not remove it from history.

## Supported versions

Until the first stable release, only the latest `0.x` release is supported. Security fixes are
released under a new immutable version and tag; published artifacts are not silently replaced.

## Trust boundary

Source links, platform metadata, downloaded media, AI output, repository content, branch names,
and pull-request text are untrusted. The CLI must not give any of them control over filesystem
paths, subprocess arguments, Git refs, shell commands, network destinations, or credentials.
