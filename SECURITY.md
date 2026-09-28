# Security policy

## Supported versions

Only the latest tagged release receives security fixes.

## Reporting

Do not open a public issue for a suspected vulnerability. Contact the repository
maintainers through the private security-reporting channel configured on the host.

## Security invariants

- No panic or Rust-owned allocation crosses the C ABI.
- All ABI inputs are null-checked and interpreted as UTF-8.
- Native strings are released only by `de_string_free`.
- Model downloads occur only after an explicit `pull` command.
- Inference contains no cloud or remote fallback.
- The HTTP server binds only to loopback unless the operator explicitly overrides it.
- Model manifests allowlist the only release-supported model IDs.
