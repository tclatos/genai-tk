# Security policy

Security fixes target the current `main` branch. Older snapshots are not maintained
as separate security release lines. Use the reviewed toolchain and dependency
lockfile described in [the security guide](docs/security.md).

## Reporting a vulnerability

Use GitHub's [private vulnerability reporting form](https://github.com/tclatos/genai-tk/security/advisories/new)
when it is available. Include the affected commit, reproduction steps, impact,
and any proposed mitigation. Do not include live credentials or personal data.

Private reporting was disabled when checked on 2026-10-24. A repository
administrator must enable **Settings → Code security → Private vulnerability
reporting** before the private form can receive reports. Until then, open an
issue requesting a private reporting channel without publishing exploit details.

## Dependency findings

The dependency security workflow fails on advisories and audit errors, including
findings with no available fix. Known unresolved findings and mitigation limits
are recorded in [docs/security.md](docs/security.md); their presence is not a
claim that the affected deployment is safe.
