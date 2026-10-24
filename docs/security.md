# Dependency and build security

## Reviewed toolchain

The repository tests Python **3.12.15** and uv **0.13.5**. The project rejects uv
older than 0.13.4 and Python older than 3.12.15. Other Python minor versions need
their own current security patch release; the minimum version alone does not
establish their security status. See the [Python 3.12.15 security release](https://www.python.org/downloads/release/python-31215/).

uv 0.13.4 and patched certifi releases removed a compromised emSign root.
Both the package manager and applications must be updated: updating certifi
does not update uv's embedded trust store. The project requires
`certifi>=2026.10.19`. See the [uv advisory](https://github.com/astral-sh/uv/security/advisories/GHSA-hq85-5cpj-mr62)
and [certifi advisory](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2).

The bootstrap recipes download the versioned uv installer and verify its SHA-256
before execution. To install or replace an existing uv manually:

```bash
installer=$(mktemp)
curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/0.13.5/install.sh -o "$installer"
printf '%s  %s\n' '446a6087825fa73eadb045e5a2e9e2adf7df241b571228187728191d961dda1f' "$installer" | shasum -a 256 -c - && sh "$installer"
rm -f "$installer"
uv python install 3.12.15
uv sync --locked
```

On Windows, install that same reviewed uv release through the official release
artifacts and verify its published checksum. Existing virtual environments and
running processes must be refreshed/restarted; containers must be rebuilt.
Changing a manifest does not patch already deployed applications.

## Lockfile and audit workflow

Normal CI and Docker installs use `--locked` so dependency drift fails instead
of rewriting the reviewed lock. Production installs use `--no-default-groups`
to exclude development and evaluation tools. Intentional updates use `uv lock
--upgrade-package <package>` followed by review of the manifest and lock diff.
Do not delete the lockfile to bypass a resolution conflict or override upstream
constraints to force an unsupported security upgrade.

```bash
uv sync --locked --all-extras
uv run --locked pytest tests/integration_tests/security/test_dependency_security.py --no-cov
uv run --locked pytest tests/integration_tests/security/test_docker_build_context.py --include-docker --no-cov
uv run --locked cli skills validate --all
uv audit --locked --no-build --no-python-downloads --python-version 3.12.15 --python-platform linux
```

`just security-audit` audits the current platform. The separate **Dependency
security** workflow runs on pull requests, pushes, daily, and on demand. It
audits every locked extra and dependency group against Linux, macOS, and Windows
markers without installing or building dependencies. This is metadata coverage,
not proof that all three operating systems can run every optional provider.
Functional CI remains separate so its results are distinguishable from advisories.

The audit workflow explicitly runs `uv python install 3.12.15` after installing
uv. Selecting `python-version` in setup-uv does not install that interpreter;
bootstrap it before an audit that uses `--no-python-downloads`.

Native `uv audit` is experimental in the pinned version. The command exits
unsuccessfully for vulnerabilities and service/lock errors; CI preserves that
status through `pipefail` and uploads JSON evidence even on failure. There are
no ignored advisories and no automatic exemption for packages without fixes.
Missing or empty evidence after a service failure is not a clean audit. Review
the workflow log as well as its artifact.

Dependabot proposes weekly updates for uv, GitHub Actions, and Docker. Review
and test those PRs; pins need regular refresh. Actions use immutable commit
SHAs, Git requirements use reviewed commits, and Docker base/tool images use
version tags plus multi-platform digests. OS packages installed with apt remain
subject to repository updates; this is not a bit-for-bit reproducible image.

## Unresolved upstream findings (reviewed 2026-10-24)

The strict security gate is expected to remain red until these findings are
resolved or a separately reviewed, evidence-based applicability decision is
made. No exceptions are included in this change.

| Package | Remaining blocker | Operational limit pending remediation |
|---|---|---|
| Pillow 12.3.0 | Reported image-decoder advisories require Pillow 13.0.0. Core `fastembed>=0.8.0` resolves to a release requiring `pillow<13`; both [FastEmbed 0.9.0](https://github.com/qdrant/fastembed/blob/v0.9.0/pyproject.toml) and its reviewed main branch retain that cap. | Do not process untrusted images/documents in the host process. There is no verified decoder-disable mitigation here; wait for compatible upstream support or separately review replacing/removing the dependency. |
| chromadb 1.5.9 (optional extra) | The audit reports GHSA-f4j7-r4q5-qw2c, GHSA-2wm9-hf6c-p5cr, GHSA-36p7-vc44-83pf, and GHSA-xph7-9rjv-w5fr with no fixed version. See [upstream tracking](https://github.com/chroma-core/chroma/issues/7691). | Avoid the `chromadb` extra when unused. Keep any Chroma service private and treat tenant isolation/authentication assumptions as unresolved; legacy-code applicability has not been proven for every advisory. |
| diskcache 5.6.3 | GHSA-w8v5-vhqr-4h9v / CVE-2025-69872 covers pickle execution when an attacker can modify cache files. It is transitive through `py-key-value-aio[disk]`; [upstream discussion](https://github.com/grantjenks/python-diskcache/issues/357) lists no patched release. | Use a dedicated cache directory writable only by the service user. Never share writable cache volumes with untrusted users or agent sandboxes, and do not load downloaded caches. This reduces exposure; it does not resolve the advisory. |

The compatible update raises Docling to at least 2.132, SQLite checkpoints to
3.1.2, filelock to 4.2.1, and setuptools to 84.1.0. The former Requests `<2.34`
cap blocked patched Docling's `docling-core>=2.98` dependency, which requires
Requests 2.34.2. That cap is replaced with the compatible floor after verifying
the JSON decoding exception contract. Alias records in audit output can describe
the same vulnerability; report unique advisory aliases separately from packages.

The lock selects LiteLLM 1.103.8 because 1.104.2 declares `filelock<4`, which
conflicts with the fixed filelock floor. The selected release permits the patched
filelock and itself requires patched certifi; existing LLM factory regressions
cover this compatibility choice. The OpenSandbox SDK floor is 1.1.0 so executor
isolation can supply `ConnectionConfig.transport` to bound response streams.
Generated projects inherit the Python/uv floors and the pinned Hatchling backend.

## Container trust and secrets

[The Dockerfile](../deploy/Dockerfile) uses the patched certifi bundle for Python,
Requests, and curl through explicit CA environment variables. This avoids
silently reverting Python clients to an older OS bundle. Tools with independent
trust stores, including Node, still need their own vendor updates. An operator's
custom CA override must also remove compromised roots; never disable TLS checks.

[.dockerignore](../.dockerignore) excludes environment files, common credential
directories, private key files, and Streamlit secrets before the builder sees
them. Supply secrets at runtime. For dependency credentials during builds, use
BuildKit secret/SSH mounts; do not use build arguments, copied files, or tokens
embedded in dependency URLs. The regression test exports a real scratch build
containing synthetic secrets and verifies that application files, runtime skills,
and their Markdown references survive while credentials are excluded.

Report vulnerabilities through [SECURITY.md](../SECURITY.md).
