---
name: dependency-security
description: Review Python dependency advisories, uv and Python security updates, lockfiles, CI action pins, Docker trust stores, and build-secret handling. Use when updating dependencies or responding to security audit failures.
---

# Dependency and CI Security

Read [docs/security.md](../../../docs/security.md), [SECURITY.md](../../../SECURITY.md),
and [pyproject.toml](../../../pyproject.toml) before changing the security policy.

1. Verify the advisory and patched versions against upstream primary sources.
   Inspect the actual locked dependency path, extras, Python/platform markers,
   and affected code. Distinguish alias records from unique vulnerabilities.
2. Use the reviewed uv 0.13.5 / Python 3.12.15 toolchain. Refresh pins deliberately
   when a newer security release is reviewed; update bootstrap hashes, CI,
   Docker tags/digests, docs, and the lock together.
3. Make targeted `uv lock --upgrade-package <package>` changes. Preserve library
   source rules in `uv-sources-sibling-checkouts`; keep root-only references in
   dependency groups. Do not delete the lock or override upstream constraints
   to claim an unsupported upgrade is fixed.
4. Run `uv sync --locked --all-extras`, relevant regression tests, and
   `uv run --locked cli skills validate --all`. Production uses
   `uv sync --locked --no-default-groups` with only required extras.
5. Run `uv audit --locked --no-build --python-version 3.12.15 --python-platform linux`
   and repeat for macOS/Windows markers. Native audit is experimental; keep uv
   pinned and treat network/service/parser failures as failures, not clean scans.
   In CI, run `uv python install 3.12.15` before auditing with
   `--no-python-downloads`; setup-uv's `python-version` only selects an interpreter.
6. Keep findings with no fix visible. Do not add blanket ignores or
   `--ignore-until-fixed`. Any proposed exception needs specific affected-code
   evidence, an owner, an expiry, and a regression check in a separate review.
   Update the unresolved findings table instead of concealing a blocker.
   For integrations with no patched release, remove unused dependency extras or
   retire the integration with explicit configuration errors and migration docs;
   never silently reinterpret existing persistent data. The Chroma extra and
   DiskCache backend are retired. Keep Pillow 13 paired with compatible FastEmbed,
   Streamlit, and Docling releases and run the real compatibility regressions in
   [tests/integration_tests/security/](../../../tests/integration_tests/security/).
7. Verify image digests and action SHAs at upstream release refs. Keep credentials
   out of build contexts and run the Docker context regression with
   `uv run --locked pytest tests/integration_tests/security/test_docker_build_context.py --include-docker --no-cov`.

An updated certifi bundle does not patch uv's embedded certificates or an
operator's custom CA file. Refresh existing environments and container images,
then restart processes. Do not disable certificate verification.
