# Sandbox Browser: Enedis Bot-Detection Investigation

Status document for the Enedis `/indisponible` redirect issue in the Docker/OpenSandbox browser path.

## Executive Summary

The generic sandbox browser stack is in a much better state than at the start of the investigation:

- cookie and storage restoration now apply to the active context
- browser tools expose better URL/title/state diagnostics
- reconnect logic no longer tears down sessions during normal navigation churn
- the stale baked-in User-Agent override is cleared
- locale and timezone are now propagated consistently from browser config into the sandbox browser environment

Those fixes improved correctness and made the sandbox browser much closer to a real browser, but they did **not** stop Enedis from redirecting the sandbox/CDP path to `/indisponible`.

The strongest current signal is architectural:

1. A **local Playwright launch on the host** reaches the real Enedis login page and FriendlyCaptcha.
2. The **current sandbox path** (`connect_over_cdp()` to the pre-launched sandbox browser) still goes to `/indisponible`.
3. A **fresh browser launch inside the same sandbox image**, run directly in Docker outside the OpenSandbox control plane, can launch Chromium successfully on a blank page, but the real Enedis probe has not yet completed cleanly due command/session instability during the long run.

Current best hypothesis: the remaining blocker is in the **sandbox browser/runtime architecture**, most likely the pre-launched browser + CDP attach path, or a closely related sandbox-specific runtime characteristic.

## Current Architecture

```
┌─────────────┐    HTTP     ┌──────────────────────┐     CDP      ┌────────────────┐
│  Agent CLI  │ ──────────→ │  OpenSandbox Server  │ ───────────→ │ Chromium 135   │
│   (host)    │             │  (Docker sandbox)    │              │ pre-launched   │
└─────────────┘             └──────────────────────┘              └────────────────┘
                                   │
                              VNC / canvas
```

The important historical contrast is:

- **old working Enedis flow**: Playwright launched and owned a fresh browser directly
- **current failing flow**: Playwright attaches over CDP to a browser that the sandbox image has already started

That distinction is now the center of the investigation.

## Generic Changes Already Landed

These are generic infrastructure fixes, not Enedis-specific hacks:

### Browser/session fixes

- `genai_tk/tools/sandbox_browser/session.py`
  - restore cookies/localStorage into the active context
  - richer logging for navigation, XHR/fetch, failed requests, and frame navigations

- `genai_tk/tools/sandbox_browser/tools.py`
  - `browser_read_page` now returns URL and title along with text
  - `browser_wait` supports `load_state`
  - `browser_navigate` supports `wait_until`
  - `_ensure_connected()` now tolerates transient evaluate/navigation errors instead of reconnecting unnecessarily

### Sandbox browser environment fixes

- `genai_tk/agents/sandbox/aio_backend.py`
  - clear baked-in `BROWSER_USER_AGENT` unless explicitly configured
  - propagate locale/timezone through `TZ`, `LANG`, `LC_ALL`, `LANGUAGE`
  - append `--lang=...` and `--time-zone-for-testing=...` only if not already present
  - preserve generic webdriver suppression via `--disable-blink-features=AutomationControlled`

- `genai_tk/tools/sandbox_browser/models.py`
  - add `timezone_id` to browser config

- `config/sandbox.yaml`
  - default timezone set to `Europe/Paris`

- `docs/browser_control.md`
  - document the sandbox browser timezone/locale behavior

### Enedis skill updates

- `skills/custom/enedis-portal/SKILL.md`
  - keep Enedis-specific waits and diagnostics in the skill
  - prefer `domcontentloaded` + explicit idle waits
  - capture URL/title/body diagnostics instead of assuming login success

### Tests

Relevant tests were added/updated under:

- `tests/unit_tests/tools/sandbox_browser/`
- `tests/unit_tests/agents/langchain/`

The browser/session/backend tests for these fixes passed during the investigation.

## Confirmed Findings

### 1. Reconnect churn was real, but not the root cause

Earlier runs showed `_ensure_connected()` reconnecting during critical Enedis navigation. That problem is fixed. The session now survives the `auth/XUI` transition properly.

Result: the redirect to `/indisponible` still happens.

### 2. The stale User-Agent override was real, but not the root cause

The sandbox image originally forced a desktop-Mac Chrome UA that no longer matched Chromium's real version and client hints.

After clearing that override:

- `navigator.userAgent` and `userAgentData.brands` became consistent
- the browser fingerprint became materially more believable

Result: Enedis still redirects to `/indisponible`.

### 3. Locale/timezone mismatch was real, but not the root cause

The sandbox browser previously exposed the wrong locale/timezone. After the environment propagation fixes, direct probes showed:

- `navigator.language = fr-FR`
- `navigator.languages = [\"fr-FR\"]`
- locale resolved to French
- timezone resolved to `Europe/Paris`

Result: Enedis still redirects to `/indisponible`.

### 4. Public IP mismatch is probably not the main explanation

Host and sandbox public IP checks matched during the investigation, so simple Docker NAT identity does not appear to explain the behavior by itself.

### 5. WebGL remains software-rendered, but that alone is unlikely to explain the block

The sandbox renderer remains SwiftShader-based. Several launch-flag variants were tried, but they did not materially change the renderer or move the investigation forward.

This is still a possible weak signal, but it no longer looks like the best lead because:

- the host-local Playwright probe still reached the real Enedis login flow
- that makes “any headless/SwiftShader browser is blocked” unlikely

### 6. Local Playwright works on Enedis

This is the most important result so far.

A local Playwright probe on the host reached the real Enedis auth flow:

- final URL on the Enedis auth/XUI path
- Enedis title
- login email field present
- FriendlyCaptcha text present

This means Enedis is **not** simply blocking all automation, all Playwright, or all headless Chromium on this machine.

### 7. The current sandbox/CDP path still fails

After all generic fixes, the sandbox browser flow still follows the same broad pattern:

- initial page load
- auth redirect
- eventual `/indisponible`
- no login form fields

### 8. OpenSandbox/browser docs align with the current suspicion

The browser guide confirms that the normal sandbox integration pattern is to attach automation over CDP to the sandbox browser, and it explicitly calls out canvas/CDP as a path that may disconnect and require heartbeat management.

That matches two things seen in practice:

- earlier reconnect/liveness instability in our own tooling
- the broader fragility encountered while trying to run longer in-sandbox browser experiments through the sandbox command APIs

The same guide also documents alternative interaction styles such as VNC/GUI actions, which suggests the platform itself recognizes that CDP/browser-control is not equivalent to visual/manual interaction for all sites.

## Latest Fresh-Browser Experiments

### Attempt A: fresh browser inside OpenSandbox-managed sandbox

Goal: install Playwright in the sandbox container and launch `/usr/bin/chromium-browser` directly from inside the container, bypassing the pre-launched browser/CDP path.

What happened:

- Playwright installation inside the sandbox succeeded
- short direct command execution worked
- longer browser-probe commands repeatedly failed due OpenSandbox/execd command-channel instability:
  - `RemoteProtocolError`
  - incomplete chunked read
  - server disconnected without sending a response
  - sandbox disappeared from the OpenSandbox server after failure

Interpretation:

- this did **not** produce a clean Enedis result
- however, it did show that the command/control plane is unstable enough to obstruct the experiment itself

### Attempt B: same sandbox image, launched directly under Docker

To remove OpenSandbox from the equation, the exact sandbox image was launched directly with Docker and driven using `docker exec`.

Results:

1. Fresh Playwright + Chromium launch on `about:blank` worked successfully.
2. Locale/timezone in that direct-container launch looked correct.
3. The Enedis probe itself did not finish cleanly within the interactive run that was attempted, so the final Enedis outcome in this direct-image path is still unresolved.

Interpretation:

- launching a second/fresh Chromium in the sandbox image is possible
- the image itself is not fundamentally incapable of running Playwright
- the unresolved part is whether the Enedis site behaves differently in that direct fresh-image path than in the current CDP-attached path

## What Is No Longer the Best Hypothesis

These ideas are now lower priority:

- cookie banner interaction as the trigger
- `navigator.webdriver` alone
- stale/mismatched UA alone
- locale/timezone mismatch alone
- reconnect churn alone
- “all headless or all Playwright is blocked”

## Current Best Hypotheses

### H1: Pre-launched browser + CDP attach is the remaining detection vector

This is the strongest current hypothesis.

Reasons:

- it is the main architectural difference from the old working flow
- local Playwright with a fresh browser works
- current sandbox flow still uses CDP attach to a pre-launched browser
- the browser guide explicitly frames CDP/canvas as a more fragile path

Possible reasons this matters:

- debugger/CDP observability
- state inherited from sandbox browser startup wrapper
- differences between the default pre-launched context and a fresh Playwright-owned context
- sandbox browser command-line / wrapper behavior that differs from a directly launched browser

### H2: A deeper sandbox-runtime characteristic remains, but only in the managed sandbox path

If a clean direct-container fresh-browser Enedis probe succeeds, then the remaining blocker is very likely specific to:

- OpenSandbox-managed runtime behavior
- the pre-launched browser wrapper
- the CDP/browser-control path

### H3: Network or TLS-level signals still matter, but need stronger evidence

This is still possible, but it dropped in priority after the host-local Playwright success.

It should only move back up if a direct fresh-browser run inside the sandbox image still fails even when CDP is removed from the picture.

## Recommended Next Steps

### Highest-priority next experiment

Complete a **clean fresh-browser Enedis probe inside the sandbox image** and capture:

- final URL
- title
- login/email field presence
- body text
- UA/client-hints basics
- WebGL basics

Do this in the most stable execution path available, preferably:

1. direct Docker container using the same sandbox image, or
2. a more reliable detached/background execution pattern that does not depend on long streaming responses from the OpenSandbox command API

This single result is still the best discriminator.

### Decision tree after that result

#### If fresh browser in the sandbox image reaches login/FriendlyCaptcha

Then the next generic infrastructure step should be:

- move away from relying exclusively on the pre-launched browser + CDP attach model for sensitive sites

Likely options:

- support a generic “fresh browser launched inside sandbox” mode
- keep Enedis-specific behavior in `SKILL.md`, but make the launch/control primitive generic

#### If fresh browser in the sandbox image still goes to `/indisponible`

Then investigate deeper sandbox/runtime characteristics:

- network/TLS fingerprinting
- proxy/interception differences
- sandbox image startup wrapper side effects
- other managed-runtime signals independent of CDP

### Supporting follow-up experiments

If time allows, gather these next:

1. Compare `bot.sannysoft.com` and `tls.browserleaks.com/json` between:
   - host-local Playwright fresh launch
   - sandbox current CDP-attached path
   - sandbox-image direct Docker fresh launch

2. Compare the exact Chromium process args between:
   - pre-launched sandbox browser
   - fresh browser launched manually in the same image

3. Check whether the OpenSandbox-managed browser path injects extra wrapper state, environment variables, or startup files not present in a direct browser launch.

## Probe Results (2026-03-21)

Both CDP-attach and fresh-browser modes were tested against a live AIO container
(`ghcr.io/agent-infra/sandbox:latest`) using `scripts/enedis_probe.py`.

| Signal | CDP Mode | Fresh Mode |
|---|---|---|
| Verdict | BLOCKED → `/indisponible` | BLOCKED → `/indisponible` |
| User-Agent | Chrome/140 (Mac) | Chrome/140 (Mac) |
| `navigator.languages` | `["en-US"]` ❌ | `["fr-FR"]` ✅ |
| Timezone | `Asia/Singapore` ❌ | `Europe/Paris` ✅ |
| WebGL renderer | SwiftShader | SwiftShader |

### Interpretation

1. **Fresh mode fixed locale/timezone** but Enedis still blocks — locale/timezone
   is not the main trigger.

2. **CDP mode still has wrong locale/timezone** because the container was started
   via `docker run`, not via OpenSandbox (which applies `aio_backend.py` env vars).
   This confirms the `BROWSER_EXTRA_ARGS` injection only works through the managed path.

3. **SwiftShader is now the strongest remaining hypothesis.** Both modes use
   SwiftShader (software GPU renderer). A local Playwright host launch uses the
   host GPU and reaches Enedis. The SwiftShader fingerprint + Mac UA on Linux is
   likely a strong bot signal.

4. **Platform mismatch**: The sandbox bakes in a Mac UA (`Mac OS X 10_15_7`) but
   runs on Linux. JavaScript `navigator.platform` and `userAgentData.platform`
   may report Linux, contradicting the Mac UA string. Combined with SwiftShader,
   this creates a highly anomalous browser fingerprint.

### Updated Hypothesis Ranking

1. **H1 (NEW STRONGEST): SwiftShader + platform mismatch** — software GPU
   renderer + Mac-on-Linux signals are the primary detection vector
2. **H2: CDP/automation signals** — demoted; fresh mode eliminates CDP attach
   but still gets blocked
3. **H3: Network/TLS-level signals** — still possible but lower priority

### Recommended Next Steps

1. **Verify platform mismatch**: Run `navigator.platform` and
   `navigator.userAgentData.getHighEntropyValues()` in both modes
2. **Test with Linux UA**: Override the sandbox's baked-in Mac UA to a matching
   Linux Chrome UA string and re-probe
3. **Test with GPU passthrough**: If available, mount the host GPU into the
   container to get a real renderer instead of SwiftShader
4. **Compare with `bot.sannysoft.com`**: Run the bot-detection test suite in
   both the sandbox and host-local Playwright to identify the exact signals

## Bottom Line

The investigation has already removed several generic browser inconsistencies and improved the sandbox browser stack substantially.

The remaining Enedis failure now looks much less like a simple “bad browser fingerprint” problem and much more like an **architectural difference between a fresh Playwright-owned browser and the current sandbox-managed pre-launched/CDP-attached browser path**.

The most valuable next step is still to finish the clean fresh-browser-in-sandbox comparison and use that result to choose the next generic infrastructure change.

## Implementation: Fresh-Browser Mode (AIO Container)

Based on the investigation, a `launch_mode: fresh` option was added to the sandbox browser
infrastructure.  This approach works entirely within the AIO container and requires **no
additional software installation inside the container** — it reuses the container's
pre-installed `/usr/bin/chromium-browser`.

### Key Design Constraint

Installing Playwright + Chromium inside the sandbox container (~220 MB download) is
impractical for production.  The solution must work with what the AIO image already ships.

### 1. Generic `launch_mode` for `SandboxBrowserSession`

Added `launch_mode` to `SandboxBrowserConfig` (`models.py`) and `sandbox.yaml`:

- `"cdp"` (default) — attach to pre-launched sandbox Chromium via CDP (current behavior)
- `"fresh"` — kill the pre-launched browser, launch a fresh Chromium with anti-detection
  flags using the *same binary already in the container*, then HOST Playwright reconnects
  via CDP

The fresh mode (`session.py` `_connect_fresh()`) implements:
1. Kill the pre-launched Chromium via `pkill` (using `AsyncSandbox.shell.exec_command`)
2. Launch `/usr/bin/chromium-browser` with `--disable-blink-features=AutomationControlled`,
   `--lang`, `--time-zone-for-testing`, `--remote-debugging-port=9222`, `--headless=new`
3. Wait for the new browser's CDP endpoint at `{sandbox_url}/cdp/json/version`
4. Connect HOST Playwright over CDP and create a fresh, owned context with locale/timezone

Configure in `config/sandbox.yaml`:
```yaml
sandbox_browser:
  launch_mode: "fresh"  # or "cdp" (default)
```

### 2. Host-Side Enedis Probe (`scripts/enedis_probe.py`)

A diagnostic script that runs **from the host** against a live AIO container.  Tests both
CDP-attach and fresh-browser modes, comparing whether each reaches the Enedis login form
or gets redirected to `/indisponible`.

```bash
# Start a sandbox container
docker run -d --name enedis-probe -p 8080:8080 ghcr.io/agent-infra/sandbox:latest
sleep 10

# Run the probe from the host (requires: playwright, agent-sandbox, httpx)
python scripts/enedis_probe.py                       # default: localhost:8080
python scripts/enedis_probe.py http://localhost:8080  # explicit

# Cleanup
docker rm -f enedis-probe
```

The probe outputs per-mode verdicts and full JSON results, plus a summary of which
mode (if any) successfully reaches the login form.

### 3. Simplified SKILL.md

The Enedis SKILL.md uses `launch_mode: fresh` + standard browser tools for authentication.
No standalone login script is needed — the fresh mode handles bot-detection avoidance
transparently at the session layer, and the agent performs login steps via browser tools.

### Open Questions

1. **CDP proxy routing**: After killing the pre-launched Chromium and launching fresh on
   port 9222, does the sandbox nginx proxy correctly route `/cdp/` to the new process?
   The probe script is the best way to validate this end-to-end.

2. **Pre-launched browser's original port**: If the sandbox starts Chromium on a port
   other than 9222, the fresh launch on 9222 may not be reachable through `/cdp/`.
   May need to discover the original port and reuse it.

### Resolution (2026-03-21)

The investigation concluded with the **Direct Playwright** approach:

| Mode | Enedis Result |
|------|---------------|
| AIO Sandbox (CDP attach) | Blocked → `/indisponible` |
| AIO Sandbox (fresh launch) | Blocked → `/indisponible` |
| **Host-local Playwright (Direct)** | **Login form visible** |

**Root cause**: The AIO sandbox container exposes detectable signals (SwiftShader
GPU renderer, container-specific TLS fingerprint, network path indicators) that
Enedis's bot-detection identifies.  Host-local Playwright on the real host avoids
all of these.

**Solution implemented**: A parallel `direct_browser` tool suite was created with
identical 13 tools, configured as the `"Browser Agent Direct"` CLI profile.  This
mode uses host-local Playwright with anti-detection flags (`--disable-blink-features=AutomationControlled`),
locale/timezone propagation, and cookie persistence.

Validated sites in Direct mode:
- **Enedis** — login form loads, FriendlyCaptcha visible, not blocked
- **Le Monde** — homepage headlines extracted successfully
- **bot.sannysoft.com** — `webdriver: false`, anti-detection passes

See `docs/browser_control.md` for architecture and usage details.
