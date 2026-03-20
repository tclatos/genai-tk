# Sandbox Browser: Bot Detection Analysis

Status document for the Enedis `/indisponible` redirect issue.

## Current Architecture

```
┌─────────────┐    HTTP     ┌──────────────────────┐     CDP      ┌────────────────┐
│  Agent CLI   │ ──────────→│  OpenSandbox Server  │ ────────────→│ Chromium 135   │
│  (host)      │            │  (Docker container)  │              │ (headful, X11) │
└─────────────┘            └──────────────────────┘              └────────────────┘
                                    │
                               VNC viewer
```

### Design Choices (settled)

| Decision | Rationale |
|---|---|
| **Reuse default browser context** | `new_context()` over CDP doesn't inherit Chromium launch flags (UA, `--disable-blink-features`). The default context is genuine. |
| **No anti-bot JS injection** | `add_init_script()` uses CDP's `Page.addScriptToEvaluateOnNewDocument`, which is itself detectable. The headful Chromium already has real plugins/UA/webdriver=false. |
| **No forced User-Agent** | Container runs Chromium 135; forcing Chrome/131 UA creates a mismatch with `navigator.userAgentData.brands`. |
| **Only `--disable-blink-features=AutomationControlled`** | Prevents `navigator.webdriver = true`. This is the single Chromium flag we inject via `BROWSER_EXTRA_ARGS`. |
| **Site-specific logic in SKILLs only** | `session.py` and `tools.py` are generic browser infrastructure. Anti-bot workarounds go in SKILL.md files via `browser_evaluate`. |
| **`browser_evaluate` tool** | Lets SKILLs run arbitrary JS for site-specific DOM manipulation, diagnostics, or detection evasion. |

## Current Problem

The Enedis portal (`mon-compte-particulier.enedis.fr`) redirects to `/indisponible`
("Site fermé temporairement pour maintenance") when accessed from the sandbox.
The same site works fine in a regular desktop browser.

### What We Know

1. **Initial page loads fine** — `browser_navigate` to the root URL succeeds,
   returns `Title: Espace Particulier`, page renders normally.

2. **Redirect happens after ANY interaction** — originally thought to be triggered
   by the cookie consent click, but the last test proved otherwise: navigating
   directly to `/auth/login` with no click also redirects to `/indisponible`.

3. **The redirect is server-side** — the URL changes to `/indisponible`, meaning
   the server decided to redirect (likely via a 302 or SPA router redirect triggered
   by a server response).

4. **The "hardcoded" version worked before** — an earlier version of the browser
   agent (without LLM, with hardcoded navigation) reached the login page successfully.
   This suggests the detection is not purely based on Chromium fingerprinting.

### Disproved Hypotheses

| Hypothesis | Why disproved |
|---|---|
| Cookie consent click loads bot-detection scripts | Redirect still happens without clicking (direct `/auth/login` navigation) |
| `navigator.webdriver = true` | We use `--disable-blink-features=AutomationControlled`; default context inherits it |
| Mismatched User-Agent | Removed forced UA; Chromium uses its genuine Chromium/135 UA |
| `add_init_script` is detectable | Removed all JS injection from default context |
| `new_context()` doesn't inherit flags | Switched to reusing the default context |
| `--ignore-certificate-errors` alters TLS fingerprint | Removed from `BROWSER_EXTRA_ARGS` |

### Remaining Hypotheses

#### H1: CDP connection itself is detectable

Playwright's `connect_over_cdp()` attaches a debugger to the browser. Sites can
detect this via:
- `Runtime.evaluate` CDP domain being active
- Timing differences from CDP overhead
- `window.__playwright` or other Playwright-injected globals
- The DevTools protocol target list being non-empty

**Evidence for**: The "hardcoded" version may have used a different connection
method or didn't use Playwright at all.

**How to test**: Use `browser_evaluate` to check:
```js
JSON.stringify({
  webdriver: navigator.webdriver,
  plugins: navigator.plugins.length,
  languages: navigator.languages,
  hardwareConcurrency: navigator.hardwareConcurrency,
  devtoolsOpen: window.outerHeight - window.innerHeight > 160,
  __playwright: typeof window.__playwright,
  __pw_manual: typeof window.__pw_manual,
  cdcKeys: Object.keys(window).filter(k => k.startsWith('cdc_') || k.startsWith('__')),
})
```

#### H2: Network/IP-level detection

The Docker container's network stack may differ from a regular browser:
- **DNS resolution** — Docker uses its own DNS (127.0.0.11)
- **TCP fingerprint** — Docker's network namespace creates different TCP/IP
  characteristics (TTL, window size, etc.)
- **IP reputation** — The host IP may be flagged if many automated requests
  come from it, or the ISP's IP range may be in a datacenter range

**How to test**: Check the outgoing IP from inside the container:
```bash
docker exec <container> curl -s https://httpbin.org/ip
```
Compare with the host's IP. If different, Docker networking is in play.

#### H3: TLS fingerprint (JA3/JA4)

Even without `--ignore-certificate-errors`, Chromium in Docker may have a
different TLS fingerprint than regular Chrome:
- Different cipher suite ordering
- Missing or different TLS extensions
- GREASE values being different

Enedis likely uses Akamai or a similar CDN/WAF that checks JA3 fingerprints.

**How to test**: Visit a JA3 fingerprint checker from inside the sandbox:
```
browser_navigate to https://tls.browserleaks.com/json
browser_read_page
```

#### H4: Timing-based detection

The server may detect automation based on:
- Request timing patterns (too fast between page load and interaction)
- Missing background requests that a real browser would make (fonts,
  analytics, etc.)
- The page loading without executing certain JS paths

#### H5: SPA-level detection (Angular)

The Enedis portal is an Angular SPA. The `/indisponible` route may be
triggered by Angular's router based on a flag set by a detection script
that runs during initial bootstrap, not just on cookie consent.

**How to test**: Intercept network requests to see if there's an API call
that returns a "blocked" status:
```js
// In browser_evaluate before navigating
const origFetch = window.fetch;
window.fetch = async (...args) => {
  const resp = await origFetch(...args);
  if (args[0]?.toString().includes('indisponible') || resp.url.includes('indisponible')) {
    console.error('REDIRECT_DETECTED', args[0], resp.status, resp.url);
  }
  return resp;
};
```

## Ideas to Move Forward

### Approach A: Bypass CDP — use the container browser directly

Instead of connecting via CDP from the host, run a script **inside the
container** that drives the browser locally. This eliminates all CDP
artifacts.

**Implementation**: Use the sandbox's `shell.exec_command()` to run a
Python script inside the container that uses `subprocess` + `xdotool` or
a local Playwright instance (not over CDP).

**Pros**: No CDP connection = no debugger detection.
**Cons**: Much more complex; loses Playwright's rich API; hard to get
structured data back.

### Approach B: Use browser extensions instead of CDP

Install a browser extension in the sandbox Chromium that receives
navigation commands via native messaging or a local WebSocket.
Extensions run in the browser's own context and are not detectable
as automation.

**Pros**: Completely invisible to bot detection.
**Cons**: Significant development effort; extension API is limited.

### Approach C: Diagnose first, then target

Before trying more fixes, use the existing `browser_evaluate` and
`browser_get_logs` tools to gather hard data:

1. **Check fingerprint**: Navigate to `https://browserleaks.com/javascript`
   or `https://bot.sannysoft.com/` and `browser_read_page` the results.
   Compare with a real browser.

2. **Check TLS**: Navigate to `https://tls.browserleaks.com/json` and
   read the JA3 hash.

3. **Intercept the redirect**: Use `browser_evaluate` to hook
   `window.location` setter and `fetch`/`XMLHttpRequest` to find exactly
   what triggers the redirect to `/indisponible`.

4. **Compare with the hardcoded version**: Find or recreate the original
   code that worked and identify the exact difference.

### Approach D: Use network-level proxying

Route the browser's traffic through a residential proxy or the host's
network stack (instead of Docker's NAT) to eliminate network-level
fingerprint differences.

**Implementation**: `--network=host` in Docker or a SOCKS proxy.

### Approach E: Accept the detection and use the Enedis API

Enedis provides a data API (Enedis DataConnect / SGE API) for accessing
consumption and production data programmatically. This would bypass the
web portal entirely.

**Pros**: More reliable, no bot detection, structured data.
**Cons**: Requires API registration/authorization; different from the
"browser agent" use case.

## Recommended Next Step

**Approach C** (diagnose first) is the lowest effort and provides the data
needed to pick the right fix. Specifically:

1. Run the agent with a prompt like:
   ```
   Navigate to https://bot.sannysoft.com/ and read the full page content.
   Then navigate to https://tls.browserleaks.com/json and read that too.
   ```

2. Compare the results with a real browser visiting the same pages.

3. The difference will point to the exact detection vector, which can then
   be addressed with a targeted fix rather than guessing.
