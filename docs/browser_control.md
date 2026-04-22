# Browser Control

Agent-driven browser automation using a deep agent that controls a real Chromium browser.
Two modes are available depending on the target site's bot-detection level.

## Modes

| Mode | Profile | Browser location | When to use |
|------|---------|-----------------|-------------|
| **Sandbox** | `Browser Agent` | Inside Docker container (AIO) | Development, isolated environments |
| **Direct** | `Browser Agent Direct` | Host machine via Playwright | Sites with aggressive bot-detection (Enedis, SSO portals) |

Both modes expose **identical tool names** (`browser_navigate`, `browser_click`, etc.)
so SKILL.md files work unchanged with either backend.

## Architecture

### Sandbox Mode

```
User query
    │
    ▼
Deep Agent (LLM, runs on host)
    │  tool calls
    ▼
Sandbox Browser Tools  ←─ reads SKILL.md via AioSandboxBackend
    │  Playwright CDP
    ▼
AioSandboxBackend (HTTP client)
    │  HTTP
    ▼
OpenSandbox Server  ←─ manages container lifecycle + port allocation
    │  Docker
    ▼
ghcr.io/agent-infra/sandbox container
    └─ Chromium (VNC on :8080/vnc + CDP on :9222)
```

[OpenSandbox](https://github.com/alibaba/OpenSandbox) creates and manages the Docker
container lifecycle. The `AioSandboxBackend` auto-starts `opensandbox-server` if it
is not already running. See [Sandbox Support](sandbox_support.md) for setup and the
`cli sandbox` commands.

### Direct Mode

```
User query
    │
    ▼
Deep Agent (LLM, runs on host)
    │  tool calls
    ▼
Direct Browser Tools  ←─ reads SKILL.md from host filesystem
    │  Playwright API
    ▼
Host Chromium (real GPU, real UA, real network)
```

## Quick Start

### Direct Mode (recommended for bot-sensitive sites)

```bash
# 1. Install Playwright
uv sync --group browser-control
uv run playwright install chromium

# 2. Set credentials
export ENEDIS_USERNAME="your_email@example.com"
export ENEDIS_PASSWORD="your_password"

# 3. Run
uv run cli agents langchain -p "Browser Agent Direct" \
  "Get my solar panel production for this month from Enedis"
```

### Sandbox Mode

```bash
# 1. Install dependencies
uv sync --group browser-control
uv run playwright install chromium

# 2. Start sandbox server (once per session, survives CLI exit)
uv run cli sandbox start
uv run cli sandbox pull   # pre-pull image to avoid first-run delay (~20 s)

# 3. Set credentials
export ENEDIS_USERNAME="your_email@example.com"
export ENEDIS_PASSWORD="your_password"

# 4. Run the agent
uv run cli agents langchain -p "Browser Agent" \
  "Get my solar panel production for this month from Enedis"

# 5. Optional: watch the browser live
# http://localhost:8080/vnc/index.html?autoconnect=true

# 6. Stop sandbox server when done
uv run cli sandbox stop
```

Keep the sandbox warm across turns (avoids container startup overhead):
```bash
uv run cli agents langchain -p "Browser Agent" --keep-sandbox --chat "..."
```

## Browser Tools Reference

| Tool | Purpose |
|------|---------|
| `browser_navigate` | Navigate to URL; returns title + URL + text snippet |
| `browser_click` | Click element by CSS selector |
| `browser_type` | Type text into element |
| `browser_fill_credential` | Fill field from env var (**value never reaches LLM**) |
| `browser_screenshot` | Capture page as base64 PNG |
| `browser_read_page` | Extract text content (optional selector scope) |
| `browser_scroll` | Scroll page up/down |
| `browser_wait` | Wait for selector or load state |
| `browser_save_cookies` | Persist session cookies to file |
| `browser_load_cookies` | Restore session cookies from file |
| `browser_get_logs` | Return recent browser events |
| `browser_evaluate` | Execute JavaScript and return result |
| `browser_diagnose` | Return browser fingerprint + event log (debugging) |

## Credential Security

`browser_fill_credential` is one of the way to enter credentials:

```
agent → browser_fill_credential(selector="#email", credential_env="ENEDIS_USERNAME")
tool  → resolves os.environ["ENEDIS_USERNAME"] inside the tool
tool  → types value into field
tool  → returns: "Credential from $ENEDIS_USERNAME filled into '#email'."
```

The actual value never appears in LLM context, conversation history, or logs.

An allowlist in `config/sandbox.yaml` restricts which env vars the tool can
access, preventing prompt-injection attacks that try to exfiltrate other secrets:

```yaml
sandbox_browser:
  allowed_credential_envs:
    - ENEDIS_USERNAME
    - ENEDIS_PASSWORD
    - SHAREPOINT_USERNAME
    - SHAREPOINT_PASSWORD
```

## Session Caching

Credentials are entered once per session:
1. After successful login → `browser_save_cookies name="my_site"`
2. Subsequent runs → `browser_load_cookies name="my_site"` (no re-authentication)

Saved sessions are stored in `data/sessions/`. Delete this directory after
rotating credentials.

## Skills (Site-specific Knowledge)

Site knowledge is encoded in **SKILL.md** files, not Python code.
Adding support for a new site = adding one file.

| Skill | Path | Purpose |
|-------|------|---------|
| `browser-automation` | `skills/custom/browser-automation/SKILL.md` | Generic patterns, selectors, error recovery |
| `enedis-portal` | `skills/custom/enedis-portal/SKILL.md` | Enedis solar production portal |
| `sharepoint-sso` | `skills/custom/sharepoint-sso/SKILL.md` | SharePoint behind Microsoft SSO |
| `sap-portal` | `skills/custom/sap-portal/SKILL.md` | SAP Fiori/WebGUI behind SSO |

### Creating a New Site Skill

Create `skills/custom/my-site/SKILL.md`:

```markdown
---
name: my-site
description: Navigate my-site.com, handle login, and extract data
---

# My Site

## Prerequisites
- Environment variables: `MY_SITE_USER` and `MY_SITE_PASS`

## Workflow
1. browser_load_cookies name="my_site"
2. browser_navigate to https://my-site.com
3. If login required:
   - browser_fill_credential selector="#email" credential_env="MY_SITE_USER"
   - browser_fill_credential selector="#password" credential_env="MY_SITE_PASS"
   - browser_click selector="button[type='submit']"
4. browser_read_page to extract data
5. browser_save_cookies name="my_site"
```

Then add the env var names to `allowed_credential_envs` in `sandbox.yaml`.

In sandbox mode, skill directories are bind-mounted into the container at
`/mnt/skills/` so the deep agent's `SkillsMiddleware` can read them through
the container filesystem API.

## Configuration

### Browser settings (`config/sandbox.yaml`)

```yaml
sandbox_browser:
  locale: "fr-FR"
  timezone_id: "Europe/Paris"
  viewport_width: 1920
  viewport_height: 1080
  default_timeout_ms: 30000
  slow_type_ms: 60
  cookies_dir: "data/sessions"
  launch_mode: "cdp"   # "cdp" (default) or "fresh"
  allowed_credential_envs:
    - ENEDIS_USERNAME
    - ENEDIS_PASSWORD
```

`launch_mode: fresh` kills the pre-launched container browser and starts a fresh
Chromium with anti-detection flags before connecting. Use this if the default
CDP-attach mode is detected by the target site.

### Agent profiles (`config/agents/langchain.yaml`)

Both `Browser Agent` (sandbox) and `Browser Agent Direct` (host) profiles are
pre-configured with `type: deep`, planning enabled, and the appropriate browser
tool factory.

## Security Summary

| Layer | Mechanism | Protects against |
|-------|-----------|-----------------|
| Application | Allowlist + credential hiding | LLM exfiltration, prompt injection |
| Container | Docker namespaces + network isolation | Host compromise, lateral movement |
| Auditability | Open-source image + VNC visibility | Supply-chain risk, undetected actions |

**Recommended practices:**
- Store credentials in a secrets manager; inject as env vars at runtime.
- Keep `allowed_credential_envs` as short as possible.
- Enable Docker resource limits (`--memory`, `--cpus`).
- Rotate credentials regularly; clear `data/sessions/` after rotation.

## Choosing Between Sandbox and Direct Mode

Investigation (see [design/sandbox_bot_detection.md](design/sandbox_bot_detection.md))
found that Enedis and similar portals detect the AIO container due to the SwiftShader
software GPU renderer and Docker network path. Host-local Playwright avoids these signals:

| Signal | Sandbox | Direct |
|--------|---------|--------|
| GPU renderer | SwiftShader (software) | Real host GPU |
| Platform | Linux container | Matches host |
| Network path | Docker NAT | Direct |
| Enedis portal | Blocked ❌ | Works ✅ |

Use **Sandbox** for: isolated execution, untrusted pages, development/CI.  
Use **Direct** for: sites with deep fingerprinting (Enedis, SSO portals, paywalls).

## Module Structure

```
genai_tk/tools/sandbox_browser/    # AIO sandbox browser (Docker)
├── models.py         # SandboxBrowserConfig, CredentialRef, PageSummary
├── session.py        # SandboxBrowserSession (Playwright CDP or fresh)
├── tools.py          # 13 LangChain tools
└── factory.py        # create_sandbox_browser_tools()

genai_tk/tools/direct_browser/     # Host-local Playwright browser
├── models.py         # DirectBrowserConfig
├── session.py        # DirectBrowserSession (Playwright local launch)
├── tools.py          # 13 LangChain tools (same names as sandbox)
└── factory.py        # create_direct_browser_tools()
```

## Testing

```bash
# Unit tests — both suites
uv run pytest tests/unit_tests/tools/sandbox_browser/ tests/unit_tests/tools/direct_browser/ -v

# Integration tests — direct browser against live sites
uv run pytest tests/integration_tests/tools/test_direct_browser_integration.py -v

# Fingerprint comparison probe
uv run python scripts/browser_probe.py --fingerprint-only

# CLI test — sandbox (requires running sandbox server)
uv run cli sandbox start
uv run cli agents langchain -p "Browser Agent" "Navigate to example.com and read the page"

# CLI test — direct
uv run cli agents langchain -p "Browser Agent Direct" "Navigate to example.com and read the page"
```
