# Agent-Driven Browser Automation

Browser automation using a deep agent that controls a real Chromium browser.
Two tool suites are available:

- **Sandbox** (`Browser Agent`): Browser runs inside an [AIO Sandbox](https://github.com/agent-infra/sandbox) Docker container.
  Isolated and secure, but may trigger bot-detection on sites with deep fingerprinting.
- **Direct** (`Browser Agent Direct`): Browser runs on the host via Playwright.
  Uses real GPU, real platform UA, real network. Best for sites with aggressive
  bot-detection (Enedis, SSO portals, paywall sites).

Both suites expose **identical tool names** (`browser_navigate`, `browser_click`, etc.)
so SKILL.md files work unchanged with either backend.

The agent navigates websites, fills forms, handles authentication, and extracts
data — guided by site-specific **SKILL.md** files.

## Architecture

```
User query: "Get my solar panel production from Enedis"
     │
     ▼
┌───────────────────────────────────────────────┐
│  Deep Agent (LangChain, type: deep)           │
│  ├─ System prompt + browser skills            │
│  ├─ Plans multi-step workflow                  │
│  └─ Calls browser tools sequentially          │
└──────────────┬────────────────────────────────┘
               │  tool calls
       ┌───────┴───────┐
       ▼               ▼
┌──────────────┐ ┌──────────────────────────────┐
│ Direct Tools │ │  Sandbox Browser Tools       │
│ (host PW)    │ │  (AIO Docker)                │
│              │ │                              │
│ Real GPU     │ │  CDP / fresh modes           │
│ Real UA      │ │  VNC observation             │
│ Real network │ │  Docker isolation            │
└──────┬───────┘ └──────────────┬───────────────┘
       │                        │
       ▼                        ▼
   Host Chromium           AIO Container Chromium
```
User query: "Get my solar panel production from Enedis"
     │
     ▼
┌─────────────────────────────────────────┐
│  Deep Agent (LangChain, type: deep)     │
│  ├─ System prompt + browser skills      │
│  ├─ Plans multi-step workflow            │
│  └─ Calls browser tools sequentially    │
└──────────────┬──────────────────────────┘
               │  tool calls
               ▼
┌─────────────────────────────────────────┐
│  Sandbox Browser Tools (LangChain)      │
│  browser_navigate, browser_click,       │
│  browser_type, browser_fill_credential, │
│  browser_screenshot, browser_read_page, │
│  browser_scroll, browser_wait,          │
│  browser_save_cookies,                  │
│  browser_load_cookies                   │
└──────────────┬──────────────────────────┘
               │  Playwright CDP
               ▼
┌─────────────────────────────────────────┐
│  AIO Sandbox Docker Container           │
│  ├─ Real Chromium (VNC + CDP)           │
│  ├─ Shared filesystem                   │
│  └─ Shell, File, Jupyter services       │
└─────────────────────────────────────────┘
```

## Quick Start

### Direct Mode (Host-Local Playwright)

```bash
# 1. Install Playwright
uv sync --group browser-control
uv run playwright install chromium

# 2. Set credentials
export ENEDIS_USERNAME="your_email@example.com"
export ENEDIS_PASSWORD="your_password"

# 3. Run the direct browser agent
uv run cli agents langchain -p "Browser Agent Direct" \
  "Get my solar panel production for this month from Enedis portal"
```

### Sandbox Mode (AIO Docker)

### 1. Install Dependencies

```bash
uv sync --group browser-control
uv run playwright install chromium
```

### 2. Start the AIO Sandbox

```bash
docker run --security-opt seccomp=unconfined --rm -it -p 8080:8080 \
  ghcr.io/agent-infra/sandbox:latest
```

### 3. Set Credentials

```bash
export ENEDIS_USERNAME="your_email@example.com"
export ENEDIS_PASSWORD="your_password"
```

### 4. Run the Browser Agent

```bash
uv run cli agents langchain -p "Browser Agent" --sandbox docker \
  "Get my solar panel production for this month from Enedis portal"
```

### 5. Watch (Optional)

Open VNC to see the browser in real-time:
`http://localhost:8080/vnc/index.html?autoconnect=true`

## Browser Tools Reference

| Tool | Input | Returns |
|---|---|---|
| `browser_navigate` | `url` | Page title + URL + text snippet |
| `browser_click` | `selector` | Page state after click |
| `browser_type` | `selector`, `text` | Success confirmation |
| `browser_fill_credential` | `selector`, `credential_env` | "Credential filled" (**value hidden**) |
| `browser_screenshot` | *(none)* | Base64 PNG |
| `browser_read_page` | `selector` (optional) | Page text content |
| `browser_scroll` | `direction`, `amount` | Viewport state |
| `browser_wait` | `selector`, `timeout_ms` | Success/timeout |
| `browser_save_cookies` | `name` | File path |
| `browser_load_cookies` | `name` | Success/failure |
| `browser_get_logs` | `last_n` (optional) | Recent browser events |
| `browser_evaluate` | `expression` | JavaScript result |
| `browser_diagnose` | *(none)* | Browser fingerprint JSON + event log |

## Credential Security

`browser_fill_credential` is the only way to enter credentials:

1. Agent calls: `browser_fill_credential(selector="#email", credential_env="ENEDIS_USERNAME")`
2. Tool resolves `os.environ["ENEDIS_USERNAME"]` → types value into the field
3. Returns only: `"Credential from $ENEDIS_USERNAME filled into '#email'."`
4. The actual password/email **never appears** in the LLM context

An **allowlist** in `config/basic/sandbox.yaml` restricts which env vars the
tool can access, preventing the LLM from being tricked into exfiltrating other
env vars:

```yaml
sandbox_browser:
  allowed_credential_envs:
    - ENEDIS_USERNAME
    - ENEDIS_PASSWORD
    - SHAREPOINT_USERNAME
    - SHAREPOINT_PASSWORD
```

## Security Assessment

This browser automation approach provides multiple independent layers of
protection that make credential-based authentication safe to use with LLMs.

### 1. Credential Isolation (LLM-Level)

**Credentials never reach the LLM context.**

- The agent calls `browser_fill_credential` with only the **env var name**
  (e.g. `ENEDIS_USERNAME`), not the actual value.
- The credential is resolved via `os.environ` **only inside the tool**, in
  server memory, and is never serialised into the conversation.
- The tool returns only a confirmation string — the password never appears in
  any LLM response, log line, or conversation turn.
- The **allowlist** in `sandbox.yaml` ensures the LLM cannot be prompt-injected
  into requesting arbitrary env vars (AWS keys, API tokens, etc.).

### 2. Docker Sandboxing (Infrastructure-Level)

The browser runs inside a Docker container (`ghcr.io/agent-infra/sandbox:latest`),
isolated from:

- **Host filesystem**: SSH keys, `.aws/config`, `~/.kube/config` are not
  visible inside the container.
- **Host network**: The container has its own network namespace — it cannot
  sniff host traffic or reach other services directly.
- **Host processes**: Kernel namespaces prevent container processes from
  seeing or signalling host processes.
- **Other containers**: Network and PID namespaces isolate containers from
  each other.

Even if malicious JavaScript in a webpage tried to exfiltrate data, it would
be trapped inside the sandbox with no path to the host or other services.

### 3. Open Sandbox (Transparency & Auditability)

The AIO sandbox is **open-source**:

- The exact browser environment is defined in code — no black-box behaviour.
- Source can be audited to verify no telemetry or credential capture is
  happening inside the container.
- **VNC** (`http://localhost:8080/vnc`) lets authorised users watch the browser
  in real-time, enabling human verification and audit trails — important for
  regulated environments.

### 4. Session Token Strategy

Credentials are typed into the browser **only once** per session:

- After a successful login, `browser_save_cookies` persists the session.
- Subsequent agent runs use `browser_load_cookies` to restore the session
  without re-entering credentials, minimising the credential exposure window.

### 5. Real Browser (Anti-Bot Avoidance)

The sandbox runs a **full Chromium** instance (not headless), which:

- Avoids bot-detection rejections on SSO portals (SAP, SharePoint, Enedis).
- Supports modern SPAs, WebSockets, and MFA dialogs that headless Chrome
  struggles with.
- Allows visual verification via VNC for unexpected prompts (CAPTCHA, MFA).

### 6. Defense-in-Depth Summary

| Layer | Mechanism | Protects Against |
|---|---|---|
| **Application** | Allowlist + credential hiding in `browser_fill_credential` | LLM exfiltration, prompt injection |
| **System** | Docker kernel namespaces + network isolation | Host compromise, lateral movement |
| **Operational** | Open-source sandbox + VNC visibility | Supply-chain risk, undetected malicious actions |

A compromise at any single layer does not automatically expose credentials,
because the other layers remain intact.

### Recommended Production Practices

- Store credentials in a secrets manager (Vault, AWS Secrets Manager) and
  inject them as env vars at runtime — never commit to source control.
- Keep `allowed_credential_envs` as short as possible; list only what is
  strictly needed.
- Enable Docker resource limits (`--memory`, `--cpus`) to prevent runaway
  containers.
- Rotate portal credentials regularly; invalidate saved session files after
  rotation (`data/sessions/`).

## Skills

Site-specific knowledge is encoded as **SKILL.md** files, not Python code.
Adding a new site = adding a new skill file.

### Available Skills

| Skill | Location | Purpose |
|---|---|---|
| `browser-automation` | `skills/custom/browser-automation/SKILL.md` | Generic browser patterns, CSS selectors, error recovery |
| `enedis-portal` | `skills/custom/enedis-portal/SKILL.md` | Enedis solar production portal (login + data extraction) |
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

Then add the credential env vars to the allowlist in `sandbox.yaml`.

## Configuration

### Agent Profile (`config/basic/agents/langchain.yaml`)

The `Browser Agent` profile is pre-configured with `type: deep`, planning
enabled, and the `create_sandbox_browser_tools` factory.

### Browser Settings (`config/basic/sandbox.yaml`)

```yaml
sandbox_browser:
  locale: "fr-FR"
  timezone_id: "Europe/Paris"
  viewport_width: 1920
  viewport_height: 1080
  default_timeout_ms: 30000
  slow_type_ms: 60
  cookies_dir: "data/sessions"
  allowed_credential_envs:
    - ENEDIS_USERNAME
    - ENEDIS_PASSWORD
```

## Module Structure

```
genai_tk/tools/sandbox_browser/    # AIO sandbox browser (Docker)
├── __init__.py       # Public exports
├── models.py         # SandboxBrowserConfig, CredentialRef, PageSummary
├── session.py        # SandboxBrowserSession (Playwright CDP or fresh)
├── tools.py          # 13 LangChain tools
└── factory.py        # create_sandbox_browser_tools()

genai_tk/tools/direct_browser/     # Host-local Playwright browser
├── __init__.py       # Public exports
├── models.py         # DirectBrowserConfig
├── session.py        # DirectBrowserSession (Playwright local launch)
├── tools.py          # 13 LangChain tools (same names as sandbox)
└── factory.py        # create_direct_browser_tools()
```

## Agent Profiles

| Profile | Tool Suite | Use Case |
|---|---|---|
| `Browser Agent` | `sandbox_browser` | Generic automation, Docker isolation |
| `Browser Agent Direct` | `direct_browser` | Bot-resistant sites, deep fingerprinting |

## Testing

```bash
# Unit tests — both suites (68 tests)
uv run pytest tests/unit_tests/tools/sandbox_browser/ tests/unit_tests/tools/direct_browser/ -v

# Integration tests — direct browser against live sites (17 tests, requires playwright)
uv run pytest tests/integration_tests/tools/test_direct_browser_integration.py -v

# All browser tests together (85 tests)
uv run pytest tests/unit_tests/tools/sandbox_browser/ tests/unit_tests/tools/direct_browser/ \
  tests/integration_tests/tools/test_direct_browser_integration.py -v

# Fingerprint comparison probe
uv run python scripts/browser_probe.py --fingerprint-only

# CLI agent test — sandbox (requires running AIO sandbox)
docker run --security-opt seccomp=unconfined --rm -d -p 8080:8080 ghcr.io/agent-infra/sandbox:latest
uv run cli agents langchain -p "Browser Agent" --sandbox docker "Navigate to example.com and read the page"

# CLI agent test — direct
uv run cli agents langchain -p "Browser Agent Direct" "Navigate to example.com and read the page"
```

### Validated Sites (Direct Mode)

| Site | URL | Result |
|------|-----|--------|
| example.com | `https://example.com` | Page read, fingerprint collected |
| Le Monde | `https://www.lemonde.fr` | Headlines extracted from homepage |
| Enedis portal | `https://mon-compte-particulier.enedis.fr` | **Login form visible** (not blocked) |
| bot.sannysoft.com | `https://bot.sannysoft.com` | `webdriver: false`, anti-detection passes |