# Agent-Driven Browser Automation

Browser automation using a deep agent that controls a real Chromium browser
inside an [AIO Sandbox](https://github.com/agent-infra/sandbox) container.
The agent navigates websites, fills forms, handles authentication, and extracts
data — guided by site-specific **SKILL.md** files.

## Architecture

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
  viewport_width: 1920
  viewport_height: 1080
  default_timeout_ms: 30000
  slow_type_ms: 60
  anti_bot_js: true
  cookies_dir: "data/sessions"
  allowed_credential_envs:
    - ENEDIS_USERNAME
    - ENEDIS_PASSWORD
```

## Module Structure

```
genai_tk/tools/sandbox_browser/
├── __init__.py       # Public exports
├── models.py         # SandboxBrowserConfig, CredentialRef, PageSummary
├── session.py        # SandboxBrowserSession (Playwright CDP connection)
├── tools.py          # All 10 LangChain tools
└── factory.py        # create_sandbox_browser_tools()
```

## Testing

```bash
# Unit tests
uv run pytest tests/unit_tests/tools/sandbox_browser/ -v

# Integration test (requires running AIO sandbox)
docker run --security-opt seccomp=unconfined --rm -d -p 8080:8080 ghcr.io/agent-infra/sandbox:latest
uv run cli agents langchain -p "Browser Agent" --sandbox docker "Navigate to example.com and read the page"
```