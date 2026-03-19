# Authenticated Web Scraper Tool

A comprehensive browser automation tool for scraping authenticated websites without exposing credentials to LLMs. Built on Playwright with anti-bot detection, MFA support, and LangChain integration.

## Features

- **Multiple Auth Mechanisms**: Form-based login, OAuth redirects, OAuth pop-ups, pre-saved sessions, custom handlers
- **Anti-Bot Mitigations**: User-agent rotation, viewport randomization, human-like typing delays, navigator.webdriver spoofing
- **Cookie Consent Automation**: Auto-dismiss GDPR/cookie banners (Didomi, Axeptio, Tarteaucitron, OneTrust, CookieBot)
- **Session Reuse**: Cache authenticated sessions to skip re-login on subsequent scrapes
- **LangChain Integration**: `AuthenticatedWebScraperTool` for agent-driven content extraction
- **Secret Management**: Credentials from environment variables or encrypted files (never logged)
- **LLM-Safe**: Scraper runs locally; only extracted content is sent to LLM

## Quick Start

### 1. Install Dependencies

```bash
uv sync --group browser-control
uv run playwright install chromium
```

### 2. Configure a Scraper

Create a YAML file under `config/basic/web_scrapers/` (e.g., `my_site.yaml`):

```yaml
web_scrapers:
  my_scraper:
    description: "Login-protected site"
    browser:
      headless: false
      user_agent: rotating
      viewport: { width: 1920, height: 1080 }
      timeout_ms: 30000
    auth:
      type: form
      login_url: "https://example.com/login"
      credentials:
        username: { env: MY_USERNAME }
        password: { env: MY_PASSWORD }
      selectors:
        username_selector: "#login-input"
        password_selector: "#password-input"
        submit_selector: "button[type=submit]"
        success_url_pattern: "dashboard|home"
    cookie_consent:
      enabled: true
      strategy: auto
      timeout_ms: 8000
    targets:
      - name: homepage
        url: "https://example.com/home"
        wait_for: networkidle
        wait_for_selector: ".main-content"
        extract:
          type: text
          selector: ".main-content"
```

### 3. Set Credentials

```bash
export MY_USERNAME="user@example.com"
export MY_PASSWORD="secret"
```

### 4. Use the CLI

**Capture the session** (opens browser for interactive login/MFA):
```bash
cli browser capture my_scraper
```

**Scrape a page** (no LLM):
```bash
cli browser scrape my_scraper -t homepage
cli browser scrape my_scraper -t homepage -o result.txt
```

**Ask an LLM agent** (with scraper as a tool):
```bash
cli browser run "Summarize the homepage content" -c my_scraper
cli browser run "Find the latest updates" -c my_scraper -t homepage -m gpt_41mini
```

**List available scrapers**:
```bash
cli browser list
```

## Config Schema

### `WebScraperConfig`

Top-level configuration for a scraper.

- **`name`** (str): Scraper identifier
- **`description`** (str): Human-readable description
- **`browser`** (BrowserConfig): Playwright browser settings
- **`auth`** (AuthConfig): Authentication method and credentials
- **`cookie_consent`** (CookieConsentConfig): Cookie banner handling
- **`targets`** (list[TargetConfig]): Pages to scrape

### `BrowserConfig`

Playwright browser launch settings.

- **`headless`** (bool): Run browser in headless mode (default: True)
  - Set to `False` for interactive auth/MFA pop-ups
- **`user_agent`** (str): User-Agent strategy
  - `rotating`: Random UA from pool each session
  - `random`: New random UA each page load
  - Literal string: Use as-is (e.g., `"Mozilla/5.0 Windows..."`)
  - Default: `rotating`
- **`viewport`** (ViewportConfig): Window dimensions
  - `width` (int): Default 1920
  - `height` (int): Default 1080
  - `jitter` (bool): Add ±30px randomness (anti-bot)
- **`locale`** (str): Locale string (e.g., `"en-US"`, `"fr-FR"`)
- **`timeout_ms`** (int): Page operation timeout (default: 30000)
- **`slow_mo_ms`** (int): Artificial delay between actions (default: 0)
- **`java_script_enabled`** (bool): Enable JavaScript (default: True)

### `AuthConfig`

Authentication mechanism.

- **`type`** (str): One of:
  - `form`: Username/password HTML form
  - `oauth_redirect`: OAuth with redirect (no pop-up)
  - `oauth_popup`: OAuth with browser pop-up window
  - `storage_state`: Pre-saved session (skip auth entirely)
  - `none`: No authentication needed
  - `custom`: Custom handler (provide `custom_handler` function)
- **`login_url`** (str): URL to navigate to for auth (required for form/oauth_redirect)
- **`credentials`** (AuthCredentials): Username/password with env/file sources
  - Fields: `username`, `password`
  - Each can be: `{env: VAR_NAME}` or `{file: /path/to/file}`
- **`selectors`** (AuthSelectors): CSS selectors for form fields
  - `username_selector`, `password_selector`, `submit_selector`, `success_url_pattern`
  - Smart defaults provided for common patterns
- **`success_url_pattern`** (str): Regex or substring to detect successful login
- **`session`** (SessionConfig): Session cache settings
  - `storage_state_path`: Where to save browser session (supports `{name}` placeholder)
  - `check_validity`: Verify cookies not expired before reuse
- **`mfa_handler`** (str): Custom MFA handler function path (e.g., `"myapp.auth:handle_mfa"`)

### `CookieConsentConfig`

GDPR/cookie banner handling.

- **`enabled`** (bool): Enable consent automation
- **`strategy`** (str): One of:
  - `auto`: Try common banner patterns in sequence
  - `custom`: Use `custom_handler` function only
- **`timeout_ms`** (int): Per-button timeout
- **`custom_handler`** (str): Function path for custom logic

### `TargetConfig`

A page to scrape.

- **`name`** (str): Target identifier (e.g., `"homepage"`, `"product_page"`)
- **`description`** (str): What this page contains
- **`url`** (str): Full URL to navigate to
- **`wait_for`** (str): Wait strategy (`networkidle`, `load`, `domcontentloaded`)
- **`wait_for_selector`** (str): CSS selector to wait for before extracting
- **`wait_for_selector_timeout_ms`** (int): Timeout for selector (default: 5000)
- **`extract`** (ExtractConfig): Content extraction method

### `ExtractConfig`

How to extract content from the page.

- **`type`** (str): One of:
  - `text`: Plain text from selector (`.innerText`)
  - `dom`: HTML from selector (`.innerHTML`)
  - `screenshot`: Full-page PNG, base64-encoded
  - `custom`: Custom extractor function
- **`selector`** (str): CSS selector to extract from (default: `"body"`)
- **`custom_extractor`** (str): Function path if `type: custom`

## Example: Enedis Solar Production

Monitor your Enedis solar panel output:

```yaml
web_scrapers:
  enedis_production:
    description: "Enedis solar production tracker"
    browser:
      headless: false
      user_agent: rotating
    auth:
      type: form
      login_url: "https://mon-compte-particulier.enedis.fr/auth/login"
      credentials:
        username: { env: ENEDIS_USERNAME }
        password: { env: ENEDIS_PASSWORD }
      selectors:
        username_selector: "input[name=login]"
        password_selector: "input[name=password]"
        submit_selector: "button[type=submit]"
    cookie_consent:
      enabled: true
      strategy: auto
    targets:
      - name: production_daily
        url: "https://mon-compte-particulier.enedis.fr/energy/daily"
        wait_for: networkidle
        wait_for_selector: ".highcharts-root, table, [data-chart]"
        extract:
          type: text
          selector: ".production-data, table"
      - name: production_monthly
        url: "https://mon-compte-particulier.enedis.fr/energy/monthly"
        wait_for: networkidle
        extract:
          type: text
          selector: ".monthly-stats"
```

Then:
```bash
export ENEDIS_USERNAME="your@email.com"
export ENEDIS_PASSWORD="your_password"

cli browser capture enedis_production    # one-time; saves session
cli browser scrape enedis_production -t production_daily
cli browser run "What was my solar production yesterday?" -c enedis_production
```

## Python API

### Using the Scraper Tool Directly

```python
import asyncio
from genai_tk.tools.browser.scraper_session import run_scraper
from genai_tk.tools.browser.config_loader import load_web_scraper_config

async def main():
    config = load_web_scraper_config("enedis_production")
    content = await run_scraper(config, target_name="production_daily")
    print(content[:500])

asyncio.run(main())
```

### Using in a LangChain Agent

```python
from genai_tk.agents.langchain.langchain_agent import LangchainAgent
from genai_tk.tools.browser.factory import create_web_scraper_tool

# Create the scraper tool(s)
tools = create_web_scraper_tool("enedis_production")

# Build an agent with the tool
agent = LangchainAgent(
    profile_name="Research",  # or None for ad-hoc react agent
    tools=tools,
    llm="gpt_41mini"
)

# Ask a question
result = await agent.arun("What was my solar production yesterday?")
print(result)
```

### Custom Authentication Handler

Implement custom auth logic (e.g., for CAPTCHA, security questions):

```python
# myapp/auth.py
from playwright.async_api import Page

async def handle_custom_auth(page: Page, auth_config, scraper_name: str) -> None:
    """Custom auth handler called by AuthConfig.custom_handler."""
    # Navigate to login
    await page.goto(auth_config.login_url)
    
    # Fill username
    await page.fill("#username", "user@example.com")
    await page.fill("#password", "secret")
    
    # Custom: solve security question
    await page.fill("#security-answer", "dog")
    
    # Submit
    await page.click("button[type=submit]")
    await page.wait_for_load_state("networkidle")
    
    # Save session
    from genai_tk.tools.browser.session_manager import SessionManager
    await SessionManager._save_session(page, auth_config, scraper_name)
```

Reference in YAML:
```yaml
auth:
  type: custom
  login_url: "https://example.com/login"
  custom_handler: "myapp.auth:handle_custom_auth"
```

## Architecture

### Core Modules

- **`models.py`**: Pydantic config models (BrowserConfig, AuthConfig, etc.)
- **`user_agents.py`**: Rotating Chrome user-agent pool for anti-bot mitigation
- **`session_manager.py`**: Session persistence and expiry checking
- **`auth_handlers.py`**: Auth method implementations (form, OAuth, etc.)
- **`cookie_consent.py`**: GDPR banner auto-dismissal
- **`config_loader.py`**: YAML config loading with OmegaConf
- **`scraper_session.py`**: Main Playwright orchestrator (async context manager)
- **`langchain_tool.py`**: LangChain BaseTool wrapper
- **`factory.py`**: Tool factory for YAML/config-based creation
- **`cli_commands.py`**: CLI command group (capture, scrape, run, list)

### Data Flow

```
YAML Config
    ↓
config_loader.py (OmegaConf + Pydantic validation)
    ↓
ScraperSession (Playwright browser + anti-bot)
    ├─ auth_handlers.py (authenticate)
    ├─ cookie_consent.py (dismiss banners)
    └─ extract (text/DOM/screenshot)
    ↓
[Plain Text Content]
    ↓
LangChain Agent (if using langchain_tool.py)
    ↓
LLM Response
```

## Anti-Bot Mitigations

The `ScraperSession` applies several techniques to avoid detection:

1. **User-Agent Rotation**: Cycles through 25+ Chrome UA variants per session
2. **Navigator Spoofing**: Removes `navigator.webdriver` fingerprint via `add_init_script`
3. **Viewport Jitter**: ±30px randomization on window dimensions
4. **Slow Typing**: Human-like character-by-character typing in auth forms (±20% delay jitter)
5. **Locale Matching**: Sets `Accept-Language` header to match browser locale
6. **Chrome Runtime Spoofing**: Injects `window.chrome.runtime` to bypass headless checks

All techniques are applied automatically; no configuration needed.

## Testing

Run unit tests:
```bash
uv run pytest tests/unit_tests/tools/browser/ -q
```

Current coverage:
- 47 unit tests across 4 test files
- Config loading, session management, models, user agents

## Limitations & Future Work

### Current Limitations

- Playwright is synchronous on reads but async on actions—if you need fully sync API, wrap with `asyncio.run()`
- MFA handlers must be manually implemented per site (no universal solution)
- Screenshot-based extraction doesn't work with LLM vision models yet
- No persistent database for multiple account sessions

### Future Enhancements

- Vision LLM integration for screenshot extraction
- MFA template library (TOTP, SMS, email)
- Multi-account session management
- Proxy support for distributed scraping
- Browser pool for parallel scraping

## Troubleshooting

### "playwright is required"
```bash
uv sync --group browser-control
```

### "Executable doesn't exist" / Chromium not found
```bash
uv run playwright install chromium
```

### Custom env vars not found
```bash
export MY_VAR=value
cli browser capture my_scraper
```

### Session "too old" error
```bash
cli browser capture my_scraper --force
```
(Deletes the cached session and re-authenticates.)

### Browser window won't open / headless mode issues
Ensure `headless: false` in BrowserConfig for interactive auth:
```yaml
browser:
  headless: false
```

## Contributing

When adding auth methods or extraction logic:

1. Add model to `models.py` with Pydantic validation
2. Implement handler in `auth_handlers.py` or extraction function
3. Add integration tests in `tests/unit_tests/tools/browser/`
4. Update this README with examples

## License

Same as genai-tk. See LICENSE file.
