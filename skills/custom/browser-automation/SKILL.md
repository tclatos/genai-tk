---
name: browser-automation
description: Generic browser automation patterns for navigating websites, filling forms, handling authentication, and extracting data using browser tools (sandbox or direct Playwright)
---

# Browser Automation Skill

You control a real Chromium browser using these tools. The tools work identically
whether running inside an AIO sandbox container ("Browser Agent" profile) or as
a host-local Playwright browser ("Browser Agent Direct" profile).

**Available tools:**
- `browser_navigate` — go to a URL
- `browser_click` — click elements (CSS selector or text)
- `browser_type` — type into form fields
- `browser_fill_credential` — securely fill credentials from env vars (value never visible to you)
- `browser_screenshot` — take a screenshot (use sparingly, costs tokens)
- `browser_read_page` — extract text from page or element
- `browser_scroll` — scroll the page
- `browser_wait` — wait for an element or a duration
- `browser_save_cookies` — persist session to disk
- `browser_load_cookies` — restore a saved session
- `browser_get_logs` — retrieve browser event log for debugging
- `browser_evaluate` — execute JavaScript in the page
- `browser_diagnose` — collect browser fingerprint diagnostics (UA, platform, WebGL, timezone)

**Choosing a browser backend:**
- **Sandbox** (`Browser Agent`): Runs in Docker container. Isolated, secure.
  Good for general automation. May fail on sites with deep fingerprinting.
- **Direct** (`Browser Agent Direct`): Runs on host. Real GPU, real platform.
  Best for sites with aggressive bot-detection (Enedis, SSO portals).

## Core Workflow

### Phase 1 — Plan
1. Identify the target website and goal
2. Check if a saved session exists (`browser_load_cookies`)
3. Plan the navigation path

### Phase 2 — Navigate & Observe
1. `browser_navigate` to the target URL
2. `browser_read_page` to understand the page layout
3. Identify relevant elements (forms, buttons, links, data)

### Phase 3 — Interact
1. `browser_click` for buttons and links
2. `browser_type` for regular text fields
3. `browser_fill_credential` for username/password fields (**always** use this for credentials)
4. `browser_read_page` after each action to see the updated state
5. `browser_screenshot` only when text extraction is insufficient (charts, captchas, visual verification)

### Phase 4 — Extract & Report
1. `browser_read_page` with a specific selector to get the target data
2. Summarise the extracted data for the user
3. `browser_save_cookies` if you authenticated successfully (for next time)

## CSS Selector Reference

### Basic Selectors
| Pattern | Matches | Example |
|---|---|---|
| `#id` | Element by ID | `#search-box` |
| `.class` | Element by class | `.btn-primary` |
| `tag` | By element type | `button`, `input` |
| `[attr=val]` | By attribute | `[data-testid="submit"]` |
| `tag.class` | Combined | `button.primary` |

### Form Selectors
| Pattern | Matches |
|---|---|
| `input[type="email"]` | Email fields |
| `input[type="password"]` | Password fields |
| `input[type="search"]` | Search boxes |
| `input[name="q"]` | Search query inputs |
| `textarea` | Multi-line text areas |
| `select[name="country"]` | Dropdown menus |
| `button[type="submit"]` | Submit buttons |

### Playwright Text Selectors
| Pattern | Matches |
|---|---|
| `button:has-text("Login")` | Button containing text "Login" |
| `a:has-text("Next")` | Link containing text "Next" |
| `text=Accept all` | Element with exact text |

### Navigation Selectors
| Pattern | Matches |
|---|---|
| `a[href*="login"]` | Links containing "login" in href |
| `a[href*="dashboard"]` | Dashboard links |
| `nav a` | Navigation menu links |
| `[role="navigation"] a` | ARIA navigation links |

## Common Authentication Patterns

### Simple Login Form
```
1. browser_navigate → login page
2. browser_read_page → identify email and password fields
3. browser_fill_credential → email field with USERNAME env var
4. browser_fill_credential → password field with PASSWORD env var
5. browser_click → submit button
6. browser_wait → wait for dashboard/success URL
7. browser_read_page → verify successful login
8. browser_save_cookies → save session for next time
```

### OAuth/SSO Redirect
```
1. browser_navigate → application login page
2. browser_click → "Sign in with SSO" or provider button
3. browser_wait → wait for IdP login page to load
4. browser_read_page → identify credential fields on IdP page
5. browser_fill_credential → fill credentials on IdP page
6. browser_click → submit on IdP page
7. browser_wait → wait for redirect back to application
8. browser_read_page → verify successful login
9. browser_save_cookies → save session
```

### Cookie Consent Banners
Many sites show cookie consent banners that block interaction.  Dismiss them first:
```
1. browser_read_page → check for consent banner text
2. browser_click → try these selectors in order:
   - button:has-text("Accept all")
   - button:has-text("Tout accepter")
   - #tarteaucitronAllAllowed
   - #onetrust-accept-btn-handler
   - #didomi-notice-agree-button
   - .axeptio_btn_acceptAll
   - #popin_tc_privacy_button_3
```

## Error Recovery

| Problem | Recovery |
|---|---|
| Element not found | Try alternative selector, use text selector, scroll the page |
| Page timeout | Retry navigation, check URL is correct |
| Login failed | Read error messages, check credential env var names |
| CAPTCHA detected | Take screenshot, inform user — you cannot solve CAPTCHAs |
| Popup/modal blocking | Click dismiss/close button first |
| Cookie consent banner | Dismiss with "Accept all" button (see above) |
| Session expired | Delete saved cookies, re-authenticate from scratch |
| Wrong page after click | Use browser_read_page to verify, navigate back |

## Security Rules

- **ALWAYS** use `browser_fill_credential` for passwords and usernames — NEVER type credentials with `browser_type`
- Verify the domain before filling credentials (check the URL)
- Never store passwords or credential values in any output
- Check for HTTPS before submitting sensitive data
- Report suspicious redirects or unexpected domains to the user
- If you see a phishing indicator (misspelled domain, unusual URL), STOP and alert the user
