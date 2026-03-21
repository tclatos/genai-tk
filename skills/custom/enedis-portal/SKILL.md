---
name: enedis-portal
description: Navigate the Enedis customer portal to retrieve solar photovoltaic production data, handling authentication, cookie consent, and data extraction
browser_backend: direct
---

# Enedis Solar Production Portal

Retrieve solar panel production data from the Enedis "Mon Compte Particulier" portal.

## Prerequisites

- Environment variables: `ENEDIS_USERNAME` (email) and `ENEDIS_PASSWORD`
- **Recommended: use "Browser Agent Direct" profile** (host-local Playwright).
  Enedis has aggressive bot-detection that blocks the AIO sandbox browser
  (SwiftShader GPU, Mac-on-Linux platform mismatch). The direct browser uses
  the host GPU and real platform, which passes Enedis fingerprinting.
- If using sandbox: configure `launch_mode: fresh` (avoids pre-launched CDP detection)
- Browser tools: `browser_navigate`, `browser_read_page`, `browser_wait`,
  `browser_load_cookies`, `browser_save_cookies`, `browser_screenshot`,
  `browser_evaluate`, `browser_fill_credential`, `browser_get_logs`,
  `browser_diagnose`

## Workflow

### Step 1 — Try to Restore Session

```
1. browser_load_cookies name="enedis"
2. If successful:
   → browser_navigate to https://mon-compte-particulier.enedis.fr/visualiser-vos-mesures-production wait_until="domcontentloaded"
   → browser_wait load_state="networkidle" timeout_ms=10000
   → browser_read_page — check if we're on the production page (not redirected to login or /indisponible)
   → If logged in, skip to Step 3
   → If redirected to login page, continue to Step 2
3. If no saved session, continue to Step 2
```

### Step 2 — Authenticate

Navigate to the Enedis login page and perform Okta two-step login using
browser tools. The `launch_mode: fresh` configuration ensures the sandbox
browser is launched fresh with anti-detection flags, avoiding the
`/indisponible` redirect that occurs with the pre-launched browser.

```
1. browser_navigate to https://mon-compte-particulier.enedis.fr/auth/login wait_until="domcontentloaded"

2. Remove the cookie consent overlay (never click accept):
   browser_evaluate code="document.querySelector('#popin_tc_privacy')?.remove(); document.querySelector('#popin_tc_privacy_container')?.remove(); document.querySelector('.tc-privacy-wrapper')?.remove(); document.body.style.overflow = 'auto';"

3. browser_wait load_state="networkidle" timeout_ms=15000

4. Check current URL:
   → If redirected to /indisponible: report "Site is blocking the sandbox browser" and stop
   → If on login page: continue

5. Fill username:
   browser_fill_credential selector="input[type='email'], input[name='username'], #username" credential_env="ENEDIS_USERNAME"

6. Look for password field. If not visible, submit username first:
   browser_evaluate code="document.querySelector('button[type=submit]')?.click()"
   browser_wait selector="input[type='password']" timeout_ms=15000

7. Fill password:
   browser_fill_credential selector="input[type='password']" credential_env="ENEDIS_PASSWORD"

8. Submit login:
   browser_evaluate code="document.querySelector('button[type=submit]')?.click()"

9. browser_wait load_state="networkidle" timeout_ms=20000

10. Check result:
    browser_read_page
    → If URL contains "captcha" or page shows FriendlyCaptcha:
      browser_screenshot
      Inform user: "A CAPTCHA is displayed. Please solve it via VNC, then tell me to continue."
    → If URL is on the Enedis dashboard or production page: success → Step 3
    → If URL is /indisponible or login page still: report failure with browser_get_logs last_n=80

11. browser_save_cookies name="enedis"
```

### Step 3 — Navigate to Production Data

```
1. browser_navigate to https://mon-compte-particulier.enedis.fr/visualiser-vos-mesures-production wait_until="domcontentloaded"
2. browser_wait load_state="networkidle" timeout_ms=15000
3. browser_wait selector=".highcharts-root, .donnees-mensuelles, table, [class*='mesures'], [class*='production']" timeout_ms=20000
4. browser_read_page to extract the production data
```

For monthly data:
```
1. browser_navigate to https://mon-compte-particulier.enedis.fr/visualiser-vos-mesures-production?periode=mois wait_until="domcontentloaded"
2. browser_wait load_state="networkidle" timeout_ms=15000
3. browser_wait selector="table, [class*='mesures'], .highcharts-root" timeout_ms=20000
4. browser_read_page to extract monthly production figures
```

### Step 4 — Extract and Report

```
1. browser_read_page to get the production data text
2. If the data is in a chart/graph and text extraction is insufficient:
   → browser_screenshot to capture the visual chart
3. Parse the extracted text for production numbers (kWh)
4. Present the data to the user:
   - Daily production (if available)
   - Monthly totals
   - Comparison with previous periods
5. browser_save_cookies name="enedis" (refresh the saved session)
```

## Troubleshooting

| Issue | Solution |
|---|---|
| Redirected to `/indisponible` | Verify `launch_mode: fresh` is set in sandbox config. If fresh mode still fails, deeper investigation needed. |
| CAPTCHA appears | Inform user to solve via VNC, then continue workflow |
| Session expired | Re-authenticate from Step 2 |
| Page loads but no data visible | Wait longer, scroll down, try `browser_screenshot` |
| Login form not found | Check if site is under maintenance; try `browser_screenshot` for diagnostics |

## Important Notes

- The Enedis portal is a French site — all text is in French
- Production data may take a few seconds to load (SPA with async data loading)
- The Highcharts graph loads asynchronously — always wait for `networkidle` or the chart selector
- Save cookies after EVERY successful operation to avoid re-authentication
- The `launch_mode: fresh` setting kills the sandbox's pre-launched Chromium and launches a fresh
  instance with anti-detection flags — this is transparent and handled by the browser session layer
