---
name: enedis-portal
description: Navigate the Enedis customer portal to retrieve solar photovoltaic production data, handling authentication, cookie consent, and data extraction
---

# Enedis Solar Production Portal

This skill guides you to retrieve solar panel production data from the Enedis
"Mon Compte Particulier" portal.

## Prerequisites

- Environment variables set: `ENEDIS_USERNAME` (email) and `ENEDIS_PASSWORD`
- Browser tools available: `browser_navigate`, `browser_click`, `browser_type`,
  `browser_fill_credential`, `browser_read_page`, `browser_wait`, `browser_save_cookies`,
  `browser_load_cookies`, `browser_get_logs`, `browser_evaluate`

## Workflow

### Step 1 — Try to Restore Session

```
1. browser_load_cookies name="enedis"
2. If successful:
   → browser_navigate to https://mon-compte-particulier.enedis.fr/visualiser-vos-mesures-production wait_until="domcontentloaded"
   → browser_wait load_state="networkidle" timeout_ms=10000
   → browser_read_page to check if we're logged in (look for production data or dashboard content)
   → If logged in, skip to Step 4
   → If redirected to login page, continue to Step 2
3. If no saved session, continue to Step 2
```

### Step 2 — Dismiss Cookie Overlay and Navigate to Login

IMPORTANT: Do NOT click the cookie consent "Tout accepter" button.
Clicking it loads tracking/security scripts that block automated browsers.
Instead, navigate directly to the login page and dismiss the cookie overlay
via JavaScript if it appears.

```
1. browser_navigate to https://mon-compte-particulier.enedis.fr/auth/login wait_until="domcontentloaded"
2. browser_wait selector="body" timeout_ms=10000
3. browser_evaluate expression="document.querySelector('#popin_tc_privacy')?.remove(); document.querySelector('#popin_tc_privacy_container')?.remove(); document.querySelector('.tc-privacy-wrapper')?.remove(); document.body.style.overflow='auto'; 'overlay removed'"
   (This removes the cookie popup overlay from the DOM without triggering consent scripts)
4. browser_wait load_state="networkidle" timeout_ms=10000
5. browser_read_page
   (Check the reported URL and title. If the page already shows maintenance text or an unexpected route before the login form appears, collect diagnostics before retrying.)
6. If the login form is still not visible or the page reports maintenance / unavailable content:
   → browser_evaluate expression="({href: window.location.href, title: document.title, readyState: document.readyState, bodyText: document.body?.innerText?.slice(0, 1500) ?? ''})"
   → browser_get_logs last_n=80
   → Report the observed URL, title, page text, and recent browser logs to the user instead of guessing.
```

### Step 3 — Authenticate

The login form should now be visible (possibly behind a loading spinner).
Enedis uses an Okta-based form login.

```
1. browser_wait selector="input[type='email'], input[name='username'], #username" timeout_ms=15000
2. browser_read_page to identify the form structure and confirm the URL is still a login flow
3. Fill the email/username field:
   browser_fill_credential
     selector="input[type='email'], input[name='username'], #username"
     credential_env="ENEDIS_USERNAME"

4. Check if password field is visible. Some Enedis login forms are two-step
   (username first, then password on next screen).
   → If password field is NOT visible, click the submit/continue button first:
     browser_click selector="button[type='submit'], button:has-text('Continuer'), button:has-text('Suivant')"
     browser_wait selector="input[type='password']" timeout_ms=10000

5. Fill the password field:
   browser_fill_credential
     selector="input[type='password'], input[name='password'], #password"
     credential_env="ENEDIS_PASSWORD"

6. Submit the login form:
   browser_click selector="button[type='submit'], button:has-text('Se connecter'), button:has-text('Connexion')"

7. Wait for successful login — the URL should contain "/espace-client/" or similar:
   browser_wait load_state="networkidle" timeout_ms=20000
   browser_read_page to verify we're on the dashboard (not still on login page)

8. If login failed:
   → browser_read_page to check for error messages
   → Look for: ".error", "[role='alert']", ".message-erreur"
   → browser_get_logs last_n=80 if the URL/title/body do not match the expected login or dashboard state
   → Report the error to the user

9. If a CAPTCHA appears (FriendlyCaptcha):
   → browser_screenshot to show it to the user
   → Inform the user: "A CAPTCHA is displayed. Please solve it manually on the VNC view at http://localhost:8080/vnc/index.html"
   → browser_wait timeout_ms=120000 (wait up to 2 minutes for manual solving)
   → Then retry the submit

10. Save the session for future use:
    browser_save_cookies name="enedis"
```

### Step 4 — Navigate to Production Data

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

### Step 5 — Extract and Report

```
1. browser_read_page to get the production data text
2. If the data is in a chart/graph and text extraction is insufficient:
   → browser_screenshot to capture the visual chart
3. Parse the extracted text for production numbers (kWh)
4. Present the data to the user in a clear format:
   - Daily production (if available)
   - Monthly totals
   - Comparison with previous periods
5. browser_save_cookies name="enedis" (refresh the saved session)
```

## Troubleshooting

| Issue | Solution |
|---|---|
| Redirected to login after loading cookies | Session expired — re-authenticate from Step 2 |
| "Votre session a expiré" message | Re-authenticate from Step 2 |
| CAPTCHA appears | Show VNC URL to user, wait for manual solving |
| Two-step login form | Submit username first, wait for password field |
| Cookie banner reappears | Dismiss again via browser_evaluate (do NOT click accept) |
| URL/title/body switches to an unavailable or maintenance page | Run `browser_read_page`, `browser_evaluate`, and `browser_get_logs` to capture the exact URL, title, page text, and recent navigations before concluding the site blocked the sandbox |
| Page loads but no data visible | Wait longer, scroll down, try `browser_screenshot` |
| "Accès refusé" or 403 error | Session invalid — clear cookies and re-authenticate |

## Important Notes

- The Enedis portal is a French site — all text is in French
- The URL locale is `fr-FR` — this is configured in the browser settings
- Production data may take a few seconds to load (SPA with async data loading)
- The Highcharts graph loads asynchronously — always wait for `networkidle` or the chart selector
- Save cookies after EVERY successful login to avoid re-authentication next time
