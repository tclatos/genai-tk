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
  `browser_load_cookies`

## Workflow

### Step 1 — Try to Restore Session

```
1. browser_load_cookies name="enedis"
2. If successful:
   → browser_navigate to https://mon-compte-particulier.enedis.fr/visualiser-vos-mesures-production
   → browser_read_page to check if we're logged in (look for production data or dashboard content)
   → If logged in, skip to Step 4
   → If redirected to login page, continue to Step 2
3. If no saved session, continue to Step 2
```

### Step 2 — Handle Cookie Consent

The Enedis site uses TC Privacy (Tarteaucitron) for cookie consent.  Dismiss
it before attempting login.

```
1. browser_navigate to https://mon-compte-particulier.enedis.fr
2. browser_wait selector="#popin_tc_privacy_button_3" timeout_ms=8000
3. browser_click selector="#popin_tc_privacy_button_3"
   (This is the "Tout accepter" button on the TC Privacy banner)
4. If that fails, try:
   - button:has-text("Tout accepter")
   - button:has-text("Accepter")
   - #tarteaucitronAllAllowed
```

### Step 3 — Authenticate

The login page is at `https://mon-compte-particulier.enedis.fr/auth/login`.
Enedis uses an Okta-based form login.

```
1. browser_navigate to https://mon-compte-particulier.enedis.fr/auth/login
2. browser_wait selector="input[type='email'], input[name='username'], #username" timeout_ms=15000
3. browser_read_page to identify the form structure

4. Fill the email/username field:
   browser_fill_credential
     selector="input[type='email'], input[name='username'], #username"
     credential_env="ENEDIS_USERNAME"

5. Check if password field is visible.  Some Enedis login forms are two-step
   (username first, then password on next screen).
   → If password field is NOT visible, click the submit/continue button first:
     browser_click selector="button[type='submit'], button:has-text('Continuer'), button:has-text('Suivant')"
     browser_wait selector="input[type='password']" timeout_ms=10000

6. Fill the password field:
   browser_fill_credential
     selector="input[type='password'], input[name='password'], #password"
     credential_env="ENEDIS_PASSWORD"

7. Submit the login form:
   browser_click selector="button[type='submit'], button:has-text('Se connecter'), button:has-text('Connexion')"

8. Wait for successful login — the URL should contain "/espace-client/" or similar:
   browser_wait selector="body" timeout_ms=20000
   browser_read_page to verify we're on the dashboard (not still on login page)

9. If login failed:
   → browser_read_page to check for error messages
   → Look for: ".error", "[role='alert']", ".message-erreur"
   → Report the error to the user

10. If a CAPTCHA appears (FriendlyCaptcha):
    → browser_screenshot to show it to the user
    → Inform the user: "A CAPTCHA is displayed. Please solve it manually on the VNC view at http://localhost:8080/vnc/index.html"
    → browser_wait timeout_ms=120000 (wait up to 2 minutes for manual solving)
    → Then retry the submit

11. Save the session for future use:
    browser_save_cookies name="enedis"
```

### Step 4 — Navigate to Production Data

```
1. browser_navigate to https://mon-compte-particulier.enedis.fr/visualiser-vos-mesures-production
2. browser_wait selector=".highcharts-root, .donnees-mensuelles, table, [class*='mesures'], [class*='production']" timeout_ms=20000
3. browser_read_page to extract the production data
```

For monthly data:
```
1. browser_navigate to https://mon-compte-particulier.enedis.fr/visualiser-vos-mesures-production?periode=mois
2. browser_wait selector="table, [class*='mesures'], .highcharts-root" timeout_ms=20000
3. browser_read_page to extract monthly production figures
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
| Cookie banner reappears | Dismiss again before proceeding |
| Page loads but no data visible | Wait longer, scroll down, try `browser_screenshot` |
| "Accès refusé" or 403 error | Session invalid — clear cookies and re-authenticate |

## Important Notes

- The Enedis portal is a French site — all text is in French
- The URL locale is `fr-FR` — this is configured in the browser settings
- Production data may take a few seconds to load (SPA with async data loading)
- The Highcharts graph loads asynchronously — always wait for `networkidle` or the chart selector
- Save cookies after EVERY successful login to avoid re-authentication next time
