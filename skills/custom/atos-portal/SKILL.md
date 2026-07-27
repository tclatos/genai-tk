---
name: atos-portal
description: Navigate Atos MyAtos portal (SAP Fiori) behind SSO authentication with manual user login
browser_backend: direct
---

# Atos MyAtos Portal Access

This skill guides you to access the Atos MyAtos portal (SAP Fiori-based) behind corporate Single Sign-On (SSO).
**All credentials are entered manually by the user** — the agent never handles passwords or secrets.

## Prerequisites

- Browser tools available (works with "Browser Agent Direct" profile)
- The Atos portal URL: `https://wac.das.myatos.net/portal/auth.jsp`

## Authentication Flow

### Step 1 — Try Saved Session

```
1. browser_load_cookies name="atos-portal"
2. If successful, navigate to https://nextgen.myatos.net/sap/flp
3. browser_read_page — if you see "MyAtos" or "Page d'accueil", login is OK
4. If redirected to login page, continue to Step 2
```

### Step 2 — Manual SSO Login (user enters everything)

The agent navigates to the login page, then **waits for the user** to fill in credentials manually:

```
1. browser_navigate to https://wac.das.myatos.net/portal/auth.jsp
2. browser_read_page to confirm the login page is displayed
3. Inform the user: "Please enter your DAS ID, password, and AUTH code in the browser, then click AUTH to log in."
4. browser_wait_for_user message="Waiting for user to log in via Atos SSO..." timeout_seconds=180
5. After URL changes (redirect to portal), browser_read_page to confirm login success
6. browser_wait with load_state="networkidle" to ensure full page load
7. browser_save_cookies name="atos-portal"
```

**IMPORTANT:** Do NOT use `browser_type` or `browser_fill_credential` for credentials.
The user must enter DAS ID, password, and AUTH code manually in the browser window.

### Step 3 — MyAtos Portal Navigation

The Atos MyAtos portal is a SAP Fiori-based interface:

```
1. browser_read_page to see available apps and tiles
2. The portal displays:
   - "Page d'accueil" (Home page)
   - "General Apps" section
   - "My Tasks" section
3. browser_click on the target app or tile
4. browser_wait for the app to load
5. browser_read_page to extract data
```

## Login Form Details

### URL
- **Auth Page**: `https://wac.das.myatos.net/portal/auth.jsp`
- **Portal Page**: `https://nextgen.myatos.net/sap/flp`

### Form Fields (filled by user, not the agent)
- **DAS ID**: Text input for username (Atos employee ID)
- **Password**: Password input
- **AUTH Code**: Field for 2FA/OTP
- **AUTH Button**: Submit button

### Alternative Login Options
The page also offers alternative authentication methods:
- PKI (certificate-based)
- OTP (one-time password)
- DAS (default)
- BULL (legacy)

## Redirect Flow

1. User submits credentials at `https://wac.das.myatos.net/portal/auth.jsp`
2. System validates at `https://wac.das.myatos.net/sso_auth/authenticate`
3. On success, redirects to `https://nextgen.myatos.net/sap/flp#Shell-home`

## Notes

- The Atos portal is a SAP Fiori application — expect SPA behavior
- Always wait for `networkidle` after login to ensure full page load
- The portal title is "MyAtos"
- Session cookies are essential — save them after successful authentication
- If authentication fails, the page displays: "Authentication Unsuccessful"
- The portal is for internal Atos use only (© Atos S.E. 2026)

## Example Workflow

```python
# Load saved session
browser_load_cookies(name="atos-portal")
browser_navigate(url="https://nextgen.myatos.net/sap/flp")

# If not authenticated, go to login page
browser_navigate(url="https://wac.das.myatos.net/portal/auth.jsp")
browser_read_page()

# Wait for user to log in manually
browser_wait_for_user(message="Please log in with your DAS ID, password and AUTH code", timeout_seconds=180)

# Confirm login and save session
browser_wait(load_state="networkidle", timeout_ms=10000)
browser_read_page()
browser_save_cookies(name="atos-portal")
```

## Troubleshooting

- **"Authentication Unsuccessful"**: Ask the user to check credentials and retry
- **Page not loading**: Wait longer with `browser_wait(load_state="networkidle")`
- **Session expired**: Delete saved cookies and re-authenticate
- **User too slow**: Increase `timeout_seconds` in `browser_wait_for_user`
