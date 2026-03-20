---
name: sharepoint-sso
description: Navigate SharePoint sites behind Microsoft OAuth/SSO, handling MSAL login flows and document access
---

# SharePoint SSO Portal

This skill guides you to access SharePoint sites behind Microsoft Single Sign-On (OAuth).

## Prerequisites

- Environment variables: `SHAREPOINT_USERNAME` (email) and `SHAREPOINT_PASSWORD`
- Browser tools available

## Authentication Flow

Microsoft SSO typically follows this pattern:

### Step 1 — Try Saved Session

```
1. browser_load_cookies name="sharepoint"
2. If successful, navigate to the target SharePoint URL
3. If redirected to login, continue to Step 2
```

### Step 2 — Microsoft Login

```
1. browser_navigate to the SharePoint site URL
2. You will be redirected to login.microsoftonline.com
3. browser_wait selector="input[type='email'], input[name='loginfmt']" timeout_ms=15000

4. Fill email:
   browser_fill_credential
     selector="input[type='email'], input[name='loginfmt']"
     credential_env="SHAREPOINT_USERNAME"

5. Click Next:
   browser_click selector="input[type='submit'], #idSIButton9"

6. Wait for password page:
   browser_wait selector="input[type='password'], input[name='passwd']" timeout_ms=10000

7. Fill password:
   browser_fill_credential
     selector="input[type='password'], input[name='passwd']"
     credential_env="SHAREPOINT_PASSWORD"

8. Click Sign In:
   browser_click selector="input[type='submit'], #idSIButton9"

9. Handle "Stay signed in?" prompt:
   browser_wait selector="#idSIButton9, #idBtn_Back" timeout_ms=10000
   browser_click selector="#idSIButton9"  (Yes, stay signed in)

10. Wait for redirect back to SharePoint:
    browser_wait timeout_ms=15000
    browser_read_page to verify successful login

11. browser_save_cookies name="sharepoint"
```

### Step 3 — Navigate SharePoint

```
1. browser_navigate to the target document library or page
2. browser_read_page to see available content
3. Navigate as needed using browser_click on links
```

## MFA Handling

If MFA is configured:
1. browser_screenshot to show the MFA prompt
2. Inform user: "Microsoft MFA is required. Please complete verification on VNC."
3. browser_wait timeout_ms=120000
4. browser_read_page to verify MFA was completed

## Notes

- Microsoft login pages change frequently — if selectors fail, use browser_read_page and adapt
- SharePoint SPAs may take time to load — use browser_wait generously
- Some SharePoint sites require conditional access policies — these cannot be bypassed
