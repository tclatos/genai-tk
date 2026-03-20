---
name: sap-portal
description: Navigate SAP web portals (Fiori, WebGUI) behind SSO, handling authentication and data extraction
---

# SAP Portal Access

This skill guides you to access SAP web portals (SAP Fiori, SAP WebGUI, SAP BTP)
behind corporate Single Sign-On.

## Prerequisites

- Environment variables: `SAP_USERNAME` and `SAP_PASSWORD`
- Browser tools available
- The SAP portal URL (provided by the user)

## Authentication Flow

### Step 1 — Try Saved Session

```
1. browser_load_cookies name="sap"
2. If successful, navigate to the target SAP URL
3. If redirected to login, continue to Step 2
```

### Step 2 — SSO Login

SAP portals typically use corporate IdP (ADFS, Okta, Azure AD, etc.):

```
1. browser_navigate to the SAP portal URL
2. You will likely be redirected to the corporate IdP
3. browser_read_page to identify the login form
4. browser_fill_credential with SAP_USERNAME and SAP_PASSWORD
5. browser_click to submit
6. Handle any MFA prompts (screenshot + inform user)
7. browser_wait for redirect back to SAP
8. browser_save_cookies name="sap"
```

### Step 3 — SAP Fiori Navigation

SAP Fiori uses a tile-based UI:
```
1. browser_read_page to see available tiles/apps
2. browser_click on the target tile or app
3. browser_wait for the app to load (SAP apps can be slow)
4. browser_read_page to extract data
```

## Notes

- SAP Fiori apps are heavy SPAs — always wait for full load
- SAP WebGUI renders server-side — page structure differs significantly
- Data extraction from SAP tables may require scrolling and multiple reads
- This is a stub skill — expand with specific SAP portal details when available
