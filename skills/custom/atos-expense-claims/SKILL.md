---
name: atos-expense-claims
description: Navigate to the Atos MyAtos Expense Claims (Notes de Frais) menu from the portal home page
browser_backend: direct
---

# Atos MyAtos Expense Claims Navigation

This skill guides you to navigate to the Expense Claims (Notes de Frais) menu in the Atos MyAtos portal.

**Prerequisites:**
- User must already be authenticated on the Atos MyAtos portal
- The portal home page must be accessible at `https://nextgen.myatos.net/sap/flp#Shell-home`
- This skill does NOT handle authentication — use the `atos-portal` skill for login

## Navigation Flow

### Step 1 — Verify Portal Access

```
1. browser_navigate to https://nextgen.myatos.net/sap/flp
2. browser_wait with load_state="domcontentloaded" (timeout: 5000ms)
3. browser_read_page to confirm you see the MyAtos portal home page
4. Verify the page contains "Page d'accueil" or "Mes applications"
```

### Step 2 — Locate and Click Expense Claims Menu

The Expense Claims menu is typically located in the "Mes applications" (My Applications) section:

```
1. browser_read_page to see all available menu items
2. Look for the link/button with text "Note de frais" (Expense Claims)
3. browser_click on the "Note de frais" link using selector: a:has-text("Note de frais")
4. browser_wait with load_state="networkidle" (timeout: 5000ms) to ensure full page load
```

### Step 3 — Confirm Expense Claims Page Loaded

```
1. browser_read_page to verify the Expense Claims page is displayed
2. Confirm the page title contains "My expenses" or "Poste de travail personne en déplacement"
3. The URL should change to: https://nextgen.myatos.net/sap/flp#Expense-claim
```

## Page Structure

### MyAtos Portal Home Page
- **URL**: `https://nextgen.myatos.net/sap/flp#Shell-home`
- **Title**: "MyAtos" or "Page d'accueil"
- **Sections**:
  - "Page d'accueil" (Home)
  - "Mes applications" (My Applications)
  - "General Apps"
  - "My Tasks"

### Expense Claims Page
- **URL**: `https://nextgen.myatos.net/sap/flp#Expense-claim`
- **Title**: "My expenses" or "Poste de travail personne en déplacement"
- **Content**: Expense claims list and management interface

## CSS Selectors

| Element | Selector |
|---------|----------|
| Expense Claims Link | `a:has-text("Note de frais")` |
| Portal Home Link | `a:has-text("Page d'accueil")` |

## Error Handling

### Not Authenticated
If you see a login page instead of the portal home:
- The session has expired or is invalid
- Use the `atos-portal` skill to re-authenticate
- Then retry this skill

### Page Not Loading
If the Expense Claims page fails to load:
- Wait longer with `browser_wait` (increase timeout to 10000ms)
- Check for JavaScript errors with `browser_get_logs`
- Verify the URL changed to `#Expense-claim`

## Example Workflow

```python
# Assuming user is already authenticated
browser_navigate(url="https://nextgen.myatos.net/sap/flp")
browser_wait(load_state="domcontentloaded", timeout_ms=5000)
browser_read_page()

# Click on Expense Claims
browser_click(selector='a:has-text("Note de frais")')
browser_wait(load_state="networkidle", timeout_ms=5000)
browser_read_page()

# Verify we're on the Expense Claims page
# Extract data or perform further actions
```

## Notes

- The Atos MyAtos portal is a SAP Fiori-based SPA (Single Page Application)
- Navigation uses hash-based routing (`#Shell-home`, `#Expense-claim`)
- Always wait for `networkidle` after clicking to ensure full page load
- The portal is in French — menu items use French labels
- This skill assumes the user has access to the Expense Claims module
- If the "Note de frais" link is not visible, the user may not have the required permissions

## Related Skills

- **atos-portal**: Handles authentication and initial portal access
- **browser-automation**: Generic browser automation patterns
