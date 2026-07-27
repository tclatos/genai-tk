# Atos MyAtos Expense Claims Navigation Skill

Navigate to the Expense Claims (Notes de Frais) menu in the Atos MyAtos portal.

## Quick Start

This skill assumes you are **already authenticated** on the Atos MyAtos portal. It provides a straightforward path to the Expense Claims menu.

### Prerequisites
- Active session on Atos MyAtos portal
- Access to the Expense Claims module
- Browser tools available

### Basic Usage

```python
from browser_tools import browser_navigate, browser_click, browser_wait, browser_read_page

# Navigate to portal home
browser_navigate(url="https://nextgen.myatos.net/sap/flp")
browser_wait(load_state="domcontentloaded", timeout_ms=5000)

# Click on Expense Claims menu
browser_click(selector='a:has-text("Note de frais")')
browser_wait(load_state="networkidle", timeout_ms=5000)

# Verify page loaded
page_content = browser_read_page()
print(page_content)
```

## What This Skill Does

1. **Navigates to the MyAtos portal home page**
2. **Locates the "Note de frais" (Expense Claims) menu item**
3. **Clicks the menu to open the Expense Claims interface**
4. **Waits for the page to fully load**
5. **Confirms successful navigation**

## What This Skill Does NOT Do

- ❌ Handle authentication (use `atos-portal` skill for login)
- ❌ Manage credentials
- ❌ Create or submit expense claims
- ❌ Extract detailed expense data

## File Structure

```
atos-expense-claims/
├── SKILL.md           # Full skill documentation
├── README.md          # This file
├── example.py         # Python example implementation
└── selectors.json     # CSS selectors reference
```

## Troubleshooting

### "Note de frais" link not found
- Verify you are authenticated
- Check if you have permission to access Expense Claims
- Use `browser_read_page()` to see all available menu items

### Page not loading after click
- Increase the wait timeout: `browser_wait(load_state="networkidle", timeout_ms=10000)`
- Check browser logs: `browser_get_logs()`
- Verify the URL changed to `#Expense-claim`

### Session expired
- Use the `atos-portal` skill to re-authenticate
- Then retry this skill

## Related Skills

- **atos-portal**: Authentication and portal access
- **browser-automation**: Generic browser patterns
- **sap-portal**: General SAP portal navigation

## Support

For issues or questions, refer to:
- `SKILL.md` for detailed documentation
- `example.py` for code examples
- `selectors.json` for CSS selector reference
