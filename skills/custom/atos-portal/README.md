# Atos Portal Skill

This skill provides automated access to the Atos MyAtos portal, a SAP Fiori-based employee portal used by Atos employees.

## Quick Start

1. Ensure you have `ATOS_USERNAME` and `ATOS_PASSWORD` environment variables set
2. Use `browser_load_cookies(name="atos-portal")` to restore a saved session
3. If not authenticated, navigate to `https://wac.das.myatos.net/portal/auth.jsp` and follow the login flow
4. After successful login, you'll be redirected to `https://nextgen.myatos.net/sap/flp`

## Key Features

- **SSO Authentication**: Handles Atos DAS ID login
- **Session Management**: Save and restore authenticated sessions
- **SAP Fiori Navigation**: Access apps and tiles in the MyAtos portal
- **Error Handling**: Detects authentication failures and provides guidance

## Files

- `SKILL.md` - Complete skill documentation with authentication flow and examples
- `README.md` - This file

## Related Skills

- `sap-portal` - General SAP portal access (Fiori, WebGUI)
- `sharepoint-sso` - Microsoft OAuth/SSO authentication
- `browser-automation` - Generic browser automation patterns
