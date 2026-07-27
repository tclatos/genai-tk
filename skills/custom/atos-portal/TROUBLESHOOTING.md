# Atos Portal Troubleshooting Guide

## Common Issues and Solutions

### 1. "Authentication Unsuccessful" Error

**Symptom**: After entering credentials and clicking AUTH, you see "Authentication Unsuccessful" message.

**Causes**:
- Incorrect DAS ID or password
- Account locked or disabled
- Network connectivity issues
- Session timeout

**Solutions**:
1. Verify your DAS ID (Atos employee ID) is correct
2. Verify your password is correct (check for caps lock, special characters)
3. If you have 2FA enabled, ensure you enter the AUTH Code
4. Wait a few minutes and retry (account may be temporarily locked)
5. Check your network connection
6. Clear browser cache: `browser_load_cookies(name="atos-portal")` will fail, forcing re-login

### 2. Page Not Loading After Login

**Symptom**: Login appears successful but the MyAtos portal page doesn't load.

**Causes**:
- Network timeout
- JavaScript not executing
- Browser not waiting long enough for page load

**Solutions**:
1. Increase wait timeout: `browser_wait(load_state="networkidle", timeout_ms=15000)`
2. Check browser logs: `browser_get_logs(last_n=50)`
3. Take a screenshot: `browser_screenshot()` to see current state
4. Try navigating directly: `browser_navigate(url="https://nextgen.myatos.net/sap/flp")`

### 3. Session Expired

**Symptom**: You were previously logged in, but now you're back at the login page.

**Causes**:
- Session timeout (typically 30-60 minutes of inactivity)
- Browser cookies were cleared
- Multiple concurrent sessions

**Solutions**:
1. Re-authenticate using the login flow
2. Don't rely on saved cookies for long-running processes
3. Implement session refresh: periodically navigate to portal and check if still authenticated

### 4. 2FA/AUTH Code Required

**Symptom**: Login form shows "AUTH Code" field that needs to be filled.

**Causes**:
- Your account has 2FA enabled
- This is a security feature for sensitive accounts

**Solutions**:
1. Use `browser_wait_for_user()` to pause and let user enter code manually
2. If you have access to the OTP generator, you can automate:
   ```python
   browser_fill_credential(
       selector="input[placeholder*='AUTH']",
       credential_env="ATOS_AUTH_CODE"
   )
   ```
3. Contact Atos IT if you don't have access to 2FA

### 5. Alternative Login Methods

**Symptom**: You want to use PKI, OTP, or BULL authentication instead of DAS.

**Causes**:
- DAS credentials not available
- Preference for certificate-based authentication
- Legacy system requirements

**Solutions**:
1. Click on alternative login option (PKI, OTP, BULL)
2. Follow the specific authentication flow for that method
3. Modify the skill to handle alternative methods

### 6. Portal Apps Not Loading

**Symptom**: You're logged in to MyAtos, but apps/tiles don't load when clicked.

**Causes**:
- App is still loading (SAP Fiori apps can be slow)
- JavaScript error in the app
- Missing permissions for the app

**Solutions**:
1. Wait longer: `browser_wait(load_state="networkidle", timeout_ms=15000)`
2. Check browser console: `browser_get_logs()`
3. Try a different app to verify portal is working
4. Contact Atos IT if you don't have permission for an app

### 7. Cookies Not Saving

**Symptom**: `browser_save_cookies(name="atos-portal")` succeeds but cookies aren't restored next time.

**Causes**:
- Session file not found or corrupted
- Cookies expired between saves
- File system permissions issue

**Solutions**:
1. Check session file exists: `/data/sessions/atos-portal_session.json`
2. Verify file is readable and not corrupted
3. Re-authenticate and save cookies again
4. Use `browser_load_cookies()` with error handling

### 8. Network/Proxy Issues

**Symptom**: Can't reach `wac.das.myatos.net` or `nextgen.myatos.net`.

**Causes**:
- Corporate proxy blocking access
- VPN not connected
- DNS resolution issues
- Firewall rules

**Solutions**:
1. Verify you're on the corporate network or VPN
2. Check proxy settings in browser configuration
3. Try accessing the URL directly in a browser first
4. Contact Atos IT for network troubleshooting

## Debug Workflow

When troubleshooting, follow this workflow:

```python
# 1. Check current page state
page = browser_read_page()
print(f"URL: {page['url']}")
print(f"Title: {page['title']}")

# 2. Take screenshot for visual inspection
screenshot = browser_screenshot()

# 3. Check browser logs for errors
logs = browser_get_logs(last_n=50)
for log in logs:
    if log['level'] == 'error':
        print(f"Error: {log['message']}")

# 4. Check if element exists
try:
    browser_click(selector="button:has-text('AUTH')")
except Exception as e:
    print(f"Element not found: {e}")

# 5. Wait longer if needed
browser_wait(load_state="networkidle", timeout_ms=20000)
```

## Getting Help

If you can't resolve the issue:

1. **Collect diagnostic information**:
   - Screenshot of the error
   - Browser logs: `browser_get_logs()`
   - Current URL and page title
   - Steps to reproduce

2. **Contact Atos IT Support**:
   - Provide your DAS ID
   - Describe the issue
   - Include diagnostic information

3. **Check Atos Portal Status**:
   - Visit `https://wac.das.myatos.net/portal/auth.jsp` directly
   - Check if the portal is accessible
   - Look for maintenance messages

## Related Documentation

- `SKILL.md` - Complete skill documentation
- `selectors.json` - CSS selectors for form elements
- `example.py` - Example usage code
