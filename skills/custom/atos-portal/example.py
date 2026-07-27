"""
Example: Accessing the Atos MyAtos Portal

This example demonstrates how to authenticate to the Atos MyAtos portal
and navigate to access employee information.
"""

from browser_tools import (
    browser_load_cookies,
    browser_navigate,
    browser_read_page,
    browser_fill_credential,
    browser_click,
    browser_wait,
    browser_save_cookies,
    browser_wait_for_user,
)


def login_to_atos_portal():
    """
    Authenticate to the Atos MyAtos portal using saved session or credentials.
    """
    
    # Step 1: Try to load saved session
    print("Attempting to load saved Atos session...")
    session_loaded = browser_load_cookies(name="atos-portal")
    
    if session_loaded:
        print("Session loaded successfully. Navigating to portal...")
        browser_navigate(url="https://nextgen.myatos.net/sap/flp")
        browser_wait(load_state="networkidle", timeout_ms=10000)
        page_content = browser_read_page()
        
        if "MyAtos" in page_content.get("title", ""):
            print("✓ Already authenticated!")
            return True
    
    # Step 2: Perform SSO login
    print("Session not available or expired. Performing login...")
    browser_navigate(url="https://wac.das.myatos.net/portal/auth.jsp")
    browser_wait(timeout_ms=3000)
    
    page_content = browser_read_page()
    print("Login page loaded. Filling credentials...")
    
    # Fill DAS ID
    browser_fill_credential(
        selector="input[placeholder*='DAS']",
        credential_env="ATOS_USERNAME"
    )
    
    # Fill Password
    browser_fill_credential(
        selector="input[type='password']",
        credential_env="ATOS_PASSWORD"
    )
    
    # Check if AUTH Code is required (2FA)
    if "AUTH Code" in page_content:
        print("⚠ 2FA AUTH Code required. Please enter it manually.")
        browser_wait_for_user(
            message="Please enter your AUTH Code and click AUTH to continue.",
            timeout_seconds=120
        )
    else:
        # Submit the form
        print("Submitting credentials...")
        browser_click(selector="button:has-text('AUTH')")
    
    # Step 3: Wait for redirect and page load
    print("Waiting for authentication and redirect...")
    browser_wait(load_state="networkidle", timeout_ms=15000)
    
    page_content = browser_read_page()
    
    if "MyAtos" in page_content.get("title", ""):
        print("✓ Authentication successful!")
        
        # Save session for future use
        print("Saving session...")
        browser_save_cookies(name="atos-portal")
        return True
    else:
        print("✗ Authentication failed. Please check your credentials.")
        return False


def navigate_to_app(app_name):
    """
    Navigate to a specific app in the MyAtos portal.
    
    Args:
        app_name: Name of the app to navigate to (e.g., "My Tasks")
    """
    print(f"Navigating to {app_name}...")
    
    # Read current page to find the app
    page_content = browser_read_page()
    
    # Click on the app
    browser_click(selector=f"a:has-text('{app_name}')")
    
    # Wait for app to load
    browser_wait(load_state="networkidle", timeout_ms=10000)
    
    page_content = browser_read_page()
    print(f"✓ Loaded {app_name}")
    
    return page_content


def main():
    """Main example workflow."""
    
    print("=" * 60)
    print("Atos MyAtos Portal Access Example")
    print("=" * 60)
    
    # Authenticate
    if not login_to_atos_portal():
        print("Failed to authenticate. Exiting.")
        return
    
    # Read portal content
    print("\nReading portal content...")
    page_content = browser_read_page()
    print(f"Portal Title: {page_content.get('title', 'N/A')}")
    print(f"Portal URL: {page_content.get('url', 'N/A')}")
    
    # Example: Navigate to My Tasks
    # tasks_content = navigate_to_app("My Tasks")
    # print(f"\nTasks content:\n{tasks_content}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
