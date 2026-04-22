#!/usr/bin/env python3
"""
Example: Navigate to Atos MyAtos Expense Claims Menu

This example demonstrates how to navigate to the Expense Claims (Notes de Frais)
menu in the Atos MyAtos portal, assuming the user is already authenticated.

Prerequisites:
- User must be authenticated on the Atos MyAtos portal
- Browser tools must be available
"""

from browser_tools import (
    browser_navigate,
    browser_click,
    browser_wait,
    browser_read_page,
)


def navigate_to_expense_claims():
    """
    Navigate to the Expense Claims menu in Atos MyAtos portal.
    
    Assumes user is already authenticated.
    
    Returns:
        dict: Page content and status information
    """
    
    print("Step 1: Navigate to MyAtos portal home page...")
    browser_navigate(url="https://nextgen.myatos.net/sap/flp")
    
    print("Step 2: Wait for page to load...")
    browser_wait(load_state="domcontentloaded", timeout_ms=5000)
    
    print("Step 3: Read page content to verify portal is loaded...")
    page_content = browser_read_page()
    print(f"Portal page loaded: {page_content.get('title', 'Unknown')}")
    
    # Verify we're on the portal home page
    if "MyAtos" not in page_content.get("title", "") and \
       "Page d'accueil" not in page_content.get("body", ""):
        print("⚠️  Warning: May not be on the correct portal page")
        print(f"Page title: {page_content.get('title')}")
    
    print("\nStep 4: Click on 'Note de frais' (Expense Claims) menu...")
    browser_click(selector='a:has-text("Note de frais")')
    
    print("Step 5: Wait for Expense Claims page to load...")
    browser_wait(load_state="networkidle", timeout_ms=5000)
    
    print("Step 6: Verify Expense Claims page is displayed...")
    expense_page = browser_read_page()
    
    # Verify we're on the Expense Claims page
    if "My expenses" in expense_page.get("body", "") or \
       "Poste de travail personne en déplacement" in expense_page.get("body", ""):
        print("✅ Successfully navigated to Expense Claims menu!")
        print(f"Page title: {expense_page.get('title')}")
        print(f"URL: {expense_page.get('url')}")
        return {
            "status": "success",
            "page_title": expense_page.get("title"),
            "url": expense_page.get("url"),
            "content": expense_page.get("body", "")[:500]  # First 500 chars
        }
    else:
        print("⚠️  Warning: Expense Claims page may not have loaded correctly")
        print(f"Page title: {expense_page.get('title')}")
        print(f"URL: {expense_page.get('url')}")
        return {
            "status": "warning",
            "page_title": expense_page.get("title"),
            "url": expense_page.get("url"),
            "message": "Page loaded but content may not be as expected"
        }


def navigate_to_expense_claims_with_error_handling():
    """
    Navigate to Expense Claims with comprehensive error handling.
    
    Returns:
        dict: Result with status and details
    """
    
    try:
        print("=" * 60)
        print("Atos MyAtos Expense Claims Navigation")
        print("=" * 60)
        
        # Step 1: Navigate to portal
        print("\n[1/6] Navigating to MyAtos portal...")
        browser_navigate(url="https://nextgen.myatos.net/sap/flp")
        
        # Step 2: Wait for DOM to load
        print("[2/6] Waiting for page to load...")
        browser_wait(load_state="domcontentloaded", timeout_ms=5000)
        
        # Step 3: Verify portal is accessible
        print("[3/6] Verifying portal access...")
        portal_page = browser_read_page()
        
        if "MyAtos" not in portal_page.get("title", ""):
            print("⚠️  Portal title unexpected. Checking page content...")
        
        # Step 4: Click Expense Claims menu
        print("[4/6] Clicking 'Note de frais' menu...")
        try:
            browser_click(selector='a:has-text("Note de frais")')
        except Exception as e:
            print(f"❌ Error clicking menu: {e}")
            print("Available page content:")
            print(portal_page.get("body", "")[:1000])
            return {
                "status": "error",
                "error": str(e),
                "message": "Could not find or click 'Note de frais' menu"
            }
        
        # Step 5: Wait for Expense Claims page
        print("[5/6] Waiting for Expense Claims page to load...")
        browser_wait(load_state="networkidle", timeout_ms=5000)
        
        # Step 6: Verify final page
        print("[6/6] Verifying Expense Claims page...")
        expense_page = browser_read_page()
        
        # Check if we're on the correct page
        is_correct_page = (
            "My expenses" in expense_page.get("body", "") or
            "Poste de travail personne en déplacement" in expense_page.get("body", "") or
            "Expense-claim" in expense_page.get("url", "")
        )
        
        if is_correct_page:
            print("\n✅ SUCCESS: Expense Claims menu opened!")
            print(f"   URL: {expense_page.get('url')}")
            print(f"   Title: {expense_page.get('title')}")
            return {
                "status": "success",
                "url": expense_page.get("url"),
                "title": expense_page.get("title"),
                "message": "Successfully navigated to Expense Claims"
            }
        else:
            print("\n⚠️  WARNING: Page loaded but may not be Expense Claims")
            print(f"   URL: {expense_page.get('url')}")
            print(f"   Title: {expense_page.get('title')}")
            return {
                "status": "warning",
                "url": expense_page.get("url"),
                "title": expense_page.get("title"),
                "message": "Page loaded but content verification inconclusive"
            }
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message": "Navigation failed"
        }


if __name__ == "__main__":
    # Run the navigation with error handling
    result = navigate_to_expense_claims_with_error_handling()
    
    print("\n" + "=" * 60)
    print("Result Summary:")
    print("=" * 60)
    for key, value in result.items():
        print(f"{key}: {value}")
