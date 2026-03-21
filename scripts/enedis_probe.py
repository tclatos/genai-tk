#!/usr/bin/env python3
"""Enedis probe — compare CDP-attach vs fresh-browser modes against the Enedis login page.

Runs from the HOST against a live AIO sandbox container. Tests both browser
connection strategies to determine which (if either) reaches the Enedis
login form vs getting redirected to ``/indisponible``.

Prerequisites:
    - An AIO sandbox container running (``cli sandbox start`` or manual ``docker run``)
    - Host Python environment with ``playwright``, ``agent-sandbox``, ``httpx``

Usage:
    # Start a sandbox container (if not already running)
    docker run -d --name enedis-probe -p 8080:8080 ghcr.io/agent-infra/sandbox:latest
    sleep 10

    # Run the probe from the host
    python scripts/enedis_probe.py                       # default: localhost:8080
    python scripts/enedis_probe.py http://localhost:8080  # explicit URL

    # Cleanup
    docker rm -f enedis-probe
"""

from __future__ import annotations

import asyncio
import json
import sys


async def _probe_enedis(page: object) -> dict:
    """Navigate to Enedis login and capture key signals."""
    result: dict = {
        "final_url": "",
        "title": "",
        "has_email_field": False,
        "has_captcha": False,
        "body_text_snippet": "",
        "user_agent": "",
        "languages": "",
        "timezone": "",
        "webgl_renderer": "",
        "error": None,
    }

    try:
        # Browser fingerprint basics
        fp = await page.evaluate(
            """() => ({
            userAgent: navigator.userAgent,
            languages: JSON.stringify(navigator.languages),
            timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
        })"""
        )
        result["user_agent"] = fp.get("userAgent", "")
        result["languages"] = fp.get("languages", "")
        result["timezone"] = fp.get("timezone", "")

        # WebGL renderer
        result["webgl_renderer"] = await page.evaluate(
            """() => {
            try {
                const c = document.createElement('canvas');
                const gl = c.getContext('webgl') || c.getContext('experimental-webgl');
                if (!gl) return 'no-webgl';
                const ext = gl.getExtension('WEBGL_debug_renderer_info');
                return ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : 'no-ext';
            } catch(e) { return 'error: ' + e.message; }
        }"""
        )

        # Navigate to Enedis login
        await page.goto(
            "https://mon-compte-particulier.enedis.fr/auth/login",
            wait_until="domcontentloaded",
            timeout=30000,
        )

        # Dismiss cookie overlay via JS
        await page.evaluate(
            """() => {
            document.querySelector('#popin_tc_privacy')?.remove();
            document.querySelector('#popin_tc_privacy_container')?.remove();
            document.querySelector('.tc-privacy-wrapper')?.remove();
            document.body.style.overflow = 'auto';
        }"""
        )

        try:
            await page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass

        await asyncio.sleep(3)

        result["final_url"] = page.url
        result["title"] = await page.title()
        email_field = await page.query_selector("input[type='email'], input[name='username'], #username")
        result["has_email_field"] = email_field is not None
        body_text = await page.evaluate("() => document.body?.innerText?.slice(0, 2000) ?? ''")
        result["body_text_snippet"] = body_text
        result["has_captcha"] = "captcha" in body_text.lower()

    except Exception as exc:
        result["error"] = str(exc)

    return result


async def probe_cdp_mode(sandbox_url: str) -> dict:
    """Test the current CDP-attach mode (connect to pre-launched browser)."""
    from agent_sandbox import Sandbox
    from playwright.async_api import async_playwright

    client = Sandbox(base_url=sandbox_url)
    browser_info = client.browser.get_info().data
    cdp_url = browser_info.cdp_url

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(cdp_url)
        contexts = browser.contexts
        if contexts:
            context = contexts[0]
            page = context.pages[0] if context.pages else await context.new_page()
        else:
            context = await browser.new_context(
                locale="fr-FR",
                timezone_id="Europe/Paris",
                viewport={"width": 1920, "height": 1080},
                ignore_https_errors=True,
            )
            page = await context.new_page()

        result = await _probe_enedis(page)
        await browser.close()

    return result


async def probe_fresh_mode(sandbox_url: str) -> dict:
    """Test the fresh-browser mode (kill pre-launched, launch new, reconnect)."""
    import httpx
    from agent_sandbox import AsyncSandbox, Sandbox
    from playwright.async_api import async_playwright

    # Discover the CDP port the pre-launched browser uses so the proxy keeps routing
    try:
        sync_client = Sandbox(base_url=sandbox_url)
        browser_info = sync_client.browser.get_info().data
        from urllib.parse import urlparse

        cdp_port = urlparse(browser_info.cdp_url).port or 9222
    except Exception:
        cdp_port = 9222

    client = AsyncSandbox(base_url=sandbox_url)

    # Kill the pre-launched Chromium
    await client.shell.exec_command(command="pkill -f chromium || pkill -f chrome || true")
    await asyncio.sleep(2)

    # Launch fresh Chromium with anti-detection flags on the same port
    launch_cmd = (
        "nohup /usr/bin/chromium-browser"
        " --no-first-run --no-default-browser-check"
        " --disable-blink-features=AutomationControlled"
        " --lang=fr-FR"
        " --time-zone-for-testing=Europe/Paris"
        " --remote-debugging-address=0.0.0.0"
        f" --remote-debugging-port={cdp_port}"
        " --headless=new"
        " --window-size=1920,1080"
        " about:blank"
        " > /tmp/chromium-fresh.log 2>&1 &"
    )
    await client.shell.exec_command(command=launch_cmd)

    # Wait for fresh browser CDP endpoint
    cdp_url = None
    for _ in range(15):
        await asyncio.sleep(1)
        try:
            async with httpx.AsyncClient(trust_env=False) as hc:
                resp = await hc.get(f"{sandbox_url}/cdp/json/version", timeout=3.0)
                if resp.status_code == 200:
                    cdp_url = resp.json().get("webSocketDebuggerUrl", "")
                    if cdp_url:
                        break
        except Exception:
            continue

    if not cdp_url:
        return {"error": "Fresh Chromium did not start within 15s"}

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(cdp_url)
        context = await browser.new_context(
            locale="fr-FR",
            timezone_id="Europe/Paris",
            viewport={"width": 1920, "height": 1080},
            ignore_https_errors=True,
        )
        page = await context.new_page()
        result = await _probe_enedis(page)
        await context.close()
        await browser.close()

    return result


def _verdict(result: dict) -> str:
    url = result.get("final_url", "")
    if result.get("error"):
        return f"ERROR: {result['error']}"
    if "/indisponible" in url:
        return "BLOCKED → /indisponible"
    if result.get("has_email_field"):
        return "SUCCESS → login form reached"
    if result.get("has_captcha"):
        return "CAPTCHA → FriendlyCaptcha detected"
    return f"UNCLEAR → {url}"


async def main(sandbox_url: str) -> None:
    print(f"=== Enedis Probe — sandbox at {sandbox_url} ===\n")

    # --- Test 1: CDP mode (pre-launched browser) ---
    print("▶ Mode 1: CDP attach (pre-launched browser)")
    cdp_result = await probe_cdp_mode(sandbox_url)
    print(f"  Verdict: {_verdict(cdp_result)}")
    print(f"  URL: {cdp_result.get('final_url', 'N/A')}")
    print(f"  UA: {cdp_result.get('user_agent', 'N/A')[:80]}")
    print(f"  WebGL: {cdp_result.get('webgl_renderer', 'N/A')}")
    print()

    # --- Test 2: Fresh mode (kill + relaunch) ---
    print("▶ Mode 2: Fresh browser (kill + relaunch inside container)")
    fresh_result = await probe_fresh_mode(sandbox_url)
    print(f"  Verdict: {_verdict(fresh_result)}")
    print(f"  URL: {fresh_result.get('final_url', 'N/A')}")
    print(f"  UA: {fresh_result.get('user_agent', 'N/A')[:80]}")
    print(f"  WebGL: {fresh_result.get('webgl_renderer', 'N/A')}")
    print()

    # --- Full JSON ---
    print("=== Full Results (JSON) ===")
    print(json.dumps({"cdp_mode": cdp_result, "fresh_mode": fresh_result}, indent=2, ensure_ascii=False))

    # Exit code
    cdp_ok = cdp_result.get("has_email_field") or cdp_result.get("has_captcha")
    fresh_ok = fresh_result.get("has_email_field") or fresh_result.get("has_captcha")
    if fresh_ok and not cdp_ok:
        print("\n✅ Fresh mode works, CDP mode blocked — use launch_mode: fresh", file=sys.stderr)
    elif fresh_ok and cdp_ok:
        print("\n✅ Both modes work", file=sys.stderr)
    elif not fresh_ok and not cdp_ok:
        print("\n❌ Both modes blocked — deeper investigation needed", file=sys.stderr)
        sys.exit(1)
    else:
        print("\n⚠️  CDP works but fresh doesn't — unexpected", file=sys.stderr)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    asyncio.run(main(url))
