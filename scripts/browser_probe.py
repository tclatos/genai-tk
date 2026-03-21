#!/usr/bin/env python3
"""Browser probe: compare sandbox, direct Playwright, and fingerprint diagnostics.

Tests multiple browser access paths against a target URL and collects browser
fingerprint signals to identify bot-detection triggers.

Usage:
    # Test all modes against a target URL
    uv run python scripts/browser_probe.py https://mon-compte-particulier.enedis.fr/auth/login

    # Quick fingerprint comparison (uses bot.sannysoft.com)
    uv run python scripts/browser_probe.py --fingerprint-only

    # Test direct mode only
    uv run python scripts/browser_probe.py --mode direct https://example.com

    # Test sandbox mode (requires running AIO container)
    uv run python scripts/browser_probe.py --mode sandbox https://example.com

Modes:
    direct      — Host-local Playwright (real GPU, real platform)
    sandbox-cdp — AIO sandbox pre-launched browser via CDP
    sandbox-fresh — AIO sandbox fresh Chromium launch
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Fingerprint probes (JavaScript snippets)
# ---------------------------------------------------------------------------

FINGERPRINT_JS = """
(() => {
    const fp = {};
    fp.userAgent = navigator.userAgent;
    fp.platform = navigator.platform;
    fp.languages = navigator.languages;
    fp.language = navigator.language;
    fp.hardwareConcurrency = navigator.hardwareConcurrency;
    fp.deviceMemory = navigator.deviceMemory || 'N/A';
    fp.webdriver = navigator.webdriver;
    fp.cookieEnabled = navigator.cookieEnabled;
    fp.doNotTrack = navigator.doNotTrack;
    fp.maxTouchPoints = navigator.maxTouchPoints;
    fp.timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    fp.timezoneOffset = new Date().getTimezoneOffset();

    // WebGL
    try {
        const canvas = document.createElement('canvas');
        const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (gl) {
            const debugExt = gl.getExtension('WEBGL_debug_renderer_info');
            fp.webglVendor = debugExt ? gl.getParameter(debugExt.UNMASKED_VENDOR_WEBGL) : 'N/A';
            fp.webglRenderer = debugExt ? gl.getParameter(debugExt.UNMASKED_RENDERER_WEBGL) : 'N/A';
        }
    } catch(e) { fp.webglVendor = 'error'; fp.webglRenderer = 'error'; }

    // Screen
    fp.screenWidth = screen.width;
    fp.screenHeight = screen.height;
    fp.colorDepth = screen.colorDepth;

    // High-entropy UA client hints (async, returns promise)
    fp.uaDataBrands = navigator.userAgentData?.brands?.map(b => b.brand + '/' + b.version) || [];
    fp.uaDataMobile = navigator.userAgentData?.mobile;
    fp.uaDataPlatform = navigator.userAgentData?.platform || 'N/A';

    return fp;
})()
"""


@dataclass
class ProbeResult:
    mode: str
    url: str = ""
    final_url: str = ""
    title: str = ""
    has_login_form: bool = False
    has_captcha: bool = False
    is_blocked: bool = False
    fingerprint: dict = field(default_factory=dict)
    error: str | None = None
    body_snippet: str = ""


async def probe_direct(target_url: str, headless: bool = True) -> ProbeResult:
    """Probe using host-local Playwright (direct mode)."""
    from playwright.async_api import async_playwright

    result = ProbeResult(mode="direct", url=target_url)

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=headless,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--lang=fr-FR",
                    "--window-size=1920,1080",
                ],
            )
            context = await browser.new_context(
                locale="fr-FR",
                timezone_id="Europe/Paris",
                viewport={"width": 1920, "height": 1080},
                ignore_https_errors=True,
            )
            page = await context.new_page()

            await page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=15000)

            result.final_url = page.url
            result.title = await page.title()

            # Fingerprint
            result.fingerprint = await page.evaluate(FINGERPRINT_JS)

            # Check page state
            body = await page.inner_text("body")
            result.body_snippet = body[:500]
            result.is_blocked = "/indisponible" in page.url or "indisponible" in body.lower()
            result.has_login_form = bool(
                await page.query_selector("input[type='email'], input[type='password'], input[name='username']")
            )
            result.has_captcha = "captcha" in body.lower() or "friendlycaptcha" in body.lower()

            await browser.close()
    except Exception as exc:
        result.error = str(exc)

    return result


async def probe_sandbox_cdp(sandbox_url: str, target_url: str) -> ProbeResult:
    """Probe using AIO sandbox CDP-attach mode."""
    from agent_sandbox import Sandbox
    from playwright.async_api import async_playwright

    result = ProbeResult(mode="sandbox-cdp", url=target_url)

    try:
        client = Sandbox(base_url=sandbox_url)
        browser_info = client.browser.get_info().data
        cdp_url = browser_info.cdp_url
        print(f"  CDP URL: {cdp_url}")

        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(cdp_url)
            contexts = browser.contexts
            if contexts:
                context = contexts[0]
                pages = context.pages
                page = pages[0] if pages else await context.new_page()
            else:
                context = await browser.new_context(locale="fr-FR", timezone_id="Europe/Paris")
                page = await context.new_page()

            await page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=15000)

            result.final_url = page.url
            result.title = await page.title()
            result.fingerprint = await page.evaluate(FINGERPRINT_JS)

            body = await page.inner_text("body")
            result.body_snippet = body[:500]
            result.is_blocked = "/indisponible" in page.url or "indisponible" in body.lower()
            result.has_login_form = bool(
                await page.query_selector("input[type='email'], input[type='password'], input[name='username']")
            )
            result.has_captcha = "captcha" in body.lower() or "friendlycaptcha" in body.lower()

            await browser.close()
    except Exception as exc:
        result.error = str(exc)

    return result


async def probe_sandbox_fresh(sandbox_url: str, target_url: str) -> ProbeResult:
    """Probe using AIO sandbox fresh-browser mode."""
    from urllib.parse import urlparse

    from agent_sandbox import AsyncSandbox, Sandbox
    from playwright.async_api import async_playwright

    result = ProbeResult(mode="sandbox-fresh", url=target_url)

    try:
        sync_client = Sandbox(base_url=sandbox_url)
        try:
            browser_info = sync_client.browser.get_info().data
            original_cdp_url = browser_info.cdp_url
            parsed = urlparse(original_cdp_url)
            cdp_port = parsed.port or 9222
        except Exception:
            cdp_port = 9222

        client = AsyncSandbox(base_url=sandbox_url)

        # Kill pre-launched browser
        await client.shell.exec_command(command="pkill -f chromium || pkill -f chrome || true")
        await asyncio.sleep(2)

        # Launch fresh Chromium
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

        # Wait for CDP
        import httpx

        cdp_ws_url = None
        for _ in range(15):
            await asyncio.sleep(1)
            try:
                async with httpx.AsyncClient(trust_env=False) as hc:
                    resp = await hc.get(f"{sandbox_url}/cdp/json/version", timeout=3.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        cdp_ws_url = data.get("webSocketDebuggerUrl", "")
                        if cdp_ws_url:
                            break
            except Exception:
                continue

        if not cdp_ws_url:
            result.error = "Fresh Chromium did not start within 15s"
            return result

        async with async_playwright() as pw:
            browser = await pw.chromium.connect_over_cdp(cdp_ws_url)
            context = await browser.new_context(
                locale="fr-FR",
                timezone_id="Europe/Paris",
                viewport={"width": 1920, "height": 1080},
                ignore_https_errors=True,
            )
            page = await context.new_page()

            await page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_load_state("networkidle", timeout=15000)

            result.final_url = page.url
            result.title = await page.title()
            result.fingerprint = await page.evaluate(FINGERPRINT_JS)

            body = await page.inner_text("body")
            result.body_snippet = body[:500]
            result.is_blocked = "/indisponible" in page.url or "indisponible" in body.lower()
            result.has_login_form = bool(
                await page.query_selector("input[type='email'], input[type='password'], input[name='username']")
            )
            result.has_captcha = "captcha" in body.lower() or "friendlycaptcha" in body.lower()

            await browser.close()
    except Exception as exc:
        result.error = str(exc)

    return result


def _verdict(r: ProbeResult) -> str:
    if r.error:
        return f"ERROR: {r.error[:100]}"
    if r.is_blocked:
        return "BLOCKED → /indisponible"
    if r.has_captcha:
        return "CAPTCHA → solve required"
    if r.has_login_form:
        return "SUCCESS → login form reached"
    return f"UNKNOWN → {r.final_url}"


def _print_result(r: ProbeResult) -> None:
    print(f"\n{'=' * 60}")
    print(f"Mode: {r.mode}")
    print(f"Verdict: {_verdict(r)}")
    print(f"URL: {r.url} → {r.final_url}")
    print(f"Title: {r.title}")
    if r.fingerprint:
        fp = r.fingerprint
        print(f"  User-Agent: {fp.get('userAgent', 'N/A')[:80]}")
        print(f"  Platform: {fp.get('platform', 'N/A')}")
        print(f"  UA Data Platform: {fp.get('uaDataPlatform', 'N/A')}")
        print(f"  Languages: {fp.get('languages', 'N/A')}")
        print(f"  Timezone: {fp.get('timezone', 'N/A')}")
        print(f"  WebDriver: {fp.get('webdriver', 'N/A')}")
        print(f"  WebGL Vendor: {fp.get('webglVendor', 'N/A')}")
        print(f"  WebGL Renderer: {fp.get('webglRenderer', 'N/A')}")
        print(f"  Screen: {fp.get('screenWidth')}x{fp.get('screenHeight')} ({fp.get('colorDepth')}bit)")
    if r.error:
        print(f"  Error: {r.error[:200]}")
    print(f"  Body snippet: {r.body_snippet[:200]}")


def _print_comparison(results: list[ProbeResult]) -> None:
    print(f"\n{'=' * 60}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 60}")
    headers = ["Signal"] + [r.mode for r in results]
    rows = [
        ["Verdict"] + [_verdict(r) for r in results],
        ["WebGL Renderer"] + [r.fingerprint.get("webglRenderer", "N/A")[:30] for r in results],
        ["Platform"] + [r.fingerprint.get("platform", "N/A") for r in results],
        ["UA Platform"] + [r.fingerprint.get("uaDataPlatform", "N/A") for r in results],
        ["WebDriver"] + [str(r.fingerprint.get("webdriver", "N/A")) for r in results],
        ["Timezone"] + [r.fingerprint.get("timezone", "N/A") for r in results],
        ["Languages"] + [str(r.fingerprint.get("languages", "N/A")) for r in results],
    ]

    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(fmt.format(*row))


async def main() -> None:
    parser = argparse.ArgumentParser(description="Browser probe: compare access modes and fingerprints")
    parser.add_argument(
        "url", nargs="?", default="https://mon-compte-particulier.enedis.fr/auth/login", help="Target URL"
    )
    parser.add_argument("--mode", choices=["direct", "sandbox-cdp", "sandbox-fresh", "all"], default="all")
    parser.add_argument("--sandbox-url", default="http://localhost:8080", help="AIO sandbox URL")
    parser.add_argument("--fingerprint-only", action="store_true", help="Use bot.sannysoft.com instead of target URL")
    parser.add_argument("--headless", action="store_true", default=True, help="Run direct mode headless")
    parser.add_argument("--headed", action="store_true", help="Run direct mode headed")
    args = parser.parse_args()

    target_url = "https://bot.sannysoft.com/" if args.fingerprint_only else args.url
    headless = not args.headed

    results: list[ProbeResult] = []

    if args.mode in ("direct", "all"):
        print(f"\n--- Probing DIRECT mode against {target_url} ---")
        r = await probe_direct(target_url, headless=headless)
        _print_result(r)
        results.append(r)

    if args.mode in ("sandbox-cdp", "all"):
        print(f"\n--- Probing SANDBOX CDP mode against {target_url} ---")
        r = await probe_sandbox_cdp(args.sandbox_url, target_url)
        _print_result(r)
        results.append(r)

    if args.mode in ("sandbox-fresh", "all"):
        print(f"\n--- Probing SANDBOX FRESH mode against {target_url} ---")
        r = await probe_sandbox_fresh(args.sandbox_url, target_url)
        _print_result(r)
        results.append(r)

    if len(results) > 1:
        _print_comparison(results)

    # Write JSON results
    json_results = []
    for r in results:
        json_results.append(
            {
                "mode": r.mode,
                "verdict": _verdict(r),
                "url": r.url,
                "final_url": r.final_url,
                "title": r.title,
                "fingerprint": r.fingerprint,
                "is_blocked": r.is_blocked,
                "has_login_form": r.has_login_form,
                "has_captcha": r.has_captcha,
                "error": r.error,
            }
        )
    print("\n--- JSON results ---")
    print(json.dumps(json_results, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())
