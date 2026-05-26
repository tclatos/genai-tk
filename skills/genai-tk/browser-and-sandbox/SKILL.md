---
name: genai-tk-browser-and-sandbox
description: Work on browser automation, sandbox browser tools, direct Playwright tools, AioSandbox backend, and sandbox CLI support in genai-tk.
---

# GenAI Toolkit Browser And Sandbox

## Read First

- `docs/browser_control.md`
- `docs/sandbox_support.md`
- `docs/design/sandbox_backend.md`
- `genai_tk/agents/tools/sandbox_browser/`
- `genai_tk/agents/tools/direct_browser/`
- `genai_tk/agents/sandbox/`

## Modes

| Mode | Profile | Code | Use for |
|---|---|---|---|
| Sandbox | Browser Agent | `genai_tk/agents/tools/sandbox_browser/`, `genai_tk/agents/sandbox/aio_backend.py` | Isolated browser runs in Docker/OpenSandbox |
| Direct | Browser Agent Direct | `genai_tk/agents/tools/direct_browser/` | Bot-sensitive sites needing host Chromium |

Both modes should expose the same tool names so site skills under `skills/custom/` work unchanged.

## Tool Contract

Keep these names stable unless docs and skills are updated together:

- `browser_navigate`
- `browser_click`
- `browser_type`
- `browser_fill_credential`
- `browser_screenshot`
- `browser_read_page`
- `browser_scroll`
- `browser_wait`
- `browser_save_cookies`
- `browser_load_cookies`
- `browser_get_logs`
- `browser_evaluate`
- `browser_diagnose`

## Change Workflow

1. Decide if the change applies to sandbox, direct, or both.
2. Preserve identical tool names and similar return shapes across both backends.
3. Use `browser_fill_credential` for secrets; never expose credential values to the model.
4. Update `skills/custom/browser-automation/SKILL.md` when behavior or tool names change.
5. Prefer mocked/unit tests for tool plumbing and mark real browser tests as integration.

## Commands

```bash
uv sync --group browser-control
uv run playwright install chromium
uv run cli sandbox start
uv run cli sandbox pull
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/tools/sandbox_browser -q
```

## Avoid

- Do not solve or bypass CAPTCHAs.
- Do not log credentials or credential environment values.
- Do not make sandbox-only assumptions in generic site skills.
