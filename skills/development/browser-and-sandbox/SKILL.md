---
name: browser-and-sandbox
description: Work on browser automation, sandbox browser tools, direct Playwright tools, AioSandbox backend, and sandbox CLI support in genai-tk.
---

# GenAI Toolkit Browser And Sandbox

## Read First

- `docs/browser_control.md`
- `docs/sandbox_support.md`
- `docs/design/06_sandbox_and_codeact.md`
- `genai_tk/agents/tools/sandbox_browser/`
- `genai_tk/agents/tools/direct_browser/`
- `genai_tk/agents/sandbox/`

## Modes

| Mode | Profile | Code | Use for |
|---|---|---|---|
| Sandbox | Browser Agent | `genai_tk/agents/tools/sandbox_browser/`, `genai_tk/agents/sandbox/aio_backend.py` | Isolated browser runs in Docker/OpenSandbox |
| Direct | Browser Agent Direct | `genai_tk/agents/tools/direct_browser/` | Bot-sensitive sites needing host Chromium |

Both modes should expose the same tool names so site skills under `skills/custom/` work unchanged.

## Python Executor Trust Boundary

- The Python tool defaults to `docker`; `auto` and `sandbox` require isolation
  too. Only explicit `local` permits trusted host execution. A sandbox failure
  must never cause host fallback.
- A default Python executor owns a dedicated backend, omits configured
  environment variables/volumes, and does not implicitly acquire the shared
  `SandboxManager` backend. Explicit `backend=` can grant those capabilities
  within a caller-owned trust boundary; it is not per-worker tenant isolation.
- The worker relays allowlisted host tool calls over the existing sandbox
  command transport as bounded JSON. Do not add public host RPC listeners or
  deserialize worker-controlled pickle data.
- AIO CPU/memory limits are container settings. The generated local server also
  applies process and privilege restrictions; external servers must enforce
  equivalent policies. Network egress remains deployment-dependent.
- A timeout or infrastructure failure invalidates the executor and retires its
  owned container. For a supplied backend, the caller must retire the runtime
  after timeout. Use `aclose()` for async cleanup and surface termination failures.
  Never retry a host tool with an uncertain outcome automatically.
- Generic AIO command deadlines retire the backend on timeout or cancellation;
  do not replace this with a client-only timeout that leaves code running remotely.
- AIO `max_command_response_bytes` defaults to 16 MiB for the entire command
  response, including SSE framing. Enforce it in the host HTTP byte stream before
  SDK framing, reject compressed command bodies, close aborted responses, and
  retire the backend on violations. Worker-side limits or SDK output callbacks
  alone cannot bound an unterminated SSE frame. File-transfer responses are outside
  this limit. Configurable SDK transports require `opensandbox>=1.1.0`; fail closed
  with upgrade guidance if unavailable.
- See [the Python interpreter skill](../python-interpreter/SKILL.md) and
  [the sandbox design](../../../docs/design/06_sandbox_and_codeact.md).

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
uv sync --extra harnessing --extra browser
uv run playwright install chromium
uv run cli sandbox start
uv run cli sandbox pull
GENAITK_PROFILE=pytest uv run pytest tests/unit_tests/tools/sandbox_browser -q
```

## Avoid

- Do not solve or bypass CAPTCHAs.
- Do not log credentials or credential environment values.
- Do not make sandbox-only assumptions in generic site skills.
