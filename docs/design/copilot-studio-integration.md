# LangChain / DeepAgent Integration with Microsoft Copilot Studio

> **Terminology note**
> - **Microsoft Copilot Studio** — Microsoft's low-code platform (part of Microsoft 365) for building, configuring, and publishing AI agents to Teams, M365 apps, and other channels. Formerly Power Virtual Agents.
> - **GitHub Copilot** — The AI coding assistant in VS Code. Unrelated to this document.
> - **Microsoft 365 Agents SDK** — The Python SDK for building bots/agents that connect to Microsoft Copilot Studio, Teams, and other M365 channels via Azure Bot Service.

---

## Overview

This document describes how to expose LangChain/DeepAgent-based agents (with Skills, Tools, and shell command execution) as a bot registered in **Microsoft Copilot Studio**, using the **Microsoft 365 Agents SDK for Python**. Streamlit remains a parallel direct-call UI to the same agent backend.

---

## Architecture

```
                     ┌─────────────────────────────────────────┐
                     │          Python Backend                  │
                     │                                          │
  Copilot Studio ←→ Azure Bot Service ←→  M365 Agent endpoint  │
  (Teams, M365)       (channel routing)   (M365 Agents SDK)    │
                                                 ↓             │
  Streamlit UI ───────────────────────→  LangChain Agent       │
  (direct in-process)                     (DeepAgent/ReAct)    │
                                           ↓    ↓    ↓         │
                                         Tools Skills Shell    │
                                          (genai-tk sandbox)   │
                     └─────────────────────────────────────────┘
```

The Streamlit pages in `genai_blueprint/webapp/pages/demos/` continue to call LangChain agents **directly in-process** — no M365 Agents SDK layer involved. Microsoft Copilot Studio is an additional channel.

---

## Integration Models

### Model A — LangChain as Orchestrator (Recommended)

LangChain handles **all** orchestration. Microsoft Copilot Studio acts solely as a user-facing channel and conversation router.

- Your DeepAgent retains full control of planning, tool selection, skill loading, and shell execution
- Copilot Studio's built-in generative AI orchestration is bypassed
- Users interact via Teams, M365 Chat, or any connected channel

### Model B — Copilot Studio as Orchestrator (Actions)

Copilot Studio's built-in AI orchestrates and calls your LangChain capabilities as **OpenAPI Actions**.

- **Not recommended** for DeepAgent use cases — Copilot Studio's orchestrator overrides your agent's planning loop
- Suitable only for exposing discrete, stateless tool endpoints

---

## Implementation

### 1. Microsoft 365 Agents SDK for Python

The SDK is at [github.com/microsoft/agents-sdk-python](https://github.com/microsoft/agents-sdk-python) (preview as of 2026).

```bash
uv add microsoft-agents-hosting-aiohttp microsoft-agents-bot-builder
```

Add to `pyproject.toml`:
```toml
[project.optional-dependencies]
m365-agent = [
    "microsoft-agents-hosting-aiohttp",
    "microsoft-agents-bot-builder",
]
```

### 2. Agent Handler

> **Design note:** the agent definition (compiled graph, tools, profile) is
> separated from per-session state.  Rebuilding the full agent on every turn
> adds latency and discards caches.  The pattern below caches the compiled
> agent per profile key and injects only the session-scoped state at invocation
> time.

```python
# genai_blueprint/main/m365_agent.py
from __future__ import annotations

from functools import lru_cache

from microsoft.agents.bot_builder import ActivityHandler, TurnContext, MessageFactory
from microsoft.agents.core.models import ChannelAccount

from genai_tk.agents.langchain.config import resolve_profile
from genai_tk.config_mgmt.config_mngr import global_config
from genai_blueprint.main.agent_factory import build_agent  # your existing factory


@lru_cache(maxsize=16)
def _cached_agent(profile_key: str):
    """Return a compiled agent for the given profile, cached for the process lifetime.

    Only the agent *definition* is cached — session state is injected per turn.
    """
    config = global_config()
    profile = resolve_profile(config, profile_key)
    return build_agent(profile)


def _session_id(turn_context: TurnContext) -> str:
    """Composite session key: tenant + AAD user + conversation.

    Using just conversation.id risks collisions when conversation IDs are
    recycled across tenants or when the same user opens the bot on a new device.
    """
    activity = turn_context.activity
    tenant = (activity.channel_data or {}).get("tenant", {}).get("id", "unknown")
    user_id = activity.from_property.id if activity.from_property else "anon"
    conv_id = activity.conversation.id
    return f"{tenant}:{user_id}:{conv_id}"


class LangChainAgent(ActivityHandler):
    """M365 Agents SDK handler that delegates to the LangChain DeepAgent."""

    async def on_message_activity(self, turn_context: TurnContext) -> None:
        user_input = turn_context.activity.text or ""
        if not user_input.strip():
            return

        config = global_config()
        profile_key = config.get("m365_agent.agent_profile", "research")
        agent = _cached_agent(profile_key)
        session_id = _session_id(turn_context)

        result = await agent.ainvoke(
            {"input": user_input},
            config={
                "configurable": {
                    "session_id": session_id,
                    # propagate AAD identity for tool-level authz
                    "user_id": turn_context.activity.from_property.id if turn_context.activity.from_property else None,
                }
            },
        )
        await turn_context.send_activity(
            MessageFactory.text(result.get("output", ""))
        )

    async def on_members_added_activity(
        self, members_added: list[ChannelAccount], turn_context: TurnContext
    ) -> None:
        for member in members_added:
            if member.id != turn_context.activity.recipient.id:
                await turn_context.send_activity(
                    MessageFactory.text("Hello! How can I help you today?")
                )
```

### 3. Bot Endpoint (aiohttp)

```python
# genai_blueprint/main/m365_app.py
from aiohttp import web
from microsoft.agents.hosting.aiohttp import CloudAdapter, Request, Response
from microsoft.agents.core import ConfigurationBotFrameworkAuthentication

from genai_tk.config_mgmt.config_mngr import global_config
from genai_blueprint.main.m365_agent import LangChainAgent


def _make_adapter() -> CloudAdapter:
    config = global_config()
    auth_config = ConfigurationBotFrameworkAuthentication(
        MicrosoftAppId=config.get("m365_agent.app_id", ""),
        MicrosoftAppPassword=config.get("m365_agent.app_password", ""),
    )
    return CloudAdapter(auth_config)


_adapter = _make_adapter()
_agent = LangChainAgent()


async def messages(req: web.Request) -> web.Response:
    return await _adapter.process(Request(req), _agent, Response())


app = web.Application()
app.router.add_post("/api/messages", messages)
```

### 4. Conversation Memory / Session Keying

Use the **composite key** `{tenant}:{user_id}:{conversation_id}` (computed by
`_session_id()` above) rather than `conversation.id` alone.  A bare
conversation ID can collide across tenants and breaks multi-device continuity.

```python
from langchain_community.chat_message_histories import SQLChatMessageHistory

history = SQLChatMessageHistory(
    session_id=session_id,   # composite — see _session_id() helper
    connection_string="sqlite:///data/sessions/m365_sessions.db",
)
```

**Memory key dimensions:**

| Dimension | Source field | Why it matters |
|---|---|---|
| Tenant | `activity.channel_data.tenant.id` | Prevents cross-tenant leakage |
| AAD User | `activity.from_property.id` | Per-user personalization |
| Conversation | `activity.conversation.id` | Thread isolation |

For persistent cross-device memory, store history in a shared backend
(Postgres / Redis) keyed by `{tenant}:{user_id}` without the conversation
suffix, and fetch it as read-only context on each new conversation.

### 5. Configuration

Add to `config/app_conf.yaml`:

```yaml
m365_agent:
  app_id: ${oc.env:M365_BOT_APP_ID,}
  app_password: ${oc.env:M365_BOT_APP_PASSWORD,}
  agent_profile: research   # default genai-tk agent profile key
```

### 6. Azure Bot Resource Registration

1. Create an **Azure Bot** resource in the Azure Portal (free F0 tier for development)
2. Under **Configuration**, set the **Messaging endpoint**:
   - Production: `https://<your-host>/api/messages`
   - Local dev: use `ngrok http 8000` and set the ngrok HTTPS URL
3. Copy the **App ID** and generate a **Client Secret** (App Password) — store them in `.env`, never in YAML committed to source control:
   ```bash
   export M365_BOT_APP_ID=<your-app-id>
   export M365_BOT_APP_PASSWORD=<your-client-secret>
   ```

### 7. Connect to Microsoft Copilot Studio

1. Open **Microsoft Copilot Studio** → create or open an agent
2. Go to **Settings → Channels → Azure Bot Service**
3. Provide the Azure Bot resource App ID and connection details
4. Under the agent's **Generative AI** settings, configure fallback to your registered Azure Bot so Copilot Studio routes to your LangChain agent
5. Publish the agent to Teams or other M365 channels

---

## Compatibility Matrix

| Feature | Status | Notes |
|---|---|---|
| LangChain Tools (`@tool`) | ✅ Full | Executed server-side; invisible to Copilot Studio |
| SKILL.md / Skills system | ✅ Full | Agent reads skills during planning; not exposed to M365 |
| Shell / sandbox (`genai-tk sandbox`) | ✅ Full | Runs in Docker on your server |
| DeepAgent / ReAct planning loop | ✅ Full | Model A only — Copilot Studio orchestration bypassed |
| Streamlit UI | ✅ Full | Direct in-process calls; no M365 SDK layer needed |
| Streaming responses | ⚠️ **Critical** | Without streaming, UX latency is visible on DeepAgent calls; SDK supports activity streaming — see section 5a |
| Conversation memory | ⚠️ Explicit | Key on `conversation.id` — see section 4 above |
| File attachments | ⚠️ Manual | Signed URLs expire; validate, download, and preprocess before passing to agent — see section 5b |
| Copilot Studio topics/dialogs | ❌ Bypassed | Intentional — LangChain handles all orchestration |

---

## Security Considerations

- **App Password (client secret)** — store exclusively in environment variables; never commit to source control
- All shell commands and tool executions run on **your server** inside the genai-tk sandbox — Microsoft Copilot Studio never has access to the execution environment
- Request authentication is validated automatically by `CloudAdapter.process()` using Azure Bot Service tokens
- Use HTTPS for the `/api/messages` endpoint — Azure Bot Service requires it in production

### User Identity Propagation and RBAC

The AAD user identity arriving in the activity must flow into the agent so that
tools can enforce per-user authorization:

```python
# In agent tool implementation — example guard
from genai_tk.utils.basic_auth import AuthConfig

def _require_role(user_id: str, required_role: str) -> None:
    """Raise PermissionError if user does not hold required_role."""
    # Integrate with your AAD group / role resolution here
    allowed = resolve_user_roles(user_id)  # returns set[str]
    if required_role not in allowed:
        raise PermissionError(f"User {user_id!r} lacks role {required_role!r}")
```

**Checklist:**

| Control | Implementation |
|---|---|
| Identity source | `activity.from_property.id` (AAD OID) |
| Propagation | via `configurable["user_id"]` in `agent.ainvoke()` |
| Tool-level authz | check roles before executing sensitive tools (shell, file write) |
| Audit log | record `{user_id, tool_name, inputs, timestamp}` for every tool call |
| Secret isolation | sandbox never exposes host credentials to model output |

> **Risk without this:** a user can request `"run shell command"` and the
> agent executes it without any user-level gate.  Tool-level checks are the
> last line of defence after the model.

---

## Streaming Strategy

Without streaming, users see a blank window until the DeepAgent finishes
planning and executing — this feels slower than native Copilot.  The M365
Agents SDK supports activity streaming; the simplest approach is **chunked
text activities**:

```python
# Streaming with incremental activity updates (pseudo-code)
async for chunk in agent.astream({"input": user_input}, config=run_config):
    if text := chunk.get("output_chunk"):
        await turn_context.send_activity(MessageFactory.text(text))
```

For full token-level streaming, use `astream_events` and send each
`on_llm_new_token` event as a separate activity update.  Note that some
channels (Teams) buffer activities; test the actual UX in your target channel
before optimising further.

---

## Observability and Traces

Copilot Studio sees only the final text output.  Without structured traces,
debugging multi-step agent failures is very hard.

### Recommended trace structure

```python
from pydantic import BaseModel

class ToolTrace(BaseModel):
    step_id: str
    tool_name: str
    inputs: dict
    output_summary: str
    duration_ms: int

class ConversationTrace(BaseModel):
    session_id: str
    conversation_id: str
    user_id: str
    steps: list[ToolTrace]
    total_duration_ms: int
```

Store traces in the same SQLite/Postgres store as memory, keyed by
`session_id`.  Optionally append a brief `[N steps executed]` footer to the
agent response for user-visible explainability.

Integrate with `genai-tk`'s monitoring config (`MonitoringConfig`) to route
traces to LangSmith or Langfuse when `monitoring.enabled: true`.

---

## Timeouts and Guardrails

DeepAgents can loop, chain tools indefinitely, or run expensive tasks.  Define
explicit ceilings in `config/app_conf.yaml`:

```yaml
m365_agent:
  app_id: ${oc.env:M365_BOT_APP_ID,}
  app_password: ${oc.env:M365_BOT_APP_PASSWORD,}
  agent_profile: research
  guardrails:
    max_steps: 15              # max tool-call iterations
    max_tool_calls: 30         # cumulative tool invocations per turn
    timeout_seconds: 120       # wall-clock limit for ainvoke()
    fallback_response: >-
      I wasn't able to complete this request within the allowed time.
      Please try a more focused question.
```

Enforce them in the handler:

```python
import asyncio

async def on_message_activity(self, turn_context: TurnContext) -> None:
    ...
    timeout = config.get("m365_agent.guardrails.timeout_seconds", 120)
    fallback = config.get("m365_agent.guardrails.fallback_response", "Sorry, I timed out.")
    try:
        result = await asyncio.wait_for(
            agent.ainvoke({"input": user_input}, config=run_config),
            timeout=timeout,
        )
        reply = result.get("output", "")
    except asyncio.TimeoutError:
        reply = fallback
    await turn_context.send_activity(MessageFactory.text(reply))
```

---

## Attachment Handling

The SDK passes attachment URLs — these are **signed, expiring URLs** that
must be fetched promptly.  Do not pass raw URLs directly to the model.

**Ingestion pipeline:**

1. Receive `activity.attachments` in `on_message_activity`
2. Download immediately (before the URL expires — typically 1 hour)
3. Validate MIME type against an allow-list (`application/pdf`, `image/*`, etc.)
4. Run security checks (virus scan in enterprise deployments)
5. Preprocess into model-ready form (PDF → markdown via genai-tk `markdownize`, image → base64)
6. Pass processed content as part of `user_input` to the agent

```python
import aiohttp

ASYNC def _ingest_attachments(attachments: list) -> list[str]:
    """Download, validate, and convert attachments to text snippets."""
    allowed_types = {"application/pdf", "text/plain", "image/png", "image/jpeg"}
    snippets = []
    async with aiohttp.ClientSession() as session:
        for att in attachments:
            if att.content_type not in allowed_types:
                continue  # silently skip unsupported types
            async with session.get(att.content_url) as resp:
                raw = await resp.read()
            snippets.append(_preprocess(raw, att.content_type))
    return snippets
```

---

## Future Evolution

The current architecture is intentionally layered so that the cognitive backend
can evolve without touching the channel layer:

```
Copilot Studio  = interaction layer
Azure Bot       = channel abstraction
M365 SDK        = protocol layer
genai-tk        = cognitive runtime      ← evolves here
sandbox/tools   = execution layer
```

The natural next step (compatible with the existing architecture) is to replace
the simple agent backend with:

- **Knowledge Graph** (`genai-graph` / EKG) as a world model
- **Persistent memory layer** keyed per-user across conversations
- **Multi-agent coordination** — route by classifier to specialised profiles
- **Dynamic profile routing** — `profile_key` resolved from message classifier
  rather than static config

None of these changes require touching the M365 SDK layer.

---

## SDK Status (as of mid-2026)

| Package | Status | Notes |
|---|---|---|
| `microsoft-agents-hosting-aiohttp` | Preview | Part of the M365 Agents SDK; actively developed |
| `microsoft-agents-bot-builder` | Preview | Activity handler abstractions |
| `botbuilder-python` | Stable / maintenance | Older SDK; still works but not the recommended path for new M365 projects |

Track the official SDK at [github.com/microsoft/agents-sdk-python](https://github.com/microsoft/agents-sdk-python) for API changes before going to production.

---

## Local Development Checklist

```bash
# 1. Start the M365 agent endpoint
uv run python -m aiohttp.web genai_blueprint.main.m365_app:app --port 8000

# 2. Tunnel to localhost (Azure Bot Service requires a public HTTPS URL)
ngrok http 8000

# 3. Set Azure Bot messaging endpoint to the ngrok HTTPS URL
# 4. Use Bot Framework Emulator for offline local testing (bypasses auth)
```

---

## References

- [Microsoft 365 Agents SDK for Python](https://github.com/microsoft/agents-sdk-python)
- [Microsoft Copilot Studio documentation](https://learn.microsoft.com/en-us/microsoft-copilot-studio/)
- [Azure Bot Service — Connect to channels](https://learn.microsoft.com/en-us/azure/bot-service/bot-service-manage-channels)
- [Copilot Studio — Use Azure Bot Service bots](https://learn.microsoft.com/en-us/microsoft-copilot-studio/publication-connect-bot-to-azure-bot-service-channels)
- [genai-tk agents documentation](../../genai-tk/docs/agents.md)
- [genai-blueprint FastAPI app](../genai_blueprint/main/fastapi_app.py)
