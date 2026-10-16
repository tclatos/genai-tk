---
name: agent-user-interaction-protocol
description: Build, extend, or connect agent frontends (Console CLI, Terminal TUI, Streamlit, or Remote Web Apps via HTTP/WebSocket/SSE) using the Agent-User Interaction Protocol (AUIP) and NeMo Relay lifecycle bridge. Use whenever designing client interfaces or exposing agent streaming over network APIs.
---

# Agent-User Interaction Protocol (AUIP) & Multimodal Frontends

## Problem

Historically, autonomous agents printed output directly to terminal `stdout` via custom middlewares (e.g. `RichToolCallMiddleware`). This coupled agent execution to local terminals, failed for headless environments (APIs, batch benchmarks, background tasks), created runtime fragmentation between LangChain and DeerFlow, and made it impossible for web apps or remote frontends to receive structured, interactive events.

**AUIP** decouples execution from presentation:
1. **Telemetry & Lifecycle Tap**: NeMo Relay captures execution scopes (`agent`, `llm`, `tool`) and semantic marks (`skill.load`, planning updates, HITL).
2. **Standardized Wire Protocol**: AUIP normalizes all streaming tokens, thoughts, tool calls, results, and human-in-the-loop requests into strongly-typed Pydantic v2 events.
3. **Multimodal Client Adapters**: Frontends (Console CLI, Terminal TUI, Streamlit, and Remote Web Apps) consume the same stream independently.

## Architecture

```
Agent Execution ──► NeMo Relay Scopes ──► RelayAuipBridge ──► AUIP Event Stream
 (LC / DeerFlow)                                                     │
                                       ┌─────────────────────────────┼─────────────────────────────┐
                                       ▼                             ▼                             ▼
                              Console CLI / TUI             Streamlit Workbench           Remote Web App (SSE)
```

## Outbound AUIP Event Types

All outbound events derive from `AuipEvent` (`genai_tk.protocols.auip`):

| Event Type (`type`) | Payload Model | Description |
|---|---|---|
| `stream.token` | `AuipTokenEvent` | Incremental assistant response text chunks |
| `stream.thinking` | `AuipThinkingEvent` | Model internal chain-of-thought scratchpad |
| `phase.change` | `AuipPhaseEvent` | Graph node / phase / subagent transition |
| `tool.call` | `AuipToolCallEvent` | Tool name, invocation arguments, call ID |
| `tool.result` | `AuipToolResultEvent` | Execution output, status (`success`/`error`), duration (ms) |
| `artifact.emit` | `AuipArtifactEvent` | Rich artifacts (code, table, SVG, Mermaid diagram, file) |
| `hitl.request` | `AuipHitlRequestEvent` | Human-in-the-loop prompt, question, or approval request |
| `telemetry.usage` | `AuipUsageTelemetryEvent` | Token counts, duration, model, cost |
| `turn.error` | `AuipErrorEvent` | Unhandled error or safety violation |
| `turn.end` | `AuipEndEvent` | Turn completion signal |

## Consuming AUIP in Python Frontends

### 1. From Any BaseHarness

```python
from genai_tk.agents.harness import create_harness

harness = create_harness("research")
try:
    async for event in harness.astream_auip("Investigate quantum computing trends"):
        if event.type == "stream.token":
            print(event.text, end="", flush=True)
        elif event.type == "tool.call":
            print(f"\n[Tool Invoked]: {event.tool_name}({event.args})")
        elif event.type == "tool.result":
            print(f"[Tool Result]: {event.content[:100]} ({event.duration_ms}ms)")
finally:
    await harness.aclose()
```

### 2. Using Rich Terminal Renderer

```python
from genai_tk.agents.harness.chat_repl import astream_turn

# Uses RichAuipRenderer under the hood
full_answer = await astream_turn(harness, query, show_trace=True)
```

---

## Official AG-UI Standard Support & `cli auip serve`

`genai-tk` natively supports the [AG-UI protocol](https://github.com/ag-ui-protocol/ag-ui) standard via `harness_events_to_ag_ui()` ([genai_tk/protocols/auip/ag_ui_bridge.py](/home/tcl/prj/genai-tk/genai_tk/protocols/auip/ag_ui_bridge.py)) and the built-in server in [genai_tk/protocols/auip/server.py](/home/tcl/prj/genai-tk/genai_tk/protocols/auip/server.py).

> **Troubleshooting:** an empty chat usually means agent creation failed (e.g. missing optional extra). Run `uv sync --extra harnessing`; errors are shown in the chat and server log.

### 1. Launching the AUIP / AG-UI Server

```bash
# Start on default port 8000 (profile from config/auip.yaml or --profile)
uv run cli auip serve

# Start with a specific profile on custom port
uv run cli auip serve --profile coding --port 8080

# Inspect configured endpoints and available agent profiles
uv run cli auip info
```

### 2. Endpoints Exposed

- `POST /agent`: Official AG-UI SSE stream consumed by `@assistant-ui/react-ag-ui` and `@ag-ui/client`.
- `POST /api/v1/auip/stream`: Canonical AUIP SSE stream for custom web applications.
- `GET /profiles`: Enumerate all available LangChain and DeerFlow agent profiles for UI dropdowns.
- `GET /health`: Server health and active default profile.

### 3. Dedicated React Web Client (`auip-client`)

A complete reference web client is available in `/home/tcl/prj/auip-client`:
- Built with React, Vite, Tailwind CSS, `@assistant-ui/react`, `@assistant-ui/react-ag-ui`, and `@ag-ui/client`.
- Supports dynamic profile switching via dropdown, real-time token streaming, thinking/reasoning blocks, and tool call inspection badges.
- Run with:
  ```bash
  cd /home/tcl/prj/auip-client
  npm install
  npm run dev
  ```

---

## Exposing AUIP Over Remote HTTP / Server-Sent Events (SSE)

To connect remote web frontends (React, Vue, Next.js, Electron, or mobile apps) to `genai-tk` agents, implement an SSE gateway with FastAPI:

### 1. FastAPI Server Implementation

```python
from collections.abc import AsyncIterator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from genai_tk.agents.harness import create_harness
from genai_tk.protocols.auip.models import AuipInboundMessage, AuipUserMessage

app = FastAPI(title="GenAI Toolkit AUIP Gateway")


class ChatRequest(BaseModel):
    profile: str = "default"
    message: str
    thread_id: str | None = None


async def auip_sse_generator(profile: str, query: str, thread_id: str | None) -> AsyncIterator[str]:
    harness = create_harness(profile)
    try:
        async for event in harness.astream_auip(query, thread_id=thread_id):
            # Format according to W3C Server-Sent Events specification:
            # event: <type>
            # data: <json>
            json_payload = event.model_dump_json()
            yield f"event: {event.type}\ndata: {json_payload}\n\n"
    finally:
        await harness.aclose()


@app.post("/api/v1/sessions/stream")
async def stream_chat(req: ChatRequest):
    return StreamingResponse(
        auip_sse_generator(req.profile, req.message, req.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
```

### 2. Client-Side Consumption (JavaScript / TypeScript)

Using standard browser `fetch` or EventSource:

```typescript
async function streamAgentResponse(query: string, onEvent: (ev: any) => void) {
  const response = await fetch("/api/v1/sessions/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile: "research", message: query }),
  });

  const reader = response.body?.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader!.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n\n");
    buffer = lines.pop() || "";

    for (const chunk of lines) {
      if (!chunk.trim()) continue;
      const dataMatch = chunk.match(/^data:\s*(.+)$/m);
      if (dataMatch) {
        const auipEvent = JSON.parse(dataMatch[1]);
        onEvent(auipEvent);
      }
    }
  }
}

// React / UI integration:
streamAgentResponse("What is RAG?", (event) => {
  switch (event.type) {
    case "stream.token":
      setAnswer((prev) => prev + event.text);
      break;
    case "tool.call":
      setActiveTools((prev) => [...prev, { name: event.tool_name, args: event.args }]);
      break;
    case "tool.result":
      updateToolResult(event.call_id, event.content, event.duration_ms);
      break;
    case "hitl.request":
      openApprovalModal(event.question, event.options);
      break;
  }
});
```

---

## Exposing AUIP Over WebSockets (Bidirectional HITL)

For real-time human-in-the-loop confirmation before running dangerous tools (e.g. bash commands or database mutations), use WebSockets:

```python
from fastapi import WebSocket, WebSocketDisconnect


@app.websocket("/ws/v1/chat")
async def chat_websocket(websocket: WebSocket):
    await websocket.accept()
    harness = create_harness("coding")
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "user.message":
                query = data.get("text", "")
                async for event in harness.astream_auip(query):
                    await websocket.send_text(event.model_dump_json())
                    # If human approval is required:
                    if event.type == "hitl.request":
                        # Wait for client's inbound decision
                        resp = await websocket.receive_json()
                        # Pass decision back into execution context
    except WebSocketDisconnect:
        pass
    finally:
        await harness.aclose()
```

## Anti-Patterns to Avoid

- **Never print to stdout inside agent middlewares**: Do not use `rich.console.Console().print` in tool definitions or execution middlewares. Emit events instead.
- **Never couple frontends to specific engines**: Frontend consumers must only depend on `AuipEvent` schemas, never on LangChain callbacks or DeerFlow internals.
- **Always close the harness**: Wrap streaming calls in `try ... finally: await harness.aclose()` to clean up background containers, sandbox connections, and trajectory flushes.
