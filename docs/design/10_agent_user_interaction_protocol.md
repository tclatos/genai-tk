# Design 10: Agent-User Interaction Protocol (AUIP) & Multimodal Frontends

## 1. Architectural Challenge: Decoupling Execution from Presentation

In early autonomous agent systems, execution and presentation were tightly coupled:
- **Terminal Coupling**: Middlewares such as `RichToolCallMiddleware` hooked into tool execution loops to write ANSI escape codes directly to `stdout`.
- **Headless Failure**: In background tasks, batch evaluations (e.g. FinanceBench, OfficeQA), and cloud API servers, hardcoded terminal prints polluted application logs with unrenderable formatting.
- **Framework Fragmentation**: Display logic written for LangChain did not function for DeerFlow or custom graph engines.
- **Unidirectional Display**: Terminal logging middlewares could not support bidirectional Human-in-the-Loop (HITL) scenarios (e.g. asking for user confirmation before executing destructive commands).

`genai-tk` resolves these issues through the **Agent-User Interaction Protocol (AUIP)**, establishing a clean boundary between agent runtimes and presentation frontends.

```mermaid
flowchart TD
    subgraph Execution["Agent Execution Runtimes"]
        LC["LangChain / DeepAgents"]
        DF["DeerFlow Harness"]
    end

    subgraph Observability["Telemetry & Event Bus"]
        RELAY["NVIDIA NeMo Relay Engine"]
        BRIDGE["RelayAuipBridge (genai_tk/protocols/auip/relay_bridge.py)"]
    end

    subgraph Protocol["AUIP Event & Control Hub"]
        OUTBOUND["AUIP Outbound Stream (Tokens, Tools, Artifacts, HITL)"]
        INBOUND["AUIP Inbound Channel (Prompts, Approvals, Interrupts)"]
    end

    subgraph Frontends["Multimodal Client Adapters"]
        CLI["1. Console CLI (RichAuipRenderer)"]
        TUI["2. Terminal TUI (Textual Cockpit)"]
        ST["3. Streamlit Workbench (StreamlitAuipConsumer)"]
        WEB["4. Remote Web App (FastAPI SSE / WebSockets)"]
    end

    Execution --> RELAY
    RELAY --> BRIDGE
    BRIDGE --> OUTBOUND
    Execution -.->|Direct Token Streaming| OUTBOUND

    OUTBOUND --> CLI
    OUTBOUND --> TUI
    OUTBOUND --> ST
    OUTBOUND --> WEB

    CLI --> INBOUND
    TUI --> INBOUND
    ST --> INBOUND
    WEB --> INBOUND
    INBOUND --> Execution
```

---

## 2. NeMo Relay as the Universal Lifecycle Tap

Rather than injecting presentation middlewares into agent execution stacks, `genai-tk` leverages **NeMo Relay** ([Design 09: Observability, Trajectories & Evaluation Architecture](09_observability_and_trajectories.md)) as the universal event tap:
- **Hierarchical Scopes**: NeMo Relay captures `Agent` start/stop, `Step` transitions, `LLM` token usage, and `Tool` invocations across both LangChain and DeerFlow.
- **Structured Attributes**: Captures exact tool names, arguments, return values, timing, and errors without touching standard output.
- **Semantic Marks**: Records events like `skill.load`, planning updates, and subagent handoffs.
- **Live Event Listener Bus**: `add_relay_event_listener()` allows `RelayAuipBridge` to tap into live events without altering the execution pipeline.

---

## 3. Formal AUIP Specification

The protocol defines typed Pydantic v2 schemas in `genai_tk.protocols.auip`:

### Outbound Events (Agent -> Client)

| Event Type (`type`) | Payload Model | Purpose |
|---|---|---|
| `stream.token` | `AuipTokenEvent` | Incremental assistant response text chunks |
| `stream.thinking` | `AuipThinkingEvent` | Model internal reasoning / scratchpad trace |
| `phase.change` | `AuipPhaseEvent` | Cognitive node transition (e.g. `planner`, `researcher`) or planning todos |
| `tool.call` | `AuipToolCallEvent` | External tool call invocation details and authorization state |
| `tool.result` | `AuipToolResultEvent` | Captured tool output, execution duration (ms), and status |
| `artifact.emit` | `AuipArtifactEvent` | Rich outputs (code, tables, Mermaid diagrams, file paths) |
| `hitl.request` | `AuipHitlRequestEvent` | Clarification questions or authorization approval requests |
| `telemetry.usage` | `AuipUsageTelemetryEvent` | Turn-level prompt/completion tokens, latency, cost, model ID |
| `turn.error` | `AuipErrorEvent` | Unhandled exceptions or safety violations |
| `turn.end` | `AuipEndEvent` | Turn conclusion signal |

### Inbound Control Messages (Client -> Agent)

| Message Type (`type`) | Payload Model | Purpose |
|---|---|---|
| `user.message` | `AuipUserMessage` | User prompt initiating or continuing a turn |
| `hitl.response` | `AuipHitlResponse` | User reply to a pending `hitl.request` |
| `tool.decision` | `AuipToolDecision` | Human approval, rejection, or amendment of a pending tool call |
| `control.interrupt`| `AuipInterruptControl` | Cancellation, pause, or steering signals |

---

## 4. Multimodal Frontends

AUIP enables a single agent backend to serve diverse client form factors:

### 1. Console CLI (`cli agents run`)
- Consumes `harness.astream_auip()`.
- Uses `RichAuipRenderer` (`genai_tk/protocols/auip/renderers/rich_console.py`).
- Renders live spinners for tool invocations, one-line summary briefs with duration, loaded skill badges, and formatted Markdown answers.

### 2. Terminal TUI (`cli agents tui`)
- Textual-based split-screen cockpit.
- Left pane displays the live nested cognitive step tree with collapsible tool parameter/result inspectors.
- Right pane displays streaming conversational markdown and interactive approval prompts.

### 3. Streamlit Webapp (`genai_tk/webapp`)
- Driven by `stream_harness_turn()` in `genai_tk/webapp/ui_components/harness_workbench.py`.
- Maps AUIP events to native `st.status()`, `st.chat_message()`, `st.expander()`, and interactive code/Mermaid artifact visualizers.

### 4. Remote Web App Gateway (SSE / WebSockets) & AG-UI Standard Interoperability
- **Official AG-UI Protocol Bridge**: `harness_events_to_ag_ui()` in [genai_tk/protocols/auip/ag_ui_bridge.py](/home/tcl/prj/genai-tk/genai_tk/protocols/auip/ag_ui_bridge.py) adapts canonical agent harness events into official AG-UI standard events (`RUN_STARTED`, `TEXT_MESSAGE_CONTENT`, `TOOL_CALL_START`, `TOOL_CALL_RESULT`, `REASONING_*`, etc.) encoded via `ag-ui-protocol`.
- **FastAPI Gateway**: Implemented in [genai_tk/protocols/auip/server.py](/home/tcl/prj/genai-tk/genai_tk/protocols/auip/server.py):
  - `POST /agent`: Standard AG-UI Server-Sent Events endpoint consumed out-of-the-box by `@assistant-ui/react-ag-ui` and `@ag-ui/client`.
  - `POST /api/v1/auip/stream`: Canonical AUIP SSE endpoint.
  - `GET /profiles`: Enumerate discoverable agent profiles across LangChain and DeerFlow.
  - `GET /health`: Liveness probe.
- **Dedicated Web Client**: [auip-client](/home/tcl/prj/auip-client), a modern React + Vite + TypeScript web interface built with `@assistant-ui/react` and `@assistant-ui/react-ag-ui`, featuring real-time token streaming, collapsible chain-of-thought blocks, tool call inspection badges, and a live agent profile selector.
- **CLI Commands**: `cli auip serve --profile <profile> --port <port>` and `cli auip info` provided by [genai_tk/protocols/auip/cli_commands.py](/home/tcl/prj/genai-tk/genai_tk/protocols/auip/cli_commands.py).
- Detailed implementation patterns and client code are documented in the skill `agent-user-interaction-protocol`.

---

## 5. Deprecation & Retirement of `RichToolCallMiddleware`

`RichToolCallMiddleware` in `genai_tk/agents/langchain/middleware/rich_middleware.py` is formally deprecated:
- Removed from default configurations (`config/examples/agents/defaults.yaml`, `config/agents/langchain/browser.yaml`).
- Removed from sibling projects (`wiki`, `genai-graph`, `rfq-analyst`, `ekg-atos`).
- Retained as a backward-compatibility stub emitting `DeprecationWarning` if referenced by legacy third-party configurations.
