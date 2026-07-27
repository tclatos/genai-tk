# DeepAgents & DeerFlow: A LangGraph-First Unification Analysis

**Status:** Independent analysis (second report)
**Date:** 2026-07-26
**Question:** Can we simplify DeepAgents/DeerFlow integration given both run on LangGraph, and make the top LangGraph object accessible?

## Bottom Line

Yes. Both runtimes already produce the same object — a `CompiledStateGraph`
from `langchain.agents.create_agent` — and DeerFlow's embedded client already
builds a **checkpointer-attached** instance of that graph in process. The
integration is not simplified today only because that graph is private and the
DeerFlow streaming path wraps a *synchronous* generator in a thread+queue
bridge, then translates it twice.



## Method and Evidence Base

This analysis reads the installed code, not documentation. Verified facts:

- Installed DeerFlow is `deerflow_harness-2.1.0` (dist-info present). The
  dependency is declared `deerflow-harness[tui]>=2.1.0` (`pyproject.toml:122`)
  with a git-source override tracking `rev = "main"` (`pyproject.toml:210`).
  So the version is currently pinned at 2.1.0; the `main` override is a
  future-drift risk, not a statement about current capability.
- `deepagents.create_deep_agent` and `langchain.agents.create_agent` both
  return a compiled LangGraph graph (`genai_tk/agents/langchain/factory.py`,
  `_create_deep_agent` at line 205, `_create_react_agent` at line 189).
- `LangChainHarness` streams that graph with
  `agent.astream_events(..., version="v2")` and one translator
  (`genai_tk/agents/harness/langchain_harness.py:131`, translator at line 146).
  The agent itself is held in a private `self._agent` (line 75).
- `DeerFlowClient._ensure_agent` builds the DeerFlow graph with
  `create_agent(**kwargs)` **and passes a checkpointer**
  (`deerflow/client.py:376-384`), using the full `build_middlewares(...)` chain
  (line 348), `apply_prompt_template` (line 362), `get_thread_state_schema`
  (line 374), tool resolution, authorization, and deferred-MCP assembly. The
  compiled graph is stored in private `self._agent` (`client.py:210`, `:384`).
  There is no public `agent` property.
- `EmbeddedDeerFlowClient.stream_message` runs `client.stream(...)` (a **sync**
  generator) on a background `threading.Thread` and bridges to async via
  `queue.Queue` + `run_in_executor` (`genai_tk/agents/deer_flow/embedded_client.py:511-535`).
- That client's `_translate_event` only handles `messages-tuple` events and
  emits `TokenEvent`/`ToolCallEvent`/`ToolResultEvent`/`ClarificationEvent`
  (`embedded_client.py:594-629`). It never emits a `NodeEvent`, so the
  `NodeEvent`/`ClarificationEvent` dataclasses it defines are partially dead.
- `DeerFlowHarness._translate_deerflow_event` then maps those local dataclasses
  to harness events (`genai_tk/agents/harness/deerflow_harness.py:151-167`).
  Two translations sit between the graph and the harness event model.
- `create_harness` dispatches to `LangChainHarness` or `DeerFlowHarness` by
  `profile.harness` (`genai_tk/agents/harness/registry.py:102-120`). Neither
  subclass exposes its graph.
- The MCP agent tool calls `create_langchain_agent(...)` and
  `agent.ainvoke(...)` directly, bypassing `BaseHarness`, and hard-codes
  `thread_id: "mcp_default"` (`genai_tk/mcp/agent_tool.py:64`, `:106-111`).

### A public, checkpointer-aware SDK factory 

DeerFlow 2.1.0 ships `create_deerflow_agent` in `deerflow/agents/factory.py`,
exported from `deerflow.agents` (`deerflow/agents/__init__.py:4`, lazy export
at lines 16-20). Its signature accepts `checkpointer: BaseCheckpointSaver |
None = None` (`factory.py:75`) and forwards it to `create_agent(...,
checkpointer=checkpointer, ...)` (`factory.py:168`). Its own docstring calls
it "the SDK-level entry point sitting between the raw
`langchain.agents.create_agent` primitive and the config-driven
`make_lead_agent` application factory."


### `make_lead_agent` not passing a checkpointer is by design, not a gap



## Direct Answers

**Can we simplify the integration because both are on LangGraph?**
Yes. The two harnesses wrap the same `CompiledStateGraph` type. The
asymmetry is purely interface debt: the LangChain path uses native async
`astream_events`; the DeerFlow path uses a sync generator bridged through a
thread and translated twice. Unifying on `astream_events` is mechanical once
the DeerFlow graph is reachable.

**Can we make the top LangGraph object accessible?**
Yes, and it is the single most useful change. Add a read accessor to
`BaseHarness` and expose the compiled graph plus its checkpointer. For
`LangChainHarness` this is trivial (the graph already exists in `self._agent`).
For `DeerFlowHarness` it requires threading the graph `DeerFlowClient` already
builds out through `EmbeddedDeerFlowClient`.

**Does this require dropping DeerFlow features?**
No. Because we expose the graph `DeerFlowClient` already assembles — not a
rebuilt one — the full lead-agent middleware chain (DynamicContext,
SkillActivation, SkillToolPolicy, DurableContext, summarization, title,
memory, view-image, subagent limits, loop detection, token budget, system
message coalescing, terminal/safety finish-reason, clarification, deferred
MCP tool-search, authorization) is preserved unchanged.

## What "Make the Graph Accessible" Concretely Means

Add to `BaseHarness`:

```python path=/home/tcl/prj/genai-tk/genai_tk/agents/harness/base.py start=null
@property
def graph(self): ...
@property
def checkpointer(self): ...
```

- `LangChainHarness.graph` returns `self._agent` (already built lazily by
  `_ensure_agent`, `langchain_harness.py:77`). A single async `await
  self._ensure_agent()` guard is enough; the graph is static per harness
  instance because the profile is fixed at construction.
- `DeerFlowHarness.graph` returns the compiled graph `DeerFlowClient` builds.
  This needs two small exposures:
  - `EmbeddedDeerFlowClient` gains `agent` and `checkpointer` properties
    returning `self._client._agent` and `self._client._checkpointer`
    (`embedded_client.py` wraps `DeerFlowClient` at line 412).
  - `DeerFlowHarness.graph` calls `await self._ensure_client()` then returns
    `client.client._agent`.

**Caveat that must be documented:** the DeerFlow graph is not a single static
object. `DeerFlowClient._ensure_agent` caches it keyed on `(model_name,
thinking_enabled, is_plan_mode, subagent_enabled, max_concurrent_subagents,
max_total_subagents, agent_name, available_skills,
checkpoint_channel_mode, authorization_identity)` (`client.py:280-294`). It is
recreated when those change. So `harness.graph` for DeerFlow means "the graph
for the currently resolved configuration," valid after the first `astream`
(or after an explicit ensure call). The accessor should be async and ensure
the agent, returning the cached graph for the resolved config. This is not a
defect; it is the same laziness `LangChainHarness` already has, just with a
wider cache key.

Once `graph` and `checkpointer` are exposed, any consumer can do:

```python path=null start=null
graph = await harness.graph
async for ev in graph.astream_events(
    {"messages": msg},
    config={"configurable": {"thread_id": tid}, "callbacks": cbs},
    version="v2",
):
    ...
```

and get native LangGraph streaming, native `ainvoke`, and native checkpointer
persistence — without going through the harness event model at all when they
do not want to.

## The Simplification the Accessor Unlocks

With the graph reachable, `DeerFlowHarness.astream` can stop calling
`client.stream_message()` and instead drive `graph.astream_events(version="v2")`
through the *existing* `_translate_langchain_event` translator. Concretely,
introduce one shared streaming helper used by both harnesses:

```text
BaseHarness.astream
  -> shared LangGraph stream adapter
       -> graph.astream_events(version="v2")
       -> _translate_langchain_event (shared)
  -> harness StreamEvents
```

What this deletes from the DeerFlow path:

- The `threading.Thread` + `queue.Queue` + `run_in_executor` sync-to-async
  bridge (`embedded_client.py:511-535`). The graph is async-native; the bridge
  exists only because `DeerFlowClient.stream()` is a sync generator.
- The `StreamEvent`-to-local-dataclass translation
  (`embedded_client._translate_event`, `embedded_client.py:570-629`).
- The local-dataclass-to-harness-event translation
  (`DeerFlowHarness._translate_deerflow_event`, `deerflow_harness.py:151-167`).
- The local DeerFlow dataclasses that become unused
  (`TokenEvent`/`NodeEvent`/`ToolCallEvent`/`ToolResultEvent`/`ErrorEvent`/
  `ClarificationEvent` in `embedded_client.py:73-142`), after callers migrate.

What it adds back (correctly):

- `NodeEvent` visibility for DeerFlow phase nodes (planner, researcher, coder,
  reporter). The current DeerFlow path never emits `NodeEvent`s because
  `_translate_event` only inspects `messages-tuple` events. `astream_events`
  surfaces graph nodes via `on_chain_start`, and the shared translator already
  filters internal nodes (`_INTERNAL_NODE_NAMES`, `langchain_harness.py:35-46`).
  DeerFlow's phase node names pass that filter, so this is a net improvement.

What it keeps (full parity):

- `DeerFlowClient` still owns config/profile resolution, `prepare_profile()`,
  `config_bridge` config generation, model/mode selection, the full
  `build_middlewares(...)` chain, tool resolution, authorization, deferred MCP
  tool-search assembly, and checkpointer lifetime. The simplification touches
  only the *streaming boundary*, not the *graph construction* boundary. That is
  why no DeerFlow feature is lost.

## Why `create_deerflow_agent` Is Not the Right Tool Here

`create_deerflow_agent` is real, public, and checkpointer-aware. But it is the wrong
instrument for a full-parity simplification because its
`_assemble_from_features` assembles a **reduced** middleware chain
(`factory.py:178-349`) — sandbox infra, dangling-tool-call, tool-error,
todo, title, memory, view-image, subagent-limit, loop-detection,
token-budget, clarification. It omits the lead-agent-only middlewares that
`build_middlewares` adds: `DynamicContextMiddleware`,
`SkillActivationMiddleware`, `SkillToolPolicyMiddleware`,
`DurableContextMiddleware`, `SystemMessageCoalescingMiddleware`,
`TerminalResponseMiddleware`, `ModelLengthFinishReasonMiddleware`,
`SafetyFinishReasonMiddleware`, and the configured-extension middlewares
(`deerflow/agents/lead_agent/agent.py:319-472`).

Since full DeerFlow parity is required, rebuilding the graph via
`create_deerflow_agent` would drop skills activation/policy, durable context,
system-message coalescing, terminal/safety finish-reason handling, and
configured extensions. The correct move is therefore to **expose and stream
the graph `DeerFlowClient` already builds**, not to swap factories.
`create_deerflow_agent` remains useful for greenfield SDK callers who want a
minimal DeerFlow-flavoured agent without the lead-agent machinery; it is not
the unification vehicle.

`create_deerflow_agent` *does* accept `middleware=` (full takeover), so in
principle one could pass `build_middlewares(...)` into it. But doing so
reimplements most of `DeerFlowClient._ensure_agent` (tool resolution,
authorization, deferred assembly, prompt template, state schema) for no gain
over simply exposing the graph `DeerFlowClient` already produces.

## One Semantic the Shared Translator Must Handle

The current DeerFlow path special-cases the `ask_clarification` tool into a
`ClarificationEvent` (human-in-the-loop signal) at `embedded_client.py:618`.
The shared `_translate_langchain_event` would otherwise emit that as a
`ToolResultEvent`. The fix is small and contained: in the shared translator,
detect `on_tool_end` with `name == "ask_clarification"` and emit a
`ClarificationEvent` instead. This is the only DeerFlow-specific branch needed;
everything else (`TokenEvent`, `ToolCallEvent`, `ToolResultEvent`, `UsageEvent`,
`NodeEvent`, `ErrorEvent`, `EndEvent`) maps identically.

## MCP Agent-as-Tool

The graph accessor makes the fix straightforward. `register_agent_tool` bypasses `BaseHarness` and hard-codes
`mcp_default` (`agent_tool.py:64`, `:106-111`). With `create_harness` already
returning a `BaseHarness` whose `.graph` and `.astream` are available, the MCP
invoker should:

- Resolve the profile via `create_harness(profile, force_memory_checkpointer=True)`.
- Stream via `harness.astream(query, thread_id=<isolated>)` (or call the graph
  directly when a raw result is preferred).
- Derive the thread ID per MCP session/call instead of the shared literal.
- Return a structured result (text, run_id, artifacts, usage, error), not only
  the final message content.
- Call `harness.aclose()` in a `finally` for sandbox-backed DeepAgents.

This removes the duplicated profile lookup and agent construction in
`agent_tool._build_agent` and the cross-client state-bleed risk in one move,
without touching `server_builder`'s FastMCP/transport/ordinary-tool
responsibilities 

##  Other Points

- Keeping the small Pydantic harness event model as a consumer boundary (CLI,
  Streamlit, MCP) rather than pushing raw LangGraph callbacks outward.
- A bare DeepAgent graph cannot back the DeerFlow web UI; the frontend needs
  the Gateway + LangGraph thread/run/artifact APIs, not just a graph.
- `prepare_profile()` / `config_bridge.py` remain necessary in either path.
- Treating the `rev = "main"` git override as a drift risk worth pinning.
- The MCP `mcp_default` isolation problem and the agent-tool bypass of
  `BaseHarness`.
- Not building a DeerFlow-Gateway replacement unless a concrete product
  requirement justifies it.

## Proposed Implementation Order

1. **Expose the graph.** Add `graph` and `checkpointer` properties to
   `BaseHarness`; implement for `LangChainHarness` (trivial) and
   `DeerFlowHarness` (via new `EmbeddedDeerFlowClient.agent`/`.checkpointer`
   properties). Add a unit test that asserts
   `isinstance(await harness.graph, CompiledStateGraph)` for a react, a deep,
   and a DeerFlow profile.
2. **Share the streamer.** Extract `_translate_langchain_event` into a shared
   `LangGraphStreamAdapter` (or a `LangGraphHarness` mixin) that owns
   `astream_events(version="v2")` invocation, monitoring-callback attachment,
   internal-node filtering, `EndEvent`, and the `ask_clarification` →
   `ClarificationEvent` branch. `LangChainHarness.astream` delegates to it.
3. **Migrate `DeerFlowHarness.astream`.** Drive `harness.graph.astream_events`
   through the shared adapter instead of `client.stream_message`. Keep
   `DeerFlowClient` for construction only. Add the two-turn same-`thread_id`
   persistence test against the SQLite checkpointer.
4. **Retire the dead layers.** Remove the thread+queue bridge, the local
   DeerFlow dataclasses, and `_translate_deerflow_event` once no caller
   references them. Keep `EmbeddedDeerFlowClient` as the construction/checkpointer
   holder and the passthrough for `list_models`/`list_skills`/memory/MCP config.
5. **Harden MCP.** Refactor `agent_tool.py` to invoke `create_harness` +
   `harness.astream` with isolated thread IDs and a structured Pydantic result;
   add a `type: deep` profile invocation test and a two-session isolation test.

Steps 1-2 are non-destructive and independently shippable. Step 3 is the
behaviour-change gated by the persistence test. Steps 4-5 are cleanup that
follows.

## Risks and Controls

- **Graph re-creation on config change (DeerFlow).** Document that
  `harness.graph` reflects the currently resolved configuration; the accessor
  ensures the agent before returning. Control: a test that changes
  `model_name` and asserts a new graph object with the old checkpointer.
- **`ask_clarification` HITL semantics.** Control: an explicit test that a
  DeerFlow run halting on `ask_clarification` emits a `ClarificationEvent`
  through the shared translator.
- **Upstream drift from `rev = "main"`.** Pin a tested commit and keep a
  contract manifest (`create_deerflow_agent` signature, `DeerFlowClient._agent`
  presence, `build_middlewares` name). Control: a smoke import + signature test
  in CI.
- **Sandbox lifecycle for DeepAgents.** `harness.aclose()` must still run in
  `finally`; the graph accessor does not change cleanup ownership.
- **MCP thread isolation.** Never share a literal default; derive per session
  or per call.

## Recommendation

Expose the compiled LangGraph graph and its checkpointer on every harness,
then unify both harnesses' streaming on `astream_events(version="v2")` through
one translator, keeping `DeerFlowClient` as the graph *constructor* (full
parity) and dropping it only as the graph *streamer*. Refactor the MCP
agent tool onto `BaseHarness` with isolated thread IDs.

This answers the actual question — making the top LangGraph
object accessible — and it deletes the real wart (the sync thread+queue
bridge and the double translation) rather than reasoning around it. Do not
swap to `create_deerflow_agent` for this; it sacrifices lead-agent parity for
a simplification that exposing the existing graph already delivers.
