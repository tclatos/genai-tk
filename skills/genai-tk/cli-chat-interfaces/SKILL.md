---
name: cli-chat-interfaces
description: Add single-shot or interactive chat commands to any Typer CLI (genai-tk built-ins or scaffolded projects) using the shared harness chat REPL — command history, line editing, slash commands, and rendered tool-call/streaming output. Use whenever a new CLI subcommand needs to send a query to an agent/harness and print or converse with the response.
---

# CLI Chat Interfaces

## Problem

Every new agent-backed CLI command needs a way to (a) send one query and print
the response, and (b) optionally drop into an interactive multi-turn REPL. It's
tempting to hand-roll this with `typer.prompt()` / `input()` in a `while True`
loop — but that gives you no command history, no line editing (arrow keys,
Ctrl-R search), no consistent slash commands, and no shared event rendering
(tool calls, node traces, errors). `cli agents run --chat` already solved this
once; reuse it instead of reinventing it per command.

## Read First

- `genai_tk/agents/harness/chat_repl.py` — the shared implementation
- `genai_tk/agents/harness/commands.py` — reference usage (`cli agents run`)
- `docs/harness.md`, `docs/cli.md`

## Use This, Not A Hand-Rolled Loop

```python
from genai_tk.agents.harness import create_harness
from genai_tk.agents.harness.chat_repl import run_chat_repl, stream_turn
import asyncio

harness = create_harness(profile_key, llm_override=llm)
try:
    if query:
        # stream_turn is sync — it wraps its own asyncio.run(). Call it directly;
        # do NOT await it or wrap it in another asyncio.run() (raises "asyncio.run()
        # cannot be called from a running event loop").
        stream_turn(harness, query, show_trace=trace, console=console)
    else:
        # run_chat_repl is async — interactive multi-turn REPL with prompt_toolkit
        # history/auto-suggest and /help /info /clear /quit slash commands.
        asyncio.run(run_chat_repl(harness, show_trace=trace, console=console))
except KeyboardInterrupt:
    console.print("\n[yellow]Interrupted.[/yellow]")
finally:
    asyncio.run(harness.aclose())
```

This works for **any** harness (LangChain react/deep/custom, DeerFlow) — the
functions operate on the canonical `BaseHarness`/event model, not on a specific
framework.

## What You Get For Free

- `PromptSession` with `FileHistory` (`.agents.input.history`) + `AutoSuggestFromHistory` — arrow-key history, Ctrl-R search.
- Slash commands: `/help`, `/info` (shows harness/profile/model), `/clear` (new thread), `/quit`.
- Consistent Rich rendering of `TokenEvent`, `ToolCallEvent`, `ToolResultEvent`, `NodeEvent` (trace), `ClarificationEvent`, `ErrorEvent` across every command that uses it.
- `--trace` (node-level execution trace) and `--json` (raw NDJSON events) support for free if you thread those flags through to `stream_turn`/rendering.

## Avoid

- Do not write your own `while True: typer.prompt(...)` loop for a new chat-style command — extend `chat_repl.py` instead if you need new behavior (e.g. a new slash command), so every command benefits.
- Do not call `stream_turn()` from inside code already running under `asyncio.run()` — it manages its own event loop.
- Do not duplicate the tool-call/token rendering logic inline; that belongs in `_astream_turn()` in `chat_repl.py` so all callers stay consistent.

## Example Fix Applied

`prjtest`'s scaffolded `cli pentest chat` command originally hand-rolled a
`typer.prompt("You")` loop with manual `TokenEvent` printing (no history, no
line editing, `exit`/`quit`/`q` to leave). It was rewritten to call
`stream_turn`/`run_chat_repl` exactly as shown above — see
`prjtest/prjtest/commands/pentest_commands.py`.
