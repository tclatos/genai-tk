# GenAI-TK Workflow DSL vs Kestra

## Purpose

This note compares the current GenAI-TK workflow DSL, as implemented on top of Prefect, with Kestra as an orchestration platform.

The comparison is intentionally about **functional behavior**, not UI.

It also distinguishes between three things that are easy to conflate:

1. **Native DSL features** available directly in YAML.
2. **Python escape hatches** that are possible because the DSL ultimately runs Python and Prefect.
3. **Roadmap features** described in the redesign notes but not yet the main authoring/runtime model.

That distinction matters. If we compare only the theoretical ceiling of "you can always write custom Python", then the current DSL will look broader than it is in practice. For a platform choice, the more useful question is: **what does each system make easy, standard, and repeatable for a team?**

## Executive Summary

The current GenAI-TK workflow DSL is best understood as a **Python-first orchestration layer** for AI and document-processing pipelines. It adds a clean YAML authoring surface for DAG composition, presets, interpolation, fan-out, caching, and scheduling, while delegating execution to Prefect and step logic to Python callables or Prefect flows.

Kestra is broader. It is a **general orchestration platform** with its own first-class flow language, plugin/task ecosystem, typed inputs and outputs, event-driven triggers, richer control-flow constructs, worker/task-runner execution model, and more platform-native operational semantics.

The practical conclusion is:

- If the team wants a workflow layer tightly coupled to a Python codebase, especially for AI ETL, document pipelines, RAG ingestion, and knowledge-graph builds, the current DSL is a good fit.
- If the team wants a general-purpose orchestrator for multiple teams, multiple execution environments, event-driven automation, or low-code authoring outside Python, Kestra is functionally stronger.
- The current DSL is closer to **"Prefect + a project-specific YAML compiler"** than to **"Kestra as a workflow platform"**.

## Current GenAI-TK DSL: What It Already Does Well

Based on [workflows.md](../workflows.md), [prefect.md](../prefect.md), and the current workflow runtime implementation, the DSL already provides a meaningful functional layer on top of Prefect:

- YAML-defined single-step workflows and multi-step DAG pipelines.
- Reusable workflow composition by referencing other workflows from `run:`.
- Parameter defaults, presets, and CLI overrides.
- `${values.*}` interpolation and `${steps.<id>.result.*}` references.
- Parallel execution of independent branches.
- `foreach` fan-out with bounded concurrency.
- Step retries, tags, and failure policy (`abort`, `skip`, `continue`).
- Optional inline execution to flatten Prefect subflows.
- Workflow validation, dry-run planning, and CLI execution.
- Prefect deployment serving with cron or interval scheduling.
- Built-in manifest and hybrid caching patterns that are especially useful for file-oriented ETL.
- Workflow-level artifacts and step-specific artifact publishing through Prefect.

This is already more than a thin wrapper. For the current repo use cases, it is a practical orchestration tool.

## Core Architectural Difference

The main difference is not syntax. It is **where workflow semantics live**.

### GenAI-TK DSL

- Workflow definitions are YAML.
- Execution logic lives mostly in Python callables or Prefect flows.
- The DSL compiles to a Prefect flow using `PrefectFlowFactory`.
- It is embedded inside a Python application and configuration system.
- The main extension mechanism is "write more Python".

### Kestra

- Workflow definitions are YAML.
- Workflow semantics are much more native to the orchestration layer itself.
- Execution primitives are provided by tasks, flowable tasks, triggers, expressions, task runners, and plugins.
- It is a standalone orchestration platform, not just a library inside an app.
- The main extension mechanism is "use platform tasks/plugins" and, when needed, custom code tasks or plugins.

This leads to very different tradeoffs.

## Functional Comparison

### 1. Authoring Model and Extension Style

**Current DSL**

- Best when a team is comfortable expressing business logic as Python functions or Prefect flows.
- YAML mostly wires together Python units.
- Reuse is strong when workflows are built from existing in-repo code.
- Changes often imply code changes, not only YAML changes.

**Kestra**

- Best when a team wants more orchestration behavior to be expressed directly in the workflow language.
- YAML references tasks/plugins rather than Python callables by dotted path.
- Better suited to mixed-language or polyglot teams.
- More workflow changes can stay declarative.

**Assessment**

The current DSL is stronger as a **codebase-local orchestration layer**. Kestra is stronger as a **shared orchestration language and platform**.

### 2. Inputs, Parameters, and Validation

**Current DSL**

- Has `defaults`, `presets`, `params`, and CLI overrides.
- `params` currently document requiredness and descriptions, but they are not a rich typed schema.
- Runtime correctness depends heavily on Python signatures and step implementations.
- This is enough for internal engineering teams, but it is not a strong contract surface for broad self-service use.

**Kestra**

- Has first-class typed inputs such as string, int, float, boolean, date/time, duration, file, JSON, YAML, URI, secret, arrays, and select-like types.
- Validates inputs before execution creation.
- Supports validators, ranges, regex checks, nested/grouped inputs, and dynamic input population.

**Assessment**

Kestra is functionally ahead on **typed runtime contract definition**. The current DSL is adequate for developer-owned workflows but weaker for workflows exposed to less technical users or multiple consuming teams.

### 3. Control Flow and Orchestration Constructs

**Current DSL**

- Supports DAG dependencies via `after:` and `wait_for:`.
- Supports sub-workflow expansion.
- Supports `foreach` fan-out.
- Supports step-level failure policy and retries.
- Does not expose a broad set of native control-flow constructs in the DSL itself.

**Kestra**

- Provides flowable tasks for orchestration behavior such as parallelism, switches, if/else style branching, loops, and scoped error handling.
- More orchestration logic is represented explicitly in the workflow graph rather than hidden in step code.

**Assessment**

For straight DAG ETL, the current DSL is sufficient. For workflows that need more **branching, dynamic orchestration, and explicit control-flow semantics**, Kestra is more expressive out of the box.

### 4. Outputs and Data Passing Between Steps

**Current DSL**

- Supports `${steps.<id>.result.*}` references.
- In practice, many existing workflows still communicate through directories and files, not strongly modeled step outputs.
- The redesign note explicitly calls out that outputs are not yet fully first-class.

**Kestra**

- Has first-class task outputs and flow outputs.
- Outputs are addressable throughout the flow and across subflows.
- Internal storage URIs are part of the execution model for files.
- Dynamic-task outputs have explicit addressing conventions.

**Assessment**

Kestra has the stronger **dataflow model**. The current DSL works well for file-system-centric AI ETL, but it is less mature for explicit, typed inter-step contracts.

### 5. Triggers and Scheduling

**Current DSL**

- Supports immediate runs, dry-runs, and serving a workflow as a Prefect deployment.
- Scheduling is currently centered on Prefect deployment serving with cron or interval options.
- There is no comparable native DSL surface for webhook triggers, polling triggers, flow-completion triggers, or realtime event triggers.

**Kestra**

- Has first-class schedule, flow, webhook, polling, and realtime trigger models.
- Trigger conditions, trigger-specific inputs, and trigger lifecycle behaviors are part of the platform.
- Plugin-based triggers broaden the event surface further.

**Assessment**

If the decision depends on **event-driven orchestration**, Kestra is functionally in another category.

### 6. Execution Placement and Runtime Isolation

**Current DSL**

- Main execution path is a generated Prefect flow using a local `ThreadPoolTaskRunner`.
- It can serve Prefect deployments.
- The runtime models already contain `work_pool`, `work_queue`, and a `deployment` step kind, which shows the design is moving toward broader placement options.
- However, those capabilities are not yet the dominant user-facing model described in the docs.

**Kestra**

- Workers and task runners are a core concept.
- Compute can be isolated or offloaded to remote environments through task runners.
- This is part of the normal authoring and runtime model, not an edge capability.

**Assessment**

Kestra is substantially stronger if the team needs **runtime isolation, heterogeneous execution backends, or platform-level compute placement**.

### 7. Error Handling and Recovery Semantics

**Current DSL**

- Has step retries and `on_failure` behavior.
- Failure handling is mostly per-step and exception-driven.
- Compensating behavior is usually encoded as explicit downstream steps or Python logic.

**Kestra**

- Has task retries, flow-level retries, global and local error handlers, `allowFailure`, `allowWarning`, and replay/restart concepts.
- Error semantics are more explicit in the orchestration model.

**Assessment**

Kestra has a richer **workflow-level failure model**. The current DSL is good enough for engineering-owned pipelines, but less expressive for complex operational runbooks.

### 8. Caching and Incremental Processing

**Current DSL**

- This is one area where the current stack is unusually strong.
- `cache: manifest` and `cache: hybrid` are native workflow concepts.
- Several built-in flows already use manifests and content hashing for incremental file processing.
- This is a strong fit for document conversion, extraction, anonymization, and ingestion pipelines.

**Kestra**

- Has strong execution history, outputs, retries, and replay semantics.
- But the core docs emphasize orchestration and storage more than a built-in, content-hash-oriented incremental file-processing model.
- Equivalent behavior is likely to be task- or plugin-specific rather than a central workflow convention.

**Assessment**

For **incremental AI/file ETL**, the current DSL has a meaningful advantage in opinionated built-in behavior.

### 9. Platform Scope and Governance

**Current DSL**

- Lives inside the application repo and config system.
- Governance is mainly whatever the application, repo, deployment process, and Prefect setup provide.
- Excellent for a team that wants workflow code to remain inside the same engineering boundary as the application.

**Kestra**

- Has a broader platform model with namespaces, secrets, trigger administration, and platform-level operational concepts.
- Better suited to being a shared control plane across many workflows and teams.

**Assessment**

If the target is a **workflow platform for an organization**, Kestra is the more natural fit. If the target is a **workflow subsystem inside a Python product**, the current DSL is more lightweight and aligned.

## Side-by-Side Summary

| Dimension | Current GenAI-TK DSL | Kestra |
|---|---|---|
| Primary identity | Python-first DSL on top of Prefect | Standalone orchestration platform |
| Best fit | AI ETL and document pipelines inside one Python codebase | Multi-team orchestration across systems and runtimes |
| Reuse model | Reuse Python callables and Prefect flows | Reuse tasks, plugins, subflows, triggers |
| Input model | Defaults, presets, light param metadata | Strongly typed inputs with validation |
| Control flow | DAG + sub-workflows + foreach | Richer native orchestration constructs |
| Outputs | Basic step result references, often file-based handoff | First-class task and flow outputs |
| Scheduling | Prefect deployment serve, cron, interval | Native schedule and event triggers |
| Event-driven automation | Limited in current DSL | Strong, first-class |
| Runtime placement | Mostly local Prefect flow runtime, emerging remote hooks | Native workers/task runners |
| Error model | Retries + per-step failure policy | Global/local handlers, retries, replay/restart |
| Incremental file ETL | Strong manifest/hybrid caching story | Less central as a workflow primitive |
| Polyglot support | Indirect, usually via Python wrappers | Much stronger |

## Where the Current DSL Is Genuinely Close to Kestra

The current DSL is not "toy". It is already close to Kestra in a few specific areas:

- YAML-authored workflow composition.
- Reusable sub-workflows.
- Parameter presets for concrete environments or data sources.
- Parallel branch execution for DAG-shaped pipelines.
- Fan-out mapping over collections.
- Scheduling for repeated batch runs.
- Step metadata such as retries and tags.

If the team's workflows are mostly batch-style DAGs over Python steps, these similarities are real.

## Where It Is Not Close Yet

It is not yet close to Kestra in these areas:

- breadth of native orchestration constructs;
- typed inputs/outputs as first-class contracts;
- event and trigger model;
- worker/task-runner execution architecture;
- platform-level isolation and governance;
- non-Python-first authoring experience;
- explicit error-handling and replay semantics.

The redesign note in [workflow_engine_prefect_redesign.md](workflow_engine_prefect_redesign.md) is important here: it already identifies several of these gaps, especially around first-class outputs, centralized caching semantics, and more truly Prefect-native runtime behavior. That document should be read as evidence that the current system is evolving, not as evidence that those capabilities are already present.

## Decision Guidance

### Choose the current GenAI-TK DSL if:

- most workflow logic already lives in Python;
- the main workloads are document pipelines, RAG ingestion, extraction, and similar AI ETL;
- the same engineering team owns both application code and workflows;
- you value tight code reuse over broad platform abstraction;
- incremental file-processing caches are important;
- event-driven triggers and heterogeneous runtime placement are not central requirements.

### Choose Kestra if:

- workflows are meant to become a shared platform capability across teams;
- you need more than cron and interval scheduling;
- external events, webhooks, polling, or flow-chaining triggers matter;
- workflows should be operable and extensible without writing much Python glue;
- execution isolation and remote compute placement are important from day one;
- you want stronger native contracts for inputs, outputs, and error handling.

## Bottom Line

The current GenAI-TK workflow DSL is a strong **project-level orchestration layer** for Python-native AI workflows. It is productive, composable, and especially good at incremental file-processing pipelines.

Kestra is a stronger **orchestration platform**. It offers broader native workflow semantics, richer trigger and execution models, and a more complete operational abstraction.

So the real choice is not just between two YAML syntaxes.

It is between:

- **an embedded Python/Prefect workflow subsystem optimized for this codebase**, and
- **a dedicated orchestration platform optimized for broader workflow standardization**.

For a single project or a small number of Python-centric AI pipelines, the current DSL is credible and likely simpler.

For a longer-term platform bet across teams, runtimes, and event sources, Kestra is functionally ahead.