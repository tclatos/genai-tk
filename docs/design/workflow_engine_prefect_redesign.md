# Workflow Engine Redesign: Prefect-Native, Factory-Based Orchestration

## Executive Summary

The current workflow DSL is useful for simple pipelines, but the runtime architecture does not match the abstraction it presents.

Today, the system:

- models workflow concerns such as templates, sub-workflows, cache, materialization, and concurrency in YAML and Pydantic;
- resolves those definitions into an expanded DAG;
- then executes the DAG in a custom Python loop that invokes steps sequentially.

That approach works for basic ETL, but it underuses Prefect and creates architectural friction in the exact areas we now want to improve:

- true workflow-level parallelism;
- generic, reusable caching;
- distributed execution via Prefect deployments and work pools;
- first-class step outputs and artifacts;
- consistency with the toolkit's existing factory pattern.

This document proposes a redesign in which the YAML layer becomes a compile-time front-end for a Prefect-native runtime.

The central recommendation is:

1. Keep YAML as the authoring experience.
2. Compile YAML into a normalized workflow graph.
3. Use a `PrefectFlowFactory` to build a real Prefect flow object from that graph.
4. Use explicit cache and artifact backends instead of per-flow manifest code.
5. Treat distributed execution and step placement as first-class runtime concepts.

The result is a workflow system that remains approachable for configuration-driven AI ETL, while becoming much closer to real Prefect orchestration and much easier to extend.

## Scope and Intent

This proposal targets the workflow engine currently documented in [docs/workflows.md](../workflows.md) and implemented primarily in:

- [genai_tk/workflow/models.py](../../genai_tk/workflow/models.py)
- [genai_tk/workflow/resolver.py](../../genai_tk/workflow/resolver.py)
- [genai_tk/workflow/executor.py](../../genai_tk/workflow/executor.py)
- [genai_tk/workflow/prefect/run.py](../../genai_tk/workflow/prefect/run.py)

It also considers representative workflows in:

- [config/workflows.yaml](../../config/workflows.yaml)
- [../../genai-graph/config/ekg_workflows.yaml](../../../genai-graph/config/ekg_workflows.yaml)

This is a design proposal, not an implementation plan with full code.

## Current State

## What Works Well Today

Several parts of the current design are already good and should be preserved:

- YAML-defined workflows are readable and easy to compose.
- `workflow_profiles` are a practical CLI abstraction.
- `step_templates` reduce duplication.
- `uses_workflow` is a useful composition primitive.
- OmegaConf interpolation is a good fit for environment and path binding.
- Individual Prefect flows already exist for important ETL steps.

These are valuable foundations. The problem is not the idea of YAML workflows. The problem is the current runtime architecture beneath them.

## The Architectural Mismatch

The current system models workflow semantics in the DSL, but execution is still handled by a custom sequential executor.

In practice, the runtime flow is:

1. Load and validate YAML into Pydantic models.
2. Expand templates and sub-workflows.
3. Topologically sort steps.
4. Execute each step in a Python `for` loop.
5. Call the referenced flow or function via `run_flow_ephemeral`.

That means the top-level orchestration is not truly Prefect-native. Prefect is used as a helper around individual flows, not as the real workflow runtime.

This is the root cause behind most current limitations.

## Diagnosis of the Current Design

## 1. Workflow-Level Parallelism Is Not Really Modeled or Enforced

The DSL exposes a `concurrency` field, but workflow execution is still a sequential loop over ordered steps.

This creates two issues:

- the configuration implies behavior the runtime does not fully enforce;
- actual concurrency lives inside individual flow implementations rather than in the orchestration layer.

For example:

- `markdownize_flow` has its own Prefect task runner and batching model;
- `ppt2pdf_flow` has its own manifest and parallel file conversion logic;
- `rag_file_ingestion_flow` has its own batching and submission model.

So concurrency is fragmented across flows instead of being expressed at the workflow level.

## 2. Cache and Materialization Are Declared Centrally but Implemented Locally

The workflow models already define `cache` and `materialization`, but those concepts are not actually enforced centrally by the runtime.

Instead, each flow reimplements its own version of incremental processing:

- markdownize maintains one manifest shape;
- ppt2pdf maintains another;
- anonymization maintains another;
- BAML extraction maintains another;
- RAG ingestion uses vector-store-specific dedup logic.

This creates duplicated logic and inconsistent semantics.

## 3. Step Outputs Are Not First-Class

Most workflows are wired together through shared directories and profile variables rather than through explicit step outputs.

This is workable for simple pipelines, but it limits:

- composability;
- inspection;
- downstream references;
- validation of inter-step contracts;
- more advanced ETL and distributed patterns.

The `outputs` field exists in the models but is not yet a meaningful runtime concept.

## 4. Orchestration Concerns Leak into Wrapper Steps

Some wrapper steps explicitly manage Prefect runtime context or additional orchestration behavior. This is a sign that orchestration is not centralized strongly enough.

That makes wrappers heavier than they should be and obscures the boundary between:

- business logic;
- step implementation;
- workflow runtime concerns.

## 5. The Current DSL Is DAG ETL, Not Really Pregel

The current workflows are DAG-style ETL orchestration, not general Pregel-style iterative graph computation.

That distinction matters.

Pregel semantics may absolutely be useful for some workloads, but they should be modeled as:

- a step kind;
- a specialized backend;
- or an execution strategy inside a step.

They should not define the top-level orchestration abstraction for every workflow.

The current examples in `config/workflows.yaml` and `config/ekg_workflows.yaml` are fundamentally ETL orchestration graphs.

## Design Goals

The redesigned system should satisfy the following goals.

## Functional Goals

- Keep YAML authoring for workflows and profiles.
- Preserve sub-workflow composition and templates.
- Support simple pipelines without requiring Python glue.
- Support complex AI ETL workflows and multi-stage document pipelines.
- Support KG construction and enrichment pipelines.
- Support mixed local and remote step execution.

## Runtime Goals

- Make Prefect the actual orchestration runtime.
- Enable true DAG-driven parallelism.
- Support distributed execution via deployments and work pools.
- Provide consistent retry, tagging, and run metadata.
- Make outputs and artifacts addressable by downstream steps.

## Architectural Goals

- Follow the toolkit's factory pattern.
- Separate compile-time concerns from runtime concerns.
- Remove duplicated cache logic from individual flows where possible.
- Preserve backward compatibility for a migration period, but do not freeze the current DSL shape forever.

## Recommended Architecture

## Overview

The workflow system should be split into three phases:

1. **Parse**: load YAML into declarative Pydantic models.
2. **Compile**: normalize, expand, and validate the workflow into a compiled graph.
3. **Execute**: use a `PrefectFlowFactory` to generate a real Prefect flow object from the compiled graph.

This gives a much cleaner separation between configuration, graph semantics, and runtime behavior.

## Phase 1: Parse

This phase remains close to the current approach.

Responsibilities:

- load YAML from merged config;
- validate workflow and profile shapes;
- preserve authoring-level concepts such as templates, defaults, and profiles.

The existing `WorkflowSpec`, `StepSpec`, and `WorkflowProfileSpec` provide a usable starting point, though they should evolve.

## Phase 2: Compile

Introduce a `WorkflowCompiler` that transforms declarative specs into a normalized internal graph.

Responsibilities:

- expand `step_templates`;
- expand sub-workflows recursively;
- validate dependency references;
- normalize step execution policy;
- normalize cache and artifact policy;
- resolve output references;
- compute the final DAG for runtime submission.

This phase should produce compiled runtime objects such as:

- `CompiledWorkflow`
- `CompiledStep`
- `ExecutionPolicy`
- `CachePolicy`
- `ArtifactPolicy`
- `ResolvedReference`

## Phase 3: Execute

Execution should no longer be a custom topological `for` loop.

Instead, a `PrefectFlowFactory` should take a compiled workflow and return a Prefect flow object that:

- creates Prefect tasks or subflows for each compiled step;
- wires dependencies via Prefect futures and `wait_for`;
- submits independent steps concurrently when allowed;
- optionally calls Prefect deployments for remote work;
- publishes results and artifacts in a uniform way.

In this design, Prefect is not merely a helper inside steps. Prefect becomes the actual workflow runtime.

## Factory-Based Design

The workflow redesign should align with the toolkit's existing preference for factories.

## `PrefectFlowFactory`

This is the central object in the new architecture.

Responsibilities:

- accept a compiled workflow and resolved profile values;
- build a Prefect flow object;
- apply flow-level runtime configuration;
- expose a `get()` or `flow_factory()` style API aligned with other factories in the toolkit.

Suggested shape:

```python
from typing import Any

from pydantic import BaseModel, Field


class PrefectFlowFactory(BaseModel):
    workflow_name: str
    profile_name: str | None = None
    values: dict[str, Any] = Field(default_factory=dict)

    def get(self):
        return self.flow_factory()

    def flow_factory(self):
        ...
```

This follows the same broad pattern used elsewhere in the toolkit: resolve configuration, normalize it, then expose a `get()` method that returns the executable runtime object.

## `WorkflowCompiler`

Responsibilities:

- turn author-level YAML into compiled runtime models;
- own expansion, normalization, and validation;
- isolate the resolver from runtime-specific behavior.

This is the right place to evolve the DSL without pushing that complexity into execution code.

## `PrefectStepFactory`

Responsibilities:

- resolve one compiled step into the correct Prefect execution unit;
- wrap callables as tasks when needed;
- detect local flows versus remote deployments;
- apply step-level retry, cache, tag, and artifact policy.

## `CacheBackendFactory`

Responsibilities:

- build the cache backend for a step;
- encapsulate reuse policy;
- remove manifest logic from individual flows where practical.

Supported backends should include at least:

- `none`
- `prefect_result`
- `manifest`
- `hybrid`

## `ArtifactStoreFactory`

Responsibilities:

- resolve where step outputs and artifacts are published;
- support filesystem and object-storage-backed artifact stores;
- integrate with Prefect result and artifact publishing where relevant.

## Proposed Runtime Model

## Step Kinds

The runtime should support several explicit step kinds.

### `task`

A Python callable executed as a Prefect task.

Best for:

- small pure transforms;
- lightweight metadata extraction;
- helper computations;
- orchestration support steps.

### `flow`

A local Prefect flow executed as a subflow.

Best for:

- existing flows such as markdownize, ppt2pdf, BAML extraction, and RAG ingestion;
- larger multi-task step implementations.

### `workflow`

A reference to another YAML workflow.

This preserves the value of sub-workflow composition while making it part of a single normalized runtime model.

### `deployment`

A step executed via a Prefect deployment and routed to a work pool or queue.

Best for:

- long-running or resource-intensive jobs;
- distributed execution;
- isolated environments;
- scalable batch ETL.

### `factory`

An advanced escape hatch for cases where the executable unit itself must be created dynamically from config.

Best for:

- KG builders;
- specialized ingest jobs;
- domain-specific step builders.

### `callable`

A plain dotted-path callable that the runtime may wrap as a Prefect task.

This is mainly a compatibility and convenience kind.

## DSL Recommendations

## Keep the Current High-Level Shape

Retain:

- `workflows:`
- `workflow_profiles:`
- `step_templates:`
- `defaults:`

These are good abstractions and already understandable to users.

## Normalize the Step Schema

The current schema uses separate fields such as `uses`, `uses_workflow`, `inputs`, and `params`.

That should be simplified.

Recommended target schema:

```yaml
steps:
  - id: markdownize
    invoke:
      kind: flow
      target: genai_tk.workflow.prefect.flows.markdownize_flow.markdownize_flow
    wait_for:
      - ppt_to_pdf
    with:
      base_dir: "${values.pdf_dir}"
      output_dir: "${values.md_dir}"
      pathspecs:
        - "**/*.pdf"
      batch_size: "${values.batch_size}"
      converter: "${values.converter}"
    cache:
      backend: manifest
      level: item
      key:
        include:
          - inputs
          - params
          - code
    execution:
      tags:
        - docs
        - ocr
    artifacts:
      publish_result: true
      publish_metadata: true
```

## Rationale for These Changes

### Replace `uses` and `uses_workflow` with `invoke`

This unifies execution semantics and avoids having multiple mutually exclusive execution fields.

### Replace `needs` with `wait_for`

`wait_for` is closer to Prefect terminology and more explicit about runtime dependency semantics.

Backward compatibility can keep `needs` as an alias for a migration period.

### Replace `inputs` and `params` with `with`

At runtime, both become keyword arguments. The distinction adds complexity without strong value.

### Move execution concerns into an `execution` block

This is the right place for:

- retries;
- tags;
- work pool;
- work queue;
- runner hints;
- concurrency limits.

## Parallelism Model

The current `concurrency: serial|parallel|auto` field should be retired or heavily de-emphasized.

It is not expressive enough and it does not map cleanly to real Prefect orchestration.

Parallelism should instead be modeled at three levels.

## 1. DAG Parallelism

If two steps are independent in the DAG, they should be eligible to run concurrently.

This should be the default behavior of the generated Prefect flow.

No explicit `parallel` flag is needed for this.

## 2. Fan-Out Parallelism

When a step needs to run over many items, files, batches, or partitions, that should be explicit in the DSL.

For example:

```yaml
steps:
  - id: enrich_partition
    invoke:
      kind: deployment
      target: data-enrichment/prod
    foreach:
      from: "${steps.partition.result.partitions}"
      as: partition
      concurrency_limit: 8
    with:
      partition_id: "${item}"
```

This is far more expressive than a step-level `parallel` flag.

## 3. Infrastructure Placement

Some steps should run:

- locally;
- as in-process tasks;
- as subflows;
- remotely via deployment;
- in specific work pools.

That belongs in explicit execution policy, not in a generic concurrency field.

## Caching Model

Caching should be redesigned around two complementary layers.

## 1. Prefect Result Reuse

Use this for steps where the returned Python value is the primary output and the step is effectively pure.

Best for:

- lightweight transforms;
- metadata derivation;
- configuration expansion;
- data selection and planning.

Strengths:

- native to Prefect;
- observable;
- simple.

Limitations:

- insufficient for many ETL steps whose primary effect is writing artifacts or mutating downstream systems.

## 2. Shared Manifest Cache

Use this for side-effecting ETL and incremental file or dataset processing.

Best for:

- OCR;
- conversion;
- anonymization;
- BAML extraction;
- vector ingestion;
- graph loading;
- artifact generation.

This should become a reusable library rather than repeated manifest code inside each flow.

Suggested shared models:

```python
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class CacheRecord(BaseModel):
    key: str
    fingerprint: str
    status: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    code_version: str | None = None
    processed_at: datetime


class ManifestCache(BaseModel):
    records: dict[str, CacheRecord] = Field(default_factory=dict)
```

Capabilities should include:

- manifest load and save;
- per-item fingerprinting;
- code fingerprint invalidation;
- artifact metadata recording;
- invalidation strategies;
- force-rebuild override.

## 3. Hybrid Cache Mode

For many AI ETL steps, the correct answer is hybrid caching:

- Prefect result reuse for top-level step metadata;
- manifest-based incremental processing for files, partitions, or artifacts.

This should be an explicit supported mode rather than an accidental mix of unrelated implementations.

## Artifacts and Materialization

The current `materialization` concept should be reframed around two separate concerns.

## Result Persistence

This describes whether the returned Python object should be persisted and how.

Examples:

- none;
- JSON;
- pickle;
- Prefect result storage block.

## Artifact Publication

This describes the side outputs of a step.

Examples:

- output directory;
- manifest path;
- HTML report;
- vector-store stats;
- KG database path;
- summary metrics.

This distinction is important because AI ETL steps often care more about artifact state than return values.

## Step Outputs and References

The new runtime should allow downstream steps to reference outputs from upstream steps directly.

Examples:

```yaml
with:
  base_dir: "${steps.markdownize.result.output_dir}"
```

```yaml
with:
  manifest_path: "${steps.extract.artifacts.manifest}"
```

```yaml
with:
  kg_path: "${steps.build_graph.result.db_path}"
```

This is a major improvement over today's reliance on shared profile values and hand-managed directories.

## Implications for Existing Example Workflows

## `config/workflows.yaml`

Current examples such as:

- markdownize-only profiles;
- RAG ingestion;
- anonymize and ingest;

fit well into the proposed model and would largely become clearer.

These workflows are straightforward DAG ETL examples and do not require Pregel-style semantics.

## `genai-graph/config/ekg_workflows.yaml`

The KG workflows are where the redesign becomes especially valuable.

KG workflows often need:

- heavy compute;
- long runtimes;
- local and remote resources;
- artifact publication;
- incremental rebuild behavior;
- better placement control.

Those should become first-class runtime concerns rather than being hidden inside wrapper steps.

## Pregel Positioning

If Pregel-powered execution is required for some use cases, it should be introduced as a step kind or backend.

For example:

```yaml
steps:
  - id: graph_reasoning
    invoke:
      kind: factory
      target: my_package.workflow.factories.PregelStepFactory
    with:
      graph_input: "${steps.prepare.result.graph_path}"
      max_supersteps: 20
```

That keeps the top-level workflow model consistent while still enabling Pregel execution where it is actually needed.

## Migration Strategy

The redesign should be rolled out in stages.

## Phase 1: Introduce the Compiler and Factories

- Add `WorkflowCompiler`.
- Add `PrefectFlowFactory`.
- Add `PrefectStepFactory`.
- Keep the current YAML mostly compatible.
- Support compatibility aliases:
  - `uses` -> `invoke.target`
  - `uses_workflow` -> `invoke.kind: workflow`
  - `needs` -> `wait_for`
  - `inputs` and `params` -> `with`

## Phase 2: Centralize Cache and Artifact Backends

- Introduce shared manifest cache support.
- Introduce explicit result persistence and artifact publication policies.
- Refactor existing flows to reuse common cache primitives where practical.

Priority candidates:

- markdownize;
- ppt2pdf;
- anonymize;
- BAML extraction;
- RAG ingestion.

## Phase 3: Add Distributed Execution Support

- Support `deployment` step kind.
- Add work-pool and work-queue placement.
- Add tags and resource-oriented execution policy.
- Make remote execution first-class rather than wrapper-driven.

## Phase 4: Simplify the DSL

- Deprecate `concurrency`.
- Deprecate separate `inputs` and `params`.
- Deprecate orchestration-heavy wrapper patterns when no longer needed.

## Alternatives Considered

## Alternative A: Keep the Current Executor and Add More Features

This is the lowest-effort path, but it is not the right long-term design.

It would preserve the fundamental mismatch between:

- a rich workflow DSL;
- a custom, largely sequential runtime.

That path increases complexity without solving the orchestration problem at the root.

## Alternative B: Drop YAML Composition and Use Raw Prefect Only

This would move closer to Prefect, but it would throw away one of the system's strongest features: approachable, composable, configuration-driven workflows.

That is too high a usability cost.

## Alternative C: Thin YAML Compiler to Prefect

This is the recommended approach.

It preserves the value of YAML authoring while making Prefect the actual runtime model.

## Key Recommendations

If only a few changes are adopted, they should be these.

1. Stop evolving the current top-level custom executor.
2. Introduce a `WorkflowCompiler` and `PrefectFlowFactory`.
3. Treat graph parallelism as a DAG property rather than a step flag.
4. Replace per-flow manifest implementations with shared cache primitives.
5. Make step outputs and artifacts first-class.
6. Model distributed execution explicitly with step kinds and execution policy.
7. Treat Pregel as a specialized execution backend, not the top-level orchestration model.

## Open Design Questions

These questions should be resolved before implementation begins.

## 1. Backward Compatibility Horizon

Should the current YAML DSL be preserved for one migration cycle only, or for an extended period?

Recommendation:

- keep compatibility aliases for one migration cycle;
- make the new normalized schema the target model.

## 2. Step Output Contracts

Do we want weakly typed step outputs, or should step results optionally declare schemas?

Recommendation:

- start with flexible dictionaries and artifact references;
- add optional typed result schemas later for important step families.

## 3. Deployment Support Timing

Should remote Prefect deployment support be available in phase 1 or phase 3?

Recommendation:

- design for it in phase 1;
- fully operationalize it in phase 3.

## 4. Mapping and Partitioned Execution

Do we want `foreach` or `map_over` semantics in the first redesign iteration?

Recommendation:

- reserve the model now;
- implement after the compiler and flow factory exist.

## Final Recommendation

The right next-generation workflow system for this toolkit is:

- YAML-authored;
- Prefect-native at runtime;
- factory-based in architecture;
- artifact-aware;
- cache-pluggable;
- deployment-capable;
- and explicitly designed for AI ETL rather than pretending every workflow is a Pregel computation.

The most important structural move is to replace the current executor-centric design with a compiler plus `PrefectFlowFactory`.

That single change makes the rest of the redesign coherent:

- parallelism becomes a property of the graph and runtime;
- cache becomes a reusable backend rather than per-flow code;
- distributed execution becomes a supported step kind;
- artifacts and outputs become real workflow objects;
- and the system becomes much closer to Prefect while staying aligned with the toolkit's factory pattern.