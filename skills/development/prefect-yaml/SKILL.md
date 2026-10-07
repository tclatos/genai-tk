---
name: prefect-yaml
description: Declarative YAML DSL, typed contracts, manifest caching, and execution runtime for Prefect workflows. Use when authoring, modifying, or executing workflows, pipelines, step tasks, presets, and routing rules in prefect-yaml or consuming projects.
---

# Prefect YAML DSL (`prefect-yaml`)

## Overview

`prefect-yaml` is a standalone, Python-first workflow authoring surface on top of Prefect. It compiles declarative YAML workflow definitions into Prefect `@flow` and `@task` DAGs without requiring boilerplate Python orchestration glue.

Source repository: [https://github.com/tclatos/prefect-yaml](https://github.com/tclatos/prefect-yaml)

## Core Capabilities

1. **Typed Input Contracts (`inputs:`)**: Strongly typed inputs (`int`, `str`, `float`, `bool`, `path`, `enum`, `list`, `dict`) with constraints (`choices`, `min`, `max`, `regex`) and automatic CLI coercion.
2. **Predictable Value Precedence**: `defaults → presets → CLI --set KEY=VALUE` with OmegaConf `${values.*}` interpolation.
3. **DAG Composition**: Steps define dependencies using `after:` (alias `wait_for:`). Independent steps run in parallel.
4. **Sub-workflow Inlining**: Steps can reference other workflows via `run: <workflow_name>`. Steps are recursively expanded and leaf dependencies rewritten.
5. **Manifest Caching**: Steps with `cache: manifest` compute deterministic xxHash3 fingerprints over resolved inputs. Fresh steps are skipped.
6. **Generic Routing**:
   - `router:` partitions item collections across parallel workflows via rules (`pathspec`, `regex`, prefix).
   - `switch:` evaluates dynamic values to choose workflow branches.
7. **Execution Modes**: Standard subflows with full Prefect UI visibility, or `inline: true` subflow flattening.

## Workflow Definition Syntax

```yaml
workflows:
  data_pipeline:
    description: "Sample pipeline with typed inputs and steps"
    inputs:
      source_dir:
        type: path
        required: true
        description: "Source directory containing raw files"
      batch_size:
        type: int
        default: 10
        minimum: 1
        maximum: 100
      mode:
        type: enum
        choices: [fast, standard, deep]
        default: standard
    defaults:
      source_dir: "/data/input"
      batch_size: 10
    presets:
      production:
        source_dir: "/var/data/prod"
        batch_size: 50
        mode: deep
    pipeline:
      - id: step_extract
        run: my_module.extract_step
        with:
          input_dir: "${values.source_dir}"
          batch_size: "${values.batch_size}"

      - id: step_transform
        run: my_module.transform_step
        after: [step_extract]
        with:
          extracted: "${steps.step_extract.result.output_path}"
          mode: "${values.mode}"
        cache: manifest
```

## Python Step Decoration (`@workflow`)

Decorate plain callables to register them by name:

```python
from prefect_yaml import workflow


@workflow(name="extract_step", description="Extract records from directory")
def extract_step(*, input_dir: str, batch_size: int = 10) -> dict:
    return {"output_path": f"{input_dir}/extracted.json", "count": 42}
```

## Generic Routing Syntax

```yaml
workflows:
  partitioned_etl:
    pipeline:
      - id: partition_sources
        router:
          items: "${values.sources}"
          rules:
            - match: "**/*.pdf"
              run: pdf_converter_flow
              with:
                files: "${items}"
            - match: "https://**"
              run: web_fetcher_flow
              with:
                urls: "${items}"
          default:
            run: fallback_processor
            with:
              items: "${items}"
```

## CLI Commands

```bash
# List workflows and presets
uv run prefect-yaml list

# Inspect contracts, presets, and DAG
uv run prefect-yaml show <workflow_name>

# Dry-run: inspect plan, values, and cache status without running
uv run prefect-yaml run <workflow_name>[/<preset>] --dry-run

# Run workflow with CLI overrides
uv run prefect-yaml run <workflow_name>/<preset> --set batch_size=20

# Bypass cache manifests
uv run prefect-yaml run <workflow_name> --force

# Validate definitions in directory
uv run prefect-yaml validate

# Manage local Prefect server daemon
uv run prefect-yaml server start
uv run prefect-yaml server status
uv run prefect-yaml server stop
uv run prefect-yaml server ui
```
