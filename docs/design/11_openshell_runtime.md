# OpenShell Runtime Integration for genai-tk

> **Work in progress.** OpenShell is not yet a generated `cli init` feature and
> there is no stable genai-tk OpenShell SDK wrapper. This document describes the
> current intended design, the available whole-worker integration, known
> limitations, and the work required to make deployment repeatable for every
> scaffolded project.

**Status:** experimental integration design, with a working local
whole-worker deployment

**Evidence baseline:** OpenShell `v0.1.10` and the current genai-tk branch

## Summary

OpenShell should be an **optional containment and credential-control layer
around a genai-tk worker**. It should not replace genai-tk's agent harness, model
factory, Deep Agents, DeerFlow, MCP registry, or NeMo Relay.

The recommended shape is:

```text
host application / orchestrator
    |
    | authenticated OpenShell API
    v
OpenShell Gateway
    |
    +-- compute driver
    |
    +-- per-sandbox Supervisor
            |
            +-- policy and provider credentials
            |
            v
        genai-tk worker process
        (LangChain / Deep Agents / DeerFlow)
            |
            +-- approved model, MCP, data, and SaaS endpoints
            +-- NeMo Relay -> OpenShell OTLP relay -> collector
```

The whole-worker design is preferred because it contains:

- model clients;
- custom Python tools;
- LangChain and LangGraph execution;
- Deep Agent shell and file tools;
- DeerFlow and its local tools;
- MCP clients and local stdio MCP servers;
- NeMo Relay and application dependencies;
- provider placeholders and their request paths.

The alternative—implementing `SandboxBackendProtocol` only for one Deep Agent
tool backend—protects fewer surfaces. It remains a possible later extension,
not the primary security design.

## 1. Rationale

### 1.1 Why a container alone is not enough

A conventional worker container usually receives a real API key, retains broad
outbound network access, and relies on its mounts and application code to
enforce security. That is sufficient for trusted, deterministic jobs but not
for an autonomous agent with shell access, generated code, browser tools,
dynamic packages, or many external APIs.

OpenShell adds:

- deny-by-default outbound network mediation;
- binary-specific and endpoint-specific policy;
- HTTP method/path, GraphQL, WebSocket, MCP, and native TCP controls;
- opaque provider placeholders instead of real credentials in the workload;
- endpoint-bound credential substitution;
- Landlock filesystem policy;
- seccomp and non-root/no-capability process controls;
- OCSF security events;
- workspace, OIDC/mTLS, Kubernetes, and gateway operating surfaces;
- an OTLP relay that can tag agent spans with a sandbox identity.

OpenShell therefore addresses deployment and trust boundaries that genai-tk
does not currently own.

### 1.2 Why OpenShell should remain outside genai-tk orchestration

genai-tk already owns:

- agent profile selection;
- model and embedding construction;
- LangChain and DeerFlow harness execution;
- middleware and context propagation;
- sandbox-tool protocol selection;
- MCP registration;
- NeMo Relay semantic agent spans and ATOF trajectories.

Duplicating those responsibilities in a gateway would create two orchestration
systems. OpenShell should instead provide the runtime boundary in which the
existing genai-tk process runs.

### 1.3 Why the whole-worker boundary is preferred

OpenShell is framework-neutral. A genai-tk worker needs no special LangChain,
Deep Agents, or DeerFlow connector when the complete process is sandboxed.

Whole-worker containment also avoids:

- nested Docker;
- Docker socket mounts;
- running OpenSandbox inside OpenShell;
- split policy between host model clients and sandboxed tools;
- host-only custom tools bypassing the sandbox policy.

## 2. Current design

### 2.1 Host orchestrator

The host orchestrator is application-owned. It is not OpenShell's Supervisor.

It should:

1. authenticate the caller;
2. determine tenant, profile, model, workspace, and retention policy;
3. call a pinned OpenShell gateway through mTLS or OIDC;
4. create or lease a sandbox from a versioned template;
5. attach pre-created provider records;
6. pass only non-secret environment variables;
7. start the canonical genai-tk command;
8. stream stdout/events to the caller;
9. retain sandbox ID, genai-tk run ID, and thread ID for correlation;
10. export required trajectories and delete the sandbox when appropriate.

The host must never pass its `.env`, OpenShell operator credentials, or real
provider credentials to the worker.

### 2.2 Worker image

The worker image should contain:

- Python 3.12;
- the project package and a locked dependency set;
- only the extras needed by that worker profile;
- `config/`, safe skill files, and immutable runtime assets;
- any generated clients, such as BAML output;
- a model metadata snapshot if the runtime would otherwise fetch it;
- an explicit non-root user;
- no package manager or build toolchain unless the worker truly requires it;
- no Docker socket, gateway credentials, or provider secrets.

The image entry point should be an explicit Python command. Prefer:

```text
python -m genai_tk.main.cli agents run <profile> ...
```

The canonical process must be declared explicitly when creating the OpenShell
sandbox. Do not rely on an image's development-oriented `CMD`.

### 2.3 Sandbox policy

The base worker policy should:

- require Landlock rather than silently accepting a best-effort fallback;
- set a numeric non-root workload identity;
- grant only the minimum readable runtime paths;
- grant writable access only to a designated workspace plus temporary paths;
- leave outbound network denied unless a provider/profile contributes a rule;
- avoid broad host, binary, method, and wildcard grants.

A minimal whole-worker policy for an image with `/app`, `/opt/venv`, and UID/GID
1000 is:

```yaml
version: 1
filesystem_policy:
  include_workdir: true
  read_only:
    - /app
    - /opt/venv
    - /usr
    - /lib
    - /etc
  read_write:
    - /app/data
    - /tmp
    - /home/appuser
landlock:
  compatibility: hard_requirement
process:
  run_as_user: "1000"
  run_as_group: "1000"
```

The exact paths and identity must match the final image.

### 2.4 Provider profile

Provider profiles define:

- the environment variable expected by the application;
- the credential storage name;
- allowed endpoints and port;
- protocol and L7 enforcement;
- credential placement;
- exact binary paths allowed to use that credential.

Conceptually, an OpenRouter profile used by a Python worker is:

```yaml
id: project-openrouter
display_name: Project OpenRouter
category: inference
inference_capable: true
credentials:
  - name: api_key
    env_vars: [OPENROUTER_API_KEY]
    required: true
    auth_style: bearer
    header_name: authorization
discovery:
  credentials: [api_key]
endpoints:
  - host: openrouter.ai
    port: 443
    protocol: rest
    access: read-write
    enforcement: enforce
binaries:
  - /usr/local/bin/python3.12
  - /opt/venv/bin/python3.12
```

Copy and tailor provider profiles. A binary path or base URL change can
invalidate the binding.

### 2.5 LangChain, Deep Agents, and DeerFlow

#### LangChain and custom graphs

No graph changes are required. Run the complete CLI or AUIP worker inside
OpenShell and attach model/MCP/data providers.

#### Deep Agents

Use the whole worker with `backend: none` or an in-process/local backend.
Do not select the OpenSandbox Docker backend inside OpenShell.

The current genai-tk implementation avoids importing the AIO/OpenSandbox adapter
when the backend is `none`. This allows a whole-worker image to omit optional
`aio-sandbox` packages.

An optional `OpenShellBackend` could later implement
`SandboxBackendProtocol` for deployments where the reasoning process remains
outside the sandbox. Such an adapter must:

- wrap the synchronous Python SDK without blocking the event loop;
- implement shell, file, search, upload, and download semantics;
- use argument arrays and safe path handling;
- provide idempotent lifecycle, cancellation, and orphan recovery;
- avoid assumptions about OpenSandbox's internal `_sandbox` object.

This adapter is not required for whole-worker containment.

#### DeerFlow

Run the entire DeerFlow worker inside OpenShell with `sandbox: local`. Do not
enable DeerFlow's nested Docker backend.

There is currently no drop-in OpenShell provider for individual DeerFlow tools.
Adding one would require a DeerFlow extension and a pinned compatibility test.

### 2.6 NeMo Relay and OpenShell telemetry

NeMo Relay creates semantic agent, model, and tool spans. OpenShell secures the
transport:

```text
genai-tk / NeMo Relay
    -> OTEL_EXPORTER_OTLP_ENDPOINT injected by OpenShell
    -> Supervisor relay
    -> operator-selected collector
```

OpenShell can add:

- `openshell.sandbox.id`;
- `openshell.telemetry.source=agent`;
- gateway and supervisor spans;
- OCSF security decisions;
- Prometheus metrics.

The current NeMo Relay setup accepts an explicitly configured `http://` OTLP
endpoint for OpenShell's private relay. Collector addresses and collector
credentials remain outside the workload.

ATOF trajectories remain files owned by genai-tk. For ephemeral sandboxes, the
host must download or persist `/app/data/trajectories` before deleting the
sandbox.

OTLP and ATOF can contain prompts, tool arguments, and responses. They must be
classified and retained as sensitive application data.

## 3. Current limitations

### 3.1 Not implemented in `cli init`

A scaffolded project does not currently receive:

- `deploy/openshell/`;
- a worker Dockerfile;
- OpenShell policy/provider profiles;
- a sandbox template;
- OpenShell Python dependency wiring;
- a host launch wrapper;
- OpenShell doctor checks;
- policy/ATOF export tests.

The deployment steps in this document must currently be added to the project
manually.

### 3.2 No genai-tk OpenShell client abstraction yet

Projects must either call the OpenShell CLI/SDK directly or create their own
wrapper. A future genai-tk wrapper should use Pydantic v2 configuration and
cover:

- gateway selection and authentication;
- sandbox create/start/exec/delete;
- provider attachment;
- bounded timeouts and retries;
- cancellation and idempotency;
- sandbox/trajectory correlation;
- explicit OpenShell error mapping;
- cleanup and orphan recovery.

### 3.3 No universal worker dependency profile

Scaffolded projects inherit a broad development dependency set. A production
worker must remove UI, notebook, OCR, model-training, CUDA, and development-only
dependencies when the selected profile does not use them.

An RFQ worker build, for example, had to exclude document ingestion/UI
dependencies that pulled Torch/CUDA through unrelated optional capability
paths. Each production worker should have an explicit, locked dependency
profile rather than mutating the general lockfile during container assembly.

### 3.4 Framework support differs

| Workload | Current integration |
|---|---|
| LangChain ReAct | Whole worker |
| Deep Agents | Whole worker; optional protocol backend not implemented |
| DeerFlow | Whole worker with `sandbox: local` |
| Remote MCP | Native network policy |
| Local stdio MCP | Include server in worker image |
| NeMo Relay | ATOF plus OpenShell OTLP relay |
| OpenSandbox/AIO tools | Do not nest inside OpenShell |

### 3.5 Security boundaries OpenShell does not provide

OpenShell does not:

- prevent prompt injection;
- decide whether an allowed action is appropriate for the business;
- scrub arbitrary prompts, responses, or logs;
- protect data after it is intentionally sent to an allowed endpoint;
- replace application authentication or tenant authorization;
- make an audit-only or L4-only rule equivalent to L7 enforcement;
- make a vulnerable worker image safe;
- replace image scanning, signing, SBOM review, and dependency patching;
- provide a production HA database in the Helm chart.

Kubernetes security also depends on a CNI that actually enforces ingress and
egress `NetworkPolicy`.

### 3.6 Platform and lifecycle constraints

- Linux requires the documented Landlock/seccomp capability baseline.
- WSL 2 is experimental in the evaluated OpenShell release.
- Static filesystem/process changes require sandbox recreation.
- Newly attached provider placeholders may require a newly launched process.
- Binary authorization is sensitive to executable path and identity.
- Agent traces and OCSF records require an external durable collector.
- Kubernetes HA requires externally operated PostgreSQL.

## 4. Deploy a scaffolded genai-tk project

This is the current manual procedure. Replace project names, profile names,
endpoints, and paths with the values for the generated project.

### Step 1: Scaffold and identify the worker

```bash
uv run cli init --name "My Agent"
uv sync
```

Select or define:

- one agent profile;
- its model provider;
- its writable paths;
- its required MCP/data/browser endpoints;
- its required optional extras;
- its canonical command.

For a first deployment, choose a small model-only Deep Agent and deliberately
avoid tools that require private databases, dynamic packages, or browser
automation.

### Step 2: Add a worker-only dependency profile

Create a production dependency set that excludes development and unrelated
profile extras. Commit its lockfile.

Verify:

```bash
uv tree
uv sync --locked
uv run cli agents list
```

Do not install `aio-sandbox` merely because a Deep Agent uses `backend: none`.

### Step 3: Build a non-root worker image

Add a `deploy/openshell/Dockerfile` that:

- uses Python 3.12;
- copies the locked project;
- installs only the worker dependencies;
- copies `config/`, package code, and safe skills;
- copies generated clients not included by the Python package;
- excludes `.env`, tests, notebooks, build tools, and uploaded customer data;
- creates an explicit non-root user;
- writes only to approved workspace paths;
- declares a stable Python entry point.

Build and inspect it:

```bash
docker build \
  -f deploy/openshell/Dockerfile \
  -t my-agent-openshell:dev \
  .

docker image inspect my-agent-openshell:dev \
  --format 'user={{.Config.User}} size={{.Size}}'
```

The image should not contain real credentials or a Docker socket.

### Step 4: Add the sandbox policy

Create `deploy/openshell/policy.yaml` from the minimal policy above. Add only
filesystem paths that the real image uses. Provider profiles contribute their
own network rules; do not duplicate a broad allowlist in the base policy.

Validate the file:

```bash
openshell policy lint deploy/openshell/policy.yaml
```

### Step 5: Add and validate provider profiles

For each model, MCP, or tool credential:

1. copy the matching OpenShell example profile;
2. replace endpoints with the actual base URL and port;
3. replace binary paths with paths from the final image;
4. choose `enforce` and the narrowest access preset;
5. lint and import the profile.

```bash
openshell profile lint -f deploy/openshell/openrouter.yaml
openshell profile import -f deploy/openshell/openrouter.yaml --global
```

Create the provider from operator-held environment state:

```bash
openshell provider create \
  --name my-agent-openrouter \
  --type project-openrouter \
  --from-existing \
  --global-profile
```

Do not store the resulting secret in the worker image, manifest, or repository.

### Step 6: Install and qualify the gateway

Pin the stable OpenShell release used by the project. For local evaluation, use
the documented Docker or package-managed gateway with mTLS and loopback
publication.

For production:

- pin gateway, supervisor, and sandbox runtime artifacts;
- enable OIDC or another authenticated gateway mode;
- qualify the native Linux kernel or Kubernetes CNI;
- configure a durable OCSF/OTLP collector;
- disable or explicitly govern anonymous OpenShell telemetry.

Confirm the gateway before launching a worker:

```bash
openshell status
```

### Step 7: Run one canonical command

Use the explicit canonical command rather than relying on `CMD`:

```bash
openshell sandbox create \
  --name my-agent-smoke \
  --from my-agent-openshell:dev \
  --provider my-agent-openrouter \
  --policy deploy/openshell/policy.yaml \
  --cpu 2 \
  --memory 4Gi \
  --label project=my-agent \
  --env GENAITK_PROFILE=local \
  --no-tty \
  -- python -m genai_tk.main.cli agents run default \
     "Reply exactly with OPENSHELL_OK" --raw
```

For a retained service, use `--detach` and run the project's API/AUIP worker
instead of a one-shot CLI command.

### Step 8: Verify the security gates

Do not promote an integration until all of these checks pass:

1. the real provider key is absent from the workload environment, command line,
   and image;
2. the workload runs as the expected non-root UID/GID;
3. `/var/run/docker.sock` is absent;
4. approved workspace writes succeed and unapproved path writes fail;
5. the approved model request succeeds;
6. an unapproved destination fails;
7. the deny decision appears in OpenShell logs/OCSF;
8. the model request appears as an allowed L4/L7 decision;
9. NeMo Relay writes an ATOF trajectory;
10. an agent OTLP span contains `openshell.sandbox.id`;
11. no collector credential or real provider secret appears in workload output.

A simple denial probe is:

```bash
openshell sandbox exec -n my-agent-smoke -- \
  python -c \
  'import urllib.request; urllib.request.urlopen("https://example.com", timeout=5)'
```

The command must fail for a policy that does not allow `example.com`.

### Step 9: Export state before deletion

An ephemeral sandbox does not provide durable ATOF storage. Export it first:

```bash
openshell sandbox download \
  my-agent-smoke \
  /app/data/trajectories \
  ./out/trajectories
```

Then delete the sandbox:

```bash
openshell sandbox delete my-agent-smoke
```

For a retained multi-turn worker, map the tenant, genai-tk thread, OpenShell
sandbox, and trajectory identifiers to a durable workspace.

### Step 10: Add CI and production controls

Before production use:

- lint every policy and provider profile;
- run the OpenShell policy prover;
- scan and sign the worker image;
- publish an immutable image digest;
- test no-secret image layers and environment output;
- test CNI enforcement rather than assuming `NetworkPolicy` works;
- test cancellation, timeout, deletion, and orphan recovery;
- test gateway restart and interrupted output streaming;
- define OTLP/OCSF/ATOF retention and redaction;
- define PostgreSQL backup and recovery if gateway HA is enabled;
- retain a known-good image and policy for rollback.

## 5. Observed proof points

A local OpenShell `v0.1.10` run with a real genai-tk RFQ Deep Agent proved the
basic design:

- a real model call succeeded through OpenRouter;
- the workload received an opaque credential placeholder;
- an unapproved endpoint was denied with an OCSF event;
- no Docker socket was present;
- the worker ran as UID/GID 1000;
- an ATOF trajectory was exported;
- NeMo Relay spans carried the exact OpenShell sandbox ID;
- the agent ran without optional OpenSandbox packages.

That evidence is useful, but it is not production qualification. The test host
was WSL 2, which the evaluated OpenShell release classifies as experimental.

## 6. What to do next

The next genai-tk work should remove the manual gaps rather than add an
agent-specific OpenShell adapter:

1. Add `deploy/openshell/` to scaffold output behind an optional `init` flag.
2. Generate a worker-only dependency extra from the selected agent profile.
3. Add Pydantic v2 configuration models for gateway, image, template,
   providers, labels, retention, and timeouts.
4. Add a small host execution client around the OpenShell Python SDK.
5. Add `cli info doctor` checks for gateway authentication, policy/provider
   readiness, OTLP relay, and writable workspace access.
6. Add capability-oriented policy templates:
   - model-only;
   - MCP;
   - browser;
   - code execution;
   - private database/graph.
7. Add negative policy tests and CI policy-prover checks.
8. Make ATOF export part of sandbox teardown.
9. Resolve trace service naming and explicit redaction configuration.
10. Qualify the same worker on supported native Linux and Kubernetes.
11. Implement a Deep Agents `OpenShellBackend` only after a concrete use case
    requires reasoning outside the sandbox.

## References

- [OpenShell repository](https://github.com/ai-openshell/openshell)
- [OpenShell architecture](https://docs.nvidia.com/openshell/latest/about/architecture)
- [OpenShell support matrix](https://docs.nvidia.com/openshell/latest/about/support-matrix)
- [OpenShell sandbox policies](https://docs.nvidia.com/openshell/latest/how-it-works/policies/overview)
- [OpenShell provider profiles](https://docs.nvidia.com/openshell/latest/how-it-works/providers/profiles)
- [OpenShell Python SDK](https://docs.nvidia.com/openshell/latest/sdk/python)
- [OpenShell observability](https://docs.nvidia.com/openshell/latest/observability/overview)
- [genai-tk sandbox support](../sandbox_support.md)
- [Agent harness and middleware design](05_agent_harness_and_middleware.md)
- [Sandbox and CodeAct design](06_sandbox_and_codeact.md)
- [Observability and trajectories design](09_observability_and_trajectories.md)
- [Project scaffolding](../scaffolding.md)
