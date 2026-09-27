# Sandbox Support & Execution Architecture

Sandboxes provide isolated, safe execution environments for AI agents to run shell commands, interact with filesystems, and execute Python code.

All agent frameworks in **genai-tk** (LangChain, DeepAgents, and DeerFlow) support sandboxes through a unified 3-layer architecture and shared configuration.

---

## 1. Architectural Layers: Decoupling Sandboxes, Protocols, and Executors

The sandbox subsystem is structured into three clean, decoupled layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Layer 3: Python Execution Tools (Agent Action Language)                     │
│  - LocalPythonExecutor: In-process AST-evaluated interpreter (no eval)      │
│  - SandboxedPythonExecutor: Persistent state + NumPy/Pandas + Host RPC      │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ executes on
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ Layer 2: Generic Runtime Protocol (SandboxBackendProtocol)                  │
│  Standard container contract: aexecute(), aread(), awrite(), aedit()        │
│  (Not Python-specific; works for shell commands, filesystem, search)        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ implemented by
┌──────────────────────────────────────▼──────────────────────────────────────┐
│ Layer 1: Isolation Runtimes (SandboxBackendFactory)                         │
│  - Local Host: deepagents.backends.local_shell:LocalShellBackend            │
│  - OpenSandbox Docker: genai_tk.agents.sandbox:AioSandboxBackend            │
│  - Cloud Sandboxes: E2B, Modal, Daytona, Fly Machines (via YAML class path) │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Layer 1: Isolation Runtimes (Infrastructure)
Where execution physically occurs:
* **Local Host (`local`)**: Runs commands directly in host processes. Fast, zero-overhead, suitable for development and trusted tasks.
* **OpenSandbox Docker (`docker` / `aio_sandbox`)**: The default all-in-one local container (`ghcr.io/agent-infra/sandbox:latest`, ~13GB) including Chromium with CDP + VNC, Python 3.11 with data science packages, Node.js, and an in-container `execd` daemon.
* **Cloud Sandboxes (`e2b`, `modal`, `daytona`)**: Managed remote microVMs or container endpoints for elastic scaling and zero local Docker dependencies.

### Layer 2: Generic Protocol (`SandboxBackendProtocol`)
Defined by `deepagents.backends.protocol.SandboxBackendProtocol`. It specifies a standard async contract:
* `aexecute(command, timeout)` — run shell commands
* `aread(path, offset, limit)` / `awrite(path, content)` / `aedit(path, old, new)` — file I/O
* `agrep_raw(pattern, path)` / `aglob_info(pattern, path)` — search and discovery
* `aupload_files(...)` / `adownload_files(...)` — bulk file transfer

> **Key Insight:** `SandboxBackendProtocol` is **not Python-specific**; it is a general runtime environment abstraction.

### Layer 3: Python Execution Actions (`BasePythonExecutor`)
How agents execute stateful Python code blocks:
* **`LocalPythonExecutor`**: In-process safe AST interpreter with import allow-lists and operation limits.
* **`SandboxedPythonExecutor`** (aliased to `DockerPythonExecutor`): Executes code inside any backend conforming to `SandboxBackendProtocol`, maintaining variable state across turns and bridging host tools via an ephemeral RPC server.

---

## 2. Decision Matrix: When to Use Which Runtime

| Runtime | Isolation Level | Cold Start | Data Science Libs | Browser & VNC | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`local`** | Process-only | Instant (0s) | Host-installed | Host Playwright | Local development, unit tests, trusted code |
| **`docker` (OpenSandbox)** | Full Container | ~5s (warm daemon) | Pre-installed (NumPy, Pandas, SciPy) | Built-in Chromium + VNC | Production, financial calculations, complex document QA, untrusted code |
| **`e2b` / `modal`** | Cloud MicroVM | ~2-4s | Custom template | Cloud headless | Production cloud deployments without local Docker daemon |

---

## 3. Extensibility: `SandboxBackendFactory`

New sandbox technologies (such as E2B, Modal, Daytona, or custom container runtimes) can be introduced **without modifying agent profiles or core harness code**.

### Registering via YAML Configuration

Declare the qualified Python class name in `config/sandbox.yaml` under `sandbox.backends`:

```yaml
sandbox:
  default: docker

  # Custom and pluggable backends:
  backends:
    docker:
      class: "genai_tk.agents.sandbox.aio_backend.AioSandboxBackend"
    local:
      class: "deepagents.backends.local_shell.LocalShellBackend"
    e2b:
      class: "my_package.sandboxes.e2b:E2bSandboxBackend"
    modal:
      class: "my_package.sandboxes.modal:ModalSandboxBackend"
```

### Programmatic Resolution & Instantiation

```python
from genai_tk.agents.sandbox import SandboxBackendFactory

# Create any registered backend:
backend = SandboxBackendFactory.create("docker")
# Or create custom cloud backend:
# e2b_backend = SandboxBackendFactory.create("e2b", api_key="sk-...")
```

---

## 4. Lifecycle & Session Management (`SandboxManager`)

To eliminate container startup churn (~28s per run), `SandboxManager` provides a thread-safe singleton that preserves sandbox backends across multi-turn agent sessions:

```python
from genai_tk.agents.sandbox import SandboxManager

# Get or lazily start the shared session backend
backend = await SandboxManager.aget_shared_backend()
```

---

## 5. OpenSandbox Setup & CLI Commands

When using the `docker` backend, genai-tk orchestrates **Alibaba OpenSandbox** and the `agent-infra/sandbox` image.

### Installation

```bash
# Install optional sandbox dependencies
uv sync --group aio-sandbox
# Or during project initialization
uv run cli init --with-sandbox
```

### CLI Management

```bash
cli sandbox start   # Start opensandbox-server daemon in background
cli sandbox pull    # Pre-pull the ~13GB Docker image
cli sandbox status  # Verify daemon health and image availability
cli sandbox stop    # Stop background daemon
```

### Running Agents with Sandboxes

```bash
# Run with local AST execution (fast)
cli agents run chat --sandbox local "Compute compound interest"

# Run with isolated Docker sandbox (with stateful NumPy / Pandas)
cli agents run research --sandbox docker "Analyze financial tables"

# Interactive multi-turn chat with container reuse
cli agents run "Browser Agent" --sandbox docker --chat
```

### Live VNC Viewer (Browser Agents)

When running browser agents in Docker, observe actions live in your browser:
```
http://localhost:8080/vnc/index.html?autoconnect=true
```

---

## 6. Shared Configuration Reference (`config/sandbox.yaml`)

```yaml
sandbox:
  default: local

  # Docker-specific OpenSandbox settings:
  docker:
    aio:
      image: "ghcr.io/agent-infra/sandbox:latest"
      opensandbox_server_url: "http://localhost:8080"
      startup_timeout: 60.0
      work_dir: "/home/user"
      env_vars:
        DEBUG: "0"
      # Skill directories are mounted read-only automatically to /mnt/skills
      volumes: []

  # Cloud Sandbox Settings (Optional):
  e2b:
    api_key: ${oc.env:E2B_API_KEY,null}
    timeout: 300
```

