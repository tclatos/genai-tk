# Sandbox Support

Sandboxes provide isolated, containerised environments for safe code execution.
All agent frameworks in genai-tk support sandboxes through a unified configuration.

## Terminology — who's who

Several similarly-named projects are involved. They are **not** competing
alternatives — each plays a distinct role in the stack:

| Name | What it is | Role in genai-tk |
|------|------------|-------------------|
| **DeepAgents** (`deepagents` PyPI package) | LangChain's agent framework. Defines the `SandboxBackendProtocol` interface consumed by the `deep` agent profile. Ships only `LocalShellBackend` — no Docker backend of its own. | The interface genai-tk's LangChain harness implements against. |
| **Alibaba OpenSandbox** (`opensandbox`, `opensandbox-server`, `agent-sandbox` PyPI packages) | An orchestrator SDK + local server that manages Docker container lifecycle and exposes an HTTP client. | Wrapped by genai-tk's [`AioSandboxBackend`](../genai_tk/agents/sandbox/aio_backend.py), genai-tk's own open-source `SandboxBackendProtocol` implementation for the LangChain harness. |
| **agent-infra/sandbox** ("AIO Sandbox") | The Docker **image** (`ghcr.io/agent-infra/sandbox:latest`, ~13GB) — Chromium, Python, Node.js, a REST shell/file API, and a VNC web viewer. | The container that OpenSandbox starts and manages — the payload, not the orchestrator. |
| **`opensandbox/execd`** | A small (~114MB) auxiliary image pinned by `opensandbox-server`'s Docker runtime (`[runtime].execd_image` in its TOML config). It runs the lightweight `execd` command/filesystem daemon that `AioSandboxBackend` actually talks HTTP to inside the sandbox container. | Downloaded automatically the first time `opensandbox-server` starts a Docker-backed sandbox — **not something you configure directly**. genai-tk pins its version in [`_OPENSANDBOX_EXECD_IMAGE`](../genai_tk/agents/sandbox/config.py) (currently `opensandbox/execd:v1.0.16`), written into the generated server TOML by `write_server_config()`. |
| **DeerFlow** (`deerflow-harness`) | A separate agent harness (ByteDance, LangGraph-based). Has its own sandbox provider, `deerflow.community.aio_sandbox:AioSandboxProvider`, which manages Docker containers directly and does **not** go through `opensandbox-server`. | An independent harness with an independent backend implementation. |

genai-tk deliberately **configures both harnesses to use the same
`ghcr.io/agent-infra/sandbox` image and the same `sandbox.docker.aio` config
block** (see [Shared sandbox config](#shared-sandbox-config-both-harnesses)
below) — this is a genai-tk design choice for consistency, not an inherent
property of DeepAgents or DeerFlow. Each harness is free to use a different
backend/image; nothing requires them to match. `opensandbox/execd` is
unrelated to that choice — it's an implementation detail of the LangChain
path only, pulled by `opensandbox-server` regardless of which AIO image you
point it at.

```
deepagents.SandboxBackendProtocol (interface)
  ← AioSandboxBackend (genai-tk, wraps Alibaba OpenSandbox SDK)
    ← opensandbox-server (orchestrator, manages Docker)
      ├─ pulls opensandbox/execd:v1.0.16  (in-container command/file daemon)
      └─ pulls ghcr.io/agent-infra/sandbox container (the actual sandbox)

deerflow AioSandboxProvider (independent implementation)
  ← manages ghcr.io/agent-infra/sandbox container directly (no opensandbox-server, no execd image)
```

## Sandbox Types

| Type | When to use | Docker required? |
|------|-------------|------------------|
| `local` | Development, testing, trusted code | ✗ (no) |
| `docker` | Production, untrusted code, full isolation | ✓ (yes) |

The `docker` type uses [OpenSandbox](https://github.com/alibaba/OpenSandbox)
(`ghcr.io/agent-infra/sandbox:latest`) — an open-source AIO container with
Chromium, Python, Node.js, a REST file/shell API, and a VNC web viewer.

---

## Installation

Sandbox support is **optional** (heavy dependencies). Install it only when needed:

### With `cli init` (recommended)

```bash
uv run cli init --with-sandbox

# This installs the aio-sandbox group:
# - agent-sandbox
# - opensandbox
# - opensandbox-server
```

### Manual installation

```bash
# Install the aio-sandbox optional group
uv sync --group aio-sandbox

# Or add individually
uv add agent-sandbox opensandbox opensandbox-server
```

### First-time setup (Docker only)

After installation, initialize the OpenSandbox server:

```bash
# Generate server config (~/.sandbox.toml)
opensandbox-server init-config ~/.sandbox.toml --example docker

# Verify installation
cli sandbox status
```

### Warm the server (recommended)

Run these once per machine boot to cut per-invocation latency from ~28 s to ~5 s:

```bash
cli sandbox start   # start the background daemon
cli sandbox pull    # pre-pull the Docker image
cli sandbox status  # verify everything is ready
```

---

## CLI commands

| Command | Purpose |
|---------|---------|
| `cli sandbox start` | Start opensandbox-server as a background daemon |
| `cli sandbox stop` | Stop the daemon |
| `cli sandbox status` | Show daemon health, image cache, and HTTP reachability |
| `cli sandbox pull` | Pre-pull the Docker image |

---

## Architecture

```
Agent Code (host process)
    │
    └─ AioSandboxBackend (HTTP client)
            │  auto-starts opensandbox-server if not running
            │  HTTP
            ▼
       OpenSandbox Server (localhost:8080)
            │  manages Docker lifecycle + port allocation
            ▼
       ghcr.io/agent-infra/sandbox container
            ├─ REST API (shell, file, Jupyter)
            ├─ Chromium (CDP + VNC)
            └─ /mnt/skills/   ← skill dirs mounted here
```

The `AioSandboxBackend` auto-starts `opensandbox-server` if not already running,
using the binary from your Python environment (compatible with `uv`/virtualenvs).

---

## Using sandboxes with agents

### Local sandbox (no Docker, fast)

```bash
# DeerFlow profile — sandbox selected via --sandbox
cli agents run chat --sandbox local "write code and run it"

# LangChain deep agent — sandbox is profile-driven (set backend: aio_sandbox
# on the profile); --sandbox is ignored for LangChain profiles
cli agents run coding "write code and run it"
```

Or in Python:

```python
from genai_tk.agents.langchain.langchain_agent import LangchainAgent

agent = LangchainAgent(llm="gpt_41mini@openai", sandbox="local")
result = await agent.run("Write a Python script and run it")
```

### Docker sandbox (production, isolated)

**Prerequisite**: Must have Docker running and sandbox installed via `cli init --with-sandbox`.

```bash
# Start the sandbox server once per boot
cli sandbox start && cli sandbox pull

# DeerFlow profile — docker sandbox
cli agents run research --sandbox docker --chat

# LangChain deep agent — sandbox from profile backend config
cli agents run coding "rm -rf /important/files"
```

### Browser agents with Docker

Observe the browser live via VNC while the agent is working:

```bash
cli agents run "Browser Agent" --chat "Find the weather"

# In another terminal:
# Open http://localhost:8080/vnc/index.html?autoconnect=true
```

---

## Configuration

All sandbox settings live in `config/sandbox.yaml`:

```yaml
sandbox:
  default: "local"             # Use local by default
  docker:
    aio:
      opensandbox_server_url: "http://localhost:8080"
      startup_timeout: 60.0    # wait 60s for server to start
      work_dir: "/home/user"
      entrypoint: ["/opt/gem/run.sh"]
      env_vars: {}             # extra env vars passed to container
```

### Volume mounts and skills

When a `deep` agent runs with `--sandbox docker`, skill directories are
automatically bind-mounted into the container **read-only**:

```
Host: ~/project/skills/custom  →  Container: /mnt/skills/custom  (read-only)
```

This happens automatically — no manual configuration needed. See
[browser_control.md](browser_control.md#skills-site-specific-knowledge) for
details about browser agent skills.

### Shared sandbox config (both harnesses)

`sandbox.docker.aio` in `config/sandbox.yaml` is the single source of truth for
**both** the LangChain deepagent harness (`AioSandboxBackend`) and the DeerFlow
harness (`AioSandboxProvider`). `env_vars` and `volumes` are forwarded into the
generated DeerFlow `config.yaml` (`environment` / `mounts`) so configuring them
once applies to either runtime:

```yaml
sandbox:
  docker:
    aio:
      env_vars:
        DEBUG: "1"
      volumes:
        - host_path: /home/me/data
          container_path: /mnt/data
          read_only: true
```

Note: the LangChain harness starts the sandbox via `opensandbox-server` while
DeerFlow's provider manages Docker containers directly; `opensandbox_server_url`
is only used by the LangChain path.

---

## Advanced usage

### Multi-turn chat (container reuse)

For interactive sessions, `--chat` keeps the agent (and its sandbox container)
alive across turns, avoiding per-turn startup overhead:

```bash
cli agents run "Browser Agent" --chat
```

### VNC viewer

Watch the browser agent work in real-time:

```
http://localhost:8080/vnc/index.html?autoconnect=true
```

Use the VNC viewer to:
- See what the browser is doing
- Manually intervene (click, type) if needed
- Debug agent navigation issues

### Custom work directory

Change where code executes inside the container:

```yaml
sandbox:
  docker:
    aio:
      work_dir: "/workspace"
```

### Environment variables

Pass secrets/config into the sandbox:

```yaml
sandbox:
  docker:
    aio:
      env_vars:
        API_KEY: ${oc.env:MY_API_KEY}
        DEBUG: "1"
```

---

## Security

The Docker sandbox provides strong isolation:

| Aspect | Details |
|--------|---------|
| **Filesystem** | Container cannot see host SSH keys, `.aws/`, `.kube/`, etc. |
| **Network** | Container has its own network namespace |
| **Processes** | Kernel namespaces prevent container from signalling host |
| **Skills** | Mounted read-only — container cannot modify host skills |
| **Resources** | Configure limits: `--memory 2g --cpus 2` (in production) |

### Production security practices

1. **Rotate secrets** — Clear `data/sessions/` after changing API keys
2. **Resource limits** — Keep memory/CPU limits reasonable
3. **Docker socket** — Use non-privileged socket (`DOCKER_HOST`)
4. **Image updates** — Periodically pull latest sandbox image: `cli sandbox pull`

---

## Troubleshooting

**Q: "aio-sandbox is not installed"**

Install it:
```bash
uv sync --group aio-sandbox
# or
cli init --with-sandbox
```

**Q: "opensandbox-server is not running"**

The backend auto-starts it. If it doesn't:
```bash
cli sandbox start
cli sandbox status
```

**Q: Docker connection refused**

Verify Docker is running and the socket is accessible:
```bash
docker ps
echo $DOCKER_HOST
```

**Q: Container startup timeout (60s exceeded)**

The sandbox image is being pulled for the first time. This is slow on poor connections.
Run `cli sandbox pull` once during setup to pre-cache it.

**Q: Too slow for interactive use**

Use `--sandbox local` instead. `docker` has ~5-10s startup overhead per invocation.

**Q: How do I run production code safely?**

1. Use `--sandbox docker` with resource limits
2. Use `--keep-sandbox` for multi-turn sessions
3. Monitor container resources: `docker stats`
4. Clear session data after rotating secrets

**Q: I see `opensandbox/execd:v1.0.16` in `docker images` — where did that come from?**

`opensandbox-server` pulls it automatically the first time it starts a
Docker-backed sandbox — it's the in-container command/filesystem daemon that
`AioSandboxBackend` talks to, separate from the `ghcr.io/agent-infra/sandbox`
AIO image. You didn't configure it directly; it's pinned in genai-tk via
`_OPENSANDBOX_EXECD_IMAGE` in
[`genai_tk/agents/sandbox/config.py`](../genai_tk/agents/sandbox/config.py).
See [Terminology — who's who](#terminology--whos-who) for the full picture.

---

## Implementation reference

See [design/sandbox_backend.md](design/sandbox_backend.md) (if present) for
internal architecture and the `AioSandboxBackend` protocol implementation.

## Related projects & future considerations

- **Per-harness backends are independent by design.** Nothing requires the
  LangChain `deep` agent and DeerFlow to share a container image or backend
  implementation — genai-tk currently points both at
  `ghcr.io/agent-infra/sandbox` as a deliberate choice for operational
  consistency (one image to pull/cache, one config block). Swapping either
  harness to a different backend does not affect the other.
- **[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)** (Apache-2.0) is a
  policy-enforcing sandbox runtime (Rust gateway + supervisor, Docker/Podman/
  MicroVM/K8s drivers) built for coding-agent CLIs, with declarative YAML
  network/filesystem/process egress policies and credential injection that
  `AioSandboxBackend` does not currently provide. It's alpha software today
  (single-tenant, "proof-of-life") but worth revisiting as a
  `SandboxBackendProtocol` implementation once it stabilizes — particularly
  for use cases needing network egress policy enforcement rather than plain
  container isolation.
