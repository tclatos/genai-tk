# Sandbox Support

Sandboxes provide isolated, containerised environments for safe code execution.
All agent frameworks in genai-tk support sandboxes through a unified configuration.

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
# LangChain
cli agents langchain --sandbox local "write code and run it"

# Deer-flow
cli agents deerflow --sandbox local --chat

# SmolAgents
cli agents smolagents --executor local "build a calculator"
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

# LangChain
cli agents langchain --sandbox docker "rm -rf /important/files"

# Deer-flow with code execution
cli agents deerflow -p research --sandbox docker --chat

# SmolAgents
cli agents smolagents --executor docker "write and test code"
```

### Browser agents with Docker

Observe the browser live via VNC while the agent is working:

```bash
cli agents langchain -p "Browser Agent" --sandbox docker --chat "Find the weather"

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

---

## Advanced usage

### Keep-sandbox flag (multi-turn chat)

For interactive chat sessions, reuse the same container across turns to avoid
per-turn startup overhead:

```bash
cli agents langchain -p "Browser Agent" --sandbox docker --keep-sandbox --chat
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

---

## Implementation reference

See [design/sandbox_backend.md](design/sandbox_backend.md) (if present) for
internal architecture and the `AioSandboxBackend` protocol implementation.
