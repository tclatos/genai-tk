# Sandbox Support

Sandboxes provide isolated, containerised environments for safe code execution.
All agent frameworks in genai-tk support sandboxes through a unified configuration.

## Sandbox Types

| Type | When to use |
|------|-------------|
| `local` | Development, trusted code, no Docker needed |
| `docker` | Production, untrusted code, full isolation |

The `docker` type uses [OpenSandbox](https://github.com/alibaba/OpenSandbox)
(`ghcr.io/agent-infra/sandbox:latest`) — an open-source AIO container with
Chromium, Python, Node.js, a REST file/shell API, and a VNC web viewer.

## Setup

### One-time initialisation

```bash
# Install OpenSandbox server
uv add opensandbox-server

# Generate server config (~/.sandbox.toml)
opensandbox-server init-config ~/.sandbox.toml --example docker
```

### Warm the server (recommended)

Run these once per machine boot to cut per-invocation latency from ~28 s to ~5 s:

```bash
cli sandbox start   # start the background daemon
cli sandbox pull    # pre-pull the Docker image
cli sandbox status  # verify everything is ready
```

### `cli sandbox` commands

| Command | Purpose |
|---------|---------|
| `cli sandbox start` | Start opensandbox-server as a background daemon |
| `cli sandbox stop` | Stop the daemon |
| `cli sandbox status` | Show daemon health, image cache, and HTTP reachability |
| `cli sandbox pull` | Pre-pull the Docker image |

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
            └─ /mnt/skills/   ← skill dirs mounted here (deep agents)
```

The `AioSandboxBackend` auto-starts `opensandbox-server` using the binary from
the active Python environment (compatible with `uv`/virtualenvs).

## Using Sandboxes with Agents

### LangChain agents

```bash
cli agents langchain --sandbox docker "bash ls -la /home/user"
cli agents langchain --sandbox local "run some code"
```

Or in code:
```python
from genai_tk.agents.langchain.langchain_agent import LangchainAgent

agent = LangchainAgent(llm="gpt_41mini@openai", sandbox="docker")
await agent.run("Write a Python script and run it")
```

### Deer-flow

```bash
cli agents deerflow --sandbox docker -p "Research Assistant" --chat
```

### SmolAgents

```bash
cli agents smolagents --executor docker "write and execute code"
cli agents smolagents --executor e2b "cloud execution"   # requires E2B_API_KEY
```

## Configuration

All sandbox settings live in `config/sandbox.yaml`:

```yaml
sandbox:
  default: "local"
  docker:
    aio:
      opensandbox_server_url: "http://localhost:8080"
      startup_timeout: 60.0
      work_dir: "/home/user"
      entrypoint: ["/opt/gem/run.sh"]
      env_vars: {}
```

### Volume mounts and skills (deep agents with Docker)

When a `deep` agent runs with `--sandbox docker`, skill directories are
automatically bind-mounted into the container so the `SkillsMiddleware` can
read them via the container filesystem API:

```
Host: ~/project/skills/custom  →  Container: /mnt/skills/custom  (read-only)
```

This happens automatically — no manual configuration needed. See
[browser_control.md](browser_control.md#skills-site-specific-knowledge) for
details about browser agent skills.

## Keep-Sandbox Flag

For multi-turn chat, use `--keep-sandbox` to reuse the same container across
turns (avoids per-turn startup overhead):

```bash
cli agents langchain -p "Browser Agent" --sandbox docker --keep-sandbox --chat "..."
```

## VNC Access

When using the Docker sandbox with a browser agent, observe the browser live:

```
http://localhost:8080/vnc/index.html?autoconnect=true
```

## Security Notes

The Docker sandbox provides strong isolation:

- **Filesystem**: container cannot see host SSH keys, `.aws/`, `.kube/`, etc.
- **Network**: container has its own network namespace.
- **Processes**: kernel namespaces prevent container processes from signalling host.

Skill directories are mounted **read-only**. The container cannot modify host skills.

For production:
- Enable Docker resource limits: `--memory 2g --cpus 2`
- Keep `DOCKER_HOST` pointing to a non-privileged socket
- Rotate secrets regularly; clear `data/sessions/` after credential rotation

## Implementation Reference

See [design/sandbox_backend.md](design/sandbox_backend.md) for the
`AioSandboxBackend` protocol implementation details and low-level API.
