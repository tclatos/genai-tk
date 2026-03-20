# Sandbox Support

Unified sandbox integration for secure, isolated code execution across all AI agent frameworks in GenAI Toolkit.

## Overview

Sandbox support enables agents to safely execute arbitrary code, scripts, and tools in isolated environments. This document covers:

- **Why OpenSandbox**: Rationale for the architecture choice
- **Architecture**: Unified configuration system and framework integration
- **Available Sandboxes**: Local and Docker-based options
- **CLI Usage**: Running agents with sandbox activation
- **Configuration**: YAML-based settings and customization
- **Integration Patterns**: Using sandboxes in your applications

## Why OpenSandbox?

### The Migration Story

Previously, sandbox execution used direct Docker subprocess management in Python:
- Each agent created its own container via `docker run` commands
- Port allocation was manual and error-prone (fixed host ports like 18091)
- Container cleanup was complex (tracking stale containers, handling race conditions)
- Configuration was scattered and framework-specific

### Why OpenSandbox (KISS Principle)

We migrated to [Alibaba OpenSandbox](https://github.com/alibaba/OpenSandbox) to **simplify** the architecture:

1. **Lifecycle Management**: OpenSandbox server handles all Docker container lifecycle — pulling images, creating, healthchecking, and cleanup
2. **Dynamic Port Allocation**: No more manual port management; OpenSandbox dynamically allocates ports
3. **Separation of Concerns**: Backend focuses on HTTP communication; server focuses on infrastructure
4. **Reduced Code**: Eliminated ~500 lines of Docker subprocess handling, container tracking, stale process cleanup, and port conflict resolution
5. **Reusability**: Single server instance serves multiple concurrent sandboxes via HTTP
6. **Reliability**: Server handles retries, timeouts, and edge cases we'd have to reimplement

### Sandbox Benefits

Agents often need to execute untrusted code:
- Run user-provided scripts
- Execute bash commands for file operations
- Perform computational tasks (math, data analysis)
- Interact with external tools and APIs

Sandboxes provide:
- **Process isolation** — Code runs in a separate Docker container with filesystem namespace
- **Resource limits** — CPU, memory, and process constraints managed by Docker
- **File isolation** — Sandbox working directory isolated from host
- **Network control** — Configurable network access restrictions
- **Audit trail** — Execution history logged via HTTP API

## Architecture

### OpenSandbox Server

The sandbox backend delegates all infrastructure concerns to the OpenSandbox server:

```
Agent Code
    │
    ├─ AioSandboxBackend (HTTP client)
    │       │
    │       └─ Auto-starts opensandbox-server if not running
    │              │
    │              ├─ Manages Docker container lifecycle
    │              ├─ Allocates dynamic ports
    │              ├─ Healthchecks containers
    │              └─ Cleans up on shutdown
    │
    └─ OpenSandbox Server (REST API at config.opensandbox_server_url)
            │
            └─ Docker container (ghcr.io/agent-infra/sandbox)
                 ├─ /v1/shell/exec_command
                 ├─ /v1/file/read_file
                 ├─ /v1/file/write_file
                 └─ /v1/file/replace_in_file
```

**Key Feature:** The backend automatically starts `opensandbox-server` if it detects the server isn't running at the configured URL. The server binary is resolved from the Python environment's bin directory, ensuring compatibility with virtual environments.

### Initialization

Before running sandboxed agents, initialize the OpenSandbox configuration once:

```bash
# Install server (if not already in dependencies)
uv add opensandbox-server

# Initialize configuration
opensandbox-server init-config ~/.sandbox.toml --example docker
```

This creates `~/.sandbox.toml` with Docker-based sandbox templates. The backend will use this configuration when auto-starting the server.

All sandbox technologies are configured through a single YAML file: `config/basic/sandbox.yaml`

```yaml
sandbox:
  default: "local"  # Default execution environment
  docker:
    aio:  # AioSandboxBackend (LangChain, DeepAgent, Deer Flow)
      opensandbox_server_url: "http://localhost:8080"
      startup_timeout: 60.0
      work_dir: "/home/user"
      entrypoint: ["/opt/gem/run.sh"]
      env_vars: {}
```

### Module Structure

```
genai_tk/agents/sandbox/
├── __init__.py          # Public API exports
├── models.py            # Pydantic configuration models
├── config.py            # Configuration loader functions
└── aio_backend.py       # AioSandboxBackend implementation (OpenSandbox-based)
```

**Key Classes:**

- `SandboxConfig` — Top-level configuration (Docker, E2B, WASM settings)
- `DockerAioSettings` — Docker sandbox settings for AioSandboxBackend (OpenSandbox-based lifecycle)
  - `opensandbox_server_url` — URL for OpenSandbox server (default: `http://localhost:8080`)
  - `entrypoint` — Container entrypoint (default: `["/opt/gem/run.sh"]`)
  - `startup_timeout` — API healthcheck timeout in seconds
  - `work_dir` — Default working directory in sandbox
  - `env_vars` — Extra environment variables for container
- `AioSandboxBackend` — deepagents `SandboxBackendProtocol` implementation

### Framework Integration

Each AI framework integrates sandbox support via command-line options and configuration:

#### LangChain (`genai_tk.agents.langchain`)

- **CLI Option**: `--sandbox local|docker` (`-b local|docker`)
- **Config**: `SandboxConfig` in `LlmFactory` and `LangchainAgent`
- **Files**: `commands.py`, `config.py`, `langchain_agent.py`, `sandbox_backend.py`
- **Supported Types**: Local (subprocess execution), Docker (AioSandboxBackend with OpenSandbox)

**Usage:**
```bash
uv run cli agents langchain --sandbox docker "Write to file.txt"
uv run cli agents langchain --sandbox local "Calculate something"
```

#### DeepAgent CLI (`genai_tk.agents.deepagent_cli`)

- **CLI Option**: `--sandbox local|docker`
- **Config**: Passed to `deep_agent/run` via `sandbox_bridge.py`
- **Files**: `cli_commands.py`, `models.py`, `sandbox_bridge.py`
- **Supported Types**: Local (subprocess), Docker (AioSandboxBackend with OpenSandbox)

**Usage:**
```bash
uv run cli agents deepagent --sandbox docker "Analyze code"
```

#### Deer Flow (`genai_tk.agents.deer_flow`)

- **CLI Option**: `--sandbox local|docker` 
- **Config**: Passed via `config_bridge.py` to the Deer Flow client
- **Files**: `cli_commands.py`, `config_bridge.py`
- **Supported Types**: Local, Docker (AioSandboxBackend)
- **Note**: `--input`/`-i` option added for consistent input handling

**Usage:**
```bash
uv run cli agents deerflow --input "perform calculation" --sandbox docker
```

#### SmolAgents (`genai_tk.agents.smolagents`)

- **CLI Option**: `--executor local|docker|e2b`
- **Config**: Integrated into SmolAgents' native executor selection
- **Files**: `commands.py`
- **Supported Types**: Local (thread-based), Docker (native `DockerExecutor`), E2B (cloud)

**Usage:**
```bash
uv run cli agents smolagents --executor docker "Write and execute code"
```

## Available Sandboxes

### Local Sandbox

**Type**: Process/thread-based execution on the host machine

**Pros:**
- No external dependencies or containers required
- Fastest startup time
- Direct filesystem access (careful!)

**Cons:**
- Limited isolation (shared OS resources)
- No resource limits without OS-level configuration
- May interfere with host system

**Configuration**: No special config needed; it's the default.

**Usage:**
```bash
uv run cli agents langchain --sandbox local "Run code"
# Or default:
uv run cli agents langchain "Run code"
```

### Docker Sandbox

**Type**: Isolated container using `agent-infra/sandbox` image

**Pros:**
- Strong process and filesystem isolation
- Resource limits (CPU, memory, PID count)
- Reproducible environment
- Clean container per execution

**Cons:**
- Requires Docker daemon running
- ~5-10s startup overhead per container
- Network access requires explicit configuration

**Image**: `ghcr.io/agent-infra/sandbox:latest`

**Configuration** (`config/basic/sandbox.yaml`):
```yaml
sandbox:
  docker:
    aio:
      image: "ghcr.io/agent-infra/sandbox:latest"
      host: "127.0.0.1"
      host_port: 18091
      startup_timeout: 60.0
      work_dir: "/home/user"
      env_vars: {}
    smolagents:
      image: "python:3.12-slim"
      mem_limit: "512m"
      cpu_quota: 50000
      pids_limit: 100
```

**Usage:**
```bash
# LangChain
uv run cli agents langchain --sandbox docker "bash /tmp/script.sh"

# DeepAgent
uv run cli agents deepagent --sandbox docker "Analyze results"

# Deer Flow
uv run cli agents deerflow --input "run tests" --sandbox docker

# SmolAgents
uv run cli agents smolagents --executor docker "write and test code"
```

**Docker Setup Check:**
```bash
docker ps  # Ensure Docker daemon is running
docker pull ghcr.io/agent-infra/sandbox:latest
```

### E2B Cloud Sandbox

**Type**: Managed cloud sandboxes (E2B)

**Pros:**
- No local Docker dependency
- Fully managed infrastructure
- Scales to multiple concurrent sandboxes

**Cons:**
- Requires E2B API key
- Network latency
- No local file caching

**Configuration** (`config/basic/sandbox.yaml`):
```yaml
sandbox:
  e2b:
    api_key: "${E2B_API_KEY}"  # Environment variable reference
    template: null              # E2B template ID (optional)
    timeout: 300
```

**Setup:**
1. Sign up at [e2b.dev](https://e2b.dev)
2. Get API key from dashboard
3. Set environment variable: `export E2B_API_KEY="..."`
4. Or add to `.env` file for auto-loading

**Usage:**
```bash
# SmolAgents only (native support)
uv run cli agents smolagents --executor e2b "analyze data"
```

## CLI Usage

### Optimising Docker Sandbox Startup (Recommended)

Docker sandboxes are slow on first use (~28 seconds) because the server and image need to be initialised. To avoid this cost on every agent invocation, use the **`cli sandbox`** commands to warm up the infrastructure once:

```bash
# 1. Start the opensandbox-server as a background daemon (keeps running between invocations)
cli sandbox start

# 2. Pre-pull the Docker image so there's no download stall
cli sandbox pull

# 3. Verify everything is ready
cli sandbox status
```

Once these run, every subsequent `--sandbox docker` agent call reuses the warm server and cached image, reducing startup from ~28 s to ~5–10 s.

**`cli sandbox` subcommands:**

| Command | Purpose |
|---------|---------|
| `cli sandbox start` | Launch opensandbox-server as a background daemon. Writes PID to `~/.cache/genai-tk/opensandbox-server.pid`. Returns immediately after the server becomes HTTP-ready. |
| `cli sandbox stop` | Terminate the daemon and clean up the PID file. |
| `cli sandbox status` | Show daemon status, image cache status, and HTTP reachability. Tips are printed only when action is needed. |
| `cli sandbox pull` | Run `docker pull` on the configured image (default: `ghcr.io/agent-infra/sandbox:latest`). |

**Example workflow:**

```bash
# Once per machine boot
$ cli sandbox start
opensandbox-server started (pid 12345)
Listening at: http://localhost:8080
Server is ready.
Tip: run cli sandbox pull to pre-pull the Docker image.

$ cli sandbox pull
Pulling Docker image: ghcr.io/agent-infra/sandbox:latest
Image ready: ghcr.io/agent-infra/sandbox:latest

$ cli sandbox status
           OpenSandbox Status
 Server URL              http://localhost:8080
 Docker image            ghcr.io/agent-infra/sandbox:latest  (cached locally)
 Daemon PID              12345 (alive)
 HTTP reachable          yes

# Every agent invocation now reuses the warm server + cached image
$ uv run cli agents langchain --sandbox docker "Your query"
```

### Quick Start

**LangChain:**
```bash
# Default (local) sandbox
uv run cli agents langchain "read file.txt"

# With Docker sandbox
uv run cli agents langchain --sandbox docker "bash ls -la /home/user"
```

**DeepAgent:**
```bash
# With Docker sandbox
uv run cli agents deepagent --sandbox docker "fix this code"
```

**Deer Flow:**
```bash
# With input option and Docker sandbox
uv run cli agents deerflow --input "calculate 2+2" --sandbox docker

# With profile override
uv run cli agents deerflow --input "summarize results" --sandbox docker -p "Research Assistant"
```

**SmolAgents:**
```bash
# Docker executor
uv run cli agents smolagents --executor docker "write a python function"

# E2B cloud sandbox
uv run cli agents smolagents --executor e2b "run data analysis"
```

### Full CLI Option Reference

#### LangChain (`langchain`)
```
--sandbox, -b [local|docker]  Execution environment (default: local)
```

#### DeepAgent (`deepagent`)
```
--sandbox [local|docker]       Execution environment (default: local)
```

#### Deer Flow (`deerflow`)
```
--sandbox [local|docker]       Execution environment (default: local)
--input, -i TEXT              Input text for the agent
```

#### SmolAgents (`smolagents`)
```
--executor [local|docker|e2b]  Code executor (default: local)
--input, -i TEXT              Input text for the agent
```

## Configuration

### YAML Structure

**Location**: `config/basic/sandbox.yaml`

**Format**: Top-level `sandbox` key containing environment-specific settings:

```yaml
sandbox:
  # Default sandbox type when --sandbox/--executor not specified
  default: "local"

  # Docker configuration (used by all frameworks except E2B-specific)
  docker:
    # AioSandboxBackend settings (LangChain, DeepAgent, Deer Flow)
    aio:
      image: "ghcr.io/agent-infra/sandbox:latest"
      host: "127.0.0.1"
      host_port: 18091
      startup_timeout: 60.0
      work_dir: "/home/user"
      env_vars:
        # Optional environment variables for the sandbox
        PYTHONPATH: "/workspace"

    # SmolAgents Docker executor settings
    smolagents:
      image: "python:3.12-slim"
      mem_limit: "512m"       # Memory limit
      cpu_quota: 50000        # CPU quota (microseconds per period)
      pids_limit: 100         # Max processes

  # E2B cloud sandbox (SmolAgents only)
  e2b:
    api_key: null             # Will be read from E2B_API_KEY env var
    template: null            # E2B template ID (optional)
    timeout: 300              # Timeout in seconds

  # WebAssembly / Pyodide (reserved for future use)
  wasm:
    enabled: false
```

### Environment Variable Substitution

OmegaConf supports environment variable interpolation:

```yaml
sandbox:
  e2b:
    api_key: "${E2B_API_KEY}"  # Reads E2B_API_KEY from environment
```

### Environment-Specific Configuration

Load different configurations for dev/prod:

**`config/basic/sandbox.yaml`** (base):
```yaml
sandbox:
  default: "local"
  docker:
    aio:
      image: "ghcr.io/agent-infra/sandbox:latest"
      # ...
```

**`config/basic/sandbox_prod.yaml`** (production override):
```yaml
sandbox:
  default: "docker"
  docker:
    aio:
      startup_timeout: 120.0
  e2b:
    api_key: "${E2B_API_KEY}"
```

Load with OmegaConf:
```python
from omegaconf import OmegaConf
from genai_tk.utils.config_mngr import load_config

cfg = OmegaConf.merge(
    load_config("basic/sandbox.yaml"),
    load_config("basic/sandbox_prod.yaml"),
)
```

## Python Integration

### Load Sandbox Configuration

```python
from genai_tk.agents.sandbox.config import (
    load_sandbox_config,
    get_docker_aio_settings,
    get_docker_smolagents_settings,
    get_e2b_settings,
)

# Load full config
cfg = load_sandbox_config()
print(f"Default sandbox: {cfg.default}")

# Load specific settings
aio_settings = get_docker_aio_settings()
print(f"Docker image: {aio_settings.image}")
print(f"Port: {aio_settings.host_port}")

smol_settings = get_docker_smolagents_settings()
print(f"SmolAgents memory limit: {smol_settings.mem_limit}")

e2b_settings = get_e2b_settings()
if e2b_settings.api_key:
    print("E2B API key configured")
```

### Use AioSandboxBackend Directly

```python
import asyncio
from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend
from genai_tk.agents.sandbox.config import get_docker_aio_settings

async def run_code():
    config = get_docker_aio_settings()
    async with AioSandboxBackend(config=config) as backend:
        # Execute a bash command
        result = await backend.aexecute("echo 'Hello from sandbox'")
        print(result.output)

        # List files
        files = await backend.als_info("/home/user")
        for f in files:
            print(f"  {f['path']}")

        # Write a file
        await backend.awrite("/home/user/test.txt", "Hello World")

asyncio.run(run_code())
```

### Create a Sandboxed Agent

**LangChain with Docker:**
```python
from genai_tk.core import LLMFactory
from genai_tk.agents.langchain.langchain_agent import LangchainAgent

llm = LLMFactory.create("openai/gpt-4")
agent = LangchainAgent(
    llm=llm,
    sandbox="docker",  # Enable Docker sandbox
    tools=["bash", "file"],
)

result = agent.run("Write 'test' to /tmp/output.txt and verify it")
print(result)
```

**DeepAgent with Docker:**
```python
from genai_tk.agents.deepagent_cli.sandbox_bridge import DeepAgentSandboxBridge

bridge = DeepAgentSandboxBridge(sandbox_type="docker")
response = await bridge.run_agent(
    input_text="analyze this code",
    mode="pro",
    sandbox_override="docker",
)
```

**SmolAgents with Docker:**
```python
from smolagents import CodeAgent
from genai_tk.agents.smolagents.commands import create_smol_agent

agent = create_smol_agent(
    model="openai",
    executor_type="docker",  # Docker-based code execution
)

result = agent.run("calculate fibonacci(10)")
```

## Docker Container Lifecycle

### Setup

**Prerequisites:**
1. Docker daemon running (`docker ps` works)
2. Network access to Docker socket
3. `ghcr.io/agent-infra/sandbox:latest` image available

**Pre-pull the image** (optional, auto-fetched on first use):
```bash
docker pull ghcr.io/agent-infra/sandbox:latest
```

### Container Lifecycle Events

**On agent startup with `--sandbox docker`:**

1. **Stale container cleanup**: Checks for leftover containers on port 18091 and removes them
2. **Container creation**: Starts a new container with settings from `config/basic/sandbox.yaml`
3. **Health check**: Polls the HTTP API (at `http://127.0.0.1:18091`) until `startup_timeout`
4. **Ready**: Sandbox is ready to accept tool execution requests

**During execution:**
- Commands are sent to `http://127.0.0.1:18091/execute/` endpoint
- Responses include output, stderr, exit code

**On agent shutdown:**
- Container is stopped
- Container is removed from docker

### Debugging Docker Issues

**Check running containers:**
```bash
docker ps --filter publish=18091
```

**View container logs:**
```bash
docker logs <container_id>
```

**Manual container cleanup:**
```bash
# Stop and remove the sandbox container
docker stop $(docker ps -q --filter publish=18091) 2>/dev/null || true
docker rm $(docker ps -aq --filter publish=18091) 2>/dev/null || true
```

**Test connectivity:**
```bash
curl http://127.0.0.1:18091/health
```

## Error Handling

### Common Issues

**"No Docker running"**
```
Error: Could not connect to Docker daemon
```
**Fix**: Start Docker daemon
```bash
docker run --version  # Tests connection
```

**"Docker image not found"**
```
Error: image not found: ghcr.io/agent-infra/sandbox:latest
```
**Fix**: Pull the image
```bash
docker pull ghcr.io/agent-infra/sandbox:latest
```

**"Port 18091 already in use"**
```
Error: address already in use
```
**Fix**: Stop stale container or change port in `sandbox.yaml`
```bash
docker stop $(docker ps -q --filter publish=18091)
```

**"Startup timeout"**
```
Error: Sandbox failed to start within timeout
```
**Fix**: Increase `startup_timeout` in `config/basic/sandbox.yaml` or check Docker logs

### Enabling Debug Logging

```python
import logging

# Enable debug logs for sandbox operations
logging.getLogger("genai_tk.agents.sandbox").setLevel(logging.DEBUG)
```

Or via environment:
```bash
LOGURU_LEVEL=DEBUG uv run cli agents langchain --sandbox docker "test"
```

## Testing

### Unit Tests

```bash
# Test sandbox models and config
uv run pytest tests/unit_tests/agents/sandbox/ -v

# Test LangChain integration
uv run pytest tests/unit_tests/agents/langchain/test_langchain_agent.py -v
```

### Integration Tests

```bash
# Test Docker sandbox with real container
uv run pytest tests/integration_tests/agents/test_langchain_sandbox_integration.py -v
```

## Best Practices

1. **Use Docker for untrusted code** — Local sandbox is only for development
2. **Set resource limits** — Configure `mem_limit` and `cpu_quota` for computational tasks
3. **Validate outputs** — Sandbox may return partial results on timeout
4. **Monitor startup time** — Increase `startup_timeout` for slow systems or networks
5. **Check available disk space** — Docker containers write logs and temporary files
6. **Use environment variables** — Store API keys via environment, not in YAML
7. **Profile code execution** — Log execution time to catch slow computations early

## Troubleshooting

### Agent runs but sandbox features don't work

**Issue**: `--sandbox docker` option seems to be ignored.

**Diagnosis**:
1. Check if option appears in help: `uv run cli agents langchain --help | grep sandbox`
2. Verify config file exists: `cat config/basic/sandbox.yaml`
3. Test Docker directly: `docker ps`

**Solution**: Restart the agent with fresh environment:
```bash
source .venv/bin/activate  # For venv
uv sync                    # Reinstall
uv run cli agents langchain --sandbox docker "test"
```

### Docker container keeps getting stuck

**Issue**: Container starts but HTTP API doesn't respond.

**Diagnosis**:
```bash
docker logs <container_id>  # Check container logs
curl http://127.0.0.1:18091/health  # Test API
```

**Solution**: Increase startup timeout and retry:
```yaml
sandbox:
  docker:
    aio:
      startup_timeout: 120.0  # Increase from 60
```

### E2B executor won't start

**Issue**: "E2B API key not configured" error

**Diagnosis**:
```bash
echo $E2B_API_KEY  # Check env var is set
```

**Solution**: Set API key:
```bash
export E2B_API_KEY="your-key-here"
# Or in .env:
echo "E2B_API_KEY=your-key-here" >> .env
```

## See Also

- [Agent Integration Guide](docs/deepagent_integration.md) — DeepAgent-specific setup
- [Deer Flow Integration](docs/Deer_Flow_Integration.md) — Deer Flow configuration
- [MCP Servers](docs/mcp-servers.md) — Model Context Protocol integration
- [Configuration Management](genai_tk/utils/config_mngr.py) — OmegaConf setup

## Future Enhancements

- **WASM sandbox** — Browser-based execution via Pyodide
- **Resource profiling** — Automatic memory/CPU limit detection
- **Snapshot isolation** — Container template caching for faster startup
- **Network sandboxes** — Egress/ingress filtering per execution
- **Persistent volumes** — Shared storage across multiple executions
