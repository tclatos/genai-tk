# AioSandboxBackend

`AioSandboxBackend` is a `deepagents` `BackendProtocol` implementation backed by the
[agent-infra/sandbox](https://github.com/agent-infra/sandbox) Docker container.

It manages the full container lifecycle — pulling the image, starting the container, polling until the REST API is healthy, and stopping the container on exit.

## Architecture

```
AioSandboxBackend (Python, this process)
        │
        │  HTTP  (via agent_sandbox.AsyncSandbox)
        ▼
ghcr.io/agent-infra/sandbox container
        │
        ├─ port 8091 (SANDBOX_SRV_PORT) ← REST API
        │     /v1/shell/exec_command
        │     /v1/file/read_file
        │     /v1/file/write_file
        │     /v1/file/replace_in_file
        │     /v1/file/list_path
        │
        └─ port 8080 ← nginx web UI (not used by this backend)
```

The `agent_sandbox.AsyncSandbox` SDK is a pure HTTP client (Fern-generated). It does **not** manage Docker — the backend does that via `asyncio.create_subprocess_exec("docker", ...)`.

## Usage

```python
from genai_tk.agents.langchain.sandbox_backend import AioSandboxBackend, AioSandboxBackendConfig

async with AioSandboxBackend() as backend:
    result = await backend.execute_tool("bash", {"command": "echo hello"})
    print(result.output)   # "hello\n"
    print(result.success)  # True
```

With custom config:

```python
config = AioSandboxBackendConfig(
    host_port=19091,          # host port to bind (default: 18091)
    startup_timeout=120.0,    # seconds to wait for the API to become ready
    work_dir="/workspace",    # default path for ls when no path is given
    env_vars={"MY_VAR": "1"}, # extra env vars passed to the container
)

async with AioSandboxBackend(config=config) as backend:
    ...
```

## Supported Tools

| Tool name    | Input keys                         | Description                              |
|--------------|------------------------------------|------------------------------------------|
| `bash`       | `command: str`                     | Run a shell command; returns stdout      |
| `ls`         | `path: str` (optional)             | List directory; defaults to `work_dir`   |
| `read_file`  | `path: str`                        | Read a file; returns its content         |
| `write_file` | `path: str`, `content: str`        | Write / overwrite a file                 |
| `str_replace`| `path: str`, `old_str`, `new_str`  | In-place string replacement in a file    |

All tools return a `SandboxToolResult` with:

- `output: str` — command output or file content
- `exit_code: int` — 0 on success
- `error: str | None` — error message if the operation failed
- `success: bool` — `True` when `exit_code == 0` and `error is None`

## Docker Image

```
ghcr.io/agent-infra/sandbox:latest
```

Pull manually if needed:

```bash
docker pull ghcr.io/agent-infra/sandbox:latest
```

The container exposes two ports:

| Port | Purpose |
|------|---------|
| 8080 | nginx web UI — not used |
| 8091 | REST API (`SANDBOX_SRV_PORT`) — used by this backend |

## Proxy Caveat

If an `http_proxy` (or `HTTP_PROXY`) environment variable is set, `httpx` will route **all** requests through it — including those to `localhost`.  
Both the health-poll client and the `AsyncSandbox` client are created with `trust_env=False` to bypass this.

## Exit Code Caveat

When running bash commands that intentionally exit with a non-zero code, use a subshell:

```python
# WRONG — kills the shared shell session in the container
await backend.execute_tool("bash", {"command": "exit 42"})

# CORRECT — exit code is captured; session survives
result = await backend.execute_tool("bash", {"command": "bash -c 'exit 42'"})
assert result.exit_code == 42
```

## Configuration Reference

| Field             | Type              | Default                              | Description                       |
|-------------------|-------------------|--------------------------------------|-----------------------------------|
| `image`           | `str`             | `ghcr.io/agent-infra/sandbox:latest` | Docker image to run               |
| `host`            | `str`             | `127.0.0.1`                          | Interface to bind on the host     |
| `host_port`       | `int`             | `18091`                              | Host port mapped to container 8091|
| `startup_timeout` | `float`           | `60.0`                               | Seconds to wait for the API       |
| `work_dir`        | `str`             | `/home/user`                         | Default path for `ls`             |
| `env_vars`        | `dict[str, str]`  | `{}`                                 | Extra env vars for the container  |

## Testing

Unit tests (no Docker required):

```bash
uv run pytest tests/unit_tests/agents/langchain/test_sandbox_backend.py -v
```

Integration tests (Docker required, image must be pulled):

```bash
uv run pytest tests/integration_tests/agents/test_sandbox_backend_integration.py -v -s
```
