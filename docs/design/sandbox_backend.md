# AioSandboxBackend

`AioSandboxBackend` is a `deepagents` `SandboxBackendProtocol` implementation that provides secure, isolated code execution via [Alibaba OpenSandbox](https://github.com/alibaba/OpenSandbox).

## Architecture

```
AioSandboxBackend (Python, this process)
        │
        ├─ Manages OpenSandbox lifecycle via opensandbox SDK
        │
        ├─ Spawns opensandbox-server (if not already running)
        │     └─ Listens on http://localhost:8080 (configurable)
        │
        └─ HTTP client (via agent_sandbox.AsyncSandbox)
             │  HTTP
             ▼
        OpenSandbox Server
             │
             ├─ REST API on configured port
             │     /v1/shell/exec_command
             │     /v1/file/read_file
             │     /v1/file/write_file
             │     /v1/file/replace_in_file
             │     /v1/file/list_path
             │
             └─ Manages Docker container lifecycle
                  │
                  └─ ghcr.io/agent-infra/sandbox:latest
                       ├─ port 8091 ← REST API
                       └─ port 8080 ← nginx web UI
```

The `agent_sandbox.AsyncSandbox` SDK is a pure HTTP client (Fern-generated). Container lifecycle and port management are handled by the OpenSandbox server.

## Usage

### Low-level `execute_tool`

```python
from genai_tk.agents.langchain.sandbox_backend import AioSandboxBackend, AioSandboxBackendConfig

async with AioSandboxBackend() as backend:
    result = await backend.execute_tool("bash", {"command": "echo hello"})
    print(result.output)   # "hello\n"
    print(result.success)  # True
```

### `SandboxBackendProtocol` interface

```python
async with AioSandboxBackend() as backend:
    # Execute a shell command
    resp = await backend.aexecute("ls /home/user")
    print(resp.output)      # file listing
    print(resp.exit_code)   # 0

    # List directory with metadata
    infos = await backend.als_info("/home/user")
    # [{'path': '/home/user/file.py', 'size': 1234}, ...]

    # Read a file with line numbers (supports pagination)
    text = await backend.aread("/home/user/file.py", offset=0, limit=50)
    # "1: #!/usr/bin/env python3\n2: ..."

    # Write a new file (errors if file already exists)
    result = await backend.awrite("/home/user/new.py", "print('hi')")

    # Edit a file — replace first or all occurrences
    result = await backend.aedit("/home/user/new.py", "print('hi')", "print('hello')")
    # EditResult(path=..., occurrences=1)

    # Search with grep
    matches = await backend.agrep_raw("TODO", path="/home/user", glob="*.py")
    # [GrepMatch(path=..., line=5, text='    # TODO: fix this'), ...]

    # Glob file listing
    infos = await backend.aglob_info("**/*.py", path="/home/user")

    # Bulk file upload / download
    await backend.aupload_files([("/home/user/a.txt", b"content")])
    responses = await backend.adownload_files(["/home/user/a.txt"])
    # [FileDownloadResponse(path=..., content=b"content")]
```

With custom config:

```python
config = AioSandboxBackendConfig(
    opensandbox_server_url="http://localhost:8080",  # OpenSandbox server URL
    startup_timeout=120.0,  # seconds to wait for the API to become ready
    work_dir="/workspace",  # default path for ls when no path is given
    env_vars={"MY_VAR": "1"},  # extra env vars passed to the container
    entrypoint=["/opt/gem/run.sh"],  # sandbox container entrypoint
)

async with AioSandboxBackend(config=config) as backend:
    ...
```

## Supported Tools (`execute_tool`)

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

## `SandboxBackendProtocol` Methods

`AioSandboxBackend` fully implements `deepagents.backends.protocol.SandboxBackendProtocol`.
All methods are async-native; sync counterparts (`ls_info`, `read`, etc.) raise `NotImplementedError`.

| Method | Returns | Description |
|--------|---------|-------------|
| `aexecute(command, *, timeout)` | `ExecuteResponse` | Run a shell command |
| `als_info(path)` | `list[FileInfo]` | List directory with `path` / `size` metadata |
| `aread(file_path, offset, limit)` | `str` | Read file with 1-based line numbers; paginatable |
| `awrite(file_path, content)` | `WriteResult` | Create a new file; error if it already exists |
| `aedit(file_path, old, new, replace_all)` | `EditResult` | Replace text; `replace_all=False` replaces first occurrence only |
| `agrep_raw(pattern, path, glob)` | `list[GrepMatch] \| str` | Grep for literal text; returns `GrepMatch` list or error string |
| `aglob_info(pattern, path)` | `list[FileInfo]` | Find files matching a glob (`**` supported via Python glob) |
| `aupload_files(files)` | `list[FileUploadResponse]` | Write multiple `(path, bytes)` files |
| `adownload_files(paths)` | `list[FileDownloadResponse]` | Read multiple files as `bytes` |
| `id` (property) | `str` | Container short ID when running; random hex otherwise |

## OpenSandbox Server Setup

The backend automatically starts `opensandbox-server` if it's not already running. No manual setup required, but you can configure it:

### Installation

```bash
uv add opensandbox-server
```

### Initialize Configuration

```bash
opensandbox-server init-config ~/.sandbox.toml --example docker
```

This creates `~/.sandbox.toml` with Docker-based sandbox configuration. The `opensandbox-server` manages Docker image pulling and container lifecycle internally.

### Manual Server Start

```bash
opensandbox-server --config ~/.sandbox.toml
```

The server listens on `http://localhost:8080` by default.

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

All settings live in the ``sandbox.docker.aio`` block of ``config/sandbox.yaml``
and are shared by **both** harnesses — the LangChain/deepagents harness
(``AioSandboxBackend``) and the DeerFlow harness
(``deerflow.community.aio_sandbox:AioSandboxProvider``). The DeerFlow config
bridge (:func:`genai_tk.agents.deer_flow.config_bridge.write_deer_flow_config`)
forwards ``env_vars`` → ``environment`` and ``volumes`` → ``mounts`` into the
generated DeerFlow ``config.yaml`` so a single ``sandbox.yaml`` drives both
runtimes. DeerFlow's provider manages containers directly and does not consume
``opensandbox_server_url`` (kept in the generated config for diagnostics only).

| Field                     | Type              | Default                                      | Description                              |
|---------------------------|-------------------|----------------------------------------------|------------------------------------------|
| `opensandbox_server_url`  | `str`             | `http://localhost:8080`                      | OpenSandbox server URL (LangChain harness) |
| `image`                   | `str`             | `ghcr.io/agent-infra/sandbox:latest`         | Container image (both harnesses)         |
| `entrypoint`              | `list[str]`       | `["/opt/gem/run.sh"]`                        | Sandbox container entrypoint (LangChain) |
| `startup_timeout`         | `float`           | `60.0`                                       | Seconds to wait for the API              |
| `work_dir`                | `str`             | `/home/user`                                 | Default path for `ls` / `agrep_raw`      |
| `env_vars`                | `dict[str, str]`  | `{}`                                         | Container env vars (both harnesses)      |
| `volumes`                 | `list[VolumeMountConfig]` | `[]`                              | Bind-mounts (both harnesses)             |

```yaml
sandbox:
  docker:
    aio:
      image: "ghcr.io/agent-infra/sandbox:latest"
      opensandbox_server_url: "http://localhost:8080"
      startup_timeout: 60.0
      work_dir: "/home/user"
      env_vars:
        API_KEY: ${oc.env:MY_API_KEY}
      volumes:
        - host_path: /home/me/project/skills
          container_path: /mnt/skills
          read_only: true
```

## Testing

Unit tests (no Docker required):

```bash
uv run pytest tests/unit_tests/agents/langchain/test_sandbox_backend.py -v
```

Integration tests (Docker required, image must be pulled):

```bash
uv run pytest tests/integration_tests/agents/test_sandbox_backend_integration.py -v -s
```
