# AioSandboxBackend

`AioSandboxBackend` is a `deepagents` `SandboxBackendProtocol` implementation backed by the
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
    host_port=19091,          # host port to bind (default: 18091)
    startup_timeout=120.0,    # seconds to wait for the API to become ready
    work_dir="/workspace",    # default path for ls when no path is given
    env_vars={"MY_VAR": "1"}, # extra env vars passed to the container
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
| `work_dir`        | `str`             | `/home/user`                         | Default path for `ls` / `agrep_raw` |
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
