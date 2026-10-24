# Docker Image Build

This page covers how to build and run Docker images for genai-tk based applications.

## Quick Start

```bash
# Build image for the current project
just docker-build

# Run the container (detached, port 8501)
just docker-run

# Check it started
just docker-check
just docker-logs
```

## How It Works

The build system has three pieces:

| File | Purpose |
|---|---|
| `deploy/Dockerfile` | Generic multi-stage image — parameterised with build args |
| `deploy/docker.just` | Shared recipes (`docker-build`, `docker-run`, …) — imported by the project justfile |
| `justfile` (project root) | Sets project-specific variables, then `import 'deploy/docker.just'` |

### Variables (set in the importing justfile)

| Variable | Default | Description |
|---|---|---|
| `app_name` | — | Docker image name (e.g. `rfq-pricing`) |
| `docker_tag` | `latest` | Image tag |
| `docker_pkg` | — | Python package directory (e.g. `rfq_pricing`) |
| `extras` | `streamlit` | uv optional extras to install, comma-separated |
| `install_node` | `false` | Install Node.js (`true` / `false`) |
| `extra_copy` | `""` | Extra directories to bake into image (space-separated, e.g. `skills data`) |
| `streamlit_entry` | `""` | Streamlit entry script path. Empty = auto-detected from installed package. |
| `docker_port` | `8501` | Host port to expose |
| `dockerfile` | `deploy/Dockerfile` | Path to Dockerfile |

### Build Args (Dockerfile)

The same variables map to Docker build args. You can pass them directly:

```bash
docker build -f deploy/Dockerfile \
  --build-arg PKG_NAME=rfq_pricing \
  --build-arg EXTRAS=streamlit,nlp \
  --build-arg INSTALL_NODE=false \
  -t rfq-pricing:latest .
```

## Available Recipes

| Recipe | Description |
|---|---|
| `just docker-build` | Build the image |
| `just docker-run` | Start container detached on port 8501 |
| `just docker-run-it` | Start container interactively (removed on exit) |
| `just docker-stop` | Stop and remove the container |
| `just docker-logs` | Follow container logs |
| `just docker-shell` | Open bash in the running container |
| `just docker-check` | List images matching `app_name` |
| `just docker-rmi` | Remove the image |
| `just docker-sync-time` | Resync WSL clock after hibernation |

## Runtime Configuration

### Build and dependency security

The maintained [Dockerfile](../deploy/Dockerfile) pins Python 3.12.15, uv 0.13.5,
and optional Node 24.22.1 by version and multi-platform image digest. Both install
stages use `uv sync --locked --no-default-groups`; development and evaluation
dependencies are excluded. Each comma-separated extra becomes its own argument.
Node comes from its pinned image, without executing a downloaded installer.

Run `uv python pin 3.12.15` if a downstream project has no `.python-version`, and
commit it with `uv.lock`. Copy the maintained Dockerfile and
[.dockerignore](../.dockerignore) together when creating a downstream image.
The ignore rules prevent environment files, credential directories, and private
keys from reaching any build stage while preserving runtime skill Markdown.
Inject secrets at runtime.

Python, Requests, and curl use the patched certifi bundle rather than an
unreviewed OS/custom trust bundle. Rebuild existing images after updates. See
[the security guide](security.md) for audit commands and unresolved upstream
advisories; a successfully built image is not evidence of a clean security audit.

### Environment Variables

The container reads secrets from environment variables. Mount your `.env` at runtime (the `docker-run` recipe does this automatically if the file exists):

```bash
# Automatic if .env exists in project root
just docker-run

# Or pass variables explicitly
docker run --env OPENAI_API_KEY=sk-... rfq-pricing:latest
```

`~/.env` is also auto-mounted if it exists (genai-tk convention for shared API keys).

### Config Override

To override the baked-in `config/` at runtime, set `DOCKER_CONFIG_DIR`:

```bash
DOCKER_CONFIG_DIR=/path/to/my/config just docker-run
# mounts as /app/config (read-only)
```

### Data Volume

Set `DOCKER_DATA_DIR` to mount a directory as `/data/external`:

```bash
DOCKER_DATA_DIR=/my/data just docker-run
```

## Selecting Extras

Extras control which optional dependencies are installed. They match the extras defined in `pyproject.toml`:

| Extra | Contents |
|---|---|
| `streamlit` | Streamlit web interface |
| `nlp` | spaCy, Presidio PII detection |
| `postgres` | PostgreSQL vector store |
| `baml` | BAML structured extraction |
| `chromadb` | ChromaDB vector store |
| `harnessing` | DeerFlow, DeepAgents, sandbox |
| `browser` | Playwright |
| `monitoring` | LangFuse, OpenTelemetry |
| `all` | All of the above |

```bash
# Build with NLP and monitoring
just docker-build EXTRAS=streamlit,nlp,monitoring
```

Or override via env var:
```bash
DOCKER_EXTRAS=streamlit,nlp just docker-build
```

## Adding Docker Support to a Scaffolded App

If you created your project with `cli init`, here's how to add Docker support:

**1. Create `deploy/Dockerfile`** — copy from genai-tk and adjust defaults:

```dockerfile
ARG PKG_NAME=my_app
ARG EXTRAS=streamlit
ARG INSTALL_NODE=false
ARG STREAMLIT_ENTRY=""
ARG EXTRA_COPY_DIRS="skills"

# Copy the pinned stages from genai-tk/deploy/Dockerfile unchanged.
```

**2. Create `deploy/docker.just`** — copy from genai-tk (recipes only, no variable definitions):

```just
# Full image reference
_image := app_name + ":" + docker_tag

[doc('Build the Docker image')]
docker-build *args:
    docker build --pull --rm -f "{{ dockerfile }}" \
        --build-arg PKG_NAME="{{ docker_pkg }}" \
        --build-arg EXTRAS="{{ extras }}" \
        …
```

**3. Add variables and import in your `justfile`**:

```just
# ── Docker configuration ──────────────────────────────────────────────────────
app_name        := "my-app"
docker_tag      := "latest"
extras          := "streamlit"
install_node    := "false"
extra_copy      := "skills"
streamlit_entry := ""
docker_port     := "8501"
dockerfile      := "deploy/Dockerfile"
docker_pkg      := "my_app"

import 'deploy/docker.just'
```

**4. Add `.dockerignore`** (copy from genai-tk, adjust if needed).

**5. Ensure `uv.lock` and `.python-version` are committed** — the Dockerfile uses `uv sync --locked --no-default-groups`.

## Streamlit Entry Point Resolution

| Situation | How entry is resolved |
|---|---|
| `streamlit_entry` set in justfile | Used as-is (`/app/<entry>`) |
| `streamlit_entry` empty (default) | Auto-detected: `python -c "import genai_tk; …"` finds the installed webapp |
| Building genai-tk itself | `streamlit_entry = "genai_tk/webapp/main/streamlit.py"` (explicit in justfile) |

This means scaffolded apps that use genai-tk's built-in webapp (with their own pages registered via genai-tk's page discovery) work without any extra configuration.

## Docker Prerequisites

The user running `just docker-*` must have access to the Docker daemon:

```bash
# Add yourself to the docker group (requires logout/login or WSL restart)
sudo usermod -aG docker $USER
```

Or prefix commands with `sudo`:
```bash
sudo docker build …
```

## Project-Specific Notes

### genai-tk itself

Built with `extras=all,monitoring` and `install_node=true` (some Streamlit components need Node.js). Entry: `genai_tk/webapp/main/streamlit.py`.

### rfq_pricing

Built with `extras=streamlit` and `install_node=false`. Streamlit entry is auto-detected from the installed `genai_tk` package. Extra dirs `skills data` are baked in.
