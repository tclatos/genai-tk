---
name: genai-tk-docker
description: Build, configure, and troubleshoot Docker images for genai-tk and scaffolded applications. Add Docker support to a new project, adjust extras, fix Dockerfile issues, update just recipes.
---

# Docker Image Build for genai-tk Apps

## Read First

- `docs/docker.md`
- `deploy/Dockerfile`
- `deploy/docker.just`

## Code Map

| Concern | Path |
|---|---|
| Generic Dockerfile template | `deploy/Dockerfile` |
| Shared just recipes | `deploy/docker.just` |
| genai-tk docker variables | `justfile` (top, before `import 'deploy/docker.just'`) |
| rfq_pricing docker variables | `rfq_pricing/justfile` (top) |
| rfq_pricing Dockerfile | `rfq_pricing/deploy/Dockerfile` |
| rfq_pricing docker recipes | `rfq_pricing/deploy/docker.just` |

## Architecture

The system is split into three layers:

```
justfile (project)          — defines variables (app_name, extras, …)
  └── import deploy/docker.just  — provides docker-build, docker-run, … recipes
        └── deploy/Dockerfile    — generic multi-stage uv build (uses build args)
```

**Key invariant:** `deploy/docker.just` must NOT define variables. All variables (`app_name`, `docker_tag`, `extras`, `install_node`, `extra_copy`, `streamlit_entry`, `docker_port`, `dockerfile`, `docker_pkg`) must be defined in the importing `justfile` BEFORE the `import` line. This is required because `just` forbids variable redefinition across imported files.

## Variables Reference

```just
app_name        := "my-app"          # → image name
docker_tag      := "latest"          # → image tag → _image = app_name + ":" + docker_tag
docker_pkg      := "my_pkg"          # → PKG_NAME build arg → copied into /app/<pkg>
extras          := "streamlit"       # → EXTRAS build arg → uv sync --extra <extras>
install_node    := "false"           # → INSTALL_NODE build arg (true/false)
extra_copy      := "skills data"     # → EXTRA_COPY_DIRS build arg (space-separated dirs)
streamlit_entry := ""                # → STREAMLIT_ENTRY build arg (empty = auto-detect)
docker_port     := "8501"            # → host port in docker-run
dockerfile      := "deploy/Dockerfile"
```

## Dockerfile Build Args

All build args correspond to the justfile variables above:

```dockerfile
ARG PKG_NAME=genai_tk
ARG EXTRAS=streamlit
ARG INSTALL_NODE=false
ARG STREAMLIT_ENTRY=""
ARG EXTRA_COPY_DIRS=""
```

## Streamlit Entry Resolution

- If `STREAMLIT_ENTRY` is non-empty: Dockerfile uses `/app/${STREAMLIT_ENTRY}`
- If empty: resolved at runtime via `python -c "import genai_tk; …"` from the installed package
- genai-tk itself: sets `streamlit_entry := "genai_tk/webapp/main/streamlit.py"` explicitly (doubles as the `webapp` recipe arg)

## Adding Docker Support to a New Project

1. Create `deploy/Dockerfile` — copy from `genai-tk/deploy/Dockerfile`, keep the structure unchanged. Only adjust the default ARG values at the top if needed.

2. Create `deploy/docker.just` — copy from `genai-tk/deploy/docker.just` **exactly**. Do NOT add variable definitions here.

3. In the project `justfile`, add BEFORE the first recipe:
   ```just
   app_name        := "my-app"
   docker_tag      := "latest"
   extras          := "streamlit"
   install_node    := "false"
   extra_copy      := "skills"
   streamlit_entry := ""
   docker_port     := "8501"
   dockerfile      := "deploy/Dockerfile"
   docker_pkg      := "my_pkg"

   import 'deploy/docker.just'
   ```
   Also add `set shell := ["bash", "-euc"]` if not already present (required for the `#!/usr/bin/env bash` recipe blocks).

4. Create `.dockerignore` — copy from `genai-tk/.dockerignore`, remove project-irrelevant lines.

5. Commit `uv.lock` — the Dockerfile uses `uv sync --locked`.

## Common Issues

### "variable has multiple definitions"
Variables in `docker.just` conflict with the importing justfile. Remove ALL variable definitions from `docker.just` — it must only contain recipes and the `_image` computed variable.

### "directory not found" during COPY of node dir
The Dockerfile guards against this with:
```dockerfile
RUN mkdir -p /root/.nvm/versions/node   # after the conditional nvm install
```
If the dir is missing in runtime stage, the guard `[ "$(ls -A ...)" ]` skips the node setup.

### `EXTRA_COPY_DIRS` — copying optional dirs
Uses a `RUN --mount=from=builder` loop:
```dockerfile
RUN --mount=from=builder,source=/app,target=/build \
    for dir in ${EXTRA_COPY_DIRS}; do \
        [ -d "/build/$dir" ] && cp -a "/build/$dir" "./$dir" && chown -R appuser:appuser "./$dir"; \
    done
```
Missing dirs are silently skipped. This is intentional.

### Docker permission denied
```bash
sudo usermod -aG docker $USER   # then log out / restart WSL
```

## Recipes Provided

```
docker-build      Build image (passes all ARGs)
docker-run        Start detached container; auto-mounts .env and ~/.env
docker-run-it     Start interactive container (--rm)
docker-stop       Stop and remove container by app_name
docker-logs       Follow container logs
docker-shell      exec bash in running container
docker-check      List images matching app_name
docker-rmi        Remove image
docker-sync-time  sudo hwclock -s (WSL)
```

## Do Not

- Do not put `set dotenv-load` or `set shell` in `docker.just` — they are set by the importing justfile.
- Do not bake secrets into the image. Pass them at runtime via `--env-file` or `-e`.
- Do not use `uv sync --no-lock` — always use `--locked` to get reproducible builds.
- Do not use `monitoring` extra unless you also add the monitoring env vars at runtime (LangFuse keys etc.), as it will slow startup.
