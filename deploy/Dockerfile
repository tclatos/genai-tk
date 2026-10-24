# check=skip=SecretsUsedInArgOrEnv
# Generic Dockerfile for genai-tk based applications.
# Works for the toolkit itself and for scaffolded apps (e.g. rfq_pricing).
#
# Build args:
#   PKG_NAME         Python package directory (default: genai_tk)
#   EXTRAS           Comma-separated uv extras (default: streamlit)
#   INSTALL_NODE     Install Node.js in the image (default: false)
#   STREAMLIT_ENTRY  Streamlit entrypoint script (default: <PKG_NAME>/webapp/main/streamlit.py)
#   EXTRA_COPY_DIRS  Space-separated additional dirs to copy into image (default: "")
#
# Usage:
#   docker build -f deploy/Dockerfile -t myapp:latest .
#   docker build -f deploy/Dockerfile -t myapp:latest --build-arg EXTRAS=streamlit,nlp .
#   docker build -f deploy/Dockerfile -t myapp:latest --build-arg INSTALL_NODE=true .

# ── Build args (available in both stages) ────────────────────────────────────

ARG PKG_NAME=genai_tk
ARG EXTRAS=streamlit
ARG INSTALL_NODE=false
ARG STREAMLIT_ENTRY=""
ARG EXTRA_COPY_DIRS=""

# ── Builder stage ────────────────────────────────────────────────────────────

# Keep tags for update discovery and digests for immutable image selection.
FROM ghcr.io/astral-sh/uv:0.13.5@sha256:32a7bb39b496babbd997b0a39a4d66c126bdaf5661b302bf6f06abda8deca6ff AS uv
FROM node:24.22.1-bookworm-slim@sha256:55fb5f5a527ea2c5c4a95f963e48ef90aa38c7b346054fdc8ee9957a2b55d205 AS node
FROM python:3.12.15-bookworm@sha256:14a4933e6cef421424f7a66d89a3f4eafcbd0d90e0a822251f298e05132dc735 AS builder

COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=node /usr/local/bin/node /opt/node-src/bin/node
COPY --from=node /usr/local/lib/node_modules /opt/node-src/lib/node_modules

ENV UV_PYTHON_DOWNLOADS=never \
    PATH="/opt/node/bin:$PATH"

ARG EXTRAS
ARG INSTALL_NODE

# System deps: git (for git dependencies), graphviz-dev (pygraphviz), ca-certs
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential libssl-dev graphviz-dev \
        ca-certificates openssl wget \
    && rm -rf /var/lib/apt/lists/*

# No network installer is executed for optional Node.js support.
RUN mkdir -p /opt/node \
    && if [ "$INSTALL_NODE" = "true" ]; then \
        cp -a /opt/node-src/. /opt/node/ \
        && ln -s ../lib/node_modules/npm/bin/npm-cli.js /opt/node/bin/npm \
        && ln -s ../lib/node_modules/npm/bin/npx-cli.js /opt/node/bin/npx; \
    fi \
    && rm -rf /opt/node-src

WORKDIR /app

# Install dependencies first (layer cache: only re-runs when lockfile changes)
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; set --; \
    for extra in $(printf '%s' "$EXTRAS" | tr ',' ' '); do set -- "$@" --extra "$extra"; done; \
    uv sync --locked --no-default-groups --no-install-project "$@"

# Copy source and install the project itself
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; set --; \
    for extra in $(printf '%s' "$EXTRAS" | tr ',' ' '); do set -- "$@" --extra "$extra"; done; \
    uv sync --locked --no-default-groups "$@"

# ── Runtime stage ────────────────────────────────────────────────────────────

FROM python:3.12.15-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70 AS runtime

COPY --from=uv /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

ARG PKG_NAME
ARG INSTALL_NODE
ARG STREAMLIT_ENTRY
ARG EXTRA_COPY_DIRS

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Copy virtualenv from builder
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Empty when INSTALL_NODE=false.
COPY --from=builder /opt/node /opt/node

# Create non-root user and data directories
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser \
    && mkdir -p /data/external /data/llm_cache /data/hf_models \
               /data/vector_store /data/kv_store /tmp/mcp-workspace \
    && chown -R appuser:appuser /data /home/appuser /tmp/mcp-workspace

# Copy application files with proper ownership
COPY --from=builder --chown=appuser:appuser /app/${PKG_NAME} ./${PKG_NAME}
COPY --from=builder --chown=appuser:appuser /app/config ./config
COPY --from=builder --chown=appuser:appuser /app/pyproject.toml ./
COPY --from=builder --chown=appuser:appuser /app/uv.lock ./
COPY --from=builder --chown=appuser:appuser /app/.python-version ./

# Copy optional directories (skills, data, .streamlit, use_case_data, etc.)
# Each project can pass EXTRA_COPY_DIRS="skills data .streamlit" to include them.
# We use a shell loop so missing dirs don't break the build.
RUN --mount=from=builder,source=/app,target=/build \
    for dir in ${EXTRA_COPY_DIRS}; do \
        if [ -d "/build/$dir" ]; then \
            cp -a "/build/$dir" "./$dir" \
            && chown -R appuser:appuser "./$dir"; \
        fi; \
    done

# Resolve the streamlit entry point:
#   - if STREAMLIT_ENTRY is set explicitly, use it
#   - otherwise default to <PKG_NAME>/webapp/main/streamlit.py
ENV _STREAMLIT_ENTRY=${STREAMLIT_ENTRY:-${PKG_NAME}/webapp/main/streamlit.py}

ENV PYTHONPATH="/app" \
    UV_PYTHON_DOWNLOADS=never \
    UV_LOCKED=true \
    UV_NO_DEFAULT_GROUPS=true \
    REQUESTS_CA_BUNDLE=/app/.venv/lib/python3.12/site-packages/certifi/cacert.pem \
    SSL_CERT_FILE=/app/.venv/lib/python3.12/site-packages/certifi/cacert.pem \
    CURL_CA_BUNDLE=/app/.venv/lib/python3.12/site-packages/certifi/cacert.pem

# Add Node.js to PATH only if installed
ENV PATH="/opt/node/bin:/app/.venv/bin:/usr/local/bin:/usr/bin:/bin:/sbin"

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8501/_stcore/health || exit 1

# Expand the configured entrypoint, then deliver container signals to Streamlit.
CMD ["sh", "-c", "exec streamlit run \"/app/${_STREAMLIT_ENTRY}\" --server.port=8501 --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
