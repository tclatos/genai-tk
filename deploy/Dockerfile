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

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder

ARG EXTRAS
ARG INSTALL_NODE

# System deps: git (for git dependencies), graphviz-dev (pygraphviz), ca-certs
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl build-essential libssl-dev graphviz-dev \
        ca-certificates openssl wget \
    && rm -rf /var/lib/apt/lists/*

# Conditionally install Node.js (needed for some Streamlit components)
RUN if [ "$INSTALL_NODE" = "true" ]; then \
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/master/install.sh | bash \
        && export NVM_DIR="$HOME/.nvm" \
        && [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" \
        && nvm install --lts \
        && nvm alias default node \
        && nvm use default; \
    fi
# Ensure the node dir exists even when not installing (prevents COPY failure)
RUN mkdir -p /root/.nvm/versions/node

WORKDIR /app

# Install dependencies first (layer cache: only re-runs when lockfile changes)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --extra ${EXTRAS}

# Copy source and install the project itself
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra ${EXTRAS}

# ── Runtime stage ────────────────────────────────────────────────────────────

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

ARG PKG_NAME
ARG INSTALL_NODE
ARG STREAMLIT_ENTRY
ARG EXTRA_COPY_DIRS

WORKDIR /app

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Copy virtualenv from builder
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Conditionally copy Node.js from builder
RUN if [ "$INSTALL_NODE" = "true" ]; then \
        echo "Node.js will be copied from builder"; \
    fi
COPY --from=builder /root/.nvm/versions/node /opt/node-versions-src/
RUN if [ "$INSTALL_NODE" = "true" ] && [ -d /opt/node-versions-src ] && [ "$(ls -A /opt/node-versions-src 2>/dev/null)" ]; then \
        mv /opt/node-versions-src /opt/node-versions \
        && chmod -R a+rx /opt/node-versions \
        && ln -s /opt/node-versions/* /opt/node; \
    else \
        rm -rf /opt/node-versions-src; \
    fi

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
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

# Add Node.js to PATH only if installed
ENV PATH="/opt/node/bin:/app/.venv/bin:/usr/local/bin:/usr/bin:/bin:/sbin"

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8501/_stcore/health || exit 1

# Default: start Streamlit
CMD streamlit run "/app/${_STREAMLIT_ENTRY}" \
    --server.port=8501 --server.address=0.0.0.0 --server.headless=true \
    --server.enableCORS=false --server.enableXsrfProtection=false
