# Monitoring and Tracing

GenAI Toolkit provides unified monitoring support for **LLM calls, agents, and pipelines** across multiple observability backends.

Supported backends:
- **LangSmith** — LangChain's platform (tracing, debugging, testing)
- **LangFuse** — Open-source observability (cloud or self-hosted)
- **OpenTelemetry (OTEL)** — Standard observability protocol (standalone or via LangFuse)
- **Local JSONL** — File-based trace logging (always on, no external service required)

Multiple backends can be **active simultaneously** — traces are sent to all configured backends in parallel.

## Quick Start

### 1. Configure Monitoring

Edit `config/app_conf.yaml` under your profile (e.g., `local:`) monitoring section:

```yaml
monitoring:
  backends: [langfuse, local]
  project: MyProject
  langfuse:
    host: https://cloud.langfuse.com
    public_key: ${oc.env:LANGFUSE_PUBLIC_KEY,""}
    secret_key: ${oc.env:LANGFUSE_SECRET_KEY,""}
  local_log:
    path: ${paths.data_root}/traces/llm_calls.jsonl
    include_prompts: true
```

Or use the provided aliases for cloud vs. self-hosted:

```yaml
monitoring:
  _langfuse_cloud: &langfuse_cloud
    host: https://cloud.langfuse.com
    public_key: ${oc.env:LANGFUSE_PUBLIC_KEY,""}
    secret_key: ${oc.env:LANGFUSE_SECRET_KEY,""}

  _langfuse_local: &langfuse_local
    host: ${oc.env:LANGFUSE_HOST,http://localhost:3000}
    public_key: ${oc.env:LANGFUSE_PUBLIC_KEY,""}
    secret_key: ${oc.env:LANGFUSE_SECRET_KEY,""}

  backends: [langfuse, local]
  project: MyProject
  langfuse: *langfuse_cloud    # Change to *langfuse_local for docker-compose
  local_log:
    path: ${paths.data_root}/traces/llm_calls.jsonl
    include_prompts: true
```

### 2. Set Environment Variables

Create a `.env` file in your home directory (`~/.env`) or project root with:

```bash
# LangFuse Cloud
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# OR LangFuse Self-Hosted
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...

# LangSmith (optional)
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=...

# OpenAI and other LLM keys
OPENAI_API_KEY=...
```

### 3. Run Your Application

The CLI automatically initializes monitoring at startup:

```bash
uv run cli core llm -i "Tell me a joke"
```

Traces will be sent to all active backends. Check status:

```bash
cli monitoring status
```

## Usage

### CLI Monitoring Commands

```bash
# Show active backends and their configuration
cli monitoring status

# View the local JSONL trace log (most recent first)
cli monitoring tail                    # Last 20 entries
cli monitoring tail --n 50             # Last 50 entries
cli monitoring tail --json             # Raw JSON output for piping

# Clear the local trace log
cli monitoring clear --yes

# Manage LangFuse Docker instance (self-hosted)
cli monitoring start langfuse          # Start via docker-compose
cli monitoring stop langfuse           # Stop the containers
cli monitoring open langfuse           # Open UI in browser
cli monitoring open langsmith          # Open LangSmith UI
```

### Programmatic Access

Initialize monitoring in your code:

```python
from genai_tk.utils.tracing import setup_monitoring, get_monitoring_callbacks

# Call once at startup
ctx = setup_monitoring()

# Get LangChain callbacks to pass to LLM invocations
callbacks = get_monitoring_callbacks()

# Use with LLM
from genai_tk.core.factories import get_llm

llm = get_llm("gpt-4o-mini")
response = llm.invoke("Hello", config={"callbacks": callbacks})
```

### Local Trace Log Format

Each trace entry in the JSONL file is a JSON object:

```json
{
  "ts": "2026-06-10T12:34:56.789+00:00",
  "session_id": "a1b2c3d4-...",
  "model": "gpt-4o-mini",
  "framework": "langchain",
  "prompt": "Tell me a joke",
  "response": "Why did the AI …",
  "tokens_in": 10,
  "tokens_out": 25,
  "cost_usd": 0.00012,
  "latency_ms": 450.5,
  "error": null
}
```

Fields:
- `ts` — UTC timestamp (ISO-8601)
- `session_id` — Process-scoped UUID (same for all calls in one process)
- `model` — LLM model identifier
- `framework` — Source framework (`langchain`, `litellm`, `baml`, etc.)
- `prompt` — Input text (truncated per `include_prompts` config)
- `response` — Output text (truncated per `include_prompts` config)
- `tokens_in`, `tokens_out` — Token counts (if available)
- `cost_usd` — Estimated cost (calculated from token counts + pricing DB)
- `latency_ms` — Wall-clock latency in milliseconds
- `error` — Error message (null if no error)

## Configuration Reference

### MonitoringConfig (YAML)

```yaml
monitoring:
  # List of backend names to activate
  # Valid: langsmith, langfuse, otel, local
  backends: [langfuse, local]

  # Project name for LangSmith / LangFuse
  project: MyProject

  # LangSmith settings
  langsmith:
    endpoint: https://api.smith.langchain.com  # Optional override

  # LangFuse settings
  langfuse:
    host: https://cloud.langfuse.com  # Or http://localhost:3000 for self-hosted
    otel_host: ""  # Optional separate OTEL endpoint (auto-derived from host if empty)
    public_key: ${oc.env:LANGFUSE_PUBLIC_KEY,""}
    secret_key: ${oc.env:LANGFUSE_SECRET_KEY,""}

  # OpenTelemetry settings (standalone or via LangFuse)
  otel:
    endpoint: http://localhost:4318
    headers: {}
    service_name: genai-tk

  # Local JSONL trace log
  local_log:
    enabled: true                                  # Can disable without removing from config
    path: ${paths.data_root}/traces/llm_calls.jsonl
    include_prompts: true                          # Log prompts/responses or omit for privacy
    max_prompt_chars: 2000                         # Truncate long inputs to this length
```

### Legacy Configuration

For backward compatibility, you can still use:

```yaml
monitoring:
  langsmith: true    # Equivalent to backends: [langsmith]
  project: MyProject
```

## Switching Backends

### From CLI Config

Edit `config/app_conf.yaml` to change which backends are active:

```yaml
# Only local logging
monitoring:
  backends: [local]
  local_log:
    path: ${paths.data_root}/traces/llm_calls.jsonl

# Only LangSmith
monitoring:
  backends: [langsmith]
  project: MyProject

# Cloud LangFuse only
monitoring:
  backends: [langfuse]
  project: MyProject
  langfuse:
    host: https://cloud.langfuse.com
    public_key: ${oc.env:LANGFUSE_PUBLIC_KEY,""}
    secret_key: ${oc.env:LANGFUSE_SECRET_KEY,""}

# All three
monitoring:
  backends: [langsmith, langfuse, local]
  project: MyProject
  langfuse:
    host: https://cloud.langfuse.com
    public_key: ${oc.env:LANGFUSE_PUBLIC_KEY,""}
    secret_key: ${oc.env:LANGFUSE_SECRET_KEY,""}
  local_log:
    path: ${paths.data_root}/traces/llm_calls.jsonl
```

### From Environment

Set environment variables to control behavior:

```bash
# Disable tracing entirely
export LANGFUSE_TRACING_ENABLED=false

# Use self-hosted LangFuse instead of cloud
export LANGFUSE_BASE_URL=http://localhost:3000

# Disable LangSmith
export LANGSMITH_API_KEY=

# Sample only 10% of traces (LangFuse only)
export LANGFUSE_SAMPLE_RATE=0.1
```

## Self-Hosted LangFuse

### Start with Docker Compose

A docker-compose file is included for running LangFuse locally:

```bash
cli monitoring start langfuse
```

This starts PostgreSQL + LangFuse v3 on `http://localhost:3000`.

### Configure for Self-Hosted

```yaml
monitoring:
  backends: [langfuse, local]
  project: MyProject
  langfuse:
    host: http://localhost:3000
    public_key: ${oc.env:LANGFUSE_PUBLIC_KEY,""}
    secret_key: ${oc.env:LANGFUSE_SECRET_KEY,""}
```

Then set in `.env`:

```bash
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

Create an account in the LangFuse UI at `http://localhost:3000`, copy the keys, and add to `.env`.

## Observability Integrations

### LangChain Callbacks

The monitoring system automatically registers LangChain callbacks for all active backends. When you call an LLM with:

```python
llm.invoke(input, config={"callbacks": get_monitoring_callbacks()})
```

Traces are sent to:
- **LangSmith** — via `LANGCHAIN_TRACING_V2` env var
- **LangFuse** — via `langfuse.langchain.CallbackHandler`
- **OTEL** — via `openinference-instrumentation-langchain` auto-instrumentation
- **Local** — via `LocalTraceLog` callback handler

### SmolAgents

SmolAgents is automatically instrumented via OpenInference if the backend is configured:

```python
from genai_tk.utils.tracing import setup_monitoring

setup_monitoring()  # Auto-instruments SmolAgents

# Now use SmolAgents normally — tracing is automatic
agent = CodeAgent(tools=[...])
agent.run("your task")
```

### LiteLLM

LiteLLM is automatically configured to send traces to LangFuse via OTEL when that backend is active:

```python
import litellm
from genai_tk.utils.tracing import setup_monitoring

setup_monitoring()  # Configures LiteLLM callbacks

response = litellm.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### BAML

Use the `log_llm_call` helper for BAML extraction results:

```python
from genai_tk.utils.local_trace_log import log_llm_call

log_llm_call(
    model="gpt-4o",
    framework="baml",
    prompt="Extract entities",
    response="...",
    tokens_in=100,
    tokens_out=50,
    latency_ms=250
)
```

## Troubleshooting

### Traces Not Appearing in LangFuse/LangSmith

1. **Verify backend is active:**
   ```bash
   cli monitoring status
   ```
   Should show `✓` next to the backend and `keys=set`.

2. **Check env vars are loaded:**
   ```bash
   uv run python -c "import os; print(os.environ.get('LANGFUSE_PUBLIC_KEY', 'NOT SET'))"
   ```
   Should print the key value, not `NOT SET`.

3. **Verify auth:**
   ```bash
   uv run python -c "
   from langfuse import Langfuse
   lf = Langfuse()
   print('Auth check:', lf.auth_check())
   "
   ```
   Should print `True`.

4. **Check callbacks are being passed to LLM:**
   ```python
   from genai_tk.utils.tracing import get_monitoring_callbacks
   cbs = get_monitoring_callbacks()
   print('Callbacks:', cbs)
   ```
   Should list the active handlers.

### Local Log File Not Created

1. **Check directory permissions:**
   ```bash
   ls -la $(dirname "$(uv run python -c 'from genai_tk.utils.tracing import monitoring_config; print(monitoring_config().local_log.path)')")
   ```

2. **Verify `include_prompts: true` in config** — without this, nothing is logged to local.

3. **Call an LLM:**
   ```bash
   cli core llm -i "test"
   ```
   The first call creates the log file.

### "No Langfuse client with public key X has been initialized"

This is a warning from the LangFuse v4 SDK when multiple projects are configured without an explicit key. It's safe to ignore; the CallbackHandler initializes correctly on its own.

## Advanced

### Custom Span Attributes

Add custom attributes to spans via the `metadata` field in LLM invocations:

```python
llm.invoke(
    input,
    config={
        "callbacks": get_monitoring_callbacks(),
        "metadata": {
            "user_id": "user-123",
            "session": "session-456",
            "custom_field": "value"
        }
    }
)
```

### Filtering Spans

Only certain spans are exported. Use `should_export_span` to customize:

```python
from genai_tk.utils.tracing import setup_monitoring

def my_filter(span):
    # Export only generation spans with cost > $0.01
    return span.attributes.get("gen_ai.usage.output_token_count", 0) > 50

setup_monitoring()  # Then customize in your app code
```

### Cost Calculation

Local trace cost is estimated from token counts using an embedded pricing database (`data/models_dev.json`). Costs are approximate and assume on-demand pricing; actual costs depend on your billing plan.

For accurate cost tracking, use LangFuse or LangSmith's native cost reporting.

## See Also

- [docs/core.md](core.md) — LLM factory and model selection
- [docs/agents.md](agents.md) — Agent frameworks and tool integration
- [docs/cli.md](cli.md) — CLI command reference including `cli monitoring`
- `config/examples/monitoring.yaml` — Configuration templates for different setups
