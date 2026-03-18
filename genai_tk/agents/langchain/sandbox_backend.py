"""Backward-compatible re-exports for ``AioSandboxBackend``.

The implementation has moved to :mod:`genai_tk.agents.sandbox.aio_backend`.
This module re-exports everything so existing imports keep working.

Prefer importing from the canonical location:

```python
from genai_tk.agents.sandbox import AioSandboxBackend
from genai_tk.agents.sandbox.models import DockerAioSettings
```
"""

from genai_tk.agents.sandbox.aio_backend import AioSandboxBackend, SandboxToolResult
from genai_tk.agents.sandbox.models import DockerAioSettings as AioSandboxBackendConfig

__all__ = ["AioSandboxBackend", "AioSandboxBackendConfig", "SandboxToolResult"]
