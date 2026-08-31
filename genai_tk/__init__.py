"""GenAI Toolkit - Core AI components for building applications."""

__version__ = "0.1.0"

# Import main components to make them easily accessible
try:
    import genai_tk.mcp.compat  # noqa: F401

    from . import core, extra, utils  # noqa: F401
except ImportError:
    # Handle case where dependencies aren't installed
    pass
