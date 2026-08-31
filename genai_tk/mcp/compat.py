"""Compatibility shims for MCP 2.x with legacy consumers."""

from __future__ import annotations

import sys
import types


def install_mcp_compat_shims() -> None:
    """Install backwards-compatibility shims for libraries expecting MCP 1.x layouts."""
    try:
        import mcp.server.mcpserver as mcpserver
    except (ImportError, ModuleNotFoundError):
        return

    try:
        import mcp.server.context as server_context
        import mcp.shared.context as shared_context

        if not hasattr(shared_context, "RequestContext") and hasattr(server_context, "ServerRequestContext"):
            shared_context.RequestContext = server_context.ServerRequestContext  # type: ignore[attr-defined]
    except Exception:
        pass

    if "mcp.shared.session" not in sys.modules:
        session_mod = types.ModuleType("mcp.shared.session")
        session_mod.ProgressFnT = object  # type: ignore[attr-defined]
        sys.modules["mcp.shared.session"] = session_mod

    sys.modules["mcp.server.fastmcp"] = mcpserver
    try:
        import mcp.server.mcpserver.server as server_mod

        sys.modules["mcp.server.fastmcp.server"] = server_mod
    except Exception:
        pass

    try:
        import mcp.server.mcpserver.tools as tools_mod

        sys.modules["mcp.server.fastmcp.tools"] = tools_mod
    except Exception:
        pass

    try:
        import mcp.server.mcpserver.utilities as utils_mod

        sys.modules["mcp.server.fastmcp.utilities"] = utils_mod
    except Exception:
        pass

    try:
        import mcp.server.mcpserver.utilities.func_metadata as fm_mod

        sys.modules["mcp.server.fastmcp.utilities.func_metadata"] = fm_mod
    except Exception:
        pass


install_mcp_compat_shims()
