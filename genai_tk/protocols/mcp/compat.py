"""Compatibility shims for MCP 2.x with legacy consumers."""

from __future__ import annotations

import sys
import types
from typing import Any


def install_mcp_compat_shims() -> None:
    """Install backwards-compatibility shims for libraries expecting MCP 1.x layouts."""
    try:
        import mcp.server.mcpserver as mcpserver
    except (ImportError, ModuleNotFoundError):
        return

    # 1. Alias FastMCP class and error in mcpserver if not present
    if not hasattr(mcpserver, "FastMCP"):
        mcpserver.FastMCP = mcpserver.MCPServer  # type: ignore[attr-defined]
    try:
        import mcp.server.mcpserver.exceptions as mcp_exc

        if not hasattr(mcp_exc, "FastMCPError") and hasattr(mcp_exc, "MCPServerError"):
            mcp_exc.FastMCPError = mcp_exc.MCPServerError  # type: ignore[attr-defined]
    except Exception:
        pass

    # 2. Context shims
    try:
        import mcp.server.context as server_context
        import mcp.shared.context as shared_context

        if not hasattr(shared_context, "RequestContext") and hasattr(server_context, "ServerRequestContext"):
            shared_context.RequestContext = server_context.ServerRequestContext  # type: ignore[attr-defined]
    except Exception:
        pass

    # 3. Session shims
    if "mcp.shared.session" not in sys.modules:
        session_mod = types.ModuleType("mcp.shared.session")
        session_mod.ProgressFnT = object  # type: ignore[attr-defined]
        sys.modules["mcp.shared.session"] = session_mod

    # 4. Version shims
    if "mcp.shared.version" not in sys.modules:
        try:
            import mcp.types.version as types_version

            sys.modules["mcp.shared.version"] = types_version
        except Exception:
            pass

    # 5. FastMCP module shims (mcp.server.fastmcp.*)
    sys.modules["mcp.server.fastmcp"] = mcpserver
    try:
        import mcp.server.mcpserver.server as server_mod

        if not hasattr(server_mod, "FastMCP"):
            server_mod.FastMCP = server_mod.MCPServer  # type: ignore[attr-defined]
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

    try:
        import mcp.server.mcpserver.exceptions as exc_mod

        sys.modules["mcp.server.fastmcp.exceptions"] = exc_mod
    except Exception:
        pass

    try:
        import mcp.server.mcpserver.prompts as prompts_mod

        sys.modules["mcp.server.fastmcp.prompts"] = prompts_mod
    except Exception:
        pass

    try:
        import mcp.server.mcpserver.resources as resources_mod

        sys.modules["mcp.server.fastmcp.resources"] = resources_mod
    except Exception:
        pass

    # 6. McpError alias
    try:
        import mcp
        import mcp.types as mcp_types

        if hasattr(mcp, "MCPError"):
            if not hasattr(mcp, "McpError"):
                mcp.McpError = mcp.MCPError  # type: ignore[attr-defined]
            if not hasattr(mcp_types, "McpError"):
                mcp_types.McpError = mcp.MCPError  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        import mcp.shared.exceptions as shared_exc

        if hasattr(shared_exc, "MCPError") and not hasattr(shared_exc, "McpError"):
            shared_exc.McpError = shared_exc.MCPError  # type: ignore[attr-defined]
    except Exception:
        pass

    # 7. Wire model attribute camelCase compatibility (Tool.inputSchema, CallToolResult.isError, etc.)
    try:
        import mcp.types as mcp_types

        if hasattr(mcp_types, "Tool") and not hasattr(mcp_types.Tool, "inputSchema"):
            mcp_types.Tool.inputSchema = property(lambda self: self.input_schema)  # type: ignore[attr-defined]
        if hasattr(mcp_types, "CallToolResult"):
            if not hasattr(mcp_types.CallToolResult, "isError"):
                mcp_types.CallToolResult.isError = property(lambda self: self.is_error)  # type: ignore[attr-defined]
            if not hasattr(mcp_types.CallToolResult, "structuredContent"):
                mcp_types.CallToolResult.structuredContent = property(  # type: ignore[attr-defined]
                    lambda self: getattr(self, "structured_content", None)
                )
        if hasattr(mcp_types, "ListToolsResult") and not hasattr(mcp_types.ListToolsResult, "nextCursor"):
            mcp_types.ListToolsResult.nextCursor = property(lambda self: self.next_cursor)  # type: ignore[attr-defined]
    except Exception:
        pass

    # 8. ClientSession list_* methods cursor parameter backwards compatibility
    try:
        from mcp.client.session import ClientSession
        from mcp.types import PaginatedRequestParams

        for method_name in ("list_tools", "list_resources", "list_prompts", "list_resource_templates"):
            if hasattr(ClientSession, method_name):
                orig_method = getattr(ClientSession, method_name)

                def _make_compat_list(orig_fn):
                    async def _compat_list(self, *args, cursor: str | None = None, params: Any = None, **kwargs):
                        if cursor is not None and params is None:
                            params = PaginatedRequestParams(cursor=cursor)
                        return await orig_fn(self, *args, params=params, **kwargs)

                    return _compat_list

                setattr(ClientSession, method_name, _make_compat_list(orig_method))
    except Exception:
        pass


install_mcp_compat_shims()
