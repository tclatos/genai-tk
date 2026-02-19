"""MCP server exposure package for genai-tk.

This package allows exposing LangChain tools and agents as MCP servers,
configurable via YAML, launchable via CLI or standalone scripts.

Example:
    ```bash
    # Serve a configured MCP server over stdio
    uv run cli mcp serve --name search

    # List available servers
    uv run cli mcp list

    # Generate a standalone script
    uv run cli mcp generate --name search --output server_search.py
    ```
"""
