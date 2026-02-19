"""Generate standalone MCP server Python scripts from configuration.

A generated script delegates to ``genai_tk.mcp.server_builder.serve`` and can
be launched directly with ``uv run``, added as a ``[project.scripts]`` entry
in ``pyproject.toml``, or published via ``uvx``.

Example:
    ```python
    from pathlib import Path
    from genai_tk.mcp.script_generator import generate_server_script

    code = generate_server_script("search")
    Path("server_search.py").write_text(code)
    ```
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def generate_server_script(
    name: str,
    config_path: Path | str | None = None,
) -> str:
    """Render a standalone ``server_<name>.py`` script as a string.

    The rendered script imports ``genai_tk.mcp.server_builder.serve`` at
    runtime and calls it with the provided server name and config path.

    Args:
        name: MCP server name (must exist in the config file).
        config_path: Absolute path to ``servers.yaml``.  When ``None`` the
            generated script lets ``serve()`` auto-detect the path.

    Returns:
        Python source code as a string.

    Example:
        ```python
        code = generate_server_script("chinook", config_path="/project/config/mcp/servers.yaml")
        Path("server_chinook.py").write_text(code)
        ```
    """
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        keep_trailing_newline=True,
    )
    template = env.get_template("server_script.py.jinja")

    safe_name = name.lower().replace(" ", "_").replace("-", "_")
    config_repr = repr(str(config_path)) if config_path is not None else "None"

    code = template.render(
        server_name=name,
        module_name=f"server_{safe_name}",
        script_filename=f"server_{safe_name}.py",
        config_path_repr=config_repr,
        generated_at=datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
    )
    return code


def write_server_script(
    name: str,
    output: Path | str | None = None,
    config_path: Path | str | None = None,
) -> Path:
    """Generate and write a standalone server script to disk.

    Args:
        name: MCP server name.
        output: Destination file path. Defaults to ``server_<name>.py`` in the
            current working directory.
        config_path: Optional path to the YAML config file.

    Returns:
        The path of the written file.

    Example:
        ```python
        path = write_server_script("search")
        print(f"Script written to {path}")
        ```
    """
    if output is None:
        safe = name.lower().replace(" ", "_").replace("-", "_")
        output = Path.cwd() / f"server_{safe}.py"

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    code = generate_server_script(name, config_path=config_path)
    output.write_text(code, encoding="utf-8")
    return output
