#!/usr/bin/env python3
"""
Create Rich hyperlinks to open Office documents from WSL on Windows
"""

import os
import subprocess

from rich import print
from rich.console import Console
from rich.panel import Panel


def get_windows_path(wsl_path):
    """Convert WSL path to a forward-slash Windows UNC path (e.g. //wsl.localhost/Ubuntu/...)"""
    try:
        return subprocess.check_output(["wslpath", "-m", wsl_path]).decode().strip()
    except Exception as e:
        print(f"Error converting path: {e}")
        return None


def create_file_link_panel(file_path):
    """Create a Rich panel with clickable file link"""
    console = Console()

    windows_path = get_windows_path(file_path)
    if not windows_path:
        return

    # windows_path is like //wsl.localhost/Ubuntu/...; prepend "file:" to get a valid file URI
    file_url = f"file:{windows_path}"
    file_name = os.path.basename(file_path)

    console.print(
        Panel(
            f"[bold]Generated File:[/bold]\n"
            f"[link={file_url}]{file_name}[/link]\n"
            f"\n[green]Click the link above to open in Windows![/green]",
            title="📄 Document Created",
            border_style="blue",
        )
    )


# Usage
if __name__ == "__main__":
    dir_path = "/home/tcl/prj/genai-tk/.deer-flow/threads/331369d81e96789b/user-data/outputs"
    file_path = "/home/tcl/prj/genai-tk/.deer-flow/threads/331369d81e96789b/user-data/outputs/einstein_quotes.pptx"
    create_file_link_panel(dir_path)
