"""CLI commands for managing the OpenSandbox server and Docker sandbox lifecycle.

Provides the ``sandbox`` command group so the slow startup cost can be paid
once (by running ``cli sandbox start``) rather than on every agent invocation.

Usage:
    ```bash
    # Start the opensandbox-server as a background daemon
    cli sandbox start

    # Pre-pull the configured Docker image so the first sandbox is fast
    cli sandbox pull

    # Check what is running
    cli sandbox status

    # Stop the daemon
    cli sandbox stop
    ```

Once ``cli sandbox start`` and ``cli sandbox pull`` have been run, every
subsequent ``cli agents langchain --sandbox docker`` invocation skips the
~1 s server startup *and* benefits from a locally cached Docker image,
reducing total startup overhead by 20-30 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from genai_tk.cli.base import CliTopCommand

# PID file written by ``start`` and removed by ``stop``
_DEFAULT_PID_FILE = Path.home() / ".cache" / "genai-tk" / "opensandbox-server.pid"


def _find_server_binary() -> str:
    """Return the path to the opensandbox-server binary."""
    import shutil

    venv_bin = Path(sys.executable).parent / "opensandbox-server"
    if venv_bin.is_file():
        return str(venv_bin)
    found = shutil.which("opensandbox-server")
    if found:
        return found
    raise RuntimeError(
        "opensandbox-server not found.\n"
        "Install: uv add opensandbox-server && opensandbox-server init-config ~/.sandbox.toml --example docker"
    )


def _read_pid(pid_file: Path) -> int | None:
    """Return the PID stored in *pid_file*, or ``None`` if the file is missing/invalid."""
    try:
        return int(pid_file.read_text().strip())
    except Exception:
        return None


def _is_pid_alive(pid: int) -> bool:
    """Return ``True`` if a process with *pid* is currently running."""
    import os

    try:
        os.kill(pid, 0)  # signal 0: probe without sending any signal
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # process exists but owned by another user


async def _check_server_http(server_url: str, timeout: float = 2.0) -> bool:
    """Return ``True`` if the opensandbox-server responds at *server_url*."""
    import httpx

    try:
        async with httpx.AsyncClient(trust_env=False) as hc:
            resp = await hc.get(f"{server_url}/v1/sandboxes", timeout=timeout)
            return resp.status_code < 500
    except Exception:
        return False


def _docker_image_local(image: str) -> bool:
    """Return ``True`` if *image* is already present in the local Docker cache."""
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False  # docker not installed


def _playwright_installed() -> bool:
    """Return ``True`` if the ``playwright`` Python package is importable."""
    import importlib.util

    return importlib.util.find_spec("playwright") is not None


class SandboxCommands(CliTopCommand):
    """Commands for managing the OpenSandbox server daemon and Docker images."""

    description: str = "Manage the OpenSandbox server and Docker sandbox lifecycle"

    def get_description(self) -> tuple[str, str]:
        return "sandbox", self.description

    def register_sub_commands(self, cli_app: typer.Typer) -> None:  # noqa: C901
        from rich.console import Console

        console = Console()

        @cli_app.command("start")
        def start(
            pid_file: str = typer.Option(str(_DEFAULT_PID_FILE), "--pid-file", help="PID file location"),
            config_url: str = typer.Option("", "--url", help="opensandbox-server URL (default: from sandbox.yaml)"),
        ) -> None:
            """Start the opensandbox-server as a background daemon.

            The server keeps running after this command exits, so every
            subsequent ``cli agents langchain --sandbox docker`` call
            reuses it and skips its ~1 s startup.
            """
            import asyncio
            import subprocess

            from genai_tk.agents.sandbox.config import get_docker_aio_settings

            pid_path = Path(pid_file)
            server_url = config_url or get_docker_aio_settings().opensandbox_server_url

            # Check if already running
            existing_pid = _read_pid(pid_path)
            if existing_pid and _is_pid_alive(existing_pid):
                if asyncio.run(_check_server_http(server_url)):
                    console.print(
                        f"[green]opensandbox-server is already running[/green] (pid {existing_pid}, {server_url})"
                    )
                    return
                # PID alive but not responding yet — let it keep booting
                console.print(
                    f"[yellow]opensandbox-server pid {existing_pid} is alive but not responding yet.[/yellow]"
                )
                return

            try:
                binary = _find_server_binary()
            except RuntimeError as exc:
                console.print(f"[red]{exc}[/red]")
                raise typer.Exit(1) from exc

            pid_path.parent.mkdir(parents=True, exist_ok=True)

            # Launch detached so it outlives this CLI process
            proc = subprocess.Popen(
                [binary],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            pid_path.write_text(str(proc.pid))
            console.print(f"[green]opensandbox-server started[/green] (pid {proc.pid})")
            console.print(f"[dim]PID file: {pid_path}[/dim]")
            console.print(f"[dim]Listening at: {server_url}[/dim]")

            # Wait up to 10 s for the server to become HTTP-reachable
            import time

            deadline = time.monotonic() + 10
            ready = False
            while time.monotonic() < deadline:
                if asyncio.run(_check_server_http(server_url)):
                    ready = True
                    break
                time.sleep(0.5)

            if ready:
                console.print("[green]Server is ready.[/green]")
            else:
                console.print("[yellow]Server started but not yet responding — it may still be initialising.[/yellow]")
            console.print("[yellow]Tip:[/yellow] run [cyan]cli sandbox pull[/cyan] to pre-pull the Docker image.")

        @cli_app.command("stop")
        def stop(
            pid_file: str = typer.Option(str(_DEFAULT_PID_FILE), "--pid-file", help="PID file location"),
        ) -> None:
            """Stop the background opensandbox-server daemon."""
            import signal as _signal

            pid_path = Path(pid_file)
            pid = _read_pid(pid_path)
            if pid is None:
                console.print("[yellow]No PID file found — nothing to stop.[/yellow]")
                return

            if not _is_pid_alive(pid):
                console.print(f"[yellow]Process {pid} is not running.[/yellow]")
                pid_path.unlink(missing_ok=True)
                return

            import os

            try:
                os.kill(pid, _signal.SIGTERM)
                console.print(f"[green]Sent SIGTERM to opensandbox-server[/green] (pid {pid})")
            except ProcessLookupError:
                console.print(f"[yellow]Process {pid} already gone.[/yellow]")
            finally:
                pid_path.unlink(missing_ok=True)

        @cli_app.command("status")
        def status(
            pid_file: str = typer.Option(str(_DEFAULT_PID_FILE), "--pid-file", help="PID file location"),
            config_url: str = typer.Option("", "--url", help="opensandbox-server URL (default: from sandbox.yaml)"),
        ) -> None:
            """Show whether the opensandbox-server is running and reachable."""
            import asyncio

            from rich.table import Table

            from genai_tk.agents.sandbox.config import get_docker_aio_settings

            pid_path = Path(pid_file)
            server_url = config_url or get_docker_aio_settings().opensandbox_server_url
            image = get_docker_aio_settings().image

            pid = _read_pid(pid_path)
            proc_alive = pid is not None and _is_pid_alive(pid)
            http_ok = asyncio.run(_check_server_http(server_url))
            image_local = _docker_image_local(image)
            playwright_ok = _playwright_installed()

            table = Table(title="OpenSandbox Status", show_header=False, box=None)
            table.add_column("Key", style="bold cyan", no_wrap=True)
            table.add_column("Value")

            table.add_row("Server URL", server_url)
            table.add_row(
                "Docker image",
                f"{image}  [green](cached locally)[/green]"
                if image_local
                else f"{image}  [yellow](not pulled yet)[/yellow]",
            )
            table.add_row(
                "playwright",
                "[green]installed[/green]" if playwright_ok else "[red]missing[/red]",
            )
            table.add_row(
                "Daemon PID",
                f"{pid} ({'alive' if proc_alive else 'dead'})" if pid else "[dim]no PID file[/dim]",
            )
            table.add_row(
                "HTTP reachable",
                "[green]yes[/green]" if http_ok else "[red]no[/red]",
            )

            console.print(table)

            tips: list[str] = []
            if not http_ok:
                if proc_alive:
                    tips.append(
                        "[yellow]Server process is alive but not yet responding — it may still be starting up.[/yellow]"
                    )
                else:
                    tips.append("run [cyan]cli sandbox start[/cyan] to start the daemon")
            if not image_local:
                tips.append("run [cyan]cli sandbox pull[/cyan] to cache the Docker image locally")
            if not playwright_ok:
                tips.append("run [cyan]uv sync --group browser-control[/cyan] to install playwright")
            for tip in tips:
                console.print(f"[yellow]Tip:[/yellow] {tip}" if not tip.startswith("[yellow]Server") else tip)

        @cli_app.command("list")
        def list_sandboxes(
            config_url: str = typer.Option("", "--url", help="opensandbox-server URL (default: from sandbox.yaml)"),
        ) -> None:
            """List active sandboxes and their VNC / API URLs.

            Shows the per-container endpoints needed for debugging:
            noVNC URL for visual inspection, API base for shell/browser access.
            """
            import httpx
            from rich.table import Table

            from genai_tk.agents.sandbox.config import get_docker_aio_settings

            server_url = config_url or get_docker_aio_settings().opensandbox_server_url
            try:
                resp = httpx.get(f"{server_url}/v1/sandboxes", timeout=5.0)
                resp.raise_for_status()
            except Exception as exc:
                console.print(f"[red]Cannot reach opensandbox-server at {server_url}:[/red] {exc}")
                raise typer.Exit(1) from exc

            data = resp.json()
            sandboxes = data if isinstance(data, list) else data.get("sandboxes", data.get("data", []))
            if not sandboxes:
                console.print("[dim]No active sandboxes.[/dim]")
                return

            table = Table(title=f"Active Sandboxes ({server_url})")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Status", style="green")
            table.add_column("API URL", style="white")
            table.add_column("VNC URL", style="blue")

            for sbx in sandboxes:
                sbx_id = sbx.get("id", sbx.get("sandbox_id", "?"))
                sbx_status = sbx.get("status", "?")
                # Build per-container URL from endpoints or port mappings
                endpoints = sbx.get("endpoints", sbx.get("ports", {}))
                api_url = ""
                vnc_url = ""
                if isinstance(endpoints, dict):
                    ep_8080 = endpoints.get("8080", endpoints.get(8080, {}))
                    if isinstance(ep_8080, dict):
                        host = ep_8080.get("endpoint", ep_8080.get("host", ""))
                        if host:
                            api_url = f"http://{host}"
                            vnc_url = f"http://{host}/vnc/index.html?autoconnect=true"
                    elif isinstance(ep_8080, str) and ep_8080:
                        api_url = f"http://{ep_8080}"
                        vnc_url = f"http://{ep_8080}/vnc/index.html?autoconnect=true"
                table.add_row(sbx_id[:12], sbx_status, api_url, vnc_url)

            console.print(table)
            console.print(
                "[dim]Open the VNC URL in a browser to see the sandbox desktop and Chromium in real time.[/dim]"
            )

        @cli_app.command("pull")
        def pull(
            image: str = typer.Option("", "--image", "-i", help="Docker image to pull (default: from sandbox.yaml)"),
        ) -> None:
            """Pre-pull the configured Docker sandbox image.

            Pulls the image so that the first ``cli agents langchain --sandbox docker``
            invocation does not stall waiting for the image download.
            """
            import subprocess

            from genai_tk.agents.sandbox.config import get_docker_aio_settings

            target = image or get_docker_aio_settings().image
            console.print(f"[cyan]Pulling Docker image:[/cyan] {target}")
            try:
                result = subprocess.run(
                    ["docker", "pull", target],
                    check=False,
                )
                if result.returncode == 0:
                    console.print(f"[green]Image ready:[/green] {target}")
                else:
                    console.print(f"[red]docker pull failed (exit {result.returncode})[/red]")
                    raise typer.Exit(result.returncode)
            except FileNotFoundError as exc:
                console.print("[red]docker CLI not found — is Docker installed and on PATH?[/red]")
                raise typer.Exit(1) from exc
