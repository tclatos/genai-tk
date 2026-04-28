"""Lightweight Jupyter notebook executor for regression testing.

Executes code cells in `.ipynb` files in order, using a shared ``exec()``
namespace. Skips empty cells and (optionally) cells that install packages.
Ideas taken from : https://medium.com/codetodeploy/automating-jupyter-notebook-testing-a-liteweight-approach-f723273eeacf

Intentionally dependency-free beyond the standard library and Rich so it can
be used in CI without a full Jupyter stack.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class CellResult:
    """Outcome of executing a single notebook cell."""

    cell_index: int
    source: str
    passed: bool
    duration: float
    error: Exception | None = None


@dataclass
class NotebookResult:
    """Aggregated outcome of running an entire notebook."""

    path: Path
    cell_results: list[CellResult] = field(default_factory=list)
    skipped: int = 0

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.cell_results)

    @property
    def failed_cells(self) -> list[CellResult]:
        return [r for r in self.cell_results if not r.passed]

    @property
    def total_duration(self) -> float:
        return sum(r.duration for r in self.cell_results)


def _should_skip(source: str, allow_pip: bool) -> bool:
    stripped = source.strip()
    if not stripped:
        return True
    if not allow_pip and ("%pip" in source or "!pip" in source):
        return True
    return False


def _strip_magics(source: str) -> str:
    """Remove IPython magic / shell-escape lines that exec() cannot handle.

    Lines starting with ``%`` (line magic) or ``!`` (shell escape) are silently
    dropped.  Cell magics (``%%``) would make the whole cell unparseable, so
    cells starting with ``%%`` are left unchanged and will raise a SyntaxError
    (which surfaces as a test failure, which is the desired behaviour).
    """
    cleaned = []
    for line in source.splitlines(keepends=True):
        lstripped = line.lstrip()
        if lstripped.startswith(("%", "!")):
            cleaned.append("# " + line)  # comment out so line numbers stay aligned
        else:
            cleaned.append(line)
    return "".join(cleaned)


def run_notebook(path: Path, allow_pip: bool = False) -> NotebookResult:
    """Execute all code cells in *path* and return a :class:`NotebookResult`.

    Args:
        path: Path to the ``.ipynb`` file.
        allow_pip: When ``True``, cells with ``%pip`` / ``!pip`` are executed
            instead of skipped.

    Returns:
        Aggregated result for the notebook.
    """
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    env: dict = {}
    result = NotebookResult(path=path)

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if _should_skip(source, allow_pip):
            result.skipped += 1
            continue

        executable = _strip_magics(source)
        t0 = time.perf_counter()
        try:
            exec(executable, env)  # noqa: S102
            duration = time.perf_counter() - t0
            result.cell_results.append(CellResult(idx, source, passed=True, duration=duration))
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - t0
            result.cell_results.append(CellResult(idx, source, passed=False, duration=duration, error=exc))
            break  # stop on first failure (notebook state is now undefined)

    return result
