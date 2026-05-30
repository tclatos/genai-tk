"""Workflow function registry.

Provides the ``@workflow`` decorator that registers a Python callable as a
named workflow.  YAML pipelines can reference registered names via
``run: <name>`` instead of writing the full dotted import path.

The decorator is purely a side-effect: it leaves the original function unchanged
while adding an entry to the module-level :data:`registry` singleton.

Usage::

    from genai_tk.workflow import workflow


    @workflow(name="kg_build", description="Build a knowledge graph")
    def kg_build_step(*, graphs: list[dict], kg_name: str = "inline") -> dict: ...


    # Use the function name as the registration name:
    @workflow
    def markdownize(*, base_dir: str, output_dir: str) -> dict: ...
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from pydantic import BaseModel


class RegisteredWorkflow(BaseModel):
    """Metadata for a Python-registered workflow callable."""

    name: str
    description: str
    dotted_path: str
    callable_: Any  # the actual function — arbitrary type
    hidden: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def get_params_schema(self) -> dict[str, dict[str, Any]]:
        """Introspect the function signature and return a param info mapping.

        Returns:
            Mapping of parameter name to ``{required, default, annotation}`` info.
        """
        sig = inspect.signature(self.callable_)
        params: dict[str, dict[str, Any]] = {}
        for pname, param in sig.parameters.items():
            if pname in ("self", "cls"):
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            info: dict[str, Any] = {}
            if param.default is inspect.Parameter.empty:
                info["required"] = True
            else:
                info["default"] = param.default
            if param.annotation is not inspect.Parameter.empty:
                info["annotation"] = param.annotation
            params[pname] = info
        return params


class WorkflowRegistry:
    """Global registry of Python-registered workflow callables.

    Use the :func:`workflow` decorator to register callables; do not
    instantiate this class directly — use the module-level singleton
    :data:`registry`.
    """

    _instance: WorkflowRegistry | None = None

    def __new__(cls) -> WorkflowRegistry:
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._entries: dict[str, RegisteredWorkflow] = {}
            cls._instance = inst
        return cls._instance

    def register(self, fn: Callable, *, name: str | None = None, description: str = "", hidden: bool = False) -> None:
        """Register *fn* under an optional *name* (defaults to ``fn.__name__``)."""
        wf_name = name or fn.__name__
        module = getattr(fn, "__module__", None) or ""
        qualname = getattr(fn, "__qualname__", fn.__name__)
        dotted_path = f"{module}.{qualname}" if module else qualname
        desc = description or (fn.__doc__ or "").strip().split("\n")[0]
        self._entries[wf_name] = RegisteredWorkflow(
            name=wf_name,
            description=desc,
            dotted_path=dotted_path,
            callable_=fn,
            hidden=hidden,
        )

    def get(self, name: str) -> RegisteredWorkflow | None:
        """Return registration metadata for *name*, or ``None`` if not found."""
        return self._entries.get(name)

    def list_all(self) -> list[RegisteredWorkflow]:
        """Return all registered workflow entries, sorted by name."""
        return sorted(self._entries.values(), key=lambda e: e.name)

    def __contains__(self, name: str) -> bool:
        return name in self._entries


#: Module-level registry singleton.
registry = WorkflowRegistry()


def workflow(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str = "",
    hidden: bool = False,
) -> Any:
    """Register a callable as a named workflow.

    Can be used as a plain decorator or called with keyword arguments::

        @workflow
        def my_step(...):
            ...

        @workflow(name="custom_name", description="Does X")
        def my_step(...):
            ...

    Args:
        fn: The callable to register (when used without arguments).
        name: Override the registration name (default: ``fn.__name__``).
        description: Human-readable description shown in ``cli workflow list``.
        hidden: When ``True``, omit from ``cli workflow list`` (still runnable).

    Returns:
        The original callable, unmodified.
    """
    if fn is not None:
        # Called as @workflow without parentheses
        registry.register(fn, name=name, description=description, hidden=hidden)
        return fn

    def decorator(fn2: Callable) -> Callable:
        registry.register(fn2, name=name, description=description, hidden=hidden)
        return fn2

    return decorator
