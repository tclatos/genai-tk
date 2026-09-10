"""GenAI Toolkit - Core AI components for building applications."""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"

_LAZY_MODULES = {
    "core": "genai_tk.core",
    "extra": "genai_tk.extra",
    "utils": "genai_tk.utils",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES:
        mod = importlib.import_module(_LAZY_MODULES[name])
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_MODULES.keys()))
