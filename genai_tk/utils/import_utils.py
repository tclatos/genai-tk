"""Dynamic import utilities for loading functions and classes by qualified name.

Qualified names follow the ``module.path:ObjectName`` convention used throughout
the configuration system.

Example:
    ```python
    from genai_tk.utils.import_utils import ImportResolver

    fn = ImportResolver.import_from_qualified("mymodule.sub:my_func")
    result = fn(arg1, arg2)

    # Short-name lookup (searches the project for a unique class definition)
    cls = ImportResolver.import_from_qualified("MySpecialClass")
    obj = cls()
    ```
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Callable


class ImportResolver:
    """Utilities for importing Python objects by qualified name.

    All methods are static; do not instantiate this class.
    """

    @staticmethod
    def split_qualified_name(qualified_name: str) -> tuple[str, str]:
        """Split a qualified name into ``(module_path, object_name)``.

        Args:
            qualified_name: Name in ``module.path:ObjectName`` format.
        """
        module_path, sep, object_name = qualified_name.partition(":")
        if not sep or not module_path or not object_name:
            raise ValueError(f"Invalid qualified name '{qualified_name}'. Expected format 'module.path:ObjectName'.")
        return module_path, object_name

    @staticmethod
    def get_module(qualified_name: str) -> str:
        """Return the module path portion of a qualified name."""
        module_path, _ = ImportResolver.split_qualified_name(qualified_name)
        return module_path

    @staticmethod
    def get_object_name(qualified_name: str) -> str:
        """Return the object name portion of a qualified name."""
        _, object_name = ImportResolver.split_qualified_name(qualified_name)
        return object_name

    @staticmethod
    def import_from_qualified(qualified_name: str) -> Callable:
        """Dynamically import and return a function, class, or object by its qualified name.

        The name can be:

        - Fully qualified: ``'module.submodule:FunctionOrClassName'``
        - Short class name: ``'ClassName'`` — scans the project for a unique match.

        Short-name lookup raises ``ValueError`` when zero or multiple matches are found.

        Examples:
            ```python
            fn = ImportResolver.import_from_qualified("mymod.utils:helper_func")
            result = fn(arg1)

            cls = ImportResolver.import_from_qualified("MyUniqueClass")
            obj = cls()
            ```
        """
        if ":" not in qualified_name:
            return ImportResolver._import_from_short_name(qualified_name)

        module_path, object_name = ImportResolver.split_qualified_name(qualified_name)
        try:
            module = importlib.import_module(module_path)
            return getattr(module, object_name)
        except ImportError as e:
            raise ImportError(f"Cannot import module '{module_path}' for '{qualified_name}': {e}") from e
        except AttributeError as e:
            raise AttributeError(
                f"Cannot find '{object_name}' in module '{module_path}' for '{qualified_name}': {e}"
            ) from e

    @staticmethod
    def _import_from_short_name(class_name: str) -> Callable:
        """Search the project for a class definition by short name and import it.

        Scans all ``.py`` files under the nearest project root (directory containing
        ``pyproject.toml`` or ``setup.py``), using AST analysis to find class
        definitions. Skips virtual-environment and cache directories.

        Args:
            class_name: Short class name, e.g. ``'CrmExtractSubGraph'``.
        """
        current_dir = Path.cwd()
        project_root = current_dir
        for parent in [current_dir, *current_dir.parents]:
            if (parent / "pyproject.toml").exists() or (parent / "setup.py").exists():
                project_root = parent
                break

        _SKIP = {".venv", "venv", "__pycache__", ".git", "node_modules", ".eggs"}
        matches: list[tuple[str, str, Path]] = []

        for py_file in project_root.rglob("*.py"):
            if any(part in py_file.parts for part in _SKIP):
                continue
            try:
                tree = ast.parse(py_file.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        rel = py_file.relative_to(project_root)
                        module_path = ".".join([*rel.parts[:-1], rel.stem])
                        matches.append((module_path, class_name, py_file))
                        break
            except (SyntaxError, UnicodeDecodeError):
                continue

        if not matches:
            raise ValueError(
                f"No class named '{class_name}' found in the project. "
                "Use the fully qualified format 'module.path:ClassName'."
            )
        if len(matches) > 1:
            details = "\n".join(f"  - {m[0]}:{m[1]} (in {m[2]})" for m in matches)
            raise ValueError(
                f"Multiple classes named '{class_name}' found:\n{details}\n"
                "Use the fully qualified format to select the right one."
            )

        mod_path, obj_name, _ = matches[0]
        try:
            return getattr(importlib.import_module(mod_path), obj_name)
        except ImportError as e:
            raise ImportError(f"Cannot import module '{mod_path}' for class '{class_name}': {e}") from e
        except AttributeError as e:
            raise AttributeError(f"Cannot find class '{obj_name}' in module '{mod_path}': {e}") from e

    @staticmethod
    def instantiate_from_qualified_names(
        qualified_names: list[str],
        logger: object | None = None,
    ) -> list[object]:
        """Import and instantiate classes from a list of qualified names.

        Each entry in *qualified_names* must be a fully-qualified class path such as
        ``mypackage.module.MyClass``.  Classes are imported via :mod:`importlib` and
        instantiated with no constructor arguments.

        Args:
            qualified_names: List of ``module.ClassName`` strings.
            logger: Optional logger for debug output (must have ``.debug()`` method).

        Returns:
            List of instantiated objects in the same order as input.

        Raises:
            ValueError: If a name is not fully-qualified.
            ImportError: If a module cannot be imported or class is not found.

        Example:
            ```python
            middlewares = ImportResolver.instantiate_from_qualified_names([
                "mymod.LoggingMiddleware",
                "mymod.ValidatingMiddleware",
            ])
            ```
        """
        instances = []
        for qname in qualified_names:
            if "." not in qname:
                raise ValueError(f"'{qname}' must be fully-qualified: module.ClassName")

            module_path, class_name = qname.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name)
            except (ImportError, AttributeError) as exc:
                raise ImportError(f"Cannot load class '{qname}': {exc}") from exc

            instances.append(cls())
            if logger is not None:
                logger.debug(f"Instantiated: {qname}")

        return instances


# ---------------------------------------------------------------------------
# Module-level convenience aliases (backward-compatible with old config_mngr exports)
# ---------------------------------------------------------------------------


def split_qualified_name(qualified_name: str) -> tuple[str, str]:
    """Alias for ``ImportResolver.split_qualified_name()``."""
    return ImportResolver.split_qualified_name(qualified_name)


def get_module_from_qualified(qualified_name: str) -> str:
    """Alias for ``ImportResolver.get_module()``."""
    return ImportResolver.get_module(qualified_name)


def get_object_name_from_qualified(qualified_name: str) -> str:
    """Alias for ``ImportResolver.get_object_name()``."""
    return ImportResolver.get_object_name(qualified_name)


def import_from_qualified(qualified_name: str) -> Callable:
    """Alias for ``ImportResolver.import_from_qualified()``."""
    return ImportResolver.import_from_qualified(qualified_name)


def instantiate_from_qualified_names(qualified_names: list[str], logger: object | None = None) -> list[object]:
    """Alias for ``ImportResolver.instantiate_from_qualified_names()``."""
    return ImportResolver.instantiate_from_qualified_names(qualified_names, logger=logger)
