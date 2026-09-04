"""In-process safe Python code executor using AST interpretation.

Based on smolagents local_python_executor with LangChain BaseTool support,
Pydantic v2 data models, and Python 3.12+ syntax.
"""

from __future__ import annotations

import ast
import builtins
import difflib
import inspect
import math
import warnings
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from functools import wraps
from importlib import import_module
from importlib.util import find_spec
from types import BuiltinFunctionType, FunctionType, ModuleType
from typing import Any

from langchain_core.tools import BaseTool
from loguru import logger

from genai_tk.agents.tools.python_executor.models import CodeOutput


class InterpreterError(ValueError):
    """Raised when the interpreter cannot evaluate Python code due to syntax errors or unsupported operations."""


class FinalAnswerException(BaseException):
    """Raised when final_answer is called to signal completion.

    Inherits from BaseException so generic except Exception blocks in agent code don't catch it.
    """

    def __init__(self, value: Any) -> None:
        self.value = value


class BreakException(Exception):
    """Internal loop control exception."""


class ContinueException(Exception):
    """Internal loop control exception."""


class ReturnException(Exception):
    """Internal function return exception."""

    def __init__(self, value: Any) -> None:
        self.value = value


class ExecutionTimeoutError(Exception):
    """Raised when code execution exceeds the maximum allowed time."""


ERRORS: dict[str, type[BaseException]] = {
    name: getattr(builtins, name)
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type) and issubclass(getattr(builtins, name), BaseException)
}

DEFAULT_MAX_LEN_OUTPUT = 50_000
MAX_OPERATIONS = 10_000_000
MAX_WHILE_ITERATIONS = 1_000_000
MAX_EXECUTION_TIME_SECONDS = 30

ALLOWED_DUNDER_METHODS = {
    "__init__",
    "__str__",
    "__repr__",
    "__len__",
    "__iter__",
    "__next__",
    "__enter__",
    "__exit__",
    "__name__",
    "__doc__",
    "__getitem__",
    "__setitem__",
    "__delitem__",
    "__contains__",
    "__call__",
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__add__",
    "__sub__",
    "__mul__",
    "__truediv__",
    "__floordiv__",
    "__mod__",
    "__pow__",
    "__and__",
    "__or__",
    "__xor__",
    "__invert__",
    "__lshift__",
    "__rshift__",
    "__neg__",
    "__pos__",
    "__abs__",
    "__format__",
    "__hash__",
    "__bool__",
    "__int__",
    "__float__",
}


def custom_print(*args: Any) -> None:
    return None


def nodunder_getattr(obj: Any, name: str, default: Any = None) -> Any:
    if name.startswith("__") and name.endswith("__") and name not in ALLOWED_DUNDER_METHODS:
        raise InterpreterError(f"Forbidden access to dunder attribute: {name}")
    return getattr(obj, name, default)


BASE_PYTHON_TOOLS: dict[str, Callable[..., Any]] = {
    "print": custom_print,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "range": range,
    "float": float,
    "int": int,
    "bool": bool,
    "str": str,
    "set": set,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "round": round,
    "ceil": math.ceil,
    "floor": math.floor,
    "log": math.log,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "degrees": math.degrees,
    "radians": math.radians,
    "pow": pow,
    "sqrt": math.sqrt,
    "len": len,
    "sum": sum,
    "max": max,
    "min": min,
    "abs": abs,
    "enumerate": enumerate,
    "zip": zip,
    "reversed": reversed,
    "sorted": sorted,
    "all": all,
    "any": any,
    "map": map,
    "filter": filter,
    "ord": ord,
    "chr": chr,
    "next": next,
    "iter": iter,
    "divmod": divmod,
    "callable": callable,
    "getattr": nodunder_getattr,
    "hasattr": hasattr,
    "setattr": setattr,
    "type": type,
    "complex": complex,
    "repr": repr,
    "bytes": bytes,
    "bytearray": bytearray,
    "hash": hash,
    "format": format,
    "id": id,
    "bin": bin,
    "oct": oct,
    "hex": hex,
    "slice": slice,
}

BASE_BUILTIN_MODULES: list[str] = [
    "collections",
    "datetime",
    "itertools",
    "math",
    "queue",
    "random",
    "re",
    "stat",
    "statistics",
    "time",
    "unicodedata",
    "json",
    "fractions",
    "decimal",
    "string",
    "copy",
    "functools",
    "typing",
]

DANGEROUS_MODULES: list[str] = [
    "builtins",
    "io",
    "multiprocessing",
    "os",
    "pathlib",
    "pty",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "posix",
    "nt",
    "ctypes",
    "threading",
    "signal",
]

DANGEROUS_FUNCTIONS: list[str] = [
    "builtins.compile",
    "builtins.eval",
    "builtins.exec",
    "builtins.globals",
    "builtins.locals",
    "builtins.__import__",
    "os.popen",
    "os.system",
    "posix.system",
]


class LangChainToolAdapter:
    """Adapts a LangChain BaseTool to be callable directly as a function within the interpreter."""

    def __init__(self, tool: BaseTool) -> None:
        self.tool = tool
        self.__name__ = tool.name
        self.__doc__ = tool.description

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs and not args:
            return self.tool.invoke(kwargs)
        if len(args) == 1 and not kwargs:
            arg = args[0]
            if isinstance(arg, dict):
                return self.tool.invoke(arg)
            return self.tool.invoke(arg)
        if args and kwargs:
            return self.tool.invoke(kwargs)
        if args:
            return self.tool.invoke(list(args))
        return self.tool.invoke({})

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self.tool.invoke(*args, **kwargs)

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.tool.run(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tool, name)


class PrintContainer:
    """Accumulates printed output during code evaluation."""

    def __init__(self) -> None:
        self.value: str = ""

    def append(self, text: str) -> PrintContainer:
        self.value += text
        return self

    def __iadd__(self, other: Any) -> PrintContainer:
        self.value += str(other)
        return self

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"PrintContainer({self.value!r})"

    def __len__(self) -> int:
        return len(self.value)


def truncate_content(content: str, max_length: int = DEFAULT_MAX_LEN_OUTPUT) -> str:
    """Truncates content to stay within max_length characters."""
    if len(content) <= max_length:
        return content
    half = max_length // 2
    return (
        content[:half]
        + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
        + content[-half:]
    )


def check_import_authorized(import_to_check: str, authorized_imports: list[str]) -> bool:
    """Checks if an import path is authorized."""
    if "*" in authorized_imports:
        base = import_to_check.split(".")[0]
        return base not in DANGEROUS_MODULES or base in authorized_imports

    for auth in authorized_imports:
        if auth == "*" or auth == import_to_check:
            return True
        if auth.endswith(".*") and import_to_check.startswith(auth[:-2]):
            return True
        if import_to_check.startswith(auth + "."):
            return True
    return False


def get_safe_module(raw_module: Any, authorized_imports: list[str], visited: set[int] | None = None) -> Any:
    """Creates a safe representation of an imported module by copying public attributes."""
    if not isinstance(raw_module, ModuleType):
        return raw_module

    if visited is None:
        visited = set()

    module_id = id(raw_module)
    if module_id in visited:
        return raw_module

    visited.add(module_id)
    safe_module = ModuleType(raw_module.__name__)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for attr_name in dir(raw_module):
            try:
                attr_value = getattr(raw_module, attr_name)
            except (ImportError, AttributeError) as ex:
                logger.debug(f"Skipping import attribute {raw_module.__name__}.{attr_name}: {ex}")
                continue
            if isinstance(attr_value, ModuleType):
                attr_value = get_safe_module(attr_value, authorized_imports, visited=visited)
            setattr(safe_module, attr_name, attr_value)

    return safe_module


def check_safer_result(
    result: Any,
    static_tools: dict[str, Callable[..., Any]] | None = None,
    authorized_imports: list[str] | None = None,
) -> None:
    """Checks if evaluation result is safe."""
    auth = authorized_imports if authorized_imports is not None else BASE_BUILTIN_MODULES
    if isinstance(result, ModuleType):
        if not check_import_authorized(result.__name__, auth):
            raise InterpreterError(f"Forbidden access to module: {result.__name__}")
    elif isinstance(result, dict) and result.get("__spec__") and "__name__" in result:
        if not check_import_authorized(result["__name__"], auth):
            raise InterpreterError(f"Forbidden access to module: {result['__name__']}")
    elif isinstance(result, (FunctionType, BuiltinFunctionType)):
        for qualified_function_name in DANGEROUS_FUNCTIONS:
            module_name, function_name = qualified_function_name.rsplit(".", 1)
            if (
                (static_tools is None or function_name not in static_tools)
                and getattr(result, "__name__", None) == function_name
                and getattr(result, "__module__", None) == module_name
            ):
                raise InterpreterError(f"Forbidden access to function: {function_name}")


def safer_eval(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to verify that AST evaluation return value is safe."""

    @wraps(func)
    def _check_return(
        expression: ast.AST,
        state: dict[str, Any],
        static_tools: dict[str, Callable[..., Any]],
        custom_tools: dict[str, Callable[..., Any]],
        authorized_imports: list[str] = BASE_BUILTIN_MODULES,
    ) -> Any:
        result = func(expression, state, static_tools, custom_tools, authorized_imports=authorized_imports)
        check_safer_result(result, static_tools, authorized_imports)
        return result

    return _check_return


def safer_func(
    func: Callable[..., Any],
    static_tools: dict[str, Callable[..., Any]] = BASE_PYTHON_TOOLS,
    authorized_imports: list[str] = BASE_BUILTIN_MODULES,
) -> Callable[..., Any]:
    """Wraps a callable to verify its return value safety."""
    if isinstance(func, type):
        return func

    @wraps(func)
    def _check_return(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        check_safer_result(result, static_tools, authorized_imports)
        return result

    return _check_return


def evaluate_attribute(
    expression: ast.Attribute,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    if (
        expression.attr.startswith("__")
        and expression.attr.endswith("__")
        and expression.attr not in ALLOWED_DUNDER_METHODS
    ):
        raise InterpreterError(f"Forbidden access to dunder attribute: {expression.attr}")
    value = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)
    return getattr(value, expression.attr)


def evaluate_unaryop(
    expression: ast.UnaryOp,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    operand = evaluate_ast(expression.operand, state, static_tools, custom_tools, authorized_imports)
    if isinstance(expression.op, ast.USub):
        return -operand
    if isinstance(expression.op, ast.UAdd):
        return operand
    if isinstance(expression.op, ast.Not):
        return not operand
    if isinstance(expression.op, ast.Invert):
        return ~operand
    raise InterpreterError(f"Unary operation {expression.op.__class__.__name__} is not supported.")


def evaluate_lambda(
    lambda_expression: ast.Lambda,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Callable[..., Any]:
    args = [arg.arg for arg in lambda_expression.args.args]

    def lambda_func(*values: Any) -> Any:
        new_state = state.copy()
        for arg, value in zip(args, values, strict=False):
            new_state[arg] = value
        return evaluate_ast(
            lambda_expression.body,
            new_state,
            static_tools,
            custom_tools,
            authorized_imports,
        )

    return lambda_func


def evaluate_while(
    while_loop: ast.While,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    iterations = 0
    while evaluate_ast(while_loop.test, state, static_tools, custom_tools, authorized_imports):
        for node in while_loop.body:
            try:
                evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
            except BreakException:
                return None
            except ContinueException:
                break
        iterations += 1
        if iterations > MAX_WHILE_ITERATIONS:
            raise InterpreterError(f"Maximum number of {MAX_WHILE_ITERATIONS} iterations in while loop exceeded")
    else:
        for node in while_loop.orelse:
            evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
    return None


def create_function(
    func_def: ast.FunctionDef,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Callable[..., Any]:
    source_code = ast.unparse(func_def)

    def new_func(*args: Any, **kwargs: Any) -> Any:
        func_state = state.copy()
        arg_names = [arg.arg for arg in func_def.args.args]
        default_values = [
            evaluate_ast(d, state, static_tools, custom_tools, authorized_imports) for d in func_def.args.defaults
        ]

        defaults = dict(zip(arg_names[-len(default_values) :], default_values, strict=False)) if default_values else {}

        for name, value in zip(arg_names, args, strict=False):
            func_state[name] = value

        for name, value in kwargs.items():
            func_state[name] = value

        if func_def.args.vararg:
            vararg_name = func_def.args.vararg.arg
            func_state[vararg_name] = args[len(arg_names) :]

        if func_def.args.kwarg:
            kwarg_name = func_def.args.kwarg.arg
            extra_kwargs = {k: v for k, v in kwargs.items() if k not in arg_names}
            func_state[kwarg_name] = extra_kwargs

        for name, value in defaults.items():
            if name not in func_state:
                func_state[name] = value

        if func_def.args.args and func_def.args.args[0].arg == "self":
            if args:
                func_state["self"] = args[0]
                func_state["__class__"] = args[0].__class__

        result = None
        try:
            for stmt in func_def.body:
                result = evaluate_ast(stmt, func_state, static_tools, custom_tools, authorized_imports)
        except ReturnException as e:
            result = e.value

        if func_def.name == "__init__":
            return None

        return result

    new_func.__ast__ = func_def  # type: ignore[attr-defined]
    new_func.__source__ = source_code  # type: ignore[attr-defined]
    new_func.__name__ = func_def.name

    return new_func


def evaluate_function_def(
    func_def: ast.FunctionDef,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Callable[..., Any]:
    fn = create_function(func_def, state, static_tools, custom_tools, authorized_imports)
    custom_tools[func_def.name] = fn
    state[func_def.name] = fn
    return fn


def evaluate_class_def(
    class_def: ast.ClassDef,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> type:
    class_name = class_def.name
    bases = [evaluate_ast(base, state, static_tools, custom_tools, authorized_imports) for base in class_def.bases]

    metaclass = type
    for base in bases:
        base_metaclass = type(base)
        if base_metaclass is not type:
            metaclass = base_metaclass
            break

    class_dict: dict[str, Any] = {}
    if hasattr(metaclass, "__prepare__"):
        class_dict = metaclass.__prepare__(class_name, tuple(bases))

    for stmt in class_def.body:
        if isinstance(stmt, ast.FunctionDef):
            class_dict[stmt.name] = create_function(stmt, state, static_tools, custom_tools, authorized_imports)
        elif isinstance(stmt, ast.Assign):
            value = evaluate_ast(stmt.value, state, static_tools, custom_tools, authorized_imports)
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    class_dict[target.id] = value
                elif isinstance(target, ast.Attribute):
                    obj = evaluate_ast(target.value, class_dict, static_tools, custom_tools, authorized_imports)
                    setattr(obj, target.attr, value)
        elif isinstance(stmt, ast.Pass):
            pass
        elif (
            isinstance(stmt, ast.Expr)
            and stmt == class_def.body[0]
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            class_dict["__doc__"] = stmt.value.value
        else:
            raise InterpreterError(f"Unsupported statement in class body: {stmt.__class__.__name__}")

    new_class = metaclass(class_name, tuple(bases), class_dict)
    state[class_name] = new_class
    return new_class


def evaluate_annassign(
    annassign: ast.AnnAssign,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    if annassign.value:
        value = evaluate_ast(annassign.value, state, static_tools, custom_tools, authorized_imports)
        set_value(annassign.target, value, state, static_tools, custom_tools, authorized_imports)
        return value
    return None


def evaluate_augassign(
    expression: ast.AugAssign,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    def get_current_value(target: ast.AST) -> Any:
        if isinstance(target, ast.Name):
            return state.get(target.id, 0)
        if isinstance(target, ast.Subscript):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
            return obj[key]
        if isinstance(target, ast.Attribute):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            return getattr(obj, target.attr)
        raise InterpreterError(f"AugAssign not supported for {type(target)} targets.")

    current_value = get_current_value(expression.target)
    value_to_add = evaluate_ast(expression.value, state, static_tools, custom_tools, authorized_imports)

    if isinstance(expression.op, ast.Add):
        current_value += value_to_add
    elif isinstance(expression.op, ast.Sub):
        current_value -= value_to_add
    elif isinstance(expression.op, ast.Mult):
        current_value *= value_to_add
    elif isinstance(expression.op, ast.Div):
        current_value /= value_to_add
    elif isinstance(expression.op, ast.FloorDiv):
        current_value //= value_to_add
    elif isinstance(expression.op, ast.Mod):
        current_value %= value_to_add
    elif isinstance(expression.op, ast.Pow):
        current_value **= value_to_add
    elif isinstance(expression.op, ast.BitAnd):
        current_value &= value_to_add
    elif isinstance(expression.op, ast.BitOr):
        current_value |= value_to_add
    elif isinstance(expression.op, ast.BitXor):
        current_value ^= value_to_add
    elif isinstance(expression.op, ast.LShift):
        current_value <<= value_to_add
    elif isinstance(expression.op, ast.RShift):
        current_value >>= value_to_add
    else:
        raise InterpreterError(f"Operation {type(expression.op).__name__} is not supported.")

    set_value(expression.target, current_value, state, static_tools, custom_tools, authorized_imports)
    return current_value


def evaluate_boolop(
    node: ast.BoolOp,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    is_and = isinstance(node.op, ast.And)
    result = None
    for value in node.values:
        result = evaluate_ast(value, state, static_tools, custom_tools, authorized_imports)
        if is_and and not result:
            return result
        if not is_and and result:
            return result
    return result


def evaluate_binop(
    binop: ast.BinOp,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    left_val = evaluate_ast(binop.left, state, static_tools, custom_tools, authorized_imports)
    right_val = evaluate_ast(binop.right, state, static_tools, custom_tools, authorized_imports)

    if isinstance(binop.op, ast.Add):
        return left_val + right_val
    if isinstance(binop.op, ast.Sub):
        return left_val - right_val
    if isinstance(binop.op, ast.Mult):
        return left_val * right_val
    if isinstance(binop.op, ast.Div):
        return left_val / right_val
    if isinstance(binop.op, ast.FloorDiv):
        return left_val // right_val
    if isinstance(binop.op, ast.Mod):
        return left_val % right_val
    if isinstance(binop.op, ast.Pow):
        return left_val**right_val
    if isinstance(binop.op, ast.BitAnd):
        return left_val & right_val
    if isinstance(binop.op, ast.BitOr):
        return left_val | right_val
    if isinstance(binop.op, ast.BitXor):
        return left_val ^ right_val
    if isinstance(binop.op, ast.LShift):
        return left_val << right_val
    if isinstance(binop.op, ast.RShift):
        return left_val >> right_val
    raise NotImplementedError(f"Binary operation {type(binop.op).__name__} is not implemented.")


def evaluate_assign(
    assign: ast.Assign,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    result = evaluate_ast(assign.value, state, static_tools, custom_tools, authorized_imports)
    for target in assign.targets:
        set_value(target, result, state, static_tools, custom_tools, authorized_imports)
    return result


def set_value(
    target: ast.AST,
    value: Any,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> None:
    if isinstance(target, ast.Name):
        if target.id in static_tools:
            raise InterpreterError(f"Cannot assign to name '{target.id}': doing this would erase an existing tool.")
        state[target.id] = value
    elif isinstance(target, (ast.Tuple, ast.List)):
        if not hasattr(value, "__iter__") or isinstance(value, (str, bytes)):
            raise InterpreterError("Cannot unpack non-iterable value")
        val_list = list(value)
        # Check for starred unpack
        has_starred = any(isinstance(elt, ast.Starred) for elt in target.elts)
        if not has_starred:
            if len(target.elts) != len(val_list):
                raise InterpreterError(f"Cannot unpack tuple of size {len(val_list)} into {len(target.elts)} elements")
            for elem, val in zip(target.elts, val_list, strict=True):
                set_value(elem, val, state, static_tools, custom_tools, authorized_imports)
        else:
            starred_idx = [i for i, elt in enumerate(target.elts) if isinstance(elt, ast.Starred)][0]
            before_count = starred_idx
            after_count = len(target.elts) - starred_idx - 1
            if len(val_list) < before_count + after_count:
                raise InterpreterError("Not enough values to unpack")
            for i in range(before_count):
                set_value(target.elts[i], val_list[i], state, static_tools, custom_tools, authorized_imports)
            starred_target = target.elts[starred_idx].value  # type: ignore[attr-defined]
            starred_slice = val_list[before_count : len(val_list) - after_count]
            set_value(starred_target, starred_slice, state, static_tools, custom_tools, authorized_imports)
            for i in range(after_count):
                set_value(
                    target.elts[starred_idx + 1 + i],
                    val_list[len(val_list) - after_count + i],
                    state,
                    static_tools,
                    custom_tools,
                    authorized_imports,
                )
    elif isinstance(target, ast.Subscript):
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        key = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
        obj[key] = value
    elif isinstance(target, ast.Attribute):
        if target.attr.startswith("__") and target.attr.endswith("__") and target.attr not in ALLOWED_DUNDER_METHODS:
            raise InterpreterError(f"Forbidden setting dunder attribute: {target.attr}")
        obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
        setattr(obj, target.attr, value)


def evaluate_call(
    call: ast.Call,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    func: Any = None
    func_name: str | None = None

    if isinstance(call.func, ast.Call):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(call.func, ast.Lambda):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
    elif isinstance(call.func, ast.Attribute):
        obj = evaluate_ast(call.func.value, state, static_tools, custom_tools, authorized_imports)
        func_name = call.func.attr
        if not hasattr(obj, func_name):
            raise InterpreterError(f"Object {obj} has no attribute '{func_name}'")
        func = getattr(obj, func_name)
    elif isinstance(call.func, ast.Name):
        func_name = call.func.id
        if func_name in state:
            func = state[func_name]
        elif func_name in static_tools:
            func = static_tools[func_name]
        elif func_name in custom_tools:
            func = custom_tools[func_name]
        elif func_name in ERRORS:
            func = ERRORS[func_name]
        else:
            raise InterpreterError(
                f"Forbidden function evaluation: '{call.func.id}' is not among allowed tools or defined in preceding code."
            )
    elif isinstance(call.func, ast.Subscript):
        func = evaluate_ast(call.func, state, static_tools, custom_tools, authorized_imports)
        if not callable(func):
            raise InterpreterError(f"Item is not callable: {call.func}")
    else:
        raise InterpreterError(f"Invalid call function expression: {call.func}")

    args = []
    for arg in call.args:
        if isinstance(arg, ast.Starred):
            args.extend(evaluate_ast(arg.value, state, static_tools, custom_tools, authorized_imports))
        else:
            args.append(evaluate_ast(arg, state, static_tools, custom_tools, authorized_imports))

    kwargs = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            starred_dict = evaluate_ast(keyword.value, state, static_tools, custom_tools, authorized_imports)
            if not isinstance(starred_dict, dict):
                raise InterpreterError(f"Cannot unpack non-dict in **kwargs: {type(starred_dict).__name__}")
            kwargs.update(starred_dict)
        else:
            kwargs[keyword.arg] = evaluate_ast(keyword.value, state, static_tools, custom_tools, authorized_imports)

    if func_name == "super":
        if not args:
            if "__class__" in state and "self" in state:
                return super(state["__class__"], state["self"])
            raise InterpreterError("super() needs arguments when called outside class method")
        cls = args[0]
        if not isinstance(cls, type):
            raise InterpreterError("super() argument 1 must be type")
        if len(args) == 1:
            return super(cls)
        if len(args) == 2:
            return super(cls, args[1])
        raise InterpreterError("super() takes at most 2 arguments")

    if func_name == "print":
        state["_print_outputs"] += " ".join(map(str, args)) + "\n"
        return None

    # Handle LangChain BaseTool instances or adapters
    if isinstance(func, BaseTool):
        func = LangChainToolAdapter(func)

    if (inspect.getmodule(func) == builtins) and inspect.isbuiltin(func) and (func not in static_tools.values()):
        raise InterpreterError(
            f"Invoking builtin function '{func_name}' that has not been explicitly allowed is forbidden."
        )

    if (
        hasattr(func, "__name__")
        and func.__name__.startswith("__")
        and func.__name__.endswith("__")
        and (func.__name__ not in static_tools)
        and (func.__name__ not in ALLOWED_DUNDER_METHODS)
    ):
        raise InterpreterError(f"Forbidden call to dunder function: {func.__name__}")

    return func(*args, **kwargs)


def evaluate_subscript(
    subscript: ast.Subscript,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    index = evaluate_ast(subscript.slice, state, static_tools, custom_tools, authorized_imports)
    value = evaluate_ast(subscript.value, state, static_tools, custom_tools, authorized_imports)
    try:
        return value[index]
    except (KeyError, IndexError, TypeError) as ex:
        err_msg = f"Could not index {value} with '{index}': {type(ex).__name__}: {ex}"
        if isinstance(index, str) and isinstance(value, Mapping):
            matches = difflib.get_close_matches(index, list(value.keys()))
            if matches:
                err_msg += f". Did you mean: {matches}"
        raise InterpreterError(err_msg) from ex


def evaluate_name(
    name: ast.Name,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    if name.id in state:
        return state[name.id]
    if name.id in static_tools:
        return safer_func(static_tools[name.id], static_tools=static_tools, authorized_imports=authorized_imports)
    if name.id in custom_tools:
        return custom_tools[name.id]
    if name.id in ERRORS:
        return ERRORS[name.id]

    matches = difflib.get_close_matches(name.id, list(state.keys()))
    if matches:
        return state[matches[0]]
    raise InterpreterError(f"The variable `{name.id}` is not defined.")


def evaluate_condition(
    condition: ast.Compare,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> bool | Any:
    left = evaluate_ast(condition.left, state, static_tools, custom_tools, authorized_imports)
    result = True
    for i, (op_node, comparator) in enumerate(zip(condition.ops, condition.comparators, strict=True)):
        op = type(op_node)
        right = evaluate_ast(comparator, state, static_tools, custom_tools, authorized_imports)
        if op == ast.Eq:
            current = left == right
        elif op == ast.NotEq:
            current = left != right
        elif op == ast.Lt:
            current = left < right
        elif op == ast.LtE:
            current = left <= right
        elif op == ast.Gt:
            current = left > right
        elif op == ast.GtE:
            current = left >= right
        elif op == ast.Is:
            current = left is right
        elif op == ast.IsNot:
            current = left is not right
        elif op == ast.In:
            current = left in right
        elif op == ast.NotIn:
            current = left not in right
        else:
            raise InterpreterError(f"Unsupported comparison operator: {op}")

        # In Python, vector boolean evaluations (like Pandas Series comparisons) return Series, not bool
        if isinstance(current, bool) and not current:
            return False
        result = current if i == 0 else (result and current)
        left = right
    return result


def evaluate_if(
    if_statement: ast.If,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    result = None
    test_result = evaluate_ast(if_statement.test, state, static_tools, custom_tools, authorized_imports)
    branch = if_statement.body if test_result else if_statement.orelse
    for line in branch:
        line_result = evaluate_ast(line, state, static_tools, custom_tools, authorized_imports)
        if line_result is not None:
            result = line_result
    return result


def evaluate_for(
    for_loop: ast.For,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Any:
    result = None
    iterator = evaluate_ast(for_loop.iter, state, static_tools, custom_tools, authorized_imports)
    for counter in iterator:
        set_value(for_loop.target, counter, state, static_tools, custom_tools, authorized_imports)
        for node in for_loop.body:
            try:
                line_result = evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
                if line_result is not None:
                    result = line_result
            except BreakException:
                return result
            except ContinueException:
                break
    else:
        for node in for_loop.orelse:
            evaluate_ast(node, state, static_tools, custom_tools, authorized_imports)
    return result


def _evaluate_comprehensions(
    comprehensions: list[ast.comprehension],
    evaluate_element: Callable[[dict[str, Any]], Any],
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Generator[Any, None, None]:
    if not comprehensions:
        yield evaluate_element(state)
        return

    comp = comprehensions[0]
    iter_value = evaluate_ast(comp.iter, state, static_tools, custom_tools, authorized_imports)
    for value in iter_value:
        new_state = state.copy()
        set_value(comp.target, value, new_state, static_tools, custom_tools, authorized_imports)
        if all(
            evaluate_ast(if_clause, new_state, static_tools, custom_tools, authorized_imports) for if_clause in comp.ifs
        ):
            yield from _evaluate_comprehensions(
                comprehensions[1:], evaluate_element, new_state, static_tools, custom_tools, authorized_imports
            )


def evaluate_listcomp(
    listcomp: ast.ListComp,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> list[Any]:
    return list(
        _evaluate_comprehensions(
            listcomp.generators,
            lambda comp_state: evaluate_ast(listcomp.elt, comp_state, static_tools, custom_tools, authorized_imports),
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
    )


def evaluate_setcomp(
    setcomp: ast.SetComp,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> set[Any]:
    return set(
        _evaluate_comprehensions(
            setcomp.generators,
            lambda comp_state: evaluate_ast(setcomp.elt, comp_state, static_tools, custom_tools, authorized_imports),
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
    )


def evaluate_dictcomp(
    dictcomp: ast.DictComp,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> dict[Any, Any]:
    return dict(
        _evaluate_comprehensions(
            dictcomp.generators,
            lambda comp_state: (
                evaluate_ast(dictcomp.key, comp_state, static_tools, custom_tools, authorized_imports),
                evaluate_ast(dictcomp.value, comp_state, static_tools, custom_tools, authorized_imports),
            ),
            state,
            static_tools,
            custom_tools,
            authorized_imports,
        )
    )


def evaluate_try(
    try_node: ast.Try,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> None:
    try:
        for stmt in try_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        matched = False
        for handler in try_node.handlers:
            if handler.type is None:
                matched = True
            else:
                exc_type = evaluate_ast(handler.type, state, static_tools, custom_tools, authorized_imports)
                if isinstance(e, exc_type):
                    matched = True

            if matched:
                if handler.name:
                    state[handler.name] = e
                for stmt in handler.body:
                    evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
                break
        if not matched:
            raise e
    else:
        if try_node.orelse:
            for stmt in try_node.orelse:
                evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    finally:
        if try_node.finalbody:
            for stmt in try_node.finalbody:
                evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)


def evaluate_raise(
    raise_node: ast.Raise,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> None:
    exc = (
        evaluate_ast(raise_node.exc, state, static_tools, custom_tools, authorized_imports) if raise_node.exc else None
    )
    cause = (
        evaluate_ast(raise_node.cause, state, static_tools, custom_tools, authorized_imports)
        if raise_node.cause
        else None
    )
    if exc is not None:
        if cause is not None:
            raise exc from cause
        raise exc
    raise InterpreterError("Re-raise is not supported without active exception")


def evaluate_assert(
    assert_node: ast.Assert,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> None:
    test_result = evaluate_ast(assert_node.test, state, static_tools, custom_tools, authorized_imports)
    if not test_result:
        msg = (
            evaluate_ast(assert_node.msg, state, static_tools, custom_tools, authorized_imports)
            if assert_node.msg
            else ast.unparse(assert_node.test)
        )
        raise AssertionError(msg)


def evaluate_with(
    with_node: ast.With,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> None:
    contexts = []
    for item in with_node.items:
        context_expr = evaluate_ast(item.context_expr, state, static_tools, custom_tools, authorized_imports)
        enter_result = context_expr.__enter__()
        contexts.append(context_expr)
        if item.optional_vars and isinstance(item.optional_vars, ast.Name):
            state[item.optional_vars.id] = enter_result

    try:
        for stmt in with_node.body:
            evaluate_ast(stmt, state, static_tools, custom_tools, authorized_imports)
    except Exception as e:
        exc_info = (type(e), e, e.__traceback__)
        for context in reversed(contexts):
            try:
                if context.__exit__(*exc_info):
                    exc_info = (None, None, None)
            except Exception as exit_exc:
                exc_info = (type(exit_exc), exit_exc, exit_exc.__traceback__)
        if exc_info[1] is not None:
            raise exc_info[1].with_traceback(exc_info[2]) from None
    else:
        for context in reversed(contexts):
            context.__exit__(None, None, None)


def evaluate_import(
    expression: ast.Import | ast.ImportFrom,
    state: dict[str, Any],
    authorized_imports: list[str],
) -> None:
    if isinstance(expression, ast.Import):
        for alias in expression.names:
            if check_import_authorized(alias.name, authorized_imports):
                raw_module = import_module(alias.name)
                state[alias.asname or alias.name] = get_safe_module(raw_module, authorized_imports)
            else:
                raise InterpreterError(
                    f"Import of '{alias.name}' is not allowed. Authorized imports: {authorized_imports}"
                )
    elif isinstance(expression, ast.ImportFrom):
        mod_name = expression.module or ""
        if check_import_authorized(mod_name, authorized_imports):
            raw_module = __import__(mod_name, fromlist=[alias.name for alias in expression.names])
            module = get_safe_module(raw_module, authorized_imports)
            if expression.names[0].name == "*":
                if hasattr(module, "__all__"):
                    for name in module.__all__:
                        state[name] = getattr(module, name)
                else:
                    for name in dir(module):
                        if not name.startswith("_"):
                            state[name] = getattr(module, name)
            else:
                for alias in expression.names:
                    if hasattr(module, alias.name):
                        state[alias.asname or alias.name] = getattr(module, alias.name)
                    else:
                        raise InterpreterError(f"Module '{mod_name}' has no attribute '{alias.name}'")
        else:
            raise InterpreterError(f"Import from '{mod_name}' is not allowed. Authorized imports: {authorized_imports}")


def evaluate_generatorexp(
    genexp: ast.GeneratorExp,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> Generator[Any, None, None]:
    return _evaluate_comprehensions(
        genexp.generators,
        lambda comp_state: evaluate_ast(genexp.elt, comp_state, static_tools, custom_tools, authorized_imports),
        state,
        static_tools,
        custom_tools,
        authorized_imports,
    )


def evaluate_delete(
    delete_node: ast.Delete,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str],
) -> None:
    for target in delete_node.targets:
        if isinstance(target, ast.Name):
            if target.id in state:
                del state[target.id]
            else:
                raise InterpreterError(f"Cannot delete name '{target.id}': not defined")
        elif isinstance(target, ast.Subscript):
            obj = evaluate_ast(target.value, state, static_tools, custom_tools, authorized_imports)
            index = evaluate_ast(target.slice, state, static_tools, custom_tools, authorized_imports)
            del obj[index]
        else:
            raise InterpreterError(f"Deletion of {type(target).__name__} targets is not supported")


@safer_eval
def evaluate_ast(
    expression: ast.AST,
    state: dict[str, Any],
    static_tools: dict[str, Callable[..., Any]],
    custom_tools: dict[str, Callable[..., Any]],
    authorized_imports: list[str] = BASE_BUILTIN_MODULES,
) -> Any:
    """Evaluates an AST node in the interpreter context."""
    if state.setdefault("_operations_count", {"counter": 0})["counter"] >= MAX_OPERATIONS:
        raise InterpreterError(f"Reached the max number of operations of {MAX_OPERATIONS}")
    state["_operations_count"]["counter"] += 1

    common_params = (state, static_tools, custom_tools, authorized_imports)

    if isinstance(expression, ast.Assign):
        return evaluate_assign(expression, *common_params)
    if isinstance(expression, ast.AnnAssign):
        return evaluate_annassign(expression, *common_params)
    if isinstance(expression, ast.AugAssign):
        return evaluate_augassign(expression, *common_params)
    if isinstance(expression, ast.Call):
        return evaluate_call(expression, *common_params)
    if isinstance(expression, ast.Constant):
        return expression.value
    if isinstance(expression, ast.Tuple):
        return tuple(evaluate_ast(elt, *common_params) for elt in expression.elts)
    if isinstance(expression, ast.GeneratorExp):
        return evaluate_generatorexp(expression, *common_params)
    if isinstance(expression, ast.ListComp):
        return evaluate_listcomp(expression, *common_params)
    if isinstance(expression, ast.DictComp):
        return evaluate_dictcomp(expression, *common_params)
    if isinstance(expression, ast.SetComp):
        return evaluate_setcomp(expression, *common_params)
    if isinstance(expression, ast.UnaryOp):
        return evaluate_unaryop(expression, *common_params)
    if isinstance(expression, ast.Starred):
        return evaluate_ast(expression.value, *common_params)
    if isinstance(expression, ast.BoolOp):
        return evaluate_boolop(expression, *common_params)
    if isinstance(expression, ast.Break):
        raise BreakException()
    if isinstance(expression, ast.Continue):
        raise ContinueException()
    if isinstance(expression, ast.BinOp):
        return evaluate_binop(expression, *common_params)
    if isinstance(expression, ast.Compare):
        return evaluate_condition(expression, *common_params)
    if isinstance(expression, ast.Lambda):
        return evaluate_lambda(expression, *common_params)
    if isinstance(expression, ast.FunctionDef):
        return evaluate_function_def(expression, *common_params)
    if isinstance(expression, ast.Dict):
        keys = (evaluate_ast(k, *common_params) for k in expression.keys if k is not None)
        values = (evaluate_ast(v, *common_params) for v in expression.values)
        return dict(zip(keys, values, strict=True))
    if isinstance(expression, ast.Expr):
        return evaluate_ast(expression.value, *common_params)
    if isinstance(expression, ast.For):
        return evaluate_for(expression, *common_params)
    if isinstance(expression, ast.FormattedValue):
        value = evaluate_ast(expression.value, *common_params)
        if not expression.format_spec:
            return value
        format_spec = evaluate_ast(expression.format_spec, *common_params)
        return format(value, format_spec)
    if isinstance(expression, ast.If):
        return evaluate_if(expression, *common_params)
    if hasattr(ast, "Index") and isinstance(expression, ast.Index):  # type: ignore[attr-defined]
        return evaluate_ast(expression.value, *common_params)  # type: ignore[attr-defined]
    if isinstance(expression, ast.JoinedStr):
        return "".join(str(evaluate_ast(v, *common_params)) for v in expression.values)
    if isinstance(expression, ast.List):
        return [evaluate_ast(elt, *common_params) for elt in expression.elts]
    if isinstance(expression, ast.Name):
        return evaluate_name(expression, *common_params)
    if isinstance(expression, ast.Subscript):
        return evaluate_subscript(expression, *common_params)
    if isinstance(expression, ast.IfExp):
        test_val = evaluate_ast(expression.test, *common_params)
        return evaluate_ast(expression.body if test_val else expression.orelse, *common_params)
    if isinstance(expression, ast.Attribute):
        return evaluate_attribute(expression, *common_params)
    if isinstance(expression, ast.Slice):
        return slice(
            evaluate_ast(expression.lower, *common_params) if expression.lower is not None else None,
            evaluate_ast(expression.upper, *common_params) if expression.upper is not None else None,
            evaluate_ast(expression.step, *common_params) if expression.step is not None else None,
        )
    if isinstance(expression, ast.While):
        return evaluate_while(expression, *common_params)
    if isinstance(expression, (ast.Import, ast.ImportFrom)):
        return evaluate_import(expression, state, authorized_imports)
    if isinstance(expression, ast.ClassDef):
        return evaluate_class_def(expression, *common_params)
    if isinstance(expression, ast.Try):
        return evaluate_try(expression, *common_params)
    if isinstance(expression, ast.Raise):
        return evaluate_raise(expression, *common_params)
    if isinstance(expression, ast.Assert):
        return evaluate_assert(expression, *common_params)
    if isinstance(expression, ast.With):
        return evaluate_with(expression, *common_params)
    if isinstance(expression, ast.Set):
        return {evaluate_ast(elt, *common_params) for elt in expression.elts}
    if isinstance(expression, ast.Return):
        raise ReturnException(evaluate_ast(expression.value, *common_params) if expression.value else None)
    if isinstance(expression, ast.Pass):
        return None
    if isinstance(expression, ast.Delete):
        return evaluate_delete(expression, *common_params)

    raise InterpreterError(f"AST node '{expression.__class__.__name__}' is not supported.")


def evaluate_python_code(
    code: str,
    static_tools: dict[str, Callable[..., Any]] | None = None,
    custom_tools: dict[str, Callable[..., Any]] | None = None,
    state: dict[str, Any] | None = None,
    authorized_imports: list[str] = BASE_BUILTIN_MODULES,
    max_print_outputs_length: int = DEFAULT_MAX_LEN_OUTPUT,
    timeout_seconds: int | None = MAX_EXECUTION_TIME_SECONDS,
) -> tuple[Any, bool]:
    """Executes Python code in-process using AST evaluation.

    Args:
        code: Python source code string to execute.
        static_tools: Pre-configured tools / functions available in the environment.
        custom_tools: User-defined tools / functions that can be overwritten.
        state: State dictionary to preserve variables across runs.
        authorized_imports: List of package/module names permitted to be imported.
        max_print_outputs_length: Max character length of captured print logs.
        timeout_seconds: Execution timeout in seconds.

    Returns:
        tuple (result, is_final_answer)
    """
    try:
        expression = ast.parse(code)
    except SyntaxError as e:
        raise InterpreterError(
            f"Code parsing failed on line {e.lineno} due to: {type(e).__name__}: {e}\n{e.text}{' ' * (e.offset or 0)}^"
        ) from e

    if state is None:
        state = {}

    tools_map = static_tools.copy() if static_tools is not None else {}
    custom_map = custom_tools if custom_tools is not None else {}

    state["_print_outputs"] = PrintContainer()
    state["_operations_count"] = {"counter": 0}

    if "final_answer" in tools_map:
        orig_fa = tools_map["final_answer"]

        def final_answer_wrapper(*args: Any, **kwargs: Any) -> None:
            raise FinalAnswerException(orig_fa(*args, **kwargs))

        tools_map["final_answer"] = final_answer_wrapper

    def _execute() -> tuple[Any, bool]:
        result = None
        try:
            for node in expression.body:
                result = evaluate_ast(node, state, tools_map, custom_map, authorized_imports)
            state["_print_outputs"].value = truncate_content(
                str(state["_print_outputs"]), max_length=max_print_outputs_length
            )
            return result, False
        except FinalAnswerException as e:
            state["_print_outputs"].value = truncate_content(
                str(state["_print_outputs"]), max_length=max_print_outputs_length
            )
            return e.value, True
        except Exception as e:
            state["_print_outputs"].value = truncate_content(
                str(state["_print_outputs"]), max_length=max_print_outputs_length
            )
            source_seg = ""
            try:
                source_seg = ast.get_source_segment(code, node) or ""
            except Exception:
                pass
            raise InterpreterError(
                f"Code execution failed at line '{source_seg}' due to {type(e).__name__}: {e}"
            ) from e

    if timeout_seconds is not None and timeout_seconds > 0:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_execute)
            try:
                return future.result(timeout=timeout_seconds)
            except FuturesTimeoutError as e:
                raise ExecutionTimeoutError(f"Code execution exceeded timeout of {timeout_seconds} seconds") from e

    return _execute()


class LocalPythonExecutor:
    """In-process Python code executor with AST-based execution and LangChain tool integration."""

    def __init__(
        self,
        additional_authorized_imports: list[str] | None = None,
        max_print_outputs_length: int = DEFAULT_MAX_LEN_OUTPUT,
        additional_functions: dict[str, Callable[..., Any]] | None = None,
        tools: dict[str, BaseTool | Callable[..., Any]] | list[BaseTool] | None = None,
        timeout_seconds: int | None = MAX_EXECUTION_TIME_SECONDS,
        initial_state: dict[str, Any] | None = None,
    ) -> None:
        self.additional_authorized_imports = additional_authorized_imports or []
        self.authorized_imports = list(set(BASE_BUILTIN_MODULES) | set(self.additional_authorized_imports))
        self._check_authorized_imports()

        self.max_print_outputs_length = max_print_outputs_length
        self.timeout_seconds = timeout_seconds
        self.custom_tools: dict[str, Callable[..., Any]] = {}
        self.state: dict[str, Any] = {"__name__": "__main__"}
        if initial_state:
            self.state.update(initial_state)

        self.additional_functions = additional_functions or {}
        self.static_tools: dict[str, Callable[..., Any]] = {}
        self._init_tools(tools)

    def _check_authorized_imports(self) -> None:
        missing = [
            imp.split(".")[0] for imp in self.authorized_imports if imp != "*" and find_spec(imp.split(".")[0]) is None
        ]
        if missing:
            raise InterpreterError(
                f"Non-installed authorized modules: {', '.join(missing)}. Please install them or remove from authorized_imports."
            )

    def _init_tools(self, tools: dict[str, BaseTool | Callable[..., Any]] | list[BaseTool] | None) -> None:
        adapted_tools: dict[str, Callable[..., Any]] = {}
        if isinstance(tools, list):
            for t in tools:
                if isinstance(t, BaseTool):
                    adapted_tools[t.name] = LangChainToolAdapter(t)
                elif callable(t):
                    adapted_tools[getattr(t, "__name__", str(t))] = t
        elif isinstance(tools, dict):
            for name, t in tools.items():
                if isinstance(t, BaseTool):
                    adapted_tools[name] = LangChainToolAdapter(t)
                elif callable(t):
                    adapted_tools[name] = t

        self.static_tools = {**BASE_PYTHON_TOOLS, **adapted_tools, **self.additional_functions}

    def send_tools(self, tools: dict[str, BaseTool | Callable[..., Any]] | list[BaseTool]) -> None:
        """Register or update tools available to the executor."""
        self._init_tools(tools)

    def send_variables(self, variables: dict[str, Any]) -> None:
        """Inject variables into the executor's shared state."""
        self.state.update(variables)

    def reset(self) -> None:
        """Reset internal state."""
        self.state = {"__name__": "__main__"}
        self.custom_tools.clear()

    def __call__(self, code: str) -> CodeOutput:
        """Executes a code snippet and returns structured CodeOutput."""
        try:
            output, is_final_answer = evaluate_python_code(
                code,
                static_tools=self.static_tools,
                custom_tools=self.custom_tools,
                state=self.state,
                authorized_imports=self.authorized_imports,
                max_print_outputs_length=self.max_print_outputs_length,
                timeout_seconds=self.timeout_seconds,
            )
            logs = str(self.state.get("_print_outputs", ""))
            return CodeOutput(output=output, logs=logs, is_final_answer=is_final_answer)
        except Exception as e:
            logs = str(self.state.get("_print_outputs", ""))
            return CodeOutput(output=None, logs=logs, is_final_answer=False, error=str(e))
