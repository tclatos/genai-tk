"""Implement ``once``, a decorator that ensures the wrapped function is called once and returns the same result.

Typically used for thread-safe singleton instance creation.  Inspired by the
``once`` keyword in the Eiffel programming language.
"""

from __future__ import annotations

import inspect
from threading import Lock
from typing import Any, Callable, TypeVar

import wrapt

R = TypeVar("R")


def _make_hashable(obj: Any) -> Any:
    """Recursively convert an object to a hashable representation."""
    if obj is None or isinstance(obj, (int, float, str, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    if hasattr(obj, "__dict__"):
        return _make_hashable(vars(obj))
    return str(obj)


def once(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator that caches the result and returns the same value on subsequent calls.

    Thread-safe via double-checked locking.  Results are keyed by arguments so
    different call signatures produce independent cached values.

    Attach ``.invalidate()`` on the result to clear the cache.

    Example:
        ```python
        class MyClass(BaseModel):
            model_config = ConfigDict(frozen=True)

            @once
            def singleton() -> "MyClass":
                return MyClass()


        obj = MyClass.singleton()
        MyClass.singleton.invalidate()  # reset


        @once
        def get_singleton() -> MyClass:
            return MyClass()


        get_singleton.invalidate()
        ```
    """
    inner: Callable = func.__func__ if isinstance(func, staticmethod) else func  # type: ignore[union-attr]

    _cache: dict = {}
    _lock = Lock()

    @wrapt.decorator
    def _wrapper(wrapped: Callable, instance: Any, args: tuple, kwargs: dict) -> Any:  # noqa: ARG001
        sig = inspect.signature(wrapped)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        key = tuple(sorted((k, _make_hashable(v)) for k, v in bound.arguments.items()))
        if key not in _cache:
            with _lock:
                if key not in _cache:
                    _cache[key] = wrapped(*args, **kwargs)
        return _cache[key]

    def invalidate() -> None:
        with _lock:
            _cache.clear()

    wrapped_fn = _wrapper(inner)
    wrapped_fn.invalidate = invalidate  # type: ignore[attr-defined]
    result = staticmethod(wrapped_fn)
    result.invalidate = invalidate  # type: ignore[attr-defined]
    return result  # type: ignore[return-value]


# TEST


if __name__ == "__main__":
    from pydantic import BaseModel, ConfigDict

    class TestClass1(BaseModel):
        model_config = ConfigDict(frozen=True)
        a: int
        b: int = 1

        @once
        def singleton() -> TestClass1:
            """Returns a singleton instance of the class"""
            return TestClass1(a=1)

        @once
        def singleton2(a: int, b: int) -> TestClass1:  # type: ignore
            """Returns a singleton instance of the class"""
            return TestClass1(a=a, b=b)

    # Usage example:
    obj1 = TestClass1.singleton()
    obj2 = TestClass1.singleton()
    assert obj1 is obj2  # True - same instance

    @once
    def get_my_class_singleton() -> TestClass1:
        """for test"""
        return TestClass1(a=4)

    obj3 = get_my_class_singleton()
    obj4 = get_my_class_singleton()
    assert obj3 is obj4  # True - same instance

    assert obj1 is not obj3

    @once
    def do_something(x: int) -> TestClass1:
        return TestClass1(a=x)

    obj5 = do_something(1)
    obj6 = do_something(1)
    obj7 = do_something(2)

    assert obj5 is obj6
    assert obj5 is not obj7
    assert obj7.a == 2

    @once
    def do_something_2(x: int, y: int) -> TestClass1:
        return TestClass1(a=x + y)

    # Test multiple arguments
    obj8 = do_something_2(1, 2)
    obj9 = do_something_2(1, 2)
    obj10 = do_something_2(2, 1)

    assert obj8 is obj9  # Same args - same instance
    assert obj8 is not obj10  # Different args - different instance
    assert obj8.a == 3
    assert obj10.a == 3  # Same sum but different args

    # Test invalidation
    obj14 = TestClass1.singleton()
    TestClass1.singleton.invalidate()  # type: ignore
    obj15 = TestClass1.singleton()
    assert obj14 is not obj15

    # Test function-based invalidation
    obj16 = get_my_class_singleton()
    get_my_class_singleton.invalidate()  # type: ignore
    obj17 = get_my_class_singleton()
    assert obj16 is not obj17

    obj11 = TestClass1.singleton2(a=1, b=2)
    obj12 = TestClass1.singleton2(a=1, b=2)
    obj13 = TestClass1.singleton2(a=3, b=4)
    assert obj11 is obj12
    assert obj13 is not obj11
