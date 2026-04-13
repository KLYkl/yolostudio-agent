from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import Any, TypeVar

T = TypeVar('T')


def _resolve(awaitable: Awaitable[T]) -> T:
    iterator = awaitable.__await__()
    send_value: Any = None
    while True:
        try:
            yielded = iterator.send(send_value)
        except StopIteration as exc:
            return exc.value

        if inspect.isawaitable(yielded):
            send_value = _resolve(yielded)
            continue

        if yielded is None:
            send_value = None
            continue

        raise RuntimeError(f'unsupported awaitable yield: {yielded!r}')


def run(awaitable: Awaitable[T]) -> T:
    return _resolve(awaitable)