from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable


PostModelUpdate = dict[str, Any]
PostModelMiddleware = Callable[[dict[str, Any]], PostModelUpdate | Awaitable[PostModelUpdate]]


async def run_after_model_middlewares(
    state: dict[str, Any],
    middlewares: list[PostModelMiddleware],
) -> PostModelUpdate:
    for middleware in middlewares:
        update = middleware(state)
        if inspect.isawaitable(update):
            update = await update
        if update:
            return dict(update)
    return {}
