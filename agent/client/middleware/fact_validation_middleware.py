from __future__ import annotations

from typing import Any, Callable


def build_fact_validation_middleware(
    *,
    replace_last_ai_message: Callable[[list[Any], str], dict[str, Any]],
    route_reporter: Callable[[str, dict[str, Any]], None] | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    del replace_last_ai_message, route_reporter

    def after_model(state: dict[str, Any]) -> dict[str, Any]:
        del state
        return {}

    return after_model
