from __future__ import annotations

from typing import Any, Awaitable, Callable


def build_cached_reply_middleware(
    planner_llm: Any,
    *,
    replace_last_ai_message: Callable[[list[Any], str], dict[str, Any]],
    message_text: Callable[[Any], str],
    merge_grounded_sections: Callable[[list[str]], str],
    route_reporter: Callable[[str, dict[str, Any]], None] | None = None,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    del planner_llm, replace_last_ai_message, message_text, merge_grounded_sections, route_reporter

    async def after_model(state: dict[str, Any]) -> dict[str, Any]:
        del state
        return {}

    return after_model
