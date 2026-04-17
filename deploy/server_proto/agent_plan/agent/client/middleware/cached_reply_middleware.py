from __future__ import annotations

from typing import Any, Awaitable, Callable

from langchain_core.messages import AIMessage

from yolostudio_agent.agent.client.cached_tool_reply_service import resolve_cached_tool_reply
from yolostudio_agent.agent.client.grounded_reply_builder import build_grounded_tool_reply
from yolostudio_agent.agent.client.reply_renderer import (
    render_multi_tool_result_message as render_multi_tool_result_message_reply,
    render_tool_result_message as render_tool_result_message_reply,
)


RouteReporter = Callable[[str, dict[str, Any]], None]
ReplaceLastAiMessage = Callable[[list[Any], str], dict[str, Any]]
InvokeRendererText = Callable[..., Awaitable[str]]
RenderToolResultMessage = Callable[[str, dict[str, Any]], Awaitable[str]]


def build_cached_reply_middleware(
    planner_llm: Any,
    *,
    replace_last_ai_message: ReplaceLastAiMessage,
    message_text: Callable[[Any], str],
    merge_grounded_sections: Callable[[list[str]], str],
    route_reporter: RouteReporter | None = None,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    async def _invoke_renderer_text(
        *,
        messages: list[Any],
        failure_event: str = '',
        failure_payload: dict[str, Any] | None = None,
    ) -> str:
        del failure_event, failure_payload
        if planner_llm is None:
            return ''
        try:
            response = await planner_llm.ainvoke(messages)
        except Exception:
            return ''
        return message_text(getattr(response, 'content', response)).strip()

    async def _render_tool_result_message(tool_name: str, parsed: dict[str, Any]) -> str:
        async def _render_multi_tool_result_message(
            applied_results: list[tuple[str, dict[str, Any]]],
            *,
            objective: str = '',
            extra_notes: list[str] | None = None,
        ) -> str:
            return await render_multi_tool_result_message_reply(
                planner_llm=planner_llm,
                applied_results=applied_results,
                objective=objective,
                extra_notes=extra_notes,
                invoke_renderer_text=_invoke_renderer_text,
                render_tool_result_message=_render_tool_result_message,
                build_grounded_tool_reply=build_grounded_tool_reply,
                merge_grounded_sections=merge_grounded_sections,
            )

        return await render_tool_result_message_reply(
            planner_llm=planner_llm,
            tool_name=tool_name,
            parsed=parsed,
            render_multi_tool_result_message=_render_multi_tool_result_message,
            invoke_renderer_text=_invoke_renderer_text,
            build_grounded_tool_reply=build_grounded_tool_reply,
            merge_grounded_sections=merge_grounded_sections,
        )

    async def after_model(state: dict[str, Any]) -> dict[str, Any]:
        messages = list(state.get('messages') or [])
        if not messages or not isinstance(messages[-1], AIMessage) or not getattr(messages[-1], 'tool_calls', None):
            return {}
        cached_tool_result = resolve_cached_tool_reply(messages)
        if not cached_tool_result:
            return {}
        tool_name, payload = cached_tool_result
        reply = await _render_tool_result_message(tool_name, payload)
        if not reply:
            return {}
        if route_reporter is not None:
            route_reporter(
                'post-hook-override',
                {
                    'override_kind': 'cached_reply',
                    'tool': tool_name,
                },
            )
        return replace_last_ai_message(messages, reply)

    return after_model
