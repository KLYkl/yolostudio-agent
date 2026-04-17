from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage

from yolostudio_agent.agent.client import intent_parsing
from yolostudio_agent.agent.client.dataset_fact_service import (
    build_dataset_fact_followup_reply_from_messages,
    extract_dataset_fact_context_from_state,
)


RouteReporter = Callable[[str, dict[str, Any]], None]
ReplaceLastAiMessage = Callable[[list[Any], str], dict[str, Any]]


def build_fact_validation_middleware(
    *,
    replace_last_ai_message: ReplaceLastAiMessage,
    route_reporter: RouteReporter | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def after_model(state: dict[str, Any]) -> dict[str, Any]:
        messages = list(state.get('messages') or [])
        if not messages:
            return {}
        if not isinstance(messages[-1], AIMessage) or not getattr(messages[-1], 'tool_calls', None):
            return {}
        user_text = ''
        for message in reversed(messages):
            if isinstance(message, HumanMessage) and isinstance(message.content, str):
                user_text = message.content
                break
        if not user_text:
            return {}
        requested_dataset_path = intent_parsing.extract_dataset_path_from_text(user_text)
        reply = build_dataset_fact_followup_reply_from_messages(
            messages,
            user_text=user_text,
            requested_dataset_path=requested_dataset_path,
            dataset_fact_context=extract_dataset_fact_context_from_state(state),
        )
        if not reply:
            return {}
        if route_reporter is not None:
            route_reporter(
                'post-hook-override',
                {
                    'override_kind': 'dataset_fact',
                    'requested_dataset_path': requested_dataset_path,
                },
            )
        return replace_last_ai_message(messages, reply)

    return after_model
