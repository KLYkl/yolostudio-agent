from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


ToolCallBuilder = Callable[[list[Any]], tuple[str, dict[str, object]]]


class HookedToolCallGraph:
    def __init__(
        self,
        *,
        planner_llm: Any,
        tool_name: str = '',
        tool_args: dict[str, object] | None = None,
        tool_call_builder: ToolCallBuilder | None = None,
        state_overrides: dict[str, Any] | None = None,
    ) -> None:
        self._planner_llm = planner_llm
        self._tool_name = str(tool_name or '').strip()
        self._tool_args = dict(tool_args or {})
        self._tool_call_builder = tool_call_builder
        self._state_overrides = dict(state_overrides or {})

    def get_state(self, config):
        del config
        return None

    async def _render_reply(
        self,
        *,
        tool_name: str,
        tool_args: dict[str, object],
        payload: dict[str, Any],
        messages: list[Any],
    ) -> str:
        if self._planner_llm is None:
            return ''
        facts = {
            'tool_name': tool_name,
            'tool_args': dict(tool_args),
            'state_overrides': dict(self._state_overrides),
            'message_text': [str(getattr(message, 'content', message)) for message in messages if str(getattr(message, 'content', message)).strip()],
            'graph_state_keys': sorted(str(key) for key in payload.keys()),
        }
        response = await self._planner_llm.ainvoke(
            [
                SystemMessage(content='结果说明器'),
                HumanMessage(content=str(facts)),
            ]
        )
        return str(getattr(response, 'content', response)).strip()

    async def ainvoke(self, payload, config=None):
        del config
        messages = list(payload['messages'])

        if self._tool_call_builder is not None:
            tool_name, tool_args = self._tool_call_builder(messages)
        else:
            tool_name, tool_args = self._tool_name, dict(self._tool_args)
        messages.append(
            AIMessage(
                content='',
                tool_calls=[{'id': 'tc-1', 'name': tool_name, 'args': dict(tool_args)}],
            )
        )
        reply = await self._render_reply(
            tool_name=tool_name,
            tool_args=tool_args,
            payload=payload,
            messages=messages,
        )
        if reply:
            messages.append(AIMessage(content=reply))
        return {'messages': messages}
