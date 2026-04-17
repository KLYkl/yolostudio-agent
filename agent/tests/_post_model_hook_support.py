from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage

from yolostudio_agent.agent.client.agent_client import _build_agent_post_model_hook


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
        self._hook = _build_agent_post_model_hook(planner_llm)

    def get_state(self, config):
        del config
        return None

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
        hook_state = dict(payload)
        hook_state['messages'] = messages
        if self._state_overrides:
            hook_state.update(self._state_overrides)
        update = await self._hook(hook_state)
        updated_messages = list(update.get('messages') or [])
        if updated_messages and getattr(updated_messages[0], 'id', '') == '__remove_all__':
            updated_messages = updated_messages[1:]
        return {'messages': updated_messages or messages}
