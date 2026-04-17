from __future__ import annotations

from typing import Any, Callable

from langchain_core.messages import AIMessage, SystemMessage

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
        snapshot_messages: list[str] | None = None,
    ) -> None:
        self._planner_llm = planner_llm
        self._tool_name = str(tool_name or '').strip()
        self._tool_args = dict(tool_args or {})
        self._tool_call_builder = tool_call_builder
        self._snapshot_messages = [str(item).strip() for item in list(snapshot_messages or []) if str(item).strip()]
        self._hook = _build_agent_post_model_hook(planner_llm)

    def get_state(self, config):
        del config
        return None

    @staticmethod
    def _has_snapshot(messages: list[Any], snapshot: str) -> bool:
        prefix = snapshot.split('=', 1)[0] + '=' if '=' in snapshot else snapshot
        return any(str(getattr(message, 'content', '')).startswith(prefix) for message in messages)

    async def ainvoke(self, payload, config=None):
        del config
        messages = list(payload['messages'])
        insert_at = 2 if len(messages) >= 2 else len(messages)
        for snapshot in self._snapshot_messages:
            if not self._has_snapshot(messages, snapshot):
                messages.insert(insert_at, SystemMessage(content=snapshot))
                insert_at += 1

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
        update = await self._hook({'messages': messages})
        updated_messages = list(update.get('messages') or [])
        if updated_messages and getattr(updated_messages[0], 'id', '') == '__remove_all__':
            updated_messages = updated_messages[1:]
        return {'messages': updated_messages or messages}
