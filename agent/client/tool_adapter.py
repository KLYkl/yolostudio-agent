from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.tools import BaseTool, StructuredTool


def _stringify_tool_result(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get('type') == 'text' and item.get('text'):
                    parts.append(str(item['text']))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return '\n'.join(part for part in parts if part)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def adapt_tool_for_chat_model(tool: BaseTool) -> BaseTool:
    async def _arun(**kwargs: Any) -> str:
        result = await tool.ainvoke(kwargs)
        return _stringify_tool_result(result)

    def _run(**kwargs: Any) -> str:
        result = tool.invoke(kwargs)
        return _stringify_tool_result(result)

    return StructuredTool.from_function(
        func=_run,
        coroutine=_arun,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        return_direct=False,
    )


def adapt_tools_for_chat_model(tools: list[BaseTool]) -> list[BaseTool]:
    return [adapt_tool_for_chat_model(tool) for tool in tools]
