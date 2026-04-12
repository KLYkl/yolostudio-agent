from __future__ import annotations

import json

from langchain_core.tools import tool
from langchain_ollama import ChatOllama


@tool
def add(a: int, b: int) -> int:
    """两数相加"""
    return a + b


def main() -> int:
    llm = ChatOllama(model="gemma4:e4b", base_url="http://127.0.0.1:11434")
    llm_with_tools = llm.bind_tools([add])
    result = llm_with_tools.invoke("请计算 3 + 5")
    tool_calls = getattr(result, "tool_calls", None)
    print(json.dumps(tool_calls, ensure_ascii=False, indent=2))
    if not tool_calls:
        return 1
    first = tool_calls[0]
    if first.get("name") != "add":
        return 2
    args = first.get("args", {})
    if args.get("a") != 3 or args.get("b") != 5:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
