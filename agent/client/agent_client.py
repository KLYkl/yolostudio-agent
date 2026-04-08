from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

SYSTEM_PROMPT = """你是 YoloStudio Agent，负责帮助用户完成数据准备和训练管理。

工作原则：
1. 优先使用 MCP tools 获取真实结果，不要凭空猜测文件、数据集状态或训练状态。
2. 回答用简洁中文，先给结果，再给必要说明。
3. 涉及会改动数据或启动长任务的动作，不要在自然语言里先追问确认；参数足够时直接生成对应 tool 调用，由外部客户端拦截并做人审确认。
4. 工具失败时，直接说明失败原因，并给出下一步建议。
5. 如果用户的问题不需要工具，直接回答；如果需要工具，优先选择最少必要工具。"""

HIGH_RISK_TOOLS = {"start_training", "split_dataset", "augment_dataset"}


@dataclass(slots=True)
class AgentSettings:
    model: str = os.getenv("YOLOSTUDIO_AGENT_MODEL", "gemma4:e4b")
    ollama_url: str = os.getenv("YOLOSTUDIO_OLLAMA_URL", "http://127.0.0.1:11434")
    mcp_url: str = os.getenv("YOLOSTUDIO_MCP_URL", "http://127.0.0.1:8080/mcp")
    max_history_messages: int = int(os.getenv("YOLOSTUDIO_MAX_HISTORY_MESSAGES", "12"))


class YoloStudioAgentClient:
    def __init__(self, graph: Any, settings: AgentSettings) -> None:
        self.graph = graph
        self.settings = settings
        self._messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        self._turn_index = 0

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def preview(self) -> str:
        return (
            f"YoloStudio Agent 已就绪 ({self.settings.model})\n"
            f"MCP Server: {self.settings.mcp_url} | Ollama: {self.settings.ollama_url}"
        )

    async def chat(self, user_text: str, auto_approve: bool = False) -> dict[str, Any]:
        self._messages.append(HumanMessage(content=user_text))
        self._turn_index += 1
        thread_id = f"cli-turn-{self._turn_index}"
        config = {"configurable": {"thread_id": thread_id}}

        self._trim_history()
        result = await self.graph.ainvoke({"messages": self._messages}, config=config)
        pending = self._get_pending_tool_call(config)
        if pending:
            if pending["name"] in HIGH_RISK_TOOLS:
                if auto_approve:
                    result = await self.graph.ainvoke(Command(resume="approved"), config=config)
                    final_message = self._extract_final_ai(result["messages"])
                    self._messages = self._normalize_history(result["messages"])
                    return {
                        "status": "completed",
                        "message": final_message.content if final_message else "",
                        "tool_call": pending,
                        "approved": True,
                    }
                return {
                    "status": "needs_confirmation",
                    "message": self._build_confirmation_prompt(pending),
                    "tool_call": pending,
                    "thread_id": thread_id,
                }
            result = await self.graph.ainvoke(Command(resume="approved"), config=config)

        final_message = self._extract_final_ai(result["messages"])
        self._messages = self._normalize_history(result["messages"])
        self._trim_history()
        return {
            "status": "completed",
            "message": final_message.content if final_message else "",
            "tool_call": pending,
        }

    async def confirm(self, thread_id: str, approved: bool) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        pending = self._get_pending_tool_call(config)
        if not pending:
            return {"status": "error", "message": "当前没有待确认的高风险操作。"}

        if not approved:
            cancel_message = self._build_cancel_message(pending)
            self._messages.append(AIMessage(content=cancel_message))
            self._trim_history()
            return {
                "status": "cancelled",
                "message": cancel_message,
                "tool_call": pending,
            }

        result = await self.graph.ainvoke(Command(resume="approved"), config=config)
        final_message = self._extract_final_ai(result["messages"])
        self._messages = self._normalize_history(result["messages"])
        self._trim_history()
        return {
            "status": "completed",
            "message": final_message.content if final_message else "",
            "tool_call": pending,
            "approved": True,
        }

    def _get_pending_tool_call(self, config: dict[str, Any]) -> dict[str, Any] | None:
        state = self.graph.get_state(config)
        if not state or not getattr(state, "next", None):
            return None
        if tuple(state.next) != ("tools",):
            return None
        messages = state.values.get("messages", [])
        if not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            return None
        tool_call = last_message.tool_calls[0]
        return {
            "id": tool_call.get("id"),
            "name": tool_call.get("name"),
            "args": tool_call.get("args", {}),
        }

    @staticmethod
    def _extract_final_ai(messages: list[BaseMessage]) -> AIMessage | None:
        for message in reversed(messages):
            if isinstance(message, AIMessage) and message.content:
                return message
        return None

    @staticmethod
    def _normalize_history(messages: list[BaseMessage]) -> list[BaseMessage]:
        return list(messages)

    def _trim_history(self) -> None:
        max_history = max(2, self.settings.max_history_messages)
        system_messages = [message for message in self._messages if isinstance(message, SystemMessage)]
        non_system_messages = [message for message in self._messages if not isinstance(message, SystemMessage)]
        self._messages = system_messages[:1] + non_system_messages[-max_history:]

    @staticmethod
    def _build_confirmation_prompt(tool_call: dict[str, Any]) -> str:
        args = tool_call.get("args", {})
        pretty_args = "\n".join(f"  - {k}: {v}" for k, v in args.items()) or "  - 无参数"
        return (
            f"检测到高风险操作：{tool_call['name']}\n"
            f"参数摘要：\n{pretty_args}\n"
            "确认执行？(y/n)"
        )

    @staticmethod
    def _build_cancel_message(tool_call: dict[str, Any]) -> str:
        return f"已取消操作：{tool_call['name']}。如需继续，请调整参数后重新下达指令。"


async def build_agent_client(settings: AgentSettings | None = None) -> YoloStudioAgentClient:
    settings = settings or AgentSettings()
    client = MultiServerMCPClient(
        {
            "yolostudio": {
                "transport": "streamable-http",
                "url": settings.mcp_url,
            }
        }
    )
    tools = await client.get_tools()
    llm = ChatOllama(model=settings.model, base_url=settings.ollama_url)
    graph = create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],
    )
    return YoloStudioAgentClient(graph=graph, settings=settings)


async def build_agent():
    return await build_agent_client()

