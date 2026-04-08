from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

from agent_plan.agent.client.context_builder import ContextBuilder
from agent_plan.agent.client.llm_factory import LlmProviderSettings, build_llm, provider_summary
from agent_plan.agent.client.event_retriever import EventRetriever
from agent_plan.agent.client.memory_store import MemoryStore
from agent_plan.agent.client.session_state import SessionState, utc_now
from agent_plan.agent.client.tool_adapter import adapt_tools_for_chat_model
from agent_plan.agent.client.tool_result_parser import parse_tool_message

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
    provider: str = os.getenv("YOLOSTUDIO_LLM_PROVIDER", "ollama")
    model: str = os.getenv("YOLOSTUDIO_AGENT_MODEL", "gemma4:e4b")
    base_url: str = os.getenv("YOLOSTUDIO_LLM_BASE_URL", "")
    api_key: str = os.getenv("YOLOSTUDIO_LLM_API_KEY", "")
    temperature: float = float(os.getenv("YOLOSTUDIO_LLM_TEMPERATURE", "0"))
    ollama_url: str = os.getenv("YOLOSTUDIO_OLLAMA_URL", "http://127.0.0.1:11434")
    mcp_url: str = os.getenv("YOLOSTUDIO_MCP_URL", "http://127.0.0.1:8080/mcp")
    max_history_messages: int = int(os.getenv("YOLOSTUDIO_MAX_HISTORY_MESSAGES", "12"))
    session_id: str = os.getenv("YOLOSTUDIO_SESSION_ID", "default")
    memory_root: str = os.getenv("YOLOSTUDIO_MEMORY_ROOT", str(Path(__file__).resolve().parents[2] / "memory"))

    def to_llm_settings(self) -> LlmProviderSettings:
        base_url = self.base_url or (self.ollama_url if self.provider.strip().lower() == 'ollama' else '')
        return LlmProviderSettings(
            provider=self.provider,
            model=self.model,
            base_url=base_url,
            api_key=self.api_key,
            temperature=self.temperature,
        )


class YoloStudioAgentClient:
    def __init__(self, graph: Any, settings: AgentSettings) -> None:
        self.graph = graph
        self.settings = settings
        self._messages: list[BaseMessage] = []
        self._turn_index = 0
        self.memory = MemoryStore(settings.memory_root)
        self.context_builder = ContextBuilder(SYSTEM_PROMPT)
        self.event_retriever = EventRetriever(self.memory)
        self.session_state: SessionState = self.memory.load_state(settings.session_id)
        self._sync_preferences()
        self.memory.save_state(self.session_state)

    @property
    def messages(self) -> list[BaseMessage]:
        return list(self._messages)

    def preview(self) -> str:
        return (
            f"YoloStudio Agent 已就绪 ({self.settings.model})\n"
            f"MCP Server: {self.settings.mcp_url} | LLM: {provider_summary(self.settings.to_llm_settings())}\n"
            f"Session: {self.session_state.session_id}"
        )

    async def chat(self, user_text: str, auto_approve: bool = False) -> dict[str, Any]:
        self._messages.append(HumanMessage(content=user_text))
        self._turn_index += 1
        thread_id = f"{self.session_state.session_id}-turn-{self._turn_index}"
        config = {"configurable": {"thread_id": thread_id}}

        self._trim_history()
        digest = self.event_retriever.build_digest(self.session_state.session_id, self.session_state)
        built_messages = self.context_builder.build_messages(self.session_state, self._messages, digest=digest)
        result = await self.graph.ainvoke({"messages": built_messages}, config=config)
        pending = self._get_pending_tool_call(config)
        if pending:
            if pending["name"] in HIGH_RISK_TOOLS:
                self._set_pending_confirmation(thread_id, pending)
                if auto_approve:
                    result = await self.graph.ainvoke(Command(resume="approved"), config=config)
                    self._clear_pending_confirmation()
                    self._apply_tool_results(result["messages"], built_messages_len=len(built_messages))
                    final_message = self._extract_final_ai(result["messages"])
                    self._messages.append(final_message or AIMessage(content=""))
                    self._trim_history()
                    self.memory.save_state(self.session_state)
                    return {"status": "completed", "message": final_message.content if final_message else "", "tool_call": pending, "approved": True}
                self.memory.save_state(self.session_state)
                return {
                    "status": "needs_confirmation",
                    "message": self._build_confirmation_prompt(pending),
                    "tool_call": pending,
                    "thread_id": thread_id,
                }
            result = await self.graph.ainvoke(Command(resume="approved"), config=config)

        self._apply_tool_results(result["messages"], built_messages_len=len(built_messages))
        final_message = self._extract_final_ai(result["messages"])
        if final_message:
            self._messages.append(AIMessage(content=final_message.content))
        self._trim_history()
        self.memory.save_state(self.session_state)
        return {
            "status": "completed",
            "message": final_message.content if final_message else "",
            "tool_call": pending,
        }

    async def confirm(self, thread_id: str, approved: bool) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        pending = self._get_pending_tool_call(config) or self._pending_from_state()
        if not pending:
            return {"status": "error", "message": "当前没有待确认的高风险操作。"}

        if not approved:
            cancel_message = self._build_cancel_message(pending)
            self._messages.append(AIMessage(content=cancel_message))
            self._clear_pending_confirmation()
            self._trim_history()
            self.memory.append_event(self.session_state.session_id, "confirmation_cancelled", {"tool": pending["name"], "args": pending.get("args", {})})
            self.memory.save_state(self.session_state)
            return {
                "status": "cancelled",
                "message": cancel_message,
                "tool_call": pending,
            }

        result = await self.graph.ainvoke(Command(resume="approved"), config=config)
        self.memory.append_event(self.session_state.session_id, "confirmation_approved", {"tool": pending["name"], "args": pending.get("args", {})})
        self._clear_pending_confirmation()
        self._apply_tool_results(result["messages"], built_messages_len=0)
        final_message = self._extract_final_ai(result["messages"])
        if final_message:
            self._messages.append(AIMessage(content=final_message.content))
        self._trim_history()
        self.memory.save_state(self.session_state)
        return {
            "status": "completed",
            "message": final_message.content if final_message else "",
            "tool_call": pending,
            "approved": True,
        }

    def _sync_preferences(self) -> None:
        # settings.model 是 LLM 模型，不应污染训练默认模型上下文。
        if self.session_state.preferences.default_model == self.settings.model:
            self.session_state.preferences.default_model = ""
        if self.session_state.preferences.language != "zh-CN":
            self.session_state.preferences.language = "zh-CN"

    def _set_pending_confirmation(self, thread_id: str, pending: dict[str, Any]) -> None:
        pc = self.session_state.pending_confirmation
        pc.thread_id = thread_id
        pc.tool_name = pending["name"]
        pc.tool_args = pending.get("args", {})
        pc.created_at = utc_now()
        self.memory.append_event(self.session_state.session_id, "confirmation_requested", {"tool": pc.tool_name, "args": pc.tool_args, "thread_id": thread_id})

    def _clear_pending_confirmation(self) -> None:
        self.session_state.pending_confirmation.thread_id = ""
        self.session_state.pending_confirmation.tool_name = ""
        self.session_state.pending_confirmation.tool_args = {}
        self.session_state.pending_confirmation.created_at = ""

    def _pending_from_state(self) -> dict[str, Any] | None:
        pc = self.session_state.pending_confirmation
        if not pc.tool_name:
            return None
        return {"name": pc.tool_name, "args": pc.tool_args, "id": None}

    def _apply_tool_results(self, messages: list[BaseMessage], built_messages_len: int) -> None:
        delta_messages = messages[built_messages_len:] if built_messages_len <= len(messages) else messages
        tool_args_by_id: dict[str, dict[str, Any]] = {}
        for message in delta_messages:
            if isinstance(message, AIMessage):
                for tool_call in getattr(message, 'tool_calls', []) or []:
                    if tool_call.get('id'):
                        tool_args_by_id[tool_call['id']] = tool_call.get('args', {})
        for message in delta_messages:
            if not isinstance(message, ToolMessage):
                continue
            parsed = parse_tool_message(message)
            tool_name = message.name or "unknown_tool"
            tool_args = tool_args_by_id.get(message.tool_call_id or '', {})
            self.memory.append_event(self.session_state.session_id, "tool_result", {"tool": tool_name, "args": tool_args, "result": parsed})
            self._apply_to_state(tool_name, parsed, tool_args)

    def _apply_to_state(self, tool_name: str, result: dict[str, Any], tool_args: dict[str, Any] | None = None) -> None:
        ds = self.session_state.active_dataset
        tr = self.session_state.active_training

        tool_args = tool_args or {}
        if tool_name == "scan_dataset" and result.get("ok"):
            ds.img_dir = str(tool_args.get('img_dir', ds.img_dir))
            ds.label_dir = str(tool_args.get('label_dir', ds.label_dir))
            detected_yaml = result.get("detected_data_yaml") or ""
            if detected_yaml:
                ds.data_yaml = str(detected_yaml)
            if summary := result.get("summary"):
                ds.last_scan = {
                    "total_images": result.get("total_images"),
                    "labeled_images": result.get("labeled_images"),
                    "missing_labels": result.get("missing_labels"),
                    "empty_labels": result.get("empty_labels"),
                    "summary": summary,
                    "detected_data_yaml": detected_yaml,
                }
        elif tool_name == "validate_dataset" and result.get("ok"):
            ds.img_dir = str(tool_args.get('img_dir', ds.img_dir))
            ds.label_dir = str(tool_args.get('label_dir', ds.label_dir))
            ds.last_validate = {
                "issue_count": result.get("issue_count"),
                "has_issues": result.get("has_issues"),
            }
        elif tool_name == "split_dataset" and result.get("ok"):
            ds.img_dir = str(tool_args.get('img_dir', ds.img_dir))
            ds.label_dir = str(tool_args.get('label_dir', ds.label_dir))
            ds.last_split = {
                "train_path": result.get("train_path"),
                "val_path": result.get("val_path"),
                "train_count": result.get("train_count"),
                "val_count": result.get("val_count"),
                "output_dir": result.get("output_dir"),
                "suggested_yaml_path": result.get("suggested_yaml_path"),
            }
        elif tool_name == "generate_yaml" and result.get("ok"):
            output_path = result.get("output_path") or ""
            if output_path:
                ds.data_yaml = str(output_path)
        elif tool_name == "training_readiness" and result.get("ok"):
            resolved_yaml = result.get("resolved_data_yaml") or ""
            if resolved_yaml:
                ds.data_yaml = str(resolved_yaml)
        elif tool_name == "start_training" and result.get("ok"):
            tr.running = True
            resolved_args = result.get("resolved_args") or {}
            tr.model = str(resolved_args.get("model") or tool_args.get("model", tr.model))
            tr.data_yaml = str(resolved_args.get("data_yaml") or tool_args.get("data_yaml", tr.data_yaml))
            if tr.data_yaml:
                ds.data_yaml = tr.data_yaml
            tr.device = result.get("device", "")
            tr.pid = result.get("pid")
            tr.log_file = result.get("log_file", "")
            tr.started_at = result.get("started_at")
            tr.last_start_result = result
        elif tool_name == "check_training_status":
            tr.last_status = result
            tr.running = bool(result.get("running"))
            tr.device = result.get("device", tr.device)
            tr.pid = result.get("pid", tr.pid)
            tr.log_file = result.get("log_file", tr.log_file)
            tr.started_at = result.get("started_at", tr.started_at)
            command = result.get("command") or []
            for part in command:
                if isinstance(part, str) and part.startswith("model="):
                    tr.model = part.split("=", 1)[1]
                if isinstance(part, str) and part.startswith("data="):
                    tr.data_yaml = part.split("=", 1)[1]
        elif tool_name == "stop_training" and result.get("ok"):
            tr.running = False
            tr.last_status = result

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

    def _trim_history(self) -> None:
        max_history = max(2, self.settings.max_history_messages)
        if len(self._messages) <= max_history:
            return
        self._messages = self._messages[-max_history:]

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
    tools = adapt_tools_for_chat_model(await client.get_tools())
    llm = build_llm(settings.to_llm_settings())
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
