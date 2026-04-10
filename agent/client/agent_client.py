from __future__ import annotations

import json
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
from agent_plan.agent.client.event_retriever import EventRetriever
from agent_plan.agent.client.llm_factory import LlmProviderSettings, build_llm, provider_summary
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
5. 如果用户的问题不需要工具，直接回答；如果需要工具，优先选择最少必要工具。

数据集目录约定：
- 用户提供的路径可能是数据集根目录（如 /home/kly/test_dataset/），而不是直接的图片目录。
- scan_dataset、validate_dataset、training_readiness 都支持直接传 dataset root，工具层会自动解析 images/ 和 labels/ 子目录。
- 当用户表达“用这个数据训练”“按默认比例划分再训练”这类需求时，优先使用 prepare_dataset_for_training 先把数据准备到可训练状态。
- 不要自己猜测子目录名称；优先依赖工具返回的 img_dir / label_dir / data_yaml。

训练约定：
- device 默认传 auto，GPU 分配由服务器端策略决定。
- training_readiness 是训练前检查的优先入口；如果 data_yaml 已明确且参数完整，也可以直接 start_training。
- check_gpu_status 仅在用户明确询问 GPU 状态时使用；不要在每次训练前机械地多调一次。"""

HIGH_RISK_TOOLS = {"start_training", "split_dataset", "augment_dataset", "prepare_dataset_for_training"}


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
    def __init__(self, graph: Any, settings: AgentSettings, tool_registry: dict[str, Any]) -> None:
        self.graph = graph
        self.settings = settings
        self.tool_registry = tool_registry
        self._messages: list[BaseMessage] = []
        self._turn_index = 0
        self._applied_tool_call_ids: set[str] = set()
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

    def context_summary(self) -> str:
        digest = self.event_retriever.build_digest(self.session_state.session_id, self.session_state)
        return self.context_builder.build_state_summary(self.session_state, digest)

    def session_summary(self) -> str:
        return (
            f"Session ID: {self.session_state.session_id}\n"
            f"Created: {self.session_state.created_at}\n"
            f"Updated: {self.session_state.updated_at}\n"
            f"History Length: {len(self._messages)}\n"
            f"Turn Index: {self._turn_index}"
        )

    async def direct_tool(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        tool = self.tool_registry.get(tool_name)
        if not tool:
            return {"ok": False, "error": f"未找到工具: {tool_name}"}
        payload = await tool.ainvoke(kwargs)
        parsed = self._normalize_tool_output(payload)
        self.memory.append_event(self.session_state.session_id, "tool_result", {"tool": tool_name, "args": kwargs, "result": parsed})
        self._apply_to_state(tool_name, parsed, kwargs)
        self.memory.save_state(self.session_state)
        return parsed

    async def chat(self, user_text: str, auto_approve: bool = False) -> dict[str, Any]:
        self._messages.append(HumanMessage(content=user_text))
        self._turn_index += 1
        thread_id = f"{self.session_state.session_id}-turn-{self._turn_index}"
        config = {"configurable": {"thread_id": thread_id}}

        self._trim_history()
        digest = self.event_retriever.build_digest(self.session_state.session_id, self.session_state)
        built_messages = self.context_builder.build_messages(self.session_state, self._messages, digest=digest)
        result = await self.graph.ainvoke({"messages": built_messages}, config=config)

        while True:
            pending = self._get_pending_tool_call(config)
            if not pending:
                break
            if pending["name"] in HIGH_RISK_TOOLS and not auto_approve:
                self._set_pending_confirmation(thread_id, pending)
                self.memory.save_state(self.session_state)
                return {
                    "status": "needs_confirmation",
                    "message": self._build_confirmation_prompt(pending),
                    "tool_call": pending,
                    "thread_id": thread_id,
                }
            result = await self.graph.ainvoke(Command(resume="approved"), config=config)

        self._apply_tool_results(result["messages"], built_messages_len=len(built_messages))
        final_text = self._extract_or_fallback(result["messages"])
        self._messages.append(AIMessage(content=final_text))
        self._trim_history()
        self.memory.save_state(self.session_state)
        return {
            "status": "completed",
            "message": final_text,
            "tool_call": None,
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

        self.memory.append_event(self.session_state.session_id, "confirmation_approved", {"tool": pending["name"], "args": pending.get("args", {})})
        self._clear_pending_confirmation()
        result = await self.graph.ainvoke(Command(resume="approved"), config=config)
        self._apply_tool_results(result["messages"], built_messages_len=0)

        while True:
            next_pending = self._get_pending_tool_call(config)
            if not next_pending:
                break
            if next_pending["name"] in HIGH_RISK_TOOLS:
                self._set_pending_confirmation(thread_id, next_pending)
                self.memory.save_state(self.session_state)
                return {
                    "status": "needs_confirmation",
                    "message": self._build_confirmation_prompt(next_pending),
                    "tool_call": next_pending,
                    "thread_id": thread_id,
                }
            result = await self.graph.ainvoke(Command(resume="approved"), config=config)
            self._apply_tool_results(result["messages"], built_messages_len=0)

        final_text = self._extract_or_fallback(result["messages"])
        self._messages.append(AIMessage(content=final_text))
        self._trim_history()
        self.memory.save_state(self.session_state)
        return {
            "status": "completed",
            "message": final_text,
            "tool_call": pending,
            "approved": True,
        }

    def _sync_preferences(self) -> None:
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
            if message.tool_call_id and message.tool_call_id in self._applied_tool_call_ids:
                continue
            parsed = parse_tool_message(message)
            tool_name = message.name or "unknown_tool"
            tool_args = tool_args_by_id.get(message.tool_call_id or '', {})
            self.memory.append_event(self.session_state.session_id, "tool_result", {"tool": tool_name, "args": tool_args, "result": parsed})
            self._apply_to_state(tool_name, parsed, tool_args)
            if message.tool_call_id:
                self._applied_tool_call_ids.add(message.tool_call_id)

    def _apply_to_state(self, tool_name: str, result: dict[str, Any], tool_args: dict[str, Any] | None = None) -> None:
        ds = self.session_state.active_dataset
        tr = self.session_state.active_training

        tool_args = tool_args or {}
        if tool_name == "scan_dataset" and result.get("ok"):
            ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
            ds.img_dir = str(result.get('resolved_img_dir') or tool_args.get('img_dir', ds.img_dir))
            ds.label_dir = str(result.get('resolved_label_dir') or tool_args.get('label_dir', ds.label_dir))
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
            ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
            ds.img_dir = str(result.get('resolved_img_dir') or tool_args.get('img_dir', ds.img_dir))
            ds.label_dir = str(result.get('resolved_label_dir') or tool_args.get('label_dir', ds.label_dir))
            ds.last_validate = {
                "issue_count": result.get("issue_count"),
                "has_issues": result.get("has_issues"),
            }
        elif tool_name == "split_dataset" and result.get("ok"):
            ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
            ds.img_dir = str(result.get('img_dir') or tool_args.get('img_dir', ds.img_dir))
            ds.label_dir = str(result.get('label_dir') or tool_args.get('label_dir', ds.label_dir))
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
            ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
            ds.img_dir = str(result.get('resolved_img_dir') or ds.img_dir)
            ds.label_dir = str(result.get('resolved_label_dir') or ds.label_dir)
            resolved_yaml = result.get("resolved_data_yaml") or ""
            if resolved_yaml:
                ds.data_yaml = str(resolved_yaml)
        elif tool_name == "prepare_dataset_for_training" and result.get("ok"):
            ds.dataset_root = str(result.get('dataset_root') or ds.dataset_root)
            ds.img_dir = str(result.get('img_dir') or ds.img_dir)
            ds.label_dir = str(result.get('label_dir') or ds.label_dir)
            if result.get('data_yaml'):
                ds.data_yaml = str(result['data_yaml'])
            for step in result.get('steps_completed', []):
                step_name = step.get('step')
                if step_name == 'scan' and step.get('ok'):
                    ds.last_scan = {
                        'total_images': step.get('total_images'),
                        'labeled_images': step.get('labeled_images'),
                        'missing_labels': step.get('missing_labels'),
                        'empty_labels': step.get('empty_labels'),
                        'summary': step.get('summary'),
                        'detected_data_yaml': step.get('detected_data_yaml', ''),
                    }
                elif step_name == 'validate' and step.get('ok'):
                    ds.last_validate = {
                        'issue_count': step.get('issue_count'),
                        'has_issues': step.get('has_issues'),
                    }
                elif step_name == 'split' and step.get('ok'):
                    ds.last_split = {
                        'train_path': step.get('train_path'),
                        'val_path': step.get('val_path'),
                        'train_count': step.get('train_count'),
                        'val_count': step.get('val_count'),
                        'output_dir': step.get('output_dir'),
                        'suggested_yaml_path': step.get('suggested_yaml_path'),
                    }
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

    def _extract_or_fallback(self, messages: list[BaseMessage]) -> str:
        final_message = self._extract_final_ai(messages)
        final_text = final_message.content.strip() if final_message and isinstance(final_message.content, str) else ""
        return final_text or self._build_empty_reply_fallback(messages)

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

    def _build_empty_reply_fallback(self, messages: list[BaseMessage]) -> str:
        tool_calls: list[str] = []
        tool_errors: list[str] = []
        for msg in messages:
            if isinstance(msg, AIMessage):
                for tc in getattr(msg, 'tool_calls', []) or []:
                    name = tc.get('name')
                    if name and name not in tool_calls:
                        tool_calls.append(name)
            if isinstance(msg, ToolMessage):
                parsed = parse_tool_message(msg)
                if not parsed.get('ok', True):
                    tool_errors.append(f"{msg.name}: {parsed.get('error', '未知错误')}")

        parts = ["Agent 未能生成有效回复。"]
        if tool_calls:
            parts.append(f"已调用工具: {', '.join(tool_calls)}")
        if tool_errors:
            parts.append(f"工具错误: {'; '.join(tool_errors)}")
        parts.append("可输入 /context 查看当前状态，或换一种方式描述需求。")
        return "\n".join(parts)

    @staticmethod
    def _normalize_tool_output(payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            texts: list[str] = []
            for item in payload:
                if isinstance(item, dict):
                    texts.append(item.get('text', ''))
                else:
                    texts.append(str(item))
            raw = "\n".join(part for part in texts if part).strip()
            if not raw:
                return {"ok": True, "raw": raw}
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {"ok": True, "value": parsed}
            except Exception:
                return {"ok": True, "raw": raw}
        return {"ok": True, "raw": str(payload)}


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
    raw_tools = await client.get_tools()
    tools = adapt_tools_for_chat_model(raw_tools)
    llm = build_llm(settings.to_llm_settings())
    graph = create_react_agent(
        llm,
        tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=MemorySaver(),
        interrupt_before=["tools"],
    )
    tool_registry = {tool.name: tool for tool in raw_tools}
    return YoloStudioAgentClient(graph=graph, settings=settings, tool_registry=tool_registry)


async def build_agent():
    return await build_agent_client()
